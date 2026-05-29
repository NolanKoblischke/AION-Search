"""
Step 2: Generate LLM descriptions (OpenAI batch or OpenRouter direct).

CLI:
    uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-4.1-mini
    uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-5-mini
    uv run python src/experiments/gz10/step2_describe.py --submit --model qwen/qwen3-vl-235b-a22b-thinking
    uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-4.1-mini --no-images
    uv run python src/experiments/gz10/step2_describe.py --check <BATCH_INFO_JSON>
    uv run python src/experiments/gz10/step2_describe.py --embed <DESCRIPTIONS_RAW_PARQUET>

Resume interrupted upload:
    uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-5-mini --resume <CHECKPOINT_FILE>

Notes:
- OpenAI models use the batch API; OpenRouter models use direct calls.
- Set OPENAI_API_KEY for OpenAI and OPENROUTER_API_KEY for OpenRouter.
- Use --no-images to send prompt-only requests (no images created or uploaded).
- Uploads are checkpointed to upload_checkpoint.jsonl, allowing resume on failure.
- Images come from the HuggingFace dataset (image_rgb column).

Outputs:
- Runs are stored under data/gz10/llm_results/<model>_<provider>_<timestamp>/...
- Embeddings are saved next to the raw file by default (descriptions_with_embeddings.parquet)
"""

import argparse
import asyncio
import base64
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from src.experiments.gz10.constants import (
    CLASS_LABELS,
    DATA_DIR,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    IMAGE_EMBEDDINGS_PARQUET,
)
from src.utils.openai_utils import (
    MAX_BATCH_REQUESTS,
    OpenAIBatchProcessor,
    OpenAIEmbeddingGenerator,
    load_model_configs,
    read_prompt,
    wait_for_file_processing,
)

load_dotenv()

PARALLEL_UPLOADS = 10
PARALLEL_REQUESTS = 30
MAX_UPLOAD_RETRIES = 5
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.5
RETRY_JITTER = 0.5

RESULTS_BASE_DIR = Path(DATA_DIR) / "llm_results"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def infer_provider(model: str, provider: str) -> str:
    if provider in {"openai", "openrouter"}:
        return provider
    if provider != "auto":
        raise ValueError(f"Unknown provider: {provider}")
    if "/" in model:
        return "openrouter"
    return "openai"


def sanitize_model_name(model: str) -> str:
    sanitized = []
    for char in model:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_")


def resolve_results_dir(override: str | None) -> Path:
    if override:
        return Path(override)
    return RESULTS_BASE_DIR


def build_run_dir(
    results_dir: Path,
    model: str,
    provider: str,
    timestamp: str,
    run_name: str | None,
    mode: str,
) -> Path:
    model_slug = sanitize_model_name(model)
    suffix = f"_{run_name}" if run_name else ""
    return results_dir / f"{model_slug}_{provider}_{mode}_{timestamp}{suffix}"


def resolve_embed_output_path(descriptions_file: Path, output_override: str | None) -> Path:
    if output_override:
        return Path(output_override)
    if descriptions_file.name == "descriptions_raw.parquet":
        return descriptions_file.with_name("descriptions_with_embeddings.parquet")
    return descriptions_file.with_name(f"{descriptions_file.stem}_with_embeddings.parquet")


def load_hf_dataset():
    """Load test split from HuggingFace dataset."""
    print(f"Loading {HF_DATASET_REPO} split=test...")
    ds = load_dataset(HF_DATASET_REPO, split="test", revision=HF_DATASET_REVISION)
    print(f"Loaded {len(ds)} samples")
    return ds


def save_pngs_from_hf(ds, output_dir: Path) -> list[tuple[str, Path]]:
    """Save PNG images from HF dataset image_rgb column to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Saving {len(ds)} PNG images...")
    for row in tqdm(ds):
        idx = row["Galaxy10_DECals_index"]
        object_id = f"gz10_{idx:06d}"
        png_path = output_dir / f"{object_id}_gz10.png"

        if not png_path.exists():
            image = row["image_rgb"]
            image.save(png_path, format="PNG")

        results.append((object_id, png_path))

    return results


def build_batch_jsonl(
    file_mappings: list[tuple],
    prompt: str,
    model_config: dict,
    output_path: Path,
    no_images: bool,
):
    """Create a JSONL file with batch requests."""
    with output_path.open("w") as f:
        for object_id, file_id in file_mappings:
            content = [{"type": "input_text", "text": prompt}]
            if not no_images:
                content.append({"type": "input_image", "file_id": file_id})

            body = {
                "model": model_config["model_name"],
                "input": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            }

            reasoning_effort = model_config.get("reasoning_effort")
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}

            req = {
                "custom_id": object_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(req) + "\n")


async def upload_images(
    client, paths: list[Path], checkpoint_file: Path | None = None
) -> list[tuple]:
    """Upload images to OpenAI and return (object_id, file_id) pairs.

    Saves progress to checkpoint_file as uploads complete, allowing resume on failure.
    """
    sem = asyncio.Semaphore(PARALLEL_UPLOADS)

    # Load existing checkpoint if available
    already_uploaded = {}
    if checkpoint_file and checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                already_uploaded[data["object_id"]] = data["file_id"]
        print(f"Found {len(already_uploaded)} already uploaded images")

    # Filter out already uploaded paths
    paths_to_upload = []
    for p in paths:
        stem = p.stem
        object_id = stem[:-5] if stem.endswith("_gz10") else stem
        if object_id not in already_uploaded:
            paths_to_upload.append(p)

    if not paths_to_upload:
        print("All images already uploaded, using checkpoint data")
        return [(oid, fid) for oid, fid in already_uploaded.items()]

    print(f"Uploading {len(paths_to_upload)} images ({len(already_uploaded)} already done)")

    # Open checkpoint file for appending
    checkpoint_fh = None
    if checkpoint_file:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_fh = open(checkpoint_file, "a")

    async def upload_one(path: Path) -> tuple:
        for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
            try:
                async with sem:
                    with open(path, "rb") as f:
                        response = await client.files.create(file=f, purpose="vision")
                # Extract object_id from filename
                stem = path.stem
                if stem.endswith("_gz10"):
                    object_id = stem[:-5]
                else:
                    object_id = stem
                return object_id, response.id
            except Exception as exc:
                if attempt == MAX_UPLOAD_RETRIES:
                    print(f"Failed to upload {path} after {attempt} attempts: {exc}")
                    return None, None
                sleep_s = (
                    RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    + random.random() * RETRY_JITTER
                )
                await asyncio.sleep(sleep_s)
        return None, None

    tasks = [asyncio.create_task(upload_one(p)) for p in paths_to_upload]
    new_results = []

    with tqdm(total=len(tasks), desc="Uploading images") as pbar:
        for task in asyncio.as_completed(tasks):
            result = await task
            if result[0] is not None and result[1] is not None:
                new_results.append(result)
                # Write to checkpoint immediately
                if checkpoint_fh:
                    checkpoint_fh.write(
                        json.dumps({"object_id": result[0], "file_id": result[1]}) + "\n"
                    )
                    checkpoint_fh.flush()
            pbar.update(1)

    if checkpoint_fh:
        checkpoint_fh.close()

    # Combine already uploaded with newly uploaded
    all_results = [(oid, fid) for oid, fid in already_uploaded.items()]
    all_results.extend(new_results)

    return all_results


def png_to_data_url(png_path: Path) -> str:
    """Convert a PNG file to a base64 data URL."""
    b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def describe_one(
    object_id: str,
    png_path: Path | None,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    no_images: bool,
) -> dict:
    """Send one request to OpenRouter and get description."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                content = [{"type": "text", "text": prompt}]
                if not no_images:
                    img_url = png_to_data_url(png_path)
                    content.append({"type": "image_url", "image_url": {"url": img_url}})
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )

                text = response.choices[0].message.content or ""
                usage = response.usage

                return {
                    "object_id": object_id,
                    "description": text.strip(),
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "success": True,
                }
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"Failed to describe {object_id} after {attempt} attempts: {exc}")
                return {
                    "object_id": object_id,
                    "description": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "success": False,
                    "error": str(exc),
                }
            sleep_s = (
                RETRY_BASE_DELAY * (2 ** (attempt - 1))
                + random.random() * RETRY_JITTER
            )
            await asyncio.sleep(sleep_s)

    return {
        "object_id": object_id,
        "description": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "success": False,
    }


def build_metadata_df():
    """Load metadata from image embeddings parquet for merging with results."""
    df = pq.read_table(IMAGE_EMBEDDINGS_PARQUET).to_pandas()
    df["object_id"] = df["Galaxy10_DECals_index"].apply(lambda x: f"gz10_{x:06d}")
    return df


def build_descriptions_table(df_merged: pd.DataFrame) -> pa.Table:
    """Build a standardized descriptions parquet table from merged data."""
    return pa.table(
        {
            "ra": pa.array(df_merged["ra"].values, type=pa.float64()),
            "dec": pa.array(df_merged["dec"].values, type=pa.float64()),
            "label": pa.array(df_merged["label"].values, type=pa.int64()),
            "label_name": pa.array(
                [CLASS_LABELS.get(int(l), "Unknown") for l in df_merged["label"]],
                type=pa.string(),
            ),
            "Galaxy10_DECals_index": pa.array(
                df_merged["Galaxy10_DECals_index"].values, type=pa.int64()
            ),
            "object_id": pa.array(df_merged["object_id"].values, type=pa.string()),
            "description": pa.array(df_merged["description"].values, type=pa.string()),
        }
    )


async def submit_openai_batch(args):
    """Submit batch job for OpenAI descriptions."""
    model_configs = load_model_configs()
    if args.model not in model_configs:
        print(f"Model {args.model} not found. Available: {list(model_configs.keys())}")
        print("Tip: OpenRouter models include a '/' (e.g., qwen/...).")
        return

    model_config = model_configs[args.model]
    prompt = read_prompt(args.prompt)

    # Handle resume mode vs new submission
    if hasattr(args, "resume") and args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            return
        batch_folder = checkpoint_path.parent
        folder_name = batch_folder.name
        parts = folder_name.rsplit("_", 2)
        if len(parts) >= 2:
            timestamp = f"{parts[-2]}_{parts[-1]}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = None

    if args.no_images:
        # No need to load HF dataset -- get object IDs from local metadata
        df_meta = build_metadata_df()
        if args.limit:
            df_meta = df_meta.head(args.limit)
        image_results = [
            (f"gz10_{idx:06d}", None)
            for idx in df_meta["Galaxy10_DECals_index"].values
        ]
        file_mappings = [(object_id, None) for object_id, _ in image_results]
    else:
        # Load images from HF dataset
        ds = load_hf_dataset()
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        png_dir = Path(DATA_DIR) / "pngs"
        image_results = save_pngs_from_hf(ds, png_dir)

    processor = OpenAIBatchProcessor()
    client = processor.client
    if not args.no_images:
        # Determine checkpoint file location
        if checkpoint_path is None:
            results_dir = resolve_results_dir(args.results_dir)
            batch_folder = build_run_dir(
                results_dir,
                args.model,
                "openai",
                timestamp,
                args.run_name,
                "batch",
            )
            batch_folder.mkdir(parents=True, exist_ok=True)
            checkpoint_path = batch_folder / "upload_checkpoint.jsonl"

        # Upload images with checkpointing
        paths = [r[1] for r in image_results]
        file_mappings = await upload_images(client, paths, checkpoint_path)

        print(f"Total file mappings: {len(file_mappings)} images")

        # Wait for files to be processed
        file_ids = [fid for _, fid in file_mappings]
        await wait_for_file_processing(client, file_ids)

    # Create batch jobs (split if needed)
    mode = "batch_blind" if args.no_images else "batch"
    if not args.no_images and checkpoint_path:
        batch_folder = checkpoint_path.parent
    else:
        results_dir = resolve_results_dir(args.results_dir)
        batch_folder = build_run_dir(
            results_dir,
            args.model,
            "openai",
            timestamp,
            args.run_name,
            mode,
        )
        batch_folder.mkdir(parents=True, exist_ok=True)

    all_batch_infos = []
    batch_num = 0

    for i in range(0, len(file_mappings), MAX_BATCH_REQUESTS):
        batch_num += 1
        batch_mappings = file_mappings[i : i + MAX_BATCH_REQUESTS]

        batch_jsonl_path = batch_folder / f"batch_input_{timestamp}_part{batch_num}.jsonl"
        build_batch_jsonl(
            batch_mappings, prompt, model_config, batch_jsonl_path, args.no_images
        )

        # Upload batch input file
        input_file_id = await processor.upload_file(batch_jsonl_path, purpose="batch")

        # Create batch job
        batch = await client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "job": f"gz10_descriptions_{timestamp}_batch_{batch_num}",
                "model": model_config["formatted_name"],
                "total_requests": str(len(batch_mappings)),
            },
        )

        batch_info = {
            "batch_id": batch.id,
            "batch_number": batch_num,
            "total_requests": len(batch_mappings),
            "batch_info_path": str(
                batch_folder / f"batch_info_{timestamp}_part{batch_num}.json"
            ),
        }

        # Save individual batch info
        with open(batch_info["batch_info_path"], "w") as f:
            json.dump(
                {
                    **batch_info,
                    "timestamp": timestamp,
                    "provider": "openai",
                    "model": model_config,
                    "prompt_file": args.prompt,
                },
                f,
                indent=2,
            )

        all_batch_infos.append(batch_info)
        print(f"Created batch job {batch_num}: {batch.id}")

    # Save master batch info
    master_info_path = batch_folder / f"batch_info_{timestamp}_master.json"
    with open(master_info_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "provider": "openai",
                "model": model_config,
                "total_requests": len(file_mappings),
                "num_batches": len(all_batch_infos),
                "batch_folder": str(batch_folder),
                "batches": all_batch_infos,
                "prompt_file": args.prompt,
            },
            f,
            indent=2,
        )

    print("\nBatch processing submitted!")
    print(f"Master batch info: {master_info_path}")
    print("\nTo check status, run:")
    print(f"  uv run python src/experiments/gz10/step2_describe.py --check {master_info_path}")


def check_batch(args):
    """Check batch status and download results."""
    with open(args.batch_info, "r") as f:
        batch_info = json.load(f)

    processor = OpenAIBatchProcessor()
    is_master = "batches" in batch_info

    all_results_paths = []
    all_completed = True

    if is_master:
        print(f"Checking {len(batch_info['batches'])} batch job(s)...")
        for batch_data in batch_info["batches"]:
            with open(batch_data["batch_info_path"], "r") as f:
                sub_batch_info = json.load(f)

            batch_id = sub_batch_info["batch_id"]
            batch = processor.check_batch_status(batch_id)

            if batch is None:
                print(f"Failed to retrieve batch {batch_id}")
                all_completed = False
                continue

            print(f"Batch {batch_id}: {batch.status}")

            if batch.status == "completed":
                output_dir = Path(batch_info["batch_folder"]) / "results"
                output_dir.mkdir(parents=True, exist_ok=True)

                batch_num = sub_batch_info.get("batch_number", 1)
                output_path = output_dir / f"batch_results_part{batch_num}.jsonl"
                processor.download_batch_results(batch.output_file_id, output_path)
                all_results_paths.append(output_path)
                print(f"Downloaded results to {output_path}")

            elif batch.status in ["failed", "expired", "cancelled"]:
                print(f"Batch failed with status: {batch.status}")
                all_completed = False
            else:
                all_completed = False
    else:
        batch_id = batch_info["batch_id"]
        batch = processor.check_batch_status(batch_id)
        print(f"Batch {batch_id}: {batch.status}")

        if batch.status == "completed":
            output_dir = Path(batch_info.get("batch_folder", RESULTS_BASE_DIR)) / "results"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "batch_results.jsonl"
            processor.download_batch_results(batch.output_file_id, output_path)
            all_results_paths.append(output_path)
            all_completed = True

    if not all_completed:
        print("\nNot all batches completed yet. Run --check again later.")
        return

    if not all_results_paths:
        print("No results to process.")
        return

    # Process results and save intermediate parquet
    print("\nProcessing results...")
    model_config = batch_info["model"]
    input_price = model_config.get("input_price", 0) / 1_000_000
    output_price = model_config.get("output_price", 0) / 1_000_000

    results = []
    for result_path in all_results_paths:
        with open(result_path, "r") as f:
            for line in f:
                result_data = json.loads(line.strip())

                if result_data["response"]["status_code"] != 200:
                    continue

                custom_id = result_data["custom_id"]
                body = result_data["response"]["body"]

                output_text = ""
                for output_msg in body.get("output", []):
                    if "content" in output_msg:
                        for content in output_msg["content"]:
                            if "text" in content and content["text"].strip():
                                output_text = content["text"]
                                break
                    if output_text:
                        break

                if not output_text.strip():
                    continue

                usage = body.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                llm_cost = (input_tokens * input_price + output_tokens * output_price) / 2

                results.append(
                    {
                        "object_id": custom_id,
                        "description": output_text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_cost": llm_cost,
                    }
                )

    print(f"Processed {len(results)} descriptions")

    # Merge with metadata
    intermediate_path = Path(batch_info["batch_folder"]) / "descriptions_raw.parquet"
    df_metadata = build_metadata_df()
    descriptions_df = pd.DataFrame(results)
    df_merged = df_metadata.merge(descriptions_df, on="object_id", how="inner")

    table = build_descriptions_table(df_merged)
    pq.write_table(table, intermediate_path, compression="snappy")
    print(f"Saved raw descriptions to {intermediate_path}")
    print("\nTo embed descriptions, run:")
    print(f"  uv run python src/experiments/gz10/step2_describe.py --embed {intermediate_path}")


async def submit_openrouter(args):
    """Generate descriptions via OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY is not set. Aborting.")
        return

    prompt = read_prompt(args.prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.no_images:
        # No need to load HF dataset -- get object IDs from local metadata
        df_meta = build_metadata_df()
        if args.limit:
            df_meta = df_meta.head(args.limit)
        image_results = [
            (f"gz10_{idx:06d}", None)
            for idx in df_meta["Galaxy10_DECals_index"].values
        ]
    else:
        # Load images from HF dataset
        ds = load_hf_dataset()
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        png_dir = Path(DATA_DIR) / "pngs"
        image_results = save_pngs_from_hf(ds, png_dir)

    print(f"\nGenerating descriptions for {len(image_results)} images...")
    print(f"Model: {args.model}")
    print(f"Parallel requests: {PARALLEL_REQUESTS}")

    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # Run parallel requests
    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)
    tasks = [
        asyncio.create_task(
            describe_one(
                object_id,
                png_path,
                prompt,
                args.model,
                semaphore,
                client,
                args.no_images,
            )
        )
        for object_id, png_path in image_results
    ]

    results = []
    with tqdm(total=len(tasks), desc="Getting descriptions") as pbar:
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            pbar.update(1)

    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]
    failed_count = len(results) - len(successful_results)

    print(f"\nSuccessful: {len(successful_results)}, Failed: {failed_count}")

    if not successful_results:
        print("No successful descriptions. Exiting.")
        return

    # Calculate token usage
    total_input = sum(r["input_tokens"] for r in successful_results)
    total_output = sum(r["output_tokens"] for r in successful_results)
    print(f"Total tokens - Input: {total_input:,}, Output: {total_output:,}")

    # Save intermediate results
    results_dir = resolve_results_dir(args.results_dir)
    mode = "direct_blind" if args.no_images else "direct"
    results_folder = build_run_dir(
        results_dir,
        args.model,
        "openrouter",
        timestamp,
        args.run_name,
        mode,
    )
    results_folder.mkdir(parents=True, exist_ok=True)

    # Merge with metadata
    df_metadata = build_metadata_df()
    descriptions_data = [
        {"object_id": r["object_id"], "description": r["description"]}
        for r in successful_results
        if r["description"].strip()
    ]
    descriptions_df = pd.DataFrame(descriptions_data)
    df_merged = df_metadata.merge(descriptions_df, on="object_id", how="inner")

    table = build_descriptions_table(df_merged)
    intermediate_path = results_folder / "descriptions_raw.parquet"
    pq.write_table(table, intermediate_path, compression="snappy")
    print(f"\nSaved raw descriptions to {intermediate_path}")
    print("\nTo embed descriptions, run:")
    print(f"  uv run python src/experiments/gz10/step2_describe.py --embed {intermediate_path}")


def embed_descriptions(args):
    """Embed descriptions with text-embedding-3-large."""
    descriptions_file = Path(args.descriptions_file)
    print(f"Loading descriptions from {descriptions_file}...")
    df = pq.read_table(descriptions_file).to_pandas()
    print(f"Loaded {len(df)} descriptions")

    descriptions = df["description"].tolist()

    print("Generating embeddings...")
    generator = OpenAIEmbeddingGenerator()
    embeddings, total_tokens = generator.process_embeddings_with_rate_limit(
        texts=descriptions,
        batch_size=100,
        model="text-embedding-3-large",
        dimensions=None,
        desc="Embedding descriptions",
    )

    embeddings_list = [np.array(e, dtype=np.float32).tolist() for e in embeddings]

    # Save final descriptions parquet
    table = pa.table(
        {
            "ra": pa.array(df["ra"].values, type=pa.float64()),
            "dec": pa.array(df["dec"].values, type=pa.float64()),
            "label": pa.array(df["label"].values, type=pa.int64()),
            "label_name": pa.array(df["label_name"].values, type=pa.string()),
            "Galaxy10_DECals_index": pa.array(df["Galaxy10_DECals_index"].values, type=pa.int64()),
            "object_id": pa.array(df["object_id"].values, type=pa.string()),
            "description": pa.array(df["description"].values, type=pa.string()),
            "description_embedding": pa.array(
                embeddings_list, type=pa.list_(pa.float32())
            ),
        }
    )

    output_path = resolve_embed_output_path(descriptions_file, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved descriptions with embeddings to {output_path}")
    print(f"Total tokens used: {total_tokens:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM descriptions for GZ10")
    parser.add_argument("--submit", action="store_true", help="Submit generation job")
    parser.add_argument(
        "--check", type=str, metavar="BATCH_INFO", help="Check OpenAI batch status and download"
    )
    parser.add_argument(
        "--embed", type=str, metavar="DESCRIPTIONS_FILE", help="Embed descriptions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model ID (OpenAI or OpenRouter)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "openrouter"],
        help="Provider override (default: auto)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="src/prompts/general_promptv4.txt",
        help="Prompt file path",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image creation/upload and send prompt-only requests",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory base",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name suffix for output folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path override for --embed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="CHECKPOINT_FILE",
        help="Resume upload from checkpoint file (upload_checkpoint.jsonl)",
    )

    args = parser.parse_args()

    if args.submit:
        provider = infer_provider(args.model, args.provider)
        if provider == "openai":
            asyncio.run(submit_openai_batch(args))
        else:
            asyncio.run(submit_openrouter(args))
    elif args.check:
        args.batch_info = args.check
        check_batch(args)
    elif args.embed:
        args.descriptions_file = args.embed
        embed_descriptions(args)
    else:
        print("Usage:")
        print("  uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-4.1-mini")
        print("  uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-5-mini")
        print("  uv run python src/experiments/gz10/step2_describe.py --submit --model qwen/qwen3-vl-235b-a22b-thinking")
        print("  uv run python src/experiments/gz10/step2_describe.py --submit --model gpt-4.1-mini --no-images")
        print("  uv run python src/experiments/gz10/step2_describe.py --check BATCH_INFO_FILE")
        print("  uv run python src/experiments/gz10/step2_describe.py --embed DESCRIPTIONS_FILE")
        print("")
        print("Resume interrupted upload:")
        print("  uv run python src/experiments/gz10/step2_describe.py --submit --model MODEL --resume CHECKPOINT_FILE")


if __name__ == "__main__":
    main()
