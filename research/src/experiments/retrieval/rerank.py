"""
VLM-based reranking for retrieval evaluation experiments.

Reranks images across all folds and computes average metrics.
Supports lens, spiral, and merger datasets.

CLI:
    uv run python src/experiments/retrieval/rerank.py --images-dir data/retrieval_results/full/rerank/default/lens/images --dataset lens --model gpt-4.1-nano
    uv run python src/experiments/retrieval/rerank.py --images-dir data/retrieval_results/full/rerank/default/spiral/images --dataset spiral --model gpt-4.1
"""

import argparse
import base64
import io
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from src.evals.eval_utils import setup_logging

load_dotenv()


class GalaxyRanking(BaseModel):
    """Structured output for galaxy ranking."""
    ranking: int


def load_models_info(models_file: str = "src/utils/models.jsonl") -> dict:
    """Load model pricing information from JSONL file."""
    models = {}
    with open(models_file, 'r') as f:
        for line in f:
            if line.strip():
                model_data = json.loads(line)
                models[model_data['id']] = model_data
    return models


def dcg(r: np.ndarray) -> float:
    """Compute Discounted Cumulative Gain (DCG)."""
    return float(np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2))))


def ndcg_score(relevances: np.ndarray, k: int, all_relevances: np.ndarray = None) -> float:
    """Compute NDCG@k.

    Args:
        relevances: Relevance scores in ranked order (for actual DCG).
        k: Number of top results to consider.
        all_relevances: Full population relevances for ideal DCG.
            If None, uses relevances (only correct when relevances covers the full population).
    """
    actual_dcg = dcg(relevances[:k])
    ideal_relevances = all_relevances if all_relevances is not None else relevances
    ideal_dcg = dcg(np.sort(ideal_relevances)[::-1][:k])
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def center_zoom(image: Image.Image, zoom_factor: float) -> Image.Image:
    """Apply center zoom to an image.

    Extracts the center (S/Z)x(S/Z) region and scales it back to original size.
    """
    if zoom_factor <= 1.0:
        return image

    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def encode_image(image_path: str, zoom_factor: float = 1.0) -> str:
    """Encode image to base64, optionally applying center zoom."""
    if zoom_factor > 1.0:
        img = Image.open(image_path)
        img = center_zoom(img, zoom_factor)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def get_provider(model_name: str) -> str:
    """Determine the API provider based on model name."""
    if model_name.startswith("gemini"):
        return "gemini"
    return "openai"


def get_question_for_dataset(dataset: str) -> str:
    """Get the appropriate VLM question for each dataset type."""
    questions = {
        'lens': "Does this galaxy image display signs of gravitational lensing? Rank 1-10 where 10 means you are entirely sure there are signs of gravitational lensing and 1 being you are entirely sure there are no signs of gravitational lensing.",
        'spiral': "Does this galaxy have visible spiral arms? Rank 1-10 where 10 means you are entirely sure there are spiral arms and 1 being you are entirely sure there are no spiral arms.",
        'merger': "Does this galaxy image show signs of actively merging galaxies? Rank 1-10 where 10 means you are entirely sure it displays active merging and 1 being you are entirely sure it does not display active merging.",
    }
    return questions[dataset]


def process_single_image(args: tuple) -> tuple:
    """Process a single image for ranking."""
    index, image_path, api_key, model_name, question, reasoning_effort, thinking_budget, zoom_factor = args

    try:
        ranking, input_tokens, output_tokens = get_vlm_ranking(
            image_path, api_key, model_name, question, reasoning_effort, thinking_budget, zoom_factor
        )
        return index, ranking, input_tokens, output_tokens
    except Exception as e:
        logging.error(f"Error processing image {index}: {e}")
        return index, 5, 0, 0


MAX_RETRIES = 3
RETRY_DELAY = 2
TASK_TIMEOUT = 60  # seconds per image


def get_vlm_ranking_openai(
    image_path: str, api_key: str, model: str, question: str,
    reasoning_effort: str = None, zoom_factor: float = 1.0
) -> tuple:
    """Get VLM ranking for a galaxy image using OpenAI."""
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_path, zoom_factor)

    request_kwargs = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    },
                    {
                        "type": "input_text",
                        "text": question
                    }
                ]
            }
        ],
        "tools": [],
        "store": True,
        "text_format": GalaxyRanking
    }
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.parse(**request_kwargs)

            parsed_response = getattr(response, 'output_parsed', None)
            if parsed_response is None:
                raise ValueError("API returned empty or null response")

            ranking = parsed_response.ranking

            input_tokens = getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0
            output_tokens = getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0

            return ranking, input_tokens or 0, output_tokens or 0
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise


def get_vlm_ranking_gemini(
    image_path: str, api_key: str, model: str, question: str,
    thinking_budget: int = None, zoom_factor: float = 1.0
) -> tuple:
    """Get VLM ranking for a galaxy image using Gemini."""
    # Add HTTP timeout to prevent hanging
    http_options = types.HttpOptions(timeout=TASK_TIMEOUT * 1000)  # timeout in ms
    client = genai.Client(api_key=api_key, http_options=http_options)

    if zoom_factor > 1.0:
        img = Image.open(image_path)
        img = center_zoom(img, zoom_factor)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    parts = [
        types.Part.from_bytes(mime_type="image/png", data=image_bytes),
        types.Part.from_text(text=question),
    ]

    contents = [types.Content(role="user", parts=parts)]

    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": GalaxyRanking,
    }
    if thinking_budget is not None and thinking_budget != -1:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

            response_text = response.text if hasattr(response, 'text') else None
            if response_text is None or response_text.strip() == "":
                raise ValueError("API returned empty or null response")

            parsed = GalaxyRanking.model_validate_json(response_text)
            ranking = parsed.ranking

            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0

            # Include thinking tokens if available
            if (hasattr(response, 'usage_metadata') and
                hasattr(response.usage_metadata, 'thoughts_token_count') and
                response.usage_metadata.thoughts_token_count is not None and
                output_tokens is not None):
                output_tokens = output_tokens + response.usage_metadata.thoughts_token_count

            return ranking, input_tokens or 0, output_tokens or 0
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise


def get_vlm_ranking(
    image_path: str, api_key: str, model: str, question: str,
    reasoning_effort: str = None, thinking_budget: int = None, zoom_factor: float = 1.0
) -> tuple:
    """Get VLM ranking for a galaxy image. Dispatches to appropriate provider."""
    provider = get_provider(model)

    if provider == "gemini":
        return get_vlm_ranking_gemini(image_path, api_key, model, question, thinking_budget, zoom_factor)
    else:
        return get_vlm_ranking_openai(image_path, api_key, model, question, reasoning_effort, zoom_factor)


def rerank_fold(
    fold_dir: Path,
    dataset: str,
    api_key: str,
    model_info: dict,
    max_workers: int,
    k_values: list,
    logger: logging.Logger,
    all_relevances: np.ndarray = None,
    zoom_factor: float = 1.0,
) -> dict:
    """Rerank a single fold and compute metrics.

    Args:
        all_relevances: Full fold relevances for ideal DCG computation.
            If None, uses only the top-k images' relevances (incorrect for proper NDCG).
    """

    model_name = model_info['model_name']
    reasoning_effort = model_info.get('reasoning_effort', None)
    thinking_budget = model_info.get('thinking_budget', None)
    input_price = model_info['input_price']
    output_price = model_info['output_price']

    # Load metadata
    metadata_path = fold_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.warning(f"No metadata found in {fold_dir}")
        return None

    metadata_df = pd.read_csv(metadata_path)
    n_images = len(metadata_df)

    # Find image files
    image_files = sorted(fold_dir.glob("rank_*.png"))
    if not image_files:
        logger.warning(f"No images found in {fold_dir}")
        return None

    # Create mapping from rank to image path
    rank_to_image = {}
    for img_path in image_files:
        parts = img_path.stem.split("_")
        rank = int(parts[1])
        rank_to_image[rank] = str(img_path)

    # Get question for this dataset
    question = get_question_for_dataset(dataset)

    # Prepare arguments for parallel processing
    process_args = []
    for rank in range(1, n_images + 1):
        if rank in rank_to_image:
            process_args.append((
                rank - 1,  # 0-indexed
                rank_to_image[rank],
                api_key,
                model_name,
                question,
                reasoning_effort,
                thinking_budget,
                zoom_factor,
            ))

    # Process images in parallel
    vlm_scores = np.zeros(n_images)
    total_input_tokens = 0
    total_output_tokens = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, args): args[0] for args in process_args}
        completed_indices = set()

        with tqdm(total=len(process_args), desc=f"  Images", leave=False) as pbar:
            try:
                # Set overall timeout: TASK_TIMEOUT per image, with buffer
                overall_timeout = len(process_args) * TASK_TIMEOUT / max_workers + 300
                for future in as_completed(futures, timeout=overall_timeout):
                    index = futures[future]
                    try:
                        index, ranking, input_tokens, output_tokens = future.result(timeout=TASK_TIMEOUT)
                        vlm_scores[index] = ranking
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        completed_indices.add(index)
                    except TimeoutError:
                        logger.warning(f"Timeout processing image {index}, using default score 5")
                        vlm_scores[index] = 5
                        completed_indices.add(index)
                    except Exception as e:
                        logger.error(f"Error processing image {index}: {e}")
                        vlm_scores[index] = 5
                        completed_indices.add(index)
                    pbar.update(1)
            except TimeoutError:
                # Handle overall timeout - mark remaining as default
                for future, index in futures.items():
                    if index not in completed_indices:
                        logger.warning(f"Overall timeout, image {index} using default score 5")
                        vlm_scores[index] = 5
                        pbar.update(1)
                        future.cancel()

    # Calculate cost
    total_cost = (total_input_tokens * input_price + total_output_tokens * output_price) / 1_000_000

    # Get original relevance scores and similarities
    similarities = metadata_df['similarity'].values
    relevances = metadata_df['relevance'].values

    # Reranked order: by VLM score (desc), then by similarity (desc) for ties
    rerank_keys = [(-vlm_scores[i], -similarities[i]) for i in range(n_images)]
    reranked_order = np.array(sorted(range(n_images), key=lambda i: rerank_keys[i]))

    # Compute metrics
    results = {
        'n_images': n_images,
        'n_scored': int(np.sum(vlm_scores > 0)),
        'total_cost': float(total_cost),
        'total_input_tokens': int(total_input_tokens),
        'total_output_tokens': int(total_output_tokens),
    }

    # Reranked relevances
    reranked_relevances = relevances[reranked_order]

    for k in k_values:
        if k <= n_images:
            orig_ndcg = ndcg_score(relevances, k, all_relevances)
            reranked_ndcg = ndcg_score(reranked_relevances, k, all_relevances)

            results[f'original_ndcg@{k}'] = float(orig_ndcg)
            results[f'reranked_ndcg@{k}'] = float(reranked_ndcg)

    # Build results dataframe with new_rank and llm_score columns
    new_ranks = np.zeros(n_images, dtype=int)
    for new_rank_idx, old_idx in enumerate(reranked_order):
        new_ranks[old_idx] = new_rank_idx + 1

    results_df = pd.DataFrame({
        'old_rank': metadata_df['rank'].values,
        'fold_index': metadata_df['fold_index'].values,
        'global_index': metadata_df['global_index'].values,
        'ra': metadata_df['ra'].values,
        'dec': metadata_df['dec'].values,
        'similarity': metadata_df['similarity'].values,
        'relevance': relevances,
        'new_rank': new_ranks,
        'llm_score': vlm_scores,
    })

    return results, results_df


def main():
    parser = argparse.ArgumentParser(description="VLM reranking for k-fold experiments")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing fold subdirectories with images")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["lens", "spiral", "merger"],
                        help="Dataset type (lens, spiral, merger)")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano",
                        help="Model ID from models.jsonl")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Max parallel workers for VLM calls")
    parser.add_argument("--k-values", type=int, nargs='+', default=[10],
                        help="k values for NDCG metrics")
    parser.add_argument("--folds", type=int, nargs='+', default=None,
                        help="Specific folds to process (default: all)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <images-dir>/../<model>)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Center zoom factor (e.g., 2.0 for 2x zoom)")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load model info
    models_info = load_models_info()
    if args.model not in models_info:
        raise ValueError(f"Model {args.model} not found in models.jsonl")
    model_info = models_info[args.model]

    # Get API key based on provider
    provider = get_provider(model_info['model_name'])
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

    logger.info(f"Using model: {model_info['formatted_name']} (provider: {provider})")
    if args.zoom > 1.0:
        logger.info(f"Using {args.zoom}x center zoom")

    images_dir = Path(args.images_dir)
    dataset = args.dataset

    logger.info(f"Dataset: {dataset}")

    # Load objects CSV from config.json for full-population ideal DCG
    config_path = images_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {images_dir}")
    with open(config_path) as f:
        config = json.load(f)
    if 'objects_csv' not in config:
        raise ValueError(f"objects_csv not found in {config_path}. Re-run download_images.py.")
    objects_df = pd.read_csv(config['objects_csv'])
    logger.info(f"Loaded {len(objects_df)} objects from {config['objects_csv']}")

    # Build per-fold full relevances for ideal DCG
    fold_all_relevances = {}
    for fold_idx, group in objects_df.groupby('fold_index'):
        fold_all_relevances[int(fold_idx)] = group['relevance'].values

    # Find fold directories
    fold_dirs = sorted(images_dir.glob("fold_*"))
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {images_dir}")

    # Filter to requested folds
    if args.folds is not None:
        fold_dirs = [d for d in fold_dirs if int(d.name.split("_")[1]) in args.folds]

    logger.info(f"Found {len(fold_dirs)} fold directories")

    # Create output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = images_dir.parent / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each fold
    all_fold_results = []
    all_fold_dfs = []
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    n_images_total = 0

    for fold_dir in tqdm(fold_dirs, desc="Processing folds"):
        fold_num = int(fold_dir.name.split("_")[1])
        logger.info(f"\nProcessing fold {fold_num}")

        result = rerank_fold(
            fold_dir=fold_dir,
            dataset=dataset,
            api_key=api_key,
            model_info=model_info,
            max_workers=args.max_workers,
            k_values=args.k_values,
            logger=logger,
            all_relevances=fold_all_relevances.get(fold_num),
            zoom_factor=args.zoom,
        )

        if result is not None:
            fold_results, fold_df = result
            fold_results['fold'] = fold_num
            all_fold_results.append(fold_results)
            total_cost += fold_results['total_cost']
            total_input_tokens += fold_results['total_input_tokens']
            total_output_tokens += fold_results['total_output_tokens']
            n_images_total += fold_results['n_images']

            # Add fold column to dataframe
            fold_df.insert(0, 'fold', fold_num)
            all_fold_dfs.append(fold_df)

            # Print fold summary
            for k in args.k_values:
                if f'original_ndcg@{k}' in fold_results:
                    orig = fold_results[f'original_ndcg@{k}']
                    reranked = fold_results[f'reranked_ndcg@{k}']
                    logger.info(f"  Fold {fold_num} NDCG@{k}: {orig:.4f} -> {reranked:.4f} ({reranked - orig:+.4f})")

    # Build combined results.csv (ordered by fold, then new_rank)
    if all_fold_dfs:
        combined_df = pd.concat(all_fold_dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['fold', 'new_rank']).reset_index(drop=True)
        combined_df.to_csv(output_dir / "results.csv", index=False)

    # Build metadata.json
    input_price = model_info['input_price']
    output_price = model_info['output_price']

    ndcg_results = {}
    for k in args.k_values:
        orig_key = f'original_ndcg@{k}'
        rerank_key = f'reranked_ndcg@{k}'

        if all_fold_results and orig_key in all_fold_results[0]:
            orig_values = [r[orig_key] for r in all_fold_results]
            rerank_values = [r[rerank_key] for r in all_fold_results]

            per_fold = []
            for r in all_fold_results:
                per_fold.append({
                    'fold': r['fold'],
                    'original': r[orig_key],
                    'reranked': r[rerank_key],
                })

            ndcg_results[str(k)] = {
                'mean_original': float(np.mean(orig_values)),
                'std_original': float(np.std(orig_values)),
                'mean_reranked': float(np.mean(rerank_values)),
                'std_reranked': float(np.std(rerank_values)),
                'per_fold': per_fold,
            }

    metadata = {
        'model_id': args.model,
        'model_name': model_info['formatted_name'],
        'provider': provider,
        'dataset': dataset,
        'zoom_factor': args.zoom,
        'n_folds': len(all_fold_results),
        'n_images_total': n_images_total,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_price_per_million': input_price,
        'output_price_per_million': output_price,
        'total_cost': round(total_cost, 4),
        'k_values': args.k_values,
        'ndcg_results': ndcg_results,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"RERANKING SUMMARY - {dataset.upper()}")
    print(f"{'='*70}")
    print(f"Model: {model_info['formatted_name']}")
    print(f"Folds processed: {len(all_fold_results)}")
    print(f"Total images: {n_images_total}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"\nResults:")

    for k in args.k_values:
        k_str = str(k)
        if k_str in ndcg_results:
            nr = ndcg_results[k_str]
            print(f"\n  NDCG@{k}:")
            print(f"    Original:    {nr['mean_original']:.4f} +/- {nr['std_original']:.4f}")
            print(f"    Reranked:    {nr['mean_reranked']:.4f} +/- {nr['std_reranked']:.4f}")
            print(f"    Improvement: {nr['mean_reranked'] - nr['mean_original']:+.4f}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
