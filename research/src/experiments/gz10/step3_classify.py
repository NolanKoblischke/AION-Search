"""
Step 3: Run all classification methods.

CLI:
    uv run python src/experiments/gz10/step3_classify.py --aion
    uv run python src/experiments/gz10/step3_classify.py --llm-similarity --descriptions <path>
    uv run python src/experiments/gz10/step3_classify.py --llm-judge --descriptions <path>
    uv run python src/experiments/gz10/step3_classify.py --all --descriptions <path>

Methods:
1. --aion: AION search embedding similarity classification
   Output: data/gz10/gz10_aion_classifications.parquet
2. --llm-similarity: LLM description embedding similarity classification
   Output: <descriptions_dir>/llm_similarity_classifications.parquet
3. --llm-judge: Gemini judge classification from description text
   Output: <descriptions_dir>/llm_judge_classifications.parquet
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.experiments.gz10.constants import (
    AION_CLASSIFICATIONS_PARQUET,
    CLASS_LABELS,
    IMAGE_EMBEDDINGS_PARQUET,
    LABEL_EMBEDDINGS_PARQUET,
)

load_dotenv()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


def print_accuracy(true_labels: np.ndarray, predictions: np.ndarray, method_name: str):
    """Print accuracy summary."""
    accuracy = accuracy_score(true_labels, predictions)
    n_classes = len(CLASS_LABELS)
    uniform_random = 1.0 / n_classes
    print(f"\n{method_name}: {accuracy:.2%} accuracy ({len(true_labels)} samples, {accuracy / uniform_random:.1f}x over random)")


def classify_aion():
    """Classify using AION search embeddings."""
    print("Loading image embeddings...")
    df = pq.read_table(IMAGE_EMBEDDINGS_PARQUET).to_pandas()
    print(f"Loaded {len(df)} samples")

    print("Loading label embeddings...")
    label_df = pq.read_table(LABEL_EMBEDDINGS_PARQUET).to_pandas()

    # Extract embeddings
    galaxy_embeddings = np.array(df["aion_search_embedding"].tolist(), dtype=np.float32)
    label_embeddings = np.array(
        label_df["aion_search_text_embedding"].tolist(), dtype=np.float32
    )

    print(f"Galaxy embeddings shape: {galaxy_embeddings.shape}")
    print(f"Label embeddings shape: {label_embeddings.shape}")

    # Compute similarities
    print("Computing similarities...")
    similarities = cosine_similarity(galaxy_embeddings, label_embeddings)
    predictions = np.argmax(similarities, axis=1)
    true_labels = df["label"].values

    # Save results
    output_dir = Path(AION_CLASSIFICATIONS_PARQUET).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "ra": pa.array(df["ra"].values, type=pa.float64()),
            "dec": pa.array(df["dec"].values, type=pa.float64()),
            "label": pa.array(true_labels.astype(np.int64), type=pa.int64()),
            "label_name": pa.array(
                [CLASS_LABELS.get(int(l), "Unknown") for l in true_labels],
                type=pa.string(),
            ),
            "predicted_label": pa.array(predictions.astype(np.int64), type=pa.int64()),
            "predicted_label_name": pa.array(
                [CLASS_LABELS.get(int(p), "Unknown") for p in predictions],
                type=pa.string(),
            ),
            "label_similarities": pa.array(
                [row.astype(np.float32).tolist() for row in similarities],
                type=pa.list_(pa.float32()),
            ),
            "aion_mean_embedding": df["aion_mean_embedding"].tolist(),
            "aion_search_embedding": df["aion_search_embedding"].tolist(),
        }
    )

    pq.write_table(table, AION_CLASSIFICATIONS_PARQUET, compression="snappy")
    print(f"Saved to {AION_CLASSIFICATIONS_PARQUET}")

    print_accuracy(true_labels, predictions, "AION Classification")


def classify_llm_similarity(descriptions_path: Path):
    """Classify using LLM description embeddings."""
    print(f"Loading descriptions from {descriptions_path}...")
    df = pq.read_table(descriptions_path).to_pandas()
    print(f"Loaded {len(df)} samples")

    print("Loading label embeddings...")
    label_df = pq.read_table(LABEL_EMBEDDINGS_PARQUET).to_pandas()

    # Extract embeddings
    description_embeddings = np.array(
        df["description_embedding"].tolist(), dtype=np.float32
    )
    label_embeddings = np.array(
        label_df["text_embedding_3_large"].tolist(), dtype=np.float32
    )

    print(f"Description embeddings shape: {description_embeddings.shape}")
    print(f"Label embeddings shape: {label_embeddings.shape}")

    # Compute similarities
    print("Computing similarities...")
    similarities = cosine_similarity(description_embeddings, label_embeddings)
    predictions = np.argmax(similarities, axis=1)
    true_labels = df["label"].values

    # Save results to same directory as descriptions
    output_dir = descriptions_path.parent
    output_path = output_dir / "llm_similarity_classifications.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "ra": pa.array(df["ra"].values, type=pa.float64()),
            "dec": pa.array(df["dec"].values, type=pa.float64()),
            "label": pa.array(true_labels.astype(np.int64), type=pa.int64()),
            "label_name": pa.array(
                [CLASS_LABELS.get(int(l), "Unknown") for l in true_labels],
                type=pa.string(),
            ),
            "predicted_label": pa.array(predictions.astype(np.int64), type=pa.int64()),
            "predicted_label_name": pa.array(
                [CLASS_LABELS.get(int(p), "Unknown") for p in predictions],
                type=pa.string(),
            ),
            "label_similarities": pa.array(
                [row.astype(np.float32).tolist() for row in similarities],
                type=pa.list_(pa.float32()),
            ),
            "description": pa.array(df["description"].values, type=pa.string()),
        }
    )

    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved to {output_path}")

    print_accuracy(true_labels, predictions, "LLM Similarity Classification")


class GalaxyClassification(BaseModel):
    """Classification result for a galaxy description."""

    classification: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] = Field(
        ..., description="The class index (0-9) that best matches the galaxy description"
    )


def build_classification_prompt(description: str) -> str:
    """Build the prompt for galaxy classification."""
    class_descriptions = "\n".join(
        [f"  {idx}: {name}" for idx, name in CLASS_LABELS.items()]
    )
    return f"""<task>You are an expert astronomer classifying galaxies based on their descriptions.
Classify the following galaxy description into one of 10 morphological classes.
Based on the descriptions, classify the image into the most appropriate class (0-9).
</task>
<classes>
{class_descriptions}
</classes>
<description>
{description}
</description>"""


def classify_single_description(args: tuple) -> dict:
    """Classify a single description using Gemini."""
    idx, description, label, model, ra, dec, galaxy10_index = args

    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = build_classification_prompt(description)

    config = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GalaxyClassification,
    )

    result = {
        "idx": idx,
        "ra": ra,
        "dec": dec,
        "Galaxy10_DECals_index": galaxy10_index,
        "true_label": label,
        "predicted_label": None,
        "error": None,
    }

    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        parsed = response.parsed

        if parsed is None:
            result["error"] = "Null parsed result"
            return result

        if isinstance(parsed, dict):
            classification = GalaxyClassification(**parsed)
        else:
            classification = parsed

        result["predicted_label"] = int(classification.classification)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            result["error"] = f"Rate limited: {error_msg}"
        else:
            result["error"] = error_msg

    return result


def classify_llm_judge(descriptions_path: Path, model: str = "gemini-2.5-flash", cores: int = 10):
    """Classify using Gemini as a judge."""
    print(f"Loading descriptions from {descriptions_path}...")
    df = pq.read_table(descriptions_path).to_pandas()
    print(f"Loaded {len(df)} samples")
    print(f"Model: {model}")
    print(f"Parallel workers: {cores}")

    # Prepare arguments
    args_list = [
        (
            i,
            df.iloc[i]["description"],
            df.iloc[i]["label"],
            model,
            df.iloc[i]["ra"],
            df.iloc[i]["dec"],
            df.iloc[i]["Galaxy10_DECals_index"],
        )
        for i in range(len(df))
    ]

    # Process with multiprocessing
    print(f"\nClassifying with {cores} workers...")
    results = []
    error_count = 0

    with Pool(cores) as pool:
        for result in tqdm(
            pool.imap_unordered(classify_single_description, args_list),
            total=len(args_list),
            desc="Classifying descriptions",
        ):
            results.append(result)
            if result["error"]:
                error_count += 1

    # Sort by index
    results.sort(key=lambda x: x["idx"])

    print(f"\nProcessed {len(results)} samples")
    print(f"Errors: {error_count}")

    valid_results = [r for r in results if r["predicted_label"] is not None]
    print(f"Valid predictions: {len(valid_results)}")

    if len(valid_results) == 0:
        print("No valid predictions. Check for API errors.")
        return

    # Extract predictions
    true_labels = np.array([r["true_label"] for r in valid_results])
    predictions = np.array([r["predicted_label"] for r in valid_results])

    # Save results to same directory as descriptions
    output_dir = descriptions_path.parent
    output_path = output_dir / "llm_judge_classifications.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build table from valid results
    table = pa.table(
        {
            "ra": pa.array([r["ra"] for r in valid_results], type=pa.float64()),
            "dec": pa.array([r["dec"] for r in valid_results], type=pa.float64()),
            "label": pa.array(true_labels.astype(np.int64), type=pa.int64()),
            "label_name": pa.array(
                [CLASS_LABELS.get(int(l), "Unknown") for l in true_labels],
                type=pa.string(),
            ),
            "predicted_label": pa.array(predictions.astype(np.int64), type=pa.int64()),
            "predicted_label_name": pa.array(
                [CLASS_LABELS.get(int(p), "Unknown") for p in predictions],
                type=pa.string(),
            ),
            "description": pa.array(
                [df.iloc[r["idx"]]["description"] for r in valid_results],
                type=pa.string(),
            ),
        }
    )

    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved to {output_path}")

    print_accuracy(true_labels, predictions, "LLM Judge Classification")


def main():
    parser = argparse.ArgumentParser(description="Classify GZ10 galaxies")
    parser.add_argument(
        "--aion", action="store_true", help="Run AION similarity classification"
    )
    parser.add_argument(
        "--llm-similarity",
        action="store_true",
        help="Run LLM embedding similarity classification",
    )
    parser.add_argument(
        "--llm-judge", action="store_true", help="Run Gemini judge classification"
    )
    parser.add_argument("--all", action="store_true", help="Run all classification methods")
    parser.add_argument(
        "--descriptions",
        type=str,
        help="Path to descriptions parquet (required for --llm-similarity and --llm-judge)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model for judge classification",
    )
    parser.add_argument(
        "--cores", type=int, default=10, help="Number of parallel workers for judge"
    )

    args = parser.parse_args()

    if args.all:
        args.aion = True
        args.llm_similarity = True
        args.llm_judge = True

    # Validate descriptions path for LLM methods
    if (args.llm_similarity or args.llm_judge) and not args.descriptions:
        print("Error: --descriptions is required for --llm-similarity and --llm-judge")
        print(
            "Example: --descriptions data/gz10/llm_results/gpt-4.1-mini_openai_batch_YYYYMMDD_HHMMSS/descriptions_with_embeddings.parquet"
        )
        return

    descriptions_path = Path(args.descriptions) if args.descriptions else None

    if args.aion:
        print("\n" + "=" * 60)
        print("Running AION Classification")
        print("=" * 60)
        classify_aion()

    if args.llm_similarity:
        print("\n" + "=" * 60)
        print("Running LLM Similarity Classification")
        print("=" * 60)
        classify_llm_similarity(descriptions_path)

    if args.llm_judge:
        print("\n" + "=" * 60)
        print("Running LLM Judge Classification")
        print("=" * 60)
        classify_llm_judge(descriptions_path, model=args.judge_model, cores=args.cores)

    if not (args.aion or args.llm_similarity or args.llm_judge):
        print("Usage:")
        print("  uv run python src/experiments/gz10/step3_classify.py --aion")
        print("  uv run python src/experiments/gz10/step3_classify.py --llm-similarity --descriptions <path>")
        print("  uv run python src/experiments/gz10/step3_classify.py --llm-judge --descriptions <path>")
        print("  uv run python src/experiments/gz10/step3_classify.py --all --descriptions <path>")


if __name__ == "__main__":
    main()
