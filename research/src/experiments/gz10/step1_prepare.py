"""
Step 1: Generate AION and CLIP embeddings from the Galaxy10 HF dataset.

This script:
1. Loads the Galaxy10 AION benchmark dataset from HuggingFace
2. Deserializes 4-band (g,r,i,z) cutouts
3. Encodes cutouts with AION model (mean-pooled embeddings)
4. Projects through CLIP image_projector for search embeddings
5. Embeds label names with text-embedding-3-large
6. Projects label embeddings through CLIP text_projector

CLI:
    uv run python src/experiments/gz10/step1_prepare.py
    uv run python src/experiments/gz10/step1_prepare.py --split train
    uv run python src/experiments/gz10/step1_prepare.py --batch-size 64

Outputs:
- data/gz10/gz10_image_embeddings.parquet
- data/gz10/gz10_label_embeddings.parquet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage
from aion.model import AION

from src.experiments.gz10.constants import (
    CLASS_LABELS,
    CLIP_CHECKPOINT,
    IMAGE_EMBEDDINGS_PARQUET,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    LABEL_EMBEDDINGS_PARQUET,
)
from src.experiments.gz10.model_utils import load_clip_model

load_dotenv()

BAND_NAMES = ["DES-G", "DES-R", "DES-I", "DES-Z"]
CUTOUT_SHAPE = (4, 96, 96)


def deserialize_cutouts(dataset) -> np.ndarray:
    """Convert image_bands from HF dataset to numpy array of shape (N, 4, 96, 96)."""
    cutouts = []
    for row in tqdm(dataset, desc="Loading cutouts"):
        arr = np.array(row["image_bands"], dtype=np.float32)
        cutouts.append(arr)
    return np.stack(cutouts, axis=0)


def encode_with_aion(
    cutouts: np.ndarray, device: str, batch_size: int = 32
) -> np.ndarray:
    """Encode cutouts with AION model and return mean-pooled embeddings."""
    print("Loading AION model and codec...")
    codec_manager = CodecManager(device=device)
    aion = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    aion.eval()

    all_embeddings = []
    print("Encoding with AION...")
    with torch.no_grad():
        for i in tqdm(range(0, len(cutouts), batch_size)):
            batch = cutouts[i : i + batch_size]
            image_flux = torch.tensor(batch.astype("float32")).to(device)
            image = LegacySurveyImage(flux=image_flux, bands=BAND_NAMES)
            tokens = codec_manager.encode(image)
            tok_image = tokens["tok_image"]

            embeddings = aion.encode({"tok_image": tok_image}, num_encoder_tokens=600)
            mean_embeddings = embeddings.mean(dim=1)
            all_embeddings.append(mean_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def project_with_clip(embeddings: np.ndarray, device: str, model_path: str = CLIP_CHECKPOINT) -> np.ndarray:
    """Project AION embeddings through CLIP image projector."""
    print("Loading CLIP model...")
    model, _ = load_clip_model(model_path, device)

    print("Generating search embeddings...")
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        search_embeddings = model.image_projector(embeddings_tensor).cpu().numpy()

    return search_embeddings


def embed_labels_with_openai() -> tuple[np.ndarray, list[str]]:
    """Embed class label names with OpenAI text-embedding-3-large."""
    print("Embedding class labels with OpenAI...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    label_texts = [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]
    response = client.embeddings.create(input=label_texts, model="text-embedding-3-large")

    embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
    return embeddings, label_texts


def project_text_with_clip(
    text_embeddings: np.ndarray, device: str, model_path: str = CLIP_CHECKPOINT
) -> np.ndarray:
    """Project text embeddings through CLIP text projector."""
    print("Loading CLIP model for text projection...")
    model, _ = load_clip_model(model_path, device)

    print("Projecting text embeddings...")
    with torch.no_grad():
        text_tensor = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
        projected = model.text_projector(text_tensor)
        return projected.cpu().numpy()


def save_cutouts_parquet(
    dataset,
    aion_mean: np.ndarray,
    aion_search: np.ndarray,
    output_path: str,
):
    """Save metadata and embeddings to parquet."""
    table = pa.table(
        {
            "ra": pa.array(dataset["ra"], type=pa.float64()),
            "dec": pa.array(dataset["dec"], type=pa.float64()),
            "label": pa.array(dataset["label"], type=pa.int64()),
            "label_name": pa.array(dataset["label_name"], type=pa.string()),
            "Galaxy10_DECals_index": pa.array(
                dataset["Galaxy10_DECals_index"], type=pa.int64()
            ),
            "aion_mean_embedding": pa.array(
                [row.astype(np.float32).tolist() for row in aion_mean],
                type=pa.list_(pa.float32()),
            ),
            "aion_search_embedding": pa.array(
                [row.astype(np.float32).tolist() for row in aion_search],
                type=pa.list_(pa.float32()),
            ),
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved {len(table)} rows to {output_path}")


def save_label_embeddings_parquet(
    label_texts: list[str],
    text_embeddings: np.ndarray,
    aion_search_text: np.ndarray,
    output_path: str,
):
    """Save label embeddings to parquet."""
    table = pa.table(
        {
            "label": pa.array(list(range(len(label_texts))), type=pa.int64()),
            "label_name": pa.array(label_texts, type=pa.string()),
            "text_embedding_3_large": pa.array(
                [row.astype(np.float32).tolist() for row in text_embeddings],
                type=pa.list_(pa.float32()),
            ),
            "aion_search_text_embedding": pa.array(
                [row.astype(np.float32).tolist() for row in aion_search_text],
                type=pa.list_(pa.float32()),
            ),
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved label embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GZ10 embeddings")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for AION encoding"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="HF dataset split to process (default: test)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=CLIP_CHECKPOINT,
        help=(
            "Optional local AION-Search PyTorch checkpoint. If this path does "
            "not exist, weights are downloaded from astronolan/aion-search."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset from HuggingFace
    print(f"Loading {HF_DATASET_REPO} split={args.split}...")
    ds = load_dataset(HF_DATASET_REPO, split=args.split, revision=HF_DATASET_REVISION)
    print(f"Loaded {len(ds)} galaxies")

    # Deserialize cutouts
    cutouts = deserialize_cutouts(ds)
    print(f"Cutouts shape: {cutouts.shape}")

    # Encode with AION
    aion_mean = encode_with_aion(cutouts, device, batch_size=args.batch_size)
    print(f"AION embeddings shape: {aion_mean.shape}")

    # Project through CLIP
    aion_search = project_with_clip(aion_mean, device, model_path=args.model_path)
    print(f"Search embeddings shape: {aion_search.shape}")

    # Save cutouts parquet
    save_cutouts_parquet(ds, aion_mean, aion_search, IMAGE_EMBEDDINGS_PARQUET)

    # Embed and project labels
    label_embeddings, label_texts = embed_labels_with_openai()
    aion_search_text = project_text_with_clip(label_embeddings, device, model_path=args.model_path)

    # Save label embeddings
    save_label_embeddings_parquet(
        label_texts, label_embeddings, aion_search_text, LABEL_EMBEDDINGS_PARQUET
    )

    print("\n" + "=" * 50)
    print("Step 1 complete!")
    print(f"  Galaxies: {len(ds)}")
    print(f"  Split: {args.split}")
    print(f"  Image embeddings: {IMAGE_EMBEDDINGS_PARQUET}")
    print(f"  Label embeddings: {LABEL_EMBEDDINGS_PARQUET}")


if __name__ == "__main__":
    main()
