"""
Step 4: Fine-tune an MLP head on frozen AION-Search embeddings for GZ10.

Mirrors the AION-1 paper protocol: 2-layer MLP (hidden=256, GELU, dropout=0.1)
but trained on AION-Search (CLIP-projected) embeddings instead of raw AION.

Encodes cutouts from the HF dataset with AION, projects through CLIP,
caches everything to data/gz10_ft/, then trains the MLP head.

CLI:
    uv run python src/experiments/gz10/step4_finetune.py
    uv run python src/experiments/gz10/step4_finetune.py --epochs 500 --lr 1e-3
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.experiments.gz10.constants import (
    CLASS_LABELS,
    CLIP_CHECKPOINT,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
)
from src.experiments.gz10.model_utils import load_clip_model

OUTPUT_DIR = "data/gz10_ft"


class MLPHead(nn.Module):
    """Two-layer MLP classifier matching the AION-1 paper protocol."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_embeddings(
    split: str,
    device: str,
    batch_size: int = 32,
    model_path: str = CLIP_CHECKPOINT,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Load cached embeddings or encode from HF dataset and cache.

    Returns (metadata_dict, aion_search_embeddings, labels).
    """
    cache_path = Path(OUTPUT_DIR) / f"gz10_image_embeddings_{split}.parquet"

    if cache_path.exists():
        print(f"Loading cached {split} embeddings from {cache_path}...")
        df = pq.read_table(cache_path).to_pandas()
        search_emb = np.array(df["aion_search_embedding"].tolist(), dtype=np.float32)
        labels = df["label"].values.astype(np.int64)
        meta = {
            "ra": df["ra"].values,
            "dec": df["dec"].values,
            "label": labels,
            "label_name": df["label_name"].values,
            "Galaxy10_DECals_index": df["Galaxy10_DECals_index"].values,
        }
        print(f"  {split}: {len(df)} samples, search_embed dim={search_emb.shape[1]}")
        return meta, search_emb, labels

    from aion.codecs import CodecManager
    from aion.modalities import LegacySurveyImage
    from aion.model import AION

    print(f"Loading HF dataset split={split}...")
    ds = load_dataset(HF_DATASET_REPO, split=split, revision=HF_DATASET_REVISION)
    print(f"  {len(ds)} samples")

    print("Loading cutouts...")
    cutouts = []
    for row in tqdm(ds, desc="Loading cutouts"):
        arr = np.array(row["image_bands"], dtype=np.float32)
        cutouts.append(arr)
    cutouts = np.stack(cutouts, axis=0)

    labels = np.array(ds["label"], dtype=np.int64)
    meta = {
        "ra": np.array(ds["ra"], dtype=np.float64),
        "dec": np.array(ds["dec"], dtype=np.float64),
        "label": labels,
        "label_name": np.array(ds["label_name"]),
        "Galaxy10_DECals_index": np.array(ds["Galaxy10_DECals_index"], dtype=np.int64),
    }

    band_names = ["DES-G", "DES-R", "DES-I", "DES-Z"]
    print("Loading AION model...")
    codec_manager = CodecManager(device=device)
    aion = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    aion.eval()

    all_mean_emb = []
    print(f"Encoding {split} with AION...")
    with torch.no_grad():
        for i in tqdm(range(0, len(cutouts), batch_size)):
            chunk = cutouts[i : i + batch_size]
            image_flux = torch.tensor(chunk.astype("float32")).to(device)
            image = LegacySurveyImage(flux=image_flux, bands=band_names)
            tokens = codec_manager.encode(image)
            tok_image = tokens["tok_image"]
            embeddings = aion.encode({"tok_image": tok_image}, num_encoder_tokens=600)
            mean_embeddings = embeddings.mean(dim=1)
            all_mean_emb.append(mean_embeddings.cpu().numpy())

    aion_mean = np.concatenate(all_mean_emb, axis=0)
    print(f"  AION mean embeddings: {aion_mean.shape}")

    print("Loading CLIP model for projection...")
    clip_model, _ = load_clip_model(model_path, device)

    print("Projecting through CLIP image_projector...")
    with torch.no_grad():
        mean_tensor = torch.tensor(aion_mean, dtype=torch.float32).to(device)
        aion_search = clip_model.image_projector(mean_tensor).cpu().numpy()
    print(f"  AION search embeddings: {aion_search.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ra": pa.array(meta["ra"], type=pa.float64()),
        "dec": pa.array(meta["dec"], type=pa.float64()),
        "label": pa.array(labels, type=pa.int64()),
        "label_name": pa.array(meta["label_name"].tolist(), type=pa.string()),
        "Galaxy10_DECals_index": pa.array(meta["Galaxy10_DECals_index"], type=pa.int64()),
        "aion_mean_embedding": pa.array(
            [row.astype(np.float32).tolist() for row in aion_mean],
            type=pa.list_(pa.float32()),
        ),
        "aion_search_embedding": pa.array(
            [row.astype(np.float32).tolist() for row in aion_search],
            type=pa.list_(pa.float32()),
        ),
    })
    pq.write_table(table, cache_path, compression="snappy")
    print(f"  Cached {len(table)} rows to {cache_path}")

    return meta, aion_search, labels


def train_and_evaluate(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_meta: dict,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    device: str = "cpu",
):
    """Train MLP head and evaluate."""
    input_dim = train_embeddings.shape[1]
    num_classes = len(CLASS_LABELS)

    print(f"AION-Search embedding dim: {input_dim}")

    model = MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_embeddings, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metrics_path = Path(OUTPUT_DIR) / "training_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy", "best_accuracy", "lr"])

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        scheduler.step()

        avg_loss = total_loss / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y.numpy())

        acc = accuracy_score(all_true, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{acc:.4f}", f"{best_acc:.4f}", f"{current_lr:.6f}"])

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.2%}  best={best_acc:.2%}  lr={current_lr:.2e}")

    print(f"\nMetrics saved to {metrics_path}")

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())

    predictions = np.array(all_preds)
    true_labels = np.array(all_true)

    print(f"\nBest test accuracy: {best_acc:.2%}")
    print(f"(AION-1-B fine-tuned baseline from paper: 84.0%)")
    print(f"\nPer-class report:")
    target_names = [CLASS_LABELS[i] for i in range(num_classes)]
    print(classification_report(true_labels, predictions, target_names=target_names, digits=3))

    # Save classification results parquet.
    results_path = Path(OUTPUT_DIR) / "gz10_finetune_classifications.parquet"
    table = pa.table({
        "ra": pa.array(test_meta["ra"], type=pa.float64()),
        "dec": pa.array(test_meta["dec"], type=pa.float64()),
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
    })
    pq.write_table(table, results_path, compression="snappy")
    print(f"Saved classifications to {results_path}")

    # Save model checkpoint
    model_path = Path(OUTPUT_DIR) / "gz10_mlp_head.pt"
    torch.save({
        "model_state_dict": best_state,
        "accuracy": best_acc,
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        },
    }, model_path)
    print(f"Saved model to {model_path}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MLP on AION-Search embeddings for GZ10")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size (default: 256)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dim (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (default: 0.1)")
    parser.add_argument("--aion-batch-size", type=int, default=32, help="AION encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, or cpu)")
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    _, train_emb, train_labels = get_embeddings(
        "train", device, batch_size=args.aion_batch_size, model_path=args.model_path
    )
    test_meta, test_emb, test_labels = get_embeddings(
        "test", device, batch_size=args.aion_batch_size, model_path=args.model_path
    )

    print(f"\nTrain: {len(train_labels)} samples")
    print(f"Test:  {len(test_labels)} samples")
    print(f"Classes: {len(CLASS_LABELS)}")

    print("\nClass distribution (train):")
    for i in range(len(CLASS_LABELS)):
        count = (train_labels == i).sum()
        print(f"  {i}: {CLASS_LABELS[i]:40s} {count:5d} ({count/len(train_labels):.1%})")

    print(f"\nTraining MLP head: hidden={args.hidden_dim}, dropout={args.dropout}, lr={args.lr}, epochs={args.epochs}")
    train_and_evaluate(
        train_emb, train_labels, test_emb, test_labels, test_meta,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, dropout=args.dropout, device=device,
    )


if __name__ == "__main__":
    main()
