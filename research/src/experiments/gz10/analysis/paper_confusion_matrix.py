"""
Side-by-side confusion matrix figure for the paper.

CLI:
    uv run python src/experiments/gz10/analysis/paper_confusion_matrix.py \
        --left data/gz10/gz10_aion_classifications.parquet \
        --right data/gz10/llm_results/.../llm_judge_classifications.parquet

    uv run python src/experiments/gz10/analysis/paper_confusion_matrix.py \
        --left data/gz10/gz10_aion_classifications.parquet \
        --right data/gz10/llm_results/.../llm_judge_classifications.parquet \
        --output figures/gz10_confusion_matrices.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from src.experiments.gz10.constants import CLASS_LABELS


LABEL_NAMES_SHORT = [
    CLASS_LABELS[i].replace(" Galaxies", "").replace(" Galaxy", "")
    for i in range(len(CLASS_LABELS))
]


def compute_cm(df):
    """Compute row-normalized confusion matrix and accuracy from a dataframe."""
    true_labels = df["label"].values.astype(int)
    predictions = df["predicted_label"].values.astype(int)
    n_classes = len(CLASS_LABELS)

    cm = confusion_matrix(true_labels, predictions, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype(float) / row_sums
    accuracy = accuracy_score(true_labels, predictions)

    return cm_normalized, accuracy


def draw_cm(ax, cm_normalized, title, show_ylabel):
    """Draw a single confusion matrix on an axes."""
    n_classes = len(CLASS_LABELS)

    cm_percent = (cm_normalized * 100).astype(int)
    annot_labels = np.array([[f"{val}%" for val in row] for row in cm_percent])

    sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=LABEL_NAMES_SHORT,
        yticklabels=LABEL_NAMES_SHORT,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"fontsize": 14},
        cbar=False,
        ax=ax,
    )

    # Bold diagonal annotations
    for text_obj in ax.texts:
        row = int(text_obj.get_position()[1])
        col = int(text_obj.get_position()[0])
        if row == col:
            text_obj.set_fontweight("bold")

    for i in range(n_classes):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=1.5)
        )

    ax.set_title(title, fontsize=19, pad=12)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

    if show_ylabel:
        ax.set_ylabel("True", fontsize=19)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side confusion matrices for paper"
    )
    parser.add_argument(
        "--left",
        type=str,
        required=True,
        help="Left panel classification parquet",
    )
    parser.add_argument(
        "--right",
        type=str,
        required=True,
        help="Right panel classification parquet",
    )
    parser.add_argument(
        "--left-title",
        type=str,
        default="AION-Search",
    )
    parser.add_argument(
        "--right-title",
        type=str,
        default="GPT-4.1-mini",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gz10/gz10_paper_confusion_matrices.png",
        help="Output file path (.png or .pdf)",
    )
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)

    if not left_path.exists():
        print(f"File not found: {left_path}")
        return
    if not right_path.exists():
        print(f"File not found: {right_path}")
        return

    df_left = pq.read_table(left_path).to_pandas()
    df_right = pq.read_table(right_path).to_pandas()

    cm_left, acc_left = compute_cm(df_left)
    cm_right, acc_right = compute_cm(df_right)

    print(f"Left  ({args.left_title}): {acc_left:.2%} accuracy, {len(df_left)} samples")
    print(f"Right ({args.right_title}): {acc_right:.2%} accuracy, {len(df_right)} samples")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8))

    left_title = f"(a) {args.left_title} ({acc_left:.1%})"
    right_title = f"(b) {args.right_title} ({acc_right:.1%})"

    draw_cm(ax_left, cm_left, left_title, show_ylabel=True)
    draw_cm(ax_right, cm_right, right_title, show_ylabel=False)

    plt.tight_layout(w_pad=0)
    plt.subplots_adjust(wspace=0.05, bottom=0.28)
    fig.canvas.draw()
    left_bbox = ax_left.get_position()
    right_bbox = ax_right.get_position()
    mid_x = (left_bbox.x0 + right_bbox.x1) / 2
    fig.text(mid_x, 0.005, "Predicted", ha="center", fontsize=19)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
