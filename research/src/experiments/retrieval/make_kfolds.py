"""
Generate consistent k-fold assignments for lens and GZ5 datasets.
"""

import argparse
from pathlib import Path
import logging

import h5py
import numpy as np
import pandas as pd

from src.evals.eval_utils import setup_logging


def assign_folds(n_items: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    fold_assignments = np.empty(n_items, dtype=int)
    fold_size = n_items // n_folds

    for fold in range(n_folds):
        start = fold * fold_size
        end = n_items if fold == n_folds - 1 else (fold + 1) * fold_size
        fold_assignments[perm[start:end]] = fold

    return fold_assignments


def build_gz5_kfolds(
    hdf5_path: str,
    n_folds: int,
    seed: int,
    output_path: Path,
    logger: logging.Logger
) -> None:
    logger.info(f"Loading GZ5 data from: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        data_table = f['__astropy_table__']
        n_rows = len(data_table)
        ra = np.array(data_table['ra'][:])
        dec = np.array(data_table['dec'][:])
        votes = np.array(data_table['smooth-or-featured_total-votes'][:])

    valid_mask = votes >= 3
    valid_indices = np.where(valid_mask)[0]
    logger.info(f"Total GZ5 rows: {n_rows:,}")
    logger.info(f"Rows after quality filter (votes >= 3): {len(valid_indices):,}")

    fold_assignments = assign_folds(len(valid_indices), n_folds, seed)

    df = pd.DataFrame({
        'row_index': valid_indices,
        'ra': ra[valid_indices],
        'dec': dec[valid_indices],
        'kfold': fold_assignments
    }).sort_values('row_index')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved GZ5 k-folds to: {output_path}")


def build_lens_kfolds(
    hdf5_path: str,
    n_folds: int,
    seed: int,
    output_path: Path,
    logger: logging.Logger
) -> None:
    logger.info(f"Loading lens data from: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        data_table = f['__astropy_table__']
        n_rows = len(data_table)
        ra = np.array(data_table['ra'][:])
        dec = np.array(data_table['dec'][:])

    indices = np.arange(n_rows)
    fold_assignments = assign_folds(len(indices), n_folds, seed)

    df = pd.DataFrame({
        'row_index': indices,
        'ra': ra,
        'dec': dec,
        'kfold': fold_assignments
    }).sort_values('row_index')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved lens k-folds to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate k-fold assignments for lens and GZ5 datasets")
    parser.add_argument("--gz5-path", type=str, default="data/gz5_base_embedded.hdf5",
                        help="Path to gz5_base_embedded.hdf5")
    parser.add_argument("--lens-path", type=str,
                        default="data/lens/lens_parent_sample_v1_embedded_oct24_base.hdf5",
                        help="Path to lens parent sample HDF5")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/retrieval_results/10fold/kfolds",
                        help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    build_gz5_kfolds(
        args.gz5_path,
        args.n_folds,
        args.seed,
        output_dir / "gz5.csv",
        logger
    )
    build_lens_kfolds(
        args.lens_path,
        args.n_folds,
        args.seed,
        output_dir / "lens.csv",
        logger
    )


if __name__ == "__main__":
    main()
