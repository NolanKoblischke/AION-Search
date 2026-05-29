"""
Lens relevance sensitivity analysis for AION and AION-Search.

Re-scores existing retrieval result CSVs under different relevance
definitions. The retrieved objects and their rankings are unchanged --
only the relevance column is recomputed.

Sensitivity axes:
  - Matching radius for relevance (e.g. 1" vs 30")
  - Additional catalogs for relevance (e.g. adding LensCat)

Input CSVs (produced by eval_retrieval.py):
  aion_lens:        query_index, fold_index, rank, global_index, ra, dec, similarity, relevance
  aion_search_lens: rank, fold_index, global_index, ra, dec, similarity, relevance

Usage examples:
  # Config A: HSC+ML at 30", 10-fold, AION-1
  uv run -m src.experiments.retrieval.lens_selection_sensitivity.eval_lens_sensitivity \
    --objects-csv data/retrieval_results/10fold/aion_lens/aion_lens_kfold_objects.csv \
    --match-arcsec 30

  # Config B: HSC+ML+LensCat at 1", full, AION-Search
  uv run -m src.experiments.retrieval.lens_selection_sensitivity.eval_lens_sensitivity \
    --objects-csv data/retrieval_results/full/aion_search_lens/default/aion_search_lens_kfold_objects.csv \
    --lenscat-path data/lens/lenscat.csv

  # Config C: HSC+ML+LensCat at 30", 10-fold, AION-Search
  uv run -m src.experiments.retrieval.lens_selection_sensitivity.eval_lens_sensitivity \
    --objects-csv data/retrieval_results/10fold/aion_search_lens/default/aion_search_lens_kfold_objects.csv \
    --lenscat-path data/lens/lenscat.csv --match-arcsec 30
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from src.experiments.retrieval.eval_retrieval import ndcg_at_k
from src.evals.eval_utils import setup_logging


# ---------------------------------------------------------------------------
# Relevance recomputation
# ---------------------------------------------------------------------------

def build_lens_catalog(
    hsc_path: str,
    masterlens_path: str,
    lenscat_path: str | None,
) -> SkyCoord:
    """Load and combine lens catalogs into a single SkyCoord."""
    hsc = pd.read_csv(hsc_path)
    masterlens = pd.read_csv(masterlens_path)
    catalog_dfs = [hsc[['ra', 'dec']], masterlens[['ra', 'dec']]]

    if lenscat_path is not None:
        lenscat = pd.read_csv(lenscat_path)
        lenscat_coords = lenscat[['RA [deg]', 'DEC [deg]']].rename(
            columns={'RA [deg]': 'ra', 'DEC [deg]': 'dec'}
        )
        catalog_dfs.append(lenscat_coords)

    combined = pd.concat(catalog_dfs, ignore_index=True)
    return SkyCoord(
        ra=combined['ra'].values * u.degree,
        dec=combined['dec'].values * u.degree,
    )


def recompute_relevance(
    ra: np.ndarray,
    dec: np.ndarray,
    lens_coords: SkyCoord,
    match_radius_arcsec: float,
) -> np.ndarray:
    """Return binary relevance for each object based on proximity to catalog."""
    obj_coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    _, sep, _ = obj_coords.match_to_catalog_sky(lens_coords)
    return (sep.arcsec <= match_radius_arcsec).astype(np.float32)


# ---------------------------------------------------------------------------
# NDCG calculation from re-scored CSV
# ---------------------------------------------------------------------------

def compute_ndcg_image_query(df: pd.DataFrame, k: int,
                             total_relevant: int) -> dict:
    """Compute per-fold mean NDCG@k for image-query (aion_lens) results.

    Expects columns: query_index, fold_index, rank, relevance.

    total_relevant is the number of relevant objects in the full parent
    sample (not just the top-N retrieved).  Since total_relevant >> k,
    the ideal DCG always assumes all top-k slots are filled with relevant
    items, matching the normalization used by eval_retrieval.py.
    """
    # Build an all_relevances array with enough 1s for correct IDCG.
    # ndcg_at_k sorts this and takes top-k, so we just need >= k ones.
    ideal_relevances = np.ones(max(total_relevant, k), dtype=np.float32)

    fold_ndcgs = {}
    for fold, fold_df in df.groupby('fold_index'):
        query_scores = []
        for _, q_df in fold_df.groupby('query_index'):
            q_df = q_df.sort_values('rank')
            ranked_rel = q_df['relevance'].values
            q_ndcg = ndcg_at_k(ranked_rel, k, ideal_relevances)
            query_scores.append(q_ndcg)
        fold_ndcgs[int(fold)] = float(np.mean(query_scores)) if query_scores else 0.0
    return fold_ndcgs


def compute_ndcg_text_query(df: pd.DataFrame, k: int) -> dict:
    """Compute per-fold NDCG@k for text-query (aion_search_lens) results.

    Expects columns: fold_index, rank, relevance.
    """
    fold_ndcgs = {}
    for fold, fold_df in df.groupby('fold_index'):
        fold_df = fold_df.sort_values('rank')
        ranked_rel = fold_df['relevance'].values
        fold_ndcgs[int(fold)] = float(ndcg_at_k(ranked_rel, k, ranked_rel))
    return fold_ndcgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score lens retrieval CSVs under different relevance definitions"
    )
    parser.add_argument("--objects-csv", type=str, required=True,
                        help="Path to an existing *_objects.csv from eval_retrieval.py")

    # Catalog configuration
    parser.add_argument("--hsc-path", type=str, default="data/lens/hsc_lenses.csv")
    parser.add_argument("--masterlens-path", type=str, default="data/lens/masterlens.csv")
    parser.add_argument("--lenscat-path", type=str, default=None,
                        help="Path to LensCat CSV. Adds LensCat to relevance catalog.")
    parser.add_argument("--match-arcsec", type=float, default=1.0,
                        help="Matching radius for relevance (default: 1.0)")

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated next to input CSV)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Detect task type from CSV columns
    objects_path = Path(args.objects_csv)
    df = pd.read_csv(objects_path)
    is_image_query = 'query_index' in df.columns
    task_label = "aion_lens" if is_image_query else "aion_search_lens"
    logger.info(f"Detected task: {task_label} ({len(df):,} rows)")

    # Build catalog and recompute relevance
    catalogs_label = "HSC + MasterLens"
    if args.lenscat_path:
        catalogs_label += " + LensCat"
    logger.info(f"Relevance config: {catalogs_label}, {args.match_arcsec}\"")

    lens_coords = build_lens_catalog(
        args.hsc_path, args.masterlens_path, args.lenscat_path,
    )
    logger.info(f"Combined lens catalog: {len(lens_coords)} entries")

    old_relevant = int(df['relevance'].sum())
    df['relevance'] = recompute_relevance(
        df['ra'].values, df['dec'].values, lens_coords, args.match_arcsec,
    )
    new_relevant = int(df['relevance'].sum())
    logger.info(f"Relevance: {old_relevant} -> {new_relevant} relevant rows "
                f"(out of {len(df):,})")

    # Compute NDCG
    if is_image_query:
        fold_ndcgs = compute_ndcg_image_query(df, args.k, new_relevant)
    else:
        fold_ndcgs = compute_ndcg_text_query(df, args.k)

    ndcg_values = list(fold_ndcgs.values())
    mean_ndcg = float(np.mean(ndcg_values))
    std_ndcg = float(np.std(ndcg_values))

    logger.info(f"NDCG@{args.k}: {mean_ndcg:.4f} +/- {std_ndcg:.4f}")
    for fold, val in sorted(fold_ndcgs.items()):
        logger.info(f"  Fold {fold}: {val:.4f}")

    # Build output directory
    if args.output_dir is None:
        radius_tag = f"{args.match_arcsec:.0f}arcsec"
        catalog_tag = "hsc_ml_lenscat" if args.lenscat_path else "hsc_ml"
        output_dir = objects_path.parent / "sensitivity" / f"{catalog_tag}_{radius_tag}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results = {
        'summary': {
            f'mean_ndcg@{args.k}': mean_ndcg,
            f'std_ndcg@{args.k}': std_ndcg,
            'n_folds': len(fold_ndcgs),
            'relevance_catalogs': catalogs_label,
            'relevance_radius_arcsec': args.match_arcsec,
            'total_relevant_in_results': new_relevant,
            'source_csv': str(objects_path),
        },
        'folds': [
            {'fold': fold, f'ndcg@{args.k}': val}
            for fold, val in sorted(fold_ndcgs.items())
        ],
    }
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # Save re-scored CSV
    rescored_path = output_dir / objects_path.name
    df.to_csv(rescored_path, index=False)
    logger.info(f"Re-scored CSV saved to: {rescored_path}")


if __name__ == "__main__":
    main()
