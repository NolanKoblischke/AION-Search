#!/usr/bin/env python3
"""
Analyze reranked results to compute NDCG@10 with all possible shuffles of tied VLM scores.
This version correctly handles ties by allowing any galaxy with the same score to potentially
appear in the top-k, while preserving score ordering.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

# Try to import tqdm, fall back to simple progress indicator if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        """Simple fallback progress indicator."""
        if desc:
            print(f"{desc}...")
        return iterable


def dcg(r):
    """Compute Discounted Cumulative Gain (DCG)."""
    return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))


def ndcg_score(relevances, top_k):
    """Compute NDCG@k using fraction method."""
    actual_dcg = dcg(relevances[:top_k])
    ideal_dcg = dcg(np.sort(relevances)[::-1][:top_k])
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def analyze_reranked_ties(csv_path: str, k: int = 10, n_shuffles: int = 1000):
    """
    Analyze reranked CSV and compute NDCG@k with random shuffles of tied scores.
    
    This correctly handles tied scores by:
    1. Grouping all galaxies by their VLM score
    2. For each shuffle, randomly ordering galaxies within each score group
    3. Reconstructing the full ranking preserving score order but with randomized ties
    
    Args:
        csv_path: Path to reranked CSV file
        k: Top-k for NDCG calculation
        n_shuffles: Number of random shuffles to perform
    
    Returns:
        Dictionary with results
    """
    # Load data
    logging.info(f"Loading reranked data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Detect evaluation type from columns
    if 'spiral_fraction' in df.columns:
        eval_type = 'spiral_fraction'
        relevance_col = 'spiral_fraction'
    elif 'merger_fraction' in df.columns:
        eval_type = 'merger_fraction'
        relevance_col = 'merger_fraction'
    elif 'is_lens' in df.columns:
        eval_type = 'is_lens'
        relevance_col = 'is_lens'
    else:
        raise ValueError("Could not determine evaluation type from columns")
    
    logging.info(f"Detected evaluation type: {eval_type}")
    
    # Get VLM scores and relevance scores for entire dataset
    vlm_scores = df['vlm_score'].values
    relevance_scores = df[relevance_col].values
    
    # Group galaxies by VLM score
    unique_scores = np.unique(vlm_scores)
    unique_scores = unique_scores[::-1]  # Sort descending
    
    score_groups = {}
    for score in unique_scores:
        indices = np.where(vlm_scores == score)[0]
        if len(indices) > 1:  # Only care about tied scores
            score_groups[score] = indices
    
    # Find which scores could affect top k
    scores_affecting_top_k = []
    cumulative_count = 0
    for score in unique_scores:
        indices = np.where(vlm_scores == score)[0]
        if cumulative_count < k:  # This score group overlaps with top k
            scores_affecting_top_k.append(score)
        cumulative_count += len(indices)
        if cumulative_count >= k and score not in scores_affecting_top_k:
            # Check if any galaxies with this score are in top k
            if any(idx < k for idx in indices):
                scores_affecting_top_k.append(score)
    
    # Log analysis
    logging.info(f"\nAnalyzing scores that could affect top {k}:")
    for score in scores_affecting_top_k:
        indices = np.where(vlm_scores == score)[0]
        in_top_k = sum(1 for idx in indices if idx < k)
        logging.info(f"  Score {score}: {len(indices)} galaxies total, {in_top_k} currently in top {k}")
    
    # Compute NDCG with random shuffles
    ndcg_scores = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Original NDCG (no shuffling)
    original_ndcg = ndcg_score(relevance_scores, k)
    
    for shuffle_idx in tqdm(range(n_shuffles), desc="Computing NDCG for shuffled orderings"):
        # Build new ordering by score group
        new_order = []
        
        for score in unique_scores:
            indices = np.where(vlm_scores == score)[0]
            if score in score_groups:  # This score has ties
                # Shuffle the indices for this score group
                shuffled_indices = indices.copy()
                rng.shuffle(shuffled_indices)
                new_order.extend(shuffled_indices)
            else:
                # Single galaxy with this score, keep as is
                new_order.extend(indices)
        
        # Get relevance scores in new order
        new_order = np.array(new_order)
        shuffled_relevances = relevance_scores[new_order]
        
        # Compute NDCG@k
        ndcg = ndcg_score(shuffled_relevances, k)
        ndcg_scores.append(ndcg)
    
    # Compute statistics
    mean_ndcg = np.mean(ndcg_scores)
    std_ndcg = np.std(ndcg_scores)
    min_ndcg = np.min(ndcg_scores)
    max_ndcg = np.max(ndcg_scores)
    
    # Count unique scores in top k
    top_k_scores = vlm_scores[:k]
    top_k_unique, top_k_counts = np.unique(top_k_scores, return_counts=True)
    
    results = {
        'csv_path': csv_path,
        'eval_type': eval_type,
        'k': k,
        'n_shuffles': n_shuffles,
        'original_ndcg': original_ndcg,
        'mean_ndcg': mean_ndcg,
        'std_ndcg': std_ndcg,
        'min_ndcg': min_ndcg,
        'max_ndcg': max_ndcg,
        'n_scores_affecting_top_k': len(scores_affecting_top_k),
        'top_k_score_distribution': [(float(score), int(count)) for score, count in zip(top_k_unique, top_k_counts)],
        'scores_affecting_top_k': [(float(score), len(np.where(vlm_scores == score)[0])) 
                                   for score in scores_affecting_top_k]
    }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze reranked results for tied VLM scores impact on NDCG@k"
    )
    
    parser.add_argument("csv_path", type=str,
                       help="Path to reranked CSV file")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k for NDCG calculation (default: 10)")
    parser.add_argument("--n-shuffles", type=int, default=1000,
                       help="Number of random shuffles to perform (default: 1000)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    results = analyze_reranked_ties(
        csv_path=args.csv_path,
        k=args.k,
        n_shuffles=args.n_shuffles
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RERANKED TIES ANALYSIS")
    print(f"{'='*60}")
    print(f"File: {Path(args.csv_path).name}")
    print(f"Evaluation type: {results['eval_type']}")
    print(f"k: {results['k']}")
    print(f"\nCurrent top {results['k']} score distribution:")
    for score, count in results['top_k_score_distribution']:
        print(f"  Score {score}: {count} galaxies")
    print(f"\nScores that could affect top {results['k']}:")
    for score, total_count in results['scores_affecting_top_k']:
        # Find how many are currently in top k
        current_in_top_k = next(count for s, count in results['top_k_score_distribution'] if s == score)
        print(f"  Score {score}: {total_count} galaxies total ({current_in_top_k} currently in top {results['k']})")
    print(f"\nRandom shuffles performed: {results['n_shuffles']:,}")
    print(f"\nNDCG@{results['k']} Results:")
    print(f"  Original (no shuffle): {results['original_ndcg']:.4f}")
    print(f"  Mean ± Std:           {results['mean_ndcg']:.4f} ± {results['std_ndcg']:.4f}")
    print(f"  Range:                [{results['min_ndcg']:.4f}, {results['max_ndcg']:.4f}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()