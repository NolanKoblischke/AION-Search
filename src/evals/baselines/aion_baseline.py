#!/usr/bin/env python3
"""
Generalized AION baseline for semantic search evaluation.
Uses mean embeddings to find similar objects based on high-scoring examples as queries.
Works with any evaluation type (lens, merger, spiral).
"""

import numpy as np
from numpy.random import default_rng
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import h5py
import matplotlib.pyplot as plt
from src.plotting_scripts.default_plot import process_galaxy_image

from src.evals.eval_utils import (
    setup_logging,
    load_aion_embeddings,
    compute_ndcg_at_k,
    append_eval_summary,
    EVAL_CONFIGS
)

def select_lens_queries(object_ids: list, eval_grades, eval_config) -> list:
    """Selects grade 'A' objects as queries for lens evaluation."""
    return [object_ids[i] for i, grade in enumerate(eval_grades) if grade == 'A']

def select_merger_queries(object_ids: list, eval_grades, eval_config) -> list:
    """Selects top 50 objects as queries for merger evaluation."""
    scores = [(i, eval_config.grade_converter(eval_grades[i])) for i in range(len(object_ids))]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    n_queries = min(50, len(scores))
    return [object_ids[i] for i, score in scores[:n_queries] if score > 0]

def select_spiral_queries(object_ids: list, eval_grades, eval_config) -> list:
    """Selects top 50 objects as queries for spiral evaluation."""
    scores = [(i, eval_config.grade_converter(eval_grades[i])) for i in range(len(object_ids))]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    n_queries = min(50, len(scores))
    return [object_ids[i] for i, score in scores[:n_queries] if score > 0]


QUERY_SELECTION_FNS = {
    'lens_legacy': select_lens_queries,
    'lens_hsc': select_lens_queries,
    'mergers': select_merger_queries,
    'spirals': select_spiral_queries,
}


def load_images_for_eval(eval_name, object_ids, eval_config):
    """Load images for a specific evaluation type."""
    if eval_name in ['mergers', 'spirals']:
        return load_merger_spiral_images(eval_name, object_ids, eval_config)
    elif eval_name in ['lens_legacy', 'lens_hsc']:
        return load_lens_images(eval_name, object_ids, eval_config)
    else:
        raise ValueError(f"Image loading not implemented for {eval_name}")


def load_merger_spiral_images(eval_name, object_ids, eval_config):
    """Load images for merger or spiral galaxies from GZ5 dataset."""
    # Load the CSV to get gz5 indices
    if eval_name == 'mergers':
        csv_path = Path("data/evals/mergers_spirals/mergers_selected.csv")
    else:  # spirals
        csv_path = Path("data/evals/mergers_spirals/spirals_selected.csv")
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Create mapping from object_id to gz5_index
    id_to_gz5_idx = dict(zip(df['object_id'].astype(str), df['gz5_legacysurvey_index']))
    
    # Load images from GZ5
    gz5_file = Path("data/gz5_legacysurvey_images.hdf5")
    images = {}
    
    with h5py.File(gz5_file, 'r') as f:
        data_table = f["__astropy_table__"]
        
        for oid in object_ids:
            if str(oid) in id_to_gz5_idx:
                gz5_idx = int(id_to_gz5_idx[str(oid)])
                images[str(oid)] = data_table[gz5_idx]['image_array']
    
    return images


def load_lens_images(eval_name, object_ids, eval_config):
    """Load images for lens evaluations from FITS file."""
    # Load the CSV to get fits indices
    csv_path = Path("data/evals/lens/lens_eval_objects.csv")
    
    import pandas as pd
    import fitsio
    
    df = pd.read_csv(csv_path)
    
    # Create mapping from object_id to fits_index
    id_to_fits_idx = dict(zip(df['object_id'].astype(str), df['fits_index']))
    
    # Default FITS file path
    fits_path = Path("data/lens_image_catalog_part_000.fits")
    
    images = {}
    
    try:
        with fitsio.FITS(str(fits_path)) as f:
            # Check available columns
            col_names = f[1].get_colnames()
            
            # Determine which image column to use based on eval_name
            if eval_name == 'lens_legacy' and 'legacysurvey_image' in col_names:
                image_col = 'legacysurvey_image'
            elif eval_name == 'lens_hsc' and 'hsc_image' in col_names:
                image_col = 'hsc_image'
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Image column not found for {eval_name}")
                return images
            
            # Get FITS indices for requested object IDs
            fits_indices_to_read = []
            oid_to_fits_idx_map = {}
            
            for oid in object_ids:
                if str(oid) in id_to_fits_idx:
                    fits_idx = int(id_to_fits_idx[str(oid)])
                    fits_indices_to_read.append(fits_idx)
                    oid_to_fits_idx_map[fits_idx] = str(oid)
            
            if fits_indices_to_read:
                # Read all images at once
                all_images = f[1].read_column(image_col, rows=fits_indices_to_read)
                
                # Map images to object IDs
                for i, fits_idx in enumerate(fits_indices_to_read):
                    oid = oid_to_fits_idx_map[fits_idx]
                    img = all_images[i]
                    
                    # Handle Legacy images with 5 channels
                    if eval_name == 'lens_legacy' and img.shape[0] == 5:
                        img = img[:4]  # Remove NaN padding from 5th channel
                    
                    images[oid] = img
                    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load lens images: {e}")
    
    return images


def plot_retrieval_results(eval_name, query_ids, object_ids, embeddings, eval_grades, eval_config, 
                          grade_converter, final_metrics, output_file="retrieval_results.png", n_queries=5, k=10):
    """Plot top k results for n queries with relevance scores."""
    logger = logging.getLogger(__name__)
    
    # Select up to n_queries
    plot_queries = query_ids[:n_queries]
    logger.info(f"Creating visualization for {len(plot_queries)} queries")
    
    # Calculate dataset-wide statistics needed for standardization
    all_relevance_scores = [grade_converter(grade) for grade in eval_grades]
    dataset_avg_relevance = float(np.mean(all_relevance_scores))
    
    # Load images for all objects we might need
    all_needed_ids = set(plot_queries)
    
    # Get top k for each query to know which images to load and compute metrics
    query_results = {}
    for query_id in plot_queries:
        query_idx = object_ids.index(query_id)
        query_embedding = embeddings[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        similarities[query_idx] = -1.0
        top_indices = np.argsort(similarities)[::-1][:k+1]
        top_indices = [idx for idx in top_indices if object_ids[idx] != query_id][:k]
        
        # Compute metrics for this query
        retrieved_relevance_scores_k = np.array([grade_converter(eval_grades[idx]) for idx in top_indices])
        
        # nDCG@k
        ndcg_at_k = compute_ndcg_at_k(retrieved_relevance_scores_k, k, all_relevance_scores)
        
        query_results[query_id] = {
            'top_indices': top_indices,
            'similarities': similarities,
            'ndcg': ndcg_at_k
        }
        
        for idx in top_indices:
            all_needed_ids.add(object_ids[idx])
    
    logger.info(f"Loading images for {len(all_needed_ids)} objects...")
    images = load_images_for_eval(eval_name, list(all_needed_ids), eval_config)
    
    # Create figure with extra space for metrics
    fig, axes = plt.subplots(len(plot_queries), k+1, figsize=(24, 4*len(plot_queries)))
    if len(plot_queries) == 1:
        axes = axes.reshape(1, -1)
    
    # Add extra space on the left for metrics
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.05)
    
    # Main title
    fig.suptitle(f'{eval_name.upper()} Retrieval Results - Top {k} Retrieved Objects', fontsize=16)
    
    # Process each query
    for q_idx, query_id in enumerate(plot_queries):
        logger.info(f"Processing query {q_idx+1}/{len(plot_queries)}: {query_id}")
        
        # Find query index
        query_idx = object_ids.index(query_id)
        
        # Get precomputed results
        top_indices = query_results[query_id]['top_indices']
        similarities = query_results[query_id]['similarities']
        
        # Plot query image
        ax = axes[q_idx, 0]
        if str(query_id) in images:
            img = process_galaxy_image(images[str(query_id)])
            ax.imshow(img)
            query_relevance = grade_converter(eval_grades[query_idx])
            # For lens evaluations, show both grade and relevance
            if eval_name in ['lens_legacy', 'lens_hsc']:
                query_grade = eval_grades[query_idx]
                ax.set_title(f'Query: {query_id}\nGrade: {query_grade}, Rel: {query_relevance:.3f}', fontsize=8)
            else:
                ax.set_title(f'Query: {query_id}\nRelevance: {query_relevance:.3f}', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            ax.set_title(f'Query: {query_id}', fontsize=8)
        ax.axis('off')
        
        # Get metrics for this query
        query_ndcg = query_results[query_id]['ndcg']
        # Add query label and metrics to the left
        metrics_text = (f'Query {q_idx+1}\n'
                       f'nDCG: {query_ndcg:.3f}')
        ax.text(-0.18, 0.5, metrics_text, transform=ax.transAxes, 
                va='center', ha='right', fontsize=8, weight='bold')
        
        # Plot retrieved objects
        for i, idx in enumerate(top_indices):
            ax = axes[q_idx, i+1]
            retrieved_id = object_ids[idx]
            
            if str(retrieved_id) in images:
                img = process_galaxy_image(images[str(retrieved_id)])
                ax.imshow(img)
                
                # Get relevance score
                relevance = grade_converter(eval_grades[idx])
                similarity = similarities[idx]
                
                # For lens evaluations, show both grade and relevance
                if eval_name in ['lens_legacy', 'lens_hsc']:
                    grade = eval_grades[idx]
                    ax.set_title(f'{retrieved_id}\nSim: {similarity:.3f}\nGrade: {grade}, Rel: {relevance:.3f}', fontsize=6)
                else:
                    ax.set_title(f'{retrieved_id}\nSim: {similarity:.3f}\nRel: {relevance:.3f}', fontsize=6)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                ax.set_title(f'{retrieved_id}', fontsize=6)
            
            ax.axis('off')
    
    # Add column headers
    for i in range(k+1):
        if i == 0:
            label = 'Query'
        else:
            label = f'Rank {i}'
        # Adjust x position to account for left margin
        x_pos = 0.12 + (i + 0.5) * (0.86 / (k + 1))
        fig.text(x_pos, 0.97, label, ha='center', va='top', fontsize=10)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    plt.close()


def evaluate_aion_baseline(
    eval_name: str,
    eval_config,
    k_values: list = [10, 1000],
    plot_results: bool = False,
    output_dir: Path = None
):
    """
    Evaluate AION baseline performance.
    
    Args:
        eval_name: Name of evaluation
        eval_config: Evaluation configuration
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Load grade mapping from the embeddings file
    object_ids, embeddings, eval_grades = load_aion_embeddings(
        eval_config.embeddings_path,
        use_mean=True,
        aion_model='aion-base'  # This might need to be configurable
    )

    # Select query objects based on evaluation type
    if eval_name not in QUERY_SELECTION_FNS:
        logger.error(f"No query selection function for eval_name: {eval_name}")
        return {}

    query_selection_fn = QUERY_SELECTION_FNS[eval_name]
    query_ids = query_selection_fn(object_ids=object_ids, eval_grades=eval_grades, eval_config=eval_config)
    logger.info(f"Using {len(query_ids)} objects as queries for {eval_name}")
    
    if len(query_ids) == 0:
        logger.error("No suitable query objects found!")
        return {}
    
    # Create mapping from object_id to index
    id_to_idx = {oid: i for i, oid in enumerate(object_ids)}
    
    # Get indices of query objects
    query_indices = [id_to_idx[oid] for oid in query_ids if oid in id_to_idx]
    logger.info(f"Found {len(query_indices)} query objects with embeddings")
    
    # Initialize result structures
    results = {k: {'ndcg_scores': []} for k in k_values}
    
    # For lens evaluations, also track lenses@k for each query
    if eval_name in ['lens_legacy', 'lens_hsc']:
        for k in k_values:
            results[k]['lenses_at_k'] = []

    # Calculate overall average relevance score for the dataset (used as random baseline)
    all_relevance_scores = [eval_config.grade_converter(grade) for grade in eval_grades]
    dataset_avg_relevance = float(np.mean(all_relevance_scores))

    # Process each query
    logger.info(f"Computing metrics for k values: {k_values}")
    
    for i, query_idx in enumerate(query_indices):
        query_embedding = embeddings[query_idx].reshape(1, -1)
        
        # Compute cosine similarities to all embeddings
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Don't match to yourself
        similarities[query_idx] = -1.0
        
        # Get top indices
        max_k = max(k_values)
        top_indices_all = np.argsort(similarities)[::-1][:min(max_k+1, len(similarities))]
        
        # Remove self from results
        top_indices_all = [idx for idx in top_indices_all if idx != query_idx][:max_k]
        
        # Compute metrics for each k value
        query_metrics = {}
        for k in k_values:
            # Get top k indices
            top_indices_k = top_indices_all[:k]
            
            # Compute relevance scores using grade converter
            retrieved_relevance_scores_k = []
            for idx in top_indices_k:
                weight = eval_config.grade_converter(eval_grades[idx])
                retrieved_relevance_scores_k.append(weight)
            
            retrieved_relevance_scores_k = np.array(retrieved_relevance_scores_k)
            
            ndcg_score = compute_ndcg_at_k(retrieved_relevance_scores_k, k, all_relevance_scores)
            results[k]['ndcg_scores'].append(ndcg_score)
            query_metrics[f'ndcg@{k}'] = ndcg_score
            
            # For lens evaluations, count number of true lenses in top-k
            if eval_name in ['lens_legacy', 'lens_hsc']:
                num_lenses = sum(1 for idx in top_indices_k if eval_grades[idx] in ['A', 'B', 'C'])
                results[k]['lenses_at_k'].append(num_lenses)
                query_metrics[f'lenses@{k}'] = num_lenses
        
        # Log metrics for EVERY query (not just every 10)
        logger.info(f"  Query {i+1}/{len(query_indices)} (ID: {object_ids[query_idx]}):")
        for k in k_values:
            if eval_name in ['lens_legacy', 'lens_hsc']:
                logger.info(f"    NDCG@{k} = {query_metrics[f'ndcg@{k}']:.4f}, Lenses@{k} = {query_metrics[f'lenses@{k}']}")
            else:
                logger.info(f"    NDCG@{k} = {query_metrics[f'ndcg@{k}']:.4f}")

    logger.info(f"Dataset average relevance score: {dataset_avg_relevance:.4f}")
    
    rng = default_rng(42)  # reproducible random baseline
    final_metrics = {}
    
    for k in k_values:
        ndcg_array = np.array(results[k]['ndcg_scores'])
        # Actual metrics from queries
        final_metrics[f'ndcg@{k}'] = float(ndcg_array.mean())
        
        # For lens evaluations, compute average lenses@k
        if eval_name in ['lens_legacy', 'lens_hsc'] and 'lenses_at_k' in results[k]:
            lenses_array = np.array(results[k]['lenses_at_k'])
            final_metrics[f'lenses@{k}'] = float(lenses_array.mean())
        
        # -------- Random baseline (median of 10 random orderings) --------
        rand_ndcgs = []
        for _ in range(10):
            rand_idx = rng.permutation(len(object_ids))[:k]
            rand_rel = np.array([eval_config.grade_converter(eval_grades[idx]) for idx in rand_idx])
            rand_ndcg = compute_ndcg_at_k(rand_rel, k, all_relevance_scores)
            rand_ndcgs.append(rand_ndcg)
        
        final_metrics[f'random_ndcg@{k}'] = float(np.median(rand_ndcgs))
        
        # -------- Ideal baseline (perfect ranking) --------
        # Sort all objects by relevance score
        sorted_relevance_scores = sorted(all_relevance_scores, reverse=True)
        
        # Compute ideal nDCG@k based on actual dataset
        # This is the NDCG you would get if you perfectly ranked all items
        ideal_relevance_k = np.array(sorted_relevance_scores[:k])
        ideal_ndcg = compute_ndcg_at_k(ideal_relevance_k, k, all_relevance_scores)
        final_metrics[f'ideal_ndcg@{k}'] = ideal_ndcg
        
        
        # No standardized metrics - removed sdcg and std_avg_rel calculations
    
    # dataset stats
    final_metrics['num_queries']   = len(query_indices)
    final_metrics['total_objects'] = len(object_ids)

    # Create visualization if requested
    if plot_results and output_dir is not None:
        try:
            plot_file = output_dir / f'{eval_name}_retrieval_visualization.png'
            logger.info(f"\nCreating retrieval visualization...")
            
            plot_retrieval_results(
                eval_name=eval_name,
                query_ids=query_ids,
                object_ids=object_ids,
                embeddings=embeddings,
                eval_grades=eval_grades,
                eval_config=eval_config,
                grade_converter=eval_config.grade_converter,
                final_metrics=final_metrics,
                output_file=str(plot_file),
                n_queries=5,
                k=10
            )
            logger.info(f"Visualization saved to: {plot_file}")
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")

    return final_metrics


def main():
    """Main function to run AION baseline evaluation."""
    parser = argparse.ArgumentParser(description='AION baseline for semantic search evaluation')
    parser.add_argument('--eval-name', type=str, required=True,
                       choices=['lens_legacy', 'lens_hsc', 'spirals', 'mergers'],
                       help='Evaluation to run')
    parser.add_argument('--k-values', type=int, nargs='+',
                       default=[10, 1000],
                       help='k values for metrics')
    parser.add_argument('--output-dir', type=str,
                       default='data/eval_results',
                       help='Output directory for results')
    parser.add_argument('--aion-model', type=str, default='aion-base',
                       choices=['aion-base', 'aion-large', 'aion-xlarge'],
                       help='AION model size')
    parser.add_argument('--plot', action='store_true',
                       help='Create visualization of retrieval results')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path("data/logs") / f'{args.eval_name}_aion_baseline_{timestamp}.log'
    logger = setup_logging("INFO", str(log_file))
    
    logger.info("="*60)
    logger.info("AION Baseline Evaluation")
    logger.info("="*60)
    logger.info(f"Evaluation: {args.eval_name}")
    logger.info(f"k values: {args.k_values}")
    logger.info(f"AION model: {args.aion_model}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get evaluation configuration
    if args.eval_name not in EVAL_CONFIGS:
        raise ValueError(f"Unknown evaluation: {args.eval_name}")
    
    eval_config = EVAL_CONFIGS[args.eval_name]
    
    # Always use default embeddings path from config
    embeddings_path = eval_config.embeddings_path
    logger.info(f"Embeddings: {embeddings_path}")
    
    # Run baseline evaluation
    metrics = evaluate_aion_baseline(
        eval_name=args.eval_name,
        eval_config=eval_config,
        k_values=args.k_values,
        plot_results=args.plot,
        output_dir=output_dir
    )
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'{args.eval_name}_aion_baseline_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write(f"AION Baseline Results for {args.eval_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Embeddings: {embeddings_path}\n")
        f.write(f"AION model: {args.aion_model}\n")
        
        for k in args.k_values:
            f.write(f"\n=== Metrics @{k} ===\n")
            f.write(f"\nRaw Metrics:\n")
            f.write(f"  nDCG@{k}:                     {metrics[f'ndcg@{k}']:.4f}\n")
            
            # Add Lenses@k for lens evaluations (now averaged across queries)
            if args.eval_name in ['lens_legacy', 'lens_hsc'] and f'lenses@{k}' in metrics:
                f.write(f"  Lenses@{k} (avg):             {metrics[f'lenses@{k}']:.4f}\n")
            
            f.write(f"\nBaselines:\n")
            f.write(f"  Random nDCG@{k}:              {metrics[f'random_ndcg@{k}']:.4f}\n")
            f.write(f"  Ideal nDCG@{k}:               {metrics[f'ideal_ndcg@{k}']:.4f}\n")
            
        
        f.write(f"\nDataset statistics:\n")
        f.write(f"Number of queries: {metrics['num_queries']}\n")
        f.write(f"Total objects: {metrics['total_objects']}\n")
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Append to summary
    summary_file = append_eval_summary(
        output_dir=output_dir,
        eval_name=args.eval_name,
        model_name='aion_baseline',
        metrics=metrics,
        additional_info={
            'embeddings_file': str(embeddings_path),
            'aion_model': args.aion_model,
            'note': 'AION baseline using high-scoring objects as queries'
        }
    )
    logger.info(f"Summary appended to {summary_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary:")
    logger.info("="*60)
    
    
    logger.info(f"\n{args.eval_name} Results:")
    for k in args.k_values:
        logger.info(f"\n  Metrics @{k}:")
        logger.info(f"    Raw:")
        logger.info(f"      nDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        # Add Lenses@k for lens evaluations (now averaged across queries)
        if args.eval_name in ['lens_legacy', 'lens_hsc'] and f'lenses@{k}' in metrics:
            logger.info(f"      Lenses@{k} (avg): {metrics[f'lenses@{k}']:.4f}")
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()