"""
Optimized multi-model rerank experiments with single question.
Loads model and similarities once, supports multiple runs per configuration.
"""

import argparse
import numpy as np
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd

# Import necessary functions
from src.experiments.rerank.rerank_single_question import (
    setup_logging, get_full_dataset_similarities, rerank_galaxies_single_question,
    load_models_info, save_image_for_gpt, zoom_image,
    process_galaxy_image, GalaxyRanking, encode_image,
    get_gpt4_ranking, compute_ndcg_at_k
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_experiment_directory(base_dir: str = "data/experiments/rerank") -> Path:
    """Create a timestamped directory for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"multi_model_optimized_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def process_single_image_multi(args: Tuple[int, str, str, str, str, int]) -> Tuple[int, float, str, int, int]:
    """
    Process a single image multiple times and average the scores.
    
    Args:
        args: Tuple of (index, image_path, api_key, model_id, model_name, best_of_m)
        
    Returns:
        Tuple of (index, avg_ranking, combined_explanation, total_input_tokens, total_output_tokens)
    """
    index, image_path, api_key, model_id, model_name, best_of_m = args
    
    scores = []
    explanations = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i in range(best_of_m):
        try:
            ranking, explanation, input_tokens, output_tokens = get_gpt4_ranking(
                image_path, api_key, model_name
            )
            scores.append(ranking)
            explanations.append(f"[Eval {i+1}] {explanation}")
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        except Exception as e:
            logging.error(f"Error in evaluation {i+1} for image {index}: {e}")
            scores.append(5)  # Default score
            explanations.append(f"[Eval {i+1}] Error occurred during processing")
    
    # Calculate average score
    avg_score = np.mean(scores) if scores else 1.0
    combined_explanation = " | ".join(explanations)
    
    return index, avg_score, combined_explanation, total_input_tokens, total_output_tokens


def rerank_with_cached_data(
    object_ids: List[str],
    similarities: np.ndarray,
    eval_grades: List[str],
    images_to_rerank: List[np.ndarray],
    top_indices_to_rerank: np.ndarray,
    api_key: str,
    model_id: str,
    model_info: Dict,
    output_dir: Path,
    zoom: bool,
    k_values: List[int],
    run_number: int,
    best_of_m: int,
    logger: logging.Logger
) -> Dict:
    """
    Perform reranking using pre-loaded data.
    This avoids reloading the model and recalculating similarities.
    """
    n_total = len(object_ids)
    model_name = model_info['model_name']
    input_price_per_million = model_info['input_price']
    output_price_per_million = model_info['output_price']
    
    # Create directory for images
    images_dir = output_dir / f"galaxy_images_run{run_number}"
    images_dir.mkdir(exist_ok=True)
    
    # Process and save images for GPT evaluation
    logger.info(f"Run {run_number}: Processing and saving images...")
    image_paths = []
    for i, idx in enumerate(top_indices_to_rerank):
        image_path = save_image_for_gpt(images_to_rerank[i], str(images_dir), object_ids[idx], zoom=zoom)
        image_paths.append(image_path)
    
    # Get GPT rankings
    logger.info(f"Run {run_number}: Getting GPT rankings for top {len(top_indices_to_rerank)} objects...")
    if best_of_m > 1:
        logger.info(f"Using best-of-{best_of_m} evaluations per image")
    
    # Prepare arguments for parallel processing
    process_args = [
        (i, image_paths[i], api_key, model_id, model_name, best_of_m)
        for i in range(len(top_indices_to_rerank))
    ]
    
    # Process images in parallel
    gpt_scores = np.zeros(n_total, dtype=np.float32)  # Changed to float32 for average scores
    gpt_explanations = [""] * n_total
    # Track per-image tokens and costs
    per_image_input_tokens = np.zeros(n_total, dtype=np.int32)
    per_image_output_tokens = np.zeros(n_total, dtype=np.int32)
    per_image_costs = np.zeros(n_total, dtype=np.float32)
    total_input_tokens = 0
    total_output_tokens = 0
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(process_single_image_multi, args): args[0] 
            for args in process_args
        }
        
        with tqdm(total=len(top_indices_to_rerank), desc=f"Run {run_number} - Ranking galaxies") as pbar:
            for future in as_completed(future_to_index):
                try:
                    index, avg_ranking, explanation, input_tokens, output_tokens = future.result()
                    obj_idx = top_indices_to_rerank[index]
                    
                    gpt_scores[obj_idx] = avg_ranking
                    gpt_explanations[obj_idx] = explanation
                    
                    # Store per-image token counts
                    per_image_input_tokens[obj_idx] = input_tokens
                    per_image_output_tokens[obj_idx] = output_tokens
                    
                    # Calculate per-image cost
                    image_cost = (input_tokens * input_price_per_million + 
                                 output_tokens * output_price_per_million) / 1_000_000
                    per_image_costs[obj_idx] = image_cost
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    pbar.update(1)
                    
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Error processing image {index}: {e}")
                    pbar.update(1)
    
    # Calculate costs
    total_cost = (total_input_tokens * input_price_per_million + total_output_tokens * output_price_per_million) / 1_000_000
    cost_per_image = total_cost / len(top_indices_to_rerank) if len(top_indices_to_rerank) > 0 else 0
    
    # Get original ranking (sorted by similarity descending)
    original_indices = np.argsort(similarities)[::-1]
    
    # Create new ranking
    reranked_indices = original_indices.copy()
    
    # Get scores and similarities for top objects
    top_scores = gpt_scores[top_indices_to_rerank]
    top_similarities = similarities[top_indices_to_rerank]
    
    # Sort by score (desc) then similarity (desc)
    sorted_top_indices = np.lexsort((top_similarities[::-1], top_scores))[::-1]
    
    # Replace the top indices
    reranked_indices[:len(top_indices_to_rerank)] = top_indices_to_rerank[sorted_top_indices]
    
    # Calculate ranks
    new_ranks = np.zeros(n_total, dtype=int)
    for rank, idx in enumerate(reranked_indices):
        new_ranks[idx] = rank + 1
    
    original_ranks = np.zeros(n_total, dtype=int)
    for rank, idx in enumerate(original_indices):
        original_ranks[idx] = rank + 1
    
    # Get relevance scores
    relevance_scores = np.array([1.0 if grade in ['A', 'B', 'C'] else 0.0 for grade in eval_grades])
    
    # Calculate metrics
    results = {
        'run_number': run_number,
        'total_cost': total_cost,
        'cost_per_image': cost_per_image,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens
    }
    
    # Calculate metrics for each k
    for k in k_values:
        if k <= n_total:
            # Original metrics
            original_relevance_k = relevance_scores[original_indices[:k]]
            original_lenses = int(np.sum(original_relevance_k))
            results[f'original_lenses@{k}'] = original_lenses
            
            # Reranked metrics
            reranked_relevance_k = relevance_scores[reranked_indices[:k]]
            reranked_lenses = int(np.sum(reranked_relevance_k))
            results[f'reranked_lenses@{k}'] = reranked_lenses
            results[f'improvement_lenses@{k}'] = reranked_lenses - original_lenses
    
    # Save run-specific data with token counts and costs
    np.savez_compressed(
        output_dir / f"reranked_data_run{run_number}.npz",
        reranked_indices=reranked_indices,
        gpt_scores=gpt_scores,
        new_ranks=new_ranks,
        per_image_input_tokens=per_image_input_tokens,
        per_image_output_tokens=per_image_output_tokens,
        per_image_costs=per_image_costs,
        total_cost=total_cost,
        cost_per_image=cost_per_image,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        model_id=model_id,
        model_name=model_name,
        best_of_m=best_of_m,
        top_k=len(top_indices_to_rerank)
    )
    
    return results


def run_all_experiments(
    checkpoint_path: str,
    api_key: str,
    models: List[str],
    top_k_values: List[int],
    n_runs: int = 1,
    best_of_m_values: List[int] = [1],
    zoom: bool = False,
    cache_hdf5: str = None,
    logger: logging.Logger = None
) -> Dict:
    """
    Run all rerank experiments with optimized data loading.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create experiment directory
    exp_dir = create_experiment_directory()
    logger.info(f"Created experiment directory: {exp_dir}")
    
    # Load model pricing info once
    models_info = load_models_info()
    
    # Load model and compute similarities ONCE
    logger.info("Loading model and computing similarities (this will be reused for all experiments)...")
    start_load = time.time()
    
    object_ids, similarities, eval_grades, hsc_images, model_config = get_full_dataset_similarities(
        checkpoint_path=checkpoint_path,
        eval_name='lens_hsc',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger
    )
    
    load_time = time.time() - start_load
    logger.info(f"Model and similarities loaded in {load_time:.2f} seconds")
    
    n_total = len(object_ids)
    logger.info(f"Total objects in dataset: {n_total}")
    
    # Get original ranking
    original_indices = np.argsort(similarities)[::-1]
    
    # Load RA/DEC data once
    ra_dec_df = pd.read_csv("data/evals/lens/lens_eval_objects.csv")
    oid_to_radec = {str(row['object_id']): (row['ra'], row['dec']) 
                     for _, row in ra_dec_df.iterrows()}
    
    ra_values = []
    dec_values = []
    for oid in object_ids:
        if oid in oid_to_radec:
            ra, dec = oid_to_radec[oid]
            ra_values.append(ra)
            dec_values.append(dec)
        else:
            ra_values.append(np.nan)
            dec_values.append(np.nan)
    
    # Save common data once
    common_data_file = exp_dir / "common_data.npz"
    np.savez_compressed(
        common_data_file,
        object_ids=np.array(object_ids),
        ra=np.array(ra_values),
        dec=np.array(dec_values),
        eval_grades=np.array(eval_grades),
        similarities=similarities,
        original_indices=original_indices
    )
    
    # Summary data
    summary = {
        'experiment_dir': str(exp_dir),
        'checkpoint_path': checkpoint_path,
        'models': models,
        'top_k_values': top_k_values,
        'n_runs': n_runs,
        'best_of_m_values': best_of_m_values,
        'zoom': zoom,
        'model_load_time': load_time,
        'experiments': []
    }
    
    # Pre-load images for different top-k values
    images_cache = {}
    for top_k in top_k_values:
        logger.info(f"Pre-loading images for top-{top_k}...")
        top_indices = original_indices[:top_k]
        
        if hsc_images is None:
            # Load from pre-saved files
            if cache_hdf5 is None:
                raise ValueError("--cache-hdf5 must be specified when hsc_images is not available")
            
            if not Path(cache_hdf5).exists():
                raise ValueError(f"Cache HDF5 file not found: {cache_hdf5}")
            
            top_k_file = cache_hdf5
            
            logger.info(f"Loading images from {top_k_file}")
            with h5py.File(top_k_file, 'r') as f:
                top_k_object_ids = [f['object_id'][i].decode('utf-8') if isinstance(f['object_id'][i], bytes) 
                                   else f['object_id'][i] for i in range(f['object_id'].shape[0])]
                top_k_images = f['hsc_image'][:]
            
            # Create mapping
            oid_to_image = dict(zip(top_k_object_ids, top_k_images))
            
            # Get images for our top indices
            images_to_rerank = []
            for idx in top_indices:
                oid = object_ids[idx]
                if oid in oid_to_image:
                    images_to_rerank.append(oid_to_image[oid])
                else:
                    logger.warning(f"Image not found for object {oid}")
                    raise ValueError(f"Image not found for object {oid} in cache file {top_k_file}")
        else:
            images_to_rerank = hsc_images[top_indices]
        
        images_cache[top_k] = (images_to_rerank, top_indices)
    
    # Run experiments
    total_experiments = len(models) * len(top_k_values) * len(best_of_m_values) * n_runs
    experiment_num = 0
    
    for model_id in models:
        if model_id not in models_info:
            logger.error(f"Model {model_id} not found in models.jsonl")
            continue
            
        model_info = models_info[model_id]
        
        for top_k in top_k_values:
            # Get cached images and indices
            images_to_rerank, top_indices_to_rerank = images_cache[top_k]
            
            for best_of_m in best_of_m_values:
                # Run multiple times if requested
                run_results = []
                
                for run in range(n_runs):
                    experiment_num += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Running experiment {experiment_num}/{total_experiments}")
                    logger.info(f"Model: {model_id}, Top-K: {top_k}, Best-of-M: {best_of_m}, Run: {run+1}/{n_runs}")
                    logger.info(f"{'='*60}")
                
                    start_time = time.time()
                    
                    try:
                        # Create output directory for this configuration
                        run_dir = exp_dir / f"{model_id}_top{top_k}_m{best_of_m}"
                        run_dir.mkdir(exist_ok=True)
                    
                        # Run reranking with cached data
                        run_result = rerank_with_cached_data(
                            object_ids=object_ids,
                            similarities=similarities,
                            eval_grades=eval_grades,
                            images_to_rerank=images_to_rerank,
                            top_indices_to_rerank=top_indices_to_rerank,
                            api_key=api_key,
                            model_id=model_id,
                            model_info=model_info,
                            output_dir=run_dir,
                            zoom=zoom,
                            k_values=[10, 20, 30, 50, 100, 200, 500, 1000],
                            run_number=run+1,
                            best_of_m=best_of_m,
                            logger=logger
                        )
                        
                        run_result['elapsed_time'] = time.time() - start_time
                        run_results.append(run_result)
                        
                    except Exception as e:
                        logger.error(f"Error in experiment run {run+1}: {e}")
                
                if run_results:
                    # Calculate statistics across runs
                    experiment_summary = {
                        'model_id': model_id,
                        'model_name': model_info['formatted_name'],
                        'top_k': top_k,
                        'best_of_m': best_of_m,
                        'n_runs': len(run_results),
                        'status': 'success',
                        'output_dir': str(run_dir)
                    }
                
                    # Calculate means and standard errors
                    for key in run_results[0].keys():
                        if key in ['run_number', 'elapsed_time']:
                            continue
                        
                        values = [r[key] for r in run_results]
                        if isinstance(values[0], (int, float)):
                            experiment_summary[f'{key}_mean'] = np.mean(values)
                            if len(values) > 1:
                                experiment_summary[f'{key}_stderr'] = np.std(values, ddof=1) / np.sqrt(len(values))
                            else:
                                experiment_summary[f'{key}_stderr'] = 0
                    
                    # Save detailed run results
                    with open(run_dir / 'run_results.json', 'w') as f:
                        json.dump(run_results, f, indent=2)
                    
                    summary['experiments'].append(experiment_summary)
                else:
                    summary['experiments'].append({
                        'model_id': model_id,
                        'top_k': top_k,
                        'best_of_m': best_of_m,
                        'status': 'failed',
                        'error': 'All runs failed'
                    })
                
                # Save summary after each configuration
                with open(exp_dir / "experiment_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
    
    # Calculate total costs
    total_cost_mean = sum(exp.get('total_cost_mean', 0) for exp in summary['experiments'] if exp['status'] == 'success')
    total_tokens_mean = sum(exp.get('total_input_tokens_mean', 0) + exp.get('total_output_tokens_mean', 0) 
                           for exp in summary['experiments'] if exp['status'] == 'success')
    
    summary['total_cost_mean'] = total_cost_mean
    summary['total_tokens_mean'] = total_tokens_mean
    
    # Save final summary
    with open(exp_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All experiments completed!")
    logger.info(f"Mean total cost: ${total_cost_mean:.4f}")
    logger.info(f"Mean total tokens: {total_tokens_mean:,.0f}")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"{'='*60}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Optimized multi-model rerank experiments")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"],
                       help="Model IDs to use")
    parser.add_argument("--top-k-values", type=int, nargs='+', 
                       default=[100, 1000],
                       help="Top-k values to test")
    parser.add_argument("--n-runs", type=int, default=1,
                       help="Number of runs per configuration (default: 1)")
    parser.add_argument("--best-of-m-values", type=int, nargs='+', default=[1],
                       help="List of best-of-m values to test (default: [1])")
    parser.add_argument("--zoom", action="store_true",
                       help="Apply 50% zoom to center of images")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--cache-hdf5", type=str, default=None,
                       help="Path to cached HDF5 file with pre-computed images")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Add missing imports
    global torch, h5py
    import torch
    import h5py
    
    # Run all experiments
    logger.info("Starting optimized multi-model rerank experiments")
    logger.info(f"Models: {args.models}")
    logger.info(f"Top-K values: {args.top_k_values}")
    logger.info(f"Number of runs per configuration: {args.n_runs}")
    logger.info(f"Best-of-M values to test: {args.best_of_m_values}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    if args.zoom:
        logger.info("Zoom mode enabled")
    
    summary = run_all_experiments(
        checkpoint_path=args.checkpoint,
        api_key=api_key,
        models=args.models,
        top_k_values=args.top_k_values,
        n_runs=args.n_runs,
        best_of_m_values=args.best_of_m_values,
        zoom=args.zoom,
        cache_hdf5=args.cache_hdf5,
        logger=logger
    )
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZED MULTI-MODEL RERANK EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Experiment directory: {summary['experiment_dir']}")
    print(f"Total configurations: {len(summary['experiments'])}")
    print(f"Runs per configuration: {args.n_runs}")
    print(f"Best-of-M values tested: {args.best_of_m_values}")
    print(f"Successful: {sum(1 for e in summary['experiments'] if e['status'] == 'success')}")
    print(f"Failed: {sum(1 for e in summary['experiments'] if e['status'] == 'failed')}")
    print(f"Mean total cost: ${summary['total_cost_mean']:.4f}")
    print(f"Mean total tokens: {summary['total_tokens_mean']:,.0f}")
    print(f"Model load time: {summary['model_load_time']:.2f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()