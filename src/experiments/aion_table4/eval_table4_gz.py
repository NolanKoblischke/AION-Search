"""
Evaluate CLIP model on Galaxy Zoo 5 dataset with AION embeddings.
This script evaluates the trained CLIP model on merger and spiral detection tasks.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import torch
import logging
from typing import Dict, Tuple, List
import json
from datetime import datetime
import pandas as pd
from PIL import Image
import os

# Import evaluation utilities
from src.evals.eval_utils import generate_query_embedding, setup_logging
from src.clip.models.clip_model import GalaxyClipModel
from src.plotting_scripts.default_plot import process_galaxy_image


def dcg(r):
    """Compute Discounted Cumulative Gain (DCG)."""
    return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))


def ndcg_score(relevances, top_k):
    """Compute NDCG@k using fraction method.
    
    Args:
        relevances: Relevance scores in ranked order
        top_k: Number of top results to consider
    
    Returns:
        NDCG@k score
    """
    actual_dcg = dcg(relevances[:top_k])
    ideal_dcg = dcg(np.sort(relevances)[::-1][:top_k])
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def load_clip_model(model_path: str, device: str = 'cuda') -> GalaxyClipModel:
    """Load pretrained CLIP model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Create model instance
    model = GalaxyClipModel(
        image_input_dim=model_config['image_input_dim'],
        text_input_dim=model_config['text_input_dim'],
        embedding_dim=model_config['embedding_dim'],
        use_mean_embeddings=model_config['use_mean_embeddings']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_table4_gz(
    hdf5_path: str,
    model_path: str,
    device: str = 'cuda',
    batch_size: int = 512,
    k: int = 10,
    output_dir: str = None,
    logger: logging.Logger = None,
    apply_quality_filter: bool = True,
    save_top_m_images: int = 20
) -> Dict[str, float]:
    """
    Evaluate CLIP model on Galaxy Zoo 5 dataset.
    
    Args:
        hdf5_path: Path to gz5_base_embedded.hdf5 file
        model_path: Path to pretrained CLIP model
        device: Device to use for computation
        batch_size: Batch size for processing embeddings
        k: Top-k for NDCG calculation
        output_dir: Optional directory to save results
        logger: Optional logger instance
    
    Returns:
        Dictionary with evaluation metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load CLIP model
    logger.info(f"Loading CLIP model from: {model_path}")
    model = load_clip_model(model_path, device)
    
    # Query texts
    queries = {
        'spirals': 'visible spiral arms',
        'mergers': 'merging'
    }
    
    # Generate query embeddings
    logger.info("Generating query embeddings...")
    query_embeddings = {}
    for query_name, query_text in queries.items():
        logger.info(f"  Generating embedding for: '{query_text}'")
        # Generate OpenAI embedding
        openai_embedding = generate_query_embedding(query_text)
        
        # Project through CLIP model
        query_tensor = torch.tensor(openai_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            query_features = model.text_projector(query_tensor)
        query_features = query_features.cpu().numpy()
        query_embeddings[query_name] = query_features
    
    # Process HDF5 file
    logger.info(f"Processing HDF5 file: {hdf5_path}")
    
    # Lists to store results
    all_spiral_scores = []
    all_merger_scores = []
    all_spiral_similarities = []
    all_merger_similarities = []
    all_object_ids = []
    all_file_indices = []
    
    # Load image data file for later retrieval
    images_hdf5_path = hdf5_path.replace('gz5_base_embedded.hdf5', 'gz5_legacysurvey_images.hdf5')
    if not os.path.exists(images_hdf5_path):
        logger.warning(f"Images file not found at: {images_hdf5_path}")
        logger.warning("Will not save galaxy images")
        save_top_m_images = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        # Access the astropy table
        data_table = f['__astropy_table__']
        n_rows_original = len(data_table)
        logger.info(f"Total galaxies in dataset: {n_rows_original:,}")
        
        # Apply quality filter if requested (matching lanusse2024)
        if apply_quality_filter:
            # First, load all vote counts
            vote_counts = np.array(data_table['smooth-or-featured_total-votes'])
            quality_mask = vote_counts >= 3
            valid_indices = np.where(quality_mask)[0]
            n_rows = len(valid_indices)
            logger.info(f"After quality filter (votes >= 3): {n_rows:,} galaxies ({n_rows/n_rows_original*100:.1f}%)")
        else:
            valid_indices = np.arange(n_rows_original)
            n_rows = n_rows_original
        
        # Process in batches
        for batch_idx in tqdm(range(0, len(valid_indices), batch_size), desc="Processing galaxies"):
            batch_end = min(batch_idx + batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_idx:batch_end]
            
            # Load batch data using valid indices
            data_chunk = data_table[batch_indices]
            
            # Extract AION embeddings (768-dimensional)
            aion_embeddings = np.array(data_chunk['embeddings'])
            
            # Extract relevance scores
            merger_fractions = np.array(data_chunk['merging_merger_fraction'])
            spiral_fractions = np.array(data_chunk['has-spiral-arms_yes_fraction'])
            
            # Extract object IDs
            object_ids = data_chunk['object_id']
            if isinstance(object_ids[0], bytes):
                object_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else oid for oid in object_ids]
            
            # Project AION embeddings through CLIP model
            embeddings_tensor = torch.tensor(aion_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                clip_features = model.image_projector(embeddings_tensor)
            clip_features = clip_features.cpu().numpy()
            
            # Compute similarities for both queries
            spiral_sims = clip_features @ query_embeddings['spirals'].T
            merger_sims = clip_features @ query_embeddings['mergers'].T
            
            # Store results
            all_spiral_scores.extend(spiral_fractions.tolist())
            all_merger_scores.extend(merger_fractions.tolist())
            all_spiral_similarities.extend(spiral_sims.squeeze().tolist())
            all_merger_similarities.extend(merger_sims.squeeze().tolist())
            all_object_ids.extend(object_ids)
            all_file_indices.extend(batch_indices.tolist())
    
    # Convert to numpy arrays
    all_spiral_scores = np.array(all_spiral_scores)
    all_merger_scores = np.array(all_merger_scores)
    all_spiral_similarities = np.array(all_spiral_similarities)
    all_merger_similarities = np.array(all_merger_similarities)
    all_object_ids = np.array(all_object_ids)
    all_file_indices = np.array(all_file_indices)
    
    # Compute NDCG@k for spirals
    logger.info("\nComputing NDCG scores...")
    
    # Sort by similarity scores
    spiral_indices = np.argsort(all_spiral_similarities)[::-1]
    merger_indices = np.argsort(all_merger_similarities)[::-1]
    
    # Get relevance scores in ranked order
    spiral_relevances_ranked = all_spiral_scores[spiral_indices]
    merger_relevances_ranked = all_merger_scores[merger_indices]
    
    # Compute NDCG@k
    spiral_ndcg = ndcg_score(spiral_relevances_ranked, k)
    merger_ndcg = ndcg_score(merger_relevances_ranked, k)
    
    # Compute random baselines
    logger.info("\nComputing random baselines (50 shuffles)...")
    from numpy.random import default_rng
    rng = default_rng(42)  # Fixed seed for reproducibility
    
    spiral_random_ndcgs = []
    merger_random_ndcgs = []
    
    for i in range(50):
        # Random shuffle for spirals
        random_indices = rng.permutation(len(all_spiral_scores))
        random_spiral_relevances = all_spiral_scores[random_indices]
        spiral_random_ndcgs.append(ndcg_score(random_spiral_relevances, k))
        
        # Random shuffle for mergers
        random_indices = rng.permutation(len(all_merger_scores))
        random_merger_relevances = all_merger_scores[random_indices]
        merger_random_ndcgs.append(ndcg_score(random_merger_relevances, k))
    
    spiral_random_baseline = np.mean(spiral_random_ndcgs)
    spiral_random_std = np.std(spiral_random_ndcgs)
    merger_random_baseline = np.mean(merger_random_ndcgs)
    merger_random_std = np.std(merger_random_ndcgs)
    
    # Debug info
    logger.info("\nDebug NDCG calculation:")
    logger.info(f"  Spiral - Retrieved relevances (top {k}): {spiral_relevances_ranked[:k]}")
    logger.info(f"  Spiral - Ideal relevances: {np.sort(spiral_relevances_ranked)[::-1][:k]}")
    logger.info(f"  Merger - Retrieved relevances (top {k}): {merger_relevances_ranked[:k]}")
    logger.info(f"  Merger - Ideal relevances: {np.sort(merger_relevances_ranked)[::-1][:k]}")
    
    # Compute additional statistics
    results = {
        'spiral_ndcg@10': spiral_ndcg,
        'merger_ndcg@10': merger_ndcg,
        'avg_ndcg@10': (spiral_ndcg + merger_ndcg) / 2,
        'total_galaxies': n_rows,
        'spiral_avg_fraction': float(np.mean(all_spiral_scores)),
        'merger_avg_fraction': float(np.mean(all_merger_scores)),
        'spiral_top10_avg_fraction': float(np.mean(spiral_relevances_ranked[:k])),
        'merger_top10_avg_fraction': float(np.mean(merger_relevances_ranked[:k])),
        # Additional debug stats
        'spiral_max_fraction': float(np.max(all_spiral_scores)),
        'merger_max_fraction': float(np.max(all_merger_scores)),
        'spiral_num_above_0.5': int(np.sum(np.array(all_spiral_scores) > 0.5)),
        'merger_num_above_0.5': int(np.sum(np.array(all_merger_scores) > 0.5)),
        # Random baselines
        'spiral_random_ndcg@10': spiral_random_baseline,
        'spiral_random_ndcg@10_std': spiral_random_std,
        'merger_random_ndcg@10': merger_random_baseline,
        'merger_random_ndcg@10_std': merger_random_std,
    }
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"  Quality filter applied: {apply_quality_filter}")
    logger.info(f"  Spiral NDCG@{k}: {spiral_ndcg:.4f}")
    logger.info(f"  Merger NDCG@{k}: {merger_ndcg:.4f}")
    logger.info(f"  Average NDCG@{k}: {results['avg_ndcg@10']:.4f}")
    logger.info(f"\nRandom Baselines (50 shuffles):")
    logger.info(f"  Spiral random NDCG@{k}: {spiral_random_baseline:.4f} ± {spiral_random_std:.4f}")
    logger.info(f"  Merger random NDCG@{k}: {merger_random_baseline:.4f} ± {merger_random_std:.4f}")
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Spiral avg fraction (dataset): {results['spiral_avg_fraction']:.4f}")
    logger.info(f"  Spiral avg fraction (top {k}): {results['spiral_top10_avg_fraction']:.4f}")
    logger.info(f"  Merger avg fraction (dataset): {results['merger_avg_fraction']:.4f}")
    logger.info(f"  Merger avg fraction (top {k}): {results['merger_top10_avg_fraction']:.4f}")
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_path / f"table4_gz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Save ALL galaxies as CSV files (sorted by similarity)
        # All spirals
        all_spiral_data = []
        for i, idx in enumerate(spiral_indices):  # Save all galaxies
            all_spiral_data.append({
                'rank': i + 1,
                'object_id': all_object_ids[idx],
                'file_index': all_file_indices[idx],
                'similarity': float(all_spiral_similarities[idx]),
                'spiral_fraction': float(all_spiral_scores[idx])
            })
        
        spiral_df = pd.DataFrame(all_spiral_data)
        spiral_csv = output_path / f"table4_gz_all_spirals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        spiral_df.to_csv(spiral_csv, index=False)
        logger.info(f"All spirals saved to: {spiral_csv} ({len(spiral_df)} galaxies)")
        
        # All mergers
        all_merger_data = []
        for i, idx in enumerate(merger_indices):  # Save all galaxies
            all_merger_data.append({
                'rank': i + 1,
                'object_id': all_object_ids[idx],
                'file_index': all_file_indices[idx],
                'similarity': float(all_merger_similarities[idx]),
                'merger_fraction': float(all_merger_scores[idx])
            })
        
        merger_df = pd.DataFrame(all_merger_data)
        merger_csv = output_path / f"table4_gz_all_mergers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        merger_df.to_csv(merger_csv, index=False)
        logger.info(f"All mergers saved to: {merger_csv} ({len(merger_df)} galaxies)")
        
        # Save top M images if requested
        if save_top_m_images > 0 and os.path.exists(images_hdf5_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            images_base_dir = output_path / f"images_{timestamp}"
            
            logger.info(f"\nSaving top {save_top_m_images} images for each query...")
            
            # Get top object IDs for spirals and mergers
            top_spiral_ids = {}
            for idx in spiral_indices[:save_top_m_images]:
                obj_id = str(all_object_ids[idx])
                top_spiral_ids[obj_id] = idx
            
            top_merger_ids = {}
            for idx in merger_indices[:save_top_m_images]:
                obj_id = str(all_object_ids[idx])
                top_merger_ids[obj_id] = idx
            
            # All target object IDs
            all_target_ids = set(top_spiral_ids.keys()) | set(top_merger_ids.keys())
            logger.info(f"  Looking for {len(all_target_ids)} unique object IDs")
            
            # Create directories
            spiral_img_dir = images_base_dir / "spirals"
            spiral_img_dir.mkdir(parents=True, exist_ok=True)
            merger_img_dir = images_base_dir / "mergers"
            merger_img_dir.mkdir(parents=True, exist_ok=True)
            
            # Track which images we've saved
            saved_spiral_ids = set()
            saved_merger_ids = set()
            
            with h5py.File(images_hdf5_path, 'r') as img_f:
                img_table = img_f['__astropy_table__']
                total_galaxies = len(img_table)
                
                logger.info(f"  Total galaxies in image file: {total_galaxies:,}")
                
                # First, try to build an index if the dataset is not too large
                # This is faster for finding specific objects
                logger.info("  Building object ID index...")
                object_id_to_index = {}
                
                # Process in chunks to build index
                index_chunk_size = 10000
                n_chunks = (total_galaxies + index_chunk_size - 1) // index_chunk_size
                
                found_count = 0
                for chunk_idx in tqdm(range(n_chunks), desc="Building index"):
                    start_idx = chunk_idx * index_chunk_size
                    end_idx = min(start_idx + index_chunk_size, total_galaxies)
                    
                    # Load chunk of object IDs only (much faster than loading images)
                    chunk_object_ids = img_table['object_id'][start_idx:end_idx]
                    
                    # Check each object ID
                    for local_idx, obj_id in enumerate(chunk_object_ids):
                        if isinstance(obj_id, bytes):
                            obj_id = obj_id.decode('utf-8')
                        obj_id = str(obj_id)
                        
                        # Only store indices for objects we care about
                        if obj_id in all_target_ids:
                            global_idx = start_idx + local_idx
                            object_id_to_index[obj_id] = global_idx
                            found_count += 1
                            
                            # Early exit if we found all objects
                            if found_count == len(all_target_ids):
                                logger.info(f"  Found all {found_count} target objects, stopping index build")
                                break
                    
                    if found_count == len(all_target_ids):
                        break
                
                logger.info(f"  Index built. Found {len(object_id_to_index)}/{len(all_target_ids)} target objects")
                
                # Now load and save the images using batch loading
                logger.info("  Loading and saving images...")
                
                # Collect all indices for spirals that we found
                spiral_indices_to_load = []
                spiral_obj_ids_ordered = []
                for obj_id in top_spiral_ids.keys():
                    if obj_id in object_id_to_index:
                        spiral_indices_to_load.append(object_id_to_index[obj_id])
                        spiral_obj_ids_ordered.append(obj_id)
                
                # Collect all indices for mergers that we found
                merger_indices_to_load = []
                merger_obj_ids_ordered = []
                for obj_id in top_merger_ids.keys():
                    if obj_id in object_id_to_index:
                        merger_indices_to_load.append(object_id_to_index[obj_id])
                        merger_obj_ids_ordered.append(obj_id)
                
                # Load ALL spiral images at once using fancy indexing
                if spiral_indices_to_load:
                    logger.info(f"  Loading {len(spiral_indices_to_load)} spiral images in one batch...")
                    try:
                        # Sort indices for potentially better disk access
                        sorted_pairs = sorted(zip(spiral_indices_to_load, spiral_obj_ids_ordered))
                        spiral_indices_sorted = [idx for idx, _ in sorted_pairs]
                        spiral_obj_ids_sorted = [oid for _, oid in sorted_pairs]
                        
                        # Batch load all spiral images
                        all_spiral_images = img_table['image_array'][spiral_indices_sorted]
                        
                        # Process and save them
                        logger.info("  Processing and saving spiral images...")
                        for image_array, obj_id in tqdm(zip(all_spiral_images, spiral_obj_ids_sorted), 
                                                       total=len(spiral_obj_ids_sorted),
                                                       desc="Saving spiral images"):
                            rgb_image = process_galaxy_image(image_array)
                            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_uint8)
                            img.save(spiral_img_dir / f"{obj_id}.png")
                            saved_spiral_ids.add(obj_id)
                    except Exception as e:
                        logger.error(f"Error batch loading spiral images: {e}")
                        logger.info("  Falling back to individual loading...")
                        # Fallback to individual loading if batch fails
                        for obj_id in tqdm(spiral_obj_ids_ordered, desc="Saving spiral images (fallback)"):
                            idx = object_id_to_index[obj_id]
                            image_array = np.array(img_table['image_array'][idx])
                            rgb_image = process_galaxy_image(image_array)
                            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_uint8)
                            img.save(spiral_img_dir / f"{obj_id}.png")
                            saved_spiral_ids.add(obj_id)
                
                # Load ALL merger images at once using fancy indexing
                if merger_indices_to_load:
                    logger.info(f"  Loading {len(merger_indices_to_load)} merger images in one batch...")
                    try:
                        # Sort indices for potentially better disk access
                        sorted_pairs = sorted(zip(merger_indices_to_load, merger_obj_ids_ordered))
                        merger_indices_sorted = [idx for idx, _ in sorted_pairs]
                        merger_obj_ids_sorted = [oid for _, oid in sorted_pairs]
                        
                        # Batch load all merger images
                        all_merger_images = img_table['image_array'][merger_indices_sorted]
                        
                        # Process and save them
                        logger.info("  Processing and saving merger images...")
                        for image_array, obj_id in tqdm(zip(all_merger_images, merger_obj_ids_sorted),
                                                       total=len(merger_obj_ids_sorted),
                                                       desc="Saving merger images"):
                            rgb_image = process_galaxy_image(image_array)
                            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_uint8)
                            img.save(merger_img_dir / f"{obj_id}.png")
                            saved_merger_ids.add(obj_id)
                    except Exception as e:
                        logger.error(f"Error batch loading merger images: {e}")
                        logger.info("  Falling back to individual loading...")
                        # Fallback to individual loading if batch fails
                        for obj_id in tqdm(merger_obj_ids_ordered, desc="Saving merger images (fallback)"):
                            idx = object_id_to_index[obj_id]
                            image_array = np.array(img_table['image_array'][idx])
                            rgb_image = process_galaxy_image(image_array)
                            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_uint8)
                            img.save(merger_img_dir / f"{obj_id}.png")
                            saved_merger_ids.add(obj_id)
            
            # Report results
            logger.info(f"\nImage saving complete:")
            logger.info(f"  Spirals: saved {len(saved_spiral_ids)}/{len(top_spiral_ids)} images to {spiral_img_dir}")
            logger.info(f"  Mergers: saved {len(saved_merger_ids)}/{len(top_merger_ids)} images to {merger_img_dir}")
            
            # Warn about missing images
            missing_spirals = set(top_spiral_ids.keys()) - saved_spiral_ids
            if missing_spirals:
                logger.warning(f"  Could not find {len(missing_spirals)} spiral images: {list(missing_spirals)[:5]}...")
            
            missing_mergers = set(top_merger_ids.keys()) - saved_merger_ids
            if missing_mergers:
                logger.warning(f"  Could not find {len(missing_mergers)} merger images: {list(missing_mergers)[:5]}...")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on Galaxy Zoo 5 dataset")
    
    parser.add_argument("--hdf5-path", type=str,
                       default="data/gz5_base_embedded.hdf5",
                       help="Path to gz5_base_embedded.hdf5 file")
    parser.add_argument("--model-path", type=str,
                       default="runs/data_scaling_experiment_july/summaries-only/train_size_75000/run_1/best_spirals_model.pt",
                       help="Path to pretrained CLIP model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for processing")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k for NDCG calculation")
    parser.add_argument("--output-dir", type=str, default="data/eval_results/table4_gz",
                       help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no-quality-filter", action="store_true",
                       help="Disable quality filtering (votes >= 3)")
    parser.add_argument("--save-top-m-images", type=int, default=20,
                       help="Number of top images to save for each query (default: 20)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run evaluation
    results = evaluate_table4_gz(
        hdf5_path=args.hdf5_path,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        k=args.k,
        output_dir=args.output_dir,
        logger=logger,
        apply_quality_filter=not args.no_quality_filter,
        save_top_m_images=args.save_top_m_images
    )
    
    print(f"\nEvaluation complete!")
    print(f"Average NDCG@{args.k}: {results['avg_ndcg@10']:.4f}")


if __name__ == "__main__":
    main()

