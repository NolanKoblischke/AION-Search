"""
Evaluate CLIP model on gravitational lens detection using AION embeddings.
This script evaluates the trained CLIP model on lens detection tasks.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import torch
import logging
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from PIL import Image
import os
import csv

# Import evaluation utilities
from src.evals.eval_utils import generate_query_embedding, setup_logging
from src.clip.models.clip_model import AIONSearchClipModel
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


def load_clip_model(model_path: str, device: str = 'cuda') -> AIONSearchClipModel:
    """Load pretrained CLIP model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Create model instance
    model = AIONSearchClipModel(
        image_input_dim=model_config['image_input_dim'],
        text_input_dim=model_config['text_input_dim'],
        embedding_dim=model_config['embedding_dim'],
        use_mean_embeddings=model_config['use_mean_embeddings']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_lens_catalogs(
    masterlens_path: str,
    hsc_lenses_path: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load and combine lens catalogs."""
    # Load masterlens catalog
    logger.info(f"Loading masterlens catalog from: {masterlens_path}")
    masterlens = pd.read_csv(masterlens_path)
    logger.info(f"  Loaded {len(masterlens)} lenses from masterlens")
    
    # Load HSC lenses catalog  
    logger.info(f"Loading HSC lenses catalog from: {hsc_lenses_path}")
    hsc_lenses = pd.read_csv(hsc_lenses_path)
    logger.info(f"  Loaded {len(hsc_lenses)} lenses from HSC")
    
    # Remove overlaps (keeping HSC when there's a match within 1 arcsec)
    masterlens_coords = SkyCoord(ra=masterlens['ra'].values*u.degree, 
                                 dec=masterlens['dec'].values*u.degree)
    hsc_coords = SkyCoord(ra=hsc_lenses['ra'].values*u.degree, 
                          dec=hsc_lenses['dec'].values*u.degree)
    
    idx, sep2d, _ = masterlens_coords.match_to_catalog_sky(hsc_coords)
    mask_no_overlap = sep2d.arcsec > 1.0
    masterlens_clean = masterlens[mask_no_overlap]
    
    # Combine catalogs
    lens_df = pd.concat([masterlens_clean, hsc_lenses], ignore_index=True)
    logger.info(f"  Total lenses in combined catalog: {len(lens_df)}")
    
    return lens_df


def match_catalogs(
    hdf5_path: str,
    masterlens_path: str,
    hsc_lenses_path: str,
    logger: logging.Logger,
    matching_radius: float = 1.0
) -> Tuple[Table, Dict[str, str]]:
    """
    Match lens catalogs to parent sample and return matched table and lens mapping.
    
    Returns:
        lens_table: Astropy table of matched lenses
        lens_mapping: Dict mapping object_id to lens grade
    """
    # Load lens catalogs
    lens_df = load_lens_catalogs(masterlens_path, hsc_lenses_path, logger)
    
    # Load parent sample as astropy table
    logger.info(f"Loading parent sample from: {hdf5_path}")
    parent_table = Table.read(hdf5_path)
    logger.info(f"  Loaded {len(parent_table)} objects from parent sample")
    
    # Create coordinates
    parent_coords = SkyCoord(ra=parent_table['ra']*u.degree, 
                            dec=parent_table['dec']*u.degree)
    lens_coords = SkyCoord(ra=lens_df['ra'].values*u.degree, 
                          dec=lens_df['dec'].values*u.degree)
    
    # Match lenses to parent sample
    logger.info(f"Matching lenses to parent sample (radius={matching_radius} arcsec)...")
    idx_parent, idx_lens, d2d, _ = lens_coords.search_around_sky(
        parent_coords, matching_radius*u.arcsec
    )
    
    # Create lens mapping
    lens_mapping = {}
    for i, (parent_idx, lens_idx) in enumerate(zip(idx_parent, idx_lens)):
        object_id = str(parent_table['object_id'][parent_idx])
        grade = lens_df.iloc[lens_idx]['grade']
        lens_mapping[object_id] = grade
    
    logger.info(f"  Found {len(lens_mapping)} matched lenses")
    
    # Count by grade
    grade_counts = {}
    for grade in lens_mapping.values():
        grade_str = str(grade)  # Convert to string to handle mixed types
        grade_counts[grade_str] = grade_counts.get(grade_str, 0) + 1
    for grade in sorted(grade_counts.keys()):
        logger.info(f"    Grade {grade}: {grade_counts[grade]}")
    
    # Create matched lens table
    matched_indices = list(set(idx_parent))
    lens_table = parent_table[matched_indices]
    
    return lens_table, lens_mapping


def evaluate_table4_lens(
    hdf5_path: str,
    masterlens_path: str,
    hsc_lenses_path: str,
    model_path: str,
    device: str = 'cuda',
    batch_size: int = 512,
    k: int = 10,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    matching_radius: float = 1.0,
    save_top_m_images: int = 20,
    lens_index_path: Optional[str] = None,
    skip_images: bool = False
) -> Dict[str, float]:
    """
    Evaluate CLIP model on lens detection.
    
    Args:
        hdf5_path: Path to lens parent sample HDF5 file
        masterlens_path: Path to masterlens CSV file
        hsc_lenses_path: Path to HSC lenses CSV file
        model_path: Path to pretrained CLIP model
        device: Device to use for computation
        batch_size: Batch size for processing embeddings
        k: Top-k for NDCG calculation
        output_dir: Optional directory to save results
        logger: Optional logger instance
        matching_radius: Matching radius in arcseconds
    
    Returns:
        Dictionary with evaluation metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load CLIP model
    logger.info(f"Loading CLIP model from: {model_path}")
    model = load_clip_model(model_path, device)
    
    # Query text for gravitational lens
    query_text = 'gravitational lens'
    
    # Generate query embedding
    logger.info(f"Generating query embedding for: '{query_text}'")
    openai_embedding = generate_query_embedding(query_text)
    
    # Project through CLIP model
    query_tensor = torch.tensor(openai_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.text_projector(query_tensor)
    query_features = query_features.cpu().numpy()
    
    # Match catalogs and get lens mapping
    lens_table, lens_mapping = match_catalogs(
        hdf5_path, masterlens_path, hsc_lenses_path, logger, matching_radius
    )
    
    # Process HDF5 file
    logger.info(f"Processing HDF5 file: {hdf5_path}")
    
    # Lists to store results
    all_lens_scores = []
    all_similarities = []
    all_object_ids = []
    all_ra = []
    all_dec = []
    all_file_indices = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Access the astropy table
        data_table = f['__astropy_table__']
        n_rows_original = len(data_table)
        logger.info(f"Total galaxies in dataset: {n_rows_original:,}")
        
        # For memory efficiency, process all indices (no filtering in this case)
        valid_indices = np.arange(n_rows_original)
        n_rows = len(valid_indices)
        
        # Process in batches using valid indices
        for batch_idx in tqdm(range(0, len(valid_indices), batch_size), desc="Processing galaxies"):
            batch_end = min(batch_idx + batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_idx:batch_end]
            
            # Load batch data using valid indices
            data_chunk = data_table[batch_indices]
            
            # Extract embeddings (768-dimensional)
            embeddings = np.array(data_chunk['embeddings_hsc'])
            
            # Extract object IDs
            object_ids = data_chunk['object_id']
            object_ids = [str(oid) for oid in object_ids]
            
            # Extract coordinates
            ra_values = np.array(data_chunk['ra'])
            dec_values = np.array(data_chunk['dec'])
            
            # Project embeddings through CLIP model
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                clip_features = model.image_projector(embeddings_tensor)
            clip_features = clip_features.cpu().numpy()
            
            # Compute similarities
            similarities = clip_features @ query_features.T
            similarities = similarities.squeeze()
            
            # Get relevance scores for this batch
            lens_scores = []
            for oid in object_ids:
                if oid in lens_mapping:
                    grade = lens_mapping[oid]
                    # Binary relevance: 1 for A, B, C grades, 0 otherwise
                    score = 1.0 if grade in ['A', 'B', 'C'] else 0.0
                else:
                    score = 0.0
                lens_scores.append(score)
            
            # Store results
            all_lens_scores.extend(lens_scores)
            all_similarities.extend(similarities.tolist())
            all_object_ids.extend(object_ids)
            all_ra.extend(ra_values.tolist())
            all_dec.extend(dec_values.tolist())
            all_file_indices.extend(batch_indices.tolist())
    
    # Convert to numpy arrays
    all_lens_scores = np.array(all_lens_scores)
    all_similarities = np.array(all_similarities)
    all_object_ids = np.array(all_object_ids)
    all_ra = np.array(all_ra)
    all_dec = np.array(all_dec)
    all_file_indices = np.array(all_file_indices)
    
    # Compute NDCG@k
    logger.info("\nComputing NDCG scores...")
    
    # Sort by similarity scores
    indices = np.argsort(all_similarities)[::-1]
    
    # Get relevance scores in ranked order
    relevances_ranked = all_lens_scores[indices]
    
    # Compute NDCG@k
    lens_ndcg = ndcg_score(relevances_ranked, k)
    
    # Compute random baselines
    logger.info("\nComputing random baselines (50 shuffles)...")
    from numpy.random import default_rng
    rng = default_rng(42)  # Fixed seed for reproducibility
    
    random_ndcgs = []
    for i in range(50):
        random_indices = rng.permutation(len(all_lens_scores))
        random_relevances = all_lens_scores[random_indices]
        random_ndcgs.append(ndcg_score(random_relevances, k))
    
    random_baseline = np.mean(random_ndcgs)
    random_std = np.std(random_ndcgs)
    
    # Debug info
    logger.info("\nDebug NDCG calculation:")
    logger.info(f"  Retrieved relevances (top {k}): {relevances_ranked[:k]}")
    logger.info(f"  Ideal relevances: {np.sort(relevances_ranked)[::-1][:k]}")
    
    # Compute additional statistics
    num_lenses = int(np.sum(all_lens_scores))
    results = {
        'lens_ndcg@10': lens_ndcg,
        'total_galaxies': n_rows,
        'total_lenses': num_lenses,
        'lens_fraction': float(num_lenses / n_rows),
        'lens_top10_count': int(np.sum(relevances_ranked[:k])),
        'lens_top10_fraction': float(np.sum(relevances_ranked[:k]) / k),
        # Random baselines
        'lens_random_ndcg@10': random_baseline,
        'lens_random_ndcg@10_std': random_std,
    }
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"  Lens NDCG@{k}: {lens_ndcg:.4f}")
    logger.info(f"  Random Baseline NDCG@{k}: {random_baseline:.4f} ± {random_std:.4f}")
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total lenses: {num_lenses} ({results['lens_fraction']*100:.2f}%)")
    logger.info(f"  Lenses in top {k}: {results['lens_top10_count']} ({results['lens_top10_fraction']*100:.1f}%)")
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_path / f"table4_lens_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Save ALL galaxies as CSV (sorted by similarity)
        all_data = []
        for i, idx in enumerate(indices):  # Save all galaxies
            all_data.append({
                'rank': i + 1,
                'object_id': all_object_ids[idx],
                'ra': all_ra[idx],
                'dec': all_dec[idx],
                'similarity': float(all_similarities[idx]),
                'is_lens': int(all_lens_scores[idx]),
                'grade': lens_mapping.get(all_object_ids[idx], 'N')
            })
        
        df = pd.DataFrame(all_data)
        csv_file = output_path / f"table4_lens_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"All results saved to: {csv_file} ({len(df)} galaxies)")
        
        # Save top M images if requested
        if save_top_m_images > 0 and not skip_images:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            images_base_dir = output_path / f"images_{timestamp}"
            lens_img_dir = images_base_dir / "lenses"
            lens_img_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nSaving top {save_top_m_images} lens images...")
            
            # Get base path for lens image catalog files
            lens_images_base = "data/"
            
            
            # Load CSV index to map object_ids to part files
            if not lens_index_path or not os.path.exists(lens_index_path):
                logger.error(f"Lens index CSV file not found: {lens_index_path}")
                logger.info("Please run build_lens_fits_index.py first to generate the index.")
                return results
            
            logger.info(f"Loading lens index from: {lens_index_path}")
            object_id_to_part = {}
            part_nums = set()
            
            # Also create arrays for coordinate matching
            index_ras = []
            index_decs = []
            index_object_ids = []
            index_part_info = []
            
            with open(lens_index_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    part_info_dict = {
                        'part_number': int(row['part_number']),
                        'row_index': int(row['row_index']),
                        'ra': float(row['ra']),
                        'dec': float(row['dec'])
                    }
                    object_id_to_part[row['object_id']] = part_info_dict
                    part_nums.add(int(row['part_number']))
                    
                    # Store for coordinate matching
                    index_ras.append(float(row['ra']))
                    index_decs.append(float(row['dec']))
                    index_object_ids.append(row['object_id'])
                    index_part_info.append(part_info_dict)
            
            logger.info(f"  Loaded index with {len(object_id_to_part)} entries")
            
            # Create SkyCoord for the index
            index_coords = SkyCoord(ra=index_ras*u.degree, dec=index_decs*u.degree)
            
            # Collect images until we have enough that exist in the catalog
            indices_by_part = {part_num: [] for part_num in sorted(part_nums)}
            not_found_count = 0
            coord_matched_count = 0
            found_count = 0
            search_idx = 0
            
            logger.info(f"\nSearching for top {save_top_m_images} images that exist in lens catalog...")
            
            while found_count < save_top_m_images and search_idx < len(indices):
                idx = indices[search_idx]
                object_id = all_object_ids[idx]
                
                if object_id in object_id_to_part:
                    part_info = object_id_to_part[object_id]
                    part_num = part_info['part_number']
                    row_idx = part_info['row_index']
                    
                    indices_by_part[part_num].append({
                        'global_idx': idx,
                        'row_index': row_idx,
                        'object_id': object_id,
                        'rank': search_idx + 1  # Store original rank
                    })
                    found_count += 1
                else:
                    # Try coordinate matching as fallback
                    obj_ra = all_ra[idx]
                    obj_dec = all_dec[idx]
                    obj_coord = SkyCoord(ra=obj_ra*u.degree, dec=obj_dec*u.degree)
                    
                    # Find nearest match in index
                    idx_match, sep2d, _ = obj_coord.match_to_catalog_sky(index_coords)
                    
                    if sep2d.arcsec < 1.0:  # Within 1.0 arcsec
                        matched_part_info = index_part_info[idx_match]
                        matched_obj_id = index_object_ids[idx_match]
                        
                        if search_idx < 100:  # Only log first 100 to avoid spam
                            logger.info(f"Object ID {object_id} not found in index, but matched by coordinates to {matched_obj_id} (separation: {sep2d.arcsec[0]:.3f} arcsec)")
                        coord_matched_count += 1
                        
                        part_num = matched_part_info['part_number']
                        row_idx = matched_part_info['row_index']
                        
                        indices_by_part[part_num].append({
                            'global_idx': idx,
                            'row_index': row_idx,
                            'object_id': object_id,
                            'matched_object_id': matched_obj_id,
                            'rank': search_idx + 1
                        })
                        found_count += 1
                    else:
                        if search_idx < 100:  # Only log first 100 to avoid spam
                            logger.warning(f"Object ID {object_id} not found in lens index and no close coordinate match (nearest: {sep2d.arcsec[0]:.1f} arcsec)")
                        not_found_count += 1
                
                search_idx += 1
            
            logger.info(f"\nSearched {search_idx} objects to find {found_count} images in lens catalog")
            if coord_matched_count > 0:
                logger.info(f"Matched {coord_matched_count} objects by coordinates")
            if not_found_count > 0:
                logger.info(f"Could not find {not_found_count} objects in the index")
            
            # Process each part file that has indices
            saved_count = 0
            for part_num in sorted(indices_by_part.keys()):
                if not indices_by_part[part_num]:
                    continue  # Skip if no indices in this part
                
                part_file = f"{lens_images_base}lens_image_catalog_part_{part_num:03d}.fits"
                if not os.path.exists(part_file):
                    logger.warning(f"Part file not found: {part_file}")
                    continue
                
                logger.info(f"  Processing part {part_num} with {len(indices_by_part[part_num])} galaxies...")
                
                try:
                    import fitsio
                    with fitsio.FITS(part_file) as fits:
                        # We already know exactly which rows to read from the CSV index
                        indices_to_load = [idx_info['row_index'] for idx_info in indices_by_part[part_num]]
                        object_id_order = [idx_info['object_id'] for idx_info in indices_by_part[part_num]]
                        
                        logger.info(f"    Reading {len(indices_to_load)} specific rows from FITS file...")
                        # Sort indices for potentially better disk access patterns
                        sorted_pairs = sorted(zip(indices_to_load, object_id_order))
                        indices_to_load = [p[0] for p in sorted_pairs]
                        object_id_order = [p[1] for p in sorted_pairs]
                        
                        # Read only the specific rows we need - this is memory efficient
                        data = fits[1].read_rows(indices_to_load)
                        
                        # Process the read data
                        for i, obj_id_str in enumerate(object_id_order):
                            # Get the image data
                            image_array = data['legacysurvey_image'][i]
                            rgb_image = process_galaxy_image(image_array)
                            
                            # Convert to PIL Image and save
                            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_uint8)
                            img.save(lens_img_dir / f"{obj_id_str}.png")
                            saved_count += 1
                                
                except Exception as e:
                    logger.error(f"Error processing part file {part_num}: {e}")
            
            logger.info(f"\nSaved {saved_count}/{save_top_m_images} lens images to: {lens_img_dir}")
        elif skip_images:
            logger.info("\nSkipping image saving as requested.")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on gravitational lens detection")
    
    parser.add_argument("--hdf5-path", type=str,
                       default="data/lens_parent_sample_v1_embedded_oct24_base.hdf5",
                       help="Path to lens parent sample HDF5 file")
    parser.add_argument("--masterlens-path", type=str,
                       default="data/masterlens/masterlens.csv",
                       help="Path to masterlens CSV file")
    parser.add_argument("--hsc-lenses-path", type=str,
                       default="data/hsc/hsc_lenses.csv",
                       help="Path to HSC lenses CSV file")
    parser.add_argument("--model-path", type=str,
                       default="runs/data_scaling_experiment_july/summaries-only/train_size_75000/run_1/best_spirals_model.pt",
                       help="Path to pretrained CLIP model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for processing")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k for NDCG calculation")
    parser.add_argument("--output-dir", type=str, default="data/eval_results/table4_lens",
                       help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--matching-radius", type=float, default=1.0,
                       help="Matching radius in arcseconds")
    parser.add_argument("--save-top-m-images", type=int, default=20,
                       help="Number of top images to save (default: 20)")
    parser.add_argument("--lens-index-path", type=str, default="data/evals/table4_lens/lens_fits_index.csv",
                       help="Path to pre-computed lens FITS index CSV file")
    parser.add_argument("--skip-images", action="store_true",
                       help="Skip saving images (compute metrics only)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run evaluation
    results = evaluate_table4_lens(
        hdf5_path=args.hdf5_path,
        masterlens_path=args.masterlens_path,
        hsc_lenses_path=args.hsc_lenses_path,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        k=args.k,
        output_dir=args.output_dir,
        logger=logger,
        matching_radius=args.matching_radius,
        save_top_m_images=args.save_top_m_images,
        lens_index_path=args.lens_index_path,
        skip_images=args.skip_images
    )
    
    print(f"\nEvaluation complete!")
    print(f"Lens NDCG@{args.k}: {results['lens_ndcg@10']:.4f}")


if __name__ == "__main__":
    main()