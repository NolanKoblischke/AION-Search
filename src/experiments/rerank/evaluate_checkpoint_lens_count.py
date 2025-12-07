"""
Evaluate a checkpoint on lens_hsc and count the number of true lenses in top k results.
"""

import argparse
from pathlib import Path
import logging
import torch
import numpy as np
import h5py
from typing import Dict, List, Tuple
from datetime import datetime

from src.clip.models.clip_model import GalaxyClipModel
from src.evals.eval_utils import (
    EVAL_CONFIGS, 
    load_aion_embeddings,
    generate_query_embedding,
    setup_logging
)


def count_lenses_in_top_k(
    eval_grades: List[str],
    top_indices: np.ndarray,
    k_values: List[int]
) -> Dict[str, int]:
    """
    Count the number of true lenses (grades A, B, C) in top k results.
    
    Args:
        eval_grades: List of evaluation grades for all objects
        top_indices: Indices of objects sorted by similarity (descending)
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary mapping 'lenses@k' to count
    """
    results = {}
    
    for k in k_values:
        # Get grades for top k objects
        top_k_grades = [eval_grades[idx] for idx in top_indices[:k]]
        
        # Count true lenses (grades A, B, or C)
        lens_count = sum(1 for grade in top_k_grades if grade in ['A', 'B', 'C'])
        
        results[f'lenses@{k}'] = lens_count
    
    return results


def save_top_k_to_hdf5(
    output_path: str,
    top_indices: np.ndarray,
    similarities: np.ndarray,
    k: int,
    source_hdf5_path: str,
    checkpoint_path: str,
    query_text: str,
    logger: logging.Logger = None,
    lens_csv_path: str = "data/evals/lens/lens_eval_objects.csv",
    fits_path: str = "data/lens_image_catalog_part_000.fits"
):
    """
    Save top k results to HDF5 file with mean embeddings and HSC images.
    
    Args:
        output_path: Path for output HDF5 file
        top_indices: Indices of objects sorted by similarity
        similarities: Similarity scores for all objects
        k: Number of top results to save
        source_hdf5_path: Path to source HDF5 file
        checkpoint_path: Path to checkpoint used
        query_text: Query text used
        logger: Logger instance
        lens_csv_path: Path to lens CSV for FITS indexing
        fits_path: Path to FITS file containing images
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Limit k to available objects
    k = min(k, len(top_indices))
    top_k_indices = top_indices[:k]
    
    logger.info(f"Saving top {k} results to {output_path}")
    
    # Convert indices to numpy array for efficient indexing
    top_k_indices_array = np.array(top_k_indices)
    
    # Load lens CSV for FITS indexing
    import pandas as pd
    import fitsio
    
    logger.info(f"Loading lens CSV for FITS indexing: {lens_csv_path}")
    lens_df = pd.read_csv(lens_csv_path)
    
    # Create mapping from object_id to fits_index
    oid_to_fits_idx = {int(row['object_id']): row['fits_index'] 
                      for _, row in lens_df.iterrows()}
    
    # Load data efficiently from source
    with h5py.File(source_hdf5_path, 'r') as source_f:
        # Get dimensions
        embedding_dim = source_f['aion_embedding_mean'].shape[1]
        
        # Get attributes
        source_attrs = dict(source_f.attrs)
        
        # Create output file
        with h5py.File(output_path, 'w') as out_f:
            # Create datasets
            out_f.create_dataset('object_id', 
                                shape=(k,),
                                dtype=h5py.special_dtype(vlen=str))
            
            out_f.create_dataset('ra', shape=(k,), dtype=np.float64)
            out_f.create_dataset('dec', shape=(k,), dtype=np.float64)
            
            out_f.create_dataset('eval_grade',
                                shape=(k,),
                                dtype=h5py.special_dtype(vlen=str))
            
            out_f.create_dataset('aion_embedding_mean',
                                shape=(k, embedding_dim),
                                dtype=np.float32)
            
            out_f.create_dataset('similarity',
                                shape=(k,),
                                dtype=np.float32)
            
            # Get all object IDs first to determine FITS indices
            all_object_ids = source_f['object_id'][:]
            
            # Map top k indices to object IDs and then to FITS indices
            fits_indices_to_read = []
            oid_to_top_idx = {}
            
            for i, idx in enumerate(top_k_indices_array):
                obj_id = all_object_ids[idx]
                obj_id_str = obj_id if isinstance(obj_id, str) else obj_id.decode('utf-8')
                obj_id_int = int(obj_id_str)
                
                if obj_id_int in oid_to_fits_idx:
                    fits_indices_to_read.append(oid_to_fits_idx[obj_id_int])
                    oid_to_top_idx[obj_id_int] = i
                else:
                    logger.warning(f"Object ID {obj_id_int} not found in lens CSV")
            
            # Read HSC images from FITS file
            logger.info(f"Reading {len(fits_indices_to_read)} HSC images from FITS file...")
            with fitsio.FITS(fits_path) as fits_f:
                hsc_images = fits_f[1].read_column('hsc_image', rows=fits_indices_to_read)
                
                # Get image shape from first image
                image_shape = hsc_images[0].shape
                logger.info(f"HSC image shape: {image_shape}")
                
                # Create HSC image dataset
                out_f.create_dataset('hsc_image',
                                    shape=(k,) + image_shape,
                                    dtype=hsc_images.dtype)
            
            # Process in chunks for memory efficiency
            chunk_size = min(1000, k)  # Process 1000 items at a time
            
            for start_idx in range(0, k, chunk_size):
                end_idx = min(start_idx + chunk_size, k)
                chunk_indices = top_k_indices_array[start_idx:end_idx]
                chunk_range = slice(start_idx, end_idx)
                
                # HDF5 requires sorted indices for fancy indexing
                # Sort indices and keep track of original order
                sorted_idx = np.argsort(chunk_indices)
                sorted_chunk_indices = chunk_indices[sorted_idx]
                
                # Load chunk data with sorted indices
                object_ids_chunk = source_f['object_id'][sorted_chunk_indices]
                ra_chunk = source_f['ra'][sorted_chunk_indices]
                dec_chunk = source_f['dec'][sorted_chunk_indices]
                eval_grades_chunk = source_f['eval_grade'][sorted_chunk_indices]
                embeddings_chunk = source_f['aion_embedding_mean'][sorted_chunk_indices]
                
                # Reorder back to original order
                unsort_idx = np.argsort(sorted_idx)
                object_ids_chunk = object_ids_chunk[unsort_idx]
                ra_chunk = ra_chunk[unsort_idx]
                dec_chunk = dec_chunk[unsort_idx]
                eval_grades_chunk = eval_grades_chunk[unsort_idx]
                embeddings_chunk = embeddings_chunk[unsort_idx]
                
                # Save chunk data
                # Handle string decoding for object_ids and eval_grades
                for i, local_idx in enumerate(range(start_idx, end_idx)):
                    obj_id = object_ids_chunk[i]
                    obj_id_str = obj_id if isinstance(obj_id, str) else obj_id.decode('utf-8')
                    out_f['object_id'][local_idx] = obj_id_str
                    
                    grade = eval_grades_chunk[i]
                    out_f['eval_grade'][local_idx] = grade if isinstance(grade, str) else grade.decode('utf-8')
                    
                    # Save HSC image if available
                    obj_id_int = int(obj_id_str)
                    if obj_id_int in oid_to_top_idx:
                        image_idx = fits_indices_to_read.index(oid_to_fits_idx[obj_id_int])
                        out_f['hsc_image'][local_idx] = hsc_images[image_idx]
                
                # Save numeric data directly
                out_f['ra'][chunk_range] = ra_chunk
                out_f['dec'][chunk_range] = dec_chunk
                out_f['aion_embedding_mean'][chunk_range] = embeddings_chunk
                out_f['similarity'][chunk_range] = similarities[chunk_indices]
                
                logger.info(f"  Processed {end_idx}/{k} objects")
            
            # Copy original attributes
            for key, value in source_attrs.items():
                out_f.attrs[key] = value
            
            # Add new attributes
            out_f.attrs['retrieval_date'] = datetime.now().isoformat()
            out_f.attrs['checkpoint_path'] = str(checkpoint_path)
            out_f.attrs['query_text'] = query_text
            out_f.attrs['k_value'] = k
            out_f.attrs['n_rows'] = k  # Update n_rows to reflect filtered data
            out_f.attrs['note'] = 'Contains mean embeddings and HSC images'
            out_f.attrs['survey'] = 'HSC'
    
    logger.info(f"Successfully saved {k} objects to {output_path}")


def evaluate_checkpoint_lens_count(
    checkpoint_path: str,
    eval_name: str = 'lens_hsc',
    device: str = 'cuda',
    batch_size: int = 512,
    k_values: List[int] = [10, 100, 1000],
    logger: logging.Logger = None,
    save_k: int = None,
    output_file: str = None
) -> Dict:
    """
    Evaluate a checkpoint and count lenses in top k results.
    Saves HSC images along with embeddings when save_k is specified.
    
    Args:
        checkpoint_path: Path to checkpoint file
        eval_name: Evaluation name (default: lens_hsc)
        device: Device to use
        batch_size: Batch size for processing
        k_values: List of k values to evaluate
        logger: Logger instance
        save_k: Number of top results to save with HSC images (None = don't save)
        output_file: Output HDF5 file path (auto-generated if None)
    
    Returns:
        Dictionary with evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    logger.info(f"Model config: {model_config}")
    
    # Create model
    model = GalaxyClipModel(
        image_input_dim=model_config['image_input_dim'],
        text_input_dim=model_config['text_input_dim'],
        embedding_dim=model_config['embedding_dim'],
        use_mean_embeddings=model_config['use_mean_embeddings']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")
    
    # Get evaluation configuration
    eval_config = EVAL_CONFIGS[eval_name]
    
    # Load evaluation data
    logger.info(f"Loading {eval_name} evaluation data...")
    object_ids, aion_embeddings, eval_grades = load_aion_embeddings(
        eval_config.embeddings_path,
        use_mean=model_config['use_mean_embeddings'],
        aion_model='aion-base'  # Assuming base model
    )
    
    logger.info(f"Loaded {len(object_ids)} objects")
    
    # Count total lenses in dataset
    total_lenses = sum(1 for grade in eval_grades if grade in ['A', 'B', 'C'])
    logger.info(f"Total lenses in dataset: {total_lenses}")
    
    # Generate query embedding
    cache_dir = Path(eval_config.embeddings_path).parent / "query_embeddings_cache"
    logger.info(f"Generating query embedding for: '{eval_config.query_text}'")
    query_embedding = generate_query_embedding(eval_config.query_text, cache_dir=cache_dir)
    
    # Project query through model
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.text_projector(query_tensor)
    query_features = query_features.cpu().numpy()
    
    # Project AION embeddings through model in batches
    logger.info("Projecting image embeddings...")
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(aion_embeddings), batch_size):
            batch = aion_embeddings[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            features = model.image_projector(batch_tensor)
            all_features.append(features.cpu().numpy())
    
    image_features = np.vstack(all_features)
    
    # Compute similarities and rank
    logger.info("Computing similarities and ranking...")
    similarities = image_features @ query_features.T
    similarities = similarities.squeeze()
    top_indices = np.argsort(similarities)[::-1]
    
    # Count lenses in top k
    lens_counts = count_lenses_in_top_k(eval_grades, top_indices, k_values)
    
    # Also compute recall for each k
    recall_results = {}
    for k in k_values:
        recall = lens_counts[f'lenses@{k}'] / total_lenses if total_lenses > 0 else 0.0
        recall_results[f'recall@{k}'] = recall
    
    # Combine results
    results = {
        'total_objects': len(object_ids),
        'total_lenses': total_lenses,
        **lens_counts,
        **recall_results
    }
    
    # Print detailed results for top objects
    logger.info("\nTop 20 retrieved objects:")
    logger.info("Rank | Object ID | Grade | Similarity")
    logger.info("-" * 50)
    for i in range(min(20, len(top_indices))):
        idx = top_indices[i]
        grade = eval_grades[idx]
        is_lens = " (LENS)" if grade in ['A', 'B', 'C'] else ""
        logger.info(f"{i+1:4d} | {object_ids[idx]:9s} | {grade:5s} | {similarities[idx]:10.6f}{is_lens}")
    
    # Save top k results if requested
    if save_k is not None:
        if output_file is None:
            # Auto-generate output filename
            checkpoint_name = Path(checkpoint_path).stem
            output_file = f"{checkpoint_name}_top{save_k}_{eval_name}.hdf5"
        
        save_top_k_to_hdf5(
            output_path=output_file,
            top_indices=top_indices,
            similarities=similarities,
            k=save_k,
            source_hdf5_path=eval_config.embeddings_path,
            checkpoint_path=checkpoint_path,
            query_text=eval_config.query_text,
            logger=logger
        )
        
        results['saved_file'] = output_file
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and count lenses in top k")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--eval-name", type=str, default="lens_hsc",
                       choices=list(EVAL_CONFIGS.keys()),
                       help="Evaluation dataset to use")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, auto-detect if None)")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for processing")
    parser.add_argument("--k-values", type=int, nargs='+', default=[10, 100, 1000],
                       help="k values for top-k evaluation")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Save options
    parser.add_argument("--save-k", type=int, default=None,
                       help="Number of top results to save to HDF5 (omit to skip saving)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output HDF5 file path (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {args.device}")
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Run evaluation
    results = evaluate_checkpoint_lens_count(
        checkpoint_path=str(checkpoint_path),
        eval_name=args.eval_name,
        device=args.device,
        batch_size=args.batch_size,
        k_values=args.k_values,
        logger=logger,
        save_k=args.save_k,
        output_file=args.output_file
    )
    
    # Print summary results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {args.eval_name}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Total objects in dataset: {results['total_objects']:,}")
    print(f"Total lenses in dataset: {results['total_lenses']:,}")
    print(f"{'-'*60}")
    
    for k in args.k_values:
        lens_count = results[f'lenses@{k}']
        recall = results[f'recall@{k}']
        print(f"Top-{k:4d}: {lens_count:4d} lenses found (recall: {recall:.1%})")
    
    if 'saved_file' in results:
        print(f"{'-'*60}")
        print(f"Results saved to: {results['saved_file']}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()