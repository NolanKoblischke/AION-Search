"""
Generate AION embeddings for Legacy and HSC images in lens catalog.
Produces two separate files:
- lens_aion_embeddings_legacy.hdf5
- lens_aion_embeddings_hsc.hdf5
Each with columns: object_id, ra, dec, aion_embedding_mean, aion_embedding_full, eval_grade
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse
from dotenv import load_dotenv
import fitsio

load_dotenv()

# Import the general embedding generation function
from src.evals.utils import generate_aion_embeddings, setup_logging


def main():
    """Main function to generate AION embeddings for lens Legacy and HSC images."""
    parser = argparse.ArgumentParser(description='Generate AION embeddings for lens Legacy and HSC images')
    parser.add_argument('--input', type=str, 
                       default='data/lens_image_catalog_part_000.fits',
                       help='Input FITS file with lens data')
    parser.add_argument('--output-legacy', type=str, 
                       default='data/evals/lens/lens_aion_embeddings_legacy.hdf5',
                       help='Output HDF5 file for Legacy AION embeddings')
    parser.add_argument('--output-hsc', type=str, 
                       default='data/evals/lens/lens_aion_embeddings_hsc.hdf5',
                       help='Output HDF5 file for HSC AION embeddings')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--model', type=str, default='polymathic-ai/aion-base',
                       help='AION model to use')
    parser.add_argument('--lens-csv', type=str,
                       default='data/evals/lens/lens_eval_objects.csv',
                       help='CSV file with evaluation objects to process (required)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path("data/evals/lens")
    output_dir.mkdir(exist_ok=True, parents=True)
    Path("data/logs").mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("data/logs") / f"lens_aion_embeddings_{timestamp}.log"
    logger = setup_logging("INFO", str(log_file))
    
    logger.info("="*60)
    logger.info("Lens Legacy and HSC AION Embedding Generation")
    logger.info("="*60)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output Legacy file: {args.output_legacy}")
    logger.info(f"Output HSC file: {args.output_hsc}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Lens CSV: {args.lens_csv}")
    
    # Load evaluation CSV (required)
    if not Path(args.lens_csv).exists():
        raise ValueError(f"Evaluation CSV file not found: {args.lens_csv}")
    
    logger.info("Loading evaluation CSV...")
    eval_df = pd.read_csv(args.lens_csv)
    logger.info(f"Loaded {len(eval_df)} objects from evaluation CSV")
    
    # Convert NaN grades to 'N' for non-lens
    eval_df['lensgrade'] = eval_df['lensgrade'].fillna('N')
    
    # Sort by fits_index for efficient sequential reading
    eval_df = eval_df.sort_values('fits_index')
    
    # Print statistics
    grade_counts = eval_df['lensgrade'].value_counts().sort_index()
    logger.info("Grade distribution:")
    for grade, count in grade_counts.items():
        logger.info(f"  Grade {grade}: {count} ({count/len(eval_df)*100:.1f}%)")
    
    # Prepare data for embedding generation
    object_ids = eval_df['object_id'].values
    ra_values = eval_df['ra'].values
    dec_values = eval_df['dec'].values
    lens_grades = eval_df['lensgrade'].values
    fits_indices = eval_df['fits_index'].values
    
    logger.info(f"Will process {len(fits_indices)} objects from FITS file")
    
    # Read images from FITS file
    logger.info(f"Reading images from FITS file: {args.input}")
    with fitsio.FITS(args.input) as fits_f:
        # Check available columns
        col_names = fits_f[1].get_colnames()
        has_legacy = 'legacysurvey_image' in col_names
        has_hsc = 'hsc_image' in col_names
        
        if not has_legacy and not has_hsc:
            raise ValueError("No 'legacysurvey_image' or 'hsc_image' columns found in FITS file")
        
        # Generate Legacy embeddings if available
        if has_legacy:
            logger.info("\n" + "="*60)
            logger.info("Reading Legacy Survey images...")
            legacy_images = fits_f[1].read_column('legacysurvey_image', rows=fits_indices)
            logger.info(f"Read {len(legacy_images)} Legacy images")
            
            logger.info("Generating Legacy Survey embeddings...")
            logger.info("="*60)
            
            legacy_output = generate_aion_embeddings(
                object_ids=object_ids,
                ra_values=ra_values,
                dec_values=dec_values,
                eval_grades=lens_grades,
                images=legacy_images,
                survey_name='Legacy',
                aion_model_name=args.model,
                output_file_path=args.output_legacy,
                batch_size=args.batch_size,
                device=args.device,
                logger=logger
            )
        
        # Generate HSC embeddings if available
        if has_hsc:
            logger.info("\n" + "="*60)
            logger.info("Reading HSC images...")
            hsc_images = fits_f[1].read_column('hsc_image', rows=fits_indices)
            logger.info(f"Read {len(hsc_images)} HSC images")
            
            logger.info("Generating HSC Survey embeddings...")
            logger.info("="*60)
            
            hsc_output = generate_aion_embeddings(
                object_ids=object_ids,
                ra_values=ra_values,
                dec_values=dec_values,
                eval_grades=lens_grades,
                images=hsc_images,
                survey_name='HSC',
                aion_model_name=args.model,
                output_file_path=args.output_hsc,
                batch_size=args.batch_size,
                device=args.device,
                logger=logger
            )
    
    logger.info("\n" + "="*60)
    logger.info("All embeddings generated successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()