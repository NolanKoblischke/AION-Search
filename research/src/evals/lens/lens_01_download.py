"""
Extract lens matches from lens_image_catalog_part_000.fits and save to CSV.
Matches galaxies with masterlens and hsc_lenses catalogues.
"""
import numpy as np
import pandas as pd
import fitsio
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy import units as u
from pathlib import Path
import logging
from datetime import datetime
from astropy.table import vstack


def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'lens_01_download_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger


def load_all_lens_catalogs():
    """Load all lens catalogs including lenscat and stein catalogs"""
    logger = logging.getLogger(__name__)
    logger.info("Loading all lens catalogs for cross-matching...")
    
    all_lens_coords = []
    
    # Load main lens databases
    try:
        masterlens = Table.read('data/evals/lens/catalogs/masterlens.csv')
        all_lens_coords.append(SkyCoord(ra=masterlens['ra']*u.degree, dec=masterlens['dec']*u.degree))
        logger.info(f"  Loaded {len(masterlens)} entries from masterlens.csv")
    except Exception as e:
        logger.warning(f"  Could not load masterlens.csv: {e}")
    
    try:
        hsc = Table.read('data/evals/lens/catalogs/hsc_lenses.csv')
        all_lens_coords.append(SkyCoord(ra=hsc['ra']*u.degree, dec=hsc['dec']*u.degree))
        logger.info(f"  Loaded {len(hsc)} entries from hsc_lenses.csv")
    except Exception as e:
        logger.warning(f"  Could not load hsc_lenses.csv: {e}")
    
    # Load lenscat.csv
    try:
        lenscat = Table.read('data/evals/lens/catalogs/lenscat.csv')
        all_lens_coords.append(SkyCoord(ra=lenscat['RA [deg]']*u.degree, dec=lenscat['DEC [deg]']*u.degree))
        logger.info(f"  Loaded {len(lenscat)} entries from lenscat.csv")
    except Exception as e:
        logger.warning(f"  Could not load lenscat.csv: {e}")
    
    # Load stein training lenses
    try:
        stein_training = Table.read('data/evals/lens/catalogs/stein_training_lenses_legacy.tsv', format='ascii.tab')
        all_lens_coords.append(SkyCoord(ra=stein_training['ra']*u.degree, dec=stein_training['dec']*u.degree))
        logger.info(f"  Loaded {len(stein_training)} entries from stein_training_lenses_legacy.tsv")
    except Exception as e:
        logger.warning(f"  Could not load stein_training_lenses_legacy.tsv: {e}")
    
    # Load stein new lenses
    try:
        stein_new = Table.read('data/evals/lens/catalogs/stein_new_lenses_legacy.tsv', format='ascii.tab')
        all_lens_coords.append(SkyCoord(ra=stein_new['ra']*u.degree, dec=stein_new['dec']*u.degree))
        logger.info(f"  Loaded {len(stein_new)} entries from stein_new_lenses_legacy.tsv")
    except Exception as e:
        logger.warning(f"  Could not load stein_new_lenses_legacy.tsv: {e}")
    
    # Concatenate all coordinates
    if all_lens_coords:
        combined_coords = coordinates.concatenate(all_lens_coords)
        logger.info(f"Total lens entries from all catalogs: {len(combined_coords)}")
        return combined_coords
    else:
        logger.error("No lens catalogs could be loaded!")
        return None


def merge_lens_catalogues():
    """Merge masterlens and HSC lens catalogues, keeping only ra, dec, grade"""
    logger = logging.getLogger(__name__)
    logger.info("Loading lens catalogues...")
    
    # Load lens databases with updated paths
    lenses_file = 'data/evals/lens/catalogs/masterlens.csv'
    masterlens_lenses = Table.read(lenses_file)
    logger.info(f"Loaded {len(masterlens_lenses)} masterlens entries")
    
    lenses_file = 'data/evals/lens/catalogs/hsc_lenses.csv'
    hsc_lenses = Table.read(lenses_file)
    logger.info(f"Loaded {len(hsc_lenses)} HSC lens entries")
    
    # Create sky coordinates and drop overlap
    hsc_coords = SkyCoord(
        ra=hsc_lenses['ra']*u.degree, dec=hsc_lenses['dec']*u.degree
    )
    masterlens_coords = SkyCoord(
        ra=masterlens_lenses['ra']*u.degree, dec=masterlens_lenses['dec']*u.degree
    )
    
    logger.info("Finding overlaps between catalogues...")
    idx_hsc, sep2d_hsc, _ = coordinates.match_coordinates_sky(
        masterlens_coords, hsc_coords
    )
    
    # Remove overlapping entries from masterlens
    overlap_mask = sep2d_hsc.arcsec < 1.0
    logger.info(f"Found {np.sum(overlap_mask)} overlapping entries, removing from masterlens")
    masterlens_lenses = masterlens_lenses[~overlap_mask]
    
    # Keep only needed columns and rename grade column for consistency
    masterlens_subset = masterlens_lenses['ra', 'dec', 'lensgrade']
    masterlens_subset['lensgrade'].name = 'grade'
    
    hsc_subset = hsc_lenses['ra', 'dec', 'grade']
    
    # Merge catalogues
    logger.info("Merging catalogues...")
    merged_lenses = vstack([masterlens_subset, hsc_subset])
    logger.info(f"Total merged lens entries: {len(merged_lenses)}")
    
    return merged_lenses


def normalize_lensgrade(grade):
    """
    Normalize lens grade to ensure consistent formatting.
    - "A" -> "A"
    - "B" -> "B" 
    - "C" -> "C"
    - "" -> "C" (empty grade treated as unclassified lens, only 4 out of 249 cases)
    """
    # Convert to string and strip quotes/whitespace
    grade_str = str(grade).strip().strip('"').strip("'")
    
    # Handle empty grade - treat as unclassified lens (C)
    if grade_str == "":
        return "C"
    
    # Return uppercase single letter
    return grade_str.upper()


def extract_lens_matches(n_lenses=200, n_non_lenses=20000, random_seed=42):
    """Extract lens matches and create evaluation dataset with fixed numbers of lenses and non-lenses.
    
    Args:
        n_lenses: Number of lenses to include (default 200)
        n_non_lenses: Number of non-lenses to include (default 20,000)
        random_seed: Random seed for reproducible sampling
    """
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("Lens Evaluation Dataset Creation")
    logger.info("="*60)
    
    # Configuration
    fits_file = 'data/lens_image_catalog_part_000.fits'
    output_dir = Path('data/evals/lens')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'lens_eval_objects.csv'
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load lens catalogues
    lens_catalogue = merge_lens_catalogues()
    lens_coords = SkyCoord(ra=lens_catalogue['ra']*u.degree, dec=lens_catalogue['dec']*u.degree)
    
    logger.info(f"\nProcessing {fits_file}")
    logger.info(f"Target dataset: {n_lenses} lenses, {n_non_lenses} non-lenses (random seed: {random_seed})")
    
    # Read FITS file and find matches
    with fitsio.FITS(fits_file) as f:
        nrows = f[1].get_nrows()
        logger.info(f"Number of rows in FITS file: {nrows}")
        
        # Process in batches to manage memory
        batch_size = 5000
        lens_objects = []  # Will store lens matches
        all_objects = []   # Will store all objects with their indices
        
        logger.info(f"Processing {nrows} rows in batches of {batch_size}...")
        
        for start_row in range(0, nrows, batch_size):
            end_row = min(start_row + batch_size, nrows)
            row_indices = list(range(start_row, end_row))
            
            logger.info(f"  Processing rows {start_row:,} to {end_row:,}...")
            
            # Read batch data
            object_ids_batch = f[1].read_column('object_id', rows=row_indices)
            ras_batch = f[1].read_column('ra', rows=row_indices)
            decs_batch = f[1].read_column('dec', rows=row_indices)
            
            # Store all objects with their FITS indices
            for idx, (fits_idx, obj_id, ra, dec) in enumerate(zip(row_indices, object_ids_batch, ras_batch, decs_batch)):
                all_objects.append({
                    'fits_index': fits_idx,
                    'object_id': obj_id,
                    'ra': ra,
                    'dec': dec,
                    'lensgrade': None  # Default for non-lenses
                })
            
            # Create coordinates for this batch
            fits_coords_batch = SkyCoord(ra=ras_batch*u.degree, dec=decs_batch*u.degree)
            
            # Find matches within 1 arcsecond for this batch
            idx_fits_batch, idx_lens_batch, d2d_batch, _ = lens_coords.search_around_sky(
                fits_coords_batch, 1*u.arcsec
            )
            
            # Process matches from this batch
            for i, j, _ in zip(idx_fits_batch, idx_lens_batch, d2d_batch):
                grade = lens_catalogue['grade'][j]
                
                # Convert grade to string if it's bytes
                if isinstance(grade, bytes):
                    grade = grade.decode('utf-8')
                
                # Normalize the grade
                normalized_grade = normalize_lensgrade(grade)
                
                lens_objects.append({
                    'fits_index': row_indices[i],
                    'object_id': object_ids_batch[i],
                    'ra': ras_batch[i],
                    'dec': decs_batch[i],
                    'lensgrade': normalized_grade
                })
        
        logger.info(f"Found {len(lens_objects)} lens matches out of {nrows} total objects")
        
        # Create sets for efficient lookup
        lens_indices = {obj['fits_index'] for obj in lens_objects}
        
        # Update all_objects with lens grades
        all_objects_dict = {obj['fits_index']: obj for obj in all_objects}
        for lens_obj in lens_objects:
            all_objects_dict[lens_obj['fits_index']]['lensgrade'] = lens_obj['lensgrade']
        
        # Identify non-lens objects
        non_lens_indices = [idx for idx in range(nrows) if idx not in lens_indices]
        total_non_lenses = len(non_lens_indices)
        logger.info(f"Total non-lens objects: {total_non_lenses}")
        
        # Sample lenses if we have more than requested
        if len(lens_objects) > n_lenses:
            logger.info(f"Sampling {n_lenses} lenses from {len(lens_objects)} available")
            sampled_lens_indices = np.random.choice(len(lens_objects), size=n_lenses, replace=False)
            sampled_lens_objects = [lens_objects[i] for i in sampled_lens_indices]
        else:
            logger.info(f"Using all {len(lens_objects)} available lenses (less than requested {n_lenses})")
            sampled_lens_objects = lens_objects
        
        # Sample non-lenses (oversample to account for filtering)
        # We'll sample 50% more to ensure we have enough after filtering
        oversample_factor = 1.5
        n_non_lens_oversample = min(int(n_non_lenses * oversample_factor), total_non_lenses)
        if n_non_lens_oversample < n_non_lenses:
            logger.warning(f"Only {total_non_lenses} non-lenses available for oversampling")
        sampled_non_lens_indices = np.random.choice(non_lens_indices, size=n_non_lens_oversample, replace=False)
        logger.info(f"Initially sampling {n_non_lens_oversample} non-lens objects (before filtering)")
        
        # Load all lens catalogs for filtering
        logger.info("\nLoading all lens catalogs for non-lens filtering...")
        all_lens_coords = load_all_lens_catalogs()
        
        # Filter non-lenses that are too close to known lenses
        logger.info("\nFiltering non-lenses within 5 arcsec of known lenses...")
        filtered_non_lens_indices = []
        removed_count = 0
        
        for idx in sampled_non_lens_indices:
            obj = all_objects_dict[idx]
            obj_coord = SkyCoord(ra=obj['ra']*u.degree, dec=obj['dec']*u.degree)
            
            # Check if this object is within 5 arcsec of any known lens
            if all_lens_coords is not None:
                idx_lens, d2d, _ = obj_coord.match_to_catalog_sky(all_lens_coords)
                if d2d.arcsec < 5.0:
                    removed_count += 1
                    continue  # Skip this object
            
            filtered_non_lens_indices.append(idx)
            
            # Stop if we have enough non-lenses
            if len(filtered_non_lens_indices) >= n_non_lenses:
                break
        
        logger.info(f"Removed {removed_count} non-lenses that were within 5 arcsec of known lenses")
        logger.info(f"Final non-lens sample size: {len(filtered_non_lens_indices)}")
        
        if len(filtered_non_lens_indices) < n_non_lenses:
            logger.warning(f"Could only obtain {len(filtered_non_lens_indices)} non-lenses after filtering (target was {n_non_lenses})")
        
        # Create final evaluation dataset
        eval_objects = []
        
        # Add sampled lenses
        for lens_obj in sampled_lens_objects:
            eval_objects.append(lens_obj)
        
        # Add filtered non-lenses
        for idx in filtered_non_lens_indices:
            eval_objects.append(all_objects_dict[idx])
        
        # Convert to dataframe and sort by FITS index for efficient reading
        df = pd.DataFrame(eval_objects)
        df = df.sort_values('fits_index')
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logger.info(f"\nSaved {len(df)} objects to {output_csv}")
        
        # Print statistics
        logger.info("\nEvaluation dataset statistics:")
        logger.info(f"Total objects: {len(df)}")
        
        # Grade distribution
        lens_df = df[df['lensgrade'].notna()]
        logger.info(f"\nLens objects: {len(lens_df)}")
        grade_counts = lens_df['lensgrade'].value_counts().sort_index()
        for grade, count in grade_counts.items():
            logger.info(f"  Grade {grade}: {count} ({count/len(lens_df)*100:.1f}%)")
        
        non_lens_df = df[df['lensgrade'].isna()]
        logger.info(f"\nNon-lens objects: {len(non_lens_df)}")
        logger.info(f"  Sampled from {total_non_lenses} total non-lenses")
        logger.info(f"  {removed_count} objects removed for being within 5 arcsec of known lenses")
        
        # Also save a summary
        summary_file = output_dir / 'lens_eval_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Lens Evaluation Dataset Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"FITS file: {fits_file}\n")
            f.write(f"Total objects in FITS: {nrows}\n")
            f.write(f"Random seed: {random_seed}\n")
            f.write(f"Target lenses: {n_lenses}\n")
            f.write(f"Target non-lenses: {n_non_lenses}\n\n")
            
            f.write(f"Evaluation dataset:\n")
            f.write(f"  Total objects: {len(df)}\n")
            f.write(f"  Lens objects: {len(lens_df)}\n")
            for grade, count in grade_counts.items():
                f.write(f"    Grade {grade}: {count} ({count/len(lens_df)*100:.1f}%)\n")
            f.write(f"  Non-lens objects: {len(non_lens_df)} (sampled from {total_non_lenses})\n")
            f.write(f"    {removed_count} objects removed for being within 5 arcsec of known lenses\n")
            f.write(f"\nReduction from full dataset: {(1 - len(df)/nrows)*100:.1f}%\n")
        
        logger.info(f"Saved summary to {summary_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation dataset creation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    extract_lens_matches()