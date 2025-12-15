import h5py
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

TARGET_HSC_GALAXIES = 100_000
TARGET_LEGACY_GALAXIES = 100_000
MAX_GALAXIES_PER_HEALPIX = 10_000 # Specified so that we spread our training set over many healpixels
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

HSC_BASE_PATH = Path("/AstroPile_v1/hsc/pdr3_wide_21")
LEGACY_BASE_PATH = Path("/workspace/data/MMU/legacysurvey/dr10_south_21")

# Set up logging - will be configured in main()
logger = logging.getLogger(__name__)

STATS = {
    'hsc': {
        'healpix_counts': {},
        'total_processed': 0,
        'total_cut_mag': 0,
        'total_collected': 0,
        'total_excluded': 0,
        'files_processed': 0
    },
    'legacy': {
        'healpix_counts': {},
        'total_processed': 0,
        'total_cut_mag': 0,
        'total_collected': 0,
        'total_excluded': 0,
        'files_processed': 0
    }
}




def update_log_summary():
    """Log current statistics summary"""
    logger.info("\n" + "="*60)
    logger.info(f"STATISTICS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # HSC Summary
    hsc_stats = STATS['hsc']
    logger.info("\nHSC STATISTICS:")
    logger.info(f"  Total galaxies processed: {hsc_stats['total_processed']:,}")
    logger.info(f"  Cut by r_cmodel_mag >= 19: {hsc_stats['total_cut_mag']:,}")
    logger.info(f"  Total excluded (proximity match): {hsc_stats['total_excluded']:,}")
    logger.info(f"  Total collected: {hsc_stats['total_collected']:,}")
    logger.info(f"  Files processed: {hsc_stats['files_processed']:,}")
    logger.info(f"  Healpix distribution:")
    for healpix, count in sorted(hsc_stats['healpix_counts'].items()):
        logger.info(f"    healpix {healpix}: {count:,} galaxies")
    
    # Legacy Summary
    legacy_stats = STATS['legacy']
    logger.info("\nLEGACY STATISTICS:")
    logger.info(f"  Total galaxies processed: {legacy_stats['total_processed']:,}")
    logger.info(f"  Cut by r_mag >= 19: {legacy_stats['total_cut_mag']:,}")
    logger.info(f"  Total excluded (proximity match): {legacy_stats['total_excluded']:,}")
    logger.info(f"  Total collected: {legacy_stats['total_collected']:,}")
    logger.info(f"  Files processed: {legacy_stats['files_processed']:,}")
    logger.info(f"  Healpix distribution:")
    for healpix, count in sorted(legacy_stats['healpix_counts'].items()):
        logger.info(f"    healpix {healpix}: {count:,} galaxies")
    
    logger.info("\n" + "="*60 + "\n")


def compute_legacy_r_mag(flux_r):
    """Compute r-band magnitude from flux: 22.5 - 2.5 log10(FLUX_R)"""
    # Avoid log of zero or negative flux
    r_mag = 22.5 - 2.5 * np.log10(flux_r)
    return r_mag


def apply_cuts(f, survey):
    """Apply cuts to data based on survey type and return indices that pass cuts"""
    if survey == 'hsc':
        # HSC cuts
        r_cmodel_mag = f['r_cmodel_mag'][:]
        
        n_total = len(r_cmodel_mag)
        STATS['hsc']['total_processed'] += n_total
        
        # Only magnitude cut: r_cmodel_mag < 19 (removes dim galaxies)
        mask_mag = r_cmodel_mag < 19
        n_after_mag = np.sum(mask_mag)
        n_cut_mag = n_total - n_after_mag
        STATS['hsc']['total_cut_mag'] += n_cut_mag
        
        
        indices = np.where(mask_mag)[0]
        
        cut_info = (n_total, n_cut_mag)
        
    else:  # legacy
        # Legacy Survey cuts
        flux_r = f['FLUX_R'][:]
        
        n_total = len(flux_r)
        STATS['legacy']['total_processed'] += n_total
        
        r_mag = compute_legacy_r_mag(flux_r)
        
        # Only magnitude cut: r_mag < 19 (removes dim galaxies)
        mask_mag = r_mag < 19
        n_after_mag = np.sum(mask_mag)
        n_cut_mag = n_total - n_after_mag
        STATS['legacy']['total_cut_mag'] += n_cut_mag
        
        
        indices = np.where(mask_mag)[0]
        
        cut_info = (n_total, n_cut_mag)
    
    return indices, cut_info


def get_healpix_files(base_path):
    """Get all healpix HDF5 files from a survey directory"""
    healpix_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('healpix=')])
    hdf5_files = []
    
    for healpix_dir in healpix_dirs:
        # Find HDF5 files in this healpix directory
        files = list(healpix_dir.glob("*.hdf5")) #I believe just 001-of-001.hdf5, but maybe denser populated healpixes have more
        # Filter out lock files
        files = [f for f in files if not f.name.endswith('.lock')] # ignore 001-of-001.lock
        if files:
            hdf5_files.extend(files)
    
    return hdf5_files


def collect_galaxies(survey, target_count, exclusion_coords=None):
    """Collect galaxies from specified survey until we reach the target count"""
    survey_upper = survey.upper()
    survey_lower = survey.lower()
    
    logger.info("\n" + "="*60)
    logger.info(f"Collecting {survey_upper} galaxies")
    logger.info("="*60)
    
    if exclusion_coords:
        logger.info(f"Checking proximity to {len(exclusion_coords)} existing galaxies")
        logger.info(f"Exclusion radius: 10 arcsec (0.00278 degrees)")
    
    # Get base path and random seed based on survey
    base_path = HSC_BASE_PATH if survey_lower == 'hsc' else LEGACY_BASE_PATH
    random_seed = 42 if survey_lower == 'hsc' else 123
    
    # Get all available HDF5 files
    hdf5_files = get_healpix_files(base_path)
    logger.info(f"Found {len(hdf5_files)} HDF5 files in {survey_upper} directory")
    
    # Randomize the order of files
    # Use different seed when excluding to avoid same order as first run
    if exclusion_coords:
        # Different seeds for exclusion runs
        random_seed = 999 if survey_lower == 'hsc' else 888
        logger.info(f"Using different random seed ({random_seed}) for exclusion run")
    
    np.random.seed(random_seed)
    hdf5_files = np.random.permutation(hdf5_files)
    
    # Storage for collected data
    collected_data = {
        'object_id': [],
        'ra': [],
        'dec': [],
        'healpix': [],
        'image_array': []
    }
    
    total_collected = 0
    files_processed = 0
    healpix_counts = {}  # Track galaxies per healpix
    total_excluded = 0  # Track excluded galaxies
    
    for hdf5_file in hdf5_files:
        if total_collected >= target_count:
            break
            
        # Extract healpix number from directory name
        healpix_num = int(hdf5_file.parent.name.split('=')[1])
        
        # Check if this healpix already has enough galaxies
        current_healpix_count = healpix_counts.get(healpix_num, 0)
        if current_healpix_count >= MAX_GALAXIES_PER_HEALPIX:
            continue
        
        logger.info(f"\nProcessing {hdf5_file.name} (healpix={healpix_num}, current count: {current_healpix_count})")
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # Apply cuts to get valid indices
                valid_indices, cut_info = apply_cuts(f, survey_lower)
                n_valid = len(valid_indices)
                
                # Format log message based on survey
                n_total, n_cut_mag = cut_info
                logger.info(f"  Total galaxies: {n_total}, Cut by mag: {n_cut_mag}, Passed cuts: {n_valid}")
                
                if n_valid == 0:
                    continue
                
                # Determine how many to collect from this file
                n_needed_total = target_count - total_collected
                n_needed_healpix = MAX_GALAXIES_PER_HEALPIX - current_healpix_count
                n_to_collect = min(n_valid, n_needed_total, n_needed_healpix)
                
                # Randomly sample if we have more than needed
                if n_to_collect < n_valid:
                    np.random.seed(random_seed + files_processed)
                    selected_indices = np.random.choice(valid_indices, n_to_collect, replace=False)
                else:
                    selected_indices = valid_indices
                
                # Collect data
                logger.info(f"  Collecting up to {n_to_collect} galaxies...")
                actually_collected = 0
                for idx in selected_indices:
                    # Get galaxy coordinates
                    candidate_ra = float(f['ra'][idx])
                    candidate_dec = float(f['dec'][idx])
                    
                    # Check spatial proximity to existing galaxies
                    if exclusion_coords:
                        skip_galaxy = False
                        for existing_ra, existing_dec in exclusion_coords:
                            # Calculate angular separation in degrees
                            delta_ra = abs(candidate_ra - existing_ra)
                            if delta_ra > 180:  # Handle RA wraparound
                                delta_ra = 360 - delta_ra
                            
                            # Simple approximation for small angles
                            cos_dec = np.cos(np.radians(candidate_dec))
                            separation = np.sqrt((delta_ra * cos_dec)**2 + (candidate_dec - existing_dec)**2)
                            
                            if separation < 0.00278:  # 10 arcsec in degrees
                                skip_galaxy = True
                                total_excluded += 1
                                STATS[survey_lower]['total_excluded'] = total_excluded
                                break
                        
                        if skip_galaxy:
                            continue
                    
                    # Get object ID
                    obj_id = f['object_id'][idx]
                    
                    collected_data['object_id'].append(obj_id)
                    collected_data['ra'].append(candidate_ra)
                    collected_data['dec'].append(candidate_dec)
                    collected_data['healpix'].append(healpix_num)
                    collected_data['image_array'].append(f['image_array'][idx])
                    actually_collected += 1
                
                total_collected += actually_collected
                healpix_counts[healpix_num] = current_healpix_count + actually_collected
                STATS[survey_lower]['total_collected'] = total_collected
                STATS[survey_lower]['healpix_counts'][healpix_num] = healpix_counts[healpix_num]
                
                logger.info(f"  Actually collected: {actually_collected} (excluded: {total_excluded} total)")
                logger.info(f"  Total collected so far: {total_collected:,} (healpix {healpix_num}: {healpix_counts[healpix_num]})")
                
                # Update log summary every 10 files
                if files_processed % 10 == 0:
                    update_log_summary()
                
        except Exception as e:
            logger.info(f"  Error processing file: {e}")
            continue
        
        files_processed += 1
        STATS[survey_lower]['files_processed'] = files_processed
    
    logger.info(f"\n{survey_upper} collection complete: {total_collected:,} galaxies from {files_processed} files")
    if exclusion_coords:
        logger.info(f"Total excluded due to proximity to existing galaxies: {total_excluded:,}")
    update_log_summary()
    return collected_data


def save_collected_indices(hsc_data, legacy_data, output_file):
    """Save just the indices and metadata without images for redundancy"""
    logger.info(f"\nSaving collected indices to {output_file}")
    
    with h5py.File(output_file, 'w') as out_f:
        # Prepare HSC data
        n_hsc = len(hsc_data['object_id'])
        
        # Convert HSC object_ids to consistent format
        hsc_obj_ids = []
        for oid in hsc_data['object_id']:
            if isinstance(oid, bytes):
                hsc_obj_ids.append(oid.decode())
            else:
                hsc_obj_ids.append(str(oid))
        
        # Prepare Legacy data
        n_legacy = len(legacy_data['object_id'])
        
        # Convert Legacy object_ids to consistent format
        legacy_obj_ids = []
        for oid in legacy_data['object_id']:
            if isinstance(oid, bytes):
                legacy_obj_ids.append(oid.decode())
            else:
                legacy_obj_ids.append(str(oid))
        
        # Combine all metadata (no images)
        all_object_ids = np.concatenate([
            np.array(hsc_obj_ids, dtype='S50'),
            np.array(legacy_obj_ids, dtype='S50')
        ])
        all_ra = np.concatenate([
            np.array(hsc_data['ra'], dtype=np.float64),
            np.array(legacy_data['ra'], dtype=np.float64)
        ])
        all_dec = np.concatenate([
            np.array(hsc_data['dec'], dtype=np.float64),
            np.array(legacy_data['dec'], dtype=np.float64)
        ])
        all_healpix = np.concatenate([
            np.array(hsc_data['healpix'], dtype=np.int32),
            np.array(legacy_data['healpix'], dtype=np.int32)
        ])
        
        # Create group labels
        all_groups = np.array(['HSC'] * n_hsc + ['Legacy'] * n_legacy, dtype='S10')
        
        # Save metadata datasets
        out_f.create_dataset('object_id', data=all_object_ids)
        out_f.create_dataset('ra', data=all_ra)
        out_f.create_dataset('dec', data=all_dec)
        out_f.create_dataset('healpix', data=all_healpix)
        out_f.create_dataset('group', data=all_groups)
        
        # Add metadata
        out_f.attrs['created_date'] = datetime.now().isoformat()
        out_f.attrs['n_hsc_galaxies'] = n_hsc
        out_f.attrs['n_legacy_galaxies'] = n_legacy
        out_f.attrs['total_galaxies'] = n_hsc + n_legacy
        out_f.attrs['description'] = 'Galaxy indices and metadata only (no images)'
        
        logger.info(f"  Saved indices for {n_hsc:,} HSC and {n_legacy:,} Legacy galaxies")
        logger.info(f"  Total: {n_hsc + n_legacy:,} galaxies")


def save_combined_hdf5(hsc_data, legacy_data, output_file, batch_size=1000):
    """Save combined HSC and Legacy data to a single HDF5 file with memory-efficient batch writing"""
    logger.info(f"\nSaving combined data to {output_file}")
    logger.info(f"Using memory-efficient batch writing with batch size: {batch_size}")
    
    # Get dimensions from first images
    first_hsc_img = hsc_data['image_array'][0]
    _, h, w = first_hsc_img.shape
    
    with h5py.File(output_file, 'w') as out_f:
        # Prepare metadata (small memory footprint)
        logger.info("Preparing metadata...")
        n_hsc = len(hsc_data['object_id'])
        n_legacy = len(legacy_data['object_id'])
        n_total = n_hsc + n_legacy
        
        # Convert object_ids to consistent format
        hsc_obj_ids = []
        for oid in hsc_data['object_id']:
            if isinstance(oid, bytes):
                hsc_obj_ids.append(oid.decode())
            else:
                hsc_obj_ids.append(str(oid))
        
        legacy_obj_ids = []
        for oid in legacy_data['object_id']:
            if isinstance(oid, bytes):
                legacy_obj_ids.append(oid.decode())
            else:
                legacy_obj_ids.append(str(oid))
        
        # Combine all metadata
        all_object_ids = np.concatenate([
            np.array(hsc_obj_ids, dtype='S50'),
            np.array(legacy_obj_ids, dtype='S50')
        ])
        all_ra = np.concatenate([
            np.array(hsc_data['ra'], dtype=np.float64),
            np.array(legacy_data['ra'], dtype=np.float64)
        ])
        all_dec = np.concatenate([
            np.array(hsc_data['dec'], dtype=np.float64),
            np.array(legacy_data['dec'], dtype=np.float64)
        ])
        all_healpix = np.concatenate([
            np.array(hsc_data['healpix'], dtype=np.int32),
            np.array(legacy_data['healpix'], dtype=np.int32)
        ])
        all_groups = np.array(['HSC'] * n_hsc + ['Legacy'] * n_legacy, dtype='S10')
        
        # Save metadata datasets (small, can do all at once)
        logger.info("Saving metadata datasets...")
        out_f.create_dataset('object_id', data=all_object_ids)
        out_f.create_dataset('ra', data=all_ra)
        out_f.create_dataset('dec', data=all_dec)
        out_f.create_dataset('healpix', data=all_healpix)
        out_f.create_dataset('group', data=all_groups)
        
        # Pre-allocate image array dataset with chunking
        logger.info("Pre-allocating image array dataset...")
        chunk_size = min(batch_size, 64)  # HDF5 chunk size (smaller for better performance)
        chunks = (chunk_size, 5, h, w)
        
        image_dataset = out_f.create_dataset(
            'image_array',
            shape=(n_total, 5, h, w),
            dtype=np.float32,
            chunks=chunks,
            compression='lzf',
            shuffle=True,
            fletcher32=True
        )
        
        # Write images in batches to avoid memory issues
        logger.info("Writing images in batches...")
        current_idx = 0
        
        # Process HSC images
        logger.info(f"  Processing {n_hsc:,} HSC images...")
        for start_idx in range(0, n_hsc, batch_size):
            end_idx = min(start_idx + batch_size, n_hsc)
            batch_size_actual = end_idx - start_idx
            
            # Stack only the current batch
            batch_images = np.stack(hsc_data['image_array'][start_idx:end_idx])
            
            # Write to dataset
            image_dataset[current_idx:current_idx + batch_size_actual] = batch_images
            current_idx += batch_size_actual
            
            if start_idx % (batch_size * 10) == 0:
                logger.info(f"    Processed {start_idx:,}/{n_hsc:,} HSC images")
        
        # Process Legacy images (with padding)
        logger.info(f"  Processing {n_legacy:,} Legacy images...")
        for start_idx in range(0, n_legacy, batch_size):
            end_idx = min(start_idx + batch_size, n_legacy)
            batch_size_actual = end_idx - start_idx
            
            # Stack only the current batch
            batch_images = np.stack(legacy_data['image_array'][start_idx:end_idx])
            
            # Pad to 5 channels
            batch_padded = np.full((batch_size_actual, 5, h, w), np.nan, dtype=np.float32)
            batch_padded[:, :4, :, :] = batch_images
            
            # Write to dataset
            image_dataset[current_idx:current_idx + batch_size_actual] = batch_padded
            current_idx += batch_size_actual
            
            if start_idx % (batch_size * 10) == 0:
                logger.info(f"    Processed {start_idx:,}/{n_legacy:,} Legacy images")
        
        # Add metadata
        out_f.attrs['created_date'] = datetime.now().isoformat()
        out_f.attrs['n_hsc_galaxies'] = n_hsc
        out_f.attrs['n_legacy_galaxies'] = n_legacy
        out_f.attrs['total_galaxies'] = n_total
        out_f.attrs['hsc_cuts'] = 'r_cmodel_mag < 19'
        out_f.attrs['legacy_cuts'] = 'r_mag < 19'
        out_f.attrs['structure_version'] = 'flat_v1'
        out_f.attrs['legacy_padding'] = 'Channel 5 padded with NaN'
        out_f.attrs['batch_size_used'] = batch_size
        
        logger.info(f"\nSummary:")
        logger.info(f"  HSC galaxies: {n_hsc:,}")
        logger.info(f"  Legacy galaxies: {n_legacy:,}")
        logger.info(f"  Total galaxies: {n_total:,}")
        logger.info(f"  Image shape: ({n_total}, 5, {h}, {w})")
        logger.info(f"  Chunk size: {chunks}")
        logger.info(f"  Compression: LZF (optimized for speed)")
        logger.info(f"  Memory-efficient batch size: {batch_size}")
        logger.info(f"  Structure: Flat (no groups)")
        logger.info(f"  Legacy images: Padded to 5 channels with NaN")


def load_exclusion_coordinates(hdf5_path):
    """Load RA/Dec coordinates from existing HDF5 file to exclude nearby galaxies"""
    logger.info(f"Loading exclusion coordinates from: {hdf5_path}")
    exclusion_coords = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'ra' in f and 'dec' in f:
                ra_array = f['ra'][:]
                dec_array = f['dec'][:]
                
                # Create list of (ra, dec) tuples
                for ra, dec in zip(ra_array, dec_array):
                    exclusion_coords.append((float(ra), float(dec)))
                
                logger.info(f"Loaded {len(exclusion_coords)} galaxy coordinates to check for proximity")
                logger.info(f"Will exclude galaxies within 10 arcsec (0.00278 degrees)")
            else:
                logger.warning("No 'ra' or 'dec' datasets found in exclusion file")
    except Exception as e:
        logger.error(f"Error loading exclusion file: {e}")
        raise
    
    return exclusion_coords


def main():
    """Main function to collect galaxies and create final HDF5"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect galaxies from HSC and Legacy surveys')
    parser.add_argument('--existing-hdf5-to-exclude', type=str, default=None,
                       help='Path to existing HDF5 file containing galaxies to exclude')
    parser.add_argument('--output', type=str, default='data/processed/galaxy_data.hdf5',
                       help='Output HDF5 file path')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    Path("data/logs").mkdir(exist_ok=True, parents=True)
    
    # Configure logging to both file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("data/logs") / f"galaxy_collection_log_{timestamp}.log"
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Use output path from arguments
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Mass Galaxy Labeling - Data Collection")
    logger.info("="*60)
    logger.info(f"Target HSC galaxies: {TARGET_HSC_GALAXIES:,}")
    logger.info(f"Target Legacy galaxies: {TARGET_LEGACY_GALAXIES:,}")
    logger.info(f"Max galaxies per healpix: {MAX_GALAXIES_PER_HEALPIX:,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log file: {log_file}")
    
    # Load exclusion coordinates if provided
    exclusion_coords = None
    if args.existing_hdf5_to_exclude:
        exclusion_coords = load_exclusion_coordinates(args.existing_hdf5_to_exclude)
    
    # Collect galaxies from each survey
    hsc_data = collect_galaxies('hsc', TARGET_HSC_GALAXIES, exclusion_coords)
    legacy_data = collect_galaxies('legacy', TARGET_LEGACY_GALAXIES, exclusion_coords)
    
    # Save indices first as redundancy (in case image saving fails)
    indices_file = output_file.parent / f"collected_indices_{timestamp}.hdf5"
    logger.info(f"\nSaving collected indices for redundancy...")
    save_collected_indices(hsc_data, legacy_data, indices_file)
    logger.info(f"Indices saved to: {indices_file}")
    
    # Save to combined HDF5 with memory-efficient approach
    save_combined_hdf5(hsc_data, legacy_data, output_file)
    
    # Print final summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("Data collection complete!")
    logger.info("="*60)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
    
    # Final complete statistics
    update_log_summary()
    
    # Print data structure
    logger.info("\nHDF5 file structure (flat):")
    logger.info("├── object_id       # String array of galaxy IDs")
    logger.info("├── ra              # Float64 array of right ascension")
    logger.info("├── dec             # Float64 array of declination")
    logger.info("├── healpix         # Int32 array of HEALPix region")
    logger.info("├── group           # String array: 'HSC' or 'Legacy'")
    logger.info("└── image_array     # Float32[n_galaxies, 5, 160, 160]")
    logger.info("                    # Legacy images padded with NaN in channel 5")
    logger.info("\nIndices file saved separately for redundancy/recovery")


if __name__ == "__main__":
    main()