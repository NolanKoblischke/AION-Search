import h5py
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import shutil
import logging
import asyncio
# Import our utilities
sys.path.append(str(Path(__file__).parent))
from utils.openai_utils import (
    OpenAIBatchProcessor, load_model_configs, upload_files_batch, 
    wait_for_file_processing, load_plotting_function, read_prompt,
    setup_batch_logging, create_batch_job_with_info, save_master_batch_info,
    MAX_BATCH_REQUESTS
)

load_dotenv()

# Set up logging - will be configured in main()
logger = logging.getLogger(__name__)

# Disable verbose HTTP logging from httpx (used by OpenAI)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

BATCH_SIZE = 500  # Process images in batches of 500 for better efficiency
CHUNK_SIZE = 50_000  # Process data in chunks to manage memory
PARALLEL_UPLOADS = 25  # Number of concurrent uploads for faster processing


def process_single_galaxy(args):
    """Process a single galaxy to generate image - designed for parallel execution."""
    _, row, temp_dir, plotstyle_path = args
    
    try:
        # Load plotting function in worker process to avoid pickling issues
        plot_func = load_plotting_function(plotstyle_path)
        
        # Generate plot
        image_path = plot_func(
            row['image_array'], 
            str(row['object_id']), 
            output_dir=temp_dir
        )
        
        return {
            'object_id': row['object_id'],
            'image_path': image_path,
            'ra': row['ra'],
            'dec': row['dec'],
            'healpix': row['healpix'],
            'group': row['group']
        }
        
    except Exception as e:
        logger.error(f"Error processing galaxy {row.get('object_id', 'unknown')}: {str(e)}")
        return None

async def upload_batch(client, paths: list[Path]) -> list[tuple]:
    """Upload a batch of images using centralized upload function."""
    # Use the centralized upload function which handles concurrency and error handling
    return await upload_files_batch(client, paths, purpose="vision", max_concurrent=PARALLEL_UPLOADS)

def build_batch_jsonl(file_mappings: list[tuple], prompt: str, model_id: str, output_path: Path):
    """Create a JSONL file with batch requests."""
    with output_path.open("w") as f:
        for object_id, file_id in file_mappings:
            req = {
                "custom_id": object_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model_id,
                    "input": [{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "file_id": file_id}
                        ]
                    }]
                }
            }
            f.write(json.dumps(req) + "\n")

def build_batch_jsonl_wrapper(file_mappings: list[tuple], output_path: Path, prompt: str, model_id: str):
    """Wrapper to match the signature expected by split_and_create_batches."""
    build_batch_jsonl(file_mappings, prompt, model_id, output_path)

async def wait_for_files_and_create_batch(client, file_mappings, batch_folder, timestamp, 
                                          batch_num, total_batches, prompt, model_config, processor,
                                          prompt_file=None, plotstyle=None, input_file=None):
    """Wait for files to be processed and create batch job."""
    # Extract file IDs
    file_ids = [fid for _, fid in file_mappings]
    
    if file_ids:
        # Wait for these specific files to be processed
        await wait_for_file_processing(client, file_ids)
        
        # Create batch JSONL file
        batch_jsonl_path = batch_folder / f'batch_input_{timestamp}_part{batch_num}.jsonl'
        build_batch_jsonl(file_mappings, prompt, model_config['model_name'], batch_jsonl_path)
        
        # Create batch job
        metadata = {
            "prompt": prompt_file or "galaxy description generation",
            "plotstyle": plotstyle or "default",
            "prompt_file": prompt_file,
            "input_file": input_file,
            "total_requests": str(len(file_mappings))
        }
        
        batch_info = await create_batch_job_with_info(
            processor, batch_jsonl_path, batch_num, total_batches,
            timestamp, "galaxy_descriptions", model_config, metadata, batch_folder
        )
        
        logger.info(f"Created batch job {batch_num} with {len(file_mappings)} requests")
        return batch_info
    
    return None

async def process_and_upload_batch(client, batch_data, plotstyle_path, images_dir, n_cores):
    """Process a batch of galaxies: generate images, upload, and return file mappings."""
    # Prepare arguments for parallel processing
    process_args = []
    for idx, row_dict in enumerate(batch_data):
        args_tuple = (idx, row_dict, str(images_dir), plotstyle_path)
        process_args.append(args_tuple)
    
    # Generate images in parallel
    image_results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_single_galaxy, args) for args in process_args]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                image_results.append(result)
    
    # Upload images
    image_paths = [Path(r['image_path']) for r in image_results]
    file_mappings = await upload_batch(client, image_paths)
    
    # Clean up images immediately after upload (don't wait for processing)
    for path in image_paths:
        try:
            path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
    
    # Return mappings without waiting for processing
    return file_mappings

async def main_async(args, model_config, prompt, timestamp):
    """Main async function to handle batch processing and uploads."""
    # Initialize OpenAI batch processor
    processor = OpenAIBatchProcessor()
    client = processor.client  # Get the async client for file operations
    
    # Create batch folder for this run
    batch_folder = Path(args.output_dir) / f'batch_run_{timestamp}'
    batch_folder.mkdir(parents=True, exist_ok=True)
    
    # Create temporary images directory
    images_dir = batch_folder / 'batch_images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all file mappings
    all_file_mappings = []
    
    # Store plotstyle path for worker processes
    plotstyle_path = args.plotstyle
    
    try:
        with h5py.File(args.input, 'r') as f:
            # Check for required columns
            required_cols = ['object_id', 'ra', 'dec', 'healpix', 'image_array', 'group']
            available_cols = list(f.keys())
            missing_cols = [col for col in required_cols if col not in available_cols]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Get total number of galaxies
            total_galaxies = len(f['object_id'])
            logger.info(f"Total galaxies in HDF5: {total_galaxies:,}")
            
            # Count by group for logging
            groups = f['group'][:]
            unique_groups, counts = np.unique(groups, return_counts=True)
            for group, count in zip(unique_groups, counts):
                group_str = group.decode() if isinstance(group, bytes) else str(group)
                logger.info(f"Group '{group_str}': {count:,} galaxies")
            
            # Apply start-index and limit
            if args.start_index >= total_galaxies:
                raise ValueError(f"Start index {args.start_index} is beyond total galaxies {total_galaxies}")
            
            # Calculate end index
            end_index = total_galaxies
            if args.limit:
                end_index = min(args.start_index + args.limit, total_galaxies)
            
            galaxies_to_process = end_index - args.start_index
            logger.info(f"Processing galaxies from index {args.start_index:,} to {end_index-1:,} ({galaxies_to_process:,} total)")
            
            # Track batch creation tasks
            batch_creation_tasks = []
            batch_counter = 0
            current_batch_mappings = []
            all_batch_infos = []
            
            # Progress bar
            with tqdm(total=galaxies_to_process, desc="Processing and uploading galaxies") as pbar:
                processed_count = 0
                
                # Process data in chunks
                start_idx = args.start_index
                for chunk_start in range(start_idx, end_index, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, end_index)
                    
                    # Load chunk data
                    chunk_data = {}
                    for col in required_cols:
                        chunk_data[col] = f[col][chunk_start:chunk_end]
                    
                    chunk_size = chunk_end - chunk_start
                    
                    # Process chunk in batches
                    for batch_start in range(0, chunk_size, BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, chunk_size)
                        
                        # Prepare batch data
                        batch_data = []
                        for idx in range(batch_start, batch_end):
                            row_dict = {}
                            for col in required_cols:
                                value = chunk_data[col][idx]
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                                row_dict[col] = value
                            
                            # Handle padded Legacy images
                            if row_dict['group'] == 'Legacy':
                                # Remove NaN padding from 5th channel for Legacy images
                                row_dict['image_array'] = row_dict['image_array'][:4]
                            
                            batch_data.append(row_dict)
                        
                        # Process and upload batch
                        try:
                            file_mappings = await process_and_upload_batch(
                                client, batch_data, plotstyle_path, images_dir, args.n_cores
                            )
                            all_file_mappings.extend(file_mappings)
                            current_batch_mappings.extend(file_mappings)
                            logger.info(f"Uploaded {len(file_mappings)} images from batch")
                            
                            pbar.update(len(file_mappings))
                            processed_count += len(file_mappings)
                            
                            # Check if we have enough mappings for a batch job
                            if len(current_batch_mappings) >= MAX_BATCH_REQUESTS:
                                # Split off mappings for this batch job
                                batch_job_mappings = current_batch_mappings[:MAX_BATCH_REQUESTS]
                                current_batch_mappings = current_batch_mappings[MAX_BATCH_REQUESTS:]
                                
                                # Create async task for batch creation
                                batch_counter += 1
                                task = asyncio.create_task(
                                    wait_for_files_and_create_batch(
                                        client, batch_job_mappings, batch_folder, timestamp,
                                        batch_counter, None, prompt, model_config, processor,
                                        args.prompt, args.plotstyle, args.input
                                    )
                                )
                                batch_creation_tasks.append(task)
                                logger.info(f"Started batch creation task #{batch_counter} with {len(batch_job_mappings)} requests")
                            
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            continue
            
            # Handle remaining mappings
            if current_batch_mappings:
                batch_counter += 1
                task = asyncio.create_task(
                    wait_for_files_and_create_batch(
                        client, current_batch_mappings, batch_folder, timestamp,
                        batch_counter, None, prompt, model_config, processor,
                        args.prompt, args.plotstyle, args.input
                    )
                )
                batch_creation_tasks.append(task)
                logger.info(f"Started final batch creation task #{batch_counter} with {len(current_batch_mappings)} requests")
            
            # Wait for all batch creation tasks to complete
            if batch_creation_tasks:
                logger.info(f"Waiting for {len(batch_creation_tasks)} batch creation tasks to complete...")
                batch_results = await asyncio.gather(*batch_creation_tasks, return_exceptions=True)
                
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch creation task {i+1} failed: {result}")
                    elif result:
                        all_batch_infos.append(result)
        
        # Save master batch info if we created any batches
        if all_batch_infos:
            logger.info(f"Created {len(all_batch_infos)} batch job(s) with {len(all_file_mappings)} total requests")
            
            # Save master batch info using shared function
            additional_info = {
                "prompt_file": args.prompt,
                "plotstyle": args.plotstyle,
                "input_file": args.input
            }
            
            save_master_batch_info(
                batch_folder, timestamp, model_config, len(all_file_mappings),
                all_batch_infos, additional_info
            )
        
        # Clean up images directory
        if images_dir.exists():
            shutil.rmtree(images_dir)
            logger.info(f"Cleaned up images directory: {images_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Clean up on error
        if images_dir.exists():
            shutil.rmtree(images_dir)
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate galaxy descriptions using OpenAI Batch API')
    parser.add_argument('--input', type=str, 
                       default='data/processed/galaxy_data.hdf5',
                       help='Input HDF5 file path')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                       help='Model ID (e.g., gpt-4.1-mini)')
    parser.add_argument('--prompt', type=str, default='src/prompts/general_promptv5.txt',
                       help='Prompt file path')
    parser.add_argument('--plotstyle', type=str, default='src/plotting_scripts/default_plot.py',
                       help='Plotting script file path')
    parser.add_argument('--n-cores', type=int, default=8,
                       help='Number of parallel cores to use for image generation')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for results')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting index for processing galaxies (default: 0)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Number of galaxies to process from start-index')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging using shared function
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_batch_logging('multi_batch_generate', timestamp)
    
    # Load model configurations
    model_configs = load_model_configs()
    
    # Check if model is valid
    if args.model not in model_configs:
        print(f"Model {args.model} not found in model configs")
        print(f"Available models: {list(model_configs.keys())}")
        raise ValueError(f"Model {args.model} not found in model configs")
    else:
        model_config = model_configs.get(args.model)
    
    logger.info(f"Loading input file: {args.input}")
    logger.info(f"Using model: {model_config['formatted_name']}")
    logger.info(f"Using prompt: {args.prompt}")
    logger.info(f"Using {args.n_cores} cores for image generation")
    if args.start_index > 0:
        logger.info(f"Starting from index: {args.start_index:,}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit:,} galaxies")
    
    # Read prompt
    prompt = read_prompt(args.prompt)
    
    # Validate plotting function exists
    logger.info(f"Validating plotting function from: {args.plotstyle}")
    if not Path(args.plotstyle).exists():
        raise FileNotFoundError(f"Plotting script '{args.plotstyle}' not found")
    
    # Run async main function
    asyncio.run(main_async(args, model_config, prompt, timestamp))

if __name__ == "__main__":
    main()