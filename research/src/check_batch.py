#!/usr/bin/env python3
"""
Unified batch checking script for OpenAI batch jobs.
Handles both description generation and augmented description results.
Now supports checking multiple batch jobs from a master batch info file.
Includes automatic retry functionality for failed requests.
"""

import json
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import h5py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
from collections import defaultdict
from logging.handlers import RotatingFileHandler
import asyncio

# Import our utilities
sys.path.append(str(Path(__file__).parent))
from utils.openai_utils import OpenAIBatchProcessor, process_single_result_jsonl, create_batch_job_with_info

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def save_combined_hdf5(all_results_df: pd.DataFrame, output_hdf5_path: Path):
    """Save combined results to HDF5."""
    with h5py.File(output_hdf5_path, 'w') as f:
        # Save each column with appropriate data type handling
        for col in all_results_df.columns:
            data = all_results_df[col].values
            
            # Handle different data types
            if col in ['description', 'prompt', 'model_name', 'model_id', 'prompt_filename', 
                      'plotstyle_filename', 'group', 'object_id']:
                # String columns - use variable-length strings
                dt = h5py.special_dtype(vlen=str)
                string_data = [str(item) for item in data]
                f.create_dataset(col, data=string_data, dtype=dt)
            elif col in ['ra', 'dec', 'llm_cost']:
                # Floating point numbers
                f.create_dataset(col, data=data.astype(np.float64))
            elif col in ['healpix', 'input_tokens', 'output_tokens']:
                # Integers
                f.create_dataset(col, data=data.astype(np.int64))
            else:
                # Default: try to convert to appropriate type
                try:
                    if pd.api.types.is_numeric_dtype(data):
                        if pd.api.types.is_integer_dtype(data):
                            f.create_dataset(col, data=data.astype(np.int64))
                        else:
                            f.create_dataset(col, data=data.astype(np.float64))
                    else:
                        # Fall back to string
                        dt = h5py.special_dtype(vlen=str)
                        string_data = [str(item) for item in data]
                        f.create_dataset(col, data=string_data, dtype=dt)
                except Exception as e:
                    logger.warning(f"Error saving column {col}, using string fallback: {e}")
                    dt = h5py.special_dtype(vlen=str)
                    string_data = [str(item) for item in data]
                    f.create_dataset(col, data=string_data, dtype=dt)


def process_augmented_batch_results(batch_result_path: Path) -> dict:
    """Process augmented description batch results."""
    results_by_galaxy = defaultdict(lambda: {"summaries": [], "metadata": {}})
    
    with open(batch_result_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            
            # Extract custom_id components: galaxy_idx_objectid_summaryidx
            custom_id = result['custom_id']
            parts = custom_id.split('_')
            
            # Handle different custom_id formats
            if len(parts) >= 3:
                galaxy_idx = int(parts[0])
                object_id = '_'.join(parts[1:-1])  # Handle object IDs with underscores
                summary_idx = int(parts[-1])
            else:
                logger.warning(f"Unexpected custom_id format: {custom_id}")
                continue
            
            # Check for errors
            if result.get('error') is not None:
                logger.error(f"Error for {custom_id}: {result['error']}")
                continue
            
            # Extract the generated summary
            try:
                response = result['response']['body']
                # All tasks now use Responses API format
                summary = response['output'][0]['content'][0]['text'].strip()
                
                # Store summary and metadata
                results_by_galaxy[galaxy_idx]["summaries"].append({
                    "index": summary_idx,
                    "text": summary
                })
                results_by_galaxy[galaxy_idx]["object_id"] = object_id
                results_by_galaxy[galaxy_idx]["metadata"] = {
                    "model": response.get('model', 'unknown'),
                    "usage": response.get('usage', {})
                }
            except Exception as e:
                logger.error(f"Failed to parse response for {custom_id}: {e}")
                continue
    
    return results_by_galaxy


def parse_error_file(error_file_path: Path) -> list:
    """Parse error file and extract failed custom_ids."""
    failed_ids = []
    with open(error_file_path, 'r') as f:
        for line in f:
            try:
                error_data = json.loads(line.strip())
                custom_id = error_data.get('custom_id')
                if custom_id:
                    failed_ids.append(custom_id)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line in error file: {line}")
    
    return failed_ids


def find_original_requests(input_jsonl_path: Path, failed_ids: list) -> list:
    """Find original requests for failed custom_ids."""
    failed_requests = []
    failed_ids_set = set(failed_ids)
    
    with open(input_jsonl_path, 'r') as f:
        for line in f:
            try:
                request = json.loads(line.strip())
                if request.get('custom_id') in failed_ids_set:
                    failed_requests.append(request)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line in input file: {line}")
    
    logger.info(f"Found {len(failed_requests)} failed requests out of {len(failed_ids)} failed IDs")
    return failed_requests


async def create_retry_batch(processor, failed_requests, original_batch_info, retry_number, output_dir):
    """Create a retry batch for failed requests."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    retry_folder = output_dir / f'retry_{retry_number}'
    retry_folder.mkdir(parents=True, exist_ok=True)
    
    # Create retry JSONL file
    retry_jsonl_path = retry_folder / f'retry_input_{timestamp}.jsonl'
    processor.create_batch_jsonl(failed_requests, retry_jsonl_path)
    
    # Prepare metadata for retry batch
    metadata = {
        "retry_number": str(retry_number),
        "original_batch_id": original_batch_info.get('batch_id', 'unknown'),
        "failed_requests": str(len(failed_requests))
    }
    
    # Get model config from original batch info
    model_config = original_batch_info.get('model', {})
    
    # Create retry batch job
    batch_info = await create_batch_job_with_info(
        processor,
        retry_jsonl_path,
        1,  # batch_num
        1,  # total_batches
        timestamp,
        "retry_batch",
        model_config,
        metadata,
        retry_folder
    )
    
    logger.info(f"Created retry batch {retry_number} with {len(failed_requests)} requests")
    return batch_info


def check_single_batch(processor, batch_info, output_dir, auto_retry=True, max_retries=3):
    """Check status of a single batch and download if completed."""
    batch_id = batch_info['batch_id']
    
    # Check batch status
    batch = processor.check_batch_status(batch_id)
    
    if batch is None:
        logger.error(f"Failed to retrieve batch {batch_id}")
        return None, "failed", None
    
    logger.info(f"Batch {batch_id} status: {batch.status}")
    
    if batch.status == "completed":
        logger.info(f"Batch completed - downloading results from output file: {batch.output_file_id}")
        
        # Create unique output filename
        batch_num = batch_info.get('batch_number', 1)
        output_path = output_dir / f'batch_results_part{batch_num}.jsonl'
        processor.download_batch_results(batch.output_file_id, output_path)
        
        # Download errors if any
        error_path = None
        if batch.error_file_id:
            error_path = output_dir / f'batch_errors_part{batch_num}.jsonl'
            processor.download_batch_results(batch.error_file_id, error_path)
            logger.info(f"Downloaded error file to: {error_path}")
            
            # Handle retries if enabled
            if auto_retry and error_path.exists():
                # Count failed requests
                failed_ids = parse_error_file(error_path)
                if failed_ids:
                    logger.warning(f"Found {len(failed_ids)} failed requests in batch {batch_id}")
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🔄 AUTOMATIC RETRY INITIATED")
                    logger.info(f"{'='*60}")
                    
                    # Find original input file
                    batch_jsonl = batch_info.get('batch_jsonl')
                    if batch_jsonl and Path(batch_jsonl).exists():
                        # Check current retry count
                        current_retry = batch_info.get('retry_count', 0)
                        if current_retry < max_retries:
                            logger.info(f"Attempting retry {current_retry + 1}/{max_retries}")
                            
                            # Find failed requests
                            failed_requests = find_original_requests(Path(batch_jsonl), failed_ids)
                            
                            if failed_requests:
                                # Create retry batch asynchronously
                                retry_info = asyncio.run(create_retry_batch(
                                    processor,
                                    failed_requests,
                                    batch_info,
                                    current_retry + 1,
                                    output_dir
                                ))
                                
                                # Update retry tracking
                                retry_batch_id = retry_info['batch_id']
                                logger.info(f"✅ Created retry batch: {retry_batch_id}")
                                logger.info(f"   Failed requests to retry: {len(failed_requests)}")
                                logger.info(f"   Retry batch info: {retry_info['batch_info_path']}")
                                logger.info(f"{'='*60}\n")
                                
                                # Save retry info
                                retry_info_path = output_dir / f'retry_{current_retry + 1}_info.json'
                                with open(retry_info_path, 'w') as f:
                                    json.dump({
                                        'retry_batch_id': retry_batch_id,
                                        'original_batch_id': batch_id,
                                        'retry_number': current_retry + 1,
                                        'failed_count': len(failed_requests),
                                        'retry_info': retry_info
                                    }, f, indent=2)
                        else:
                            logger.warning(f"Max retries ({max_retries}) reached for batch {batch_id}")
                    else:
                        logger.warning(f"Could not find original input file for retry: {batch_jsonl}")
        
        return output_path, "completed", error_path
        
    elif batch.status in ["failed", "expired", "cancelled"]:
        logger.error(f"Batch failed with status: {batch.status}")
        return None, batch.status, None
    else:
        logger.info(f"Batch {batch_id} not yet completed - status: {batch.status}")
        return None, batch.status, None


def check_retry_batches(output_dir):
    """Check for any retry batch info files and return their status."""
    retry_infos = []
    for retry_info_file in output_dir.glob("retry_*_info.json"):
        with open(retry_info_file, 'r') as f:
            retry_info = json.load(f)
            retry_infos.append(retry_info)
    
    return retry_infos


def check_and_download_results(processor, batch_info, output_dir, auto_retry=True, max_retries=3):
    """Check batch status and download results if completed.
    Handles both single batch and multi-batch scenarios."""
    
    # Check if this is a master batch info file
    if 'batches' in batch_info:
        # Multi-batch scenario
        logger.info(f"Processing {len(batch_info['batches'])} batch jobs")
        
        all_results_paths = []
        all_completed = True
        any_failed = False
        
        for batch_data in batch_info['batches']:
            # Load individual batch info
            with open(batch_data['batch_info_path'], 'r') as f:
                sub_batch_info = json.load(f)
            
            result_path, status, error_path = check_single_batch(processor, sub_batch_info, output_dir, auto_retry, max_retries)
            
            if status == "completed" and result_path:
                all_results_paths.append(result_path)
            elif status in ["failed", "expired", "cancelled"]:
                any_failed = True
                all_completed = False
            else:
                all_completed = False
        
        if any_failed:
            logger.error("One or more batches failed")
            return None
        elif not all_completed:
            logger.info("Not all batches are completed yet")
            return None
        else:
            logger.info(f"All {len(all_results_paths)} batches completed successfully")
            return all_results_paths
    else:
        # Single batch scenario (backward compatibility)
        result_path, status, error_path = check_single_batch(processor, batch_info, output_dir, auto_retry, max_retries)
        return [result_path] if result_path else None


def combine_multi_batch_results(results_paths, batch_info, prompt_text, additional_metadata):
    """Combine results from multiple batch files into a single DataFrame."""
    all_dfs = []
    
    for i, result_path in enumerate(results_paths):
        logger.info(f"Processing batch result {i+1}/{len(results_paths)}: {result_path}")
        
        # Process each result file
        df = process_single_result_jsonl(
            result_path,
            batch_info['model'],
            prompt_text,
            additional_metadata
        )
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} results from {len(results_paths)} batch files")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Check batch status and download results')
    parser.add_argument('batch_info_file', type=str,
                       help='Path to batch info JSON file (can be master or individual)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--base-hdf5', type=str, default=None,
                       help='Path to base galaxy_descriptions HDF5 file to augment (for augmented batches only)')
    parser.add_argument('--no-auto-retry', action='store_true',
                       help='Disable automatic retry of failed requests')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retry attempts (default: 3)')
    
    args = parser.parse_args()
    
    # Load batch info file to determine type
    with open(args.batch_info_file, 'r') as f:
        batch_info = json.load(f)
    
    # Check if this is a master batch file
    is_master = 'batches' in batch_info
    
    # Auto-detect batch type from batch info contents
    if 'num_summaries_per_galaxy' in batch_info or 'prompt_template' in batch_info:
        batch_type = 'augmented'
    elif 'plotstyle' in batch_info or 'prompt_file' in batch_info:
        batch_type = 'descriptions'
    else:
        # For master files, check the metadata
        if is_master and batch_info.get('plotstyle'):
            batch_type = 'descriptions'
        elif is_master and batch_info.get('prompt_template'):
            batch_type = 'augmented'
        else:
            # Fallback: check batch endpoint
            if batch_info.get('endpoint') == '/v1/responses':
                batch_type = 'descriptions'
            else:
                batch_type = 'augmented'
    
    # Set up logging
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path("data/logs") / f'check_batch_{batch_type}_{timestamp}.log'
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Disable verbose HTTP request logging from httpx (used by OpenAI)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger.info(f"Log file: {log_file}")
    logger.info(f"Processing {'master' if is_master else 'single'} batch file")
    logger.info(f"Auto-detected batch type: {batch_type}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use timestamp from batch info for consistency
        batch_timestamp = batch_info.get('timestamp', timestamp)
        
        # If master batch, use batch folder
        if is_master and 'batch_folder' in batch_info:
            output_dir = Path(batch_info['batch_folder']) / 'results'
        else:
            if batch_type == 'descriptions':
                output_dir = Path('data/processed') / f'batch_results_{batch_timestamp}'
            else:
                output_dir = Path('data/processed') / f'augmented_results_{batch_timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize batch processor
    processor = OpenAIBatchProcessor()
    
    # Check and download results
    auto_retry = not args.no_auto_retry
    results_paths = check_and_download_results(processor, batch_info, output_dir, auto_retry, args.max_retries)
    
    if not results_paths:
        logger.info("No results to process - batches are not yet completed")
        sys.exit(0)
    
    if batch_type == 'descriptions':
        # Process description results
        logger.info("Processing description generation results...")
        
        # Load galaxy data for location information if available
        galaxy_data = {}
        input_file = batch_info.get('input_file', 'data/processed/galaxy_data.hdf5')
        try:
            with h5py.File(input_file, 'r') as f:
                # Check if this is the new flat structure
                is_flat_structure = 'group' in f.keys() and 'object_id' in f.keys()
                
                if is_flat_structure:
                    logger.info("Loading from flat HDF5 structure")
                    object_ids = f['object_id'][:]
                    ras = f['ra'][:]
                    decs = f['dec'][:]
                    healpixs = f['healpix'][:]
                    groups = f['group'][:]
                    
                    for i in range(len(object_ids)):
                        object_id = object_ids[i]
                        if isinstance(object_id, bytes):
                            object_id = object_id.decode('utf-8')
                        group = groups[i]
                        if isinstance(group, bytes):
                            group = group.decode('utf-8')
                        
                        galaxy_data[object_id] = {
                            'ra': ras[i],
                            'dec': decs[i],
                            'healpix': healpixs[i],
                            'group': group
                        }
            logger.info(f"Loaded galaxy data for {len(galaxy_data)} objects")
        except Exception as e:
            logger.warning(f"Could not load galaxy data: {e}")
        
        # Read prompt text
        prompt_text = ""
        try:
            prompt_file = batch_info.get('prompt_file', batch_info.get('prompt', ''))
            if prompt_file and Path(prompt_file).exists():
                with open(prompt_file, 'r') as f:
                    prompt_text = f.read().strip()
        except:
            pass
        
        # Get additional metadata from batch info
        additional_metadata = {
            'prompt_filename': batch_info.get('prompt_file', ''),
            'plotstyle_filename': batch_info.get('plotstyle', '')
        }
        
        # Process results - handle multiple files if needed
        if len(results_paths) > 1:
            combined_df = combine_multi_batch_results(
                results_paths, batch_info, prompt_text, additional_metadata
            )
        else:
            # Single file processing (backward compatibility)
            results_path = results_paths[0]
            logger.info(f"Processing {results_path}")
            
            combined_df = process_single_result_jsonl(
                results_path,
                batch_info['model'],
                prompt_text,
                additional_metadata
            )
        
        # Add galaxy location data
        for idx, row in combined_df.iterrows():
            object_id = row['object_id']
            if object_id in galaxy_data:
                for key, value in galaxy_data[object_id].items():
                    combined_df.at[idx, key] = value
            else:
                combined_df.at[idx, 'ra'] = 0.0
                combined_df.at[idx, 'dec'] = 0.0
                combined_df.at[idx, 'healpix'] = 0
                combined_df.at[idx, 'group'] = 'unknown'
        
        logger.info(f"Processed {len(combined_df)} results")
        
        # Save to HDF5
        batch_timestamp = batch_info.get('timestamp', timestamp)
        hdf5_path = output_dir.parent / f'galaxy_descriptions_{batch_timestamp}.hdf5'
        save_combined_hdf5(combined_df, hdf5_path)
        
        # Print statistics
        total_input_tokens = combined_df['input_tokens'].sum()
        total_output_tokens = combined_df['output_tokens'].sum()
        total_cost = combined_df['llm_cost'].sum()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"Total galaxies processed: {len(combined_df)}")
        logger.info(f"Total input tokens: {total_input_tokens:,}")
        logger.info(f"Total output tokens: {total_output_tokens:,}")
        logger.info(f"Total cost: ${total_cost:.4f}")
        logger.info(f"Average cost per galaxy: ${total_cost/len(combined_df):.6f}")
        logger.info(f"\n✅ Results saved to:")
        logger.info(f"   HDF5: {hdf5_path}")
        logger.info(f"   JSONL: {output_dir}")
        
        # Check for retry batches
        retry_infos = check_retry_batches(output_dir)
        if retry_infos:
            logger.info(f"\n🔄 Retry batches created:")
            for retry_info in retry_infos:
                logger.info(f"   Retry {retry_info['retry_number']}: {retry_info['failed_count']} failed requests")
                logger.info(f"   Batch ID: {retry_info['retry_batch_id']}")
            logger.info(f"\nTo check retry batch status, run:")
            for retry_info in retry_infos:
                retry_batch_info_path = retry_info['retry_info']['batch_info_path']
                logger.info(f"   uv run src/check_batch.py {retry_batch_info_path}")
        
    else:  # augmented
        # Process augmented results
        logger.info("Processing augmented description results...")
        
        # For augmented results, combine from multiple files if needed
        all_results = {}
        
        for i, results_path in enumerate(results_paths):
            logger.info(f"Processing batch result {i+1}/{len(results_paths)}: {results_path}")
            batch_results = process_augmented_batch_results(results_path)
            
            # Merge results
            for galaxy_idx, result_data in batch_results.items():
                if galaxy_idx not in all_results:
                    all_results[galaxy_idx] = result_data
                else:
                    # Merge summaries
                    all_results[galaxy_idx]["summaries"].extend(result_data["summaries"])
        
        logger.info(f"Downloaded and processed results for {len(all_results)} galaxies")
        
        # Find the existing galaxy_descriptions HDF5 file to update
        if not args.base_hdf5:
            logger.error("For augmented description processing, you must specify --base-hdf5 parameter")
            logger.error("Example: uv run src/check_batch.py batch_info.json --base-hdf5 data/processed/galaxy_descriptions_20250708_123456.hdf5")
            sys.exit(1)
        
        base_hdf5_path = Path(args.base_hdf5)
        
        if not base_hdf5_path.exists():
            logger.error(f"Base HDF5 file not found: {base_hdf5_path}")
            logger.error("Please check the --base-hdf5 path is correct")
            sys.exit(1)
        
        # Load the existing HDF5 file
        logger.info(f"Loading existing HDF5 file: {base_hdf5_path}")
        with h5py.File(base_hdf5_path, 'r') as f:
            # Read all existing data
            existing_data = {}
            for key in f.keys():
                existing_data[key] = f[key][:]
                logger.info(f"Loaded existing column: {key} (shape: {f[key].shape})")
        
        # Create mapping from object_id to summaries
        object_id_to_summaries = {}
        for galaxy_idx, result_data in all_results.items():
            object_id = result_data.get("object_id")
            if object_id and "summaries" in result_data:
                # Sort summaries by index and extract text
                summaries = sorted(result_data["summaries"], key=lambda x: x["index"])
                summary_texts = [s["text"] for s in summaries]
                object_id_to_summaries[object_id] = summary_texts
        
        # Create new columns for the existing object_ids
        existing_object_ids = existing_data['object_id']
        num_objects = len(existing_object_ids)
        
        # Initialize new columns
        summaries_col = []
        summary_model_col = []
        summary_prompt_template_col = []
        
        model_name = batch_info.get('model', {}).get('formatted_name', 'unknown')
        prompt_template = batch_info.get('prompt_template', '')
        
        for i in range(num_objects):
            object_id = existing_object_ids[i]
            if isinstance(object_id, bytes):
                object_id = object_id.decode('utf-8')
            
            if object_id in object_id_to_summaries:
                summaries_col.append(object_id_to_summaries[object_id])
            else:
                summaries_col.append([])  # Empty list if no summaries found
            
            summary_model_col.append(model_name)
            summary_prompt_template_col.append(prompt_template)
        
        # Create output HDF5 file with augmented data
        batch_timestamp = batch_info.get('timestamp', timestamp)
        augmented_hdf5_path = output_dir.parent / f'galaxy_descriptions_augmented_{batch_timestamp}.hdf5'
        logger.info(f"Creating augmented HDF5 file: {augmented_hdf5_path}")
        
        with h5py.File(augmented_hdf5_path, 'w') as f:
            # Copy all existing columns
            for key, data in existing_data.items():
                if key in ['description', 'prompt', 'model_name', 'model_id', 'prompt_filename', 
                          'plotstyle_filename', 'group', 'object_id']:
                    # String columns
                    dt = h5py.special_dtype(vlen=str)
                    if isinstance(data[0], bytes):
                        string_data = [item.decode('utf-8') for item in data]
                    else:
                        string_data = [str(item) for item in data]
                    f.create_dataset(key, data=string_data, dtype=dt)
                elif key in ['ra', 'dec', 'llm_cost']:
                    # Float columns
                    f.create_dataset(key, data=data.astype(np.float64))
                elif key in ['healpix', 'input_tokens', 'output_tokens']:
                    # Integer columns
                    f.create_dataset(key, data=data.astype(np.int64))
                else:
                    # Default handling
                    try:
                        f.create_dataset(key, data=data)
                    except Exception as e:
                        logger.warning(f"Error copying column {key}, using string fallback: {e}")
                        dt = h5py.special_dtype(vlen=str)
                        string_data = [str(item) for item in data]
                        f.create_dataset(key, data=string_data, dtype=dt)
            
            # Add new augmented columns
            dt_str = h5py.special_dtype(vlen=str)
            
            # summaries column - serialize lists as JSON strings
            summaries_json = [json.dumps(summaries) for summaries in summaries_col]
            f.create_dataset('summaries', data=summaries_json, dtype=dt_str)
            
            # summary_model column - string
            f.create_dataset('summary_model', data=summary_model_col, dtype=dt_str)
            
            # summary_prompt_template column - string  
            f.create_dataset('summary_prompt_template', data=summary_prompt_template_col, dtype=dt_str)
        
        # Print statistics
        total_galaxies = len(existing_object_ids)
        galaxies_with_summaries = len([s for s in summaries_col if len(s) > 0])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Augmented Results Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"Total galaxies in base file: {total_galaxies}")
        logger.info(f"Galaxies with summaries: {galaxies_with_summaries}")
        logger.info(f"Summary model: {model_name}")
        logger.info(f"\n✅ Processing complete!")
        logger.info(f"Augmented HDF5 file saved to: {augmented_hdf5_path}")
        logger.info(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()