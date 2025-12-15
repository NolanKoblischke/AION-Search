#!/usr/bin/env python3
"""
Generate augmented descriptions for galaxies using GPT-4.1-nano via Batch API.
Creates multiple single-sentence summaries for each original description.
Now uses shared utilities for batch processing.
"""

import h5py
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
import logging
import asyncio

# Import our utilities
sys.path.append(str(Path(__file__).parent))
from utils.openai_utils import (
    OpenAIBatchProcessor, load_model_configs, 
    setup_batch_logging, create_batch_job_with_info, save_master_batch_info,
    MAX_BATCH_REQUESTS
)

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


async def create_summary_batches(
    descriptions, 
    galaxy_indices, 
    object_ids,
    model_config, 
    prompt_template,
    output_dir,
    timestamp,
    num_summaries
):
    """Create batch jobs for generating summaries, splitting into 50k request batches."""
    processor = OpenAIBatchProcessor()
    
    # Create batch folder for this run
    batch_folder = output_dir / f'augmented_batch_run_{timestamp}'
    batch_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate all requests first
    logger.info(f"Creating batch requests for {len(descriptions)} galaxies x {num_summaries} summaries")
    
    all_requests = []
    
    for idx, (desc, galaxy_idx, object_id) in enumerate(
        tqdm(zip(descriptions, galaxy_indices, object_ids), 
             total=len(descriptions), 
             desc="Preparing batch requests")
    ):
        # Decode if needed
        if isinstance(desc, bytes):
            desc = desc.decode('utf-8')
        if isinstance(object_id, bytes):
            object_id = object_id.decode('utf-8')
            
        # Create num_summaries requests for each galaxy
        for summary_idx in range(num_summaries):
            prompt = prompt_template.replace("{{original_description}}", desc)
            
            # Create unique custom ID
            custom_id = f"{galaxy_idx}_{object_id}_{summary_idx}"
            
            # Create request for Responses API (text-only)
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model_config['model_name'],
                    "input": [{
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}]
                    }]
                }
            }
            
            all_requests.append(request)
    
    total_requests = len(all_requests)
    logger.info(f"Total requests to process: {total_requests}")
    
    # Calculate number of batch jobs needed
    num_batch_jobs = (total_requests + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
    logger.info(f"Will create {num_batch_jobs} batch job(s) (max {MAX_BATCH_REQUESTS:,} requests per batch)")
    
    # Create batch jobs
    all_batch_infos = []
    
    for batch_num in range(num_batch_jobs):
        start_idx = batch_num * MAX_BATCH_REQUESTS
        end_idx = min((batch_num + 1) * MAX_BATCH_REQUESTS, total_requests)
        batch_requests = all_requests[start_idx:end_idx]
        
        # Save batch input JSONL
        batch_jsonl_path = batch_folder / f'batch_input_{timestamp}_part{batch_num+1}.jsonl'
        processor.create_batch_jsonl(batch_requests, batch_jsonl_path)
        
        # Create batch job using shared function
        metadata = {
            "prompt_template": prompt_template[:500],  # Truncate long prompts for metadata
            "num_summaries": str(num_summaries),
            "total_requests": str(len(batch_requests))
        }
        
        batch_info = await create_batch_job_with_info(
            processor, batch_jsonl_path, batch_num + 1, num_batch_jobs,
            timestamp, "galaxy_augmented_descriptions", model_config, 
            metadata, batch_folder
        )
        
        all_batch_infos.append(batch_info)
    
    # Save master batch info using shared function
    additional_info = {
        "prompt_template": prompt_template,
        "num_summaries": num_summaries
    }
    
    master_info_path = save_master_batch_info(
        batch_folder, timestamp, model_config, total_requests,
        all_batch_infos, additional_info
    )
    
    return master_info_path


def main():
    parser = argparse.ArgumentParser(description='Generate augmented descriptions using GPT-4.1-nano Batch API')
    parser.add_argument('--input', type=str, 
                       default='data/processed/galaxy_descriptions.hdf5',
                       help='Input HDF5 file with text descriptions')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='Output directory for batch files')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano',
                       help='Model ID for generating summaries')
    parser.add_argument('--prompt-template', type=str,
                       default='Please summarize the following description into a single sentence CLIP query:\n<original_description>\n{{original_description}}\n</original_description>\nOnly output the summary without any additional text.\nDo not use any of the following words: `no`, `not`, `without`, `absence`, `lack of`, `no obvious`, `no signs`, `absence of` or any other negation words or phrases. Ignore any phrases containing negation from the original description containing these words.',
                       help='Prompt template for summary generation')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of galaxies to process (for testing)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting index for processing')
    parser.add_argument('--n-summaries', type=int, default=3,
                       help='Number of summaries per galaxy (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging using shared function
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_batch_logging('generate_augmented_batch', timestamp)
    
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of summaries per galaxy: {args.n_summaries}")
    
    # Load model configuration
    model_configs = load_model_configs()
    if args.model not in model_configs:
        logger.error(f"Model {args.model} not found in configurations")
        logger.error(f"Available models: {list(model_configs.keys())}")
        sys.exit(1)
    model_config = model_configs[args.model]
    
    # Load descriptions
    logger.info("Loading descriptions from HDF5...")
    with h5py.File(args.input, 'r') as f:
        # Read all descriptions and object IDs directly
        descriptions = f['description'][:]
        object_ids = f['object_id'][:]
        
        # Apply start index and limit if specified
        if args.start_index > 0:
            descriptions = descriptions[args.start_index:]
            object_ids = object_ids[args.start_index:]
            
        if args.limit:
            descriptions = descriptions[:args.limit]
            object_ids = object_ids[:args.limit]
        
        # Create simple galaxy indices for custom_id generation
        galaxy_indices = list(range(len(descriptions)))
        
        logger.info(f"Loaded {len(descriptions)} descriptions")
    
    # Estimate cost
    # Rough estimate: ~500 tokens per request (prompt + response)
    total_requests = len(descriptions) * args.n_summaries
    estimated_tokens = total_requests * 500
    
    # Get pricing from model config
    input_price = model_config.get('input_price', 0.150) / 1_000_000  # per token
    output_price = model_config.get('output_price', 0.600) / 1_000_000
    
    # Assume 400 input tokens, 100 output tokens per request
    estimated_cost = total_requests * (400 * input_price + 100 * output_price) / 2  # Batch pricing
    
    logger.info(f"\nEstimated cost:")
    logger.info(f"  Total requests: {total_requests:,}")
    logger.info(f"  Estimated tokens: {estimated_tokens:,}")
    logger.info(f"  Estimated cost: ${estimated_cost:.2f}")
    
    # Create batch jobs
    import time
    start_time = time.time()
    
    # Run async batch creation
    batch_info_path = asyncio.run(create_summary_batches(
        descriptions,
        galaxy_indices,
        object_ids,
        model_config,
        args.prompt_template,
        output_dir,
        timestamp,
        args.n_summaries
    ))
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nBatch creation time: {elapsed_time/60:.1f} minutes")


if __name__ == "__main__":
    main()