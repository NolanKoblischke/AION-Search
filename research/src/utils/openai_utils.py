#!/usr/bin/env python3
"""
OpenAI utilities for batch processing and embedding generation.
Provides general functions for creating OpenAI batch jobs and processing results.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import logging
import asyncio
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Tuple, Optional, Any, Callable
import h5py
from logging.handlers import RotatingFileHandler

load_dotenv()

# API rate limiting for embeddings
TOKENS_PER_MINUTE = 5000000  # TPM limit for text-embedding-3-large
BATCH_SIZE = 100  # Process in batches
MAX_RETRIES = 3
RATE_LIMIT_BUFFER = 0.9  # Use 90% of rate limit to be safe
PARALLEL_UPLOADS = 25  # Number of concurrent uploads
MAX_BATCH_REQUESTS = 50_000  # Maximum requests per batch job

logger = logging.getLogger(__name__)


def load_plotting_function(plotstyle: str) -> Callable:
    """Dynamically load the plot_decals function from the specified plotting script."""
    import importlib.util
    
    # Convert to Path and check if it exists
    script_path = Path(plotstyle)
    if not script_path.exists():
        raise FileNotFoundError(f"Plotting script '{plotstyle}' not found")
    
    # Get module name from filename
    module_name = script_path.stem
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the plot_decals function
    if not hasattr(module, 'plot_decals'):
        raise AttributeError(f"Plotting script '{script_path}' does not have a 'plot_decals' function")
    
    return module.plot_decals


def read_prompt(prompt_filename: str) -> str:
    """Read prompt from file."""
    prompt_path = Path(prompt_filename)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file '{prompt_filename}' not found")
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_model_configs():
    """Load model configurations from models.jsonl."""
    models_path = Path(__file__).parent / "models.jsonl"
    models = {}
    with open(models_path, 'r') as f:
        for line in f:
            model = json.loads(line.strip())
            models[model['id']] = model
    return models


def process_single_galaxy(args):
    """Process a single galaxy to generate image - designed for parallel execution."""
    _, row, temp_dir, plot_func, plot_kwargs = args
    
    try:
        # Generate plot using the provided function
        image_path = plot_func(
            row['image_array'], 
            str(row['object_id']), 
            output_dir=temp_dir,
            **plot_kwargs
        )
        
        return {
            'object_id': row['object_id'],
            'image_path': image_path,
            'ra': row.get('ra'),
            'dec': row.get('dec')
        }
        
    except Exception as e:
        logger.error(f"Error processing galaxy {row.get('object_id', 'unknown')}: {str(e)}")
        return None


async def upload_files_batch(
    client: AsyncOpenAI, 
    file_paths: List[Path],
    purpose: str = "vision",
    max_concurrent: int = PARALLEL_UPLOADS
) -> List[Tuple[str, str]]:
    """Upload multiple files with limited concurrency.
    
    Returns list of (identifier, file_id) tuples.
    """
    sem = asyncio.Semaphore(max_concurrent)
    
    async def upload_one(path: Path) -> Tuple[str, str]:
        async with sem:
            try:
                with open(path, "rb") as f:
                    response = await client.files.create(file=f, purpose=purpose)
                # Extract identifier from filename (object_id)
                identifier = path.stem.split('_')[0]
                return identifier, response.id
            except Exception as e:
                logger.error(f"Failed to upload {path}: {e}")
                return None, None
    
    results = await asyncio.gather(*[upload_one(p) for p in file_paths])
    return [(id, fid) for id, fid in results if id is not None and fid is not None]


async def wait_for_file_processing(
    client: AsyncOpenAI,
    file_ids: List[str],
    check_interval: int = 2
):
    """Wait for all uploaded files to be processed."""
    start_time = time.time()
    logger.info(f"Waiting for {len(file_ids)} files to be processed by OpenAI...")
    
    check_count = 0
    with tqdm(total=len(file_ids), desc="Files processing", unit="files") as pbar:
        processed_count = 0
        
        while True:
            check_start = time.time()
            all_processed = True
            pending_count = 0
            
            for i, fid in enumerate(file_ids):
                try:
                    file_obj = await client.files.retrieve(fid)
                    if hasattr(file_obj, 'status') and file_obj.status != 'processed':
                        all_processed = False
                        pending_count += 1
                        
                    # Log progress every 10 files
                    if (i + 1) % 10 == 0:
                        logger.debug(f"  Checked {i+1}/{len(file_ids)} files...")
                        
                except Exception as e:
                    logger.error(f"  Error checking file {fid}: {e}")
            
            check_time = time.time() - check_start
            check_count += 1
            
            # Update progress bar
            new_processed_count = len(file_ids) - pending_count
            if new_processed_count > processed_count:
                pbar.update(new_processed_count - processed_count)
                processed_count = new_processed_count
            
            if not all_processed:
                elapsed = time.time() - start_time
                pbar.set_postfix({"pending": pending_count, "elapsed": f"{elapsed:.1f}s"})
                pbar.refresh()  # Force refresh to ensure update is displayed
                logger.debug(f"  Check #{check_count} ({check_time:.1f}s): {processed_count}/{len(file_ids)} files processed")
            
            if all_processed:
                total_time = time.time() - start_time
                logger.info(f"All {len(file_ids)} files processed in {total_time:.1f}s ({total_time/60:.1f} min)")
                break
                
            await asyncio.sleep(check_interval)


class OpenAIBatchProcessor:
    """Handles OpenAI Batch API operations for processing at scale."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.sync_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def upload_file(self, file_path: Path, purpose: str = "batch") -> str:
        """Upload a file to OpenAI and return file ID."""
        with open(file_path, "rb") as f:
            response = await self.client.files.create(file=f, purpose=purpose)
        return response.id
    
    
    def check_batch_status(self, batch_id: str) -> Optional[Any]:
        """Check the status of a batch job."""
        try:
            return self.sync_client.batches.retrieve(batch_id)
        except Exception as e:
            logger.error(f"Failed to retrieve batch {batch_id}: {e}")
            return None
    
    def download_batch_results(self, output_file_id: str, output_path: Path):
        """Download batch results to a file."""
        file_content = self.sync_client.files.content(output_file_id)
        with open(output_path, "w") as f:
            f.write(file_content.text)
    
    
    
    @staticmethod
    def create_batch_jsonl(
        requests: List[Dict[str, Any]], 
        output_path: Path
    ):
        """Write batch requests to JSONL file."""
        with open(output_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")


class OpenAIEmbeddingGenerator:
    """Handles OpenAI embedding generation with rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.tokens_per_minute = TOKENS_PER_MINUTE
        self.rate_limit_buffer = RATE_LIMIT_BUFFER
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None
    ) -> Tuple[List[List[float]], int]:
        """Generate embeddings for a batch of texts."""
        # Convert bytes to strings if needed
        processed_texts = []
        for text in texts:
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            elif text is None or text == "":
                text = " "  # Use a single space instead of empty string
                print(f"Empty text found, replacing with single space.")
            processed_texts.append(str(text))
        
        # Call OpenAI API
        params = {
            "input": processed_texts,
            "model": model
        }
        if dimensions:
            params["dimensions"] = dimensions
        
        response = self.client.embeddings.create(**params)
        
        # Extract embeddings and token usage
        embeddings = [embedding.embedding for embedding in response.data]
        total_tokens = response.usage.total_tokens
        
        return embeddings, total_tokens
    
    def process_embeddings_with_rate_limit(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
        desc: str = "Generating embeddings"
    ) -> Tuple[List[List[float]], int]:
        """Process embeddings with rate limiting and progress tracking."""
        all_embeddings = []
        total_tokens = 0
        window_start_time = time.time()
        tokens_in_window = 0
        
        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        with tqdm(total=len(texts), desc=desc) as pbar:
            for batch_idx, batch in enumerate(batches):
                # Check rate limit window
                current_time = time.time()
                elapsed = current_time - window_start_time
                
                # Reset window if 60 seconds have passed
                if elapsed >= 60:
                    window_start_time = current_time
                    tokens_in_window = 0
                
                # Estimate tokens for this batch
                estimated_tokens = len(batch) * 300  # Rough estimate
                
                # Check if we would exceed rate limit
                if tokens_in_window + estimated_tokens > self.tokens_per_minute * self.rate_limit_buffer:
                    sleep_time = 60 - elapsed
                    logger.info(f"Approaching rate limit, sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    window_start_time = time.time()
                    tokens_in_window = 0
                
                # Process batch with retries
                retry_count = 0
                while retry_count < MAX_RETRIES:
                    try:
                        embeddings, tokens = self.generate_embeddings_batch(
                            batch, model, dimensions
                        )
                        all_embeddings.extend(embeddings)
                        total_tokens += tokens
                        tokens_in_window += tokens
                        pbar.update(len(embeddings))
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"Batch {batch_idx} failed (retry {retry_count}/{MAX_RETRIES}): {e}")
                            time.sleep(2 ** retry_count)  # Exponential backoff
                        else:
                            logger.error(f"Batch {batch_idx} failed after {MAX_RETRIES} retries: {e}")
                            # Fill with zeros for failed batch
                            embedding_dim = 3072 if dimensions is None else dimensions
                            all_embeddings.extend([
                                np.zeros(embedding_dim).tolist() for _ in range(len(batch))
                            ])
                            break
        
        return all_embeddings, total_tokens


def save_embeddings_to_hdf5(
    df: pd.DataFrame, 
    embeddings: list, 
    output_path: Path,
    model_name: str = "text-embedding-3-large",
    additional_datasets: Optional[Dict[str, np.ndarray]] = None
):
    """
    Save embeddings to HDF5 with all relevant data.
    
    Args:
        df: DataFrame with at least 'object_id' and 'description' columns
        embeddings: List of embeddings
        output_path: Path to save HDF5 file
        model_name: Name of embedding model used
        additional_datasets: Optional dict of additional datasets to save
    """
    with h5py.File(output_path, 'w') as f:
        # Save object IDs
        dt_str = h5py.special_dtype(vlen=str)
        object_ids = [str(oid) for oid in df['object_id'].values]
        f.create_dataset('object_id', data=object_ids, dtype=dt_str)
        
        # Save descriptions
        descriptions = [str(desc) for desc in df['description'].values]
        f.create_dataset('description', data=descriptions, dtype=dt_str)
        
        # Save embeddings
        embeddings_array = np.array(embeddings, dtype=np.float32)
        f.create_dataset('text_embedding', data=embeddings_array)
        
        # Save embedding model information
        model_names = [model_name] * len(df)
        f.create_dataset('embedding_model', data=model_names, dtype=dt_str)
        
        # Save other metadata from DataFrame if present
        optional_cols = ['model_name', 'model_id', 'prompt', 'input_tokens', 'output_tokens', 'llm_cost']
        for col in optional_cols:
            if col in df.columns:
                if col in ['input_tokens', 'output_tokens']:
                    f.create_dataset(col, data=df[col].values.astype(np.int64))
                elif col == 'llm_cost':
                    f.create_dataset(col, data=df[col].values.astype(np.float64))
                else:
                    f.create_dataset(col, data=[str(v) for v in df[col].values], dtype=dt_str)
        
        # Save any additional datasets
        if additional_datasets:
            for name, data in additional_datasets.items():
                if isinstance(data[0], str):
                    f.create_dataset(name, data=data, dtype=dt_str)
                else:
                    f.create_dataset(name, data=data)
        
        # Add metadata attributes
        f.attrs['created_date'] = datetime.now().isoformat()
        f.attrs['n_embeddings'] = len(embeddings)
        f.attrs['embedding_model'] = model_name
        f.attrs['embedding_dim'] = embeddings_array.shape[1]


def process_single_result_jsonl(
    results_jsonl_path: Path, 
    model_config: dict,
    prompt_text: str,
    additional_metadata: Optional[Dict] = None
) -> pd.DataFrame:
    """Process a single results JSONL file and return DataFrame."""
    results = []
    
    # Get model config for cost calculation
    input_price = model_config.get('input_price', 0) / 1_000_000  # Price per token
    output_price = model_config.get('output_price', 0) / 1_000_000
    
    # Read results from JSONL
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            result_data = json.loads(line.strip())
            
            # Extract the response
            if result_data['response']['status_code'] == 200:
                custom_id = result_data['custom_id']
                body = result_data['response']['body']
                
                # Extract text from the Responses API format
                # Try to find non-empty text in output messages
                output_text = ""
                for output_msg in body.get('output', []):
                    if 'content' in output_msg:
                        for content in output_msg['content']:
                            if 'text' in content and content['text'].strip():
                                output_text = content['text']
                                break
                    if output_text:
                        break
                
                # Skip if description is empty
                if not output_text or not output_text.strip():
                    logger.warning(f"Skipping empty description for object {custom_id}")
                    continue
                
                # Extract token usage
                usage = body.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                
                # Calculate cost (batch pricing is 50% discount)
                llm_cost = (input_tokens * input_price + output_tokens * output_price) / 2
                
                # Build result dictionary
                result = {
                    'object_id': custom_id,
                    'description': output_text,
                    'model_name': model_config['formatted_name'],
                    'model_id': model_config['id'],
                    'prompt': prompt_text,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'llm_cost': llm_cost
                }
                
                # Add any additional metadata
                if additional_metadata:
                    result.update(additional_metadata)
                
                results.append(result)
    
    return pd.DataFrame(results)


def setup_batch_logging(log_prefix: str, timestamp: str) -> logging.Logger:
    """
    Set up logging for batch processing scripts.
    
    Args:
        log_prefix: Prefix for log file name (e.g., 'multi_batch_generate')
        timestamp: Timestamp string for log file
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = Path("data/logs") / f'{log_prefix}_{timestamp}.log'
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
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
    
    # Get logger for the calling module
    logger = logging.getLogger('__main__')
    logger.info(f"Log file: {log_file}")
    
    # Disable verbose HTTP request logging from httpx (used by OpenAI)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger


async def create_batch_job_with_info(
    processor: OpenAIBatchProcessor,
    batch_jsonl_path: Path,
    batch_num: int,
    total_batches: int,
    timestamp: str,
    job_prefix: str,
    model_config: dict,
    metadata: dict,
    batch_folder: Path
) -> dict:
    """
    Create a batch job and save its info file.
    
    Args:
        processor: OpenAI batch processor instance
        batch_jsonl_path: Path to batch input JSONL file
        batch_num: Batch number (1-indexed)
        total_batches: Total number of batches
        timestamp: Timestamp for the run
        job_prefix: Prefix for job name (e.g., 'galaxy_descriptions')
        model_config: Model configuration dict
        metadata: Additional metadata for the batch
        batch_folder: Folder to save batch info
        
    Returns:
        Dict with batch info including path to saved info file
    """
    # Upload batch input file
    logger.info(f"Uploading batch input file {batch_num}/{total_batches}...")
    input_file_id = await processor.upload_file(batch_jsonl_path, purpose="batch")
    logger.info(f"Batch {batch_num} input file_id: {input_file_id}")
    
    # Prepare batch metadata
    batch_metadata = {
        "job": f"{job_prefix}_{timestamp}_batch_{batch_num}",
        "model": model_config['formatted_name'],
        "batch_number": str(batch_num),
        "total_batches": str(total_batches),
        **metadata
    }
    
    # Create the batch job
    logger.info(f"Creating batch job {batch_num}/{total_batches}...")
    batch = await processor.client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=batch_metadata
    )
    
    # Count requests in the file
    with open(batch_jsonl_path, 'r') as f:
        num_requests = sum(1 for _ in f)
    
    batch_info = {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "metadata": batch_metadata,
        "batch_number": batch_num,
        "total_requests": num_requests
    }
    
    # Save individual batch info
    batch_info_path = batch_folder / f'batch_info_{timestamp}_part{batch_num}.json'
    with open(batch_info_path, 'w') as f:
        json.dump({
            **batch_info,
            "timestamp": timestamp,
            "model": model_config,
            "batch_jsonl": str(batch_jsonl_path),
            **metadata
        }, f, indent=2)
    
    logger.info(f"Batch {batch_num} created - ID: {batch_info['batch_id']}")
    
    return {
        "batch_info_path": str(batch_info_path),
        "batch_id": batch_info["batch_id"],
        "batch_number": batch_num,
        "total_requests": num_requests
    }


def save_master_batch_info(
    batch_folder: Path,
    timestamp: str,
    model_config: dict,
    total_requests: int,
    batch_infos: List[dict],
    additional_info: dict
) -> Path:
    """
    Save master batch info file that references all sub-batches.
    
    Args:
        batch_folder: Folder containing batch files
        timestamp: Timestamp for the run
        model_config: Model configuration
        total_requests: Total number of requests across all batches
        batch_infos: List of batch info dicts from create_batch_job_with_info
        additional_info: Additional information to include (e.g., prompt_file, input_file)
        
    Returns:
        Path to master batch info file
    """
    master_info_path = batch_folder / f'batch_info_{timestamp}_master.json'
    
    with open(master_info_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model": model_config,
            "total_requests": total_requests,
            "num_batches": len(batch_infos),
            "batch_folder": str(batch_folder),
            "batches": batch_infos,
            **additional_info
        }, f, indent=2)
    
    logger.info(f"\n✅ Batch processing complete!")
    logger.info(f"Created {len(batch_infos)} batch job(s)")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Batch folder: {batch_folder}")
    logger.info(f"Master batch info: {master_info_path}")
    logger.info(f"\nTo check status and download results, run:")
    logger.info(f"  uv run src/check_batch.py {master_info_path}")
    
    return master_info_path


async def split_and_create_batches(
    processor: OpenAIBatchProcessor,
    all_requests: List[dict],
    batch_folder: Path,
    timestamp: str,
    job_prefix: str,
    model_config: dict,
    build_jsonl_func: Callable,
    metadata: dict
) -> List[dict]:
    """
    Split requests into multiple batches and create batch jobs.
    
    Args:
        processor: OpenAI batch processor
        all_requests: All requests to process
        batch_folder: Folder to save batch files
        timestamp: Timestamp for the run
        job_prefix: Prefix for job names
        model_config: Model configuration
        build_jsonl_func: Function to build JSONL from requests
        metadata: Metadata for batches
        
    Returns:
        List of batch info dicts
    """
    num_batch_jobs = (len(all_requests) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
    logger.info(f"Will create {num_batch_jobs} batch job(s) (max {MAX_BATCH_REQUESTS:,} requests per batch)")
    
    all_batch_infos = []
    
    for batch_num in range(num_batch_jobs):
        start_idx = batch_num * MAX_BATCH_REQUESTS
        end_idx = min((batch_num + 1) * MAX_BATCH_REQUESTS, len(all_requests))
        batch_requests = all_requests[start_idx:end_idx]
        
        if batch_requests:
            # Create batch JSONL file
            batch_jsonl_path = batch_folder / f'batch_input_{timestamp}_part{batch_num+1}.jsonl'
            build_jsonl_func(batch_requests, batch_jsonl_path)
            
            # Create batch job
            batch_info = await create_batch_job_with_info(
                processor, batch_jsonl_path, batch_num + 1, num_batch_jobs,
                timestamp, job_prefix, model_config, metadata, batch_folder
            )
            
            all_batch_infos.append(batch_info)
    
    return all_batch_infos