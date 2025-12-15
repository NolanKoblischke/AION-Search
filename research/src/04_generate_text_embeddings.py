#!/usr/bin/env python3
"""
Generate text embeddings for galaxy descriptions using OpenAI's embedding models.

This script reads galaxy descriptions from HDF5 files and generates embeddings.
It supports both standard description files and augmented files with additional summaries.

Input formats supported:
1. Standard: HDF5 with 'description', 'object_id', 'ra', 'dec' columns
2. Augmented: HDF5 with additional 'summaries' column containing JSON-serialized lists

The script automatically detects which format is provided and processes accordingly.
"""

import h5py
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import time
import json
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
# Import our utilities
sys.path.append(str(Path(__file__).parent))
from utils.openai_utils import OpenAIEmbeddingGenerator

load_dotenv()

logger = logging.getLogger(__name__)

# Default batch size for processing
BATCH_SIZE = 100

def process_embeddings_with_generator(descriptions, batch_size=BATCH_SIZE, model="text-embedding-3-large", 
                                    dimensions=None, desc="Generating embeddings"):
    """Process embeddings using the centralized embedding generator."""
    generator = OpenAIEmbeddingGenerator()
    return generator.process_embeddings_with_rate_limit(
        texts=descriptions,
        batch_size=batch_size,
        model=model,
        dimensions=dimensions,
        desc=desc
    )

def save_embeddings_to_hdf5(input_path, output_path, embeddings_by_galaxy, model_name="text-embedding-3-large", 
                           galaxy_data_path="data/processed/galaxy_data.hdf5", text_metadata=None, 
                           ra_values=None, dec_values=None, augmented_descriptions_by_idx=None):
    """Save embeddings to a new HDF5 file with all original data plus text_embedding column."""
    logger.info(f"Reading original data from {input_path}")
    
    with h5py.File(input_path, 'r') as input_file:
        # Read all datasets
        data = {}
        for key in input_file.keys():
            dataset = input_file[key]
            data[key] = dataset[:]
            logger.info(f"Read {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # Ensure we have ra and dec
        if 'ra' not in data and ra_values is not None:
            data['ra'] = ra_values
        if 'dec' not in data and dec_values is not None:
            data['dec'] = dec_values
    
    # For the new structure, we need to flatten the embeddings
    all_embeddings = []
    for galaxy_idx in sorted(embeddings_by_galaxy.keys()):
        for emb in embeddings_by_galaxy[galaxy_idx]['embeddings']:
            all_embeddings.append(emb)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Embeddings array shape: {embeddings_array.shape}")
    
    # Add embeddings to data
    data['text_embedding'] = embeddings_array
    data['embedding_model'] = [model_name.encode('utf-8')] * len(embeddings_array)
    
    # Add galaxy_data_index - this is required for downstream processing
    if not Path(galaxy_data_path).exists():
        logger.error(f"CRITICAL: galaxy_data.hdf5 not found at {galaxy_data_path}")
        logger.error("This file is required to create galaxy_data_index mapping for downstream alignment.")
        logger.error("Please ensure galaxy_data.hdf5 exists at the expected location.")
        raise FileNotFoundError(f"Required file galaxy_data.hdf5 not found at {galaxy_data_path}")
    
    logger.info(f"Adding galaxy_data_index from {galaxy_data_path}")
    galaxy_data_mapping = {}
    
    with h5py.File(galaxy_data_path, 'r') as f:
        # Require flat structure
        if 'object_id' not in f.keys():
            raise ValueError(f"galaxy_data.hdf5 must have flat structure with 'object_id' at root level. Found keys: {list(f.keys())}")
        
        # Create mapping for flat structure
        object_ids = f['object_id'][:]
        for idx, oid in enumerate(object_ids):
            if isinstance(oid, bytes):
                oid = oid.decode('utf-8')
            galaxy_data_mapping[oid] = idx
    
    logger.info(f"Created mapping for {len(galaxy_data_mapping)} objects from galaxy_data.hdf5")
    
    # Map text embedding object IDs to galaxy data indices
    text_object_ids = data['object_id']
    galaxy_data_indices = []
    missing_count = 0
    
    for oid in text_object_ids:
        if isinstance(oid, bytes):
            oid = oid.decode('utf-8')
        
        if oid in galaxy_data_mapping:
            galaxy_data_indices.append(galaxy_data_mapping[oid])
        else:
            galaxy_data_indices.append(-1)  # Sentinel for not found
            missing_count += 1
            if missing_count <= 5:  # Log first few missing
                logger.warning(f"Object ID {oid} not found in galaxy_data.hdf5")
    
    if missing_count > 0:
        logger.warning(f"Total {missing_count} object IDs not found in galaxy_data.hdf5")
    
    data['galaxy_data_index'] = np.array(galaxy_data_indices, dtype=np.int64)
    logger.info(f"Added galaxy_data_index for {len(galaxy_data_indices)} objects ({len(galaxy_data_indices) - missing_count} valid)")
    
    # Save augmented descriptions if provided
    if augmented_descriptions_by_idx is not None:
        logger.info("Preparing augmented descriptions for saving...")
        # Create a structured dataset for augmented descriptions
        # We'll store them as a JSON string for each galaxy
        augmented_json_list = []
        for i in range(len(data['object_id'])):
            if i in augmented_descriptions_by_idx:
                augmented_json_list.append(json.dumps(augmented_descriptions_by_idx[i]))
            else:
                augmented_json_list.append(json.dumps([]))  # Empty list for galaxies without augmented
        data['augmented_descriptions'] = augmented_json_list
        logger.info(f"Added augmented descriptions for {len([x for x in augmented_json_list if x != '[]'])} galaxies")
    
    # Save to new HDF5 file
    logger.info(f"Saving to {output_path}")
    with h5py.File(output_path, 'w') as output_file:
        # Add attribute to indicate new structure
        if text_metadata:
            output_file.attrs['embeddings_by_galaxy'] = True
            # Save metadata as JSON string
            output_file.create_dataset('text_metadata', data=json.dumps(text_metadata).encode('utf-8'))
        
        for key, values in data.items():
            if key == 'text_embedding':
                # Save embeddings as float32
                output_file.create_dataset(key, data=values, dtype=np.float32)
            elif key == 'embedding_model':
                # Save embedding model name as string
                dt = h5py.special_dtype(vlen=str)
                string_data = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in values]
                output_file.create_dataset(key, data=string_data, dtype=dt)
            elif key in ['description', 'prompt', 'model_name', 'model_id', 'prompt_filename', 'plotstyle_filename', 'group', 'augmented_descriptions']:
                # String columns - use variable-length strings
                dt = h5py.special_dtype(vlen=str)
                if isinstance(values[0], bytes):
                    string_data = [item.decode('utf-8') for item in values]
                else:
                    string_data = [str(item) for item in values]
                output_file.create_dataset(key, data=string_data, dtype=dt)
            elif key in ['object_id']:
                # String IDs - use variable-length strings
                dt = h5py.special_dtype(vlen=str)
                if isinstance(values[0], bytes):
                    string_data = [item.decode('utf-8') for item in values]
                else:
                    string_data = [str(item) for item in values]
                output_file.create_dataset(key, data=string_data, dtype=dt)
            elif key in ['ra', 'dec', 'llm_cost', 'llm_time']:
                # Floating point numbers
                output_file.create_dataset(key, data=values.astype(np.float64))
            elif key in ['healpix', 'input_tokens', 'output_tokens', 'galaxy_data_index']:
                # Integers
                output_file.create_dataset(key, data=values.astype(np.int64))
            else:
                # Default handling - try to preserve original dtype
                try:
                    output_file.create_dataset(key, data=values)
                except Exception as e:
                    logger.warning(f"Error saving {key} with original dtype, using string fallback: {e}")
                    dt = h5py.special_dtype(vlen=str)
                    string_data = [str(item) for item in values]
                    output_file.create_dataset(key, data=string_data, dtype=dt)
    
    n_galaxies = len(embeddings_by_galaxy)
    n_embeddings = len(all_embeddings)
    if n_embeddings > n_galaxies:
        logger.info(f"Successfully saved {n_embeddings} embeddings for {n_galaxies} galaxies to {output_path}")
    else:
        logger.info(f"Successfully saved {n_galaxies} embeddings to {output_path}")

def inspect_output_file(output_path):
    """Inspect the output HDF5 file to verify structure."""
    logger.info(f"\n=== Inspecting output file: {output_path} ===")
    
    with h5py.File(output_path, 'r') as f:
        datasets = list(f.keys())
        logger.info(f"Datasets: {datasets}")
        
        for key in datasets:
            dataset = f[key]
            logger.info(f"{key}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            # Show sample for text_embedding
            if key == 'text_embedding':
                logger.info(f"  Embedding dimensions: {dataset.shape[1] if len(dataset.shape) > 1 else 'scalar'}")
                logger.info(f"  First embedding (first 5 dims): {dataset[0][:5] if len(dataset.shape) > 1 else dataset[0]}")
            elif key == 'description':
                logger.info(f"  Sample description length: {len(dataset[0])} characters")
            elif key == 'embedding_model':
                logger.info(f"  Embedding model: {dataset[0]}")

def main():
    parser = argparse.ArgumentParser(description='Generate text embeddings for galaxy descriptions')
    parser.add_argument('--input', type=str, 
                       default='data/processed/galaxy_descriptions.hdf5',
                       help='Input HDF5 file with galaxy descriptions (can include augmented summaries)')
    parser.add_argument('--model', type=str, default='text-embedding-3-large',
                       help='OpenAI embedding model to use')
    parser.add_argument('--dimensions', type=int, default=None,
                       help='Embedding dimensions (default: max for model)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for processing')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--galaxy-data', type=str, default='data/processed/galaxy_data.hdf5',
                       help='Path to galaxy_data.hdf5 file for creating index mapping')
    
    args = parser.parse_args()
    
    # Create output directory and logs directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path("data/logs") / f'generate_text_embeddings_{timestamp}.log'
    
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
    
    logger.info(f"Log file: {log_file}")
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Validate galaxy_data.hdf5 exists (required for downstream processing)
    galaxy_data_path = Path(args.galaxy_data)
    if not galaxy_data_path.exists():
        logger.error(f"CRITICAL: galaxy_data.hdf5 not found at {galaxy_data_path}")
        logger.error("This file is required to create galaxy_data_index mapping for downstream alignment.")
        logger.error("Please ensure galaxy_data.hdf5 exists at the expected location.")
        logger.error("You can specify a different path using --galaxy-data argument.")
        sys.exit(1)
    
    # Generate output filename (standardized name)
    output_filename = "galaxy_text_embeddings.hdf5"
    output_path = output_dir / output_filename
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Galaxy data file: {galaxy_data_path}")
    logger.info(f"Model: {args.model}")
    if args.dimensions:
        logger.info(f"Dimensions: {args.dimensions}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Load descriptions from HDF5
    logger.info("Loading data from HDF5 file...")
    descriptions = []
    object_ids = []
    augmented_summaries = []
    has_augmented = False
    ra_values = []
    dec_values = []
    
    with h5py.File(input_path, 'r') as f:
        if 'description' not in f:
            logger.error("No 'description' dataset found in input file")
            sys.exit(1)
        
        descriptions = f['description'][:]
        object_ids = f['object_id'][:]
        
        # Load ra and dec if available
        if 'ra' in f:
            ra_values = f['ra'][:]
        else:
            ra_values = np.zeros(len(descriptions))
            
        if 'dec' in f:
            dec_values = f['dec'][:]
        else:
            dec_values = np.zeros(len(descriptions))
        
        # Check for augmented summaries
        if 'summaries' in f:
            has_augmented = True
            augmented_summaries = f['summaries'][:]
            logger.info(f"Found augmented summaries column")
        
        logger.info(f"Loaded {len(descriptions)} descriptions")
        
        # Show sample description
        sample_desc = descriptions[0]
        if isinstance(sample_desc, bytes):
            sample_desc = sample_desc.decode('utf-8')
        logger.info(f"Sample description (first 200 chars): {sample_desc[:200]}...")
    
    # Convert to strings
    descriptions = [d.decode('utf-8') if isinstance(d, bytes) else d for d in descriptions]
    object_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else oid for oid in object_ids]
    
    # Parse augmented summaries if present
    augmented_descriptions_by_idx = {}
    if has_augmented:
        logger.info("Parsing augmented summaries from HDF5...")
        for i, summaries_json in enumerate(augmented_summaries):
            if isinstance(summaries_json, bytes):
                summaries_json = summaries_json.decode('utf-8')
            
            try:
                summaries_list = json.loads(summaries_json)
                if summaries_list:  # Only add if non-empty
                    augmented_descriptions_by_idx[i] = summaries_list
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse summaries for index {i}: {e}")
        
        logger.info(f"Loaded augmented descriptions for {len(augmented_descriptions_by_idx)} galaxies")
    
    # Prepare all texts for embedding
    all_texts = []
    text_metadata = []  # Track which text belongs to which galaxy
    
    for i, (desc, oid) in enumerate(zip(descriptions, object_ids)):
        # Add original description
        all_texts.append(desc)
        text_metadata.append({"galaxy_idx": i, "object_id": oid, "text_type": "original", "text_idx": 0})
        
        # Add augmented descriptions if available
        if i in augmented_descriptions_by_idx:
            for j, aug_desc in enumerate(augmented_descriptions_by_idx[i]):
                all_texts.append(aug_desc)
                text_metadata.append({"galaxy_idx": i, "object_id": oid, "text_type": "augmented", "text_idx": j+1})
    
    logger.info(f"Total texts to embed: {len(all_texts)} ({len(descriptions)} original + {len(all_texts)-len(descriptions)} augmented)")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    start_time = time.time()
    
    try:
        embeddings, total_tokens = process_embeddings_with_generator(
            all_texts, 
            batch_size=args.batch_size,
            model=args.model,
            dimensions=args.dimensions
        )
        
        embedding_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
        logger.info(f"Total tokens used: {total_tokens:,}")
        # Cost estimation: roughly $0.00013 per 1K tokens for text-embedding-3-large
        estimated_cost = (total_tokens / 1000) * 0.00013
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        sys.exit(1)
    
    # Reorganize embeddings by galaxy
    embeddings_by_galaxy = {}
    for i, metadata in enumerate(text_metadata):
        galaxy_idx = metadata["galaxy_idx"]
        if galaxy_idx not in embeddings_by_galaxy:
            embeddings_by_galaxy[galaxy_idx] = {
                "object_id": metadata["object_id"],
                "embeddings": [],
                "text_types": []
            }
        embeddings_by_galaxy[galaxy_idx]["embeddings"].append(embeddings[i])
        embeddings_by_galaxy[galaxy_idx]["text_types"].append(metadata["text_type"])
    
    # Save embeddings to new HDF5 file with augmented structure
    logger.info("Saving embeddings to HDF5 file...")
    save_embeddings_to_hdf5(input_path, output_path, embeddings_by_galaxy, args.model, 
                          galaxy_data_path=args.galaxy_data,
                          text_metadata=text_metadata, ra_values=ra_values, dec_values=dec_values,
                          augmented_descriptions_by_idx=augmented_descriptions_by_idx)
    
    # Inspect output file
    inspect_output_file(output_path)
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Embeddings: {len(embeddings)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Estimated cost: ${estimated_cost:.4f}")
    logger.info(f"Processing time: {embedding_time:.2f} seconds")

if __name__ == "__main__":
    main()