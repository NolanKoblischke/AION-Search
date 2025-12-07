"""
General evaluation utilities for semantic search evaluation.
Provides a unified framework for different evaluation types (lens, merger, spiral).
"""

import numpy as np
import h5py
import torch
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
from openai import OpenAI
import os
import json
from datetime import datetime
from tqdm import tqdm
import time
import pickle
import hashlib
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# AION imports
from aion.modalities import LegacySurveyImage, HSCImage
from aion.codecs import CodecManager
from aion.model import AION


# AION model embedding dimensions
AION_MODEL_DIMS = {
    'aion-base': 768,
    'aion-large': 1024,
    'aion-xlarge': 2048
}


@dataclass
class EvalConfig:
    """Configuration for a specific evaluation type."""
    name: str  # e.g., 'lens', 'merger', 'spiral'
    embeddings_path: str  # Path to HDF5 file with embeddings
    grade_converter: Callable[[Union[str, float]], float]  # Convert grade to weight
    query_text: str  # Query text for retrieval


# Grade converter functions
def lens_grade_converter(grade: str) -> float:
    """Convert lens grade (A, B, C, N) to weight."""
    if grade in ['A', 'B', 'C']:
        return 1.0
    return 0.0


def merger_grade_converter(vote_fraction: Union[str, float]) -> float:
    """Convert merger vote fraction to weight."""
    # Handle both string and float inputs (0.0 to 1.0)
    return float(vote_fraction)


def spiral_grade_converter(vote_fraction: Union[str, float]) -> float:
    """Convert spiral vote fraction to weight."""
    # Handle both string and float inputs (0.0 to 1.0)
    return float(vote_fraction)


# Evaluation configurations
EVAL_CONFIGS = {
    'lens_legacy': EvalConfig(
        name='lens_legacy',
        embeddings_path='data/evals/lens/lens_aion_embeddings_legacy.hdf5',
        grade_converter=lens_grade_converter,
        query_text='gravitational lens'
    ),
    'lens_hsc': EvalConfig(
        name='lens_hsc',
        embeddings_path='data/evals/lens/lens_aion_embeddings_hsc.hdf5',
        grade_converter=lens_grade_converter,
        query_text='gravitational lens'
    ),
    'mergers': EvalConfig(
        name='mergers',
        embeddings_path='data/evals/mergers_spirals/mergers_aion_embeddings.hdf5',
        grade_converter=merger_grade_converter,
        query_text='merging'
    ),
    'spirals': EvalConfig(
        name='spirals',
        embeddings_path='data/evals/mergers_spirals/spirals_aion_embeddings.hdf5',
        grade_converter=spiral_grade_converter,
        query_text='visible spiral arms'
    )
}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Disable verbose HTTP request logging from httpx (used by OpenAI)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def load_eval_data_from_hdf5(embeddings_path: str) -> Tuple[List[str], Dict[str, Union[str, float]]]:
    """
    Load evaluation data directly from HDF5 file.
    
    Returns:
        - List of object IDs
        - Mapping of object_id to grade/score
    """
    with h5py.File(embeddings_path, 'r') as f:
        # Load object IDs
        object_ids = f['object_id'][:]
        if isinstance(object_ids[0], bytes):
            object_ids = [oid.decode('utf-8') for oid in object_ids]
        
        # Load evaluation grades
        eval_grades = f['eval_grade'][:]
        if isinstance(eval_grades[0], bytes):
            eval_grades = [grade.decode('utf-8') for grade in eval_grades]
        
        # Create grade mapping
        grade_mapping = dict(zip(object_ids, eval_grades))
        
    return object_ids, grade_mapping


def load_aion_embeddings(
    embeddings_path: str,
    use_mean: bool = True,
    aion_model: str = 'aion-base'
) -> Tuple[List[str], np.ndarray, Dict[str, Union[str, float]]]:
    """
    Load pre-computed AION embeddings and evaluation data from HDF5 file.
    
    Args:
        embeddings_path: Path to HDF5 file with embeddings
        use_mean: Whether to use mean-pooled embeddings
        aion_model: AION model size ('aion-base', 'aion-large', 'aion-xlarge')
    
    Returns:
        - List of object IDs
        - Embeddings array
        - eval grades
    """
    with h5py.File(embeddings_path, 'r') as f:
        # Load object IDs
        object_ids = f['object_id'][:]
        if isinstance(object_ids[0], bytes):
            object_ids = [oid.decode('utf-8') for oid in object_ids]
        
        # Load embeddings
        embedding_key = 'aion_embedding_mean' if use_mean else 'aion_embedding_full'
        if embedding_key not in f:
            raise KeyError(f"Could not find embedding key '{embedding_key}' in HDF5 file")
        
        embeddings = f[embedding_key][:]
        
        # Verify embedding dimension matches AION model
        expected_dim = AION_MODEL_DIMS[aion_model]
        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(f"Embedding dimension {actual_dim} does not match {aion_model} ({expected_dim})")
        
        # Transpose full embeddings if needed
        if not use_mean and len(embeddings.shape) == 3:
            embeddings = embeddings.transpose(0, 2, 1)
        
        # Load evaluation grades
        eval_grades = f['eval_grade'][:]
        if isinstance(eval_grades[0], bytes):
            eval_grades = [grade.decode('utf-8') for grade in eval_grades]
    
    return object_ids, embeddings, eval_grades


def generate_query_embedding(
    query_text: str,
    model: str = "text-embedding-3-large",
    cache_dir: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """Generate embedding for query text using OpenAI with optional caching.
    
    Args:
        query_text: Text to generate embedding for
        model: OpenAI embedding model to use
        cache_dir: Optional directory to cache embeddings. If None, no caching is performed.
    
    Returns:
        Embedding array
    """
    # Create cache key from query text and model
    cache_key = hashlib.sha256(f"{query_text}_{model}".encode()).hexdigest()
    
    # Check cache if directory provided
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            logger = logging.getLogger(__name__)
            logger.info(f"Loading cached embedding for query: '{query_text}'")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Generate embedding if not cached
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=query_text,
        model=model
    )
    
    embedding = np.array(response.data[0].embedding)
    
    # Save to cache if directory provided
    if cache_dir is not None:
        logger = logging.getLogger(__name__)
        logger.info(f"Caching embedding for query: '{query_text}'")
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
    
    return embedding

def compute_ndcg_at_k(relevance_scores: np.ndarray, k: int, all_relevance_scores: np.ndarray) -> float:
    """
    Compute NDCG@k for a single query using simplified formula.
    
    Args:
        relevance_scores: Relevance scores (can be binary or graded)
        k: Number of top results to consider
        all_relevance_scores: All relevance scores in the dataset for computing ideal DCG
    
    Returns:
        NDCG@k score
    """
    # Take top k results
    relevance_k = relevance_scores[:k]
    
    # Compute DCG@k using simplified formula: relevance / log2(position)
    # Note: positions start from 2 to avoid log2(1) = 0
    positions = np.arange(2, k + 2)
    discounts = np.log2(positions)
    dcg_k = np.sum(relevance_k / discounts)
    
    # Compute IDCG@k using the sorted full dataset
    sorted_all_relevance = np.sort(all_relevance_scores)[::-1][:k]
    ideal_dcg = np.sum(sorted_all_relevance / discounts)
    
    return dcg_k / ideal_dcg if ideal_dcg > 0 else 0.0





def evaluate_retrieval_with_model(
    model,
    eval_name: str,
    device: str = 'cuda',
    use_mean_embeddings: bool = True,
    batch_size: int = 512,
    k_values: List[int] = [10, 1000],
    aion_model: str = 'aion-base',
    logger: Optional[logging.Logger] = None,
    cached_baselines: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a specific evaluation task.
    
    Args:
        model: Trained model with image_projector and text_projector
        eval_name: Name of evaluation ('lens_legacy', 'lens_hsc', 'merger', 'spiral')
        device: Device to use for computation
        use_mean_embeddings: Whether to use mean-pooled embeddings
        batch_size: Batch size for projection
        k_values: List of k values to evaluate
        aion_model: AION model size
        logger: Optional logger instance
        cached_baselines: Optional dict of cached baseline scores per evaluation
    
    Returns:
        Dictionary with evaluation metrics. If baselines were computed (not cached),
        includes '_baseline_cache' key with computed baselines.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get evaluation configuration
    if eval_name not in EVAL_CONFIGS:
        raise ValueError(f"Unknown evaluation: {eval_name}")
    
    eval_config = EVAL_CONFIGS[eval_name]
    
    embeddings_path = eval_config.embeddings_path
    
    # Load AION embeddings and evaluation data
    logger.info(f"Loading {eval_name} evaluation data from {embeddings_path}...")
    object_ids, aion_embeddings, eval_grades = load_aion_embeddings(
        embeddings_path,
        use_mean=use_mean_embeddings,
        aion_model=aion_model
    )
    
    logger.info(f"Loaded {len(object_ids)} total objects")
    
    # Generate query embedding with caching
    # Derive cache directory from embeddings path
    embeddings_dir = Path(eval_config.embeddings_path).parent
    cache_dir = embeddings_dir / "query_embeddings_cache"
    
    logger.info(f"Generating query embedding for: '{eval_config.query_text}'")
    query_embedding = generate_query_embedding(eval_config.query_text, cache_dir=cache_dir)
    
    # Project query through model
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.text_projector(query_tensor)
    query_features = query_features.cpu().numpy()
    
    # Project AION embeddings through model in batches
    all_features = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(aion_embeddings), batch_size):
            batch = aion_embeddings[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            features = model.image_projector(batch_tensor)
            all_features.append(features.cpu().numpy())
    
    image_features = np.vstack(all_features)
    
    # Compute similarities and rank
    similarities = image_features @ query_features.T
    similarities = similarities.squeeze()
    top_indices = np.argsort(similarities)[::-1]
    
    # Compute metrics for each k value
    results = {}
    max_k = max(k_values)
    
    # Get relevance scores for all retrieved objects
    relevance_scores = []
    for idx in top_indices[:max_k]:
        grade = eval_grades[idx]
        weight = eval_config.grade_converter(grade)
        relevance_scores.append(weight)
    
    relevance_scores = np.array(relevance_scores)
    
    # Get all relevance scores in the dataset for baselines
    all_relevance_scores = np.array([eval_config.grade_converter(grade) for grade in eval_grades])
    
    # Track if we need to compute baselines
    need_baselines = cached_baselines is None or eval_name not in cached_baselines
    baseline_cache = {}
    
    # Compute metrics for each k
    for k in k_values:
        if k <= len(relevance_scores):
            # Compute nDCG@k
            ndcg = compute_ndcg_at_k(relevance_scores[:k], k, all_relevance_scores)
            results[f'ndcg@{k}'] = ndcg
            
            # Use cached baselines or compute them
            if need_baselines:
                # Compute random baselines
                from numpy.random import default_rng
                rng = default_rng(42)
                rand_ndcgs = []
                for _ in range(50):
                    rand_idx = rng.permutation(len(all_relevance_scores))[:k]
                    rand_rel = all_relevance_scores[rand_idx]
                    rand_ndcg = compute_ndcg_at_k(rand_rel, k, all_relevance_scores)
                    rand_ndcgs.append(rand_ndcg)
                random_baseline = float(np.median(rand_ndcgs))
                baseline_cache[f'random_ndcg@{k}'] = random_baseline
                
                # Compute ideal baselines
                sorted_relevance = np.sort(all_relevance_scores)[::-1]
                ideal_relevance_k = sorted_relevance[:k]
                ideal_baseline = compute_ndcg_at_k(ideal_relevance_k, k, all_relevance_scores)
                baseline_cache[f'ideal_ndcg@{k}'] = ideal_baseline
                
                results[f'random_ndcg@{k}'] = random_baseline
                results[f'ideal_ndcg@{k}'] = ideal_baseline
            else:
                # Use cached baselines
                results[f'random_ndcg@{k}'] = cached_baselines[eval_name][f'random_ndcg@{k}']
                results[f'ideal_ndcg@{k}'] = cached_baselines[eval_name][f'ideal_ndcg@{k}']
    
    # General statistics
    results['total_objects'] = len(object_ids)
    
    # Log key metrics
    logger.info(f"{eval_name} evaluation results:")
    for k in k_values:
        if f'ndcg@{k}' in results:
            logger.info(f"  nDCG@{k}: {results[f'ndcg@{k}']:.4f}")
    
    # Include baseline cache in results if we computed it
    if need_baselines and baseline_cache:
        results['_baseline_cache'] = baseline_cache
    
    return results




def round_numeric_values(obj, decimals=3):
    """
    Recursively round all numeric values in a nested structure to specified decimal places.
    
    Args:
        obj: The object to process (dict, list, or primitive)
        decimals: Number of decimal places to round to
    
    Returns:
        The object with all numeric values rounded
    """
    if isinstance(obj, dict):
        return {k: round_numeric_values(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_numeric_values(v, decimals) for v in obj]
    elif isinstance(obj, (float, np.float32, np.float64)):
        return round(float(obj), decimals)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    else:
        return obj


def append_eval_summary(
    output_dir: Union[str, Path],
    eval_name: str,
    model_name: str,
    metrics: Dict[str, float],
    run_name: Optional[str] = None,
    additional_info: Optional[Dict] = None
) -> Path:
    """
    Append evaluation results to summary JSONL file.
    
    Args:
        output_dir: Output directory for results
        eval_name: Name of evaluation
        model_name: Name of model
        metrics: Evaluation metrics
        run_name: Optional run name
        additional_info: Optional additional information
    
    Returns:
        Path to summary file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a single eval_summary.jsonl file for all evaluations
    summary_file = output_dir / 'eval_summary.jsonl'
    
    # Round all numeric values in metrics to 3 decimal places
    rounded_metrics = round_numeric_values(metrics)
    
    # Prepare record
    record = {
        'timestamp': datetime.now().isoformat(),
        'eval_name': eval_name,
        'model': model_name,
        'metrics': rounded_metrics
    }
    
    if run_name:
        record['run_name'] = run_name
    
    if additional_info:
        # Also round numeric values in additional_info
        record.update(round_numeric_values(additional_info))
    
    # Append to JSONL file
    with open(summary_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    
    return summary_file


def process_legacy_batch(images, model, codec_manager, device):
    """Process a batch of Legacy images and return embeddings."""
    # Remove NaN padding from 5th channel if present
    if images.shape[1] == 5:
        images = images[:, :4]
    
    # Convert to torch tensor
    image_flux = torch.tensor(images.astype('float32')).to(device)
    
    # Create typed image
    typed_image = LegacySurveyImage(
        flux=image_flux,
        bands=["DES-G", "DES-R", "DES-I", "DES-Z"]
    )
    
    # Encode to tokens
    tokens = codec_manager.encode(typed_image)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.encode(tokens, num_encoder_tokens=576)
    
    # AION returns shape (batch, tokens, dim) = (batch, 576, 768)
    # We want (batch, dim, tokens) = (batch, 768, 576)
    embeddings = embeddings.transpose(1, 2)
    
    return embeddings.detach().cpu().numpy()


def process_hsc_batch(images, model, codec_manager, device):
    """Process a batch of HSC images and return embeddings."""
    # Convert to torch tensor
    image_flux = torch.tensor(images.astype('float32')).to(device)
    
    # Create typed image
    typed_image = HSCImage(
        flux=image_flux,
        bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
    )
    
    # Encode to tokens
    tokens = codec_manager.encode(typed_image)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.encode(tokens, num_encoder_tokens=576)
    
    # AION returns shape (batch, tokens, dim) = (batch, 576, 768)
    # We want (batch, dim, tokens) = (batch, 768, 576)
    embeddings = embeddings.transpose(1, 2)
    
    return embeddings.detach().cpu().numpy()


def generate_aion_embeddings(
    object_ids: List[str],
    ra_values: np.ndarray,
    dec_values: np.ndarray,
    eval_grades: List[Union[str, float]],
    images: np.ndarray,
    survey_name: str,
    aion_model_name: str,
    output_file_path: str,
    batch_size: int = 512,
    device: str = 'cuda',
    logger: Optional[logging.Logger] = None,
    chunk_size: int = 512
) -> str:
    """
    Generate AION embeddings for objects and save to HDF5 file.
    
    Args:
        object_ids: List of object IDs
        ra_values: Array of right ascension values
        dec_values: Array of declination values
        eval_grades: List of evaluation grades/scores
        images: Array of images to process
        survey_name: Survey name ('Legacy' or 'HSC')
        aion_model_name: AION model to use (e.g., 'polymathic-ai/aion-base')
        output_file_path: Path for output HDF5 file
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        logger: Optional logger instance
        chunk_size: HDF5 chunk size
    
    Returns:
        Path to generated HDF5 file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    n_objects = len(object_ids)
    
    # Verify input arrays have same length
    if not (len(object_ids) == len(ra_values) == len(dec_values) == len(eval_grades) == len(images)):
        raise ValueError("All input arrays must have the same length")
    
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Setup CUDA optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA optimizations enabled")
    
    # Load AION model
    logger.info(f"Loading AION model: {aion_model_name}")
    model = AION.from_pretrained(aion_model_name).to(device).eval()
    codec_manager = CodecManager(device=device)
    
    # Get expected embedding dimension
    model_size = aion_model_name.split('-')[-1]  # Extract 'base', 'large', or 'xlarge'
    embedding_dim = AION_MODEL_DIMS.get(f'aion-{model_size}', 768)
    
    # Create output file
    logger.info(f"Creating output file: {output_file_path}")
    with h5py.File(output_file_path, 'w') as f:
        # Object metadata
        f.create_dataset('object_id', 
                        shape=(n_objects,),
                        dtype=h5py.special_dtype(vlen=str))
        
        # Coordinates
        f.create_dataset('ra', shape=(n_objects,), dtype=np.float64)
        f.create_dataset('dec', shape=(n_objects,), dtype=np.float64)
        
        # Evaluation grade
        f.create_dataset('eval_grade',
                        shape=(n_objects,),
                        dtype=h5py.special_dtype(vlen=str))
        
        # Mean embeddings
        f.create_dataset('aion_embedding_mean',
                        shape=(n_objects, embedding_dim),
                        dtype=np.float32,
                        chunks=(min(chunk_size, n_objects), embedding_dim),
                        compression='lzf')
        
        # Full embeddings
        f.create_dataset('aion_embedding_full',
                        shape=(n_objects, embedding_dim, 576),
                        dtype=np.float32,
                        chunks=(min(chunk_size, n_objects), embedding_dim, 576),
                        compression='lzf')
    
    # Process images
    logger.info(f"Processing {n_objects:,} {survey_name} images...")
    
    with h5py.File(output_file_path, 'r+') as hdf5_f:
        output_idx = 0
        
        # Process in batches
        for batch_start in tqdm(range(0, n_objects, batch_size), desc=f"Processing {survey_name} objects"):
            batch_end = min(batch_start + batch_size, n_objects)
            batch_size_actual = batch_end - batch_start
            
            # Get batch data
            batch_object_ids = object_ids[batch_start:batch_end]
            batch_ra = ra_values[batch_start:batch_end]
            batch_dec = dec_values[batch_start:batch_end]
            batch_grades = eval_grades[batch_start:batch_end]
            
            # Get batch images from provided array
            batch_images = images[batch_start:batch_end]
            
            # Process batch based on survey type
            if survey_name.lower() == 'legacy':
                embeddings = process_legacy_batch(batch_images, model, codec_manager, device)
            elif survey_name.lower() == 'hsc':
                embeddings = process_hsc_batch(batch_images, model, codec_manager, device)
            else:
                raise ValueError(f"Unknown survey: {survey_name}")
            
            # Save embeddings
            hdf5_f['aion_embedding_full'][output_idx:output_idx+batch_size_actual] = embeddings
            hdf5_f['aion_embedding_mean'][output_idx:output_idx+batch_size_actual] = embeddings.mean(axis=2)
            
            # Save metadata
            for i in range(batch_size_actual):
                hdf5_f['object_id'][output_idx + i] = str(batch_object_ids[i])
                hdf5_f['ra'][output_idx + i] = batch_ra[i]
                hdf5_f['dec'][output_idx + i] = batch_dec[i]
                hdf5_f['eval_grade'][output_idx + i] = str(batch_grades[i])
            
            output_idx += batch_size_actual
            
            # Log progress every 10 batches
            if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
                elapsed = time.time() - start_time
                rate = output_idx / elapsed if elapsed > 0 else 0
                eta = (n_objects - output_idx) / rate if rate > 0 else 0
                logger.info(f"  Processed {output_idx:,}/{n_objects:,} objects "
                           f"({rate:.1f} objects/sec, ETA: {eta/60:.1f} min)")
            
            # Clear GPU cache periodically
            if device == 'cuda' and (batch_start // batch_size) % 20 == 0:
                torch.cuda.empty_cache()
        
        # Add metadata attributes
        hdf5_f.attrs['created_date'] = datetime.now().isoformat()
        hdf5_f.attrs['n_rows'] = n_objects
        hdf5_f.attrs['aion_model'] = aion_model_name
        hdf5_f.attrs['embedding_dim'] = embedding_dim
        hdf5_f.attrs['n_tokens'] = 576
        hdf5_f.attrs['survey'] = survey_name
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info(f"AION embedding generation complete!")
    logger.info(f"Output file: {output_file_path}")
    logger.info(f"Total objects processed: {n_objects:,}")
    logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
    logger.info(f"Processing rate: {n_objects/elapsed_time:.1f} objects/sec")
    
    return output_file_path