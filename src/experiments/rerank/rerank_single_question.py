"""
Simplified rerank script using only the lens question.
"""

import argparse
import h5py
import numpy as np
import json
import logging
import os
import base64
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import pandas as pd

# Load environment variables
load_dotenv()

# Import utilities
from src.evals.eval_utils import compute_ndcg_at_k, setup_logging, EVAL_CONFIGS
from src.plotting_scripts.default_plot import process_galaxy_image
from src.clip.models.clip_model import GalaxyClipModel

# The single question we care about
LENS_QUESTION = "Does this galaxy image display signs of gravitational lensing? Rank 1-10 where 10 means you are entirely sure there are signs of gravitational lensing and 1 being you are entirely sure there are no signs of gravitational lensing."


def load_models_info(models_file: str = "src/utils/models.jsonl") -> Dict[str, Dict]:
    """Load model pricing information from JSONL file."""
    models = {}
    with open(models_file, 'r') as f:
        for line in f:
            if line.strip():
                model_data = json.loads(line)
                models[model_data['id']] = model_data
    return models


class GalaxyRanking(BaseModel):
    """Structured output for galaxy ranking."""
    ranking: int
    explanation: str


def zoom_image(image_array: np.ndarray, zoom_factor: float = 0.5) -> np.ndarray:
    """Zoom into the center of an image by cropping to a percentage of the original size."""
    from PIL import Image
    
    h, w = image_array.shape[:2]
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)
    
    # Calculate center crop coordinates
    start_h = (h - new_h) // 2
    end_h = start_h + new_h
    start_w = (w - new_w) // 2
    end_w = start_w + new_w
    
    # Crop the center portion
    cropped = image_array[start_h:end_h, start_w:end_w]
    
    # Convert to PIL Image for high-quality resizing
    if cropped.dtype == np.float32 or cropped.dtype == np.float64:
        cropped_uint8 = (cropped * 255).astype(np.uint8)
    else:
        cropped_uint8 = cropped
    
    pil_img = Image.fromarray(cropped_uint8)
    
    # Resize back to original dimensions using high-quality resampling
    pil_img_resized = pil_img.resize((w, h), Image.Resampling.LANCZOS)
    
    # Convert back to numpy array and normalize if needed
    result = np.array(pil_img_resized)
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        result = result.astype(np.float32) / 255.0
    
    return result


def save_image_for_gpt(image_array: np.ndarray, temp_dir: str, object_id: str, zoom: bool = False) -> str:
    """Process and save galaxy image to temporary file for model evaluation."""
    # Process image using the standard processing function
    img_rgb = process_galaxy_image(image_array)
    
    # Apply zoom if requested
    if zoom:
        img_rgb = zoom_image(img_rgb, zoom_factor=0.5)
    
    # Save to temporary file with zoom indicator in filename
    zoom_suffix = "_zoomed" if zoom else ""
    temp_path = os.path.join(temp_dir, f"{object_id}{zoom_suffix}.png")
    plt.imsave(temp_path, img_rgb)
    
    return temp_path


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_single_image(args: Tuple[int, str, str, str, str]) -> Tuple[int, int, str, int, int]:
    """
    Process a single image for ranking.
    
    Returns:
        Tuple of (index, ranking, explanation, input_tokens, output_tokens)
    """
    index, image_path, api_key, model_id, model_name = args
    
    try:
        ranking, explanation, input_tokens, output_tokens = get_gpt4_ranking(
            image_path, api_key, model_name
        )
        return index, ranking, explanation, input_tokens, output_tokens
    except Exception as e:
        logging.error(f"Error processing image {index}: {e}")
        return index, 5, "Error occurred during processing", 0, 0


def get_gpt4_ranking(image_path: str, api_key: str, model: str = "gpt-4.1") -> Tuple[int, str, int, int]:
    """Get model ranking for a galaxy image using the lens question."""
    client = OpenAI(api_key=api_key)
    
    base64_image = encode_image(image_path)

    try:
        request_kwargs = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                        {
                            "type": "input_text",
                            "text": LENS_QUESTION,
                        }
                    ]
                }
            ],
            "tools": [],
            "store": True,
            "text_format": GalaxyRanking
        }

        response = client.responses.parse(**request_kwargs)
        
        # Extract parsed response
        parsed_response = getattr(response, 'output_parsed', None)
        if parsed_response is None:
            raise ValueError("API returned empty or null response")
        
        ranking = parsed_response.ranking
        explanation = parsed_response.explanation
        
        # Extract token counts
        input_tokens = getattr(response.usage, 'input_tokens', None) if hasattr(response, 'usage') else 0
        output_tokens = getattr(response.usage, 'output_tokens', None) if hasattr(response, 'usage') else 0
        
        return ranking, explanation, input_tokens or 0, output_tokens or 0
        
    except Exception as e:
        logging.error(f"Error getting model ranking: {e}")
        raise


def get_full_dataset_similarities(
    checkpoint_path: str,
    eval_name: str = 'lens_hsc',
    device: str = 'cuda',
    batch_size: int = 512,
    logger: logging.Logger = None
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, Dict]:
    """Load full dataset and compute similarities using the trained model."""
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
    
    # Load full dataset
    logger.info(f"Loading full {eval_name} dataset from {eval_config.embeddings_path}...")
    with h5py.File(eval_config.embeddings_path, 'r') as f:
        n_objects = f['object_id'].shape[0]
        
        # Load all data
        object_ids = [f['object_id'][i].decode('utf-8') if isinstance(f['object_id'][i], bytes) 
                     else f['object_id'][i] for i in range(n_objects)]
        eval_grades = [f['eval_grade'][i].decode('utf-8') if isinstance(f['eval_grade'][i], bytes) 
                      else f['eval_grade'][i] for i in range(n_objects)]
        
        # Load embeddings
        embedding_key = 'aion_embedding_mean' if model_config['use_mean_embeddings'] else 'aion_embedding_full'
        aion_embeddings = f[embedding_key][:]
        
        # Load HSC images if available
        if 'hsc_image' in f:
            hsc_images = f['hsc_image'][:]
        else:
            logger.info("HSC images not found in HDF5, will load from FITS later")
            hsc_images = None
    
    logger.info(f"Loaded {n_objects} objects from full dataset")
    
    # Generate query embedding
    from src.evals.eval_utils import generate_query_embedding
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
    
    # Compute similarities
    logger.info("Computing similarities...")
    similarities = image_features @ query_features.T
    similarities = similarities.squeeze()
    
    return object_ids, similarities, eval_grades, hsc_images, model_config


def rerank_galaxies_single_question(
    checkpoint_path: str,
    api_key: str,
    max_images: int = 100,
    model_id: str = "gpt-4.1",
    logger: logging.Logger = None,
    zoom: bool = False,
    output_dir: str = None,
    k_values: List[int] = [10, 100, 1000]
) -> Dict:
    """
    Rerank galaxy images using vision models with single lens question.
    
    Returns:
        Dictionary with results and metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load model pricing info
    models_info = load_models_info()
    if model_id not in models_info:
        raise ValueError(f"Model {model_id} not found in models.jsonl")
    
    model_info = models_info[model_id]
    model_name = model_info['model_name']
    input_price_per_million = model_info['input_price']
    output_price_per_million = model_info['output_price']
    
    logger.info(f"Using model: {model_info['formatted_name']} ({model_name})")
    logger.info(f"Pricing: ${input_price_per_million}/1M input, ${output_price_per_million}/1M output")
    
    # Load full dataset and compute similarities
    logger.info("Loading full dataset and computing similarities...")
    object_ids, similarities, eval_grades, hsc_images, model_config = get_full_dataset_similarities(
        checkpoint_path=checkpoint_path,
        eval_name='lens_hsc',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger
    )
    
    n_total = len(object_ids)
    logger.info(f"Total objects in dataset: {n_total}")
    
    # Get original ranking (sorted by similarity descending)
    original_indices = np.argsort(similarities)[::-1]
    
    # Get top k indices to rerank
    top_indices_to_rerank = original_indices[:max_images]
    logger.info(f"Will rerank top {max_images} objects")
    
    # Load HSC images for top k objects if not already loaded
    if hsc_images is None:
        logger.info("Loading HSC images from pre-saved files...")
        top_k_file = None
        if max_images <= 100 and Path("data/experiments/rerank/best_lens_hsc_model_top100_lens_hsc.hdf5").exists():
            top_k_file = "data/experiments/rerank/best_lens_hsc_model_top100_lens_hsc.hdf5"
        elif max_images <= 1000 and Path("data/experiments/rerank/best_lens_hsc_model_top1000_lens_hsc.hdf5").exists():
            top_k_file = "data/experiments/rerank/best_lens_hsc_model_top1000_lens_hsc.hdf5"
        
        if top_k_file:
            logger.info(f"Loading images from {top_k_file}")
            with h5py.File(top_k_file, 'r') as f:
                top_k_object_ids = [f['object_id'][i].decode('utf-8') if isinstance(f['object_id'][i], bytes) 
                                   else f['object_id'][i] for i in range(f['object_id'].shape[0])]
                top_k_images = f['hsc_image'][:]
                
            # Create mapping for quick lookup
            oid_to_image = dict(zip(top_k_object_ids, top_k_images))
            
            # Create array to store images for reranking
            images_to_rerank = []
            for idx in top_indices_to_rerank:
                oid = object_ids[idx]
                if oid in oid_to_image:
                    images_to_rerank.append(oid_to_image[oid])
                else:
                    logger.warning(f"Image not found for object {oid}")
                    images_to_rerank.append(np.zeros((424, 424, 3), dtype=np.uint8))
    else:
        images_to_rerank = hsc_images[top_indices_to_rerank]
    
    # Create directory for images
    if output_dir is None:
        output_dir = Path("data/experiments/rerank") / f"single_question_{model_id}_top{max_images}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "galaxy_images"
    images_dir.mkdir(exist_ok=True)
    logger.info(f"Saving results to {output_dir}")
    
    # Process and save images for GPT evaluation
    logger.info(f"Processing and saving images (zoom={'enabled' if zoom else 'disabled'})...")
    image_paths = []
    for i, idx in enumerate(top_indices_to_rerank):
        image_path = save_image_for_gpt(images_to_rerank[i], str(images_dir), object_ids[idx], zoom=zoom)
        image_paths.append(image_path)
    
    # Get GPT rankings for top k objects
    logger.info(f"Getting GPT rankings for top {max_images} objects...")
    
    # Prepare arguments for parallel processing
    process_args = [
        (i, image_paths[i], api_key, model_id, model_name)
        for i in range(len(top_indices_to_rerank))
    ]
    
    # Process images in parallel
    gpt_scores = np.zeros(n_total)
    gpt_explanations = [""] * n_total
    total_input_tokens = 0
    total_output_tokens = 0
    
    logger.info(f"Processing images in parallel using 5 cores...")
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(process_single_image, args): args[0] 
            for args in process_args
        }
        
        with tqdm(total=len(top_indices_to_rerank), desc="Ranking galaxies") as pbar:
            for future in as_completed(future_to_index):
                try:
                    index, ranking, explanation, input_tokens, output_tokens = future.result()
                    obj_idx = top_indices_to_rerank[index]
                    
                    gpt_scores[obj_idx] = ranking
                    gpt_explanations[obj_idx] = explanation
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    pbar.update(1)
                    
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Error processing image {index}: {e}")
                    pbar.update(1)
    
    # Calculate costs
    total_cost = (total_input_tokens * input_price_per_million + total_output_tokens * output_price_per_million) / 1_000_000
    cost_per_image = total_cost / max_images if max_images > 0 else 0
    
    logger.info(f"\nCost Summary:")
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost per image: ${cost_per_image:.4f}")
    
    # Create new ranking
    reranked_indices = original_indices.copy()
    
    # Get indices of top objects to rerank and their scores
    top_scores = gpt_scores[top_indices_to_rerank]
    top_similarities = similarities[top_indices_to_rerank]
    
    # Sort these top objects by score (desc) then similarity (desc)
    sorted_top_indices = np.lexsort((top_similarities[::-1], top_scores))[::-1]
    
    # Replace the top indices in the full ranking
    reranked_indices[:max_images] = top_indices_to_rerank[sorted_top_indices]
    
    # Calculate new ranks
    new_ranks = np.zeros(n_total, dtype=int)
    for rank, idx in enumerate(reranked_indices):
        new_ranks[idx] = rank + 1
    
    # Calculate original ranks
    original_ranks = np.zeros(n_total, dtype=int)
    for rank, idx in enumerate(original_indices):
        original_ranks[idx] = rank + 1
    
    # Get relevance scores (1 for lens grades A, B, C; 0 otherwise)
    relevance_scores = np.array([1.0 if grade in ['A', 'B', 'C'] else 0.0 for grade in eval_grades])
    
    # Calculate metrics
    results = {
        'model_id': model_id,
        'model_name': model_info['formatted_name'],
        'n_total': n_total,
        'n_reranked': max_images,
        'total_cost': total_cost,
        'cost_per_image': cost_per_image,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_price_per_million': input_price_per_million,
        'output_price_per_million': output_price_per_million,
        'checkpoint_path': checkpoint_path,
        'zoom': zoom,
        'output_dir': str(output_dir)
    }
    
    # Calculate NDCG and lens counts for each k
    for k in k_values:
        if k <= n_total:
            # Original metrics
            original_relevance_k = relevance_scores[original_indices[:k]]
            original_ndcg = compute_ndcg_at_k(original_relevance_k, k, relevance_scores)
            original_lenses = int(np.sum(original_relevance_k))
            
            results[f'original_ndcg@{k}'] = original_ndcg
            results[f'original_lenses@{k}'] = original_lenses
            
            # Reranked metrics
            reranked_relevance_k = relevance_scores[reranked_indices[:k]]
            reranked_ndcg = compute_ndcg_at_k(reranked_relevance_k, k, relevance_scores)
            reranked_lenses = int(np.sum(reranked_relevance_k))
            
            results[f'reranked_ndcg@{k}'] = reranked_ndcg
            results[f'reranked_lenses@{k}'] = reranked_lenses
            
            # Improvements
            results[f'improvement_ndcg@{k}'] = reranked_ndcg - original_ndcg
            results[f'improvement_pct_ndcg@{k}'] = ((reranked_ndcg - original_ndcg) / original_ndcg * 100) if original_ndcg > 0 else 0
            results[f'improvement_lenses@{k}'] = reranked_lenses - original_lenses
    
    # Load RA/DEC data
    ra_dec_df = pd.read_csv("data/evals/lens/lens_eval_objects.csv")
    oid_to_radec = {str(row['object_id']): (row['ra'], row['dec']) 
                     for _, row in ra_dec_df.iterrows()}
    
    # Get RA/DEC for all objects
    ra_values = []
    dec_values = []
    for oid in object_ids:
        if oid in oid_to_radec:
            ra, dec = oid_to_radec[oid]
            ra_values.append(ra)
            dec_values.append(dec)
        else:
            ra_values.append(np.nan)
            dec_values.append(np.nan)
    
    # Save comprehensive results
    np.savez_compressed(
        output_dir / "reranked_data.npz",
        reranked_indices=reranked_indices,
        object_ids=np.array(object_ids),
        ra=np.array(ra_values),
        dec=np.array(dec_values),
        eval_grades=np.array(eval_grades),
        similarities=similarities,
        gpt_scores=gpt_scores,
        original_indices=original_indices,
        new_ranks=new_ranks,
        original_ranks=original_ranks
    )
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed CSV
    csv_data = []
    for idx in range(n_total):
        row = {
            'object_id': object_ids[idx],
            'ra': ra_values[idx],
            'dec': dec_values[idx],
            'eval_grade': eval_grades[idx],
            'similarity': similarities[idx],
            'original_rank': original_ranks[idx],
            'new_rank': new_ranks[idx],
            'gpt_score': gpt_scores[idx],
            'gpt_explanation': gpt_explanations[idx],
            'was_reranked': idx in top_indices_to_rerank
        }
        csv_data.append(row)
    
    # Sort by new rank
    csv_data.sort(key=lambda x: x['new_rank'])
    
    # Write to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / "reranked_results.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Rerank galaxies using single lens question")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--max-images", type=int, default=100,
                       help="Maximum number of images to rerank with GPT (default: 100)")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                       help="Model ID from models.jsonl (default: gpt-4.1)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--zoom", action="store_true",
                       help="Apply 50% zoom to center of images before evaluation")
    parser.add_argument("--k-values", type=int, nargs='+', default=[10, 100, 1000],
                       help="k values for metrics (default: 10 100 1000)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Run reranking
    logger.info(f"Starting single question reranking with {args.model}")
    if args.zoom:
        logger.info("Zoom mode enabled: Images will be zoomed to 50% center")
    
    results = rerank_galaxies_single_question(
        checkpoint_path=args.checkpoint,
        api_key=api_key,
        max_images=args.max_images,
        model_id=args.model,
        logger=logger,
        zoom=args.zoom,
        output_dir=args.output_dir,
        k_values=args.k_values
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SINGLE QUESTION RERANKING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {results['model_name']}")
    print(f"Total objects in dataset: {results['n_total']:,}")
    print(f"Objects reranked with GPT: {results['n_reranked']:,}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Cost per image: ${results['cost_per_image']:.4f}")
    print(f"{'-'*60}")
    
    # Print metrics
    for k in args.k_values:
        if f'original_ndcg@{k}' in results:
            print(f"\nMetrics @{k}:")
            print(f"  Original - NDCG: {results[f'original_ndcg@{k}']:.4f}, Lenses: {results[f'original_lenses@{k}']}")
            
            ndcg = results[f'reranked_ndcg@{k}']
            lenses = results[f'reranked_lenses@{k}']
            improvement = results[f'improvement_ndcg@{k}']
            improvement_pct = results[f'improvement_pct_ndcg@{k}']
            lens_improvement = results[f'improvement_lenses@{k}']
            
            print(f"  Reranked - NDCG: {ndcg:.4f} ({improvement:+.4f}, {improvement_pct:+.1f}%), Lenses: {lenses} ({lens_improvement:+d})")
    
    print(f"{'='*60}\n")
        

if __name__ == "__main__":
    main()