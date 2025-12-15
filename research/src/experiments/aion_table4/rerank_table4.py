"""
Rerank Table 4 evaluation results using vision language models.
Takes CSV outputs from eval_table4_gz.py and eval_table4_lens.py and reranks using VLM scoring.
"""

import argparse
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import base64
from typing import Dict, Tuple, List
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Import utilities
from src.evals.eval_utils import setup_logging
from src.plotting_scripts.default_plot import process_galaxy_image


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


def dcg(r):
    """Compute Discounted Cumulative Gain (DCG)."""
    return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))


def ndcg_score(relevances, top_k):
    """Compute NDCG@k using fraction method.
    
    Args:
        relevances: Relevance scores in ranked order
        top_k: Number of top results to consider
    
    Returns:
        NDCG@k score
    """
    actual_dcg = dcg(relevances[:top_k])
    ideal_dcg = dcg(np.sort(relevances)[::-1][:top_k])
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_question_for_eval(eval_type: str) -> str:
    """Get the appropriate question based on evaluation type."""
    questions = {
        'spiral_fraction': "Does this galaxy have visible spiral arms? Rank 1-10 where 10 means you are entirely sure there are spiral arms and 1 being you are entirely sure there are no spiral arms.",
        'merger_fraction': "Does this galaxy image show signs of actively merging galaxies? Rank 1-10 where 10 means you are entirely sure it displays active merging and 1 being you are entirely sure it does not display active merging.",
        'is_lens': "Does this galaxy image display signs of gravitational lensing? Rank 1-10 where 10 means you are entirely sure there are signs of gravitational lensing and 1 being you are entirely sure there are no signs of gravitational lensing."
    }
    return questions.get(eval_type, "")


def detect_eval_type(df: pd.DataFrame) -> str:
    """Detect evaluation type from DataFrame columns."""
    if 'spiral_fraction' in df.columns:
        return 'spiral_fraction'
    elif 'merger_fraction' in df.columns:
        return 'merger_fraction'
    elif 'is_lens' in df.columns:
        return 'is_lens'
    else:
        raise ValueError("Could not determine evaluation type from columns")


def process_single_image(args: Tuple[int, str, str, str, str, str, str]) -> Tuple[int, int, str, int, int]:
    """
    Process a single image for ranking.
    
    Returns:
        Tuple of (index, ranking, explanation, input_tokens, output_tokens)
    """
    index, image_path, api_key, model_id, model_name, question, reasoning_effort = args

    try:
        ranking, explanation, input_tokens, output_tokens = get_gpt4_ranking(
            image_path, api_key, model_name, question, reasoning_effort
        )
        return index, ranking, explanation, input_tokens, output_tokens
    except Exception as e:
        logging.error(f"Error processing image {index}: {e}")
        return index, 5, "Error occurred during processing", 0, 0


def get_gpt4_ranking(image_path: str, api_key: str, model: str, question: str, reasoning_effort: str = None) -> Tuple[int, str, int, int]:
    """Get model ranking for a galaxy image."""
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
                            "text": question
                        }
                    ]
                }
            ],
            "tools": [],
            "store": True,
            "text_format": GalaxyRanking
        }
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

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


def rerank_table4_results(
    csv_path: str,
    images_dir: str,
    api_key: str,
    model_id: str = "gpt-4.1",
    max_images: int = 100,
    logger: logging.Logger = None,
    output_dir: str = None,
    k_values: List[int] = [10, 100, 1000]
) -> Dict:
    """
    Rerank Table 4 evaluation results using vision language models.
    
    Args:
        csv_path: Path to CSV file from eval_table4_*.py
        images_dir: Directory containing galaxy images
        api_key: OpenAI API key
        model_id: Model ID from models.jsonl
        max_images: Maximum number of images to rerank (ignored if less images in dir)
        logger: Logger instance
        output_dir: Output directory for results
        k_values: k values for NDCG computation
    
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
    reasoning_effort = model_info.get('reasoning_effort', None)
    
    logger.info(f"Using model: {model_info['formatted_name']} ({model_name})")
    if reasoning_effort:
        logger.info(f"Reasoning effort: {reasoning_effort}")
    logger.info(f"Pricing: ${input_price_per_million}/1M input, ${output_price_per_million}/1M output")
    
    # Load CSV data
    logger.info(f"Loading CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create a mapping from object_id to index in dataframe
    # Convert object_id to string to ensure consistent comparison with filenames
    object_id_to_idx = {str(row['object_id']): idx for idx, row in df.iterrows()}
    
    # Scan images directory to find what needs to be reranked
    logger.info(f"Scanning images directory: {images_dir}")
    images_path = Path(images_dir)
    
    # Look for PNG files in the directory and subdirectories
    image_files = list(images_path.glob("*.png"))
    if not image_files:
        # Try looking in subdirectories
        image_files = list(images_path.glob("*/*.png"))
    
    logger.info(f"Found {len(image_files)} PNG files")
    
    # Extract object IDs from image filenames
    images_to_rerank = []
    indices_to_rerank = []
    
    for img_file in image_files:
        # Extract object_id from filename (remove .png extension)
        object_id = img_file.stem
        
        # Find corresponding index in dataframe
        if object_id in object_id_to_idx:
            idx = object_id_to_idx[object_id]
            images_to_rerank.append(str(img_file))
            indices_to_rerank.append(idx)
        else:
            logger.warning(f"Object ID {object_id} from image not found in CSV")
    
    # Sort by original rank (index in dataframe, since CSV is already sorted by similarity)
    sorted_pairs = sorted(zip(indices_to_rerank, images_to_rerank), 
                         key=lambda x: x[0])  # Sort by index (which represents rank)
    
    if sorted_pairs:
        indices_to_rerank, images_to_rerank = zip(*sorted_pairs)
        indices_to_rerank = list(indices_to_rerank)
        images_to_rerank = list(images_to_rerank)
    
    logger.info(f"Found {len(images_to_rerank)} images in directory")
    
    # Apply max_images limit
    if len(images_to_rerank) > max_images:
        logger.info(f"Limiting to first {max_images} images (ordered by rank)")
        indices_to_rerank = indices_to_rerank[:max_images]
        images_to_rerank = images_to_rerank[:max_images]
    elif max_images > len(images_to_rerank) and len(images_to_rerank) > 0:
        logger.warning(f"max_images ({max_images}) is greater than images found in directory ({len(images_to_rerank)})")
        logger.warning(f"Will process all {len(images_to_rerank)} available images")
    
    logger.info(f"Will rerank {len(images_to_rerank)} images")
    
    # Detect evaluation type
    eval_type = detect_eval_type(df)
    logger.info(f"Detected evaluation type: {eval_type}")
    
    # Get appropriate question
    question = get_question_for_eval(eval_type)
    logger.info(f"Using question: {question}")
    
    # Determine relevance column and scoring
    if eval_type == 'spiral_fraction':
        relevance_col = 'spiral_fraction'
        relevance_scores = df[relevance_col].values
    elif eval_type == 'merger_fraction':
        relevance_col = 'merger_fraction'
        relevance_scores = df[relevance_col].values
    elif eval_type == 'is_lens':
        relevance_col = 'is_lens'
        # For lens, it's binary (1 or 0)
        relevance_scores = df[relevance_col].values
    
    n_total = len(df)
    logger.info(f"Total objects in dataset: {n_total}")
    
    # Get similarity scores from CSV
    similarities = df['similarity'].values
    
    # Original ranking is already in the CSV (sorted by similarity)
    original_indices = np.arange(n_total)
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(csv_path).parent / f"rerank_{eval_type}_{model_id}_n{len(images_to_rerank)}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")
    
    # Get GPT rankings
    logger.info(f"Getting GPT rankings for {len(images_to_rerank)} objects...")
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, image_path in enumerate(images_to_rerank):
        process_args.append((i, image_path, api_key, model_id, model_name, question, reasoning_effort))
    
    # Initialize arrays for results
    gpt_scores = np.zeros(n_total)
    gpt_explanations = [""] * n_total
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Process images in parallel
    logger.info("Processing images in parallel using 5 cores...")
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(process_single_image, args): args[0] 
            for args in process_args
        }
        
        with tqdm(total=len(process_args), desc="Ranking galaxies") as pbar:
            for future in as_completed(future_to_index):
                try:
                    index, ranking, explanation, input_tokens, output_tokens = future.result()
                    obj_idx = indices_to_rerank[index]
                    
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
    cost_per_image = total_cost / len(process_args) if process_args else 0
    
    logger.info(f"\nCost Summary:")
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost per image: ${cost_per_image:.4f}")
    
    # Create new ranking
    reranked_indices = original_indices.copy()
    
    # Separate indices into those with VLM scores and those without
    has_vlm_score = gpt_scores > 0
    vlm_indices = [i for i in range(n_total) if has_vlm_score[i]]
    no_vlm_indices = [i for i in range(n_total) if not has_vlm_score[i]]
    
    # Sort VLM-scored indices by VLM score (desc), then by similarity (desc)
    vlm_scores_for_sorting = [(gpt_scores[i], similarities[i], i) for i in vlm_indices]
    vlm_scores_for_sorting.sort(key=lambda x: (-x[0], -x[1]))
    sorted_vlm_indices = [x[2] for x in vlm_scores_for_sorting]
    
    # Sort non-VLM indices by similarity (desc) - they're already in this order
    sorted_no_vlm_indices = sorted(no_vlm_indices, key=lambda i: similarities[i], reverse=True)
    
    # Combine: VLM-scored objects first, then non-VLM objects
    reranked_indices = np.array(sorted_vlm_indices + sorted_no_vlm_indices)
    
    # Calculate metrics
    results = {
        'csv_path': csv_path,
        'eval_type': eval_type,
        'model_id': model_id,
        'model_name': model_info['formatted_name'],
        "reasoning_effort": reasoning_effort,
        'n_total': n_total,
        'n_images_found': len(images_to_rerank),
        'n_reranked': len(process_args),
        'n_vlm_scored': len(vlm_indices),
        'total_cost': total_cost,
        'cost_per_image': cost_per_image,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_price_per_million': input_price_per_million,
        'output_price_per_million': output_price_per_million,
        'output_dir': str(output_dir)
    }
    
    # Calculate NDCG for each k
    for k in k_values:
        if k <= n_total:
            # Original metrics
            # Get relevance scores in original order (already sorted by similarity)
            original_relevances_sorted = relevance_scores[original_indices]
            original_ndcg = ndcg_score(original_relevances_sorted, k)
            
            results[f'original_ndcg@{k}'] = original_ndcg
            
            # Reranked metrics
            # Get relevance scores in reranked order
            reranked_relevances_sorted = relevance_scores[reranked_indices]
            reranked_ndcg = ndcg_score(reranked_relevances_sorted, k)
            
            results[f'reranked_ndcg@{k}'] = reranked_ndcg
            
            # Improvements
            results[f'improvement_ndcg@{k}'] = reranked_ndcg - original_ndcg
            results[f'improvement_pct_ndcg@{k}'] = ((reranked_ndcg - original_ndcg) / original_ndcg * 100) if original_ndcg > 0 else 0
            
            # For lens detection, also count number of lenses
            if eval_type == 'is_lens':
                original_lenses = int(np.sum(original_relevances_sorted[:k]))
                reranked_lenses = int(np.sum(reranked_relevances_sorted[:k]))
                results[f'original_lenses@{k}'] = original_lenses
                results[f'reranked_lenses@{k}'] = reranked_lenses
                results[f'improvement_lenses@{k}'] = reranked_lenses - original_lenses
    
    # Save results
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create reranked CSV
    reranked_df = df.copy()
    
    # Add VLM columns
    reranked_df['vlm_score'] = gpt_scores[original_indices]
    reranked_df['vlm_reasoning'] = [gpt_explanations[i] for i in original_indices]
    
    # Reorder rows according to new ranking
    reranked_df = reranked_df.iloc[reranked_indices]
    reranked_df['new_rank'] = range(1, n_total + 1)
    
    # Save reranked CSV
    output_csv = output_dir / f"{Path(csv_path).stem}_reranked.csv"
    reranked_df.to_csv(output_csv, index=False)
    logger.info(f"Reranked results saved to: {output_csv}")
    
    # Create visualization of top 30 galaxies by VLM score
    logger.info("Creating visualization of top 30 galaxies by VLM score...")
    
    # Create a mapping from dataframe index to image path for quick lookup
    idx_to_image_path = {}
    for i, img_path in zip(indices_to_rerank, images_to_rerank):
        idx_to_image_path[i] = img_path
    
    # Get indices sorted by VLM score (descending)
    vlm_scores_array = np.array([gpt_scores[i] for i in original_indices])
    top_vlm_indices = np.argsort(vlm_scores_array)[::-1][:30]
    
    # Create figure with 6x5 grid
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    axes = axes.flatten()
    
    for plot_idx, idx in enumerate(top_vlm_indices):
        ax = axes[plot_idx]
        
        # Get data for this galaxy
        object_id = df.iloc[idx]['object_id']
        vlm_score = vlm_scores_array[idx]
        relevance_score = relevance_scores[idx]
        original_rank = idx + 1  # Since df is already sorted by similarity
        
        # Find new rank
        new_rank = np.where(reranked_indices == idx)[0][0] + 1
        
        # Load and display image
        if idx in idx_to_image_path:
            image_path = idx_to_image_path[idx]
            img = Image.open(image_path)
            ax.imshow(img)
        else:
            # Try to find image in subdirectories
            object_id_str = str(object_id)
            image_paths = list(Path(images_dir).glob(f"*/{object_id_str}.png"))
            if image_paths:
                img = Image.open(image_paths[0])
                ax.imshow(img)
            else:
                # Create placeholder for missing images
                ax.text(0.5, 0.5, 'Image\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        # Create title with scores and ranks
        # Determine relevance label based on eval type
        if eval_type == 'spiral_fraction':
            relevance_label = 'Spiral'
        elif eval_type == 'merger_fraction':
            relevance_label = 'Merger'
        elif eval_type == 'is_lens':
            relevance_label = 'Lens'
        
        title = f"VLM: {vlm_score:.1f}, {relevance_label}: {relevance_score:.3f}\nRank: {original_rank}→{new_rank}"
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis('off')
    
    # Remove any empty subplots
    for idx in range(30, len(axes)):
        fig.delaxes(axes[idx])
    
    # Add overall title
    fig.suptitle(f'Top 30 Galaxies by VLM Score ({eval_type})', fontsize=16, y=0.995)
    
    # Adjust layout and save
    plt.tight_layout()
    output_figure = output_dir / "top30_vlm_scores.png"
    plt.savefig(output_figure, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved to: {output_figure}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Rerank Table 4 evaluation results using VLMs")
    
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to CSV file from eval_table4_*.py")
    parser.add_argument("--images", type=str, required=True,
                       help="Directory containing galaxy images to rerank")
    parser.add_argument("--max-images", type=int, default=1000,
                       help="Maximum number of images to rerank (ignored if fewer images in directory)")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                       help="Model ID from models.jsonl (default: gpt-4.1)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--k-values", type=int, nargs='+', default=[10, 100, 1000],
                       help="k values for NDCG metrics (default: 10 100 1000)")
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
    logger.info(f"Starting reranking with {args.model}")
    
    results = rerank_table4_results(
        csv_path=args.csv,
        images_dir=args.images,
        api_key=api_key,
        model_id=args.model,
        max_images=args.max_images,
        logger=logger,
        output_dir=args.output_dir,
        k_values=args.k_values
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TABLE 4 RERANKING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Evaluation Type: {results['eval_type']}")
    print(f"Model: {results['model_name']}")
    print(f"Total objects in CSV: {results['n_total']:,}")
    print(f"Images found in directory: {results['n_images_found']:,}")
    print(f"Images successfully scored: {results['n_vlm_scored']:,}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Cost per image: ${results['cost_per_image']:.4f}")
    print(f"{'-'*60}")
    
    # Print metrics
    for k in args.k_values:
        if f'original_ndcg@{k}' in results:
            print(f"\nMetrics @{k}:")
            print(f"  Original NDCG: {results[f'original_ndcg@{k}']:.4f}")
            
            ndcg = results[f'reranked_ndcg@{k}']
            improvement = results[f'improvement_ndcg@{k}']
            improvement_pct = results[f'improvement_pct_ndcg@{k}']
            
            print(f"  Reranked NDCG: {ndcg:.4f} ({improvement:+.4f}, {improvement_pct:+.1f}%)")
            
            # For lens detection, show lens counts
            if results['eval_type'] == 'is_lens' and f'original_lenses@{k}' in results:
                orig_lenses = results[f'original_lenses@{k}']
                reranked_lenses = results[f'reranked_lenses@{k}']
                lens_improvement = results[f'improvement_lenses@{k}']
                print(f"  Original lenses: {orig_lenses}")
                print(f"  Reranked lenses: {reranked_lenses} ({lens_improvement:+d})")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()