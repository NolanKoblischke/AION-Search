"""Unified display script for all evaluation types."""

import json
import os
import sys
from pathlib import Path
import base64
import argparse
import glob
from typing import Dict, Any, List, Optional, Tuple


# Import the combined module to register configs
from .config import get_eval_config, list_eval_types


def get_most_recent_jsonl(pattern: str) -> Optional[str]:
    """Get the most recent JSONL file matching a pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda x: Path(x).stat().st_mtime)


def detect_eval_type(input_file: str) -> str:
    """Detect evaluation type from input file by checking for specific fields.
    
    Returns 'all' if multiple eval types are found in the same file.
    """
    eval_types_found = set()
    
    # Read through the file and sample records to check for mixed eval types
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    # Check first 50, middle 50, and last 50 records to get a good sample
    sample_indices = []
    sample_indices.extend(range(min(50, len(lines))))  # First 50
    if len(lines) > 100:
        middle_start = len(lines) // 2 - 25
        sample_indices.extend(range(middle_start, middle_start + 50))  # Middle 50
    if len(lines) > 50:
        sample_indices.extend(range(max(0, len(lines) - 50), len(lines)))  # Last 50
    
    # Remove duplicates and sort
    sample_indices = sorted(set(sample_indices))
    
    for i in sample_indices:
        if i < len(lines) and lines[i].strip():
            try:
                record = json.loads(lines[i])
                
                # Check for evaluation-specific fields
                if 'decision_tree' in record:
                    eval_types_found.add('galaxyzoo')
                elif 'tidal_info' in record or 'tidal_score' in record:
                    eval_types_found.add('tidal')
                elif 'lensgrade' in record or 'description_says_lens_occuring_score' in record or ('judge_results' in record and record.get('judge_results', {}).get('description_says_lens_occuring') is not None):
                    eval_types_found.add('lens')
                    
                # Short circuit if we found multiple types
                if len(eval_types_found) > 1:
                    break
                    
            except json.JSONDecodeError:
                continue
    
    if len(eval_types_found) == 0:
        raise ValueError("Could not detect evaluation type from input file")
    elif len(eval_types_found) == 1:
        return list(eval_types_found)[0]
    else:
        # Multiple eval types found
        return 'all'


def calculate_model_stats(descriptions: Dict[str, Dict[str, Any]], 
                         judging_results: Dict[str, Dict[str, Any]], 
                         score_field: str) -> Dict[str, Dict[str, Any]]:
    """Calculate summary statistics for each model, separated by evaluation type."""
    model_stats = {}
    
    # Collect data for each model
    for galaxy_id, galaxy_descriptions in descriptions.items():
        for model_name, data in galaxy_descriptions.items():
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'costs': [],
                    'times': [],
                    'scores': [],
                    'galaxyzoo_scores': [],
                    'tidal_scores': [],
                    'lens_scores': [],
                    'galaxyzoo_count': 0,
                    'tidal_count': 0,
                    'lens_count': 0,
                    # Lens confusion matrix
                    'lens_tp': 0,  # True Positives (correctly identified lenses)
                    'lens_fn': 0,  # False Negatives (missed lenses)
                    'lens_tn': 0,  # True Negatives (correctly identified non-lenses)
                    'lens_fp': 0,  # False Positives (incorrectly identified lenses)
                    # Tidal confusion matrix per class
                    'tidal_shell_tp': 0,
                    'tidal_shell_fn': 0,
                    'tidal_shell_fp': 0,
                    'tidal_stream_tp': 0,
                    'tidal_stream_fn': 0,
                    'tidal_stream_fp': 0,
                    'tidal_other_tp': 0,
                    'tidal_other_fn': 0
                }
            
            # Determine eval type for this galaxy
            eval_type = 'galaxyzoo' if 'decision_tree' in data else 'tidal' if 'tidal_info' in data or 'tidal_score' in data else 'lens' if 'lensgrade' in data or 'description_says_lens_occuring_score' in data else 'unknown'
            
            # Add cost and time if available
            if data.get('cost') is not None:
                model_stats[model_name]['costs'].append(data['cost'])
            if data.get('time') is not None:
                model_stats[model_name]['times'].append(data['time'])
            
            # Add score if available in judging results - handle dynamic score fields
            if (galaxy_id in judging_results and 
                model_name in judging_results[galaxy_id]):
                
                judge_data = judging_results[galaxy_id][model_name]
                judge_results = judge_data.get('judge_results', {})
                
                # Try different score fields based on eval type
                score = None
                if eval_type == 'galaxyzoo' and judge_data.get('decision_tree_score') is not None:
                    score = judge_data['decision_tree_score']
                    model_stats[model_name]['galaxyzoo_scores'].append(score * 100)
                    model_stats[model_name]['galaxyzoo_count'] += 1
                    
                elif eval_type == 'tidal' and judge_data.get('tidal_score') is not None:
                    score = judge_data['tidal_score']
                    model_stats[model_name]['tidal_scores'].append(score * 100)
                    model_stats[model_name]['tidal_count'] += 1
                    
                    # Track tidal confusion matrix
                    if 'judge_classification' in judge_results and 'judge_tidal_info' in judge_results:
                        judge_class = judge_results['judge_classification'].get('classification', 'Other')
                        true_class = judge_results['judge_tidal_info'].get('eval_class', 'Other')
                        
                        # Update confusion matrix for each class
                        if true_class == 'Shell':
                            if judge_class == 'Shell':
                                model_stats[model_name]['tidal_shell_tp'] += 1
                            else:
                                model_stats[model_name]['tidal_shell_fn'] += 1
                        elif true_class == 'Stream':
                            if judge_class == 'Stream':
                                model_stats[model_name]['tidal_stream_tp'] += 1
                            else:
                                model_stats[model_name]['tidal_stream_fn'] += 1
                        elif true_class == 'Other':
                            if judge_class == 'Other':
                                model_stats[model_name]['tidal_other_tp'] += 1
                            else:
                                model_stats[model_name]['tidal_other_fn'] += 1
                        
                        # Also track false positives for precision calculation
                        if judge_class == 'Shell' and true_class != 'Shell':
                            model_stats[model_name]['tidal_shell_fp'] += 1
                        elif judge_class == 'Stream' and true_class != 'Stream':
                            model_stats[model_name]['tidal_stream_fp'] += 1
                    
                elif eval_type == 'lens' and judge_data.get('description_says_lens_occuring_score') is not None:
                    score = judge_data['description_says_lens_occuring_score']
                    model_stats[model_name]['lens_scores'].append(score * 100)
                    model_stats[model_name]['lens_count'] += 1
                    
                    # Track lens confusion matrix
                    if judge_results:
                        ground_truth_lens = judge_results.get('ground_truth_lens', False)
                        description_says_lens_occuring = judge_results.get('description_says_lens_occuring', False)
                        
                        if ground_truth_lens and description_says_lens_occuring:
                            model_stats[model_name]['lens_tp'] += 1
                        elif ground_truth_lens and not description_says_lens_occuring:
                            model_stats[model_name]['lens_fn'] += 1
                        elif not ground_truth_lens and not description_says_lens_occuring:
                            model_stats[model_name]['lens_tn'] += 1
                        elif not ground_truth_lens and description_says_lens_occuring:
                            model_stats[model_name]['lens_fp'] += 1
                    
                elif judge_data.get(score_field) is not None:
                    score = judge_data[score_field]
                
                if score is not None:
                    model_stats[model_name]['scores'].append(score * 100)  # Convert to 0-100 scale
    
    # Calculate summary statistics
    summary_stats = {}
    for model_name, data in model_stats.items():
        # Calculate lens F1 score
        lens_f1 = None
        if data['lens_tp'] + data['lens_fn'] > 0:  # Have actual lens examples
            # Recall (sensitivity)
            lens_recall = data['lens_tp'] / (data['lens_tp'] + data['lens_fn'])
            # Precision
            if data['lens_tp'] + data['lens_fp'] > 0:
                lens_precision = data['lens_tp'] / (data['lens_tp'] + data['lens_fp'])
            else:
                lens_precision = 0
            # F1 score
            if lens_precision + lens_recall > 0:
                lens_f1 = 2 * (lens_precision * lens_recall) / (lens_precision + lens_recall) * 100
            else:
                lens_f1 = 0
        
        # Calculate tidal F1 scores for each class
        shell_f1 = None
        if data['tidal_shell_tp'] + data['tidal_shell_fn'] > 0:  # Have actual shell examples
            # Shell recall
            shell_recall = data['tidal_shell_tp'] / (data['tidal_shell_tp'] + data['tidal_shell_fn'])
            # Shell precision
            if data['tidal_shell_tp'] + data['tidal_shell_fp'] > 0:
                shell_precision = data['tidal_shell_tp'] / (data['tidal_shell_tp'] + data['tidal_shell_fp'])
            else:
                shell_precision = 0
            # Shell F1
            if shell_precision + shell_recall > 0:
                shell_f1 = 2 * (shell_precision * shell_recall) / (shell_precision + shell_recall) * 100
            else:
                shell_f1 = 0
        
        stream_f1 = None
        if data['tidal_stream_tp'] + data['tidal_stream_fn'] > 0:  # Have actual stream examples
            # Stream recall
            stream_recall = data['tidal_stream_tp'] / (data['tidal_stream_tp'] + data['tidal_stream_fn'])
            # Stream precision
            if data['tidal_stream_tp'] + data['tidal_stream_fp'] > 0:
                stream_precision = data['tidal_stream_tp'] / (data['tidal_stream_tp'] + data['tidal_stream_fp'])
            else:
                stream_precision = 0
            # Stream F1
            if stream_precision + stream_recall > 0:
                stream_f1 = 2 * (stream_precision * stream_recall) / (stream_precision + stream_recall) * 100
            else:
                stream_f1 = 0
        
        # Average tidal F1 (only from Shell and Stream, not Other)
        tidal_f1 = None
        valid_f1s = [f for f in [shell_f1, stream_f1] if f is not None]
        if valid_f1s:
            tidal_f1 = sum(valid_f1s) / len(valid_f1s)
        
        stats = {
            'count': len(data['scores']) if data['scores'] else len(data['costs']) if data['costs'] else 0,
            'mean_cost': sum(data['costs']) / len(data['costs']) if data['costs'] else 0,
            'mean_time': sum(data['times']) / len(data['times']) if data['times'] else 0,
            'total_cost': sum(data['costs']) if data['costs'] else 0,
            'total_time': sum(data['times']) if data['times'] else 0,
            'mean_score': sum(data['scores']) / len(data['scores']) if data['scores'] else None,
            'galaxyzoo_mean_score': sum(data['galaxyzoo_scores']) / len(data['galaxyzoo_scores']) if data['galaxyzoo_scores'] else None,
            'tidal_mean_score': sum(data['tidal_scores']) / len(data['tidal_scores']) if data['tidal_scores'] else None,
            'lens_mean_score': sum(data['lens_scores']) / len(data['lens_scores']) if data['lens_scores'] else None,
            'galaxyzoo_count': data['galaxyzoo_count'],
            'tidal_count': data['tidal_count'],
            'lens_count': data['lens_count'],
            # New F1, Precision, and Recall metrics
            'lens_f1': lens_f1,
            'lens_precision': lens_precision * 100 if 'lens_precision' in locals() else None,
            'lens_recall': lens_recall * 100 if 'lens_recall' in locals() else None,
            'tidal_f1': tidal_f1,
            'tidal_shell_f1': shell_f1,
            'tidal_shell_precision': shell_precision * 100 if 'shell_precision' in locals() else None,
            'tidal_shell_recall': shell_recall * 100 if 'shell_recall' in locals() else None,
            'tidal_stream_f1': stream_f1,
            'tidal_stream_precision': stream_precision * 100 if 'stream_precision' in locals() else None,
            'tidal_stream_recall': stream_recall * 100 if 'stream_recall' in locals() else None,
            # Confusion matrix data for debugging
            'lens_tp': data['lens_tp'],
            'lens_fn': data['lens_fn'],
            'lens_tn': data['lens_tn'],
            'lens_fp': data['lens_fp'],
            'tidal_shell_tp': data['tidal_shell_tp'],
            'tidal_shell_fn': data['tidal_shell_fn'],
            'tidal_stream_tp': data['tidal_stream_tp'],
            'tidal_stream_fn': data['tidal_stream_fn']
        }
        summary_stats[model_name] = stats
    
    return summary_stats


def extract_prompt_info(descriptions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Extract prompt information from descriptions data."""
    # Get the first available prompt from any model/galaxy
    for galaxy_id, galaxy_descriptions in descriptions.items():
        for model_name, data in galaxy_descriptions.items():
            prompt = data.get('prompt', '')
            if prompt:
                return {'prompt': prompt}
    
    return {'prompt': ''}



def load_data(eval_config, judged_file: Optional[str] = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load all galaxy data including images, descriptions, and judging results."""
    prefix = eval_config.output_prefix
    
    # Use default if not provided
    if judged_file is None:
        if eval_config.name == "all":
            # Find most recent combined file
            judged_file = get_most_recent_jsonl(f"eval/judged_all_evals_*.jsonl")
            if judged_file is None:
                judged_file = f"eval/judged_all_evals.jsonl"
        else:
            # Find most recent judged file for this eval type
            judged_file = get_most_recent_jsonl(f"eval/judged_{prefix}_descriptions_*.jsonl")
            if judged_file is None:
                judged_file = f"eval/judged_{prefix}_descriptions.jsonl"
    
    print(f"Using judged file: {judged_file}")
    
    # Load descriptions and judging results from the same file
    descriptions = {}
    judging_results = {}
    
    if Path(judged_file).exists():
        with open(judged_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    object_id = data['object_id']
                    formatted_name = data['formatted_name']
                    
                    # Extract description data
                    if object_id not in descriptions:
                        descriptions[object_id] = {}
                    
                    # Store description data (everything except judge-specific fields)
                    desc_data = {k: v for k, v in data.items() 
                               if not k.startswith('judge_') and k not in ['decision_tree_score', 'tidal_score', 'description_says_lens_occuring_score']}
                    descriptions[object_id][formatted_name] = desc_data
                    
                    # Extract judging results - dynamically determine score field
                    if object_id not in judging_results:
                        judging_results[object_id] = {}
                    
                    # Determine score field based on data content
                    score_field = None
                    if 'decision_tree_score' in data:
                        score_field = 'decision_tree_score'
                    elif 'tidal_score' in data:
                        score_field = 'tidal_score'
                    elif 'description_says_lens_occuring_score' in data:
                        score_field = 'description_says_lens_occuring_score'
                    else:
                        # Fallback to generic score field
                        score_field = eval_config.get_score_field_name()
                    
                    # Store all judge-related fields
                    judge_data = {
                        score_field: data.get(score_field),
                        'judge_model': data.get('judge_model'),
                        'judge_results': data.get('judge_results', {}),
                        'eval_type': 'galaxyzoo' if 'decision_tree' in data else 'tidal' if 'tidal_info' in data or 'tidal_score' in data else 'lens' if 'lensgrade' in data or 'description_says_lens_occuring_score' in data else 'unknown'
                    }
                    
                    # Add any additional judge fields that might exist
                    for key, value in data.items():
                        if key.startswith('judge_') and key not in judge_data:
                            judge_data[key] = value
                    
                    # Also copy the score fields directly to judge_data for easy access
                    if 'decision_tree_score' in data:
                        judge_data['decision_tree_score'] = data['decision_tree_score']
                    if 'tidal_score' in data:
                        judge_data['tidal_score'] = data['tidal_score']
                    if 'description_says_lens_occuring_score' in data:
                        judge_data['description_says_lens_occuring_score'] = data['description_says_lens_occuring_score']
                    
                    judging_results[object_id][formatted_name] = judge_data
    
    # Extract and validate prompt information
    prompt_info = extract_prompt_info(descriptions)
    
    # Get available images from the description data (which contains image_path)
    images = {}
    for object_id, desc_dict in descriptions.items():
        # Get image path from any description record (they should all have the same image_path)
        if desc_dict:
            first_desc = list(desc_dict.values())[0]
            image_path = first_desc.get('image_path')
            if image_path and Path(image_path).exists():
                # Convert image to base64 for embedding
                with open(image_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                images[object_id] = f"data:image/png;base64,{img_data}"
    
    return descriptions, images, judging_results, prompt_info


def generate_prompt_html(prompt_info: Dict[str, Any]) -> str:
    """Generate HTML section for displaying prompt."""
    prompt = prompt_info.get('prompt', '')
    if not prompt:
        return ""
    
    # Escape HTML characters in prompt
    escaped_prompt = prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    # Replace newlines with <br> for display
    formatted_prompt = escaped_prompt.replace('\\n', '<br>')
    
    return f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); backdrop-filter: blur(10px); border-radius: 20px; margin: 0 auto 25px; max-width: 1400px; padding: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
            <div class="prompt-section">
                <div class="prompt-header" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <i class="fas fa-file-alt" style="font-size: 24px; color: #667eea;"></i>
                        <h2 style="margin: 0; color: #333; font-size: 24px; font-weight: 700;">Prompt</h2>
                    </div>
                </div>
                
                <div class="prompt-content">
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px; border-left: 4px solid #667eea;">
                        <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.5; color: #333; white-space: pre-wrap; overflow-x: auto;">
{formatted_prompt}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    """


def generate_model_stats_html(model_stats: Dict[str, Dict[str, Any]], 
                             primary_color: str = "#667eea", 
                             is_combined: bool = False) -> str:
    """Generate HTML for model statistics table."""
    if not model_stats:
        return ""
    
    if is_combined:
        return generate_combined_model_stats_html(model_stats, primary_color)
    
    model_stats_html = f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); backdrop-filter: blur(10px); border-radius: 20px; margin: 0 auto 25px; max-width: 1400px; padding: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
                <i class="fas fa-chart-line" style="font-size: 24px; color: {primary_color};"></i>
                <h2 style="margin: 0; color: #333; font-size: 24px; font-weight: 700;">Model Performance Summary</h2>
            </div>
            
            <div style="overflow-x: auto; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 15px; overflow: hidden;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%); color: white;">
                            <th style="padding: 20px; text-align: left; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-robot" style="margin-right: 8px;"></i>Model
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-trophy" style="margin-right: 8px;"></i>Score
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-hashtag" style="margin-right: 8px;"></i>Count
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-dollar-sign" style="margin-right: 8px;"></i>Mean Cost
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-clock" style="margin-right: 8px;"></i>Mean Time
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-coins" style="margin-right: 8px;"></i>Total Cost
                            </th>
                            <th style="padding: 20px; text-align: center; font-weight: 700; font-size: 16px; border: none;">
                                <i class="fas fa-stopwatch" style="margin-right: 8px;"></i>Total Time
                            </th>
                        </tr>
                    </thead>
                    <tbody>
        """
    
    # Sort models by mean score (descending), then by name
    sorted_models = sorted(model_stats.items(), key=lambda x: (x[1]['mean_score'] or -1, x[0]), reverse=True)
    
    for i, (model_name, stats) in enumerate(sorted_models):
        score_display = f"{stats['mean_score']:.1f}" if stats['mean_score'] is not None else "N/A"
        
        # Create smooth gradient from red to green based on score
        if stats['mean_score'] is not None:
            # Convert score (0-100) to hue (0=red, 120=green)
            score_percentage = min(max(stats['mean_score'], 0), 100)  # Clamp between 0-100
            hue = int(score_percentage * 1.2)  # 0-120 (red to green)
            score_bg_color = f"hsl({hue}, 70%, 50%)"
            score_text_color = "white"
        else:
            score_bg_color = "#6c757d"
            score_text_color = "white"
        
        # Alternating row colors
        row_bg = "#f8f9fa" if i % 2 == 0 else "white"
        
        model_stats_html += f"""
                        <tr style="background: {row_bg}; transition: background-color 0.3s ease;" onmouseover="this.style.backgroundColor='#e3f2fd'" onmouseout="this.style.backgroundColor='{row_bg}'">
                            <td style="padding: 20px; border: none; font-weight: 600; color: #333; font-size: 15px;">
                                {model_name}
                            </td>
                            <td style="padding: 20px; border: none; text-align: center;">
                                <div style="background: {score_bg_color}; color: {score_text_color}; padding: 8px 16px; border-radius: 12px; font-weight: 700; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: inline-block; min-width: 60px;">
                                    {score_display}
                                </div>
                            </td>
                            <td style="padding: 20px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 14px;">
                                {stats['count']}
                            </td>
                            <td style="padding: 20px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 14px;">
                                ${stats['mean_cost']:.6f}
                            </td>
                            <td style="padding: 20px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 14px;">
                                {stats['mean_time']:.1f}s
                            </td>
                            <td style="padding: 20px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 14px;">
                                ${stats['total_cost']:.4f}
                            </td>
                            <td style="padding: 20px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 14px;">
                                {stats['total_time']:.1f}s
                            </td>
                        </tr>
            """
    
    model_stats_html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    return model_stats_html 


def generate_combined_model_stats_html(model_stats: Dict[str, Dict[str, Any]], primary_color: str) -> str:
    """Generate HTML for combined model statistics with separate leaderboards."""
    
    def create_leaderboard_table(title, icon, score_field, count_field, color, precision_field=None):
        # Sort models by specific score type
        sorted_models = sorted(
            [(name, stats) for name, stats in model_stats.items() if stats.get(score_field) is not None],
            key=lambda x: x[1][score_field], reverse=True
        )
        
        if not sorted_models:
            return ""
        
        # No tooltips needed anymore
        
        table_html = f"""
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
                <i class="fas {icon}" style="font-size: 18px; color: {color};"></i>
                <h3 style="margin: 0; color: #333; font-size: 18px; font-weight: 700;">{title}</h3>
            </div>
            
            <div style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%); color: white;">
                            <th style="padding: 4px 6px; text-align: left; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Model
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                {'F1 Score' if precision_field else 'Score'}
                            </th>
                            {f'''<th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Precision
                            </th>''' if precision_field else ''}
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Count
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Avg Cost
                            </th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, (model_name, stats) in enumerate(sorted_models):
            score = stats[score_field]
            count = stats[count_field]
            
            # Color based on score
            if score >= 75:
                score_bg = "hsl(120, 70%, 50%)"
            elif score >= 50:
                score_bg = "hsl(60, 70%, 50%)" 
            elif score >= 25:
                score_bg = "hsl(30, 70%, 50%)"
            else:
                score_bg = "hsl(0, 70%, 50%)"
            
            row_bg = "#f8f9fa" if i % 2 == 0 else "white"
            
            table_html += f"""
                            <tr style="background: {row_bg};">
                                <td style="padding: 3px 6px; border: none; font-weight: 600; color: #333; font-size: 10px; line-height: 1.2;">
                                    {model_name}
                                </td>
                                <td style="padding: 3px 6px; border: none; text-align: center;">
                                    <div style="background: {score_bg}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 10px; display: inline-block; min-width: 35px; line-height: 1.2;">
                                        {score:.1f}
                                    </div>
                                </td>
                                {f'''<td style="padding: 3px 6px; border: none; text-align: center;">
                                    <div style="background: #6c757d; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 10px; display: inline-block; min-width: 35px; line-height: 1.2;">
                                        {stats.get(precision_field, 0):.1f}
                                    </div>
                                </td>''' if precision_field and stats.get(precision_field) is not None else '<td style="padding: 3px 6px; border: none; text-align: center;">-</td>' if precision_field else ''}
                                <td style="padding: 3px 6px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 10px; line-height: 1.2;">
                                    {count}
                                </td>
                                <td style="padding: 3px 6px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 10px; line-height: 1.2;">
                                    ${stats['mean_cost']:.4f}
                                </td>
                            </tr>
                """
        
        table_html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return table_html
    
    # Create scoring info section
    scoring_info = f"""
    <div style="background: rgba(255,255,255,0.9); border-radius: 6px; padding: 10px; margin-bottom: 15px; font-size: 10px; color: #666; border-left: 3px solid {primary_color};">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; text-align: center;">
            <div>
                <div style="color: #6f42c1; font-weight: 600; margin-bottom: 4px;">Galaxy Zoo Accuracy</div>
                <div style="font-family: monospace; font-size: 9px; margin-bottom: 3px;">Score = Correct / Total × 100</div>
                <div style="line-height: 1.3;">Random: 34% - Worst: 0%</div>
            </div>
            <div>
                <div style="color: #20c997; font-weight: 600; margin-bottom: 4px;">Tidal F1 Score</div>
                <div style="font-family: monospace; font-size: 9px; margin-bottom: 3px;">Score = (F1_Shell + F1_Stream) / 2</div>
                <div style="line-height: 1.3;">Random: 25% - All Stream: 33% - All Shell: 17% - All Other: 0%</div>
            </div>
            <div>
                <div style="color: #e74c3c; font-weight: 600; margin-bottom: 4px;">Lens F1 Score</div>
                <div style="font-family: monospace; font-size: 9px; margin-bottom: 3px;">F1 = 2×(P×R)/(P+R)</div>
                <div style="line-height: 1.3;"> Random: 50% - All Lens: 67% - All Non-lens: 0%</div>
            </div>
        </div>
    </div>
    """

    # Create all three leaderboards
    galaxyzoo_table = create_leaderboard_table(
        "Galaxy Zoo Accuracy", 
        "fa-project-diagram", 
        "galaxyzoo_mean_score", 
        "galaxyzoo_count",
        "#6f42c1"
    )
    
    # Create custom tidal table with Shell and Stream scores
    def create_tidal_table():
        # Sort models by average tidal F1 score
        sorted_models = sorted(
            [(name, stats) for name, stats in model_stats.items() if stats.get('tidal_f1') is not None],
            key=lambda x: x[1]['tidal_f1'], reverse=True
        )
        
        if not sorted_models:
            return ""
        
        table_html = f"""
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
                <i class="fas fa-water" style="font-size: 18px; color: #20c997;"></i>
                <h3 style="margin: 0; color: #333; font-size: 18px; font-weight: 700;">Tidal Features</h3>
            </div>
            
            <div style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #20c997 0%, #764ba2 100%); color: white;">
                            <th style="padding: 4px 6px; text-align: left; font-weight: 600; font-size: 11px; border: none; width: 25%; line-height: 1.2;">
                                Model
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; width: 12%; line-height: 1.2;">
                                Metric
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; width: 15%; line-height: 1.2;">
                                Shell
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; width: 15%; line-height: 1.2;">
                                Stream
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; width: 15%; line-height: 1.2;">
                                Avg F1
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; width: 12%; line-height: 1.2;">
                                Count
                            </th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, (model_name, stats) in enumerate(sorted_models):
            # Get metrics
            shell_p = stats.get('tidal_shell_precision', 0)
            shell_r = stats.get('tidal_shell_recall', 0)
            shell_f1 = stats.get('tidal_shell_f1', 0)
            stream_p = stats.get('tidal_stream_precision', 0)
            stream_r = stats.get('tidal_stream_recall', 0)
            stream_f1 = stats.get('tidal_stream_f1', 0)
            avg_f1 = stats['tidal_f1']
            count = stats['tidal_count']
            
            row_bg = "#f8f9fa" if i % 2 == 0 else "white"
            
            # First row: Model name + Precision
            table_html += f"""
                <tr style="background: {row_bg};">
                    <td rowspan="3" style="padding: 3px 6px; border: none; font-weight: 600; color: #333; font-size: 10px; vertical-align: middle; line-height: 1.2;">
                        {model_name}
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        P
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {shell_p:.1f}
                        </div>
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {stream_p:.1f}
                        </div>
                    </td>
                    <td rowspan="1" style="padding: 2px 4px; border: none; text-align: center; color: #adb5bd; font-size: 9px; line-height: 1.2;">
                        -
                    </td>
                    <td rowspan="3" style="padding: 3px 6px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 10px; vertical-align: middle; line-height: 1.2;">
                        {count}
                    </td>
                </tr>
                <tr style="background: {row_bg};">
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        R
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {shell_r:.1f}
                        </div>
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {stream_r:.1f}
                        </div>
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center; color: #adb5bd; font-size: 9px; line-height: 1.2;">
                        -
                    </td>
                </tr>
                <tr style="background: {row_bg}; border-bottom: 1px solid #e9ecef;">
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        F1
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #28a745; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {shell_f1:.1f}
                        </div>
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #28a745; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {stream_f1:.1f}
                        </div>
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #007bff; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 700; font-size: 9px; display: inline-block; min-width: 25px; line-height: 1.2;">
                            {avg_f1:.1f}
                        </div>
                    </td>
                </tr>
            """
        
        table_html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return table_html
    
    tidal_table = create_tidal_table()
    
    # Create custom lens table with P, R, F1 as rows
    def create_lens_table():
        # Sort models by lens F1 score
        sorted_models = sorted(
            [(name, stats) for name, stats in model_stats.items() if stats.get('lens_f1') is not None],
            key=lambda x: x[1]['lens_f1'], reverse=True
        )
        
        if not sorted_models:
            return ""
        
        table_html = f"""
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
                <i class="fas fa-eye" style="font-size: 18px; color: #e74c3c;"></i>
                <h3 style="margin: 0; color: #333; font-size: 18px; font-weight: 700;">Gravitational Lens</h3>
            </div>
            
            <div style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #e74c3c 0%, #764ba2 100%); color: white;">
                            <th style="padding: 4px 6px; text-align: left; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Model
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Metric
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Score
                            </th>
                            <th style="padding: 4px 6px; text-align: center; font-weight: 600; font-size: 11px; border: none; line-height: 1.2;">
                                Count
                            </th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, (model_name, stats) in enumerate(sorted_models):
            # Get metrics
            lens_p = stats.get('lens_precision', 0)
            lens_r = stats.get('lens_recall', 0)
            lens_f1 = stats.get('lens_f1', 0)
            count = stats['lens_count']
            
            row_bg = "#f8f9fa" if i % 2 == 0 else "white"
            
            # First row: Model name + Precision
            table_html += f"""
                <tr style="background: {row_bg};">
                    <td rowspan="3" style="padding: 3px 6px; border: none; font-weight: 600; color: #333; font-size: 10px; vertical-align: middle; line-height: 1.2;">
                        {model_name}
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        P
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 30px; line-height: 1.2;">
                            {lens_p:.1f}
                        </div>
                    </td>
                    <td rowspan="3" style="padding: 3px 6px; border: none; text-align: center; font-weight: 600; color: #495057; font-size: 10px; vertical-align: middle; line-height: 1.2;">
                        {count}
                    </td>
                </tr>
                <tr style="background: {row_bg};">
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        R
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #6c757d; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 600; font-size: 9px; display: inline-block; min-width: 30px; line-height: 1.2;">
                            {lens_r:.1f}
                        </div>
                    </td>
                </tr>
                <tr style="background: {row_bg}; border-bottom: 1px solid #e9ecef;">
                    <td style="padding: 2px 4px; border: none; text-align: center; font-weight: 600; font-size: 9px; color: #666; line-height: 1.2;">
                        F1
                    </td>
                    <td style="padding: 2px 4px; border: none; text-align: center;">
                        <div style="background: #007bff; color: white; padding: 1px 3px; border-radius: 2px; font-weight: 700; font-size: 9px; display: inline-block; min-width: 30px; line-height: 1.2;">
                            {lens_f1:.1f}
                        </div>
                    </td>
                </tr>
            """
        
        table_html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return table_html
    
    lens_table = create_lens_table()
    
    # Determine layout based on how many tables we have
    tables_to_show = [t for t in [galaxyzoo_table, tidal_table, lens_table] if t]
    
    if len(tables_to_show) == 3:
        # Three column layout
        grid_style = "display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;"
    elif len(tables_to_show) == 2:
        # Two column layout
        grid_style = "display: grid; grid-template-columns: 1fr 1fr; gap: 30px;"
    else:
        # Single column layout
        grid_style = "display: block;"
    
    return f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); backdrop-filter: blur(10px); border-radius: 15px; margin: 0 auto 15px; max-width: 1400px; padding: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <i class="fas fa-chart-line" style="font-size: 18px; color: {primary_color};"></i>
                <h2 style="margin: 0; color: #333; font-size: 18px; font-weight: 700;">Model Performance Leaderboards</h2>
            </div>
            
            {scoring_info}
            
            <div style="{grid_style}">
                {galaxyzoo_table}
                {tidal_table}
                {lens_table}
            </div>
        </div>
    """


def get_css_styles(primary_color: str = "#667eea") -> str:
    """Get the CSS styles for the HTML page."""
    return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
            padding: 20px;
            margin: 0;
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 3rem 2rem 2rem 2rem;
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            border: 1px solid white;
            overflow: visible;
        }}

        .galaxy-grid {{
            display: grid;
            gap: 3rem;
            margin-top: 2rem;
        }}

        .galaxy-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            max-width: 100%;
            overflow: visible;
            will-change: transform;
        }}

        .galaxy-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        }}

        .galaxy-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f0f0;
        }}

        .galaxy-id {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #333;
        }}

        .galaxy-icon {{
            font-size: 2rem;
            background: linear-gradient(135deg, {primary_color}, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .galaxy-content {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
            overflow: visible;
        }}

        .galaxy-top-section {{
            display: grid;
            grid-template-columns: 300px 1fr;
            grid-template-rows: auto auto;
            gap: 1.5rem;
            align-items: start;
        }}

        .galaxy-top-section.two-models {{
            grid-template-columns: 300px 1fr 1fr;
            gap: 1rem;
        }}

        .galaxy-left-column {{
            grid-row: 1 / -1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}

        .galaxy-image {{
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
            background: white;
        }}

        .galaxy-image:hover {{
            transform: scale(1.02);
        }}

        .model-selector {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1rem;
            border: 2px solid #e9ecef;
        }}

        .model-selector h3 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #495057;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .model-dropdown {{
            width: 100%;
            padding: 0.8rem;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            font-size: 0.9rem;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }}

        .model-result-container {{
            grid-row: 1;
            display: flex;
            flex-direction: column;
        }}

        .decision-tree-container {{
            grid-row: 2;
            background: white;
            border-radius: 15px;
            border: 2px solid #e9ecef;
            min-height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #adb5bd;
            font-style: italic;
            overflow: visible;
        }}

        .decision-tree-container.has-content {{
            align-items: stretch;
            justify-content: stretch;
            color: inherit;
            font-style: normal;
            padding: 1rem;
            overflow: visible;
        }}

        .model-result {{
            background: white;
            border-radius: 15px;
            padding: 1rem;
            border: 2px solid #e9ecef;
            flex: 1;
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
            overflow: visible;
            word-wrap: break-word;
            min-height: 200px;
        }}

        .model-result.active {{
            border-color: {primary_color};
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}

        .model-result.empty {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: #adb5bd;
            font-style: italic;
            min-height: 200px;
        }}

        .model-result h4 {{
            font-size: 1rem;
            font-weight: 600;
            color: {primary_color};
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e9ecef;
            flex-shrink: 0;
        }}

        .model-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}

        .model-result .markdown-content {{
            flex: 1;
            overflow: visible;
            max-height: none;
        }}

        .model-result .cost-info {{
            flex-shrink: 0;
            margin-top: 1rem;
        }}

        .decision-tree-section {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            border: 2px solid #dee2e6;
            width: 100%;
            overflow: visible;
        }}

        .decision-tree-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }}

        .decision-tree-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .decision-tree-score {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.1rem;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .decision-tree-score.poor {{
            background: linear-gradient(135deg, #dc3545, #c82333);
        }}

        .decision-tree-score.fair {{
            background: linear-gradient(135deg, #fd7e14, #e55a00);
        }}

        .decision-tree-score.good {{
            background: linear-gradient(135deg, #28a745, #1e7e34);
        }}

        .decision-tree-score.excellent {{
            background: linear-gradient(135deg, #007bff, #0056b3);
        }}

        .decision-tree-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }}

        .decision-path {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 5px solid;
        }}

        .decision-path.judge {{
            border-left-color: {primary_color};
        }}

        .decision-path.volunteer {{
            border-left-color: #28a745;
        }}

        .decision-path-title {{
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .decision-path.judge .decision-path-title {{
            color: {primary_color};
        }}

        .decision-path.volunteer .decision-path-title {{
            color: #28a745;
        }}

        .path-steps {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .path-step {{
            background: #f8f9fa;
            padding: 0.8rem 1rem 0.8rem 2.5rem;
            border-radius: 8px;
            border-left: 3px solid;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            position: relative;
        }}

        .path-step.match {{
            background: #d4edda;
            border-left-color: #28a745;
            color: #155724;
        }}

        .path-step.mismatch {{
            background: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }}

        .path-step.neutral {{
            border-left-color: #6c757d;
            color: #495057;
        }}

        .path-step::before {{
            content: attr(data-step);
            position: absolute;
            left: -25px;
            top: 50%;
            transform: translateY(-50%);
            background: white;
            color: {primary_color};
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.4rem;
            border-radius: 50%;
            border: 2px solid {primary_color};
            min-width: 20px;
            text-align: center;
        }}

        .cost-info {{
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.5rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }}

        .cost-badge {{
            background: #e9ecef;
            padding: 0.2rem 0.5rem;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        @media (max-width: 1400px) {{
            .galaxy-top-section,
            .galaxy-top-section.two-models {{
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }}
            
            .galaxy-left-column {{
                flex-direction: row;
                align-items: start;
            }}
            
            .galaxy-image {{
                max-height: 300px;
                flex: 1;
            }}
            
            .model-selector {{
                flex: 1;
            }}
        }}

        @media (max-width: 900px) {{
            .galaxy-left-column {{
                flex-direction: column;
            }}
            
            .galaxy-image {{
                max-height: 350px;
            }}
            
            .decision-tree-comparison {{
                grid-template-columns: 1fr;
                gap: 1rem;
            }}
        }}

        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @media (max-width: 800px) {{
            .floating-toc {{
                display: none;
            }}
            body {{
                padding-left: 20px;
            }}
        }}

        /* Lens analysis specific styles */
        .lens-analysis-section {{
            padding: 1.5rem;
        }}

        .lens-result {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 5px solid #e74c3c;
        }}

        /* Legacy styles - keeping for backward compatibility */
        .lens-result.lens-mentioned {{
            border-left-color: #28a745;
        }}

        .lens-result.lens-not-mentioned {{
            border-left-color: #dc3545;
        }}

        /* New styles for lens cases (A, B, C classes) */
        .lens-result.lens-correct {{
            border-left-color: #28a745;
            background: #f8fff9;
        }}

        .lens-result.lens-incorrect {{
            border-left-color: #dc3545;
            background: #fff8f8;
        }}

        /* New styles for non-lens cases (N class) */
        .lens-result.non-lens-correct {{
            border-left-color: #17a2b8;
            background: #f8feff;
        }}

        .lens-result.non-lens-incorrect {{
            border-left-color: #ffc107;
            background: #fffef8;
        }}

        .lens-result-header {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Legacy header styles */
        .lens-result.lens-mentioned .lens-result-header {{
            color: #28a745;
        }}

        .lens-result.lens-not-mentioned .lens-result-header {{
            color: #dc3545;
        }}

        /* New header styles for lens cases */
        .lens-result.lens-correct .lens-result-header {{
            color: #28a745;
        }}

        .lens-result.lens-incorrect .lens-result-header {{
            color: #dc3545;
        }}

        /* New header styles for non-lens cases */
        .lens-result.non-lens-correct .lens-result-header {{
            color: #17a2b8;
        }}

        .lens-result.non-lens-incorrect .lens-result-header {{
            color: #856404;
        }}

        .lens-details {{
            display: grid;
            gap: 0.5rem;
        }}

        .detail-item {{
            font-size: 0.95rem;
            color: #495057;
        }}

        .detail-item strong {{
            color: #333;
        }}
    """ 


def generate_html(eval_config, judged_file: Optional[str] = None) -> str:
    """Generate the HTML interface for the given evaluation type."""
    descriptions, images, judging_results, prompt_info = load_data(
        eval_config, judged_file
    )
    
    # Calculate model statistics
    score_field = eval_config.get_score_field_name()
    model_stats = calculate_model_stats(descriptions, judging_results, score_field)
    
    # Count objects in each data source
    desc_count = len(descriptions)
    image_count = len(images)
    judging_count = len(judging_results)
    
    print(f"Data counts: Descriptions: {desc_count}, Images: {image_count}, Judging: {judging_count}")
    
    # Get all unique model names and use first two as defaults
    all_models = set()
    for galaxy_descriptions in descriptions.values():
        all_models.update(galaxy_descriptions.keys())
    all_models = sorted(list(all_models))
    default_model_1 = all_models[0] if len(all_models) > 0 else ""
    default_model_2 = ""  # No default for model 2
    
    # Get all galaxies that have images AND descriptions
    galaxies = []
    for object_id in images.keys():
        if object_id in descriptions:
            # Get first description to extract galaxy info
            first_desc = list(descriptions[object_id].values())[0]
            galaxy_data = {
                'object_id': object_id,
                'image': images[object_id],
                'descriptions': descriptions.get(object_id, {}),
            }
            # Add any fields from the first description that aren't already in galaxy_data
            for key, value in first_desc.items():
                if key not in galaxy_data and key not in ['result', 'formatted_name', 'model_name', 
                                                          'cost', 'time', 'image_path', 'prompt', 
                                                          'plot_script', 'thinking_budget', 'reasoning_effort']:
                    galaxy_data[key] = value
            
            galaxies.append(galaxy_data)
    
    # Sort galaxies by object_id
    galaxies.sort(key=lambda x: x['object_id'])
    
    print(f"Found {len(galaxies)} galaxies with complete data (images and descriptions)")
    
    if not galaxies:
        print("WARNING: No galaxies found with complete data! Check your data files.")
    
    # Get configuration values
    primary_color = eval_config.get_primary_color()
    display_title = eval_config.get_display_title()
    galaxy_icon = eval_config.get_galaxy_icon()
    
    # Check if this is a combined view
    is_combined = eval_config.name == "all"
    
    if is_combined:
        return generate_combined_html(galaxies, all_models, default_model_1, default_model_2, 
                                     model_stats, prompt_info, primary_color, display_title, galaxy_icon, eval_config,
                                     descriptions, judging_results, score_field)
    else:
        return generate_single_eval_html(galaxies, all_models, default_model_1, default_model_2, 
                                        model_stats, prompt_info, primary_color, display_title, galaxy_icon, eval_config,
                                        descriptions, judging_results, score_field, images, len(judging_results))


def generate_combined_html(galaxies, all_models, default_model_1, default_model_2, 
                           model_stats, prompt_info, primary_color, display_title, galaxy_icon, eval_config,
                           descriptions, judging_results, score_field):
    """Generate combined HTML for multiple evaluation types."""
    
    # Separate galaxies by eval type
    galaxyzoo_galaxies = []
    tidal_galaxies = []
    lens_galaxies = []
    
    for galaxy in galaxies:
        # Check the first description to determine eval type
        first_desc = list(galaxy['descriptions'].values())[0] if galaxy['descriptions'] else {}
        if 'decision_tree' in first_desc:
            galaxyzoo_galaxies.append(galaxy)
        elif 'tidal_info' in first_desc or 'tidal_score' in first_desc:
            tidal_galaxies.append(galaxy)
        elif 'lensgrade' in first_desc or 'description_says_lens_occuring_score' in first_desc:
            lens_galaxies.append(galaxy)
    
    # Generate model statistics HTML
    model_stats_html = generate_model_stats_html(model_stats, primary_color, is_combined=True)
    
    # Generate prompt HTML
    prompt_html = generate_prompt_html(prompt_info)
    
    # Get CSS styles
    css_styles = get_css_styles(primary_color)
    
    # Get sort options for combined view
    base_sort_options = [
        {"value": "default", "label": "Default Order"},
        {"value": "model1-score", "label": "Model 1 Score ↓"},
        {"value": "model1-score-low", "label": "Model 1 Score ↑"},
        {"value": "model2-score", "label": "Model 2 Score ↓"},
        {"value": "model2-score-low", "label": "Model 2 Score ↑"},
        {"value": "galaxy-id", "label": "Galaxy ID A→Z"}
    ]
    
    sort_options_html = '\n'.join([
        f'<option value="{opt["value"]}">{opt["label"]}</option>'
        for opt in base_sort_options
    ])
    
    # Start building HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Evaluations Viewer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        {css_styles}
        
        /* Additional styles for combined view */
        .section-header {{
            background: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 15px;
            margin: 30px auto 20px;
            max-width: 1400px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            text-align: center;
        }}
        
        .section-title {{
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .section-subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 8px 0 0 0;
        }}
        
        /* Floating Table of Contents */
        .floating-toc {{
            position: fixed;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            z-index: 1000;
            width: 200px;
            max-height: 70vh;
            overflow-y: auto;
        }}
        
        .floating-toc h3 {{
            color: #333;
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .toc-item {{
            background: white;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }}
        
        .toc-item:hover {{
            transform: translateX(3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }}
        
        .toc-item.galaxyzoo {{
            border-left-color: #6f42c1;
        }}
        
        .toc-item.tidal {{
            border-left-color: #20c997;
        }}
        
        .toc-item.lens {{
            border-left-color: #e74c3c;
        }}
        
        .toc-item-title {{
            font-weight: 700;
            font-size: 0.85rem;
            margin: 0 0 3px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .toc-item.galaxyzoo .toc-item-title {{
            color: #6f42c1;
        }}
        
        .toc-item.tidal .toc-item-title {{
            color: #20c997;
        }}
        
        .toc-item.lens .toc-item-title {{
            color: #e74c3c;
        }}
        
        .toc-item-count {{
            color: #666;
            font-size: 0.75rem;
            margin: 0 0 8px 0;
        }}
        
        .toc-item-link {{
            text-decoration: none;
            font-weight: 600;
            font-size: 0.8rem;
            padding: 6px 10px;
            border-radius: 6px;
            display: inline-block;
            transition: all 0.3s ease;
        }}
        
        .toc-item.galaxyzoo .toc-item-link {{
            color: #6f42c1;
            background: rgba(111, 66, 193, 0.1);
        }}
        
        .toc-item.galaxyzoo .toc-item-link:hover {{
            background: rgba(111, 66, 193, 0.2);
        }}
        
        .toc-item.tidal .toc-item-link {{
            color: #20c997;
            background: rgba(32, 201, 151, 0.1);
        }}
        
        .toc-item.tidal .toc-item-link:hover {{
            background: rgba(32, 201, 151, 0.2);
        }}
        
        .toc-item.lens .toc-item-link {{
            color: #e74c3c;
            background: rgba(231, 76, 60, 0.1);
        }}
        
        .toc-item.lens .toc-item-link:hover {{
            background: rgba(231, 76, 60, 0.2);
        }}
        
        /* Adjust main content for floating TOC */
        body {{
            padding-left: 230px;
        }}
        
        @media (max-width: 800px) {{
            .floating-toc {{
                display: none;
            }}
            body {{
                padding-left: 20px;
            }}
        }}
    </style>
</head>
<body>

    <!-- Floating Table of Contents -->
    <div class="floating-toc">
        <h3>
            <i class="fas fa-list"></i>
            Table of Contents
        </h3>
        <div class="toc-item galaxyzoo">
            <div class="toc-item-title">
                <i class="fas fa-project-diagram"></i>
                Galaxy Zoo
            </div>
            <div class="toc-item-count">{len(galaxyzoo_galaxies)} galaxies</div>
            <a href="#galaxyzoo-section" class="toc-item-link">
                → View Results
            </a>
        </div>
        <div class="toc-item tidal">
            <div class="toc-item-title">
                <i class="fas fa-water"></i>
                Tidal Features
            </div>
            <div class="toc-item-count">{len(tidal_galaxies)} galaxies</div>
            <a href="#tidal-section" class="toc-item-link">
                → View Results
            </a>
        </div>
        <div class="toc-item lens">
            <div class="toc-item-title">
                <i class="fas fa-eye"></i>
                Gravitational Lens Detection
            </div>
            <div class="toc-item-count">{len(lens_galaxies)} galaxies</div>
            <a href="#lens-section" class="toc-item-link">
                → View Results
            </a>
        </div>
    </div>

    <h1 style="text-align: center; color: white; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">All Evaluations Viewer</h1>

    {model_stats_html}

    {prompt_html}

    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); backdrop-filter: blur(10px); border-radius: 20px; margin: 0 auto 25px; max-width: 1400px; padding: 20px 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 30px; flex-wrap: wrap;">
            <!-- Data Stats -->
            <div style="display: flex; align-items: center; gap: 25px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, {primary_color}, #764ba2); border-radius: 50%; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Total Galaxies: <span style="color: {primary_color};">{len(galaxies)}</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, #6f42c1, #5a2c8a); border-radius: 50%; box-shadow: 0 2px 8px rgba(111, 66, 193, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Galaxy Zoo: <span style="color: #6f42c1;">{len(galaxyzoo_galaxies)}</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, #20c997, #17a2b8); border-radius: 50%; box-shadow: 0 2px 8px rgba(32, 201, 151, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Tidal: <span style="color: #20c997;">{len(tidal_galaxies)}</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, #e74c3c, #c53030); border-radius: 50%; box-shadow: 0 2px 8px rgba(231, 76, 60, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Lens: <span style="color: #e74c3c;">{len(lens_galaxies)}</span></span>
                </div>
            </div>

            <!-- Sort Control -->
            <div style="display: flex; align-items: center; gap: 12px;">
                <i class="fas fa-sort" style="color: {primary_color}; font-size: 16px;"></i>
                <span style="font-size: 14px; font-weight: 600; color: #495057;">Sort:</span>
                <div style="position: relative;">
                    <select id="sort-select" style="
                        appearance: none; 
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        border: 2px solid rgba(102, 126, 234, 0.2); 
                        border-radius: 12px; 
                        padding: 8px 35px 8px 15px; 
                        font-size: 13px; 
                        font-weight: 500; 
                        color: #495057; 
                        cursor: pointer; 
                        transition: all 0.3s ease; 
                        min-width: 180px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    " onmouseover="this.style.borderColor='rgba(102, 126, 234, 0.4)'; this.style.boxShadow='0 4px 12px rgba(102, 126, 234, 0.1)'" onmouseout="this.style.borderColor='rgba(102, 126, 234, 0.2)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
                        {sort_options_html}
                    </select>
                    <i class="fas fa-chevron-down" style="
                        position: absolute; 
                        right: 12px; 
                        top: 50%; 
                        transform: translateY(-50%); 
                        color: {primary_color}; 
                        font-size: 12px; 
                        pointer-events: none;
                    "></i>
                </div>
            </div>
        </div>
    </div>

    <!-- Galaxy Zoo Section -->
    <div id="galaxyzoo-section">
        <div class="section-header">
            <div class="section-title">
                <i class="fas fa-project-diagram"></i>
                Galaxy Zoo Classification
            </div>
            <div class="section-subtitle">{len(galaxyzoo_galaxies)} galaxies evaluated for morphological features</div>
        </div>
        
        <div class="container">
            <div class="galaxy-grid">
    """
    
    # Add Galaxy Zoo galaxies
    for i, galaxy in enumerate(galaxyzoo_galaxies):
        model_options_slot1 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_1 else ""}>{model}</option>' 
            for model in all_models
        ])
        model_options_slot2 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_2 else ""}>{model}</option>' 
            for model in all_models
        ])
        
        # Use galaxy zoo config for formatting
        from .config.galaxyzoo import GalaxyZooConfig
        gz_config = GalaxyZooConfig()
        header_info = gz_config.format_galaxy_header_info(galaxy)
        metadata_section = gz_config.format_galaxy_metadata(galaxy)
        
        html_content += f"""
            <div class="galaxy-card" data-eval-type="galaxyzoo">
                <div class="galaxy-header">
                    <i class="fas fa-project-diagram galaxy-icon"></i>
                    <div class="galaxy-id">{galaxy['object_id']}</div>
                    {header_info}
                </div>
                
                {metadata_section}
                
                <div class="galaxy-content">
                    <div class="galaxy-top-section" id="top-section-{galaxy['object_id']}">
                        <div class="galaxy-left-column">
                            <img src="{galaxy['image']}" alt="Galaxy {galaxy['object_id']}" class="galaxy-image">
                            
                            <div class="model-selector">
                                <h3><i class="fas fa-cog"></i> Model Selection</h3>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 1:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="1">
                                        <option value="">Select a model...</option>
                                        {model_options_slot1}
                                    </select>
                                </div>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 2:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="2">
                                        <option value="">None</option>
                                        {model_options_slot2}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-1">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_1} description...
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-2" style="display: none;">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_2} description...
                            </div>
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-1">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-2" style="display: none;">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>
                    </div>
                </div>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    </div>

    <!-- Tidal Features Section -->
    <div id="tidal-section">
        <div class="section-header">
            <div class="section-title">
                <i class="fas fa-water"></i>
                Tidal Features Detection
            </div>
            <div class="section-subtitle">{} galaxies evaluated for tidal disruption features</div>
        </div>
        
        <div class="container">
            <div class="galaxy-grid">
    """.format(len(tidal_galaxies))
    
    # Add Tidal galaxies
    for i, galaxy in enumerate(tidal_galaxies):
        model_options_slot1 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_1 else ""}>{model}</option>' 
            for model in all_models
        ])
        model_options_slot2 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_2 else ""}>{model}</option>' 
            for model in all_models
        ])
        
        # Use tidal config for formatting
        from .config.tidal import TidalConfig
        tidal_config = TidalConfig()
        header_info = tidal_config.format_galaxy_header_info(galaxy)
        metadata_section = tidal_config.format_galaxy_metadata(galaxy)
        
        html_content += f"""
            <div class="galaxy-card" data-eval-type="tidal">
                <div class="galaxy-header">
                    <i class="fas fa-water galaxy-icon"></i>
                    <div class="galaxy-id">{galaxy['object_id']}</div>
                    {header_info}
                </div>
                
                {metadata_section}
                
                <div class="galaxy-content">
                    <div class="galaxy-top-section" id="top-section-{galaxy['object_id']}">
                        <div class="galaxy-left-column">
                            <img src="{galaxy['image']}" alt="Galaxy {galaxy['object_id']}" class="galaxy-image">
                            
                            <div class="model-selector">
                                <h3><i class="fas fa-cog"></i> Model Selection</h3>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 1:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="1">
                                        <option value="">Select a model...</option>
                                        {model_options_slot1}
                                    </select>
                                </div>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 2:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="2">
                                        <option value="">None</option>
                                        {model_options_slot2}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-1">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_1} description...
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-2" style="display: none;">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_2} description...
                            </div>
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-1">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-2" style="display: none;">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>
                    </div>
                </div>
            </div>
        """
    
    # Close HTML and add JavaScript
    html_content += f"""
            </div>
        </div>
    </div>

    <!-- Lens Section -->
    <div id="lens-section">
        <div class="section-header">
            <div class="section-title">
                <i class="fas fa-eye"></i>
                Gravitational Lens Detection
            </div>
            <div class="section-subtitle">{len(lens_galaxies)} galaxies evaluated for gravitational lens detection</div>
        </div>
        
        <div class="container">
            <div class="galaxy-grid">
    """.format(len(lens_galaxies))
    
    # Add Lens galaxies
    for i, galaxy in enumerate(lens_galaxies):
        model_options_slot1 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_1 else ""}>{model}</option>' 
            for model in all_models
        ])
        model_options_slot2 = ''.join([
            f'<option value="{model}"{" selected" if model == default_model_2 else ""}>{model}</option>' 
            for model in all_models
        ])
        
        # Use lens config for formatting
        from .config.lens import LensConfig
        lens_config = LensConfig()
        header_info = lens_config.format_galaxy_header_info(galaxy)
        metadata_section = lens_config.format_galaxy_metadata(galaxy)
        
        html_content += f"""
            <div class="galaxy-card" data-eval-type="lens">
                <div class="galaxy-header">
                    <i class="fas fa-eye galaxy-icon"></i>
                    <div class="galaxy-id">{galaxy['object_id']}</div>
                    {header_info}
                </div>
                
                {metadata_section}
                
                <div class="galaxy-content">
                    <div class="galaxy-top-section" id="top-section-{galaxy['object_id']}">
                        <div class="galaxy-left-column">
                            <img src="{galaxy['image']}" alt="Galaxy {galaxy['object_id']}" class="galaxy-image">
                            
                            <div class="model-selector">
                                <h3><i class="fas fa-cog"></i> Model Selection</h3>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 1:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="1">
                                        <option value="">Select a model...</option>
                                        {model_options_slot1}
                                    </select>
                                </div>
                                <div>
                                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 2:</label>
                                    <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="2">
                                        <option value="">None</option>
                                        {model_options_slot2}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-1">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_1} description...
                            </div>
                        </div>

                        <div class="model-result-container" id="result-{galaxy['object_id']}-2" style="display: none;">
                            <div class="model-result empty">
                                <i class="fas fa-robot"></i> Loading {default_model_2} description...
                            </div>
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-1">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>

                        <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-2" style="display: none;">
                            <i class="fas fa-sitemap"></i> No analysis available
                        </div>
                    </div>
                </div>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    </div>

    <script>
        // Galaxy descriptions and judging results data
        const galaxyDescriptions = """ + json.dumps(descriptions, indent=2) + """;
        const judgingResults = """ + json.dumps(judging_results, indent=2) + """;
        const scoreFieldName = "dynamic";  // Handle multiple score fields dynamically
        const evalType = "all";

        // Store original order of galaxy cards
        let originalGalaxyOrder = [];

        // Include JavaScript functions for combined view
        """ + generate_combined_javascript() + """
    </script>
</body>
</html>
"""
    
    return html_content


def generate_single_eval_html(galaxies, all_models, default_model_1, default_model_2, 
                             model_stats, prompt_info, primary_color, display_title, galaxy_icon, eval_config,
                             descriptions, judging_results, score_field, images, judging_count):
    # Generate model statistics HTML
    model_stats_html = generate_model_stats_html(model_stats, primary_color)
    
    # Generate prompt HTML
    prompt_html = generate_prompt_html(prompt_info)
    
    # Get CSS styles
    css_styles = get_css_styles(primary_color)
    
    # Get sort options including eval-specific ones
    base_sort_options = [
        {"value": "default", "label": "Default Order"},
        {"value": "model1-score", "label": "Model 1 Score ↓"},
        {"value": "model1-score-low", "label": "Model 1 Score ↑"},
        {"value": "model2-score", "label": "Model 2 Score ↓"},
        {"value": "model2-score-low", "label": "Model 2 Score ↑"},
    ]
    
    eval_sort_options = eval_config.get_sort_options()
    all_sort_options = base_sort_options + eval_sort_options + [
        {"value": "galaxy-id", "label": "Galaxy ID A→Z"}
    ]
    
    sort_options_html = '\n'.join([
        f'<option value="{opt["value"]}">{opt["label"]}</option>'
        for opt in all_sort_options
    ])
    
    # Start building HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{display_title}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        {css_styles}
    </style>
</head>
<body>

    <h1 style="text-align: center; color: white; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">{display_title}</h1>

    {model_stats_html}

    {prompt_html}

    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); backdrop-filter: blur(10px); border-radius: 20px; margin: 0 auto 25px; max-width: 1400px; padding: 20px 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 30px; flex-wrap: wrap;">
            <!-- Data Stats -->
            <div style="display: flex; align-items: center; gap: 25px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, {primary_color}, #764ba2); border-radius: 50%; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Descriptions: <span style="color: {primary_color};">{len(galaxies)}</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, #28a745, #20c997); border-radius: 50%; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Images: <span style="color: #28a745;">{len(images)}</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(135deg, #fd7e14, #e55a00); border-radius: 50%; box-shadow: 0 2px 8px rgba(253, 126, 20, 0.3);"></div>
                    <span style="font-size: 14px; font-weight: 600; color: #495057;">Judging: <span style="color: #fd7e14;">{judging_count}</span></span>
                </div>
                <div style="height: 20px; width: 1px; background: rgba(108, 117, 125, 0.2);"></div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <i class="fas fa-database" style="color: {primary_color}; font-size: 14px;"></i>
                    <span style="font-size: 14px; font-weight: 700; color: #333;">Total: <span style="color: {primary_color};">{len(galaxies)}</span></span>
                </div>
            </div>

            <!-- Sort Control -->
            <div style="display: flex; align-items: center; gap: 12px;">
                <i class="fas fa-sort" style="color: {primary_color}; font-size: 16px;"></i>
                <span style="font-size: 14px; font-weight: 600; color: #495057;">Sort:</span>
                <div style="position: relative;">
                    <select id="sort-select" style="
                        appearance: none; 
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        border: 2px solid rgba(102, 126, 234, 0.2); 
                        border-radius: 12px; 
                        padding: 8px 35px 8px 15px; 
                        font-size: 13px; 
                        font-weight: 500; 
                        color: #495057; 
                        cursor: pointer; 
                        transition: all 0.3s ease; 
                        min-width: 180px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    " onmouseover="this.style.borderColor='rgba(102, 126, 234, 0.4)'; this.style.boxShadow='0 4px 12px rgba(102, 126, 234, 0.1)'" onmouseout="this.style.borderColor='rgba(102, 126, 234, 0.2)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'">
                        {sort_options_html}
                    </select>
                    <i class="fas fa-chevron-down" style="
                        position: absolute; 
                        right: 12px; 
                        top: 50%; 
                        transform: translateY(-50%); 
                        color: {primary_color}; 
                        font-size: 12px; 
                        pointer-events: none;
                    "></i>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="galaxy-grid">
"""
    
    # Generate galaxy cards
    if not galaxies:
        html_content += """
            <div style="text-align: center; background: white; padding: 30px; border-radius: 10px;">
                <h2 style="color: #dc3545;">No Galaxies Found</h2>
                <p>No galaxies with complete data (images and descriptions) were found.</p>
            </div>
        """
    else:
        for i, galaxy in enumerate(galaxies):
            # Create model options with only first two models as default
            model_options_slot1 = ''.join([
                f'<option value="{model}"{" selected" if model == default_model_1 else ""}>{model}</option>' 
                for model in all_models
            ])
            model_options_slot2 = ''.join([
                f'<option value="{model}"{" selected" if model == default_model_2 else ""}>{model}</option>' 
                for model in all_models
            ])
            
            # Get header info and metadata from config
            header_info = eval_config.format_galaxy_header_info(galaxy)
            metadata_section = eval_config.format_galaxy_metadata(galaxy)
            
            html_content += f"""
                <div class="galaxy-card fade-in" style="animation-delay: {i * 0.1}s">
                    <div class="galaxy-header">
                        <i class="fas {galaxy_icon} galaxy-icon"></i>
                        <div class="galaxy-id">{galaxy['object_id']}</div>
                        {header_info}
                    </div>
                    
                    {metadata_section}
                    
                    <div class="galaxy-content">
                        <div class="galaxy-top-section" id="top-section-{galaxy['object_id']}">
                            <div class="galaxy-left-column">
                                <img src="{galaxy['image']}" alt="Galaxy {galaxy['object_id']}" class="galaxy-image">
                                
                                <div class="model-selector">
                                    <h3><i class="fas fa-cog"></i> Model Selection</h3>
                                    <div>
                                        <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 1:</label>
                                        <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="1">
                                            <option value="">Select a model...</option>
                                            {model_options_slot1}
                                        </select>
                                    </div>
                                    <div>
                                        <label style="display: block; margin-bottom: 0.5rem; font-weight: 500;">Model 2:</label>
                                        <select class="model-dropdown" data-galaxy="{galaxy['object_id']}" data-slot="2">
                                            <option value="">None</option>
                                            {model_options_slot2}
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <div class="model-result-container" id="result-{galaxy['object_id']}-1">
                                <div class="model-result empty">
                                    <i class="fas fa-robot"></i> Loading {default_model_1} description...
                                </div>
                            </div>

                            <div class="model-result-container" id="result-{galaxy['object_id']}-2" style="display: none;">
                                <div class="model-result empty">
                                    <i class="fas fa-robot"></i> Loading {default_model_2} description...
                                </div>
                            </div>

                            <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-1">
                                <i class="fas fa-sitemap"></i> No analysis available
                            </div>

                            <div class="decision-tree-container" id="decision-tree-{galaxy['object_id']}-2" style="display: none;">
                                <i class="fas fa-sitemap"></i> No analysis available
                            </div>
                        </div>
                    </div>
                </div>
            """
    
    # Continue with JavaScript
    html_content += f"""
        </div>
    </div>

    <script>
        // Galaxy descriptions and judging results data
        const galaxyDescriptions = {json.dumps(descriptions, indent=2)};
        const judgingResults = {json.dumps(judging_results, indent=2)};
        const scoreFieldName = "{score_field}";
        const evalType = "{eval_config.name}";

        // Store original order of galaxy cards
        let originalGalaxyOrder = [];

        // Include JavaScript functions here
        """ + generate_javascript(eval_config) + """
    </script>
</body>
</html>
"""
    
    return html_content 


def load_javascript_from_file(js_file_path: str) -> str:
    """Load JavaScript content from an external file."""
    try:
        # Get the directory where display.py is located
        display_dir = Path(__file__).parent
        # Construct the full path to the JS file
        js_path = display_dir / js_file_path
        
        if js_path.exists():
            with open(js_path, 'r') as f:
                return f.read()
        else:
            print(f"Warning: JavaScript file not found: {js_path}")
            return ""
    except Exception as e:
        print(f"Error loading JavaScript file {js_file_path}: {e}")
        return ""


def generate_javascript(eval_config) -> str:
    """Generate JavaScript code for the display."""
    # Load eval-specific JavaScript from external file if available
    comparison_js = ""
    if hasattr(eval_config, 'js_file') and eval_config.js_file:
        comparison_js = load_javascript_from_file(eval_config.js_file)
    else:
        # Fallback to the old method for backward compatibility
        comparison_js = eval_config.get_comparison_javascript()
    
    comparison_function_name = eval_config.get_comparison_function_name()
    
    return f"""
        // Function to get current model names for a galaxy
        function getCurrentModels(galaxyId) {{
            const model1Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="1"]`);
            const model2Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="2"]`);
            return {{
                model1: model1Dropdown ? model1Dropdown.value : '',
                model2: model2Dropdown ? model2Dropdown.value : ''
            }};
        }}

        // Function to get score for a galaxy and model
        function getScore(galaxyId, modelName) {{
            if (!modelName || !judgingResults[galaxyId] || !judgingResults[galaxyId][modelName]) {{
                return -1; // No score available
            }}
            const score = judgingResults[galaxyId][modelName][scoreFieldName];
            return score !== null && score !== undefined ? score : -1;
        }}

        // Function to sort galaxies
        function sortGalaxies(criteria) {{
            const galaxyGrid = document.querySelector('.galaxy-grid');
            const galaxyCards = Array.from(galaxyGrid.children);
            
            if (criteria === 'default') {{
                // Restore original order
                originalGalaxyOrder.forEach(card => galaxyGrid.appendChild(card));
                return;
            }}

            galaxyCards.sort((a, b) => {{
                const galaxyIdA = a.querySelector('.galaxy-id').textContent.trim();
                const galaxyIdB = b.querySelector('.galaxy-id').textContent.trim();
                
                const modelsA = getCurrentModels(galaxyIdA);
                const modelsB = getCurrentModels(galaxyIdB);

                switch (criteria) {{
                    case 'model1-score':
                        const scoreA1 = getScore(galaxyIdA, modelsA.model1);
                        const scoreB1 = getScore(galaxyIdB, modelsB.model1);
                        return scoreB1 - scoreA1; // Highest first

                    case 'model1-score-low':
                        const scoreA1Low = getScore(galaxyIdA, modelsA.model1);
                        const scoreB1Low = getScore(galaxyIdB, modelsB.model1);
                        return scoreA1Low - scoreB1Low; // Lowest first

                    case 'model2-score':
                        const scoreA2 = getScore(galaxyIdA, modelsA.model2);
                        const scoreB2 = getScore(galaxyIdB, modelsB.model2);
                        return scoreB2 - scoreA2; // Highest first

                    case 'model2-score-low':
                        const scoreA2Low = getScore(galaxyIdA, modelsA.model2);
                        const scoreB2Low = getScore(galaxyIdB, modelsB.model2);
                        return scoreA2Low - scoreB2Low; // Lowest first

                    case 'volunteer-path-length':
                        // For Galaxy Zoo specific sorting
                        if (evalType === 'galaxyzoo') {{
                            const pathLenA = getVolunteerPathLength(galaxyIdA, modelsA.model1);
                            const pathLenB = getVolunteerPathLength(galaxyIdB, modelsB.model1);
                            return pathLenB - pathLenA; // Longest first
                        }}
                        return 0;

                    case 'volunteer-path-length-short':
                        // For Galaxy Zoo specific sorting
                        if (evalType === 'galaxyzoo') {{
                            const pathLenAShort = getVolunteerPathLength(galaxyIdA, modelsA.model1);
                            const pathLenBShort = getVolunteerPathLength(galaxyIdB, modelsB.model1);
                            return pathLenAShort - pathLenBShort; // Shortest first
                        }}
                        return 0;

                    case 'galaxy-id':
                        return galaxyIdA.localeCompare(galaxyIdB);

                    default:
                        return 0;
                }}
            }});

            // Re-append sorted cards
            galaxyCards.forEach(card => galaxyGrid.appendChild(card));
        }}

        // Function specific to Galaxy Zoo for getting volunteer path length
        function getVolunteerPathLength(galaxyId, modelName) {{
            if (!modelName || !judgingResults[galaxyId] || !judgingResults[galaxyId][modelName]) {{
                return -1;
            }}
            const judgeResults = judgingResults[galaxyId][modelName].judge_results;
            if (!judgeResults || !judgeResults.volunteer_path) {{
                return -1;
            }}
            return judgeResults.volunteer_path.length;
        }}

        // Function to update layout based on model selection
        function updateLayout(galaxyId) {{
            const topSection = document.getElementById(`top-section-${{galaxyId}}`);
            const result2Container = document.getElementById(`result-${{galaxyId}}-2`);
            const decisionTree2Container = document.getElementById(`decision-tree-${{galaxyId}}-2`);
            const model2Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="2"]`);
            
            if (model2Dropdown.value) {{
                topSection.classList.add('two-models');
                result2Container.style.display = 'block';
                decisionTree2Container.style.display = 'block';
            }} else {{
                topSection.classList.remove('two-models');
                result2Container.style.display = 'none';
                decisionTree2Container.style.display = 'none';
            }}
        }}

        // Function to toggle prompt section
        function togglePromptSection() {{
            const content = document.querySelector('.prompt-content');
            const icon = document.querySelector('.prompt-expand-icon');
            
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                icon.style.transform = 'rotate(180deg)';
            }} else {{
                content.style.display = 'none';
                icon.style.transform = 'rotate(0deg)';
            }}
        }}

        // Process default model loading
        document.addEventListener('DOMContentLoaded', function() {{
            // Store original order of galaxy cards
            const galaxyGrid = document.querySelector('.galaxy-grid');
            originalGalaxyOrder = Array.from(galaxyGrid.children);

            // Trigger loading of default selected models for both slots
            document.querySelectorAll('.model-dropdown[data-slot="1"]').forEach(dropdown => {{
                if (dropdown.value) {{
                    const event = new Event('change', {{ bubbles: true }});
                    dropdown.dispatchEvent(event);
                }}
            }});
            
            document.querySelectorAll('.model-dropdown[data-slot="2"]').forEach(dropdown => {{
                if (dropdown.value) {{
                    const event = new Event('change', {{ bubbles: true }});
                    dropdown.dispatchEvent(event);
                }}
            }});
        }});

        // Handle model selection changes
        document.addEventListener('change', function(e) {{
            if (e.target.classList.contains('model-dropdown')) {{
                const galaxyId = e.target.dataset.galaxy;
                const slot = e.target.dataset.slot;
                const modelName = e.target.value;
                const resultContainer = document.getElementById(`result-${{galaxyId}}-${{slot}}`);
                const resultDiv = resultContainer.querySelector('.model-result');
                const decisionTreeContainer = document.getElementById(`decision-tree-${{galaxyId}}-${{slot}}`);

                // Update layout
                updateLayout(galaxyId);

                if (modelName && galaxyDescriptions[galaxyId] && galaxyDescriptions[galaxyId][modelName]) {{
                    const data = galaxyDescriptions[galaxyId][modelName];
                    const judging = judgingResults[galaxyId] && judgingResults[galaxyId][modelName];
                    
                    let costTimeInfo = '';
                    if (data.cost && data.time) {{
                        costTimeInfo = `
                            <div class="cost-info">
                                <span class="cost-badge"><i class="fas fa-dollar-sign"></i> ${{data.cost.toFixed(6)}}</span>
                                <span class="cost-badge"><i class="fas fa-clock"></i> ${{data.time.toFixed(1)}}s</span>
                            </div>
                        `;
                    }}
                    
                    // Update model description
                    resultDiv.innerHTML = `
                        <h4><i class="fas fa-brain"></i> ${{modelName}}</h4>
                        <div class="model-content">
                            <div class="markdown-content">${{marked.parse(data.result || 'No description available')}}</div>
                            ${{costTimeInfo}}
                        </div>
                    `;
                    resultDiv.classList.remove('empty');
                    resultDiv.classList.add('active');

                    // Update decision tree/analysis section using eval-specific function
                    if (judging && judging[scoreFieldName] !== null && judging[scoreFieldName] !== undefined) {{
                        const score = judging[scoreFieldName];
                        const judgeResults = judging.judge_results || {{}};
                        
                        // Use eval-specific comparison function
                        const comparisonHtml = {comparison_function_name}(judgeResults, data, score, judging);
                        decisionTreeContainer.innerHTML = comparisonHtml;
                        decisionTreeContainer.classList.add('has-content');
                    }} else {{
                        decisionTreeContainer.innerHTML = '<i class="fas fa-sitemap"></i> No analysis available';
                        decisionTreeContainer.classList.remove('has-content');
                    }}
                }} else if (modelName) {{
                    resultDiv.innerHTML = `
                        <h4><i class="fas fa-exclamation-triangle"></i> ${{modelName}}</h4>
                        <div class="model-content">
                            <div class="markdown-content" style="color: #dc3545; font-style: italic;">
                                No description available for this model
                            </div>
                        </div>
                    `;
                    resultDiv.classList.remove('empty');
                    resultDiv.classList.add('active');

                    decisionTreeContainer.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No analysis available';
                    decisionTreeContainer.classList.remove('has-content');
                }} else {{
                    if (slot === "1") {{
                        resultDiv.innerHTML = '<i class="fas fa-robot"></i> Select a model to see its description';
                    }} else {{
                        resultDiv.innerHTML = '<i class="fas fa-robot"></i> Select a second model to compare';
                    }}
                    resultDiv.classList.add('empty');
                    resultDiv.classList.remove('active');

                    decisionTreeContainer.innerHTML = '<i class="fas fa-sitemap"></i> No analysis available';
                    decisionTreeContainer.classList.remove('has-content');
                }}
            }} else if (e.target.id === 'sort-select') {{
                sortGalaxies(e.target.value);
            }}
        }});

        // Include eval-specific comparison functions
        {comparison_js}

        console.log(`${{evalType}} Viewer loaded successfully!`);
        console.log(`Displaying ${{Object.keys(galaxyDescriptions).length}} galaxies`);
        console.log(`Analysis results available for ${{Object.keys(judgingResults).length}} galaxies`);
    """


def generate_combined_javascript():
    """Generate JavaScript for combined evaluation display."""
    return f"""
        // Function to get current model names for a galaxy
        function getCurrentModels(galaxyId) {{
            const model1Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="1"]`);
            const model2Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="2"]`);
            return {{
                model1: model1Dropdown ? model1Dropdown.value : '',
                model2: model2Dropdown ? model2Dropdown.value : ''
            }};
        }}

        // Function to get score for a galaxy and model
        function getScore(galaxyId, modelName, evalType) {{
            if (!modelName || !judgingResults[galaxyId] || !judgingResults[galaxyId][modelName]) {{
                return -1;
            }}
            const judgeData = judgingResults[galaxyId][modelName];
            let scoreField;
            if (evalType === 'galaxyzoo') {{
                scoreField = 'decision_tree_score';
            }} else if (evalType === 'tidal') {{
                scoreField = 'tidal_score';
            }} else if (evalType === 'lens') {{
                scoreField = 'description_says_lens_occuring_score';
            }} else {{
                return -1;
            }}
            const score = judgeData[scoreField];
            return score !== null && score !== undefined ? score : -1;
        }}

        // Function to sort galaxies
        function sortGalaxies(criteria) {{
            const galaxyCards = Array.from(document.querySelectorAll('.galaxy-card'));
            
            if (criteria === 'default') {{
                // Restore original order
                galaxyCards.forEach(card => {{
                    const evalType = card.getAttribute('data-eval-type');
                    let targetSection;
                    if (evalType === 'galaxyzoo') {{
                        targetSection = document.querySelector('#galaxyzoo-section .galaxy-grid');
                    }} else if (evalType === 'tidal') {{
                        targetSection = document.querySelector('#tidal-section .galaxy-grid');
                    }} else if (evalType === 'lens') {{
                        targetSection = document.querySelector('#lens-section .galaxy-grid');
                    }}
                    if (targetSection) {{
                        targetSection.appendChild(card);
                    }}
                }});
                return;
            }}

            galaxyCards.sort((a, b) => {{
                const galaxyIdA = a.querySelector('.galaxy-id').textContent.trim();
                const galaxyIdB = b.querySelector('.galaxy-id').textContent.trim();
                const evalTypeA = a.getAttribute('data-eval-type');
                const evalTypeB = b.getAttribute('data-eval-type');
                
                const modelsA = getCurrentModels(galaxyIdA);
                const modelsB = getCurrentModels(galaxyIdB);

                switch (criteria) {{
                    case 'model1-score':
                        const scoreA1 = getScore(galaxyIdA, modelsA.model1, evalTypeA);
                        const scoreB1 = getScore(galaxyIdB, modelsB.model1, evalTypeB);
                        return scoreB1 - scoreA1;

                    case 'model1-score-low':
                        const scoreA1Low = getScore(galaxyIdA, modelsA.model1, evalTypeA);
                        const scoreB1Low = getScore(galaxyIdB, modelsB.model1, evalTypeB);
                        return scoreA1Low - scoreB1Low;

                    case 'model2-score':
                        const scoreA2 = getScore(galaxyIdA, modelsA.model2, evalTypeA);
                        const scoreB2 = getScore(galaxyIdB, modelsB.model2, evalTypeB);
                        return scoreB2 - scoreA2;

                    case 'model2-score-low':
                        const scoreA2Low = getScore(galaxyIdA, modelsA.model2, evalTypeA);
                        const scoreB2Low = getScore(galaxyIdB, modelsB.model2, evalTypeB);
                        return scoreA2Low - scoreB2Low;

                    case 'galaxy-id':
                        return galaxyIdA.localeCompare(galaxyIdB);

                    default:
                        return 0;
                }}
            }});

            // Re-append sorted cards to their appropriate sections
            galaxyCards.forEach(card => {{
                const evalType = card.getAttribute('data-eval-type');
                let targetSection;
                if (evalType === 'galaxyzoo') {{
                    targetSection = document.querySelector('#galaxyzoo-section .galaxy-grid');
                }} else if (evalType === 'tidal') {{
                    targetSection = document.querySelector('#tidal-section .galaxy-grid');
                }} else if (evalType === 'lens') {{
                    targetSection = document.querySelector('#lens-section .galaxy-grid');
                }}
                if (targetSection) {{
                    targetSection.appendChild(card);
                }}
            }});
        }}

        // Function to update layout based on model selection
        function updateLayout(galaxyId) {{
            const topSection = document.getElementById(`top-section-${{galaxyId}}`);
            const result2Container = document.getElementById(`result-${{galaxyId}}-2`);
            const decisionTree2Container = document.getElementById(`decision-tree-${{galaxyId}}-2`);
            const model2Dropdown = document.querySelector(`[data-galaxy="${{galaxyId}}"][data-slot="2"]`);
            
            if (model2Dropdown && model2Dropdown.value) {{
                topSection.classList.add('two-models');
                result2Container.style.display = 'block';
                decisionTree2Container.style.display = 'block';
            }} else {{
                topSection.classList.remove('two-models');
                result2Container.style.display = 'none';
                decisionTree2Container.style.display = 'none';
            }}
        }}

        // Simplified function to update model result display
        function updateModelResult(galaxyId, slot, modelName) {{
            console.log(`Loading model result for galaxy ${{galaxyId}}, slot ${{slot}}, model: ${{modelName}}`);
            
            const resultContainer = document.getElementById(`result-${{galaxyId}}-${{slot}}`);
            const decisionTreeContainer = document.getElementById(`decision-tree-${{galaxyId}}-${{slot}}`);
            
            if (!resultContainer || !decisionTreeContainer) {{
                console.error(`Containers not found for galaxy ${{galaxyId}}, slot ${{slot}}`);
                return;
            }}
            
            const resultDiv = resultContainer.querySelector('.model-result');
            
            if (!modelName) {{
                resultDiv.innerHTML = slot === "1" ? 
                    '<i class="fas fa-robot"></i> Select a model to see its description' :
                    '<i class="fas fa-robot"></i> Select a second model to compare';
                resultDiv.classList.add('empty');
                resultDiv.classList.remove('active');
                decisionTreeContainer.innerHTML = '<i class="fas fa-sitemap"></i> No analysis available';
                decisionTreeContainer.classList.remove('has-content');
                return;
            }}

            // Update layout
            updateLayout(galaxyId);

            // Check if we have data for this galaxy and model
            if (!galaxyDescriptions[galaxyId]) {{
                console.error(`No data found for galaxy ${{galaxyId}}`);
                resultDiv.innerHTML = `
                    <h4><i class="fas fa-exclamation-triangle"></i> ${{modelName}}</h4>
                    <div class="model-content">
                        <div class="markdown-content" style="color: #dc3545; font-style: italic;">
                            No data found for galaxy ${{galaxyId}}
                        </div>
                    </div>
                `;
                resultDiv.classList.remove('empty');
                resultDiv.classList.add('active');
                decisionTreeContainer.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No data available';
                decisionTreeContainer.classList.remove('has-content');
                return;
            }}

            if (!galaxyDescriptions[galaxyId][modelName]) {{
                console.error(`No data found for model ${{modelName}} in galaxy ${{galaxyId}}`);
                console.log(`Available models for ${{galaxyId}}:`, Object.keys(galaxyDescriptions[galaxyId]));
                resultDiv.innerHTML = `
                    <h4><i class="fas fa-exclamation-triangle"></i> ${{modelName}}</h4>
                    <div class="model-content">
                        <div class="markdown-content" style="color: #dc3545; font-style: italic;">
                            No description available for this model in this galaxy
                        </div>
                    </div>
                `;
                resultDiv.classList.remove('empty');
                resultDiv.classList.add('active');
                decisionTreeContainer.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No data available';
                decisionTreeContainer.classList.remove('has-content');
                return;
            }}

            // Get the data
            const data = galaxyDescriptions[galaxyId][modelName];
            const judging = judgingResults[galaxyId] && judgingResults[galaxyId][modelName];
            
            console.log(`Successfully loaded data for ${{galaxyId}} - ${{modelName}}`);
            
            // Prepare cost/time info
            let costTimeInfo = '';
            if (data.cost !== undefined && data.time !== undefined) {{
                costTimeInfo = `
                    <div class="cost-info">
                        <span class="cost-badge"><i class="fas fa-dollar-sign"></i> ${{data.cost.toFixed(6)}}</span>
                        <span class="cost-badge"><i class="fas fa-clock"></i> ${{data.time.toFixed(1)}}s</span>
                    </div>
                `;
            }}
            
            // Get description and parse markdown
            const description = data.result || 'No description available';
            let markdownContent = '';
            try {{
                if (typeof marked !== 'undefined' && marked.parse) {{
                    markdownContent = marked.parse(description);
                }} else {{
                    markdownContent = description.replace(/\\n/g, '<br>');
                }}
            }} catch (error) {{
                console.error('Error parsing markdown:', error);
                markdownContent = description.replace(/\\n/g, '<br>');
            }}
            
            // Update the result display
            resultDiv.innerHTML = `
                <h4><i class="fas fa-brain"></i> ${{modelName}}</h4>
                <div class="model-content">
                    <div class="markdown-content">${{markdownContent}}</div>
                    ${{costTimeInfo}}
                </div>
            `;
            resultDiv.classList.remove('empty');
            resultDiv.classList.add('active');

            // Handle analysis section
            if (judging) {{
                // Determine eval type and score field
                const card = resultContainer.closest('.galaxy-card');
                const evalType = card ? card.getAttribute('data-eval-type') : 'unknown';
                
                let scoreField;
                if (evalType === 'galaxyzoo') {{
                    scoreField = 'decision_tree_score';
                }} else if (evalType === 'tidal') {{
                    scoreField = 'tidal_score';
                }} else if (evalType === 'lens') {{
                    scoreField = 'description_says_lens_occuring_score';
                }} else {{
                    scoreField = null;
                }}
                
                if (scoreField && judging[scoreField] !== null && judging[scoreField] !== undefined) {{
                    const score = judging[scoreField];
                    const judgeResults = judging.judge_results || {{}};
                    
                    // Use eval-specific comparison function
                    let comparisonHtml = '';
                    if (evalType === 'galaxyzoo' && typeof createDecisionTreeComparison === 'function') {{
                        comparisonHtml = createDecisionTreeComparison(judgeResults, data, score, judging);
                    }} else if (evalType === 'tidal' && typeof createTidalFeatureComparison === 'function') {{
                        comparisonHtml = createTidalFeatureComparison(judgeResults, data, score, judging);
                    }} else if (evalType === 'lens' && typeof createLensComparison === 'function') {{
                        comparisonHtml = createLensComparison(judgeResults, data, score, judging);
                    }} else {{
                        // Fallback display
                        const scoreClass = score >= 0.8 ? 'excellent' : score >= 0.6 ? 'good' : score >= 0.4 ? 'fair' : 'poor';
                        comparisonHtml = `
                            <div class="decision-tree-section">
                                <div class="decision-tree-header">
                                    <div class="decision-tree-title">
                                        <i class="fas fa-chart-line"></i>
                                        Analysis Score
                                    </div>
                                    <div class="decision-tree-score ${{scoreClass}}">
                                        <i class="fas fa-star"></i>
                                        ${{(score * 100).toFixed(1)}}%
                                    </div>
                                </div>
                                <p>Detailed analysis for ${{evalType}} evaluation.</p>
                            </div>
                        `;
                    }}
                    
                    decisionTreeContainer.innerHTML = comparisonHtml;
                    decisionTreeContainer.classList.add('has-content');
                }} else {{
                    decisionTreeContainer.innerHTML = '<i class="fas fa-sitemap"></i> No analysis score available';
                    decisionTreeContainer.classList.remove('has-content');
                }}
            }} else {{
                decisionTreeContainer.innerHTML = '<i class="fas fa-sitemap"></i> No analysis available';
                decisionTreeContainer.classList.remove('has-content');
            }}
        }}

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Combined Viewer loading...');
            console.log('Galaxy descriptions available:', Object.keys(galaxyDescriptions).length);
            console.log('Judging results available:', Object.keys(judgingResults).length);
            
            // Store original order of galaxy cards
            originalGalaxyOrder = Array.from(document.querySelectorAll('.galaxy-card'));
            console.log(`Found ${{originalGalaxyOrder.length}} galaxy cards`);
            
            // Load default models for all dropdowns that have a selected value
            document.querySelectorAll('.model-dropdown').forEach(dropdown => {{
                if (dropdown.value && dropdown.value !== '') {{
                    const galaxyId = dropdown.dataset.galaxy;
                    const slot = dropdown.dataset.slot;
                    const modelName = dropdown.value;
                    
                    console.log(`Loading default model: ${{modelName}} for galaxy ${{galaxyId}}, slot ${{slot}}`);
                    updateModelResult(galaxyId, slot, modelName);
                }}
            }});
            
            console.log('Combined Viewer loaded successfully!');
        }});

        // Handle dropdown changes
        document.addEventListener('change', function(e) {{
            if (e.target.classList.contains('model-dropdown')) {{
                const galaxyId = e.target.dataset.galaxy;
                const slot = e.target.dataset.slot;
                const modelName = e.target.value;
                
                console.log(`Model changed: ${{modelName}} for galaxy ${{galaxyId}}, slot ${{slot}}`);
                updateModelResult(galaxyId, slot, modelName);
            }} else if (e.target.id === 'sort-select') {{
                sortGalaxies(e.target.value);
            }}
        }});

        // Include evaluation-specific comparison functions
        """ + generate_javascript_for_eval_type('galaxyzoo') + """
        """ + generate_javascript_for_eval_type('tidal') + """
        """ + generate_javascript_for_eval_type('lens') + """

        // Include eval-specific comparison functions from external file
        {comparison_js}
        
        console.log('Combined JavaScript loaded successfully!');
    """


def generate_javascript_for_eval_type(eval_type):
    """Generate JavaScript comparison functions for a specific eval type."""
    if eval_type == 'galaxyzoo':
        from .config.galaxyzoo import GalaxyZooConfig
        config = GalaxyZooConfig()
        if hasattr(config, 'js_file') and config.js_file:
            return load_javascript_from_file(config.js_file)
        return config.get_comparison_javascript()
    elif eval_type == 'tidal':
        from .config.tidal import TidalConfig
        config = TidalConfig()
        if hasattr(config, 'js_file') and config.js_file:
            return load_javascript_from_file(config.js_file)
        return config.get_comparison_javascript()
    elif eval_type == 'lens':
        from .config.lens import LensConfig
        config = LensConfig()
        if hasattr(config, 'js_file') and config.js_file:
            return load_javascript_from_file(config.js_file)
        return config.get_comparison_javascript()
    return ""

def main():
    """Generate and save the HTML file."""
    # Get available evaluation types
    available_types = list_eval_types()
    available_types_with_all = available_types + ['all']

    parser = argparse.ArgumentParser(
        description='Generate Galaxy Viewer HTML interface for different evaluation types'
    )
    parser.add_argument('--eval-type', choices=available_types_with_all,
                        help=f'Type of evaluation (auto-detected if not specified). Available: {", ".join(available_types_with_all)}')
    # No --judged flag; instead, the first positional argument is the judged file
    parser.add_argument('judged_file', 
                        help='Path to judged results JSONL file (required)')
    parser.add_argument('--output', '-o',
                        help='Output HTML file path (default: same as judged file but .html)')

    args = parser.parse_args()

    judged_file = args.judged_file

    # Auto-detect eval type if not provided
    eval_type = args.eval_type
    if eval_type is None:
        eval_type = detect_eval_type(judged_file)
        print(f"Auto-detected evaluation type: {eval_type}")

    # Get evaluation configuration
    eval_config = get_eval_config(eval_type)

    # Determine output file path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(judged_file).with_suffix('.html')

    html_content = generate_html(eval_config, judged_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✨ {eval_config.display_name} Viewer created successfully!")
    print(f"📁 Saved to: {output_file.absolute()}")
    print(f"🌐 Open in browser: file://{output_file.absolute()}")


if __name__ == "__main__":
    main()