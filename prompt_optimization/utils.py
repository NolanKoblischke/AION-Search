#!/usr/bin/env python3
"""
Utility functions for prompt optimization evaluation pipeline.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from collections import defaultdict
import statistics

from galaxybench.eval.combined.generate_descriptions import generate_and_judge_unified
from galaxybench.eval.combined.config import list_eval_types
from galaxybench.eval.combined.display import generate_html, get_eval_config, detect_eval_type
from galaxybench.eval.combined.print_score import analyze_scores
from galaxybench.eval.combined.summarize_runs import calculate_run_scores, get_run_metadata, generate_html_leaderboard
from glob import glob
import os

def run_evaluation(
    eval_types: List[str],
    prompt: Union[str, Path] = "prompt_optimization/prompts/default_prompt.txt",
    plot_script: Union[str, Path] = "prompt_optimization/plotting_scripts/default_plot.py", 
    judge_model: str = "gemini-2.5-flash-preview-05-20",
    cores: int = 10,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    plot_dir: str = "prompt_optimization/plots",
    models: Optional[List[str]] = None,
    verbose: bool = True,
    precontext_parts=None
) -> Optional[str]:
    """Run the core evaluation pipeline.
    
    Args:
        eval_types: List of evaluation types to run (e.g., ["tidal", "galaxyzoo", "lens"])
        prompt: Prompt to use for generation. Can be a string or path to a .txt file
        plot_script: Path to the plotting script that contains plot_decals function
        judge_model: Model to use for judging
        cores: Number of CPU cores to use for parallel processing
        output_dir: Directory to save results
        plot_dir: Directory to save plot images
        models: List of model IDs to run. If None, all uncommented models will be used
        verbose: Whether to print progress messages
        precontext_parts: List of precontext parts (images/text) to include before the main prompt for fewshot examples
        
    Returns:
        str: Path to the generated results file if successful, None if failed
    """
    # Convert paths to strings
    prompt = str(prompt)
    plot_script = str(plot_script)
    output_dir = str(output_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION PIPELINE")
        print("=" * 60)
        print(f"Evaluation types: {', '.join(eval_types)}")
        # print(f"Prompt: {prompt}")
        print(f"Plot script: {plot_script}")
        print(f"CPU cores: {cores}")
        print(f"Judge model: {judge_model}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
    
    try:
        final_file = generate_and_judge_unified(
            eval_types=eval_types,
            prompt=prompt,
            plot_script=plot_script,
            judge_model=judge_model,
            cores=cores,
            output_dir=output_dir,
            plot_dir=plot_dir,
            models=models,
            precontext_parts=precontext_parts
        )
        
        if verbose:
            print(f"\n✨ Evaluation completed! Results: {final_file}")
        
        return final_file
        
    except Exception as e:
        if verbose:
            print(f"\nEvaluation failed: {e}")
            import traceback
            traceback.print_exc()
        return None


def generate_html_viewer(
    results_file: str,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    verbose: bool = True
) -> Optional[str]:
    """Generate HTML viewer for evaluation results.
    
    Args:
        results_file: Path to the JSONL results file
        output_dir: Directory to save HTML file
        verbose: Whether to print progress messages
        
    Returns:
        str: Path to the generated HTML file if successful, None if failed
    """
    if verbose:
        print("\n📊 Generating HTML viewer...")
    
    try:
        # Auto-detect eval type from the generated file
        eval_type_for_display = detect_eval_type(results_file)
        if verbose:
            print(f"Auto-detected evaluation type: {eval_type_for_display}")
        
        # Get evaluation configuration
        eval_config = get_eval_config(eval_type_for_display)
        
        # Generate HTML content
        html_content = generate_html(eval_config, results_file)
        
        # Save HTML file to output directory
        filename = Path(results_file).stem
        html_dir = Path(f'{output_dir}/html')
        html_dir.mkdir(parents=True, exist_ok=True)
        html_file = html_dir / f'{filename}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if verbose:
            print(f"✨ {eval_config.display_name} Viewer created successfully!")
            print(f"📁 Saved to: {html_file.absolute()}")
            print(f"🌐 Open in browser: file://{html_file.absolute()}")
        
        return str(html_file.absolute())
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Failed to generate HTML display: {e}")
            import traceback
            traceback.print_exc()
        return None


def generate_scores_analysis(
    results_file: str,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    verbose: bool = True
) -> Optional[str]:
    """Generate detailed scores analysis.
    
    Args:
        results_file: Path to the JSONL results file
        output_dir: Directory to save scores file
        verbose: Whether to print progress messages
        
    Returns:
        str: Path to the generated scores file if successful, None if failed
    """
    if verbose:
        print("\n📊 Generating scores analysis...")
    
    try:
        scores_output = analyze_scores(results_file, debug=False)
        
        # Save scores to output directory
        filename = Path(results_file).stem
        scores_dir = Path(f'{output_dir}/scores')
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = scores_dir / f'{filename}_scores.txt'
        with open(scores_file, 'w', encoding='utf-8') as f:
            f.write(scores_output)
        
        if verbose:
            print(f"✨ Scores analysis completed!")
            print(f"📁 Saved to: {scores_file.absolute()}")
        
        return str(scores_file.absolute())
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Failed to generate scores analysis: {e}")
            import traceback
            traceback.print_exc()
        return None


def generate_runs_summary(
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    verbose: bool = True
) -> Optional[Dict[str, str]]:
    """Generate runs summary comparing all evaluation runs.
    
    Args:
        output_dir: Directory containing evaluation results
        verbose: Whether to print progress messages
        
    Returns:
        dict: Paths to generated HTML and JSONL summary files, None if failed
    """
    if verbose:
        print("\n📊 Generating runs summary...")
    
    try:
        # Find all JSONL files in the output directory
        jsonl_pattern = f"{output_dir}/jsonl/*.jsonl"
        jsonl_files = glob(jsonl_pattern)
        
        if not jsonl_files:
            if verbose:
                print(f"No JSONL files found in {output_dir}/jsonl/")
            return None
        
        if verbose:
            print(f"Found {len(jsonl_files)} evaluation runs to summarize...")
        
        runs_data = []
        for jsonl_file in sorted(jsonl_files):
            filename = Path(jsonl_file).name
            
            # Extract timestamp from filename
            timestamp = filename.replace('judged_all_evals_', '').replace('.jsonl', '')
            
            # Get metadata and scores
            metadata = get_run_metadata(jsonl_file)
            scores = calculate_run_scores(jsonl_file)
            
            # Combine data
            run_data = {
                'timestamp': timestamp,
                'filename': filename,
                **metadata,
                **scores
            }
            runs_data.append(run_data)
        
        # Sort by timestamp (newest first)
        runs_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Generate HTML leaderboard
        html_content = generate_html_leaderboard(runs_data)
        
        # Save HTML summary file
        html_summary_file = Path(f"{output_dir}/runs_summary.html")
        with open(html_summary_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save JSONL summary file
        jsonl_summary_file = Path(f"{output_dir}/runs_summary.jsonl")
        with open(jsonl_summary_file, 'w', encoding='utf-8') as f:
            for run_data in runs_data:
                # Create a clean record with all important information
                summary_record = {
                    'timestamp': run_data['timestamp'],
                    'filename': run_data['filename'],
                    'prompt_filename': run_data['prompt_filename'],
                    'plot_script': run_data['plot_script'],
                    'model_name': run_data['model_name'],
                    'model_count': run_data['model_count'],
                    
                    # Galaxy Zoo scores
                    'gz_accuracy': round(run_data['gz_accuracy'], 3) if isinstance(run_data['gz_accuracy'], float) else run_data['gz_accuracy'],
                    
                    # Tidal scores
                    'tidal_f1_shell': round(run_data['tidal_f1_shell'], 3) if isinstance(run_data['tidal_f1_shell'], float) else run_data['tidal_f1_shell'],
                    'tidal_f1_stream': round(run_data['tidal_f1_stream'], 3) if isinstance(run_data['tidal_f1_stream'], float) else run_data['tidal_f1_stream'],
                    'tidal_f1_average': round(run_data['tidal_f1_average'], 3) if isinstance(run_data['tidal_f1_average'], float) else run_data['tidal_f1_average'],
                    'shell_fp': run_data['shell_fp'],
                    'shell_total': run_data['shell_total'],
                    'shell_tp': run_data['shell_tp'],
                    'stream_fp': run_data['stream_fp'],
                    'stream_total': run_data['stream_total'],
                    'stream_tp': run_data['stream_tp'],
                    
                    # Lens scores
                    'lens_f1': round(run_data['lens_f1'], 3) if isinstance(run_data['lens_f1'], float) else run_data['lens_f1'],
                    'lens_fp': run_data['lens_fp'],
                    'lens_total': run_data['lens_total'],
                    'lens_tp': run_data['lens_tp'],
                    
                    # Aggregate scores
                    'agg_score': round(run_data['agg_score'], 3) if isinstance(run_data['agg_score'], float) else run_data['agg_score'],
                    'agg_score_2': round(run_data['agg_score_2'], 3) if isinstance(run_data['agg_score_2'], float) else run_data['agg_score_2']
                }
                f.write(json.dumps(summary_record) + '\n')
        
        if verbose:
            print(f"✨ Runs summary generated successfully!")
            print(f"📁 HTML saved to: {html_summary_file.absolute()}")
            print(f"🌐 Open in browser: file://{html_summary_file.absolute()}")
            print(f"📁 JSONL saved to: {jsonl_summary_file.absolute()}")
        
        return {
            'html': str(html_summary_file.absolute()),
            'jsonl': str(jsonl_summary_file.absolute())
        }
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Failed to generate runs summary: {e}")
            import traceback
            traceback.print_exc()
        return None


def get_all_scores_from_results(results_file: str) -> Optional[Dict[str, Any]]:
    """Extract all scores from a completed evaluation run.
    
    Args:
        results_file: Path to the JSONL results file
        
    Returns:
        dict: Dictionary containing all scores, None if failed
    """
    try:
        scores = calculate_run_scores(results_file)
        return {
            # Galaxy Zoo scores
            'gz_accuracy': scores.get('gz_accuracy', 0.0),
            
            # Tidal scores
            'tidal_f1_shell': scores.get('tidal_f1_shell', 0.0),
            'tidal_f1_stream': scores.get('tidal_f1_stream', 0.0),
            'tidal_f1_average': scores.get('tidal_f1_average', 0.0),
            'shell_fp': scores.get('shell_fp', 0),
            'shell_total': scores.get('shell_total', 0),
            'shell_tp': scores.get('shell_tp', 0),
            'stream_fp': scores.get('stream_fp', 0),
            'stream_total': scores.get('stream_total', 0),
            'stream_tp': scores.get('stream_tp', 0),
            
            # Lens scores
            'lens_f1': scores.get('lens_f1', 0.0),
            'lens_fp': scores.get('lens_fp', 0),
            'lens_total': scores.get('lens_total', 0),
            'lens_tp': scores.get('lens_tp', 0),
            
            # Aggregate scores
            'agg_score': scores.get('agg_score', 0.0),
            'agg_score_2': scores.get('agg_score_2', 0.0),
            
            # Model metadata
            'model_count': scores.get('model_count', 0)
        }
    except Exception as e:
        print(f"Failed to extract scores: {e}")
        return None


def get_agg_score_2_from_results(results_file: str) -> Optional[float]:
    """Extract the agg_score_2 from a completed evaluation run.
    
    Args:
        results_file: Path to the JSONL results file
        
    Returns:
        float: The agg_score_2 value, None if failed
    """
    scores = get_all_scores_from_results(results_file)
    return scores.get('agg_score_2') if scores else None


def analyze_worst_galaxyzoo_questions(results_file: str, top_n: int = 5, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:
    """Analyze the worst-performing Galaxy Zoo questions from a specific run.
    
    Args:
        results_file: Path to the JSONL results file
        top_n: Number of worst questions to return
        verbose: Whether to print the analysis
        
    Returns:
        list: List of dictionaries containing question analysis, None if failed
    """
    try:
        # Load records from the file
        records = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Only process Galaxy Zoo records
                    if 'decision_tree' in record and 'decision_tree_score' in record:
                        records.append(record)
        
        if not records:
            if verbose:
                print("No Galaxy Zoo records found in the file.")
            return None
        
        # Extract question scores
        question_scores = defaultdict(list)
        
        for record in records:
            if record.get('decision_tree_score') is None:
                continue
                
            # Extract volunteer decision tree path
            volunteer_nodes = set()
            for node_info in record['decision_tree']:
                if 'node' in node_info:
                    volunteer_nodes.add(node_info['node'])
            
            # Extract judge decision tree path
            judge_nodes = set()
            if 'judge_results' in record and 'judge_path' in record['judge_results']:
                judge_nodes = set(record['judge_results']['judge_path'])
            
            # Calculate per-question accuracy
            for node_info in record['decision_tree']:
                if 'node' in node_info:
                    node = node_info['node']
                    question = node_info.get('question', node.split('_')[0])
                    
                    # Check if judge got this specific node correct
                    node_correct = 1.0 if node in judge_nodes else 0.0
                    question_scores[question].append(node_correct)
        
        # Calculate average scores for each question
        question_stats = []
        for question, scores in question_scores.items():
            if scores:
                avg_score = statistics.mean(scores)
                question_stats.append({
                    'question': question,
                    'average_score': avg_score,
                    'total_instances': len(scores),
                    'correct_instances': sum(scores)
                })
        
        # Sort by average score (ascending - worst first)
        question_stats.sort(key=lambda x: x['average_score'])
        
        # Get the worst N questions
        worst_questions = question_stats[:top_n]
        
        if verbose:
            print(f"\n📊 WORST-PERFORMING GALAXY ZOO QUESTIONS (Top {top_n}):")
            print("=" * 60)
            for i, q_stat in enumerate(worst_questions, 1):
                correct = int(q_stat['correct_instances'])
                total = q_stat['total_instances']
                avg_score = q_stat['average_score']
                print(f"{i}. {q_stat['question']}")
                print(f"   Score: {avg_score:.3f} ({correct}/{total} correct)")
                print()
        
        return worst_questions
        
    except Exception as e:
        if verbose:
            print(f"Failed to analyze worst questions: {e}")
        return None


def run_full_evaluation_pipeline(
    eval_types: List[str] = ["tidal"],
    prompt: Union[str, Path] = "prompt_optimization/prompts/default_prompt.txt",
    plot_script: Union[str, Path] = "prompt_optimization/plotting_scripts/default_plot.py", 
    judge_model: str = "gemini-2.5-flash-preview-05-20",
    cores: int = 10,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    plot_dir: str = "prompt_optimization/plots",
    models: Optional[List[str]] = None,
    verbose: bool = True,
    precontext_parts=None
) -> Optional[Dict[str, Any]]:
    """Run the complete evaluation pipeline with all post-processing steps.
    
    Args:
        eval_types: List of evaluation types to run
        prompt: Prompt to use for generation
        plot_script: Path to the plotting script
        judge_model: Model to use for judging
        cores: Number of CPU cores to use
        output_dir: Directory to save results
        plot_dir: Directory to save plots
        models: List of model IDs to run. If None, all uncommented models will be used
        verbose: Whether to print progress messages
        precontext_parts: List of precontext parts (images/text) to include before the main prompt for fewshot examples
        
    Returns:
        dict: Dictionary containing paths to all generated files and scores
    """
    os.makedirs("prompt_optimization/runs/jsonl", exist_ok=True)
    os.makedirs("prompt_optimization/runs/scores", exist_ok=True)
    os.makedirs("prompt_optimization/runs/html", exist_ok=True)
    # Step 1: Run evaluation
    results_file = run_evaluation(
        eval_types=eval_types,
        prompt=prompt,
        plot_script=plot_script,
        judge_model=judge_model,
        cores=cores,
        output_dir=output_dir,
        plot_dir=plot_dir,
        models=models,
        verbose=verbose,
        precontext_parts=precontext_parts
    )
    
    if not results_file:
        return None
    
    # Step 2: Generate HTML viewer
    html_file = generate_html_viewer(
        results_file=results_file,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Step 3: Generate scores analysis
    scores_file = generate_scores_analysis(
        results_file=results_file,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Step 4: Generate runs summary
    summary_files = generate_runs_summary(
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Step 5: Extract all scores
    all_scores = get_all_scores_from_results(results_file)
    
    # Step 6: Analyze worst Galaxy Zoo questions if galaxyzoo is in eval_types
    worst_questions = None
    if "galaxyzoo" in eval_types:
        worst_questions = analyze_worst_galaxyzoo_questions(
            results_file=results_file,
            top_n=5,
            verbose=verbose
        )
    
    # Create comprehensive results dictionary
    results = {
        'results_file': results_file,
        'html_file': html_file,
        'scores_file': scores_file,
        'summary_files': summary_files,
        'worst_questions': worst_questions,
    }
    
    # Add all scores to the results if extraction was successful
    if all_scores:
        results.update(all_scores)
    
    return results