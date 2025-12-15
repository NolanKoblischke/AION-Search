"""Unified judge script for all evaluation types."""

import json
import os
import sys
import fcntl
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from typing import Dict, Any

import eval.combined

from eval.combined.config import get_eval_config, list_eval_types


def detect_eval_type(input_file: str) -> str:
    """Detect evaluation type from input file by checking for specific fields."""
    # Read first record to check fields
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            record = json.loads(first_line)
            
            # Check for evaluation-specific fields
            if 'decision_tree' in record:
                return 'galaxyzoo'
            elif 'tidal_info' in record:
                return 'tidal'
            
            # Could add more detection logic for future eval types
            
    raise ValueError("Could not detect evaluation type from input file")


def write_record_safely(record: Dict[str, Any], output_file: str):
    """Safely write a single record to the output file with locking."""
    with open(output_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        try:
            f.write(json.dumps(record) + '\n')
            f.flush()  # Force write to disk
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock


def judge_single_record(args):
    """Judge a single record - for multiprocessing."""
    record, judge = args
    return judge.process_record(record)


def judge_descriptions(input_file: str, output_file: str = None, eval_type: str = None, 
                      judge_model: str = None, cores: int = 10):
    """Judge galaxy descriptions in a JSONL file.
    
    Args:
        input_file: Path to input JSONL file with galaxy descriptions
        output_file: Path to output file (default: judged_{input_filename})
        eval_type: Type of evaluation (auto-detected if not specified)
        judge_model: OpenAI or Gemini model to use for judging
        cores: Number of CPU cores to use
        
    Returns:
        tuple: (output_file_path, eval_config)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if eval_type is None:
        eval_type = detect_eval_type(input_file)
        print(f"Auto-detected evaluation type: {eval_type}")
    
    # Get evaluation configuration
    eval_config = get_eval_config(eval_type)
    
    # Create output filename with judged_ prefix if not provided
    if output_file is None:
        output_file = input_path.parent / f"judged_{input_path.name}"
    
    print(f"Processing: {input_file}")
    print(f"Evaluation type: {eval_config.display_name}")
    print(f"Output will be saved to: {output_file}")
    print(f"Judge model: {judge_model}")
    
    # Create empty output file
    open(output_file, 'w').close()
    
    # Load all records
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    print(f"Loaded {len(records)} records")
    
    # Create judge instance
    judge_class = eval_config.get_judge_class()
    judge = judge_class(model=judge_model)
    
    # Prepare arguments for multiprocessing
    args_list = [(record, judge) for record in records]
    
    # Process with multiprocessing
    print(f"Processing with {cores} cores...")
    
    processed_count = 0
    error_count = 0
    
    with Pool(cores) as pool:
        for result in tqdm(pool.imap_unordered(judge_single_record, args_list), 
                          total=len(args_list), desc=f"Judging {eval_config.display_name} descriptions"):
            # Write each result immediately with safe locking
            write_record_safely(result, output_file)
            
            score_field = eval_config.get_score_field_name()
            if result.get(score_field) is not None:
                processed_count += 1
            else:
                error_count += 1
                if 'judge_error' in result:
                    print(f"Error for {result.get('object_id', 'unknown')}: {result['judge_error']}")
    
    print(f"\nCompleted processing!")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Results saved to: {output_file}")
    
    return output_file, eval_config


def print_statistics(output_file: str, eval_config):
    """Print summary statistics for the judged results."""
    print("\n=== SUMMARY STATISTICS ===")
    scores = []
    model_scores = {}
    score_field = eval_config.get_score_field_name()
    
    with open(output_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record.get(score_field) is not None:
                scores.append(record[score_field])
                model_name = record.get('formatted_name', 'unknown')
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(record[score_field])
    
    if scores:
        print(f"Total scored descriptions: {len(scores)}")
        print(f"Average {score_field}: {sum(scores)/len(scores):.3f}")
        print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        # Show score distribution
        score_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(score_bins)-1):
            if i == len(score_bins)-2:  # Last bin should be inclusive
                count = sum(1 for s in scores if score_bins[i] <= s <= score_bins[i+1])
            else:
                count = sum(1 for s in scores if score_bins[i] <= s < score_bins[i+1])
            print(f"  Score {score_bins[i]:.1f}-{score_bins[i+1]:.1f}: {count} ({count/len(scores)*100:.1f}%)")
        
        print("\nBy model:")
        for model, model_score_list in sorted(model_scores.items()):
            avg_score = sum(model_score_list) / len(model_score_list)
            print(f"  {model}: {avg_score:.3f} (n={len(model_score_list)})")