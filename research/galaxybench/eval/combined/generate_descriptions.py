"""Unified description generation and judging script for all evaluation types."""

import json
import time
import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import fcntl
import argparse
from tqdm import tqdm
import galaxybench.eval.combined

from galaxybench.eval.utils_gemini import generate as generate_gemini
from galaxybench.eval.utils_openai import generate as generate_openai

from galaxybench.eval.combined.config import get_eval_config, list_eval_types


def load_models():
    """Load model configurations from models.jsonl."""
    with open('galaxybench/eval/models.jsonl') as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith('//')]


def filter_models_by_ids(all_models, model_ids):
    """Filter models by their id field.
    
    Args:
        all_models: List of all model configurations
        model_ids: List of model ids to include (if None, returns all models)
        
    Returns:
        List of filtered model configurations
    """
    if model_ids is None:
        return all_models
    
    filtered_models = []
    for model_config in all_models:
        if model_config['id'] in model_ids:
            filtered_models.append(model_config)
    
    # Check if any requested models were not found
    found_model_ids = {model['id'] for model in filtered_models}
    missing_models = set(model_ids) - found_model_ids
    if missing_models:
        print(f"Warning: The following model IDs were not found in models.jsonl: {', '.join(missing_models)}")
    
    return filtered_models


def load_plotting_module(plot_script_path):
    """Dynamically load plotting module from file path."""
    # Convert to absolute path to ensure it works in multiprocessing
    plot_script_path = os.path.abspath(plot_script_path)
    
    # Check if file exists
    if not os.path.exists(plot_script_path):
        raise FileNotFoundError(f"Plotting script not found: {plot_script_path}")
    
    spec = importlib.util.spec_from_file_location("plot_module", plot_script_path)
    
    # Check if spec creation was successful
    if spec is None:
        raise ImportError(f"Failed to create module spec for: {plot_script_path}")
    
    # Check if loader is available
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for: {plot_script_path}")
    
    plot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_module)
    
    # Verify the module has the required plot_decals function
    if not hasattr(plot_module, 'plot_decals'):
        raise ValueError(f"Plotting script {plot_script_path} must contain a 'plot_decals' function")
    
    return plot_module


def run_model_on_image(config, prompt, image_path, galaxy_data, plot_script_name, prompt_filename, precontext_parts=None):
    """Run a single model on an image and return results."""
    model_name = config['model_name']
    
    # Retry logic for API failures
    max_retries = 3
    for retry in range(max_retries):
        try:
            start = time.time()
            if model_name.startswith('gemini'):
                result = generate_gemini(prompt, image_path, model_name, 
                                       config.get('thinking_budget', 0), precontext_parts)
            else:
                result = generate_openai(prompt, image_path, model_name, 
                                       config.get('reasoning_effort'))
            elapsed = time.time() - start
            
            # Check if the result text is None or empty - this should trigger a retry
            if result[0] is None or (isinstance(result[0], str) and result[0].strip() == ""):
                raise ValueError(f"Model returned null or empty response")
            
            # Check if token counts are valid before calculating cost
            input_tokens = result[1] if result[1] is not None else 0
            output_tokens = result[2] if result[2] is not None else 0
            
            cost = (input_tokens * config.get('input_price', 0) + 
                   output_tokens * config.get('output_price', 0)) / 1000000
            
            # If we get here successfully, break out of retry loop
            break
            
        except Exception as e:
            if retry < max_retries - 1:
                # Check if this is a quota/resource exhausted error
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"  Retry {retry + 1}/{max_retries} for {model_name}: Quota exceeded - waiting 60 seconds...")
                    print(f"  Error details: {e}")
                    time.sleep(60)  # Wait 60 seconds for quota errors
                else:
                    print(f"  Retry {retry + 1}/{max_retries} for {model_name}: {e}")
                    time.sleep(2 ** retry)  # Exponential backoff for other errors
                continue
            else:
                # Final retry failed, re-raise the exception
                raise e
    
    # Get object_id and galaxy info from galaxy_data
    object_id = list(galaxy_data.keys())[0]  # There should be exactly one object in galaxy_data
    galaxy_info = galaxy_data[object_id]
    
    record = {
        'prompt': prompt,
        'prompt_filename': prompt_filename,
        'plot_script': plot_script_name,
        'formatted_name': config['formatted_name'],
        'model_name': model_name,
        'cost': cost,
        'time': elapsed,
        'thinking_budget': config.get('thinking_budget'),
        'reasoning_effort': config.get('reasoning_effort'),
        'image_path': image_path,
        'object_id': object_id,
        'index': galaxy_info.get('index'),
        'ra': galaxy_info.get('ra'),
        'dec': galaxy_info.get('dec'),
        'result': result[0],
        'input_tokens': input_tokens,
        'output_tokens': output_tokens
    }
    
    # Add any additional fields from galaxy_info that aren't already in record
    for key, value in galaxy_info.items():
        if key not in record and key != 'image_array':
            record[key] = value
    
    return record


def judge_record(record, judge):
    """Judge a single generated description record with retry logic for quota errors."""
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            return judge.process_record(record)
        except Exception as e:
            error_str = str(e)
            
            # Check if this is the final retry
            if retry < max_retries - 1:
                # Check if this is a quota/resource exhausted error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"  Judge retry {retry + 1}/{max_retries}: Quota exceeded - waiting 60 seconds...")
                    print(f"  Judge error details: {e}")
                    time.sleep(60)  # Wait 60 seconds for quota errors
                else:
                    print(f"  Judge retry {retry + 1}/{max_retries}: {e}")
                    time.sleep(2 ** retry)  # Exponential backoff for other errors
                continue
            else:
                # Final retry failed, return the original record with error info
                error_record = record.copy()
                error_record['judge_error'] = str(e)
                return error_record


def write_record_safely(record, output_file):
    """Safely write a single record to the output file with locking."""
    with open(output_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        try:
            f.write(json.dumps(record) + '\n')
            f.flush()  # Force write to disk
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock


def process_galaxy_with_judging(args):
    """Process a single galaxy with all models and judge the results."""
    galaxy_info, models, prompt, plot_script_path, plot_script_name, prompt_filename, eval_config, judge_model, output_file, plot_dir, precontext_parts = args
    object_id = galaxy_info['object_id']
    results = []
    
    try:
        # Load plotting module within the worker process
        plot_module = load_plotting_module(plot_script_path)
        
        # Generate image using the plotting script
        image_path = plot_module.plot_decals(galaxy_info['image_array'], object_id, output_dir=plot_dir, script_name=plot_script_name)
        
        # Create judge instance for this worker
        judge_class = eval_config.get_judge_class()
        judge = judge_class(model=judge_model)
        
        for model_config in models:
            try:
                # Step 1: Generate description
                record = run_model_on_image(model_config, prompt, image_path, {object_id: galaxy_info}, plot_script_name, prompt_filename, precontext_parts)
                
                # Step 2: Judge the description immediately
                judged_record = judge_record(record, judge)
                results.append(judged_record)
                
                # Write immediately after each model completes
                write_record_safely(judged_record, output_file)
                
                score_field = eval_config.get_score_field_name()
                score = judged_record.get(score_field, 'error')
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                
                print(f"  {object_id} - {model_config['formatted_name']}: ${judged_record['cost']:.6f} ({judged_record['time']:.1f}s) - Score: {score_str}")
            except Exception as e:
                print(f"  {object_id} - Error with {model_config['formatted_name']}: {e}")
    except Exception as e:
        print(f"  {object_id} - Error generating image: {e}")
    
    return len(results)


def process_galaxy_with_judging_safe(args):
    """Wrapper that ensures we always return a result, even on failure."""
    try:
        result_count = process_galaxy_with_judging(args)
        return result_count, None
    except Exception as e:
        import traceback
        galaxy_info = args[0]
        error_info = {
            'object_id': galaxy_info['object_id'],
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return 0, error_info


def load_prompt(prompt_arg):
    """Load prompt from string or file."""
    if prompt_arg.endswith('.txt') and os.path.exists(prompt_arg):
        with open(prompt_arg, 'r') as f:
            return f.read().strip()
    else:
        return prompt_arg


def generate_and_judge_unified(eval_types: list, prompt: str, plot_script: str, 
                              judge_model: str = "gpt-4o-mini", cores: int = 10, output_dir: str = "eval/runs", plot_dir: str = "plots", models: list = None, precontext_parts=None):
    """Generate galaxy descriptions and judge them for all specified evaluation types.
    
    This unified function replaces the previous multi-step process and directly produces
    the final judged_all_evals_timestamp.jsonl file.
    
    Args:
        eval_types: List of evaluation types to process
        prompt: Prompt to use for generation (string or path to .txt file)
        plot_script: Path to the plotting script containing plot_decals function
        judge_model: Model to use for judging (default: gpt-4o-mini). Examples: gpt-4o-mini, o4-mini, gemini-2.5-flash-preview-05-20
        cores: Number of CPU cores to use for parallel processing
        output_dir: Directory to save results (default: eval/runs)
        plot_dir: Directory to save plot images (default: plots)
        models: List of model IDs to run. If None, all uncommented models from models.jsonl will be used
        precontext_parts: List of precontext parts (images/text) to include before the main prompt for fewshot examples
        
    Returns:
        str: Path to the generated unified judged file
    """
    prompt_text = load_prompt(prompt)
    plot_script_name = Path(plot_script).stem
    plot_script_path = plot_script
    all_models = load_models()
    filtered_models = filter_models_by_ids(all_models, models)
    
    # Validate that we have models to run
    if not filtered_models:
        if models is None:
            raise ValueError("No uncommented models found in models.jsonl")
        else:
            raise ValueError(f"No matching models found for: {', '.join(models)}")
    
    # Determine prompt filename
    if prompt.endswith('.txt') and os.path.exists(prompt):
        prompt_filename = Path(prompt).name
    else:
        print(f"--------------------------------")
        print('\n\n')
        print(f"WARNING: Prompt file: {prompt[:100]} not found! Assuming it is an inline prompt...")
        print('\n\n')
        print(f"--------------------------------")
        prompt_filename = "inline_prompt"
    
    # Add current datetime to filename
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_file = f'{output_dir}/jsonl/judged_all_evals_{timestamp}.jsonl'
    
    # Create empty output file
    open(output_file, 'w').close()
    
    total_galaxies = 0
    total_records = 0
    
    print(f"=== UNIFIED EVALUATION PIPELINE ===")
    print(f"Evaluation types: {', '.join(eval_types)}")
    print(f"Judge model: {judge_model}")
    print(f"Found {len(filtered_models)} models")
    print(f"Using plotting script: {plot_script_name}")
    print(f"Using prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    for eval_type in eval_types:
        print(f"\n--- Processing {eval_type} ---")
        
        # Get evaluation configuration
        eval_config = get_eval_config(eval_type)
        
        # Load galaxy data using the appropriate loader
        loader_class = eval_config.get_loader_class()
        loader = loader_class(eval_config.hdf5_path, eval_config.hdf5_table_name, eval_config)
        galaxy_data = loader.load_galaxy_data()
        
        print(f"Found {len(galaxy_data)} galaxies for {eval_config.display_name}")
        total_galaxies += len(galaxy_data)
        
        # Prepare arguments for multiprocessing
        galaxy_list = list(galaxy_data.values())
        args_list = [(galaxy_info, filtered_models, prompt_text, plot_script_path, plot_script_name, 
                     prompt_filename, eval_config, judge_model, output_file, plot_dir, precontext_parts) 
                     for galaxy_info in galaxy_list]
        
        print(f"Processing {len(galaxy_list)} galaxies with {cores} cores...")
        
        with Pool(cores) as pool:
            # Use imap_unordered for better fault tolerance
            results_iter = pool.imap_unordered(process_galaxy_with_judging_safe, args_list)
            
            success_count = 0
            failed_galaxies = []
            
            for result_count, error_info in tqdm(results_iter, total=len(args_list), 
                                                desc=f"Processing {eval_config.display_name}"):
                if error_info:
                    failed_galaxies.append(error_info)
                    print(f"\n Failed: {error_info['object_id']} - {error_info['error']}")
                else:
                    success_count += result_count
        
        # Retry failed galaxies with reduced parallelism
        if failed_galaxies:
            print(f"\n  Retrying {len(failed_galaxies)} failed galaxies with single processing...")
            
            for error_info in tqdm(failed_galaxies, desc="Retrying failed galaxies"):
                object_id = error_info['object_id']
                # Find the original args for this galaxy
                retry_args = next((args for args in args_list 
                                 if args[0]['object_id'] == object_id), None)
                
                if retry_args:
                    try:
                        # Process synchronously for better error handling
                        retry_result = process_galaxy_with_judging(retry_args)
                        success_count += retry_result
                        print(f" Successfully processed {object_id} on retry")
                    except Exception as e:
                        print(f" Failed to process {object_id} after retry: {e}")
                        # Log detailed error information
                        import traceback
                        print(f"Traceback:\n{traceback.format_exc()}")
                else:
                    print(f" Could not find args for {object_id} to retry")
        
        eval_records = success_count
        total_records += eval_records
        print(f"Completed {eval_type}: {eval_records} records")
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total galaxies processed: {total_galaxies}")
    print(f"Total records generated: {total_records}")
    print(f"Unified results saved to: {output_file}")
    
    # Check if we have the expected number of records
    expected_records = total_galaxies * len(filtered_models)
    if total_records < expected_records:
        missing_records = expected_records - total_records
        print(f"\n WARNING: {missing_records} records missing!")
        print(f"Expected: {expected_records} records ({total_galaxies} galaxies × {len(filtered_models)} models)")
        print(f"Actual: {total_records} records")
    
    # Print evaluation type breakdown
    eval_type_counts = {}
    score_stats = {}
    
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                
                # Detect eval type from record
                eval_type = "unknown"
                if 'decision_tree' in record:
                    eval_type = "galaxyzoo"
                elif 'tidal_info' in record:
                    eval_type = "tidal"
                
                eval_type_counts[eval_type] = eval_type_counts.get(eval_type, 0) + 1
                
                # Collect score statistics
                if eval_type not in score_stats:
                    score_stats[eval_type] = []
                
                # Get the appropriate score field for this eval type
                try:
                    eval_config = get_eval_config(eval_type)
                    score_field = eval_config.get_score_field_name()
                    score = record.get(score_field)
                    if score is not None and isinstance(score, (int, float)):
                        score_stats[eval_type].append(score)
                except:
                    pass
    
    print(f"\nRecords by evaluation type:")
    for eval_type, count in eval_type_counts.items():
        print(f"  {eval_type}: {count}")
        if eval_type in score_stats and score_stats[eval_type]:
            scores = score_stats[eval_type]
            avg_score = sum(scores) / len(scores)
            print(f"    Average score: {avg_score:.3f}")
    
    print(f"\n Unified evaluation complete! Results: {output_file}")
    
    return output_file
