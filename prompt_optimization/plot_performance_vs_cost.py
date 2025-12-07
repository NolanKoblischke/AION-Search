#!/usr/bin/env python3
"""
Plot decision tree scores (performance) vs cost for different models.
Creates a scatter plot with model names as labels.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_model_prices():
    """Load model pricing information from models.jsonl."""
    models_file = Path("galaxybench/eval/models.jsonl")
    model_prices = {}
    
    with open(models_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Store by both model_name and id for flexibility
            model_id = data.get('id')
            model_name = data.get('model_name')
            price_info = {
                'input_price': data['input_price'] / 1e6,  # Convert to per-token cost
                'output_price': data['output_price'] / 1e6  # Convert to per-token cost
            }
            if model_id:
                model_prices[model_id] = price_info
            if model_name:
                model_prices[model_name] = price_info
    
    return model_prices

def analyze_runs_for_plot(jsonl_dir):
    """Analyze all JSONL files and prepare data for plotting."""
    jsonl_dir = Path(jsonl_dir)
    
    # Load model prices
    model_prices = load_model_prices()
    
    # Dictionary to store scores and costs for each unique combination
    # Key: (formatted_name, prompt_filename, plot_script)
    # Value: dict with question_id -> list of scores, and lists of costs
    combination_data = defaultdict(lambda: {'question_scores': defaultdict(list), 'costs': []})
    
    # Process all JSONL files
    for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract relevant fields
                    formatted_name = data.get("formatted_name", "unknown")
                    prompt_filename = data.get("prompt_filename", "unknown")
                    plot_script = data.get("plot_script", "unknown")
                    decision_tree_score = data.get("decision_tree_score")
                    input_tokens = data.get("input_tokens")
                    output_tokens = data.get("output_tokens")
                    model_name = data.get("model_name")
                    reasoning_effort = data.get("reasoning_effort")
                    question_id = data.get("object_id", data.get("question_id", data.get("id", "unknown")))  # Get question ID from object_id field
                    
                    # Build a lookup key based on model_name and reasoning_effort
                    lookup_key = None
                    if model_name in model_prices:
                        lookup_key = model_name
                    elif reasoning_effort and f"{model_name}-{reasoning_effort}-thinking" in model_prices:
                        lookup_key = f"{model_name}-{reasoning_effort}-thinking"
                    
                    # Calculate cost on the fly if we have token counts and model pricing
                    if decision_tree_score is not None and input_tokens is not None and output_tokens is not None and lookup_key:
                        prices = model_prices[lookup_key]
                        cost = (input_tokens * prices['input_price']) + (output_tokens * prices['output_price'])
                        
                        key = (formatted_name, prompt_filename, plot_script)
                        combination_data[key]['question_scores'][question_id].append(decision_tree_score)
                        combination_data[key]['costs'].append(cost)
                
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    
    # Calculate statistics for each combination
    plot_data = []
    for (model, prompt, plot_script), data in combination_data.items():
        if data['question_scores'] and data['costs']:
            # Step 1: Calculate per-question average
            question_averages = []
            for question_id, scores in data['question_scores'].items():
                question_averages.append(np.mean(scores))
            
            question_averages = np.array(question_averages)
            n_questions = len(question_averages)
            
            # Step 2: Calculate overall mean accuracy
            mean_score = np.mean(question_averages)
            
            # Step 3: Calculate sample variance across questions and SEM
            if n_questions > 1:
                # Sample variance across questions
                sample_variance = np.var(question_averages, ddof=1)
                # Standard error of the mean via CLT
                sem = np.sqrt(sample_variance / n_questions)
            else:
                sem = 0.0  # No variance if only one question
            
            costs_array = np.array(data['costs'])
            
            # Count perfect scores (1.0) across all attempts
            perfect_scores = 0
            total_attempts = 0
            for question_id, scores in data['question_scores'].items():
                for score in scores:
                    if score == 1.0:
                        perfect_scores += 1
                    total_attempts += 1
            
            plot_data.append({
                'model': model,
                'prompt': prompt,
                'plot_script': plot_script,
                'mean_score': mean_score,
                'sem': sem,
                'mean_cost': np.mean(costs_array),
                'n': len(costs_array),
                'n_questions': n_questions,
                'perfect_scores': perfect_scores,
                'total_attempts': total_attempts
            })
    
    # Sort by mean cost for consistent ordering
    plot_data.sort(key=lambda x: x['mean_cost'])
    
    return plot_data

def create_shortened_name(model_name, max_length=50):
    """Return model name as-is for display."""
    return model_name

def find_pareto_frontier(plot_data):
    """Find Pareto optimal points (best performance for given cost or lowest cost for given performance)."""
    # Sort by cost first
    sorted_data = sorted(plot_data, key=lambda x: x['mean_cost'])
    
    pareto_points = []
    max_score_so_far = 0
    
    for point in sorted_data:
        # A point is Pareto optimal if its score is better than any point with lower cost
        if point['mean_score'] >= max_score_so_far:
            pareto_points.append(point)
            max_score_so_far = point['mean_score']
    
    return pareto_points

def create_plot(plot_data, output_path):
    """Create scatter plot of performance vs cost."""
    
    if not plot_data:
        print("No data to plot!")
        return
    
    # Filter out specific models and any with "No Thinking"
    excluded_models = [
        "Gemini 2.5 Pro Preview 05-06 No Thinking",
        "Gemini 2.5 Pro Preview Thinking",
        "Gemini 2.5 Pro Preview 05-06 Thinking",
        "o3 Reasoning Low",
        "o4-mini Reasoning Low",
        "GPT-4o",
        "Gemini 1.5 Flash"
    ]
    plot_data = [d for d in plot_data if d['model'] not in excluded_models and "No Thinking" not in d['model']]
    
    # Extract data for plotting
    models = [d['model'].replace(' Reasoning', '').replace(' Medium', '').replace(' Max Thinking', '').replace(' Thinking', '') for d in plot_data]  # Remove "Reasoning", "Medium", "Max Thinking", and "Thinking" from labels
    mean_scores = [d['mean_score'] for d in plot_data]
    sems = [d['sem'] for d in plot_data]  # Get SEM values
    mean_costs = [d['mean_cost'] * 100000 * 0.5 for d in plot_data]  # Convert to cost per 100,000 images and divide by 2
    n_samples = [d['n'] for d in plot_data]
    n_questions = [d['n_questions'] for d in plot_data]
    
    # Find Pareto frontier points
    pareto_points = find_pareto_frontier(plot_data)
    pareto_costs = [p['mean_cost'] * 100000 * 0.5 for p in pareto_points]  # Also divide by 2
    pareto_scores = [p['mean_score'] for p in pareto_points]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9.8, 7))
    
    # Define colors for each model
    base_colors = ['#4285F4', '#34A853', '#EA4335', '#FBBC05', '#9333EA', '#EC4899', '#10B981', '#F59E0B']
    colors = []
    
    # Assign colors based on model names
    for i, d in enumerate(plot_data):
        model_name = d['model']
        if model_name.startswith('GPT-4.1'):
            colors.append('#87CEEB')  # Light blue
        elif model_name.startswith('GPT-5'):
            colors.append('#DDA0DD')  # Light purple (plum)
        else:
            # Use default colors for other models
            colors.append(base_colors[i % len(base_colors)])
    
    # Add error bars for SEM (in background with low z-order)
    ax.errorbar(mean_costs, mean_scores, yerr=sems,
                fmt='none',  # No markers (we already have scatter points)
                ecolor='gray',
                elinewidth=1.2,
                capsize=3,
                alpha=0.4,
                zorder=0)  # Put in far background
    
    # Create scatter plot on top
    scatter = ax.scatter(mean_costs, mean_scores, 
                        s=300,  # Large marker size
                        c=colors, 
                        alpha=1.0,  # Fully opaque
                        edgecolor='black', 
                        linewidth=2,
                        zorder=2)  # Put scatter points above error bars
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        ax.plot(pareto_costs, pareto_scores, 
                color='gray', 
                linestyle='--', 
                alpha=0.4, 
                linewidth=2, 
                zorder=1,  # Behind the scatter points
                label='Pareto Frontier')
    
    # Add text labels for each point
    for i, (cost, score, model, d) in enumerate(zip(mean_costs, mean_scores, models, plot_data)):
        # Position text slightly offset from point
        # Slightly larger offset for better visibility
        offset_y = 0.0055  # 10% increase from 0.005
        
        # Shift Gemini 2.5 Pro labels up by 1.5x the normal offset
        if d['model'].startswith('Gemini 2.5 Pro'):
            offset_y = offset_y * 3
        
        ax.annotate(model, 
                   (cost, score),
                   xytext=(cost, score + offset_y),
                   fontsize=12,
                   ha='center',
                   va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.1, edgecolor='none'))
    
    # Customize plot
    ax.set_xlabel('Estimated cost for 100,000 images', fontsize=22)
    ax.set_ylabel('Galaxy Classification Score', fontsize=22)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Format x-axis to show cost in dollars
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.4f}' if x < 0.01 else f'${x:.2f}'))
    
    # Set axis limits with some padding (adjusted for 100,000 images and halved costs)
    x_min = 5       # $5 for 100,000 images (halved from $10)
    x_max = 10000   # $10,000 for 100,000 images (halved from $20,000)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.34, 0.59)
    
    # Set y-axis ticks in steps of 0.05
    ax.set_yticks([0.35, 0.40, 0.45, 0.50, 0.55])
    
    # Add faint grid lines
    ax.grid(True, which='both', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Set specific x-axis tick locations for clarity (adjusted for 100,000 images)
    x_ticks = [10, 100, 1000, 5000, 20000]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'${x:,}' for x in x_ticks])  # Use comma formatting for thousands
    
    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=21.6)
    
    # Add a subtle background
    # ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Set grid behind plot elements
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nModel Summary (sorted by cost):")
    print("-" * 100)
    print(f"{'Model':<40} {'Avg Score':<12} {'SEM':<8} {'Perfect':<10} {'Avg Cost':<12} {'N':<5} {'Q':<5}")
    print("-" * 100)
    for i, d in enumerate(sorted(plot_data, key=lambda x: x['mean_cost'])):
        model_display = d['model'][:37] + "..." if len(d['model']) > 40 else d['model']
        perfect_info = f"{d.get('perfect_scores', 0)}/{d.get('total_attempts', 0)}"
        print(f"{model_display:<40} {d['mean_score']:<12.4f} {d['sem']:<8.4f} {perfect_info:<10} ${d['mean_cost']:<11.6f} {d['n']:<5} {d['n_questions']:<5}")

def main():
    import sys
    
    # Set paths with optional command-line arguments
    jsonl_dir = Path(sys.argv[1])
    
    output_path = Path(sys.argv[2])
    print(f"JSONL directory: {jsonl_dir}")
    print(f"Output path: {output_path}")
    # Analyze runs
    print("Analyzing runs for performance vs cost...")
    plot_data = analyze_runs_for_plot(jsonl_dir)
    
    if not plot_data:
        print("No data found to plot!")
        return
    
    # Create plot
    print("\nCreating plot...")
    create_plot(plot_data, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()