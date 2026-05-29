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
import argparse

def load_model_prices():
    """Load model pricing information from models.jsonl."""
    script_dir = Path(__file__).parent.parent
    models_file = script_dir / "galaxybench" / "eval" / "models.jsonl"
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

def get_excluded_models():
    return [
        "Gemini 2.5 Pro Preview 05-06 No Thinking",
        "Gemini 2.5 Pro Preview Thinking",
        "Gemini 2.5 Pro Preview 05-06 Thinking",
        "o3 Reasoning Low",
        "o4-mini Reasoning Low",
        "o4-mini Reasoning High",
        "GPT-4o",
        "Gemini 1.5 Flash",
        "Gemini 2.0 Flash",
    ]

def normalize_model_name(model_name):
    return (
        model_name.replace(' Reasoning', '')
        .replace(' Medium', '')
        .replace(' High', '')
        .replace(' Max Thinking', '')
        .replace(' Thinking', '')
        .replace(' Instruct', '')
        .replace(' Preview 06-05', '')
        .replace(' Preview 05-20', '')
        .replace('Gemini 2.0', 'Gemini 2')
        .replace('Qwen3 VL', 'Qwen3')
    )

def prepare_plot_points(plot_data):
    excluded_models = get_excluded_models()
    filtered = [d for d in plot_data if d['model'] not in excluded_models and "No Thinking" not in d['model']]
    models = [normalize_model_name(d['model']) for d in filtered]
    mean_scores = [d['mean_score'] * 100 for d in filtered]
    sems = [d['sem'] * 100 for d in filtered]
    mean_costs = [d['mean_cost'] * 100000 * 0.5 for d in filtered]
    return filtered, models, mean_scores, sems, mean_costs

def compute_even_label_layout(mean_costs, mean_scores, x_min, x_max):
    points = list(zip(mean_costs, mean_scores))
    n = len(points)
    if n == 0:
        return []

    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    x_slots_log = np.linspace(np.log10(x_min * 1.2), np.log10(x_max * 0.75), n_cols)
    y_slots = np.linspace(8, 92, n_rows)

    slots = []
    for r, y in enumerate(y_slots):
        row_xs = x_slots_log if r % 2 == 0 else x_slots_log[::-1]
        for xl in row_xs:
            slots.append((xl, y))
    slots = slots[:n]

    unassigned_points = set(range(n))
    unassigned_slots = set(range(n))
    assignment = {}

    while unassigned_points:
        best_pair = None
        best_dist = float('inf')
        for pi in unassigned_points:
            p_cost, p_score = points[pi]
            p_logx = np.log10(max(p_cost, 1e-9))
            for si in unassigned_slots:
                s_logx, s_y = slots[si]
                dx = (p_logx - s_logx) / (np.log10(x_max) - np.log10(x_min))
                dy = (p_score - s_y) / 100.0
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (pi, si)

        pi, si = best_pair
        assignment[pi] = si
        unassigned_points.remove(pi)
        unassigned_slots.remove(si)

    layout = []
    for pi in range(n):
        slot_idx = assignment[pi]
        layout.append((10 ** slots[slot_idx][0], slots[slot_idx][1]))
    return layout

def find_saved_label_position(saved_positions, model):
    if model in saved_positions:
        return saved_positions[model]

    # Backward compatibility with older saved keys before stripping " High".
    legacy_high_key = f"{model} High"
    if legacy_high_key in saved_positions:
        return saved_positions[legacy_high_key]

    # Forward compatibility in case saved file has stripped key but model includes High.
    if model.endswith(" High"):
        no_high_key = model[:-5]
        if no_high_key in saved_positions:
            return saved_positions[no_high_key]

    return None

def create_plot(plot_data, output_path, label_positions_path=None, show_arrows=True):
    """Create scatter plot of performance vs cost."""
    
    if not plot_data:
        print("No data to plot!")
        return
    
    plot_data, models, mean_scores, sems, mean_costs = prepare_plot_points(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9.8, 7))
    
    # Define colors for each model
    base_colors = ['#4285F4', '#34A853', '#EA4335', '#FBBC05', '#9333EA', '#EC4899', '#10B981', '#F59E0B']
    colors = []
    
    # Assign colors based on model names
    for i, d in enumerate(plot_data):
        model_name = d['model']
        if model_name.startswith('GPT-4.1') or model_name.startswith('GPT-4.5'):
            colors.append('#87CEEB')  # Light blue
        elif model_name.startswith('GPT-5'):
            colors.append('#DDA0DD')  # Light purple (plum)
        elif model_name.startswith('Qwen3'):
            colors.append('#E9A23B')  # Orange
        elif model_name.startswith('Gemini'):
            colors.append('#17B779')  # Green
        elif model_name.startswith('o4-mini'):
            colors.append('#E74C3C')  # Red
        else:
            # Use default colors for other models
            colors.append(base_colors[i % len(base_colors)])
    
    # Add error bars for SEM (in background with low z-order)
    ax.errorbar(mean_costs, mean_scores, yerr=sems,
                fmt='none',  # No markers (we already have scatter points)
                ecolor='#333333',
                elinewidth=1.2,
                capsize=3,
                alpha=0.8,
                zorder=0)  # Put in far background
    
    # Create scatter plot on top
    scatter = ax.scatter(mean_costs, mean_scores, 
                        s=300,  # Large marker size
                        c=colors, 
                        alpha=1.0,  # Fully opaque
                        edgecolor='black', 
                        linewidth=2,
                        zorder=2)  # Put scatter points above error bars
    
    # Customize plot
    ax.set_xlabel('Estimated cost for 100,000 images', fontsize=22)
    ax.set_ylabel('Galaxy Classification Score', fontsize=22)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Format x-axis to show cost in dollars
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.4f}' if x < 0.01 else f'${x:.2f}'))
    
    # Set axis limits with some padding (adjusted for 100,000 images and halved costs)
    x_min = 5       # $5 for 100,000 images (halved from $10)
    x_max = 10000   # $10,000 for 100,000 images
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)

    # Set y-axis ticks in steps of 10
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Add grid lines
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.6)
    ax.grid(True, which='major', alpha=0.45, linestyle='-', linewidth=1.0)
    
    # Set specific x-axis tick locations for clarity (adjusted for 100,000 images)
    x_ticks = [10, 100, 1000, 10000]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'${x:,}' for x in x_ticks])  # Use comma formatting for thousands
    
    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=21.6)
    
    # Add a subtle background
    # ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Set grid behind plot elements
    ax.set_axisbelow(True)

    saved_positions = {}
    if label_positions_path and Path(label_positions_path).exists():
        with open(label_positions_path, 'r') as f:
            raw_positions = json.load(f)
        for key, val in raw_positions.items():
            saved_positions[normalize_model_name(key)] = val

    auto_layout = compute_even_label_layout(mean_costs, mean_scores, x_min, x_max)
    for i, (cost, score, model) in enumerate(zip(mean_costs, mean_scores, models)):
        saved = find_saved_label_position(saved_positions, model)
        if saved is not None:
            label_x = float(saved["x"])
            label_y = float(saved["y"])
            ha = 'center'
        else:
            label_x, label_y = auto_layout[i]
            ha = 'left' if label_x >= cost else 'right'

        annotate_kwargs = dict(
            xy=(cost, score),
            xytext=(label_x, label_y),
            fontsize=15,
            ha=ha,
            va='center',
        )
        if show_arrows and model != "GPT-5 Nano":
            annotate_kwargs["arrowprops"] = dict(
                arrowstyle='->',
                color='#888888',
                lw=1.0,
                alpha=0.5,
                # Start arrow from bottom-center of label bbox.
                relpos=(0.5, 0.0),
                # Stop arrow at marker border (scatter s=300 => ~9.8 pt radius).
                shrinkB=10,
            )
        ax.annotate(model, **annotate_kwargs)

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
    parser = argparse.ArgumentParser(description="Plot model performance vs cost.")
    parser.add_argument("jsonl_dir", type=Path, help="Directory containing JSONL run files")
    parser.add_argument("output_path", type=Path, help="Output image path")
    parser.add_argument("--label-positions", type=Path, default=None, help="Optional JSON file with saved label positions")
    parser.add_argument("--no-arrows", action="store_true", help="Render labels without arrows")
    args = parser.parse_args()

    jsonl_dir = args.jsonl_dir
    output_path = args.output_path
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
    create_plot(
        plot_data,
        output_path,
        label_positions_path=args.label_positions,
        show_arrows=not args.no_arrows,
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
