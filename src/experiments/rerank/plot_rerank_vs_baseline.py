#!/usr/bin/env python3
"""
Standalone script to plot reranking results vs AION baseline.
Creates a vertical stack of plots showing Lenses@k vs Total Cost.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PIL import Image
import argparse

def load_aion_baseline(eval_summary_path, eval_name='lens_hsc'):
    """Load AION baseline results from eval_summary.jsonl"""
    baseline_data = {}
    
    # Read all matching records and use the last one (most recent) that has lenses@k metrics
    with open(eval_summary_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                if record.get('eval_name') == eval_name and record.get('model') == 'aion_baseline':
                    metrics = record.get('metrics', {})
                    # Check if this record has lenses@k metrics
                    if 'lenses@10' in metrics:
                        # Update baseline_data with this record's values
                        for k in [10, 20, 30, 50, 100]:
                            key = f'lenses@{k}'
                            if key in metrics:
                                baseline_data[k] = metrics[key]
            except json.JSONDecodeError:
                # Skip lines with JSON errors (e.g., multiple JSON objects on one line)
                print(f"  Skipping line {line_num} due to JSON error")
                continue
    
    return baseline_data

def load_no_rerank_data(experiment_summary_path):
    """Load no-rerank (original) values from experiment summary."""
    no_rerank_data = {}
    
    with open(experiment_summary_path, 'r') as f:
        summary = json.load(f)
    
    # Get original_lenses values from first experiment (they're the same for all)
    if summary['experiments']:
        first_exp = summary['experiments'][0]
        for k in [10, 20, 30, 50, 100]:
            key = f'original_lenses@{k}_mean'
            if key in first_exp:
                no_rerank_data[k] = first_exp[key]
    
    return no_rerank_data

def create_plot(experiment_dir, eval_summary_path, output_path='rerank_vs_baseline.png', lens_viz_csv=None, lens_viz_data_dir=None):
    """Create the main plot comparing reranking to baseline."""
    
    # Load experiment summary
    experiment_summary_path = Path(experiment_dir) / 'experiment_summary.json'
    with open(experiment_summary_path, 'r') as f:
        experiment_summary = json.load(f)
    
    # Load baseline data
    baseline_data = load_aion_baseline(eval_summary_path)
    no_rerank_data = load_no_rerank_data(experiment_summary_path)
    
    # Print baseline data for debugging
    print("\nAION Baseline values:")
    for k, value in baseline_data.items():
        print(f"  Lenses@{k}: {value}")
    
    print("\nNo Rerank values:")
    for k, value in no_rerank_data.items():
        print(f"  Lenses@{k}: {value}")
    
    # Convert experiment summary to plotting data
    plot_data = []
    for exp in experiment_summary['experiments']:
        row = {
            'Model': exp['model_id'],
            'Model Name': exp.get('model_name', exp['model_id']),
            'Best-of-M': exp['best_of_m'],
            'Total Cost ($)': exp.get('total_cost_mean', 0)
        }
        
        # Add lenses@k values with error bars
        for k in [10, 20, 30, 50, 100]:
            mean_key = f'reranked_lenses@{k}_mean'
            stderr_key = f'reranked_lenses@{k}_stderr'
            if mean_key in exp:
                row[f'Lenses@{k}'] = exp[mean_key]
                row[f'Lenses@{k}_stderr'] = exp.get(stderr_key, 0)
        
        plot_data.append(row)
    
    df = pd.DataFrame(plot_data)
    
    # Define k values to plot
    k_values = [10, 20, 30, 50, 100]
    
    # Set up the figure with vertical subplots (1.25x taller: 36 * 1.25 = 45)
    fig, axes = plt.subplots(len(k_values), 1, figsize=(12, 45))
    
    # Create subplot directory
    subplot_dir = Path(output_path).parent / 'individual_subplots'
    subplot_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors for models (different shades of blue)
    model_colors = {
        'gpt-4.1-nano': '#87CEEB',  # light blue
        'gpt-4.1-mini': '#4682B4',  # steel blue
        'gpt-4.1': '#191970'        # midnight blue
    }
    
    # Define markers for best-of-m
    m_markers = {
        1: 'o',  # circle
        5: '^'   # triangle
    }
    
    # Set font sizes
    plt.rcParams.update({'font.size': 24})
    
    # Plot for each k value
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Get column name for this k
        lenses_col = f'Lenses@{k}'
        
        if lenses_col not in df.columns:
            continue
        
        # Collect data for R² calculation
        costs = []
        lenses_values = []
        
        # Create individual figure for this subplot
        fig_individual = plt.figure(figsize=(12, 9))
        ax_individual = fig_individual.add_subplot(111)
            
        # Plot each data point
        for _, row in df.iterrows():
            model_id = row['Model']
            best_of_m = row['Best-of-M']
            total_cost = row['Total Cost ($)']
            lenses = row[lenses_col]
            
            # Store for R² calculation
            costs.append(total_cost)
            lenses_values.append(lenses)
            
            # Get color and marker
            color = model_colors.get(model_id, '#333333')
            marker = m_markers.get(best_of_m, 'o')
            
            # Get error bar if available
            stderr_col = f'Lenses@{k}_stderr'
            stderr = row.get(stderr_col, 0)
            
            # Plot the point with error bars on both axes
            for ax_to_plot in [ax, ax_individual]:
                if stderr > 0:
                    ax_to_plot.errorbar(total_cost, lenses,
                                       yerr=stderr,
                                       marker=marker,
                                       color=color,
                                       markersize=20,  # 400 s parameter ~ 20 markersize
                                       markeredgecolor='black',
                                       markeredgewidth=2,
                                       alpha=1.0,  # Fully opaque
                                       capsize=5,
                                       capthick=2,
                                       elinewidth=2,
                                       linestyle='none')
                else:
                    ax_to_plot.scatter(total_cost, lenses, 
                              color=color, 
                              marker=marker, 
                              s=400,  # Large markers
                              alpha=1.0,  # Fully opaque
                              edgecolors='black',
                              linewidth=2)
        
        # Calculate R² if we have data points
        if len(costs) > 1:
            # Use log-transformed cost for linear regression since x-axis is log scale
            log_costs = np.log10(costs).reshape(-1, 1)
            lenses_array = np.array(lenses_values)
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(log_costs, lenses_array)
            
            # Calculate R²
            pred_lenses = reg.predict(log_costs)
            r2 = r2_score(lenses_array, pred_lenses)
            
            print(f"R² for k={k}: {r2:.4f}")
        
        # Add AION baseline as horizontal line
        if k in baseline_data:
            baseline_value = baseline_data[k]
            print(f"  Plotting baseline for k={k}: {baseline_value}")
            for ax_to_plot in [ax, ax_individual]:
                ax_to_plot.axhline(y=baseline_value, 
                          color='red', 
                          linestyle='--', 
                          linewidth=3,
                          alpha=0.8,
                          label='AION-1-B Similarity Search')
        
        # Add No Rerank as horizontal line
        if k in no_rerank_data:
            no_rerank_value = no_rerank_data[k]
            print(f"  Plotting no rerank for k={k}: {no_rerank_value}")
            for ax_to_plot in [ax, ax_individual]:
                ax_to_plot.axhline(y=no_rerank_value,
                          color='purple',
                          linestyle=':',
                          linewidth=3,
                          alpha=0.8,
                          label='AION-Search')
        
        # Apply formatting to both axes
        for ax_to_plot in [ax, ax_individual]:
            # Set labels and formatting
            ax_to_plot.set_xlabel('Total Cost ($)', fontsize=24)
            ax_to_plot.set_ylabel(f'Number of Lenses\nin Top-{k}', fontsize=24)
            ax_to_plot.set_xscale('log')
        
        # Format x-axis ticks to show dollar amounts
        def format_dollars(x, pos):
            """Format x-axis values as dollar amounts."""
            if x < 1:
                return f'${x:.2f}'
            else:
                return f'${x:.1f}'
        
        # Apply formatting to both axes
        for ax_to_plot in [ax, ax_individual]:
            ax_to_plot.xaxis.set_major_formatter(FuncFormatter(format_dollars))
            # Set specific tick locations for $0.01, $0.10, $1.0, $10.0
            ax_to_plot.set_xticks([0.01, 0.1, 1.0, 10.0])
            
            # Add grid
            ax_to_plot.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Set x-axis limits
            ax_to_plot.set_xlim(0.01, 10.0)
            
            # Set y-axis limits based on k value
            ylim_map = {
                10: 5,
                20: 10,
                30: 10,
                50: 12,
                100: 15
            }
            ax_to_plot.set_ylim(0, ylim_map[k])
            
            # Remove '0' from y-axis labels
            y_ticks = ax_to_plot.get_yticks()
            y_ticks = [tick for tick in y_ticks if tick != 0]
            ax_to_plot.set_yticks(y_ticks)
            
            # Set tick label sizes
            ax_to_plot.tick_params(axis='both', which='major', labelsize=24)
            ax_to_plot.tick_params(axis='both', which='minor', labelsize=24)
        
        # Add legend to every subplot with smaller font
        # Create custom legend entries organized in three columns
        model_elements = []
        bestofm_elements = []
        baseline_elements = []
        
        # Column 1: Model colors (as squares)
        for model_id, color in model_colors.items():
            model_elements.append(Line2D([0], [0], marker='s', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=model_id, markeredgecolor='black',
                                         markeredgewidth=1))
        
        # Column 2: Best-of-m markers
        for m, marker in m_markers.items():
            bestofm_elements.append(Line2D([0], [0], marker=marker, color='w',
                                         markerfacecolor='gray', markersize=10,
                                         label=f'Best-of-{m}', markeredgecolor='black',
                                         markeredgewidth=1))
        
        # Column 3: Baseline lines
        baseline_elements.append(Line2D([0], [0], color='red', linestyle='--',
                                     linewidth=2, label='AION-1-Base'))
        baseline_elements.append(Line2D([0], [0], color='purple', linestyle=':',
                                     linewidth=2, label='AION-Search'))
        
        # Combine all elements in the desired order
        legend_elements = model_elements + bestofm_elements + baseline_elements
        
        # Add legend to both axes
        for ax_to_plot in [ax, ax_individual]:
            ax_to_plot.legend(handles=legend_elements, 
                     loc='upper left',
                     fontsize=18,  # 12 * 1.5 = 18
                     frameon=False,  # Borderless legend
                     ncol=3)  # Three columns
        
        # Save individual subplot
        fig_individual.tight_layout()
        subplot_path = subplot_dir / f'lenses_at_{k}.png'
        fig_individual.savefig(subplot_path, dpi=150, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"  Saved individual subplot to: {subplot_path}")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Create lens visualization for gpt-4.1 best-of-5
    create_lens_visualization(output_path.parent, lens_viz_csv, lens_viz_data_dir)


def create_lens_visualization(output_dir, lens_viz_csv=None, lens_viz_data_dir=None):
    """Create visualization of top 3 lenses after reranking for gpt-4.1 best-of-5."""
    
    print("\nCreating lens visualization for gpt-4.1 best-of-5...")
    
    # Use provided data_dir or default
    if lens_viz_data_dir:
        data_dir = Path(lens_viz_data_dir)
    else:
        data_dir = Path("data/experiments/rerank/multi_model_optimized_20250812_232602/gpt-4.1_top1000_m5")
    
    npz_file = data_dir / "reranked_data_run1.npz"
    image_dir = data_dir / "galaxy_images_run1"
    common_data_file = data_dir.parent / "common_data.npz"
    
    if not all(p.exists() for p in [npz_file, image_dir, common_data_file]):
        print("Error: Required files not found for lens visualization")
        return
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    common_data = np.load(common_data_file, allow_pickle=True)
    
    # Extract data
    reranked_indices = data['reranked_indices']
    gpt_scores = data['gpt_scores']
    new_ranks = data['new_ranks']
    
    all_object_ids = common_data['object_ids']
    eval_grades = common_data['eval_grades']
    original_indices = common_data['original_indices']
    
    # Find top lenses after reranking
    top_lenses = []
    for i, idx in enumerate(reranked_indices):
        if i >= 100:  # Only check top 100 for efficiency
            break
        
        eval_grade = str(eval_grades[idx])
        if eval_grade in ['A', 'B', 'C']:  # True lens
            obj_id = str(all_object_ids[idx])
            original_rank = int(np.where(original_indices == idx)[0][0] + 1)
            new_rank = i + 1  # Position in reranked list
            gpt_score = float(gpt_scores[idx])
            
            # Check if image exists
            image_path = image_dir / f"{obj_id}_zoomed.png"
            if image_path.exists():
                top_lenses.append({
                    'object_id': obj_id,
                    'original_rank': original_rank,
                    'new_rank': new_rank,
                    'gpt_score': gpt_score,
                    'eval_grade': eval_grade,
                    'image_path': image_path
                })
                
                if len(top_lenses) >= 3:
                    break
    
    if not top_lenses:
        print("No lens images found in top reranked results")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, len(top_lenses), figsize=(5*len(top_lenses), 6))
    if len(top_lenses) == 1:
        axes = [axes]
    
    plt.rcParams.update({'font.size': 14})
    
    for idx, lens in enumerate(top_lenses):
        ax = axes[idx]
        
        # Load and display image
        img = Image.open(lens['image_path'])
        ax.imshow(img)
        ax.axis('off')
        
        # Add rank information
        title = f"#{lens['original_rank']} → #{lens['new_rank']}\nVLM Score: {lens['gpt_score']:.1f}"
        ax.set_title(title, fontsize=40, pad=10)
        
    plt.tight_layout()
    
    # Save figure
    lens_viz_path = output_dir / 'top_lenses_reranked.png'
    plt.savefig(lens_viz_path, dpi=150, bbox_inches='tight')
    print(f"Lens visualization saved to: {lens_viz_path}")
    plt.close()
    
    # Save explanations to text file if available
    save_lens_explanations(data, top_lenses, output_dir)


def save_lens_explanations(data, top_lenses, output_dir):
    """Save VLM explanations for the top lenses to a text file."""
    
    # Check if explanations exist in the data
    if 'gpt_explanations' not in data:
        print("No explanations found in NPZ file")
        return
    
    explanations = data['gpt_explanations']
    explanations_path = output_dir / 'top_lenses_explanations.txt'
    
    with open(explanations_path, 'w') as f:
        f.write("VLM Explanations for Top 3 Lenses After Reranking\n")
        f.write("=" * 60 + "\n\n")
        
        for i, lens in enumerate(top_lenses, 1):
            f.write(f"Lens {i}: Object ID {lens['object_id']}\n")
            f.write(f"Original Rank: #{lens['original_rank']}\n")
            f.write(f"New Rank: #{lens['new_rank']}\n")
            f.write(f"VLM Score: {lens['gpt_score']:.1f}\n")
            f.write(f"Eval Grade: {lens['eval_grade']}\n")
            
            # Get the explanation for this object
            # Find the index in the original data
            reranked_indices = data['reranked_indices']
            idx_in_reranked = lens['new_rank'] - 1
            original_idx = reranked_indices[idx_in_reranked]
            
            if original_idx < len(explanations):
                explanation = str(explanations[original_idx])
                f.write(f"\nVLM Explanation:\n{explanation}\n")
            else:
                f.write(f"\nVLM Explanation: Not available\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
    
    print(f"Lens explanations saved to: {explanations_path}")


def main():
    """Main function to run the plotting script."""
    
    parser = argparse.ArgumentParser(description="Plot reranking results vs AION baseline with error bars")
    parser.add_argument("--experiment-dir", type=str, 
                       default="data/experiments/rerank/multi_model_optimized_20250812_232602",
                       help="Path to the experiment directory containing experiment_summary.json")
    parser.add_argument("--eval-summary", type=str,
                       default="data/eval_results/eval_summary.jsonl",
                       help="Path to the eval summary JSONL file")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output path for the plot (default: experiment_dir/plots/rerank_vs_baseline.png)")
    parser.add_argument("--lens-viz-csv", type=str, default=None,
                       help="Optional CSV file for lens visualization")
    parser.add_argument("--lens-viz-data-dir", type=str, default=None,
                       help="Optional data directory for lens visualization")
    args = parser.parse_args()
    
    # Define paths
    experiment_dir = Path(args.experiment_dir)
    eval_summary_path = Path(args.eval_summary)
    
    # Set default output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = experiment_dir / "plots" / "rerank_vs_baseline.png"
    
    # Check if files exist
    experiment_summary_path = experiment_dir / "experiment_summary.json"
    if not experiment_summary_path.exists():
        print(f"Error: experiment_summary.json not found at {experiment_summary_path}")
        return
    
    if not eval_summary_path.exists():
        print(f"Error: Eval summary file not found at {eval_summary_path}")
        return
    
    # Create the plot
    print("Creating plot...")
    create_plot(experiment_dir, eval_summary_path, output_path, args.lens_viz_csv, args.lens_viz_data_dir)
    print("Done!")

if __name__ == "__main__":
    main()