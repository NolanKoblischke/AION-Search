#!/usr/bin/env python3
"""
Script to create a summary HTML leaderboard of all evaluation runs.
Shows Prompt Filename, Plotting Filename, GZ Accuracy, Average Shell and Stream F1 Score, Lens F1 score.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from glob import glob
import os

# Add eval.combined to path for imports
sys.path.append('galaxybench/eval/combined')
from galaxybench.eval.combined.print_score import calculate_classification_metrics

def get_run_metadata(jsonl_file):
    """Extract metadata from the first record of a JSONL file."""
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return {
                    'prompt_filename': data.get('prompt_filename', 'Unknown'),
                    'plot_script': data.get('plot_script', 'Unknown'),
                    'model_name': data.get('formatted_name', data.get('model_name', 'Unknown'))
                }
    except Exception as e:
        print(f"Error reading metadata from {jsonl_file}: {e}")
    
    return {
        'prompt_filename': 'Unknown',
        'plot_script': 'Unknown', 
        'model_name': 'Unknown'
    }

def false_positive_penalty(fp_count):
    """Calculate false-positive penalty factor f(x)."""
    if fp_count == 0:
        return 1.0
    elif fp_count == 1:
        return 0.5
    elif fp_count == 2:
        return 0.1
    else:  # fp_count >= 3
        return 0.0

def calculate_run_scores(jsonl_file):
    """Calculate scores for a single run."""
    scores = {
        'gz_accuracy': 0.0,
        'tidal_f1_shell': 0.0,
        'tidal_f1_stream': 0.0,
        'tidal_f1_average': 0.0,
        'shell_fp': 0,
        'shell_total': 0,
        'shell_tp': 0,
        'stream_fp': 0,
        'stream_total': 0,
        'stream_tp': 0,
        'lens_f1': 0.0,
        'lens_fp': 0,
        'lens_total': 0,
        'lens_tp': 0,
        'agg_score': 0.0,
        'agg_score_2': 0.0,
        'model_count': 0
    }
    
    # Aggregate data per model
    model_scores = defaultdict(lambda: {
        'gz_scores': [],
        'tidal_confusion': defaultdict(lambda: defaultdict(int)),
        'tidal_total': 0,
        'lens_tp': 0, 'lens_tn': 0, 'lens_fp': 0, 'lens_fn': 0
    })
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    model = data.get("formatted_name") or data.get("model_name", "unknown")
                    judge_results = data.get("judge_results", {})
                    
                    # Galaxy Zoo scores (using decision_tree_score)
                    if 'decision_tree_score' in data and data['decision_tree_score'] is not None:
                        model_scores[model]['gz_scores'].append(data['decision_tree_score'])
                    
                    # Tidal scores (build confusion matrix)
                    if 'tidal_score' in data and judge_results:
                        judge_classification = judge_results.get('judge_classification', {})
                        tidal_info = judge_results.get('judge_tidal_info', {})
                        if tidal_info:
                            true_class = tidal_info.get('eval_class', 'Other')
                            predicted_class = judge_classification.get('classification', 'Other')
                            model_scores[model]['tidal_confusion'][true_class][predicted_class] += 1
                            model_scores[model]['tidal_total'] += 1
                    
                    # Lens scores (accumulate confusion matrix values)
                    if 'description_says_lens_occuring_score' in data and judge_results:
                        correct_prediction = judge_results.get('correct_prediction', False)
                        ground_truth_lens = judge_results.get('ground_truth_lens', False)
                        
                        # Determine if this is actually a lens (ground truth)
                        is_lens = data.get('is_lens', False)
                        lensgrade = data.get('lensgrade', 'Unknown')
                        galaxy_class = data.get('class', lensgrade)
                        is_actual_lens = galaxy_class in ['A', 'B', 'C'] or is_lens or ground_truth_lens
                        
                        # Classification metrics based on correct_prediction
                        if is_actual_lens and correct_prediction:
                            model_scores[model]['lens_tp'] += 1
                        elif is_actual_lens and not correct_prediction:
                            model_scores[model]['lens_fn'] += 1
                        elif not is_actual_lens and correct_prediction:
                            model_scores[model]['lens_tn'] += 1
                        elif not is_actual_lens and not correct_prediction:
                            model_scores[model]['lens_fp'] += 1
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line in {jsonl_file}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading {jsonl_file}: {e}")
        return scores
    
    # Calculate average scores across all models
    gz_scores_all = []
    tidal_f1_shell_all = []
    tidal_f1_stream_all = []
    shell_fp_all = []
    shell_total_all = []
    shell_tp_all = []
    stream_fp_all = []
    stream_total_all = []
    stream_tp_all = []
    lens_f1_all = []
    lens_fp_all = []
    lens_total_all = []
    lens_tp_all = []
    
    for model, data in model_scores.items():
        # Galaxy Zoo accuracy
        if data['gz_scores']:
            gz_scores_all.extend(data['gz_scores'])
        
        # Tidal F1 scores - calculate for both classes even if they don't appear
        confusion_matrix = data['tidal_confusion']
        if data['tidal_total'] > 0:
            # Always calculate both Shell and Stream F1 scores
            shell_tp = confusion_matrix.get('Shell', {}).get('Shell', 0)
            shell_total_true = sum(confusion_matrix.get('Shell', {}).values())
            shell_total_predicted = sum(confusion_matrix.get(tc, {}).get('Shell', 0) for tc in ['Shell', 'Stream', 'Other'])
            shell_fp = shell_total_predicted - shell_tp
            
            shell_recall = shell_tp / shell_total_true if shell_total_true > 0 else 0
            shell_precision = shell_tp / shell_total_predicted if shell_total_predicted > 0 else 0
            shell_f1 = 2 * shell_precision * shell_recall / (shell_precision + shell_recall) if (shell_precision + shell_recall) > 0 else 0
            
            stream_tp = confusion_matrix.get('Stream', {}).get('Stream', 0)
            stream_total_true = sum(confusion_matrix.get('Stream', {}).values())
            stream_total_predicted = sum(confusion_matrix.get(tc, {}).get('Stream', 0) for tc in ['Shell', 'Stream', 'Other'])
            stream_fp = stream_total_predicted - stream_tp
            
            stream_recall = stream_tp / stream_total_true if stream_total_true > 0 else 0
            stream_precision = stream_tp / stream_total_predicted if stream_total_predicted > 0 else 0
            stream_f1 = 2 * stream_precision * stream_recall / (stream_precision + stream_recall) if (stream_precision + stream_recall) > 0 else 0
            
            tidal_f1_shell_all.append(shell_f1)
            tidal_f1_stream_all.append(stream_f1)
            shell_fp_all.append(shell_fp)
            shell_total_all.append(shell_total_true)
            shell_tp_all.append(shell_tp)
            stream_fp_all.append(stream_fp)
            stream_total_all.append(stream_total_true)
            stream_tp_all.append(stream_tp)
        
        # Lens F1 score and FP/Total
        tp, tn, fp, fn = data['lens_tp'], data['lens_tn'], data['lens_fp'], data['lens_fn']
        if tp + tn + fp + fn > 0:
            metrics = calculate_classification_metrics(tp, tn, fp, fn)
            lens_f1_all.append(metrics['f1'])
            lens_fp_all.append(fp)
            lens_total_all.append(tp + fn)  # Total true positive instances (TP + FN)
            lens_tp_all.append(tp)
    
    # Calculate averages and sums
    scores['model_count'] = len(model_scores)
    scores['gz_accuracy'] = sum(gz_scores_all) / len(gz_scores_all) if gz_scores_all else 0.0
    scores['tidal_f1_shell'] = sum(tidal_f1_shell_all) / len(tidal_f1_shell_all) if tidal_f1_shell_all else 0.0
    scores['tidal_f1_stream'] = sum(tidal_f1_stream_all) / len(tidal_f1_stream_all) if tidal_f1_stream_all else 0.0
    scores['tidal_f1_average'] = (scores['tidal_f1_shell'] + scores['tidal_f1_stream']) / 2 if (tidal_f1_shell_all or tidal_f1_stream_all) else 0.0
    scores['shell_fp'] = sum(shell_fp_all)
    scores['shell_total'] = sum(shell_total_all)
    scores['shell_tp'] = sum(shell_tp_all)
    scores['stream_fp'] = sum(stream_fp_all)
    scores['stream_total'] = sum(stream_total_all)
    scores['stream_tp'] = sum(stream_tp_all)
    scores['lens_f1'] = sum(lens_f1_all) / len(lens_f1_all) if lens_f1_all else 0.0
    scores['lens_fp'] = sum(lens_fp_all)
    scores['lens_total'] = sum(lens_total_all)
    scores['lens_tp'] = sum(lens_tp_all)
    
    # Calculate aggregate score (average of GZ accuracy, Avg Tidal F1, and Lens F1)
    scores['agg_score'] = (scores['gz_accuracy'] + scores['tidal_f1_average'] + scores['lens_f1']) / 3
    
    # Calculate Agg. Score 2 with false-positive penalties and weights
    # Default weights: w_GZ=0.4, w_lens=0.3, w_tidal=0.3
    w_gz = 0.4
    w_lens = 0.3
    w_tidal = 0.3
    
    # GZ component
    gz_component = w_gz * scores['gz_accuracy']
    
    # Lens component with FP penalty
    lens_fp_penalty = false_positive_penalty(scores['lens_fp'])
    lens_recall = scores['lens_tp'] / scores['lens_total'] if scores['lens_total'] > 0 else 0.0
    lens_component = w_lens * lens_fp_penalty * lens_recall
    
    # Tidal component with FP penalties for both shell and stream
    shell_fp_penalty = false_positive_penalty(scores['shell_fp'])
    stream_fp_penalty = false_positive_penalty(scores['stream_fp'])
    shell_recall = scores['shell_tp'] / scores['shell_total'] if scores['shell_total'] > 0 else 0.0
    stream_recall = scores['stream_tp'] / scores['stream_total'] if scores['stream_total'] > 0 else 0.0
    tidal_component = w_tidal * shell_fp_penalty * stream_fp_penalty * (shell_recall + stream_recall) / 2
    
    scores['agg_score_2'] = gz_component + lens_component + tidal_component
    
    return scores

def generate_html_leaderboard(runs_data):
    """Generate HTML leaderboard with sortable columns."""
    
    # Calculate max values for color scaling
    max_gz = max(run["gz_accuracy"] for run in runs_data)
    max_shell_f1 = max(run["tidal_f1_shell"] for run in runs_data)
    max_stream_f1 = max(run["tidal_f1_stream"] for run in runs_data)
    max_avg_tidal_f1 = max(run["tidal_f1_average"] for run in runs_data)
    max_shell_fp = max(run["shell_fp"] for run in runs_data)
    max_shell_total = max(run["shell_total"] for run in runs_data)
    max_stream_fp = max(run["stream_fp"] for run in runs_data)
    max_stream_total = max(run["stream_total"] for run in runs_data)
    max_lens_f1 = max(run["lens_f1"] for run in runs_data)
    max_lens_fp = max(run["lens_fp"] for run in runs_data)
    max_lens_total = max(run["lens_total"] for run in runs_data)
    max_agg_score = max(run["agg_score"] for run in runs_data)
    max_agg_score_2 = max(run["agg_score_2"] for run in runs_data)
    
    # Generate table rows first
    table_rows = []
    for run in runs_data:
        timestamp = run['timestamp']
        prompt_filename = f'<span class="filename">{run["prompt_filename"]}</span>'
        plot_script = f'<span class="filename">{run["plot_script"]}</span>'
        model_name = f'<span class="model-name">{run["model_name"]}</span>'
        
        gz_accuracy = f'{run["gz_accuracy"]:.3f}' if run["gz_accuracy"] >= 0 else 'N/A'
        shell_f1 = f'{run["tidal_f1_shell"]:.3f}' if run["tidal_f1_shell"] >= 0 else 'N/A'
        stream_f1 = f'{run["tidal_f1_stream"]:.3f}' if run["tidal_f1_stream"] >= 0 else 'N/A'
        avg_tidal_f1 = f'{run["tidal_f1_average"]:.3f}' if run["tidal_f1_average"] >= 0 else 'N/A'
        shell_fp_total = f'{run["shell_fp"]}/{run["shell_total"]} ({run["shell_tp"]})' if run["shell_total"] > 0 else 'N/A'
        stream_fp_total = f'{run["stream_fp"]}/{run["stream_total"]} ({run["stream_tp"]})' if run["stream_total"] > 0 else 'N/A'
        lens_f1 = f'{run["lens_f1"]:.3f}' if run["lens_f1"] >= 0 else 'N/A'
        lens_fp_total = f'{run["lens_fp"]}/{run["lens_total"]} ({run["lens_tp"]})' if run["lens_total"] > 0 else 'N/A'
        agg_score = f'{run["agg_score"]:.3f}' if run["agg_score"] >= 0 else 'N/A'
        agg_score_2 = f'{run["agg_score_2"]:.3f}' if run["agg_score_2"] >= 0 else 'N/A'
        
        # Calculate color intensities (0-1 scale based on max values)
        # For FP/Total, lower FP ratio is better, so we invert the intensity
        gz_intensity = run["gz_accuracy"] / max_gz if max_gz > 0 else 0
        shell_intensity = run["tidal_f1_shell"] / max_shell_f1 if max_shell_f1 > 0 else 0
        stream_intensity = run["tidal_f1_stream"] / max_stream_f1 if max_stream_f1 > 0 else 0
        avg_tidal_intensity = run["tidal_f1_average"] / max_avg_tidal_f1 if max_avg_tidal_f1 > 0 else 0
        # For FP/Total ratios, we want lower values (fewer false positives) to be green
        shell_fp_ratio = run["shell_fp"] / run["shell_total"] if run["shell_total"] > 0 else 0
        max_shell_fp_ratio = max_shell_fp / max_shell_total if max_shell_total > 0 else 0
        shell_fp_intensity = 1 - (shell_fp_ratio / max_shell_fp_ratio) if max_shell_fp_ratio > 0 else 1
        
        stream_fp_ratio = run["stream_fp"] / run["stream_total"] if run["stream_total"] > 0 else 0
        max_stream_fp_ratio = max_stream_fp / max_stream_total if max_stream_total > 0 else 0
        stream_fp_intensity = 1 - (stream_fp_ratio / max_stream_fp_ratio) if max_stream_fp_ratio > 0 else 1
        
        lens_intensity = run["lens_f1"] / max_lens_f1 if max_lens_f1 > 0 else 0
        
        lens_fp_ratio = run["lens_fp"] / run["lens_total"] if run["lens_total"] > 0 else 0
        max_lens_fp_ratio = max_lens_fp / max_lens_total if max_lens_total > 0 else 0
        lens_fp_intensity = 1 - (lens_fp_ratio / max_lens_fp_ratio) if max_lens_fp_ratio > 0 else 1
        
        agg_intensity = run["agg_score"] / max_agg_score if max_agg_score > 0 else 0
        agg_intensity_2 = run["agg_score_2"] / max_agg_score_2 if max_agg_score_2 > 0 else 0
        
        row = f"""
                <tr>
                    <td><span class="timestamp">{timestamp}</span></td>
                    <td>{prompt_filename}</td>
                    <td>{plot_script}</td>
                    <td>{model_name}</td>
                    <td class="score" data-intensity="{gz_intensity:.3f}">{gz_accuracy}</td>
                    <td class="score" data-intensity="{shell_intensity:.3f}">{shell_f1}</td>
                    <td class="score" data-intensity="{stream_intensity:.3f}">{stream_f1}</td>
                    <td class="score" data-intensity="{avg_tidal_intensity:.3f}">{avg_tidal_f1}</td>
                    <td class="score" data-intensity="{shell_fp_intensity:.3f}">{shell_fp_total}</td>
                    <td class="score" data-intensity="{stream_fp_intensity:.3f}">{stream_fp_total}</td>
                    <td class="score" data-intensity="{lens_intensity:.3f}">{lens_f1}</td>
                    <td class="score" data-intensity="{lens_fp_intensity:.3f}">{lens_fp_total}</td>
                    <td class="score agg-score" data-intensity="{agg_intensity:.3f}">{agg_score}</td>
                    <td class="score agg-score" data-intensity="{agg_intensity_2:.3f}">{agg_score_2}</td>
                </tr>"""
        table_rows.append(row)
    
    # Create the full HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Galaxy Benchmark - Runs Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 25px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }}
        th, td {{
            padding: 8px 6px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
            cursor: pointer;
            position: relative;
            user-select: none;
        }}
        th:hover {{
            background-color: #2c3e50;
        }}
        th.sortable::after {{
            content: ' ↕️';
            font-size: 0.8em;
        }}
        th.sort-asc::after {{
            content: ' ↑';
            color: #3498db;
        }}
        th.sort-desc::after {{
            content: ' ↓';
            color: #3498db;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .score {{
            font-weight: bold;
            padding: 6px 8px;
            border-radius: 4px;
        }}
        .agg-score {{
            border: 2px solid #7f8c8d;
            font-weight: 900;
        }}
        .filename {{
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .model-name {{
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Galaxy Benchmark - Runs Summary</h1>
        
        <div class="summary">
            <strong>{len(runs_data)}</strong> evaluation runs analyzed
        </div>
        
        <table id="leaderboard">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(0)">Timestamp</th>
                    <th class="sortable" onclick="sortTable(1)">Prompt Filename</th>
                    <th class="sortable" onclick="sortTable(2)">Plotting Script</th>
                    <th class="sortable" onclick="sortTable(3)">Model</th>
                    <th class="sortable" onclick="sortTable(4)">GZ Accuracy</th>
                    <th class="sortable" onclick="sortTable(5)">Shell F1</th>
                    <th class="sortable" onclick="sortTable(6)">Stream F1</th>
                    <th class="sortable" onclick="sortTable(7)">Avg Tidal F1</th>
                    <th class="sortable" onclick="sortTable(8)">Shell FP/Total (TP)</th>
                    <th class="sortable" onclick="sortTable(9)">Stream FP/Total (TP)</th>
                    <th class="sortable" onclick="sortTable(10)">Lens F1</th>
                    <th class="sortable" onclick="sortTable(11)">Lens FP/Total (TP)</th>
                    <th class="sortable" onclick="sortTable(12)">Agg. Score</th>
                    <th class="sortable" onclick="sortTable(13)">Agg. Score 2</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>

    <script>
        let currentSort = {{ column: -1, direction: 'asc' }};

        function sortTable(columnIndex) {{
            const table = document.getElementById('leaderboard');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            if (currentSort.column === columnIndex) {{
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSort.direction = 'asc';
                currentSort.column = columnIndex;
            }}
            
            table.querySelectorAll('th').forEach(th => {{
                th.classList.remove('sort-asc', 'sort-desc');
            }});
            
            const header = table.querySelectorAll('th')[columnIndex];
            header.classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
            
            rows.sort((a, b) => {{
                const aValue = a.cells[columnIndex].textContent.trim();
                const bValue = b.cells[columnIndex].textContent.trim();
                
                let result = 0;
                
                if (columnIndex >= 4 && columnIndex <= 13) {{
                    // Handle FP/Total format for columns 8, 9, and 11
                    if (columnIndex === 8 || columnIndex === 9 || columnIndex === 11) {{
                        const aRatio = aValue.includes('/') ? parseFloat(aValue.split('/')[0]) / parseFloat(aValue.split('/')[1]) : 0;
                        const bRatio = bValue.includes('/') ? parseFloat(bValue.split('/')[0]) / parseFloat(bValue.split('/')[1]) : 0;
                        result = aRatio - bRatio;
                    }} else {{
                        const aNum = parseFloat(aValue) || 0;
                        const bNum = parseFloat(bValue) || 0;
                        result = aNum - bNum;
                    }}
                }} else {{
                    result = aValue.localeCompare(bValue);
                }}
                
                return currentSort.direction === 'asc' ? result : -result;
            }});
            
            rows.forEach(row => tbody.appendChild(row));
        }}

        function applyRedToGreenColor(intensity) {{
            // Muted red to green color scheme
            // Red: #c0392b (muted red), Green: #27ae60 (muted green)
            const red = [192, 57, 43];    // Muted red
            const green = [39, 174, 96];  // Muted green
            
            const r = Math.round(red[0] + (green[0] - red[0]) * intensity);
            const g = Math.round(red[1] + (green[1] - red[1]) * intensity);
            const b = Math.round(red[2] + (green[2] - red[2]) * intensity);
            
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            // Apply red-to-green color scheme based on data-intensity
            const scoreCells = document.querySelectorAll('.score');
            scoreCells.forEach(cell => {{
                const intensity = parseFloat(cell.getAttribute('data-intensity')) || 0;
                const backgroundColor = applyRedToGreenColor(intensity);
                cell.style.backgroundColor = backgroundColor;
                
                // Add text color for better contrast
                if (intensity > 0.5) {{
                    cell.style.color = 'white';
                }} else {{
                    cell.style.color = 'black';
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    return html_content

def main():
    print("🌌 Galaxy Benchmark - Runs Summary Generator")
    print("=" * 50)
    
    # Find all JSONL files in runs directory
    jsonl_files = glob("eval/runs/jsonl/*.jsonl")
    
    if not jsonl_files:
        print("No JSONL files found in eval/runs/jsonl/")
        return
    
    print(f"Found {len(jsonl_files)} evaluation runs to analyze...")
    
    runs_data = []
    
    for jsonl_file in sorted(jsonl_files):
        filename = Path(jsonl_file).name
        print(f"  Analyzing {filename}...")
        
        # Extract timestamp from filename
        timestamp = filename.replace('judged_all_evals_', '').replace('.jsonl', '')
        
        # Get metadata
        metadata = get_run_metadata(jsonl_file)
        
        # Calculate scores
        scores = calculate_run_scores(jsonl_file)
        
        # Combine data
        run_data = {
            'timestamp': timestamp,
            'filename': filename,
            **metadata,
            **scores
        }
        
        runs_data.append(run_data)
        
        print(f"    ✅ GZ: {scores['gz_accuracy']:.3f}, Tidal F1: {scores['tidal_f1_average']:.3f}, Lens F1: {scores['lens_f1']:.3f}")
    
    # Sort by timestamp (newest first)
    runs_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Generate HTML
    print(f"\n📊 Generating HTML leaderboard...")
    html_content = generate_html_leaderboard(runs_data)
    
    # Save HTML file
    html_output_file = "eval/runs_summary.html"
    with open(html_output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save JSONL file with all run data
    print(f"\n💾 Generating JSONL summary...")
    jsonl_output_file = "eval/runs_summary.jsonl"
    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
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
                'gz_accuracy': run_data['gz_accuracy'],
                
                # Tidal scores
                'tidal_f1_shell': run_data['tidal_f1_shell'],
                'tidal_f1_stream': run_data['tidal_f1_stream'],
                'tidal_f1_average': run_data['tidal_f1_average'],
                'shell_fp': run_data['shell_fp'],
                'shell_total': run_data['shell_total'],
                'shell_tp': run_data['shell_tp'],
                'stream_fp': run_data['stream_fp'],
                'stream_total': run_data['stream_total'],
                'stream_tp': run_data['stream_tp'],
                
                # Lens scores
                'lens_f1': run_data['lens_f1'],
                'lens_fp': run_data['lens_fp'],
                'lens_total': run_data['lens_total'],
                'lens_tp': run_data['lens_tp'],
                
                # Aggregate scores
                'agg_score': run_data['agg_score'],
                'agg_score_2': run_data['agg_score_2']
            }
            f.write(json.dumps(summary_record) + '\n')
    
    print(f"✨ Summary generated successfully!")
    print(f"📁 HTML saved to: {Path(html_output_file).absolute()}")
    print(f"🌐 Open in browser: file://{Path(html_output_file).absolute()}")
    print(f"📁 JSONL saved to: {Path(jsonl_output_file).absolute()}")

if __name__ == "__main__":
    main()