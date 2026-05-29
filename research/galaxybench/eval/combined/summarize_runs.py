#!/usr/bin/env python3
"""Create a compact Galaxy Zoo leaderboard for judged evaluation runs."""

import json
from glob import glob
from pathlib import Path


def get_run_metadata(jsonl_file):
    """Extract metadata from the first record of a JSONL file."""
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return {
                    "prompt_filename": data.get("prompt_filename", "Unknown"),
                    "plot_script": data.get("plot_script", "Unknown"),
                    "model_name": data.get("formatted_name", data.get("model_name", "Unknown")),
                }
    except Exception as exc:
        print(f"Error reading metadata from {jsonl_file}: {exc}")

    return {
        "prompt_filename": "Unknown",
        "plot_script": "Unknown",
        "model_name": "Unknown",
    }


def calculate_run_scores(jsonl_file):
    """Calculate Galaxy Zoo accuracy for a single judged JSONL run."""
    scores = {
        "gz_accuracy": 0.0,
        "model_count": 0,
    }
    model_scores = {}

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = data.get("decision_tree_score")
                if score is None:
                    continue

                model = data.get("formatted_name") or data.get("model_name", "unknown")
                model_scores.setdefault(model, []).append(score)
    except Exception as exc:
        print(f"Error reading {jsonl_file}: {exc}")
        return scores

    all_scores = [score for values in model_scores.values() for score in values]
    scores["model_count"] = len(model_scores)
    scores["gz_accuracy"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return scores


def _score_intensity(value, max_value):
    return value / max_value if max_value > 0 else 0.0


def generate_html_leaderboard(runs_data):
    """Generate a sortable HTML leaderboard with Galaxy Zoo accuracy only."""
    max_gz = max((run["gz_accuracy"] for run in runs_data), default=0.0)
    table_rows = []

    for run in runs_data:
        timestamp = run["timestamp"]
        prompt_filename = f'<span class="filename">{run["prompt_filename"]}</span>'
        plot_script = f'<span class="filename">{run["plot_script"]}</span>'
        model_name = f'<span class="model-name">{run["model_name"]}</span>'
        gz_accuracy = f'{run["gz_accuracy"]:.3f}'
        gz_intensity = _score_intensity(run["gz_accuracy"], max_gz)

        table_rows.append(
            f"""
                <tr>
                    <td><span class="timestamp">{timestamp}</span></td>
                    <td>{prompt_filename}</td>
                    <td>{plot_script}</td>
                    <td>{model_name}</td>
                    <td class="score" data-intensity="{gz_intensity:.3f}">{gz_accuracy}</td>
                    <td>{run["model_count"]}</td>
                </tr>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Galaxy Zoo Benchmark - Runs Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background-color: white;
            padding: 28px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #243447;
            text-align: center;
            margin-bottom: 24px;
            font-size: 2.1em;
        }}
        .summary {{
            background-color: #eef3f7;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            font-size: 13px;
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
            user-select: none;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .score {{
            font-weight: bold;
            padding: 6px 8px;
            border-radius: 4px;
        }}
        .filename {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.85em;
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .model-name, .timestamp {{
            font-size: 0.9em;
            color: #637282;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Galaxy Zoo Benchmark - Runs Summary</h1>
        <div class="summary"><strong>{len(runs_data)}</strong> evaluation runs analyzed</div>
        <table id="leaderboard">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Timestamp</th>
                    <th onclick="sortTable(1)">Prompt Filename</th>
                    <th onclick="sortTable(2)">Plotting Script</th>
                    <th onclick="sortTable(3)">Model</th>
                    <th onclick="sortTable(4)">GZ Accuracy</th>
                    <th onclick="sortTable(5)">Model Count</th>
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

            rows.sort((a, b) => {{
                const aValue = a.cells[columnIndex].textContent.trim();
                const bValue = b.cells[columnIndex].textContent.trim();
                let result;
                if (columnIndex >= 4) {{
                    result = (parseFloat(aValue) || 0) - (parseFloat(bValue) || 0);
                }} else {{
                    result = aValue.localeCompare(bValue);
                }}
                return currentSort.direction === 'asc' ? result : -result;
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}

        function colorForIntensity(intensity) {{
            const red = [192, 57, 43];
            const green = [39, 174, 96];
            const r = Math.round(red[0] + (green[0] - red[0]) * intensity);
            const g = Math.round(red[1] + (green[1] - red[1]) * intensity);
            const b = Math.round(red[2] + (green[2] - red[2]) * intensity);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('.score').forEach(cell => {{
                const intensity = parseFloat(cell.getAttribute('data-intensity')) || 0;
                cell.style.backgroundColor = colorForIntensity(intensity);
                cell.style.color = intensity > 0.5 ? 'white' : 'black';
            }});
        }});
    </script>
</body>
</html>"""


def main():
    print("Galaxy Zoo Benchmark - Runs Summary Generator")
    print("=" * 50)

    jsonl_files = glob("eval/runs/jsonl/*.jsonl")
    if not jsonl_files:
        print("No JSONL files found in eval/runs/jsonl/")
        return

    runs_data = []
    for jsonl_file in sorted(jsonl_files):
        filename = Path(jsonl_file).name
        timestamp = filename.replace("judged_all_evals_", "").replace(".jsonl", "")
        metadata = get_run_metadata(jsonl_file)
        scores = calculate_run_scores(jsonl_file)
        run_data = {
            "timestamp": timestamp,
            "filename": filename,
            **metadata,
            **scores,
        }
        runs_data.append(run_data)
        print(f"  {filename}: GZ={scores['gz_accuracy']:.3f}")

    runs_data.sort(key=lambda x: x["timestamp"], reverse=True)
    html_content = generate_html_leaderboard(runs_data)

    html_output_file = "eval/runs_summary.html"
    with open(html_output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    jsonl_output_file = "eval/runs_summary.jsonl"
    with open(jsonl_output_file, "w", encoding="utf-8") as f:
        for run_data in runs_data:
            summary_record = {
                "timestamp": run_data["timestamp"],
                "filename": run_data["filename"],
                "prompt_filename": run_data["prompt_filename"],
                "plot_script": run_data["plot_script"],
                "model_name": run_data["model_name"],
                "model_count": run_data["model_count"],
                "gz_accuracy": run_data["gz_accuracy"],
            }
            f.write(json.dumps(summary_record) + "\n")

    print("Summary generated successfully.")
    print(f"HTML saved to: {Path(html_output_file).absolute()}")
    print(f"JSONL saved to: {Path(jsonl_output_file).absolute()}")


if __name__ == "__main__":
    main()
