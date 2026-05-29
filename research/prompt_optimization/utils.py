#!/usr/bin/env python3
"""Utilities for the Galaxy Zoo prompt-optimization evaluation pipeline."""

import json
import os
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from galaxybench.eval.combined.generate_descriptions import generate_and_judge_unified
from galaxybench.eval.combined.print_score import analyze_scores
from galaxybench.eval.combined.summarize_runs import (
    calculate_run_scores,
    generate_html_leaderboard,
    get_run_metadata,
)


def run_evaluation(
    eval_types: List[str],
    prompt: Union[str, Path] = "src/prompts/general_promptv4.txt",
    plot_script: Union[str, Path] = "prompt_optimization/plotting_scripts/default_plot.py",
    judge_model: str = "gemini-2.5-flash-preview-05-20",
    cores: int = 10,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    plot_dir: str = "prompt_optimization/plots",
    models: Optional[List[str]] = None,
    verbose: bool = True,
    precontext_parts=None,
) -> Optional[str]:
    """Run the core Galaxy Zoo evaluation pipeline."""
    prompt = str(prompt)
    plot_script = str(plot_script)
    output_dir = str(output_dir)

    if verbose:
        print("\n" + "=" * 60)
        print("GALAXY ZOO EVALUATION PIPELINE")
        print("=" * 60)
        print(f"Evaluation types: {', '.join(eval_types)}")
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
            precontext_parts=precontext_parts,
        )
        if verbose:
            print(f"\nEvaluation completed. Results: {final_file}")
        return final_file
    except Exception as exc:
        if verbose:
            print(f"\nEvaluation failed: {exc}")
            import traceback

            traceback.print_exc()
        return None


def generate_scores_analysis(
    results_file: str,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    verbose: bool = True,
) -> Optional[str]:
    """Generate a text score report for one judged JSONL file."""
    if verbose:
        print("\nGenerating scores analysis...")

    try:
        scores_output = analyze_scores(results_file, debug=False)
        filename = Path(results_file).stem
        scores_dir = Path(f"{output_dir}/scores")
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = scores_dir / f"{filename}_scores.txt"
        with open(scores_file, "w", encoding="utf-8") as f:
            f.write(scores_output)

        if verbose:
            print("Scores analysis completed.")
            print(f"Saved to: {scores_file.absolute()}")
        return str(scores_file.absolute())
    except Exception as exc:
        if verbose:
            print(f"Warning: failed to generate scores analysis: {exc}")
            import traceback

            traceback.print_exc()
        return None


def generate_runs_summary(
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    verbose: bool = True,
) -> Optional[Dict[str, str]]:
    """Generate a Galaxy Zoo-only summary across all judged JSONL runs."""
    if verbose:
        print("\nGenerating runs summary...")

    try:
        jsonl_pattern = f"{output_dir}/jsonl/*.jsonl"
        jsonl_files = glob(jsonl_pattern)
        if not jsonl_files:
            if verbose:
                print(f"No JSONL files found in {output_dir}/jsonl/")
            return None

        runs_data = []
        for jsonl_file in sorted(jsonl_files):
            filename = Path(jsonl_file).name
            timestamp = filename.replace("judged_all_evals_", "").replace(".jsonl", "")
            metadata = get_run_metadata(jsonl_file)
            scores = calculate_run_scores(jsonl_file)
            runs_data.append({
                "timestamp": timestamp,
                "filename": filename,
                **metadata,
                **scores,
            })

        runs_data.sort(key=lambda x: x["timestamp"], reverse=True)
        html_content = generate_html_leaderboard(runs_data)

        html_summary_file = Path(f"{output_dir}/runs_summary.html")
        with open(html_summary_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        jsonl_summary_file = Path(f"{output_dir}/runs_summary.jsonl")
        with open(jsonl_summary_file, "w", encoding="utf-8") as f:
            for run_data in runs_data:
                summary_record = {
                    "timestamp": run_data["timestamp"],
                    "filename": run_data["filename"],
                    "prompt_filename": run_data["prompt_filename"],
                    "plot_script": run_data["plot_script"],
                    "model_name": run_data["model_name"],
                    "model_count": run_data["model_count"],
                    "gz_accuracy": round(run_data["gz_accuracy"], 3)
                    if isinstance(run_data["gz_accuracy"], float)
                    else run_data["gz_accuracy"],
                }
                f.write(json.dumps(summary_record) + "\n")

        if verbose:
            print("Runs summary generated successfully.")
            print(f"HTML saved to: {html_summary_file.absolute()}")
            print(f"JSONL saved to: {jsonl_summary_file.absolute()}")

        return {
            "html": str(html_summary_file.absolute()),
            "jsonl": str(jsonl_summary_file.absolute()),
        }
    except Exception as exc:
        if verbose:
            print(f"Warning: failed to generate runs summary: {exc}")
            import traceback

            traceback.print_exc()
        return None


def get_all_scores_from_results(results_file: str) -> Optional[Dict[str, Any]]:
    """Extract Galaxy Zoo scores from a completed evaluation run."""
    try:
        scores = calculate_run_scores(results_file)
        return {
            "gz_accuracy": scores.get("gz_accuracy", 0.0),
            "model_count": scores.get("model_count", 0),
        }
    except Exception as exc:
        print(f"Failed to extract scores: {exc}")
        return None


def analyze_worst_galaxyzoo_questions(
    results_file: str,
    top_n: int = 5,
    verbose: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Analyze the worst-performing Galaxy Zoo questions from a judged JSONL run."""
    try:
        records = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if "decision_tree" in record and "decision_tree_score" in record:
                        records.append(record)

        if not records:
            if verbose:
                print("No Galaxy Zoo records found in the file.")
            return None

        question_scores = defaultdict(list)
        for record in records:
            if record.get("decision_tree_score") is None:
                continue

            judge_nodes = set(record.get("judge_results", {}).get("judge_path", []))
            for node_info in record["decision_tree"]:
                if "node" not in node_info:
                    continue
                node = node_info["node"]
                question = node_info.get("question", node.split("_")[0])
                question_scores[question].append(1.0 if node in judge_nodes else 0.0)

        question_stats = []
        for question, scores in question_scores.items():
            if scores:
                question_stats.append({
                    "question": question,
                    "average_score": statistics.mean(scores),
                    "total_instances": len(scores),
                    "correct_instances": sum(scores),
                })

        question_stats.sort(key=lambda x: x["average_score"])
        worst_questions = question_stats[:top_n]

        if verbose:
            print(f"\nWorst-performing Galaxy Zoo questions (top {top_n}):")
            print("=" * 60)
            for i, q_stat in enumerate(worst_questions, 1):
                correct = int(q_stat["correct_instances"])
                total = q_stat["total_instances"]
                avg_score = q_stat["average_score"]
                print(f"{i}. {q_stat['question']}")
                print(f"   Score: {avg_score:.3f} ({correct}/{total} correct)")

        return worst_questions
    except Exception as exc:
        if verbose:
            print(f"Failed to analyze worst questions: {exc}")
        return None


def run_full_evaluation_pipeline(
    eval_types: List[str] = ["galaxyzoo"],
    prompt: Union[str, Path] = "src/prompts/general_promptv4.txt",
    plot_script: Union[str, Path] = "prompt_optimization/plotting_scripts/default_plot.py",
    judge_model: str = "gemini-2.5-flash-preview-05-20",
    cores: int = 10,
    output_dir: Union[str, Path] = "prompt_optimization/runs",
    plot_dir: str = "prompt_optimization/plots",
    models: Optional[List[str]] = None,
    verbose: bool = True,
    precontext_parts=None,
) -> Optional[Dict[str, Any]]:
    """Run the complete Galaxy Zoo prompt-evaluation pipeline."""
    os.makedirs(f"{output_dir}/jsonl", exist_ok=True)
    os.makedirs(f"{output_dir}/scores", exist_ok=True)

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
        precontext_parts=precontext_parts,
    )

    if not results_file:
        return None

    scores_file = generate_scores_analysis(
        results_file=results_file,
        output_dir=output_dir,
        verbose=verbose,
    )
    summary_files = generate_runs_summary(output_dir=output_dir, verbose=verbose)
    all_scores = get_all_scores_from_results(results_file)

    worst_questions = None
    if "galaxyzoo" in eval_types:
        worst_questions = analyze_worst_galaxyzoo_questions(
            results_file=results_file,
            top_n=5,
            verbose=verbose,
        )

    results = {
        "results_file": results_file,
        "scores_file": scores_file,
        "summary_files": summary_files,
        "worst_questions": worst_questions,
    }

    if all_scores:
        results.update(all_scores)

    return results
