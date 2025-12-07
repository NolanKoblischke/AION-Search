from prompt_optimization.utils import run_full_evaluation_pipeline
import os
import argparse
import numpy as np

def main():
    """Run a gz evaluation with configurable prompt and plot script, and print the accuracy."""
    parser = argparse.ArgumentParser(description="Run a gz evaluation and print the accuracy.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Path to the prompt file."
    )
    parser.add_argument(
        "--plot-script",
        type=str,
        help="Path to the plotting script."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model IDs to run. If not specified, all uncommented models from models.jsonl will be used."
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of evaluation runs to perform for each model (default: 1)."
    )
    args = parser.parse_args()

    print("Running evaluation...")

    results = {}
    models_to_run = args.models if args.models else None
    
    for model in models_to_run:
        results[model] = []
        for run_idx in range(args.n_runs):
            print(f"Running evaluation for model: {model}, run {run_idx+1}/{args.n_runs}")
            model_results = run_full_evaluation_pipeline(
                eval_types=["galaxyzoo"],
                prompt=args.prompt,
                plot_script=args.plot_script,
                cores=10,
                models=[model]
            )
            # Store each model's results
            accuracy = model_results.get('gz_accuracy')
            results[model].append(accuracy)
            print(f"Run {run_idx+1} accuracy: {accuracy:.3f}")

    print("\nSummary Statistics:")
    for model in results:
        print(f"\nModel: {model}")
        print(f"Mean accuracy: {np.mean(results[model]):.2f}")
        print(f"Std deviation: {np.std(results[model]):.2f}")


if __name__ == "__main__":
    main()