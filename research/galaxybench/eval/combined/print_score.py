#!/usr/bin/env python3
"""Calculate Galaxy Zoo decision-tree scores from judged JSONL outputs."""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path


def calculate_galaxyzoo_question_scores(judge_results):
    """Calculate per-question and per-answer scores for Galaxy Zoo evaluation."""
    judge_path = judge_results.get("judge_path", [])
    volunteer_path = judge_results.get("volunteer_path", [])

    if not volunteer_path:
        return {}, {}, {}

    question_scores = {}
    judge_set = set(judge_path)
    question_types = defaultdict(lambda: {"judge": [], "volunteer": []})

    for step in judge_path:
        question = step.split("_")[0]
        question_types[question]["judge"].append(step)

    for step in volunteer_path:
        question = step.split("_")[0]
        question_types[question]["volunteer"].append(step)

    for question_type, paths in question_types.items():
        j_steps = set(paths["judge"])
        v_steps = set(paths["volunteer"])
        if v_steps:
            question_scores[question_type] = len(j_steps.intersection(v_steps)) / len(v_steps)

    answer_scores = {}
    for v_step in volunteer_path:
        if "_" not in v_step:
            continue
        question_type, answer = v_step.split("_", 1)
        answer_key = f"{question_type}: {answer}"
        answer_scores.setdefault(answer_key, []).append(1.0 if v_step in judge_set else 0.0)

    error_analysis = {}
    for question_type, paths in question_types.items():
        v_steps = paths["volunteer"]
        j_steps = paths["judge"]
        if not v_steps:
            continue

        v_answers = [step.split("_", 1)[1] for step in v_steps if "_" in step]
        j_answers = [step.split("_", 1)[1] for step in j_steps if "_" in step]

        error_stats = {
            "total_wrong": 0,
            "wrong_type": 0,
            "judge_not_mentioned_wrong": 0,
            "judge_specific_wrong": 0,
        }

        for v_answer in v_answers:
            v_step = f"{question_type}_{v_answer}"
            if v_step in judge_set:
                continue

            error_stats["total_wrong"] += 1
            judge_answer_for_question = next(
                (j_answer for j_answer in j_answers if j_answer != "not-mentioned"),
                None,
            )

            if judge_answer_for_question is None:
                if v_answer != "not-mentioned":
                    error_stats["judge_not_mentioned_wrong"] += 1
            elif v_answer == "not-mentioned":
                error_stats["judge_specific_wrong"] += 1
            else:
                error_stats["wrong_type"] += 1

        if error_stats["total_wrong"] > 0:
            error_analysis[question_type] = error_stats

    return question_scores, answer_scores, error_analysis


def analyze_scores(jsonl_file, debug=False):
    """Analyze Galaxy Zoo scores from a judged JSONL file and return a text report."""
    jsonl_file = Path(jsonl_file)
    if not jsonl_file.exists():
        return f"Error: File {jsonl_file} not found!"

    scores = defaultdict(list)
    question_scores = defaultdict(lambda: defaultdict(list))
    answer_scores = defaultdict(lambda: defaultdict(list))
    error_analysis = defaultdict(lambda: defaultdict(lambda: {
        "total_wrong": 0,
        "wrong_type": 0,
        "judge_not_mentioned_wrong": 0,
        "judge_specific_wrong": 0,
    }))

    debug_stats = {
        "total_lines": 0,
        "valid_json": 0,
        "models_found": set(),
        "score_fields_found": set(),
    }

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            debug_stats["total_lines"] += 1
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                if debug:
                    print(f"JSON decode error on line {debug_stats['total_lines']}: {exc}")
                continue

            debug_stats["valid_json"] += 1
            model = data.get("formatted_name") or data.get("model_name", "unknown")
            debug_stats["models_found"].add(model)

            score = data.get("decision_tree_score")
            judge_results = data.get("judge_results", {})
            if score is None:
                continue

            debug_stats["score_fields_found"].add("decision_tree_score")
            scores[model].append(score)

            if judge_results:
                q_scores, a_scores, e_analysis = calculate_galaxyzoo_question_scores(judge_results)
                for q_type, q_score in q_scores.items():
                    question_scores[model][q_type].append(q_score)
                for answer_key, answer_score_list in a_scores.items():
                    answer_scores[model][answer_key].extend(answer_score_list)
                for q_type, e_stats in e_analysis.items():
                    for stat_key, stat_value in e_stats.items():
                        error_analysis[model][q_type][stat_key] += stat_value

    output = io.StringIO()

    if debug:
        output.write("Debug Information:\n")
        output.write(f"Total lines: {debug_stats['total_lines']}\n")
        output.write(f"Valid JSON lines: {debug_stats['valid_json']}\n")
        output.write(f"Models found: {sorted(debug_stats['models_found'])}\n")
        output.write(f"Score fields found: {sorted(debug_stats['score_fields_found'])}\n\n")

    output.write("Mean Galaxy Zoo Decision-Tree Scores per Model\n")
    output.write("=" * 50 + "\n")

    for model in sorted(scores.keys()):
        score_list = scores[model]
        mean_score = sum(score_list) / len(score_list)
        output.write(f"\nModel: {model}\n")
        output.write("-" * (len(model) + 8) + "\n")
        output.write(f"  galaxyzoo: {mean_score:.4f} (n={len(score_list)})\n")

        for question_type in sorted(question_scores[model].keys()):
            q_scores = question_scores[model][question_type]
            if not q_scores:
                continue
            q_mean = sum(q_scores) / len(q_scores)
            output.write(f"    - {question_type}: {q_mean:.4f} (n={len(q_scores)})\n")

            e_stats = error_analysis[model][question_type]
            if e_stats["total_wrong"] > 0:
                total_wrong = e_stats["total_wrong"]
                output.write(f"      Error breakdown (n={total_wrong} wrong):\n")
                if e_stats["wrong_type"] > 0:
                    pct = e_stats["wrong_type"] / total_wrong * 100
                    output.write(f"        Wrong type: {pct:.1f}% ({e_stats['wrong_type']})\n")
                if e_stats["judge_not_mentioned_wrong"] > 0:
                    pct = e_stats["judge_not_mentioned_wrong"] / total_wrong * 100
                    output.write(
                        f"        Judge said 'not-mentioned': {pct:.1f}% "
                        f"({e_stats['judge_not_mentioned_wrong']})\n"
                    )
                if e_stats["judge_specific_wrong"] > 0:
                    pct = e_stats["judge_specific_wrong"] / total_wrong * 100
                    output.write(
                        f"        Judge over-specific: {pct:.1f}% "
                        f"({e_stats['judge_specific_wrong']})\n"
                    )

        if answer_scores[model]:
            output.write("    Per-answer breakdown:\n")
            for answer_key in sorted(answer_scores[model].keys()):
                a_scores = answer_scores[model][answer_key]
                if a_scores:
                    a_mean = sum(a_scores) / len(a_scores)
                    output.write(f"      - {answer_key}: {a_mean:.4f} (n={len(a_scores)})\n")

    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Galaxy Zoo decision-tree scores from a judged JSONL file."
    )
    parser.add_argument("jsonl_file", type=str, help="Path to the judged JSONL file.")
    parser.add_argument("--debug", action="store_true", help="Show debug information.")
    args = parser.parse_args()

    print(analyze_scores(args.jsonl_file, debug=args.debug))


if __name__ == "__main__":
    main()
