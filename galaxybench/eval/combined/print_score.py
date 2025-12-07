#!/usr/bin/env python3
"""
Simple script to calculate and print mean scores per model per evaluation.
Automatically detects all score fields ending with '_score'.
Includes per-question breakdown when possible.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import io

def get_judge_score_fields():
    """Get score field names from available judges."""
    try:
        # Try to import judges
        judges_path = Path(__file__).parent / "judges"
        if judges_path.exists():
            sys.path.append(str(judges_path.parent))
            
            from judges.galaxyzoo import GalaxyZooJudge
            from judges.tidal import TidalJudge
            from judges.lens import LensJudge
            
            # Create judge instances with dummy model (just to get score field names)
            judges = [
                GalaxyZooJudge(model="dummy"), 
                TidalJudge(model="dummy"), 
                LensJudge(model="dummy")
            ]
            return {judge.get_score_field_name(): judge.__class__.__name__.replace('Judge', '').lower() 
                   for judge in judges}
    except Exception as e:
        print(f"Warning: Could not import judges ({e}), using fallback detection")
        pass
    
    return {}

def calculate_galaxyzoo_question_scores(judge_results):
    """Calculate per-question scores for GalaxyZoo evaluation."""
    judge_path = judge_results.get('judge_path', [])
    volunteer_path = judge_results.get('volunteer_path', [])
    
    if not volunteer_path:
        return {}, {}
    
    # Calculate overall question scores (as before)
    question_scores = {}
    judge_set = set(judge_path)
    volunteer_set = set(volunteer_path)
    
    # Group by question type
    question_types = defaultdict(lambda: {'judge': [], 'volunteer': []})
    
    for step in judge_path:
        question = step.split('_')[0]
        question_types[question]['judge'].append(step)
    
    for step in volunteer_path:
        question = step.split('_')[0]
        question_types[question]['volunteer'].append(step)
    
    # Calculate score for each question
    for question_type in question_types:
        j_steps = set(question_types[question_type]['judge'])
        v_steps = set(question_types[question_type]['volunteer'])
        if v_steps:  # Only calculate if volunteer has answers for this question
            matches = len(j_steps.intersection(v_steps))
            total = len(v_steps)
            question_scores[question_type] = matches / total if total > 0 else 0
    
    # Calculate per-answer scores with detailed error analysis
    answer_scores = {}
    
    # For each step in volunteer path, check if judge got it right
    for v_step in volunteer_path:
        if '_' in v_step:
            question_type, answer = v_step.split('_', 1)
            answer_key = f"{question_type}: {answer}"
            
            # Check if judge got this specific answer correct
            is_correct = 1.0 if v_step in judge_set else 0.0
            
            if answer_key not in answer_scores:
                answer_scores[answer_key] = []
            answer_scores[answer_key].append(is_correct)
    
    # NEW: Detailed error analysis for wrong answers
    error_analysis = {}
    
    # For each question type, analyze the errors
    for question_type in question_types:
        v_steps = question_types[question_type]['volunteer']
        j_steps = question_types[question_type]['judge']
        
        if not v_steps:
            continue
            
        # Find the specific answers for this question from both judge and volunteer
        v_answers = []
        j_answers = []
        
        for step in v_steps:
            if '_' in step:
                _, answer = step.split('_', 1)
                v_answers.append(answer)
        
        for step in j_steps:
            if '_' in step:
                _, answer = step.split('_', 1)
                j_answers.append(answer)
        
        # Analyze each volunteer answer
        error_stats = {
            'total_wrong': 0,
            'wrong_type': 0,  # Judge gave specific answer but wrong type
            'judge_not_mentioned_wrong': 0,  # Judge said not-mentioned but volunteer had specific answer
            'judge_specific_wrong': 0,  # Judge gave specific answer but volunteer said not-mentioned/different
        }
        
        for v_answer in v_answers:
            v_step = f"{question_type}_{v_answer}"
            if v_step not in judge_set:  # This is a wrong answer
                error_stats['total_wrong'] += 1
                
                # Check what the judge said for this question
                judge_answer_for_question = None
                for j_answer in j_answers:
                    if j_answer != 'not-mentioned':
                        judge_answer_for_question = j_answer
                        break
                
                if not judge_answer_for_question:
                    # Judge said not-mentioned for this question
                    if v_answer != 'not-mentioned':
                        error_stats['judge_not_mentioned_wrong'] += 1
                else:
                    # Judge gave a specific answer
                    if v_answer == 'not-mentioned':
                        error_stats['judge_specific_wrong'] += 1
                    else:
                        error_stats['wrong_type'] += 1
        
        if error_stats['total_wrong'] > 0:
            error_analysis[question_type] = error_stats
    
    return question_scores, answer_scores, error_analysis

def calculate_tidal_question_scores(judge_results):
    """Calculate per-question scores for Tidal evaluation."""
    judge_classification = judge_results.get('judge_classification', {})
    tidal_info = judge_results.get('judge_tidal_info', {})
    
    if not tidal_info:
        return {}, {}
    
    scores = {}
    answer_scores = {}
    
    # The tidal judge now does a single classification: Shell/Stream/Other
    ground_truth_class = tidal_info.get('eval_class', 'Other')  # Shell, Stream, or Other
    judge_class = judge_classification.get('classification', 'Other')  # Shell, Stream, or Other
    
    # Single question: correct classification
    is_correct = 1.0 if judge_class == ground_truth_class else 0.0
    scores['classification'] = is_correct
    
    # Per-answer breakdown
    answer_key = f"classification: {ground_truth_class}"
    if answer_key not in answer_scores:
        answer_scores[answer_key] = []
    answer_scores[answer_key].append(is_correct)
    
    return scores, answer_scores

def calculate_lens_question_scores(data, lens_score):
    """Calculate per-question scores for Lens evaluation."""
    scores = {}
    answer_scores = {}
    
    # Extract metadata
    survey = data.get('survey', 'Unknown')
    lensgrade = data.get('lensgrade', 'Unknown')
    detected = lens_score > 0
    
    # Survey breakdown
    survey_key = f"survey: {survey}"
    if survey_key not in answer_scores:
        answer_scores[survey_key] = []
    answer_scores[survey_key].append(1.0 if detected else 0.0)
    
    # Lens grade breakdown
    if lensgrade in ['A', 'B', 'C']:
        grade_key = f"grade: {lensgrade}"
        if grade_key not in answer_scores:
            answer_scores[grade_key] = []
        answer_scores[grade_key].append(1.0 if detected else 0.0)
    
    # Overall detection
    scores['detection'] = 1.0 if detected else 0.0
    
    return scores, answer_scores

def calculate_classification_metrics(tp, tn, fp, fn):
    """Calculate classification metrics from confusion matrix values."""
    metrics = {}
    
    # Basic counts
    total = tp + tn + fp + fn
    actual_positives = tp + fn
    actual_negatives = tn + fp
    predicted_positives = tp + fp
    predicted_negatives = tn + fn
    
    # Handle division by zero
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0
    
    # Core metrics
    metrics['accuracy'] = safe_divide(tp + tn, total)
    metrics['sensitivity'] = safe_divide(tp, actual_positives)  # True Positive Rate (TPR), Recall
    metrics['specificity'] = safe_divide(tn, actual_negatives)  # True Negative Rate (TNR)
    metrics['precision'] = safe_divide(tp, predicted_positives)  # Positive Predictive Value (PPV)
    metrics['npv'] = safe_divide(tn, predicted_negatives)  # Negative Predictive Value
    
    # Derived metrics
    metrics['fpr'] = safe_divide(fp, actual_negatives)  # False Positive Rate = 1 - Specificity
    metrics['fnr'] = safe_divide(fn, actual_positives)  # False Negative Rate = 1 - Sensitivity
    metrics['fdr'] = safe_divide(fp, predicted_positives)  # False Discovery Rate = 1 - Precision
    metrics['for'] = safe_divide(fn, predicted_negatives)  # False Omission Rate = 1 - NPV
    
    # F1 Score
    metrics['f1'] = safe_divide(2 * metrics['precision'] * metrics['sensitivity'], 
                               metrics['precision'] + metrics['sensitivity'])
    
    # Balanced accuracy (average of sensitivity and specificity)
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    
    # Matthews Correlation Coefficient
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    metrics['mcc'] = safe_divide(mcc_numerator, mcc_denominator)
    
    return metrics

def format_classification_report(stats, prefix=""):
    """Format classification metrics for display."""
    tp = stats.get(f'tp{prefix}', 0)
    tn = stats.get(f'tn{prefix}', 0) 
    fp = stats.get(f'fp{prefix}', 0)
    fn = stats.get(f'fn{prefix}', 0)
    
    if tp + tn + fp + fn == 0:
        return []
    
    lines = []
    lines.append(f"      └─ Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Calculate classification metrics including F1 score
    metrics = calculate_classification_metrics(tp, tn, fp, fn)
    lines.append(f"      └─ Metrics: Precision={metrics['precision']:.3f}, Recall={metrics['sensitivity']:.3f}, F1={metrics['f1']:.3f}")
    
    return lines

def format_tidal_confusion_matrix(confusion_matrix, total):
    """Format and display multi-class confusion matrix for tidal classification."""
    if total == 0:
        return []
    
    classes = ['Shell', 'Stream', 'Other']
    lines = []
    
    lines.append(f"    ═══ Tidal Confusion Matrix ═══")
    
    # Header row
    header = "          "
    for pred_class in classes:
        header += f"{pred_class:>8}"
    lines.append(header)
    
    # Matrix rows
    for true_class in classes:
        row = f"    {true_class:>6} "
        for pred_class in classes:
            count = confusion_matrix.get(true_class, {}).get(pred_class, 0)
            row += f"{count:>8}"
        lines.append(row)
    
    # Summary stats
    lines.append(f"    Total samples: {total}")
    
    # Calculate per-class metrics including F1 scores
    lines.append(f"    ─── Per-class Performance ───")
    for class_name in classes:
        true_positives = confusion_matrix.get(class_name, {}).get(class_name, 0)
        total_true = sum(confusion_matrix.get(class_name, {}).values())
        total_predicted = sum(confusion_matrix.get(tc, {}).get(class_name, 0) for tc in classes)
        
        if total_true > 0:
            recall = true_positives / total_true
            precision = true_positives / total_predicted if total_predicted > 0 else 0
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            lines.append(f"      {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={total_true})")
    
    return lines

def analyze_scores(jsonl_file, debug=False):
    """Analyze scores from a JSONL file and return formatted report."""
    jsonl_file = Path(jsonl_file)
    
    if not jsonl_file.exists():
        return f"Error: File {jsonl_file} not found!"
    
    # Get known score fields from judges (if available)
    judge_score_fields = get_judge_score_fields()
    
    # scores[model][eval_type] = [score1, score2, ...]
    scores = defaultdict(lambda: defaultdict(list))
    # question_scores[model][eval_type][question] = [score1, score2, ...]
    question_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # answer_scores[model][eval_type][answer_key] = [score1, score2, ...]
    answer_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # error_analysis[model][eval_type] = aggregated error stats
    error_analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'total_wrong': 0, 'wrong_type': 0, 'judge_not_mentioned_wrong': 0, 'judge_specific_wrong': 0
    })))
    
    # Lens-specific aggregated stats for summary
    lens_stats = defaultdict(lambda: {
        'total': 0, 'detected': 0,
        'hsc_total': 0, 'hsc_detected': 0,
        'legacy_total': 0, 'legacy_detected': 0,
        'grade_A_total': 0, 'grade_A_detected': 0,
        'grade_B_total': 0, 'grade_B_detected': 0,
        'grade_C_total': 0, 'grade_C_detected': 0,
        # Classification metrics
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,  # True/False Positives/Negatives
        'tp_hsc': 0, 'tn_hsc': 0, 'fp_hsc': 0, 'fn_hsc': 0,
        'tp_legacy': 0, 'tn_legacy': 0, 'fp_legacy': 0, 'fn_legacy': 0,
        'tp_grade_A': 0, 'fn_grade_A': 0,
        'tp_grade_B': 0, 'fn_grade_B': 0,
        'tp_grade_C': 0, 'fn_grade_C': 0
    })
    
    # Tidal-specific confusion matrix stats
    tidal_stats = defaultdict(lambda: {
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),  # [true_class][predicted_class] = count
        'total': 0
    })
    
    # Debug: Track what we're finding
    debug_stats = {
        'total_lines': 0,
        'valid_json': 0,
        'models_found': set(),
        'score_fields_found': set(),
        'eval_types_found': set()
    }
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            debug_stats['total_lines'] += 1
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                debug_stats['valid_json'] += 1
                model = data.get("formatted_name") or data.get("model_name", "unknown")
                debug_stats['models_found'].add(model)
                judge_results = data.get("judge_results", {})
                
                # Find all score fields (anything ending with '_score')
                for key, value in data.items():
                    if key.endswith('_score') and value is not None:
                        debug_stats['score_fields_found'].add(key)
                        # Use judge name if available, otherwise extract from field name
                        eval_type = judge_score_fields.get(key, key.replace('_score', ''))
                        debug_stats['eval_types_found'].add(eval_type)
                        scores[model][eval_type].append(value)
                        
                        # Calculate per-question scores if possible
                        if eval_type == 'galaxyzoo' and judge_results:
                            q_scores, a_scores, e_analysis = calculate_galaxyzoo_question_scores(judge_results)
                            for q_type, q_score in q_scores.items():
                                question_scores[model][eval_type][q_type].append(q_score)
                            for a_key, a_score_list in a_scores.items():
                                answer_scores[model][eval_type][a_key].extend(a_score_list)
                            # Aggregate error analysis
                            for q_type, e_stats in e_analysis.items():
                                for stat_key, stat_value in e_stats.items():
                                    error_analysis[model][eval_type][q_type][stat_key] += stat_value
                        
                        elif eval_type == 'tidal' and judge_results:
                            q_scores, a_scores = calculate_tidal_question_scores(judge_results)
                            for q_type, q_score in q_scores.items():
                                question_scores[model][eval_type][q_type].append(q_score)
                            for a_key, a_score_list in a_scores.items():
                                answer_scores[model][eval_type][a_key].extend(a_score_list)
                            
                            # Collect tidal confusion matrix data
                            judge_classification = judge_results.get('judge_classification', {})
                            tidal_info = judge_results.get('judge_tidal_info', {})
                            if tidal_info:
                                true_class = tidal_info.get('eval_class', 'Other')
                                predicted_class = judge_classification.get('classification', 'Other')
                                tidal_stats[model]['confusion_matrix'][true_class][predicted_class] += 1
                                tidal_stats[model]['total'] += 1
                        
                        elif key == 'description_says_lens_occuring_score':
                            eval_type = 'description_says_lens_occuring'
                            q_scores, a_scores = calculate_lens_question_scores(data, value)
                            for q_type, q_score in q_scores.items():
                                question_scores[model][eval_type][q_type].append(q_score)
                            for a_key, a_score_list in a_scores.items():
                                answer_scores[model][eval_type][a_key].extend(a_score_list)
                        
                        # Update lens stats for any lens-related evaluation
                        if key == 'description_says_lens_occuring_score' or eval_type in ['lens', 'description_says_lens_occuring']:
                            # Update lens stats for summary
                            stats = lens_stats[model]
                            stats['total'] += 1
                            
                            # Get the actual correctness from judge results
                            correct_prediction = judge_results.get('correct_prediction', False)
                            description_says_lens_occuring = judge_results.get('description_says_lens_occuring', False) 
                            ground_truth_lens = judge_results.get('ground_truth_lens', False)
                            
                            # Use description_says_lens_occuring as our "detected" for detection rate
                            detected = description_says_lens_occuring
                            if detected:
                                stats['detected'] += 1
                            
                            # Determine if this is actually a lens (ground truth)
                            is_lens = data.get('is_lens', False)
                            lensgrade = data.get('lensgrade', 'Unknown')
                            galaxy_class = data.get('class', lensgrade)
                            is_actual_lens = galaxy_class in ['A', 'B', 'C'] or is_lens or ground_truth_lens
                            
                            # Classification metrics based on correct_prediction
                            if is_actual_lens and correct_prediction:
                                stats['tp'] += 1  # True Positive: correctly detected lens
                            elif is_actual_lens and not correct_prediction:
                                stats['fn'] += 1  # False Negative: missed lens
                            elif not is_actual_lens and correct_prediction:
                                stats['tn'] += 1  # True Negative: correctly rejected non-lens
                            elif not is_actual_lens and not correct_prediction:
                                stats['fp'] += 1  # False Positive: false alarm
                            
                            survey = data.get('survey', 'Unknown')
                            if survey == 'HSC':
                                stats['hsc_total'] += 1
                                if detected:
                                    stats['hsc_detected'] += 1
                                # HSC-specific classification metrics
                                if is_actual_lens and correct_prediction:
                                    stats['tp_hsc'] += 1
                                elif is_actual_lens and not correct_prediction:
                                    stats['fn_hsc'] += 1
                                elif not is_actual_lens and correct_prediction:
                                    stats['tn_hsc'] += 1
                                elif not is_actual_lens and not correct_prediction:
                                    stats['fp_hsc'] += 1
                                    
                            elif survey == 'Legacy':
                                stats['legacy_total'] += 1
                                if detected:
                                    stats['legacy_detected'] += 1
                                # Legacy-specific classification metrics
                                if is_actual_lens and correct_prediction:
                                    stats['tp_legacy'] += 1
                                elif is_actual_lens and not correct_prediction:
                                    stats['fn_legacy'] += 1
                                elif not is_actual_lens and correct_prediction:
                                    stats['tn_legacy'] += 1
                                elif not is_actual_lens and not correct_prediction:
                                    stats['fp_legacy'] += 1
                            
                            # Grade-specific metrics (only for actual lenses)
                            if galaxy_class in ['A', 'B', 'C']:
                                stats[f'grade_{galaxy_class}_total'] += 1
                                if detected:
                                    stats[f'grade_{galaxy_class}_detected'] += 1
                                if correct_prediction:
                                    stats[f'tp_grade_{galaxy_class}'] += 1
                                else:
                                    stats[f'fn_grade_{galaxy_class}'] += 1
            
            except json.JSONDecodeError as e:
                if debug:
                    print(f"JSON decode error on line {debug_stats['total_lines']}: {e}")
                continue
            except Exception as e:
                if debug:
                    print(f"Error processing line {debug_stats['total_lines']}: {e}")
                continue
    
    # Generate output
    output = io.StringIO()
    
    if debug:
        output.write(f"Debug Information:\n")
        output.write(f"Total lines: {debug_stats['total_lines']}\n")
        output.write(f"Valid JSON lines: {debug_stats['valid_json']}\n")
        output.write(f"Models found: {sorted(debug_stats['models_found'])}\n")
        output.write(f"Score fields found: {sorted(debug_stats['score_fields_found'])}\n")
        output.write(f"Evaluation types found: {sorted(debug_stats['eval_types_found'])}\n")
        output.write("\n")
    
    # Print results
    output.write("Mean Scores per Model per Evaluation\n")
    output.write("=" * 50 + "\n")
    
    for model in sorted(scores.keys()):
        output.write(f"\nModel: {model}\n")
        output.write("-" * (len(model) + 8) + "\n")
        
        for eval_type in sorted(scores[model].keys()):
            score_list = scores[model][eval_type]
            mean_score = sum(score_list) / len(score_list)
            output.write(f"  {eval_type}: {mean_score:.4f} (n={len(score_list)})\n")
            
            # Print per-question breakdown if available
            if eval_type in question_scores[model]:
                for question_type in sorted(question_scores[model][eval_type].keys()):
                    q_scores = question_scores[model][eval_type][question_type]
                    if q_scores:
                        q_mean = sum(q_scores) / len(q_scores)
                        output.write(f"    └─ {question_type}: {q_mean:.4f} (n={len(q_scores)})\n")
                        
                        # Add error analysis for GalaxyZoo
                        if eval_type == 'galaxyzoo' and question_type in error_analysis[model][eval_type]:
                            e_stats = error_analysis[model][eval_type][question_type]
                            if e_stats['total_wrong'] > 0:
                                wrong_type_pct = (e_stats['wrong_type'] / e_stats['total_wrong']) * 100
                                not_mentioned_pct = (e_stats['judge_not_mentioned_wrong'] / e_stats['total_wrong']) * 100
                                specific_wrong_pct = (e_stats['judge_specific_wrong'] / e_stats['total_wrong']) * 100
                                
                                output.write(f"      └─ Error breakdown (n={e_stats['total_wrong']} wrong):\n")
                                if e_stats['wrong_type'] > 0:
                                    output.write(f"         • Wrong type: {wrong_type_pct:.1f}% ({e_stats['wrong_type']})\n")
                                if e_stats['judge_not_mentioned_wrong'] > 0:
                                    output.write(f"         • Judge said 'not-mentioned': {not_mentioned_pct:.1f}% ({e_stats['judge_not_mentioned_wrong']})\n")
                                if e_stats['judge_specific_wrong'] > 0:
                                    output.write(f"         • Judge over-specific: {specific_wrong_pct:.1f}% ({e_stats['judge_specific_wrong']})\n")
            
            # Print per-answer breakdown
            if eval_type in answer_scores[model] and eval_type != 'description_says_lens_occuring':
                output.write(f"    Per-answer breakdown:\n")
                for answer_key in sorted(answer_scores[model][eval_type].keys()):
                    a_scores = answer_scores[model][eval_type][answer_key]
                    if a_scores:
                        a_mean = sum(a_scores) / len(a_scores)
                        output.write(f"      └─ {answer_key}: {a_mean:.4f} (n={len(a_scores)})\n")
            
            # Special detailed lens analysis
            if (eval_type == 'description_says_lens_occuring' or eval_type == 'lens') and model in lens_stats:
                stats = lens_stats[model]
                output.write(f"    Detailed Lens Analysis:\n")
                
                # Overall Classification Performance
                output.write(f"    ═══ Overall Classification Performance ═══\n")
                overall_lines = format_classification_report(stats, "")
                for line in overall_lines:
                    output.write(line + "\n")
                
                # By Survey Type
                output.write(f"    ─── Performance by Survey Type ───\n")
                if stats['hsc_total'] > 0:
                    output.write(f"      HSC Survey:\n")
                    hsc_lines = format_classification_report(stats, "_hsc")
                    for line in hsc_lines:
                        output.write(f"  {line}\n")
                
                if stats['legacy_total'] > 0:
                    output.write(f"      Legacy Survey:\n")
                    legacy_lines = format_classification_report(stats, "_legacy")
                    for line in legacy_lines:
                        output.write(f"  {line}\n")
                
                # By Lens Grade  
                output.write(f"    ─── Performance by Lens Grade ───\n")
                for grade in ['A', 'B', 'C']:
                    tp_key = f'tp_grade_{grade}'
                    fn_key = f'fn_grade_{grade}'
                    
                    if stats.get(tp_key, 0) + stats.get(fn_key, 0) > 0:
                        output.write(f"      Grade {grade}: TP={stats[tp_key]}, FN={stats[fn_key]}\n")
            
            # Special tidal confusion matrix
            if eval_type == 'tidal' and model in tidal_stats:
                stats = tidal_stats[model]
                confusion_lines = format_tidal_confusion_matrix(stats['confusion_matrix'], stats['total'])
                for line in confusion_lines:
                    output.write(line + "\n")
    
    return output.getvalue()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate and print mean scores per model per evaluation from a JSONL file.")
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file containing evaluation results.")
    parser.add_argument("--debug", action="store_true", help="Show debug information about detected evaluation types")
    args = parser.parse_args()

    result = analyze_scores(args.jsonl_file, debug=args.debug)
    print(result)

if __name__ == "__main__":
    main()