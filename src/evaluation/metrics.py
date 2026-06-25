"""Evaluation metrics for sentence completion models."""

from typing import List, Dict, Tuple
from collections import Counter

import numpy as np

from src.data.prepare import CompletionQuestion


def accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute overall accuracy."""
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(ground_truth) if ground_truth else 0.0


def per_position_accuracy(predictions: List[str], ground_truth: List[str],
                          options: List[List[str]]) -> Dict[int, float]:
    """
    Compute accuracy broken down by the position of the correct answer
    in the options list (to detect positional bias).
    """
    position_correct = Counter()
    position_total = Counter()

    for pred, truth, opts in zip(predictions, ground_truth, options):
        if truth in opts:
            pos = opts.index(truth)
            position_total[pos] += 1
            if pred == truth:
                position_correct[pos] += 1

    return {
        pos: position_correct[pos] / position_total[pos]
        for pos in sorted(position_total.keys())
        if position_total[pos] > 0
    }


def confusion_by_option(predictions: List[str], ground_truth: List[str],
                        all_options: List[str]) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix showing which options get confused."""
    matrix = {opt: Counter() for opt in all_options}
    for pred, truth in zip(predictions, ground_truth):
        if truth in matrix:
            matrix[truth][pred] += 1
    return {k: dict(v) for k, v in matrix.items()}


def evaluate_dataset(scorer, dataset: List[CompletionQuestion],
                     verbose: bool = True) -> Dict:
    """
    Evaluate a scorer on a dataset of completion questions.

    Args:
        scorer: A BaseScorer instance.
        dataset: List of CompletionQuestion objects with known answers.
        verbose: Whether to print progress.

    Returns:
        Dictionary with evaluation results.
    """
    predictions = []
    ground_truth = []
    all_options = []
    scores_list = []
    detailed_results = []

    for i, q in enumerate(dataset):
        if q.correct is None:
            continue
        pred, scores = scorer.predict(q.question, q.options, q.blank_token)
        predictions.append(pred)
        ground_truth.append(q.correct)
        all_options.append(q.options)
        scores_list.append(scores)
        detailed_results.append({
            "question": q.question,
            "options": q.options,
            "scores": [round(s, 6) for s in scores],
            "predicted": pred,
            "correct": q.correct,
            "is_correct": pred == q.correct,
        })
        if verbose and (i + 1) % 10 == 0:
            acc_so_far = accuracy(predictions, ground_truth)
            print(f"  Processed {i + 1}/{len(dataset)} | Accuracy so far: {acc_so_far:.4f}")

    overall_acc = accuracy(predictions, ground_truth)
    pos_acc = per_position_accuracy(predictions, ground_truth, all_options)

    # Gather unique options for confusion matrix
    unique_options = sorted(set(opt for opts in all_options for opt in opts))
    confusion = confusion_by_option(predictions, ground_truth, unique_options)

    results = {
        "total_questions": len(predictions),
        "accuracy": overall_acc,
        "per_position_accuracy": pos_acc,
        "confusion_matrix": confusion,
        "detailed_results": detailed_results,
    }

    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Total Questions: {len(predictions)}")
        print(f"  Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
        print(f"  Per-Position Accuracy: {pos_acc}")

    return results


def bias_analysis(results: Dict) -> Dict:
    """
    Analyze positional bias in predictions.

    Checks if the model favors certain option positions regardless of
    the correct answer's position.
    """
    pos_acc = results["per_position_accuracy"]
    if not pos_acc:
        return {"has_bias": False}

    acc_values = list(pos_acc.values())
    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)
    max_acc = max(acc_values)
    min_acc = min(acc_values)

    # Simple bias detection: if std is high or max-min difference is large
    spread = max_acc - min_acc
    has_bias = spread > 0.15 or std_acc > 0.1

    return {
        "has_bias": has_bias,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "max_accuracy": max_acc,
        "min_accuracy": min_acc,
        "spread": spread,
        "per_position": pos_acc,
        "most_favored_position": max(pos_acc, key=pos_acc.get),
        "least_favored_position": min(pos_acc, key=pos_acc.get),
    }
