"""
Exact Match evaluation for WebWalkerQA.

Normalization:
  - Lowercase
  - Strip leading/trailing whitespace
  - Remove articles (a, an, the)
  - Collapse multiple whitespace to single space
  - Strip punctuation at boundaries

Pass@k (oracle):
  Given k answers, EM=1 if any one matches ground truth.
"""

import re
import string
from typing import Union


def _normalize(text: str) -> str:
    """Normalize a string for EM comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: Union[str, list]) -> bool:
    """
    Compute Exact Match between prediction and ground truth.

    Args:
        prediction: Model's answer string.
        ground_truth: Ground truth string or list of acceptable answers.

    Returns:
        True if normalized prediction matches any normalized ground truth.
    """
    if not prediction:
        return False

    pred_norm = _normalize(prediction)

    if isinstance(ground_truth, list):
        return any(_normalize(gt) == pred_norm for gt in ground_truth)
    return _normalize(str(ground_truth)) == pred_norm


def pass_at_k(predictions: list[str], ground_truth: Union[str, list]) -> bool:
    """
    Oracle Exact Match: True if any prediction in the list matches.

    Args:
        predictions: List of candidate answers (one per thread/pass).
        ground_truth: Ground truth string or list.

    Returns:
        True if at least one prediction is an exact match.
    """
    return any(exact_match(p, ground_truth) for p in predictions)


def compute_em_score(
    results: list[dict],
    pred_key: str = "final_answer",
    gt_key: str = "answer",
) -> dict:
    """
    Compute aggregate EM statistics over a list of result dicts.

    Args:
        results: List of per-question result dicts.
        pred_key: Key for the model's predicted answer.
        gt_key: Key for the ground truth answer.

    Returns:
        Dict with keys: em, num_correct, num_total.
    """
    num_correct = 0
    num_total = len(results)

    for r in results:
        pred = r.get(pred_key, "") or ""
        gt = r.get(gt_key, "")
        if exact_match(pred, gt):
            num_correct += 1

    em = num_correct / num_total if num_total > 0 else 0.0
    return {"em": em, "num_correct": num_correct, "num_total": num_total}
