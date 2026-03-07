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
import math
from typing import Union, List


def _normalize(text: str) -> str:
    """Normalize a string for EM comparison."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: Union[str, List]) -> bool:
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


def f1_score(prediction: str, ground_truth: Union[str, List]) -> float:
    """
    Compute word-level F1 score between prediction and ground truth.
    If multiple ground truths, return the max F1.
    """
    if not prediction:
        return 0.0

    def _get_f1(pred: str, gt: str) -> float:
        pred_tokens = _normalize(pred).split()
        gt_tokens = _normalize(gt).split()
        
        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0
            
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if isinstance(ground_truth, list):
        return max(_get_f1(prediction, gt) for gt in ground_truth)
    return _get_f1(prediction, str(ground_truth))


def pass_at_k(predictions: List[str], ground_truth: Union[str, List]) -> bool:
    """
    Oracle Exact Match: True if any prediction in the list matches.

    Args:
        predictions: List of candidate answers (one per thread/pass).
        ground_truth: Ground truth string or list.

    Returns:
        True if at least one prediction is an exact match.
    """
    return any(exact_match(p, ground_truth) for p in predictions)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Standard estimator for pass@k.
    n: total samples
    c: number of correct samples
    k: k for pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_scores(
    results: List[dict],
    pred_key: str = "final_answer",
    gt_key: str = "answer_gt",
) -> dict:
    """
    Compute aggregate EM, F1, and Pass@k statistics.
    If multiple samples are provided per question, 'final_answer' should be a list.
    """
    num_total = len(results)
    if num_total == 0:
        return {"em": 0, "f1": 0, "num_correct": 0, "num_total": 0}

    total_em = 0.0
    total_f1 = 0.0
    
    # For pass@k
    pass_k_values = [1, 2, 4, 8, 16]
    total_pass_k = {k: 0.0 for k in pass_k_values}
    has_multiple_samples = False

    for r in results:
        pred = r.get(pred_key, "")
        gt = r.get(gt_key, "")
        
        if isinstance(pred, list):
            has_multiple_samples = True
            n = len(pred)
            c = sum(1 for p in pred if exact_match(p, gt))
            
            # Pass@1 is same as average EM
            total_em += c / n
            total_f1 += sum(f1_score(p, gt) for p in pred) / n
            
            for k in pass_k_values:
                if k <= n:
                    total_pass_k[k] += estimate_pass_at_k(n, c, k)
        else:
            if exact_match(pred, gt):
                total_em += 1
            total_f1 += f1_score(pred, gt)

    scores = {
        "em": total_em / num_total,
        "f1": total_f1 / num_total,
        "num_correct": total_em,  # Total equivalent correct questions
        "num_total": num_total,
    }
    
    if has_multiple_samples:
        for k, val in total_pass_k.items():
            scores[f"pass@{k}"] = val / num_total
            
    return scores
