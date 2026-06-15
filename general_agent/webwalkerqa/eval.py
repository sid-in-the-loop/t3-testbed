import re
from typing import Union


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    True if prediction matches ground truth after normalization.
    Normalizes: strip, lower-case, collapse whitespace.
    Supports multi-answer ground truth separated by <|answer_split|>.
    """
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    gt_str = str(ground_truth)
    if "<|answer_split|>" in gt_str:
        return any(exact_match(prediction, gt) for gt in gt_str.split("<|answer_split|>"))
    a = re.sub(r"\s+", " ", str(prediction).strip().lower())
    b = re.sub(r"\s+", " ", gt_str.strip().lower())
    return a == b


def f1_score(prediction: Union[str, list], ground_truth: Union[str, list]) -> float:
    """
    Token-level F1 between prediction and ground truth.
    If both are strings, tokenize on whitespace. Used by runner.py.
    """
    if isinstance(prediction, str):
        pred_tokens = set(prediction.strip().lower().split())
    else:
        pred_tokens = set(str(t).lower() for t in prediction)
    if isinstance(ground_truth, str):
        gt_tokens = set(ground_truth.strip().lower().split())
    else:
        gt_tokens = set(str(t).lower() for t in ground_truth)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)
