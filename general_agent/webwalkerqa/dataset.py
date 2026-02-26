"""
WebWalkerQA dataset loading.

Supports:
  1. Local JSON file: list of {"id", "question", "answer"} dicts
  2. HuggingFace Hub: datasets.load_dataset("callanwu/WebWalkerQA")
  3. Auto-download to a local cache

Format expected:
  [
    {"id": "0", "question": "...", "answer": "..."},
    ...
  ]
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Default local path for the dataset
DEFAULT_LOCAL_PATH = Path(__file__).parent.parent / "data" / "webwalkerqa_200.json"

# HuggingFace dataset identifier
HF_DATASET_ID = "callanwu/WebWalkerQA"


@dataclass
class QAExample:
    """Single WebWalkerQA example."""
    id: str
    question: str
    answer: str  # Ground-truth answer (may be a list or string)


def load_dataset(
    path: Optional[str] = None,
    split: str = "test",
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[QAExample]:
    """
    Load WebWalkerQA dataset.

    Args:
        path: Path to local JSON file. If None, tries DEFAULT_LOCAL_PATH, then HF.
        split: HuggingFace split to use ("train", "test", "validation").
        max_examples: Limit dataset size (useful for debugging).
        shuffle: Shuffle examples before slicing.
        seed: Random seed for shuffling.

    Returns:
        List of QAExample objects.
    """
    if path is not None:
        examples = _load_from_json(path)
    elif DEFAULT_LOCAL_PATH.exists():
        examples = _load_from_json(str(DEFAULT_LOCAL_PATH))
    else:
        examples = _load_from_hf(split)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(examples)

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def _load_from_json(path: str) -> list[QAExample]:
    """Load from a local JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        qid = str(item.get("id", item.get("question_id", str(i))))
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("answers", item.get("gold_answer", "")))
        # Normalise: if answer is a list, keep as-is; eval will handle it
        examples.append(QAExample(id=qid, question=question, answer=answer))

    return examples


def _load_from_hf(split: str) -> list[QAExample]:
    """Download from HuggingFace Hub."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to download WebWalkerQA from HuggingFace. "
            "Install it with: pip install datasets"
        )

    print(f"[dataset] Downloading {HF_DATASET_ID} ({split}) from HuggingFace...")
    ds = hf_load(HF_DATASET_ID, split=split, trust_remote_code=True)

    examples = []
    for i, item in enumerate(ds):
        qid = str(item.get("id", item.get("question_id", str(i))))
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("answers", item.get("gold_answer", "")))
        examples.append(QAExample(id=qid, question=question, answer=answer))

    # Cache locally for next run
    _save_to_cache(examples)
    return examples


def _save_to_cache(examples: list[QAExample]) -> None:
    """Save downloaded examples to local cache."""
    DEFAULT_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": e.id, "question": e.question, "answer": e.answer} for e in examples]
    with open(DEFAULT_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[dataset] Cached {len(examples)} examples to {DEFAULT_LOCAL_PATH}")
