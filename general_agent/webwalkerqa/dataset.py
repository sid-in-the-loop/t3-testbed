import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DEFAULT_LOCAL_PATH = Path(__file__).parent.parent / "data" / "webwalkerqa_200.json"

HF_DATASET_ID = "callanwu/WebWalkerQA"


@dataclass
class QAExample:
    id: str
    question: str
    answer: str


def load_dataset(
    path: Optional[str] = None,
    split: str = "main",
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[QAExample]:
    """
    Load WebWalkerQA dataset.

    Args:
        path: Path to local JSON file. If None, tries HF.
        split: HuggingFace split to use ("main", "silver"). Default "main".
        max_examples: Limit dataset size (useful for debugging).
        shuffle: Shuffle examples before slicing.
        seed: Random seed for shuffling.

    Returns:
        List of QAExample objects.
    """
    if path is not None:
        if path.endswith(".jsonl"):
            examples = _load_from_jsonl(path)
        else:
            examples = _load_from_json(path)
    else:
        cache_path = DEFAULT_LOCAL_PATH.parent / f"webwalkerqa_{split}.json"
        if cache_path.exists():
            examples = _load_from_json(str(cache_path))
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
    """Load from a local JSON file (single JSON array)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        qid = str(item.get("id", item.get("question_id", str(i))))
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("answers", item.get("gold_answer", "")))
        examples.append(QAExample(id=qid, question=question, answer=answer))

    return examples


def _load_from_jsonl(path: str) -> list[QAExample]:
    """Load from a local JSONL file (one JSON object per line). Same schema as JSON: id, question, answer."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item.get("id", item.get("question_id", str(i))))
            question = item.get("question", item.get("query", ""))
            answer = item.get("answer", item.get("answers", item.get("gold_answer", "")))
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

    cache_path = DEFAULT_LOCAL_PATH.parent / f"webwalkerqa_{split}.json"
    _save_to_cache(examples, cache_path)
    return examples


def _save_to_cache(examples: list[QAExample], cache_path: Path) -> None:
    """Save downloaded examples to local cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": e.id, "question": e.question, "answer": e.answer} for e in examples]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[dataset] Cached {len(examples)} examples to {cache_path}")
