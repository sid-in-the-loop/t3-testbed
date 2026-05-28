"""
Prepare datasets for main table experiments.

Sources:
  1. evaluation_data/test_small_deepresearcher.json → hotpotqa, musique, 2wikimultihopqa, bamboogle
  2. evaluation_data/webwalker.json → webwalker
  3. evaluation_data/hle.json → hle
  4. evaluation_data/frames.tsv → frames
  5. data/GAIA.json → already done

Output: general_agent/data/main_table/{dataset}.json
Schema: [{"id": "{dataset}-{idx}", "question": "...", "answer": "..."}]

Usage:
  cd general_agent
  python -m webwalkerqa.scripts.prepare_datasets
"""

import csv
import json
import sys
from pathlib import Path

_GA_DIR = Path(__file__).resolve().parent.parent.parent
_TESTBED = _GA_DIR.parent
_EVAL_DIR = _TESTBED / "evaluation_data"
_OUT_DIR = _GA_DIR / "data" / "main_table"

# Map data_source values to ID prefixes (must match get_prompt_for_question routing)
SOURCE_TO_PREFIX = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wiki": "2wikimultihopqa",
    "Bamboogle": "bamboogle",
}

# Only include multi-hop datasets
INCLUDE_SOURCES = set(SOURCE_TO_PREFIX.keys())


def _first_answer(answer_str: str) -> str:
    """Extract the first (canonical) answer from <|answer_split|> delimited string."""
    if "<|answer_split|>" in str(answer_str):
        return str(answer_str).split("<|answer_split|>")[0].strip()
    return str(answer_str).strip()


def prepare_deepresearcher():
    """Split test_small_deepresearcher.json into per-dataset JSON files."""
    src = _EVAL_DIR / "test_small_deepresearcher.json"
    print(f"Reading {src}")
    with open(src) as f:
        data = json.load(f)

    buckets: dict[str, list] = {}
    for item in data:
        ds = item["data_source"]
        if ds not in INCLUDE_SOURCES:
            continue
        prefix = SOURCE_TO_PREFIX[ds]
        if prefix not in buckets:
            buckets[prefix] = []

        # Use the index from extra_info if available for a stable ID
        extra = item.get("extra_info", "")
        if isinstance(extra, str) and "index" in extra:
            try:
                idx = eval(extra)["index"]  # e.g. 'hotpotqa_5a7c634c...'
                # Strip the dataset prefix from the index to avoid duplication
                idx = str(idx)
                for strip_prefix in [f"{ds}_", f"{prefix}_"]:
                    if idx.startswith(strip_prefix):
                        idx = idx[len(strip_prefix):]
                        break
            except Exception:
                idx = item["id"]
        else:
            idx = item["id"]

        buckets[prefix].append({
            "id": f"{prefix}-{idx}",
            "question": item["question"],
            "answer": _first_answer(item["answer"]),
        })

    for prefix, examples in buckets.items():
        out_path = _OUT_DIR / f"{prefix}.json"
        with open(out_path, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"  {prefix}: {len(examples)} questions → {out_path}")


def prepare_webwalker():
    """Prepare WebWalkerQA dataset."""
    src = _EVAL_DIR / "webwalker.json"
    print(f"Reading {src}")
    with open(src) as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append({
            "id": f"webwalker-{item['id']}",
            "question": item["question"],
            "answer": str(item["answer"]).strip(),
        })

    out_path = _OUT_DIR / "webwalker.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  webwalker: {len(examples)} questions → {out_path}")


def prepare_hle():
    """Prepare HLE dataset."""
    src = _EVAL_DIR / "hle.json"
    print(f"Reading {src}")
    with open(src) as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append({
            "id": f"hle-{item['id']}",
            "question": item["question"],
            "answer": str(item["answer"]).strip(),
        })

    out_path = _OUT_DIR / "hle.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  hle: {len(examples)} questions → {out_path}")


def prepare_frames():
    """Prepare FRAMES dataset from TSV."""
    src = _EVAL_DIR / "frames.tsv"
    print(f"Reading {src}")

    examples = []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            idx = row[0].strip()
            question = row[1].strip()
            answer = row[2].strip()
            if question and answer:
                examples.append({
                    "id": f"frames-{idx}",
                    "question": question,
                    "answer": answer,
                })

    out_path = _OUT_DIR / "frames.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  frames: {len(examples)} questions → {out_path}")


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {_OUT_DIR}\n")

    prepare_deepresearcher()
    print()
    prepare_webwalker()
    print()
    prepare_hle()
    print()
    prepare_frames()

    # Summary
    print("\n" + "=" * 60)
    print("Dataset summary:")
    for p in sorted(_OUT_DIR.glob("*.json")):
        with open(p) as f:
            n = len(json.load(f))
        print(f"  {p.name}: {n} questions")
    print("=" * 60)


if __name__ == "__main__":
    main()
