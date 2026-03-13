"""
GAIA-103 diversity experiment: 9 conditions, T=12, n=4 rollouts per question.

Conditions: naive-t4, jaccard-o{16,32,48,64}, dense-o{16,32,48,64}.
Output: one JSONL per condition; summary.csv with pass@1 and pass@4 (mean over 103 questions).

Usage:
    python -m webwalkerqa.scaling_v2_experiment --all
    python -m webwalkerqa.scaling_v2_experiment --condition naive-t4
    python -m webwalkerqa.scaling_v2_experiment --aggregate-only
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

_GA_DIR = Path(__file__).parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

env_path = _GA_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found.")
    sys.exit(1)

import pandas as pd
from tqdm import tqdm

from webwalkerqa.configs import get_config
from webwalkerqa.dataset import load_dataset
from webwalkerqa.eval import exact_match
from webwalkerqa.methods import get_method

DEFAULT_MODEL = "openai/gpt-4o-mini"

# 9 conditions
GAIA_CONDITIONS = [
    "naive-t4",
    "jaccard-o16",
    "jaccard-o32",
    "jaccard-o48",
    "jaccard-o64",
    "dense-o16",
    "dense-o32",
    "dense-o48",
    "dense-o64",
]

RESULTS_DIR = _GA_DIR / "results" / "gaia_103"


def _question_result_to_jsonl_line(
    question_id: str,
    question: str,
    answer_gt: str,
    rollout_answers: List[str],
) -> Dict[str, Any]:
    """Build one JSONL record: pass_at_1 = first rollout correct, pass_at_4 = any correct."""
    pass_at_1 = 1 if exact_match(rollout_answers[0], answer_gt) else 0
    pass_at_4 = 1 if any(exact_match(a, answer_gt) for a in rollout_answers) else 0
    return {
        "question_id": question_id,
        "question": question,
        "answer_gt": answer_gt,
        "rollout_answers": rollout_answers,
        "pass_at_1": pass_at_1,
        "pass_at_4": pass_at_4,
    }


async def run_condition(
    condition_id: str,
    model: str,
    dataset: list,
    max_concurrent: int,
    output_dir: Path,
) -> None:
    """Run all questions for one condition; append one JSONL line per question."""
    config = get_config(condition_id)
    print(f"\n{'='*60}")
    print(f"Condition: {config.id}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")

    jsonl_path = output_dir / f"{condition_id}.jsonl"
    completed_qids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed_qids.add(rec["question_id"])
        print(f"Resuming: {len(completed_qids)} questions already in {jsonl_path.name}")

    semaphore = asyncio.Semaphore(max_concurrent)
    method_cls = get_method(config.method)

    async def run_one_question(example):
        if str(example.id) in completed_qids:
            return None
        async with semaphore:
            method = method_cls(model=model, config=config, verbose=False)
            result = await method.run_question(
                question_id=str(example.id),
                question=example.question,
                answer_gt=str(example.answer),
            )
            return result

    tasks = [run_one_question(ex) for ex in dataset]
    n_questions = len(dataset)
    pbar = tqdm(
        asyncio.as_completed(tasks),
        total=n_questions,
        desc=f"{config.id} ({n_questions}q × 4 rollouts/q)",
        mininterval=10.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    append_f = open(jsonl_path, "a")
    new_lines = 0

    try:
        for coro in pbar:
            result = await coro
            if result is None:
                continue
            rollout_results = result.metadata.get("rollout_results", [])
            rollout_answers = [r.get("answer", "") for r in rollout_results]
            while len(rollout_answers) < 4:
                rollout_answers.append("")
            rollout_answers = rollout_answers[:4]

            line_dict = _question_result_to_jsonl_line(
                question_id=result.question_id,
                question=result.question,
                answer_gt=result.answer_gt,
                rollout_answers=rollout_answers,
            )
            append_f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")
            append_f.flush()
            new_lines += 1
    finally:
        append_f.close()

    total = len(completed_qids) + new_lines
    print(f"Wrote results to {jsonl_path} ({total} questions total, {new_lines} new)")


def write_summary_csv(output_dir: Path) -> None:
    """Aggregate all condition JSONL files into summary.csv (pass@1, pass@4 mean over questions)."""
    rows = []
    for cond_id in GAIA_CONDITIONS:
        jsonl_path = output_dir / f"{cond_id}.jsonl"
        if not jsonl_path.exists():
            continue
        pass_1_sum = 0.0
        pass_4_sum = 0.0
        n = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pass_1_sum += rec.get("pass_at_1", 0)
                pass_4_sum += rec.get("pass_at_4", 0)
                n += 1
        if n == 0:
            continue
        rows.append({
            "condition": cond_id,
            "pass@1": pass_1_sum / n,
            "pass@4": pass_4_sum / n,
            "n_questions": n,
        })

    if not rows:
        print("No JSONL files found; skipping summary.csv")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    print(df.to_string(index=False))


async def main():
    parser = argparse.ArgumentParser(description="GAIA-103 diversity experiment")
    parser.add_argument(
        "--condition",
        type=str,
        choices=GAIA_CONDITIONS,
        help="Run a single condition",
    )
    parser.add_argument("--all", action="store_true", help="Run all 9 conditions")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only write summary.csv from existing JSONL files",
    )
    parser.add_argument(
        "--max-concurrent",
        "--jobs",
        type=int,
        default=100,
        dest="max_concurrent",
        help="Max concurrent questions in flight (each question does 4 rollouts → many OpenAI calls; default: 100)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_GA_DIR / "data" / "GAIA.json"),
        help="Path to GAIA JSON (e.g. data/GAIA.json, 103 questions)",
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        write_summary_csv(output_dir)
        return

    dataset = load_dataset(path=args.dataset)
    print(f"Loaded {len(dataset)} questions")

    if args.condition:
        conditions = [args.condition]
    elif args.all:
        conditions = GAIA_CONDITIONS
    else:
        conditions = []

    for cond_id in conditions:
        await run_condition(
            condition_id=cond_id,
            model=args.model,
            dataset=dataset,
            max_concurrent=args.max_concurrent,
            output_dir=output_dir,
        )

    # After all conditions (or single run), write summary if we have any JSONL
    write_summary_csv(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
