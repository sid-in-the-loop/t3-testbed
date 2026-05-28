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

from webwalkerqa.configs import EXPERIMENT_MATRIX, get_config
from webwalkerqa.dataset import load_dataset
from webwalkerqa.eval import exact_match
from webwalkerqa.methods import get_method

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Fixed list of 10 usual ML seeds (all < 2**32). For k=3 runs we use the first 3: 0, 1, 42.
RUN_SEEDS = [0, 1, 42, 123, 456, 789, 2024, 1234, 3141, 2718]

# All conditions from configs (includes naive-t4, jaccard-o8/16/32/48/64, dense-o8/16/32/48/64)
GAIA_CONDITIONS = sorted(EXPERIMENT_MATRIX.keys())
# Small config: 5 conditions for small-benchmark runs (one job runs all 5)
SMALL_CONDITIONS = ["naive-t4", "jaccard-o8", "jaccard-o16", "dense-o8", "dense-o16"]

RESULTS_DIR = _GA_DIR / "results" / "gaia_103"

# Benchmark names from question_id prefix (e.g. bamboogle-115 -> bamboogle). Used for splitting results.
BENCHMARK_PREFIXES = ("bamboogle", "hotpotqa", "simpleqa", "musique", "2wikimultihopqa")


def get_benchmark_from_question_id(question_id: str) -> str:
    """Return benchmark/dataset name from question_id (e.g. 'hotpotqa-7059' -> 'hotpotqa'). Falls back to 'all' for numeric IDs."""
    s = str(question_id).strip()
    if "-" in s:
        return s.split("-")[0]
    return "all"


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
    run_seed: int = 0,
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
                run_seed=run_seed,
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
            try:
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
                # Add full rollout details for comprehensive logging
                line_dict["rollout_details"] = rollout_results
                line_str = json.dumps(line_dict, ensure_ascii=False) + "\n"
                append_f.write(line_str)
                append_f.flush()
                # Also write to benchmark-specific subdir for per-dataset metrics
                benchmark = get_benchmark_from_question_id(result.question_id)
                bench_dir = output_dir / benchmark
                bench_dir.mkdir(parents=True, exist_ok=True)
                bench_path = bench_dir / f"{condition_id}.jsonl"
                with open(bench_path, "a") as bf:
                    bf.write(line_str)
                new_lines += 1
            except Exception as e:
                print(f"\nError processing question: {e}")
                continue
    finally:
        append_f.close()

    total = len(completed_qids) + new_lines
    print(f"Wrote results to {jsonl_path} ({total} questions total, {new_lines} new)")


def _aggregate_summary_for_dir(dir_path: Path) -> List[Dict[str, Any]]:
    """Compute summary rows (pass@1, pass@4 per condition) for a directory containing condition JSONLs."""
    rows = []
    for cond_id in GAIA_CONDITIONS:
        jsonl_path = dir_path / f"{cond_id}.jsonl"
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
    return rows


def write_summary_csv(output_dir: Path) -> None:
    """Aggregate condition JSONLs into summary.csv (overall and per-benchmark)."""
    # Overall summary (all questions)
    rows = _aggregate_summary_for_dir(output_dir)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to {csv_path}")
        print(df.to_string(index=False))

    # Per-benchmark summaries (bamboogle, hotpotqa, simpleqa, musique, 2wikimultihopqa, and 'all' if present)
    for name in list(BENCHMARK_PREFIXES) + ["all"]:
        bench_dir = output_dir / name
        if not bench_dir.is_dir():
            continue
        bench_rows = _aggregate_summary_for_dir(bench_dir)
        if not bench_rows:
            continue
        bench_df = pd.DataFrame(bench_rows)
        bench_csv = bench_dir / "summary.csv"
        bench_df.to_csv(bench_csv, index=False)
        print(f"\nSummary [{name}] saved to {bench_csv}")
        print(bench_df.to_string(index=False))


async def main():
    parser = argparse.ArgumentParser(description="GAIA-103 diversity experiment")
    parser.add_argument(
        "--condition",
        type=str,
        choices=GAIA_CONDITIONS,
        help="Run a single condition",
    )
    parser.add_argument("--all", action="store_true", help="Run all conditions from configs")
    parser.add_argument("--small", action="store_true", help="Run small config only (naive-t4, jaccard-o8, jaccard-o16, dense-o8, dense-o16)")
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
        "--seed",
        type=int,
        default=0,
        help="Run seed index (0..9). Mapped to RUN_SEEDS[seed %% 10] (e.g. 0->0, 1->1, 2->42). For k=3 runs use 0,1,2.",
    )
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
    elif args.small:
        conditions = SMALL_CONDITIONS
    elif args.all:
        conditions = GAIA_CONDITIONS
    else:
        conditions = []

    run_seed_value = RUN_SEEDS[args.seed % len(RUN_SEEDS)]
    for cond_id in conditions:
        await run_condition(
            condition_id=cond_id,
            model=args.model,
            dataset=dataset,
            max_concurrent=args.max_concurrent,
            output_dir=output_dir,
            run_seed=run_seed_value,
        )

    # After all conditions (or single run), write summary if we have any JSONL
    write_summary_csv(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
