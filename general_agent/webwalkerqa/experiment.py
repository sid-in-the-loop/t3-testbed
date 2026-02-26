"""
Main experiment runner for WebWalkerQA T³ PoC.

Usage:
  # Run a single config:
  python -m webwalkerqa.experiment --config A1 --model openai/gpt-4o-mini

  # Run all configs:
  python -m webwalkerqa.experiment --all --model openai/gpt-4o-mini

  # Run a specific group:
  python -m webwalkerqa.experiment --group B --model openai/gpt-4o-mini

  # Limit dataset size (for quick testing):
  python -m webwalkerqa.experiment --config A2 --model openai/gpt-4o-mini --max-examples 10 --verbose

  # Override dataset path:
  python -m webwalkerqa.experiment --config B1 --dataset /path/to/webwalkerqa.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the general_agent directory is importable
_GA_DIR = Path(__file__).parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")

from webwalkerqa.configs import EXPERIMENT_MATRIX, list_configs, get_config
from webwalkerqa.dataset import load_dataset
from webwalkerqa.eval import compute_em_score
from webwalkerqa.runner import run_question


DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_OUTPUT_BASE = _GA_DIR / "results" / "webwalkerqa"


async def run_experiment(
    config_id: str,
    model: str,
    output_base: Path,
    dataset_path: str = None,
    max_examples: int = None,
    max_concurrent: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Run a single experiment config on WebWalkerQA.

    Args:
        config_id: e.g. "A1", "B2", "Oracle"
        model: LiteLLM model string
        output_base: Base directory for results
        dataset_path: Path to local JSON (auto-downloads if None)
        max_examples: Limit dataset size
        max_concurrent: Max concurrent questions
        verbose: Print per-question output

    Returns:
        Summary dict with EM score and stats.
    """
    config = get_config(config_id)
    print(f"\n{'='*60}")
    print(f"Config: {config.id} | Method: {config.method}")
    print(f"k={config.k} threads, n={config.n} turns, t={config.t} tokens/thread/turn")
    print(f"Total tokens (est.): {config.total_tokens:,}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    # Load dataset
    examples = load_dataset(path=dataset_path, max_examples=max_examples)
    print(f"Dataset: {len(examples)} questions")

    # Output directory for this run
    model_short = model.split("/")[-1].replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"{config.id}_{model_short}" / "questions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run questions with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def run_one(example):
        async with semaphore:
            return await run_question(
                example=example,
                config=config,
                model=model,
                output_dir=output_dir,
                verbose=verbose,
            )

    tasks = [run_one(ex) for ex in examples]

    # Process with progress updates
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 10 == 0 or completed == len(examples):
            n_correct = sum(1 for r in results if r.em)
            print(f"  Progress: {completed}/{len(examples)} | EM so far: {n_correct}/{completed} "
                  f"({100*n_correct/completed:.1f}%)")

    # Compute summary
    result_dicts = [r.to_dict() for r in results]
    em_stats = compute_em_score(result_dicts)

    summary = {
        "config_id": config.id,
        "method": config.method,
        "k": config.k,
        "n": config.n,
        "t": config.t,
        "model": model,
        "timestamp": timestamp,
        "num_questions": len(examples),
        **em_stats,
        "avg_turns_used": sum(r.turns_used for r in results) / len(results) if results else 0,
        "avg_search_calls": sum(r.search_calls_used for r in results) / len(results) if results else 0,
        "total_prompt_tokens": sum(r.total_prompt_tokens for r in results),
        "total_output_tokens": sum(r.total_output_tokens for r in results),
        "num_errors": sum(1 for r in results if r.error),
    }

    # Save summary
    summary_path = output_base / f"{config.id}_{model_short}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULT — {config.id} ({config.description})")
    print(f"  EM:            {em_stats['em']:.4f} ({em_stats['num_correct']}/{em_stats['num_total']})")
    print(f"  Avg turns:     {summary['avg_turns_used']:.1f}")
    print(f"  Avg searches:  {summary['avg_search_calls']:.1f}")
    print(f"  Total tokens:  {summary['total_prompt_tokens'] + summary['total_output_tokens']:,}")
    print(f"  Summary saved: {summary_path}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="WebWalkerQA T³ PoC experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config selection
    cfg_group = parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument(
        "--config", type=str,
        choices=list(EXPERIMENT_MATRIX.keys()) + [k.lower() for k in EXPERIMENT_MATRIX.keys()],
        help="Single experiment config to run (e.g. A1, B2, Oracle)",
    )
    cfg_group.add_argument("--all", action="store_true", help="Run all configs")
    cfg_group.add_argument(
        "--group", type=str, choices=["A", "B", "C", "Oracle"],
        help="Run all configs in a group",
    )
    cfg_group.add_argument(
        "--s1-only", action="store_true", help="Run only s1 configs (A1, B1, C1)",
    )
    cfg_group.add_argument(
        "--t3-only", action="store_true", help="Run only T³ Fixed configs",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL})",
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to local WebWalkerQA JSON file (auto-downloads if not set)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit number of questions (for debugging)",
    )

    # Execution
    parser.add_argument(
        "--max-concurrent", type=int, default=4,
        help="Max concurrent question processing (default: 4)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_BASE),
        help=f"Output directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-question output")

    args = parser.parse_args()

    # Determine configs to run
    if args.config:
        configs_to_run = [args.config.upper()]
    elif args.all:
        configs_to_run = list(EXPERIMENT_MATRIX.keys())
    elif args.group:
        configs_to_run = [c.id for c in list_configs(args.group)]
    elif args.s1_only:
        configs_to_run = [c.id for c in EXPERIMENT_MATRIX.values() if c.method == "s1"]
    elif args.t3_only:
        configs_to_run = [c.id for c in EXPERIMENT_MATRIX.values() if c.method == "t3_fixed"]
    else:
        configs_to_run = []

    output_base = Path(args.output_dir)
    all_summaries = []

    for config_id in configs_to_run:
        summary = asyncio.run(run_experiment(
            config_id=config_id,
            model=args.model,
            output_base=output_base,
            dataset_path=args.dataset,
            max_examples=args.max_examples,
            max_concurrent=args.max_concurrent,
            verbose=args.verbose,
        ))
        all_summaries.append(summary)

    # Print final comparison table
    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"{'Config':<10} {'Method':<12} {'k':>4} {'t':>6} {'EM':>8} {'Searches':>10}")
        print("-" * 70)
        for s in all_summaries:
            print(
                f"{s['config_id']:<10} {s['method']:<12} {s['k']:>4} {s['t']:>6} "
                f"{s['em']:>8.4f} {s['avg_search_calls']:>10.1f}"
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
