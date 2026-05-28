#!/usr/bin/env python3
"""
Aggregate LLM-as-a-judge results into one .txt file per model.

Reads all summary_llm.csv under JUDGE_OUTPUT_BASE (per model_seed and per benchmark),
and writes a single labeled text file per model: benchmark, seed, condition, pass@1_llm, pass@4_llm, n.

Usage:
  python scripts/aggregate_llm_judge_results.py [--judge-dir RESULTS_DIR] [--output-dir OUT_DIR]
  Default: --judge-dir results/small_benchmark_seeds_judged --output-dir results/small_benchmark_seeds_judged
  (writes results/small_benchmark_seeds_judged/aggregate_llm_judge_<model>.txt)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Resolve paths relative to general_agent
_GA_DIR = Path(__file__).resolve().parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

BENCHMARKS_ORDER = ("overall", "bamboogle", "hotpotqa", "simpleqa", "musique", "2wikimultihopqa")


def parse_model_seed(dirname: str) -> tuple[str | None, int | None]:
    """Return (model, seed_int) or (None, None) if not model_seedN."""
    m = re.match(r"^(.+)_seed(\d+)$", dirname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def collect_judge_summaries(judge_base: Path) -> dict[str, list[dict]]:
    """
    judge_base: e.g. results/small_benchmark_seeds_judged
    Returns: { model: [ {"benchmark": str, "seed": int, "condition": str, "pass@1_llm": float, "pass@4_llm": float, "n": int}, ... ] }
    """
    by_model: dict[str, list[dict]] = {}

    for path in sorted(judge_base.iterdir()):
        if not path.is_dir():
            continue
        model, seed = parse_model_seed(path.name)
        if model is None or seed is None:
            continue

        if model not in by_model:
            by_model[model] = []

        # Overall (root summary_llm.csv)
        overall_csv = path / "summary_llm.csv"
        if overall_csv.exists():
            df = pd.read_csv(overall_csv)
            for _, row in df.iterrows():
                by_model[model].append({
                    "benchmark": "overall",
                    "seed": seed,
                    "condition": row["condition"],
                    "pass@1_llm": float(row["pass@1_llm"]),
                    "pass@4_llm": float(row["pass@4_llm"]),
                    "n": int(row["n"]),
                })

        # Per-benchmark
        for bench in ("bamboogle", "hotpotqa", "simpleqa", "musique", "2wikimultihopqa"):
            bench_csv = path / bench / "summary_llm.csv"
            if not bench_csv.exists():
                continue
            df = pd.read_csv(bench_csv)
            for _, row in df.iterrows():
                by_model[model].append({
                    "benchmark": bench,
                    "seed": seed,
                    "condition": row["condition"],
                    "pass@1_llm": float(row["pass@1_llm"]),
                    "pass@4_llm": float(row["pass@4_llm"]),
                    "n": int(row["n"]),
                })

    return by_model


def format_aggregate_txt(records: list[dict]) -> str:
    """Format one model's records into a single labeled text block."""
    lines = []
    for bench in BENCHMARKS_ORDER:
        bench_records = [r for r in records if r["benchmark"] == bench]
        if not bench_records:
            continue
        lines.append("")
        lines.append(f"=== benchmark: {bench} ===")
        # Group by seed
        seeds = sorted({r["seed"] for r in bench_records})
        for seed in seeds:
            seed_records = [r for r in bench_records if r["seed"] == seed]
            lines.append(f"  --- seed: {seed} ---")
            for r in sorted(seed_records, key=lambda x: x["condition"]):
                lines.append(
                    f"    {r['condition']:14}  pass@1_llm={r['pass@1_llm']:.3f}  pass@4_llm={r['pass@4_llm']:.3f}  n={r['n']}"
                )
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM-as-a-judge results into one .txt per model")
    parser.add_argument(
        "--judge-dir",
        type=str,
        default="results/small_benchmark_seeds_judged",
        help="Base dir containing model_seedN subdirs with summary_llm.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write aggregate_llm_judge_<model>.txt (default: same as judge-dir)",
    )
    args = parser.parse_args()

    judge_base = _GA_DIR / args.judge_dir
    output_dir = _GA_DIR / (args.output_dir or args.judge_dir)
    if not judge_base.is_dir():
        print(f"Judge dir not found: {judge_base}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_model = collect_judge_summaries(judge_base)
    if not by_model:
        print("No model_seed* dirs with summary_llm.csv found.", file=sys.stderr)
        sys.exit(1)

    for model in sorted(by_model.keys()):
        records = by_model[model]
        txt = f"# LLM-as-a-judge aggregate — model: {model}\n" + format_aggregate_txt(records)
        out_path = output_dir / f"aggregate_llm_judge_{model}.txt"
        out_path.write_text(txt, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
