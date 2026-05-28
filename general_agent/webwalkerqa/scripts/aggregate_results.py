"""
Aggregate main table results across 5 seeds.
Drops the single outlier run (furthest from mean) and reports mean±std over top-4.

Usage:
  cd general_agent
  python -m webwalkerqa.scripts.aggregate_results --results-dir /home/ssmurali/t3-testbed/results/main_table
  python -m webwalkerqa.scripts.aggregate_results --results-dir /home/ssmurali/t3-testbed/results/main_table --metric pass_at_4_llm
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METRICS = ["pass_at_1_llm", "pass_at_4_llm", "pass_at_1", "pass_at_4"]
CONDITIONS = [
    "sequential", "naive_parallel", "diversity_parallel",  # legacy names
    "seq", "naive_k2", "div_k2", "naive_k4", "div_k4", "naive_k8", "div_k8",  # new named-by-k
    # Pool-size ablation
    "pool_4", "pool_8", "pool_16", "pool_32",
    # Oversample-until-turn-N ablation
    "os_1", "os_2", "os_3", "os_4", "os_5", "os_6", "os_7", "os_8",
]
COND_SHORT = {
    "sequential": "Seq", "naive_parallel": "Naive", "diversity_parallel": "Div",
    "seq": "Seq(1x32)",
    "naive_k2": "Nv@2", "div_k2": "Dv@2",
    "naive_k4": "Nv@4", "div_k4": "Dv@4",
    "naive_k8": "Nv@8", "div_k8": "Dv@8",
}


def drop_outlier(values: List[float]) -> List[float]:
    """Drop the single value furthest from the mean, return remaining."""
    if len(values) <= 1:
        return values
    arr = np.array(values)
    mean = arr.mean()
    dists = np.abs(arr - mean)
    outlier_idx = int(np.argmax(dists))
    return [v for i, v in enumerate(values) if i != outlier_idx]


def collect(results_dir: Path) -> pd.DataFrame:
    """Walk results dir and collect per-run metric values."""
    rows = []
    for summary in sorted(results_dir.rglob("summary_T*.csv")):
        # Expected: results_dir/model/dataset/condition/run_N/summary_T*.csv
        try:
            rel_parts = summary.relative_to(results_dir).parts
            if len(rel_parts) < 5:
                continue
            model, dataset, condition, run = rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3]
        except ValueError:
            continue

        if condition not in CONDITIONS:
            continue

        df = pd.read_csv(summary)
        if df.empty:
            continue

        row = {"model": model, "dataset": dataset, "condition": condition, "run": run}
        for m in METRICS:
            row[m] = float(df[m].iloc[0]) if m in df.columns else None
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each (model, dataset, condition), drop outlier, compute mean±std over top-4."""
    records = []
    for (model, dataset, condition), grp in df.groupby(["model", "dataset", "condition"]):
        values = grp[metric].dropna().tolist()
        if not values:
            continue
        top4 = drop_outlier(values)
        mean = np.mean(top4)
        std = np.std(top4, ddof=1) if len(top4) > 1 else 0.0
        records.append({
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "n_runs": len(values),
            "n_used": len(top4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "all_values": [round(v, 4) for v in values],
            "used_values": [round(v, 4) for v in top4],
        })
    return pd.DataFrame(records)


def print_table(agg: pd.DataFrame, metric: str):
    """Print a pivot table: rows=dataset, cols=condition, cells=mean±std."""
    pivot_mean = agg.pivot_table(index=["model", "dataset"], columns="condition", values="mean")
    pivot_std  = agg.pivot_table(index=["model", "dataset"], columns="condition", values="std")

    # Reorder columns
    cols = [c for c in CONDITIONS if c in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_std  = pivot_std[cols]

    print(f"\n{'='*90}")
    print(f"  Metric: {metric} (%)  |  mean ± std over top-4 runs (outlier dropped)")
    print(f"{'='*90}")

    header = f"{'Model / Dataset':<32}"
    for c in cols:
        header += f"  {COND_SHORT[c]:>18}"
    print(header)
    print("-" * 90)

    prev_model = None
    for (model, dataset) in pivot_mean.index:
        if model != prev_model:
            if prev_model is not None:
                print()
            print(f"  [{model}]")
            prev_model = model
        row_str = f"    {dataset:<28}"
        for c in cols:
            m = pivot_mean.loc[(model, dataset), c] if c in pivot_mean.columns else None
            s = pivot_std.loc[(model, dataset), c] if c in pivot_std.columns else None
            if m is not None and not np.isnan(m):
                row_str += f"  {m*100:5.1f} ± {s*100:4.1f}      "
            else:
                row_str += f"  {'—':>18}"
        print(row_str)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--metric", default="pass_at_4_llm",
                        choices=METRICS, help="Primary metric to aggregate")
    parser.add_argument("--save-csv", default="", help="Save aggregated table to this CSV path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        sys.exit(1)

    print(f"Collecting from: {results_dir}")
    raw = collect(results_dir)
    if raw.empty:
        print("No summary CSVs found.")
        sys.exit(1)
    print(f"Found {len(raw)} run entries across {raw['dataset'].nunique()} datasets, "
          f"{raw['model'].nunique()} models, {raw['condition'].nunique()} conditions")

    agg = aggregate(raw, args.metric)
    print_table(agg, args.metric)

    if args.save_csv:
        agg.to_csv(args.save_csv, index=False)
        print(f"\nSaved to {args.save_csv}")


if __name__ == "__main__":
    main()
