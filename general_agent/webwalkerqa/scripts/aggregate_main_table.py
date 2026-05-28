"""
Aggregate main table results into Tables 1 and 2.

Walks results/main_table/{model}/{dataset}/ and collects summary CSVs.

Usage:
  cd general_agent
  python -m webwalkerqa.scripts.aggregate_main_table
  python -m webwalkerqa.scripts.aggregate_main_table --results-dir /path/to/results/main_table
  python -m webwalkerqa.scripts.aggregate_main_table --turns 5
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

_TESTBED = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RESULTS = _TESTBED / "results" / "main_table"

TABLE1_DATASETS = ["hotpotqa", "musique", "2wikimultihopqa", "bamboogle", "frames"]
TABLE2_DATASETS = ["GAIA", "hle", "webwalker"]
ALL_DATASETS = TABLE1_DATASETS + TABLE2_DATASETS

CONDITIONS = ["sequential", "naive_parallel", "diversity_parallel"]
CONDITION_DISPLAY = {
    "sequential": "Sequential ReAct",
    "naive_parallel": "Naive Parallel (k=4)",
    "diversity_parallel": "Diversity-Forced (k=4)",
}

MODELS_ORDER = ["gpt-4o-mini", "gpt-4.1-mini", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwq-32b-preview"]


def load_summary(results_dir: Path, model: str, dataset: str, turns: int) -> Dict:
    """Load summary CSV for a model/dataset/turns combination."""
    summary_path = results_dir / model / dataset / f"summary_T{turns}.csv"
    if not summary_path.exists():
        return {}

    rows = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["condition"]] = row
    return rows


def load_judged_summary(results_dir: Path, model: str, dataset: str, turns: int) -> Dict:
    """Load LLM-judged summary for Table 2 datasets."""
    judged_path = results_dir / model / dataset / "judged" / "summary_llm.csv"
    if not judged_path.exists():
        return {}

    rows = {}
    with open(judged_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get("condition", "")
            rows[cond] = row
    return rows


def format_table(
    results_dir: Path,
    datasets: List[str],
    models: List[str],
    turns: int,
    use_judge: bool = False,
    metric_key: str = "pass_at_4",
) -> str:
    """Build a formatted table string."""
    # Header
    header = f"{'Model':<18} {'Method':<26}"
    for ds in datasets:
        header += f" {ds:>12}"
    header += f" {'Avg':>8}"
    lines = [header, "-" * len(header)]

    for model in models:
        for cond in CONDITIONS:
            if use_judge:
                summaries = {ds: load_judged_summary(results_dir, model, ds, turns) for ds in datasets}
                # For judged, condition names may have suffixes
                def get_val(ds):
                    s = summaries[ds]
                    for key in [cond, f"{cond}_T{turns}"]:
                        if key in s:
                            return float(s[key].get(f"pass@4_llm", s[key].get("pass_at_4_llm", 0)))
                    return None
            else:
                summaries = {ds: load_summary(results_dir, model, ds, turns) for ds in datasets}
                def get_val(ds):
                    s = summaries[ds]
                    if cond in s:
                        return float(s[cond].get(metric_key, 0))
                    return None

            vals = []
            row = f"{model:<18} {CONDITION_DISPLAY.get(cond, cond):<26}"
            for ds in datasets:
                v = get_val(ds)
                if v is not None:
                    row += f" {v:>11.1%}"
                    vals.append(v)
                else:
                    row += f" {'—':>12}"

            if vals:
                avg = sum(vals) / len(vals)
                row += f" {avg:>7.1%}"
            else:
                row += f" {'—':>8}"

            lines.append(row)
        lines.append("")  # blank line between models

    return "\n".join(lines)


def format_metrics_table(
    results_dir: Path,
    datasets: List[str],
    models: List[str],
    turns: int,
) -> str:
    """Build QPD/ITC/ATC metrics table."""
    header = f"{'Model':<18} {'Method':<26} {'QPD':>8} {'ITC':>8} {'ATC':>8}"
    lines = [header, "-" * len(header)]

    for model in models:
        for cond in CONDITIONS:
            qpd_vals, itc_vals, atc_vals = [], [], []
            for ds in datasets:
                s = load_summary(results_dir, model, ds, turns)
                if cond in s:
                    qpd_vals.append(float(s[cond].get("mean_jaccard_qpd", 0)))
                    itc_vals.append(float(s[cond].get("mean_itc", 0)))
                    atc_vals.append(float(s[cond].get("mean_atc", 0)))

            def _avg(vals):
                return sum(vals) / len(vals) if vals else 0

            row = f"{model:<18} {CONDITION_DISPLAY.get(cond, cond):<26}"
            row += f" {_avg(qpd_vals):>8.4f} {_avg(itc_vals):>8.4f} {_avg(atc_vals):>8.4f}"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Aggregate main table results")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--turns", type=int, default=5)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    turns = args.turns

    # Discover models
    models = []
    if results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and d.name in MODELS_ORDER:
                models.append(d.name)
    if not models:
        models = [d.name for d in sorted(results_dir.iterdir()) if d.is_dir()] if results_dir.exists() else []

    if not models:
        print(f"No model directories found in {results_dir}")
        return

    print(f"Models found: {models}")
    print(f"Turn config: T={turns}")
    print()

    # Table 1: Multi-hop QA (Exact Match)
    print("=" * 100)
    print(f"TABLE 1: Multi-hop QA Results (pass@4, Exact Match, T_par={turns})")
    print("=" * 100)
    print(format_table(results_dir, TABLE1_DATASETS, models, turns, use_judge=False))
    print()

    # Table 2: Hard Reasoning (LLM-as-judge)
    print("=" * 100)
    print(f"TABLE 2: Hard Reasoning Results (pass@4, LLM-as-judge, T_par={turns})")
    print("=" * 100)
    print(format_table(results_dir, TABLE2_DATASETS, models, turns, use_judge=True))
    print()

    # Metrics
    print("=" * 100)
    print(f"METRICS: QPD / ITC / ATC (averaged over all datasets, T_par={turns})")
    print("=" * 100)
    print(format_metrics_table(results_dir, ALL_DATASETS, models, turns))

    # Save as CSV
    out_csv = results_dir / f"table1_T{turns}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method"] + TABLE1_DATASETS + ["avg"])
        for model in models:
            for cond in CONDITIONS:
                row = [model, cond]
                vals = []
                for ds in TABLE1_DATASETS:
                    s = load_summary(results_dir, model, ds, turns)
                    if cond in s:
                        v = float(s[cond].get("pass_at_4", 0))
                        row.append(f"{v:.4f}")
                        vals.append(v)
                    else:
                        row.append("")
                avg = sum(vals) / len(vals) if vals else ""
                row.append(f"{avg:.4f}" if isinstance(avg, float) else "")
                w.writerow(row)
    print(f"\nSaved Table 1 CSV: {out_csv}")

    out_csv2 = results_dir / f"table2_T{turns}.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method"] + TABLE2_DATASETS + ["avg"])
        for model in models:
            for cond in CONDITIONS:
                row = [model, cond]
                vals = []
                for ds in TABLE2_DATASETS:
                    s = load_judged_summary(results_dir, model, ds, turns)
                    found = False
                    for key in [cond, f"{cond}_T{turns}"]:
                        if key in s:
                            v = float(s[key].get("pass@4_llm", s[key].get("pass_at_4_llm", 0)))
                            row.append(f"{v:.4f}")
                            vals.append(v)
                            found = True
                            break
                    if not found:
                        row.append("")
                avg = sum(vals) / len(vals) if vals else ""
                row.append(f"{avg:.4f}" if isinstance(avg, float) else "")
                w.writerow(row)
    print(f"Saved Table 2 CSV: {out_csv2}")


if __name__ == "__main__":
    main()
