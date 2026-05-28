"""
Extract per-run QPD / ITC / ATC diagnostic metrics from all experiment trees.

Walks the 8 result roots (main tables + legacy + ablations), reads each run's
already-aggregated `summary_T*.csv`, and emits a single CSV at
paper_assets/metrics_all.csv.

Does NOT recompute anything — only reads `mean_jaccard_qpd`, `mean_itc`,
`mean_atc` straight from the saved summary CSVs.

Resumable: if the output CSV already exists, skip rows whose
(model, dataset, condition, k, seed, source_dir) tuple is already present.
Pass --force to re-read everything.

Usage:
  python -m webwalkerqa.scripts.extract_metrics
  python -m webwalkerqa.scripts.extract_metrics --force
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------- config ----------
RESULTS_ROOT = Path("/home/ssmurali/t3-testbed/results")
OUT_CSV = Path("/home/ssmurali/t3-testbed/paper_assets/metrics_all.csv")

# Roots we walk. Only the canonical / current experimental roots.
EXP_ROOTS = [
    "main_table_clueweb_t8",   # Canonical Table 1 (ClueWeb backend)
    "main_table_web_serper",   # Canonical Table 2 (Serper backend, web_reasoning prompt)
    "passk_ablation",          # Phase-2 k=2 ablation (naive_k2, div_k2)
    "poolsize_ablation",       # Pool-size ablation (pool_4/8/16/32)
    "oversample_ablation",     # Oversample-until-N ablation (os_1..os_8)
]

# Map directory name to (canonical condition, k_threads)
COND_DIR_MAP: Dict[str, Tuple[str, int]] = {
    # Sequential
    "sequential": ("sequential", 1),
    "seq":        ("sequential", 1),
    # Parallel — by k (legacy + new)
    "naive_parallel":      ("naive_parallel", 4),
    "diversity_parallel":  ("diversity_parallel", 4),
    "naive_k2": ("naive_parallel", 2),
    "div_k2":   ("diversity_parallel", 2),
    "naive_k4": ("naive_parallel", 4),
    "div_k4":   ("diversity_parallel", 4),
    "naive_k8": ("naive_parallel", 8),
    "div_k8":   ("diversity_parallel", 8),
}
# Pool-size ablation: pool_<P> → (diversity_parallel, 4); P stays in source_dir
POOL_RE = re.compile(r"^pool_(\d+)$")
# Oversample-until-N: os_<N>  → (diversity_parallel, 4); N stays in source_dir
OS_RE = re.compile(r"^os_(\d+)$")


# ---------- column header / metric definitions ----------
HEADER_COMMENT_LINES = [
    "# T3 paper diagnostic metrics — all per-run scalars from summary_T*.csv (read, not recomputed)",
    "# Source: webwalkerqa/run/run_main_table.py compute_qpd / compute_itc / compute_atc",
    "# QPD = Query Pairwise Diversity: mean Jaccard distance over all pairs of turn-1 queries across the k threads (per question, then averaged over questions)",
    "# ITC = Inter-Turn Coherence: per thread, mean Jaccard similarity of turn-1 query vs each later turn's query; averaged over threads then questions",
    "# ATC = Across-Thread Coherence (despite the name, this is a DISTANCE): per turn, mean pairwise Jaccard distance between threads' queries; averaged across turns and questions",
    "# Sequential (k=1) writes 0.0 for all three metrics by construction (no pair to compare).",
]
COLUMNS = ["model", "dataset", "condition", "k", "seed", "num_questions",
           "QPD", "ITC", "ATC", "source_dir"]


# ---------- core extraction ----------
def parse_condition(cond_dir: str) -> Optional[Tuple[str, int]]:
    """Return (condition, k) for known dir names, else None to skip."""
    if cond_dir in COND_DIR_MAP:
        return COND_DIR_MAP[cond_dir]
    if POOL_RE.match(cond_dir):
        return ("diversity_parallel", 4)  # pool ablation is always div_k4
    if OS_RE.match(cond_dir):
        return ("diversity_parallel", 4)  # oversample ablation is always div_k4
    return None


def parse_seed(run_name: str) -> Optional[int]:
    m = re.match(r"^run_(\d+)$", run_name)
    return int(m.group(1)) if m else None


def read_summary_metrics(summary_csv: Path) -> Optional[Dict[str, str]]:
    """Read first data row, return {n_questions, QPD, ITC, ATC} or None on parse error."""
    try:
        with open(summary_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        r = rows[0]
        return {
            "num_questions": r.get("n_questions", ""),
            "QPD": r.get("mean_jaccard_qpd", ""),
            "ITC": r.get("mean_itc", ""),
            "ATC": r.get("mean_atc", ""),
        }
    except Exception:
        return None


def walk_one_root(exp_root: str, already_seen: Set[Tuple]) -> List[Dict[str, str]]:
    """Walk results/<exp_root>/<model>/<dataset>/<cond_dir>/run_<seed>/ and yield rows."""
    rows = []
    root_dir = RESULTS_ROOT / exp_root
    if not root_dir.exists():
        return rows
    for summary in sorted(root_dir.rglob("summary_T*.csv")):
        try:
            rel = summary.relative_to(RESULTS_ROOT).parts
            # Expected: <exp_root>/<model>/<dataset>/<cond_dir>/run_<seed>/summary_T*.csv
            if len(rel) != 6:
                continue
            _, model, dataset, cond_dir, run_name, _ = rel
        except Exception:
            continue

        parsed = parse_condition(cond_dir)
        if parsed is None:
            continue
        condition, k = parsed
        seed = parse_seed(run_name)
        if seed is None:
            continue

        source_dir = "/".join(rel[:-1])  # everything except the .csv basename
        key = (model, dataset, condition, k, seed, source_dir)
        if key in already_seen:
            continue

        m = read_summary_metrics(summary)
        if m is None:
            continue

        rows.append({
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "k": k,
            "seed": seed,
            "num_questions": m["num_questions"],
            "QPD": m["QPD"],
            "ITC": m["ITC"],
            "ATC": m["ATC"],
            "source_dir": source_dir,
        })
        already_seen.add(key)
    return rows


def load_existing_keys(out_csv: Path) -> Tuple[Set[Tuple], List[Dict[str, str]]]:
    """If output CSV exists (and not --force), return (existing keys, existing rows for re-write)."""
    if not out_csv.exists():
        return set(), []
    keys: Set[Tuple] = set()
    rows: List[Dict[str, str]] = []
    with open(out_csv) as f:
        # skip leading comment lines
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for r in reader:
        try:
            key = (r["model"], r["dataset"], r["condition"], int(r["k"]), int(r["seed"]), r["source_dir"])
            keys.add(key)
            rows.append(r)
        except Exception:
            continue
    return keys, rows


def write_csv(out_csv: Path, all_rows: List[Dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        for line in HEADER_COMMENT_LINES:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)


# ---------- sanity-check report ----------
def parse_float(s: str) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def report(rows: List[Dict[str, str]]) -> None:
    n = len(rows)
    unique_tuples = {(r["model"], r["dataset"], r["condition"], int(r["k"])) for r in rows}
    print(f"\nRow count: {n}")
    print(f"Unique (model, dataset, condition, k) tuples: {len(unique_tuples)}")
    for metric in ["QPD", "ITC", "ATC"]:
        vals = [parse_float(r[metric]) for r in rows]
        non_null = [v for v in vals if v is not None]
        nulls = sum(1 for v in vals if v is None)
        if non_null:
            print(f"  {metric}: non_null={len(non_null)}  nulls={nulls}  "
                  f"min={min(non_null):.4f}  max={max(non_null):.4f}  "
                  f"mean={sum(non_null)/len(non_null):.4f}")
        else:
            print(f"  {metric}: non_null=0  nulls={nulls}  (no numeric values!)")

    # Anomaly: ATC null but QPD/ITC present
    anomalies_atc_only = []
    for r in rows:
        q, i, a = parse_float(r["QPD"]), parse_float(r["ITC"]), parse_float(r["ATC"])
        if a is None and (q is not None or i is not None):
            anomalies_atc_only.append((r["model"], r["dataset"], r["condition"], r["k"], r["seed"], r["source_dir"]))
    if anomalies_atc_only:
        print(f"\n  ATC-null but QPD/ITC present ({len(anomalies_atc_only)} rows) — possible extraction bug:")
        for a in anomalies_atc_only[:5]:
            print(f"    {a}")
        if len(anomalies_atc_only) > 5:
            print(f"    ... and {len(anomalies_atc_only) - 5} more")
    else:
        print("\n  No ATC-null-only-anomaly rows.")

    # Other anomalies
    bad = []
    for r in rows:
        q, i, a = parse_float(r["QPD"]), parse_float(r["ITC"]), parse_float(r["ATC"])
        cond = r["condition"]; k = int(r["k"])
        # ITC > 1 (it's a similarity, should be in [0,1])
        if i is not None and (i > 1.0 + 1e-6 or i < -1e-6):
            bad.append(("ITC out of [0,1]", r["model"], r["dataset"], cond, k, r["seed"], i))
        # QPD/ATC out of [0,1] (jaccard distance)
        if q is not None and (q > 1.0 + 1e-6 or q < -1e-6):
            bad.append(("QPD out of [0,1]", r["model"], r["dataset"], cond, k, r["seed"], q))
        if a is not None and (a > 1.0 + 1e-6 or a < -1e-6):
            bad.append(("ATC out of [0,1]", r["model"], r["dataset"], cond, k, r["seed"], a))
        # All-zero parallel run (suspicious — should have nonzero diversity)
        if cond != "sequential" and k >= 2 and q == 0.0 and a == 0.0:
            bad.append(("parallel run with QPD=ATC=0 (broken?)",
                        r["model"], r["dataset"], cond, k, r["seed"], None))
    if bad:
        print(f"\n  Other anomalies ({len(bad)}):")
        for b in bad[:10]:
            print(f"    {b}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")
    else:
        print("\n  No range/zero anomalies.")


# ---------- main ----------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-extract everything; ignore existing CSV.")
    args = p.parse_args()

    if args.force or not OUT_CSV.exists():
        already_seen: Set[Tuple] = set()
        kept_rows: List[Dict[str, str]] = []
        if args.force:
            print("[--force] Re-extracting all rows.")
        else:
            print(f"[fresh] Output {OUT_CSV} doesn't exist; full extraction.")
    else:
        already_seen, kept_rows = load_existing_keys(OUT_CSV)
        print(f"[resume] Found {len(kept_rows)} existing rows in {OUT_CSV}; will append new ones.")

    new_rows: List[Dict[str, str]] = []
    for root in EXP_ROOTS:
        rows = walk_one_root(root, already_seen)
        print(f"  {root:<28} → +{len(rows)} new rows")
        new_rows.extend(rows)

    all_rows = kept_rows + new_rows
    write_csv(OUT_CSV, all_rows)
    print(f"\nWrote {len(all_rows)} total rows to {OUT_CSV}")
    report(all_rows)


if __name__ == "__main__":
    main()
