"""
Figure 3 — pass@k sweep on GAIA-103 (Qwen3-8B), naive parallel vs S^3.

Data sources (LLM-judge pass rates, 3 seeds per cell):
  k=1 (sequential):  results/main_table_web_serper/qwen3-8b/gaia/seq/run_*/summary_T25.csv
                     -> column: pass_at_1_llm
  k=2:               results/passk_ablation/qwen3-8b/GAIA/{naive_k2,div_k2}/run_*/summary_T16.csv
                     -> column: pass_at_4_llm  (judging padded threads 3,4 as empty -> equals true pass@2)
  k=4:               results/main_table_web_serper/qwen3-8b/gaia/{naive_k4,div_k4}/run_*/summary_T8.csv
                     -> column: pass_at_4_llm
  k=8 (EXTRAPOLATED): results/main_table_web_serper/qwen3-8b/gaia/{naive_k8,div_k8}/run_*/{...}_T8.{csv,jsonl}
                     -> only threads 1-4 were LLM-judged. We compute pass@8 by
                        gold-match calibration:
                          delta = pass@4_llm  -  pass@4_gold(threads 1-4)
                          pass@8_llm_est = pass@8_gold(threads 1-8)  +  delta
                        where gold match is permissive substring containment.

Outputs (paper_assets/figures/fig3/):
  data.csv          long-form: model, dataset, condition, k, seed, passk, extrapolated
  passk_sweep.pdf   single panel, 3.3in x 2.8in
  passk_sweep.png   1400 dpi preview

Usage (from compute node — needs no external deps):
  python paper_assets/figures/fig3/plot_passk_sweep.py
"""

from __future__ import annotations
import csv
import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Shared style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, S3, SEQ, AXIS, GRID, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

ROOT = TESTBED_ROOT / "results"
SEEDS = [1, 2, 3]
OUT_DIR = Path(__file__).resolve().parent
CSV_OUT = OUT_DIR / "data.csv"
MODEL = "qwen3-8b"
DATASET = "gaia"


# ---------- helpers ----------
def normalize(s: str) -> str:
    return (s or "").strip().lower().rstrip(".,!?;:").strip()


def per_thread_gold(csv_path: Path, n_threads: int) -> List[List[bool]]:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            gold = normalize(r.get("gold_answer", ""))
            labels = []
            for i in range(1, n_threads + 1):
                ans = normalize(r.get(f"thread_{i}_answer", ""))
                labels.append(bool(ans) and (gold in ans or ans in gold))
            rows.append(labels)
    return rows


def passk_unbiased(per_q_labels: List[List[bool]], k: int) -> float:
    out = []
    for labs in per_q_labels:
        N, c = len(labs), sum(labs)
        if k > N:
            out.append(1.0 if c > 0 else 0.0)
            continue
        out.append(1.0 if (N - c) < k else 1.0 - comb(N - c, k) / comb(N, k))
    return float(np.mean(out)) if out else float("nan")


def read_summary_passk(run_dir: Path, col: str) -> float:
    sm = list(run_dir.glob("summary_T*.csv"))[0]
    with open(sm) as f:
        return float(next(csv.DictReader(f))[col])


# ---------- per-cell aggregation ----------
def collect_passk_per_seed() -> List[dict]:
    rows: List[dict] = []

    # k=1 sequential (both naive and S^3 inherit the same point)
    for s in SEEDS:
        v = read_summary_passk(ROOT / f"main_table_web_serper/{MODEL}/{DATASET}/seq/run_{s}", "pass_at_1_llm")
        rows.append(dict(condition="sequential", k=1, seed=s, passk=v, extrapolated=False))

    # k=2 from passk_ablation (pass_at_4_llm == true pass@2 here, threads 3,4 empty)
    for cond in ["naive_k2", "div_k2"]:
        for s in SEEDS:
            run = ROOT / f"passk_ablation/{MODEL}/GAIA/{cond}/run_{s}"
            if not run.exists():
                print(f"  MISSING: {cond} seed {s}")
                continue
            v = read_summary_passk(run, "pass_at_4_llm")
            label = "naive" if cond.startswith("naive") else "s3"
            rows.append(dict(condition=label, k=2, seed=s, passk=v, extrapolated=False))

    # k=4 main_table
    for cond in ["naive_k4", "div_k4"]:
        for s in SEEDS:
            run = ROOT / f"main_table_web_serper/{MODEL}/{DATASET}/{cond}/run_{s}"
            if not run.exists():
                print(f"  MISSING: {cond} seed {s}")
                continue
            v = read_summary_passk(run, "pass_at_4_llm")
            label = "naive" if cond.startswith("naive") else "s3"
            rows.append(dict(condition=label, k=4, seed=s, passk=v, extrapolated=False))

    # k=8 extrapolated via gold-match calibration
    # Per-seed: delta = pass@4_llm - pass@4_gold(threads 1-4)
    #          pass@8_llm_est = pass@8_gold(threads 1-8) + delta
    for cond in ["naive_k8", "div_k8"]:
        for s in SEEDS:
            run = ROOT / f"main_table_web_serper/{MODEL}/{DATASET}/{cond}/run_{s}"
            if not run.exists():
                print(f"  MISSING: {cond} seed {s}")
                continue
            data_csvs = [c for c in run.glob("*_T*.csv") if "summary" not in c.name]
            if not data_csvs:
                print(f"  MISSING raw csv: {cond} seed {s}")
                continue
            per_q_gold8 = per_thread_gold(data_csvs[0], 8)
            per_q_gold4 = [labs[:4] for labs in per_q_gold8]
            p4_gold = passk_unbiased(per_q_gold4, 4)
            p8_gold = passk_unbiased(per_q_gold8, 8)
            p4_llm = read_summary_passk(run, "pass_at_4_llm")
            delta = p4_llm - p4_gold
            p8_llm_est = max(0.0, min(1.0, p8_gold + delta))
            label = "naive" if cond.startswith("naive") else "s3"
            rows.append(dict(condition=label, k=8, seed=s, passk=p8_llm_est, extrapolated=True))
    return rows


# ---------- plot ----------
def aggregate(rows: List[dict]) -> Dict[Tuple[str, int], Tuple[float, float, float]]:
    """Return {(condition, k): (mean, std, sem)}.  sem = std / sqrt(n) — band uses SEM."""
    by_key: Dict[Tuple[str, int], List[float]] = {}
    for r in rows:
        by_key.setdefault((r["condition"], r["k"]), []).append(r["passk"])
    out = {}
    for k, vs in by_key.items():
        n = len(vs)
        std = float(np.std(vs, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        out[k] = (float(np.mean(vs)), std, sem)
    return out


def render_plot(agg: Dict[Tuple[str, int], Tuple[float, float, float]]) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    K_ALL = [1, 4, 8]

    # Build naive and S^3 series — at k=1 inherit the sequential point
    seq_mean, _, seq_sem = agg[("sequential", 1)]

    def series(cond_label: str):
        means, sems = [], []
        for k in K_ALL:
            if k == 1:
                means.append(seq_mean); sems.append(seq_sem)
            else:
                m, _, sem = agg.get((cond_label, k), (np.nan, 0.0, 0.0))
                means.append(m); sems.append(sem)
        return np.array(means), np.array(sems)

    nm, ne = series("naive")
    dm, de = series("s3")

    # ---- bands (±1 SEM) ----
    ax.fill_between(K_ALL, nm - ne, nm + ne, color=NAIVE, alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(K_ALL, dm - de, dm + de, color=S3,    alpha=0.18, linewidth=0, zorder=2)

    # ---- line segments ----
    ax.plot(K_ALL, nm, color=NAIVE, linewidth=1.5, zorder=3)
    ax.plot(K_ALL, dm, color=S3, linewidth=1.5, zorder=3)

    # ---- markers (k=8 extrapolated -> open marker) ----
    for k_idx, k in enumerate(K_ALL):
        if k == 1:
            continue
        if k == 8:
            ax.scatter([k], [nm[k_idx]], facecolor="white", edgecolor=NAIVE,
                       linewidth=1.3, marker="o", s=36, zorder=4)
            ax.scatter([k], [dm[k_idx]], facecolor="white", edgecolor=S3,
                       linewidth=1.3, marker="D", s=34, zorder=4)
        else:
            ax.scatter([k], [nm[k_idx]], color=NAIVE, marker="o", s=30, zorder=4,
                       edgecolor="white", linewidth=0.7)
            ax.scatter([k], [dm[k_idx]], color=S3, marker="D", s=30, zorder=4,
                       edgecolor="white", linewidth=0.7)

    # ---- no marker at k=1: both lines emerge from the common sequential point; ----
    # the x-axis tick at 1 + the line origin convey the baseline.

    # ---- inline legend bottom-right (markers only, no frame) ----
    handles = [
        plt.Line2D([], [], color=NAIVE, marker="o", markersize=5,
                   markeredgecolor="white", markeredgewidth=0.7, linewidth=1.5,
                   label="Standard"),
        plt.Line2D([], [], color=S3, marker="D", markersize=5,
                   markeredgecolor="white", markeredgewidth=0.7, linewidth=1.5,
                   label="Ours"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=9, handlelength=1.6, handletextpad=0.4,
              borderaxespad=0.4, labelspacing=0.3)

    # ---- axes ----
    ax.set_xlim(0.5, 8.7)
    ax.set_ylim(0.10, 0.40)
    ax.set_xticks(K_ALL)
    ax.set_xticklabels([str(k) for k in K_ALL])
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_xlabel(r"Parallel threads $k$")
    ax.set_ylabel(r"pass@$k$")

    # Faint horizontal gridlines
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Spines
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(1.0)

    plt.tight_layout()
    save_fig(fig, "passk_sweep", subdir="fig3")
    plt.close(fig)


# ---------- write csv ----------
def write_data_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "dataset", "condition", "k", "seed", "passk", "extrapolated"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"model": MODEL, "dataset": DATASET, **r})


# ---------- main ----------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_passk_per_seed()
    write_data_csv(rows, CSV_OUT)
    agg = aggregate(rows)

    # Print table
    print("\nFig 3 data table (band = ±1 SEM):\n")
    print(f"{'model':<10} {'dataset':<8} {'condition':<10} {'k':>3} {'mean':>8} {'std':>8} {'sem':>8} {'extrap?':>8}")
    seen_keys = set()
    for r in rows:
        key = (r["condition"], r["k"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        m, s, sem = agg[key]
        extr = "yes" if r["extrapolated"] else ""
        print(f"{MODEL:<10} {DATASET:<8} {r['condition']:<10} {r['k']:>3} {m:>8.4f} {s:>8.4f} {sem:>8.4f} {extr:>8}")

    print(f"\nWrote: {CSV_OUT}")
    render_plot(agg)
    print(f"Wrote: {OUT_DIR}/passk_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
