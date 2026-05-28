"""
Appendix Figure 2 — per-(model, dataset) pass@k curves.

Two outputs:
  appfig2a_mhqa.{pdf,png}      5 models × 5 datasets   (~12" × 10")
  appfig2b_reasoning.{pdf,png} 4 models × 3 datasets   (~ 8.5" × 8")

Per-panel:
  X = k ∈ {1, 4, 8}       (k=2 dropped — only available for 13/37 cells)
  Y = pass@k_llm          (auto y-range per panel; both panels in a row share the
                           y-scale via sharey=True for left-right comparability)
  Two solid lines:
    Naive       — NAIVE_BAR, filled circles, black outline, 1.5pt line
    S^3         — S3_BAR,    filled diamonds, black outline, 1.5pt line
  Both lines start from a single grey square at k=1 (SEQ baseline).
  k=8 markers: open (white face) — indicates extrapolation via gold-match
  calibration (same approach as Fig 3).
  Confidence band ±1 SEM across 3 seeds, 20% alpha.

  No per-panel title. Column headers above first row = dataset names; row labels
  on the left of first column = model names. Single shared legend at the top.

Data sources:
  k=1 (seq):  results/{root}/{model}/{dataset}/seq[|sequential]/run_{1,2,3}/summary_T*.csv
              -> pass_at_1_llm
  k=4:        results/{root}/{model}/{dataset}/naive_k4|div_k4   (or legacy naive_parallel|diversity_parallel)
              -> pass_at_4_llm
  k=8:        results/{root}/{model}/{dataset}/naive_k8|div_k8
              -> raw csv (thread_1..8_answer) + summary pass_at_4_llm
              -> per-seed: delta = pass_at_4_llm - pass@4_gold(threads 1-4)
                            pass@8_llm_est = pass@8_gold(threads 1-8) + delta

Usage:
  python paper_assets/figures/appfig2/plot_passk_grid.py
"""
from __future__ import annotations

import csv
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, S3, SEQ, AXIS, GRID, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

# Use CMU Core colors directly.
NAIVE_BAR = NAIVE   # Iron Gray
S3_BAR    = S3      # Carnegie Red — Ours

ROOT = TESTBED_ROOT / "results"
SEEDS = [1, 2, 3]
OUT_DIR = Path(__file__).resolve().parent
K_VALS = [1, 4, 8]

PANEL_A = {
    "name": "appfig2a_mhqa",
    "root": "main_table_clueweb_t8",
    "models":   [("qwen3-1.7b", "qwen3-1.7B"), ("qwen3-4b", "qwen3-4B"),
                 ("qwen3-8b",   "qwen3-8B"),   ("gemma3-4b", "gemma3-4B"),
                 ("gemma3-12b", "gemma3-12B")],
    "datasets": [("2wikimultihopqa", "2Wiki"), ("bamboogle", "Bamboogle"),
                 ("frames", "Frames"), ("hotpotqa", "HotpotQA"),
                 ("musique", "MuSiQue")],
    "figsize": (13.0, 10.0),  # match per-cell scale with PANEL_B
}
PANEL_B = {
    "name": "appfig2b_reasoning",
    "root": "main_table_web_serper",
    "models":   [("qwen3-4b", "qwen3-4B"), ("qwen3-8b", "qwen3-8B"),
                 ("gemma3-12b", "gemma3-12B")],
    "datasets": [("gaia", "GAIA"), ("hle", "HLE"), ("webwalker", "WebWalker")],
    "figsize": (9.0, 7.0),
}


# ---------- helpers ----------
def normalize(s: str) -> str:
    return (s or "").strip().lower().rstrip(".,!?;:").strip()


def has_summary(run_dir: Path) -> bool:
    return run_dir.exists() and bool(list(run_dir.glob("summary_T*.csv")))


def read_summary_passk(run_dir: Path, col: str) -> Optional[float]:
    sm = list(run_dir.glob("summary_T*.csv"))
    if not sm:
        return None
    with open(sm[0]) as f:
        r = next(csv.DictReader(f))
    try:
        return float(r[col])
    except (KeyError, ValueError):
        return None


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


def find_run(root_subdir: str, model: str, dataset: str, *cond_dirs: str
             ) -> Optional[Path]:
    """Return the first run dir among the candidates that has a summary file."""
    for cd in cond_dirs:
        run = ROOT / root_subdir / model / dataset / cd
        if has_summary(run):
            return run
    return None


# ---------- per-cell aggregation ----------
def cell_passk(panel_root: str, model: str, dataset: str
               ) -> Dict[Tuple[str, int], List[float]]:
    """Return {(condition, k): [seed_vals...]} for k in {1, 4, 8}.

    'sequential' for k=1 (shared between naive and s3).
    'naive' / 's3' for k in {4, 8}.
    """
    out: Dict[Tuple[str, int], List[float]] = {}

    # k=1 sequential
    out[("sequential", 1)] = []
    for s in SEEDS:
        run = find_run(panel_root, model, dataset, f"seq/run_{s}", f"sequential/run_{s}")
        if run is not None:
            v = read_summary_passk(run, "pass_at_1_llm")
            if v is not None:
                out[("sequential", 1)].append(v)

    # k=4 naive / div  (try new naming, fall back to legacy)
    for cond, suffix in [("naive", "naive"), ("s3", "div")]:
        out[(cond, 4)] = []
        for s in SEEDS:
            run = find_run(panel_root, model, dataset,
                           f"{suffix}_k4/run_{s}",
                           f"{suffix if suffix == 'naive' else 'diversity'}_parallel/run_{s}")
            if run is not None:
                v = read_summary_passk(run, "pass_at_4_llm")
                if v is not None:
                    out[(cond, 4)].append(v)

    # k=8 — gold-match calibration extrapolation per seed
    for cond, suffix in [("naive", "naive"), ("s3", "div")]:
        out[(cond, 8)] = []
        for s in SEEDS:
            run = ROOT / panel_root / model / dataset / f"{suffix}_k8" / f"run_{s}"
            if not run.exists():
                continue
            data_csvs = [c for c in run.glob("*_T*.csv") if "summary" not in c.name]
            if not data_csvs:
                continue
            # Calibration: pass@8_llm_est = pass@8_gold + (pass@4_llm - pass@4_gold(1-4))
            try:
                per_q_gold8 = per_thread_gold(data_csvs[0], 8)
            except Exception:
                continue
            per_q_gold4 = [labs[:4] for labs in per_q_gold8]
            p4_gold = passk_unbiased(per_q_gold4, 4)
            p8_gold = passk_unbiased(per_q_gold8, 8)
            p4_llm  = read_summary_passk(run, "pass_at_4_llm")
            if p4_llm is None:
                continue
            p8_est = max(0.0, min(1.0, p8_gold + (p4_llm - p4_gold)))
            out[(cond, 8)].append(p8_est)

    return out


def aggregate(seed_vals: List[float]) -> Tuple[float, float]:
    """Return (mean, sem). SEM = std/sqrt(n)."""
    if not seed_vals:
        return float("nan"), 0.0
    n = len(seed_vals)
    std = float(np.std(seed_vals, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return float(np.mean(seed_vals)), sem


# ---------- plot ----------
def draw_panel(ax, agg: Dict[Tuple[str, int], Tuple[float, float]]):
    """agg keys: ('sequential', 1), ('naive', 4), ('naive', 8), ('s3', 4), ('s3', 8)."""
    seq_m, seq_e = agg.get(("sequential", 1), (np.nan, 0.0))

    def series(cond: str):
        means, sems = [], []
        for k in K_VALS:
            if k == 1:
                means.append(seq_m); sems.append(seq_e)
            else:
                m, e = agg.get((cond, k), (np.nan, 0.0))
                means.append(m); sems.append(e)
        return np.array(means), np.array(sems)

    nm, ne = series("naive")
    dm, de = series("s3")

    # Bands — half-width clipped to MIN_BAND_HW for visual consistency across panels.
    # Real SEM values are unchanged in data.csv; this is purely a display floor.
    MIN_BAND_HW = 0.008
    ne_disp = np.maximum(ne, MIN_BAND_HW)
    de_disp = np.maximum(de, MIN_BAND_HW)
    if not np.all(np.isnan(nm)):
        ax.fill_between(K_VALS, nm - ne_disp, nm + ne_disp,
                        color=NAIVE_BAR, alpha=0.18, linewidth=0, zorder=2)
    if not np.all(np.isnan(dm)):
        ax.fill_between(K_VALS, dm - de_disp, dm + de_disp,
                        color=S3_BAR, alpha=0.18, linewidth=0, zorder=2)

    # Lines
    ax.plot(K_VALS, nm, color=NAIVE_BAR, linewidth=1.8, zorder=3)
    ax.plot(K_VALS, dm, color=S3_BAR,    linewidth=1.8, zorder=3)

    # Markers (k>=4 only; lines emerge from common k=1 point with no marker)
    for ki, k in enumerate(K_VALS):
        if k == 1:
            continue
        is_extrap = (k == 8)
        # Naive
        if np.isfinite(nm[ki]):
            ax.scatter([k], [nm[ki]],
                       facecolor=("white" if is_extrap else NAIVE_BAR),
                       edgecolor=("black" if not is_extrap else NAIVE_BAR),
                       linewidth=(0.8 if not is_extrap else 1.6),
                       marker="o", s=(48 if not is_extrap else 52),
                       zorder=4)
        # S^3
        if np.isfinite(dm[ki]):
            ax.scatter([k], [dm[ki]],
                       facecolor=("white" if is_extrap else S3_BAR),
                       edgecolor=("black" if not is_extrap else S3_BAR),
                       linewidth=(0.8 if not is_extrap else 1.6),
                       marker="D", s=(46 if not is_extrap else 50),
                       zorder=4)

    # No marker at k=1 — both lines emerge from the common sequential point;
    # the x-axis tick at 1 conveys the baseline.

    # X axis
    ax.set_xticks(K_VALS)
    ax.set_xticklabels([str(k) for k in K_VALS], fontsize=11)
    ax.set_xlim(0.5, max(K_VALS) + 0.7)

    # Auto y range — per-panel, with a bit of headroom.
    # Use the *displayed* (clipped) band widths so the band fits inside the panel.
    all_vals = []
    for arr_m, arr_e in [(nm, ne_disp), (dm, de_disp)]:
        for v, e in zip(arr_m, arr_e):
            if np.isfinite(v):
                all_vals.append(v + e)
                all_vals.append(v - e)
    if np.isfinite(seq_m):
        seq_e_disp = max(seq_e, MIN_BAND_HW)
        all_vals.append(seq_m + seq_e_disp); all_vals.append(seq_m - seq_e_disp)
    if all_vals:
        y_min_data = min(all_vals)
        y_max_data = max(all_vals)
        rng = y_max_data - y_min_data

        # Tight zoom so the Naive vs S^3 gap fills the panel height
        if rng < 0.05:    step = 0.02
        elif rng < 0.15:  step = 0.05
        elif rng < 0.30:  step = 0.05
        else:             step = 0.10

        y_bot = max(0.0, step * np.floor((y_min_data - step * 0.6) / step))
        y_top = step * np.ceil((y_max_data + step * 0.6) / step)
        if y_top - y_bot < 2 * step:
            y_top = y_bot + 2 * step  # ensure at least 2 ticks of room
        ax.set_ylim(y_bot, y_top)

        ticks = np.arange(y_bot, y_top + 1e-9, step)
        if len(ticks) > 5:
            ticks = ticks[::2]
        ax.set_yticks(ticks)
    else:
        ax.set_ylim(0, 1)

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(0.8)


def render_grid(panel: dict) -> List:
    apply_paper_style()
    nrows = len(panel["models"])
    ncols = len(panel["datasets"])
    fig, axes = plt.subplots(nrows, ncols, figsize=panel["figsize"],
                             sharex=True, sharey=False)
    if nrows == 1: axes = np.array([axes])
    if ncols == 1: axes = axes.reshape(-1, 1)

    missing = []
    for ri, (m_key, m_lbl) in enumerate(panel["models"]):
        for ci, (d_key, d_lbl) in enumerate(panel["datasets"]):
            ax = axes[ri][ci]
            seed_vals = cell_passk(panel["root"], m_key, d_key)
            agg_vals = {k: aggregate(v) for k, v in seed_vals.items()}
            # Need at least seq + one of {naive_4, s3_4}; otherwise hide panel
            if not seed_vals.get(("sequential", 1)) or \
               (not seed_vals.get(("naive", 4)) and not seed_vals.get(("s3", 4))):
                missing.append((panel["name"], m_key, d_key))
                ax.set_visible(False)
                continue
            draw_panel(ax, agg_vals)

            if ri == 0:
                ax.set_title(d_lbl, fontsize=13, pad=6)
            if ci == 0:
                ax.text(-0.30, 0.5, m_lbl, transform=ax.transAxes,
                        ha="right", va="center", fontsize=12,
                        rotation=0, color=AXIS)

    # Shared legend at top
    handles = [
        plt.Line2D([], [], color=NAIVE_BAR, marker="o", markersize=6,
                   markeredgecolor="black", markeredgewidth=0.6, linewidth=1.3,
                   label="Standard"),
        plt.Line2D([], [], color=S3_BAR, marker="D", markersize=6,
                   markeredgecolor="black", markeredgewidth=0.6, linewidth=1.3,
                   label="Ours"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.998),
               handlelength=1.8, handletextpad=0.5, columnspacing=2.8)

    # Single shared axis labels
    fig.supxlabel(r"$k$", fontsize=14, y=0.012)
    fig.supylabel(r"pass@$k$", fontsize=14, x=0.005)

    fig.tight_layout(rect=(0.03, 0.03, 1, 0.965))
    fig.subplots_adjust(hspace=0.22, wspace=0.20)

    save_fig(fig, panel["name"], subdir="appfig2")
    plt.close(fig)
    return missing


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_missing = []
    for panel in [PANEL_A, PANEL_B]:
        miss = render_grid(panel)
        all_missing.extend(miss)
        print(f"Wrote: {OUT_DIR}/{panel['name']}.{{pdf,png}}")

    if all_missing:
        print("\nMISSING cells (panel hidden):")
        for fname, m, d in all_missing:
            print(f"  {fname}: {m} / {d}")
    else:
        print("\nNo missing cells.")


if __name__ == "__main__":
    main()
