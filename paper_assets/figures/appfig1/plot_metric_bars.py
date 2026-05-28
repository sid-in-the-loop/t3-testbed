"""
Appendix Figure 1 — per-(model, dataset) metric bar grid.

Two outputs:
  appfig1a_mhqa.{pdf,png}      5 models × 5 datasets   (~7" × 9")
  appfig1b_reasoning.{pdf,png} 4 models × 3 datasets   (~5" × 7")

Per-panel: 6 bars grouped 3×2 by metric:
  QPD-naive | QPD-S^3   |   ITC-naive | ITC-S^3   |   ATD-naive | ATD-S^3
  Bars: NAIVE / S3 colors. ±1 std error bars over 3 seeds.

QPD and ATD are distances (higher = more diverse, "↑ better" for the diversity story);
ITC is a similarity (lower = less stuck, "↓ better").
We annotate "↑ better" / "↓ better" only on the first row, top, to save space.

Data source: paper_assets/metrics_for_plots.csv  (k=4 condition only)

Usage:
  python paper_assets/figures/appfig1/plot_metric_bars.py
"""
from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, S3, AXIS, GRID, apply_paper_style, save_fig, PAPER_ASSETS  # noqa: E402

CSV_PATH = PAPER_ASSETS / "metrics_for_plots.csv"
OUT_DIR  = Path(__file__).resolve().parent

CONDS = ["naive_k4", "div_k4"]
METRICS = ["QPD", "ATC"]   # ITC dropped; ATC display label is "ATD"
METRIC_LABELS = {"QPD": "QPD", "ATC": "ATD"}
# Per-metric hatch — solid for QPD, dotted for ATD (color = naive vs S^3).
METRIC_HATCH = {"QPD": "", "ATC": ".."}

PANEL_A = {
    "name": "appfig1a_mhqa",
    "group": "mhqa",
    "models":   [("qwen3-1.7b", "qwen3-1.7B"), ("qwen3-4b", "qwen3-4B"),
                 ("qwen3-8b",   "qwen3-8B"),   ("gemma3-4b", "gemma3-4B"),
                 ("gemma3-12b", "gemma3-12B")],
    "datasets": [("2wikimultihopqa", "2Wiki"), ("bamboogle", "Bamboogle"),
                 ("frames", "Frames"), ("hotpotqa", "HotpotQA"),
                 ("musique", "MuSiQue")],
    "figsize": (13.0, 10.0),  # full-width — appendix has the room
}
PANEL_B = {
    "name": "appfig1b_reasoning",
    "group": "reasoning",
    "models":   [("qwen3-4b", "qwen3-4B"), ("qwen3-8b", "qwen3-8B"),
                 ("gemma3-12b", "gemma3-12B")],
    "datasets": [("gaia", "GAIA"), ("hle", "HLE"), ("webwalker", "WebWalker")],
    "figsize": (9.0, 7.0),
}

# Use the canonical CMU Core colors directly (no extra "punchier" variants needed —
# Carnegie Red and Iron Gray already read clearly at small panel size).
NAIVE_BAR = NAIVE   # Iron Gray
S3_BAR    = S3      # Carnegie Red — Ours


def load_csv() -> Dict[Tuple[str, str, str, str], List[Dict[str, float]]]:
    """{(group, model, dataset, condition): [{QPD, ITC, ATC} per seed]}."""
    out: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    with open(CSV_PATH) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for r in csv.DictReader(io.StringIO("".join(lines))):
        try:
            row = {m: float(r[m]) for m in METRICS}
        except (KeyError, ValueError):
            continue
        out[(r["group"], r["model"], r["dataset"], r["condition"])].append(row)
    return out


def cell_stats(seeds: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Return {metric: (mean, std)} over seeds."""
    out: Dict[str, Tuple[float, float]] = {}
    for m in METRICS:
        vals = [s[m] for s in seeds if m in s]
        if not vals:
            out[m] = (float("nan"), 0.0)
        else:
            n = len(vals)
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            out[m] = (float(np.mean(vals)), std)
    return out


def draw_panel(ax, naive_stats, s3_stats):
    """Draw 6 bars (3 metrics × {naive, S^3}).

    Bars with a real but near-zero mean (< MIN_VISIBLE_HEIGHT) are floored to
    that height so the reader can see the bar exists. Their actual value
    is preserved in the CSV; the visual floor is purely a presentation choice.
    """
    n_metrics = len(METRICS)
    x = np.arange(n_metrics, dtype=float)
    bar_w = 0.42

    MIN_VISIBLE_HEIGHT = 0.015  # bars with smaller means render at this height

    def display_height(mean: float) -> float:
        if not np.isfinite(mean):
            return 0.0
        if 0 < mean < MIN_VISIBLE_HEIGHT:
            return MIN_VISIBLE_HEIGHT
        # exact-zero stub too (so the slot is still visible as a "near-zero" marker)
        if mean == 0.0:
            return MIN_VISIBLE_HEIGHT
        return mean

    for i, m in enumerate(METRICS):
        n_mean, n_std = naive_stats[m]
        s_mean, s_std = s3_stats[m]
        hatch = METRIC_HATCH[m]

        # Naive bar
        ax.bar(x[i] - bar_w / 2, display_height(n_mean), width=bar_w,
               color=NAIVE_BAR, edgecolor="black", linewidth=0.8,
               hatch=hatch,
               yerr=n_std if n_mean >= MIN_VISIBLE_HEIGHT else None, capsize=2.0,
               error_kw={"elinewidth": 0.7, "ecolor": "#222222", "alpha": 0.9},
               zorder=3)
        # S^3 bar
        ax.bar(x[i] + bar_w / 2, display_height(s_mean), width=bar_w,
               color=S3_BAR, edgecolor="black", linewidth=0.8,
               hatch=hatch,
               yerr=s_std if s_mean >= MIN_VISIBLE_HEIGHT else None, capsize=2.0,
               error_kw={"elinewidth": 0.7, "ecolor": "#222222", "alpha": 0.9},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    ax.set_xlim(-0.55, n_metrics - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)

    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(0.8)


def render_grid(panel: dict, store: Dict) -> None:
    apply_paper_style()
    nrows = len(panel["models"])
    ncols = len(panel["datasets"])
    fig, axes = plt.subplots(nrows, ncols, figsize=panel["figsize"],
                             sharex=True, sharey=True)
    if nrows == 1: axes = np.array([axes])
    if ncols == 1: axes = axes.reshape(-1, 1)

    missing = []
    for ri, (m_key, m_lbl) in enumerate(panel["models"]):
        for ci, (d_key, d_lbl) in enumerate(panel["datasets"]):
            ax = axes[ri][ci]
            n_seeds = store.get((panel["group"], m_key, d_key, "naive_k4"), [])
            s_seeds = store.get((panel["group"], m_key, d_key, "div_k4"),   [])
            if not n_seeds or not s_seeds:
                missing.append((panel["group"], m_key, d_key))
                ax.set_visible(False)
                continue
            draw_panel(ax, cell_stats(n_seeds), cell_stats(s_seeds))

            # Column header — dataset name above the topmost row only
            if ri == 0:
                ax.set_title(d_lbl, fontsize=10, pad=4)
            # Row label — model name on the left of the leftmost column only
            if ci == 0:
                ax.text(-0.36, 0.5, m_lbl, transform=ax.transAxes,
                        ha="right", va="center", fontsize=10,
                        rotation=0, color=AXIS)

    # Shared legend at top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NAIVE_BAR, label="Standard"),
        plt.Rectangle((0, 0), 1, 1, color=S3_BAR,    label="Ours"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.995),
               handlelength=1.4, handletextpad=0.5, columnspacing=2.4)

    fig.tight_layout(rect=(0.06, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.32, wspace=0.16)

    save_fig(fig, panel["name"], subdir="appfig1")
    plt.close(fig)
    return missing


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = load_csv()
    print(f"Loaded {sum(len(v) for v in store.values())} per-seed rows from {CSV_PATH}")

    all_missing = []
    for panel in [PANEL_A, PANEL_B]:
        miss = render_grid(panel, store)
        all_missing.extend(miss)
        print(f"Wrote: {OUT_DIR}/{panel['name']}.{{pdf,png}}")

    if all_missing:
        print("\nMISSING cells (panel hidden):")
        for g, m, d in all_missing:
            print(f"  {g} / {m} / {d}")
    else:
        print("\nNo missing cells.")


if __name__ == "__main__":
    main()
