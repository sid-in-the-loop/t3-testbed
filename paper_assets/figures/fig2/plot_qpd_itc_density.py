"""
Figure 2 — Diversity-method comparison: per-question QPD and ITC distributions.

Two panels (saved separately so the paper can lay them out side-by-side):
  qpd_density  — turn-1 query pairwise distance distribution (left panel)
  itc_density  — intra-thread coherence distribution (right panel)

Source data:
  results/support_data/qpd_distributions.csv
  results/support_data/itc_distributions.csv

Outputs (paper_assets/figures/fig2/):
  qpd_density.{pdf,png}
  itc_density.{pdf,png}

Usage:
  python paper_assets/figures/fig2/plot_qpd_itc_density.py
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Shared style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, S3, AXIS, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

DATA_DIR = TESTBED_ROOT / "results" / "support_data"

PALETTE = {
    "naive_parallel": {"color": NAIVE, "label": "Standard"},
    "greedy_jaccard": {"color": S3,    "label": "Ours"},
}

LW          = 2.0
FILL_ALPHA  = 0.13
MEAN_LW     = 1.3
MEAN_ALPHA  = 0.7
FONT_LABEL  = 13
FONT_TICK   = 11
FONT_LEGEND = 11
FONT_ANNOT  = 10


def _kde(data: List[float], x: np.ndarray, bw: float) -> np.ndarray:
    """Pure-numpy Gaussian KDE matching scipy.stats.gaussian_kde(bw_method=bw).
    scipy uses h = bw_method * std(data, ddof=1) as the per-dim bandwidth.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size < 3:
        return np.zeros_like(x)
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0:
        return np.zeros_like(x)
    h = bw * sigma
    diff = (x[:, None] - arr[None, :]) / h
    kernel = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (arr.size * h)


def _load_csv_by_condition(path: Path, cond_col: str, val_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get(val_col, "")
            if v not in ("", None):
                out.setdefault(r[cond_col], []).append(float(v))
    return out


def plot_qpd() -> None:
    apply_paper_style()
    data = _load_csv_by_condition(DATA_DIR / "qpd_distributions.csv",
                                  "condition", "jaccard_qpd")

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    x = np.linspace(0.0, 1.0, 600)
    BW = 0.35

    order = ["naive_parallel", "greedy_jaccard"]
    for cond in order:
        vals = data.get(cond, [])
        if not vals:
            continue
        cfg = PALETTE[cond]
        y = _kde(vals, x, BW)
        ax.plot(x, y, color=cfg["color"], linewidth=LW, label=cfg["label"], zorder=3)
        ax.fill_between(x, y, alpha=FILL_ALPHA, color=cfg["color"], zorder=2)
        mu = float(np.mean(vals))
        y_at_mu = float(_kde(vals, np.array([mu]), BW)[0])
        ax.vlines(mu, 0, y_at_mu, color=cfg["color"], linewidth=MEAN_LW,
                  linestyle="--", alpha=MEAN_ALPHA, zorder=4)

    ax.annotate(
        "Standard sampling\nclusters queries",
        xy=(0.22, 1.50), xytext=(0.36, 0.38),
        fontsize=FONT_ANNOT, color="black",
        ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10),
    )
    ax.annotate(
        "Ours forces\nspread",
        xy=(0.86, 3.70), xytext=(0.55, 2.55),
        fontsize=FONT_ANNOT, color="black",
        ha="center", va="top",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Turn-1 Query Diversity (QPD)", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", frameon=True, handlelength=1.8,
                    borderpad=0.6, labelspacing=0.4, fontsize=FONT_LEGEND)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "qpd_density", subdir="fig2",
             also_dirs=[DATA_DIR])  # also write into results/support_data/ for backwards-compat
    plt.close(fig)
    print("  Wrote qpd_density.{pdf,png}")


def plot_itc() -> None:
    apply_paper_style()
    data = _load_csv_by_condition(DATA_DIR / "itc_distributions.csv",
                                  "condition", "itc_score")

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    x = np.linspace(0.0, 1.0, 600)
    BW = 0.30

    order = ["naive_parallel", "greedy_jaccard"]
    for cond in order:
        vals = data.get(cond, [])
        if not vals:
            continue
        cfg = PALETTE[cond]
        y = _kde(vals, x, BW)
        ax.plot(x, y, color=cfg["color"], linewidth=LW, label=cfg["label"], zorder=3)
        ax.fill_between(x, y, alpha=FILL_ALPHA, color=cfg["color"], zorder=2)
        mu = float(np.mean(vals))
        y_at_mu = float(_kde(vals, np.array([mu]), BW)[0])
        ax.vlines(mu, 0, y_at_mu, color=cfg["color"], linewidth=MEAN_LW,
                  linestyle="--", alpha=MEAN_ALPHA, zorder=4)

    ax.annotate(
        "Ours starts settle\non better paths",
        xy=(0.17, 1.98), xytext=(0.06, 1.25),
        fontsize=FONT_ANNOT, color="black",
        ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, mutation_scale=8),
    )
    ax.annotate(
        "Standard threads stay\nfixated on initial query",
        xy=(0.47, 1.43), xytext=(0.58, 0.85),
        fontsize=FONT_ANNOT, color="black",
        ha="left", va="top",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, mutation_scale=8),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Inter-Turn Coherence (ITC)", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=True, handlelength=1.6,
                    borderpad=0.5, labelspacing=0.35,
                    bbox_to_anchor=(1.0, 1.0), fontsize=FONT_LEGEND)
    leg.get_frame().set_linewidth(0.7)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "itc_density", subdir="fig2",
             also_dirs=[DATA_DIR])
    plt.close(fig)
    print("  Wrote itc_density.{pdf,png}")


def main() -> None:
    print("=== Figure 2 plots ===")
    print("Left panel (QPD):")
    plot_qpd()
    print("Right panel (ITC):")
    plot_itc()
    print("Done.")


if __name__ == "__main__":
    main()
