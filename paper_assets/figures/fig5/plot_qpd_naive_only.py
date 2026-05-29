"""
Fig 2 — clean naive-only QPD distribution.

Single KDE curve (Naive, coral) with median tick. No legend (single condition).

Outputs:
  paper_assets/figures/fig5/qpd_naive_only.{pdf,png}
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, AXIS, GRID, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

DATA = TESTBED_ROOT / "results" / "support_data" / "qpd_distributions.csv"


def kde(values, x, bw=0.35):
    arr = np.asarray(values, dtype=float)
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0:
        return np.zeros_like(x)
    h = bw * sigma
    diff = (x[:, None] - arr[None, :]) / h
    kernel = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (arr.size * h)


def main():
    vals = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["condition"] == "naive_parallel":
                v = r.get("jaccard_qpd", "")
                if v not in ("", None):
                    vals.append(float(v))

    apply_paper_style()
    median = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.linspace(0.0, 1.0, 600)
    y = kde(vals, x, bw=0.35)

    ax.fill_between(x, y, alpha=0.20, color=NAIVE, linewidth=0, zorder=2)
    ax.plot(x, y, color=NAIVE, linewidth=2.0, zorder=3)

    # Median dashed line — drawn full-height so the line stays clear of the curve.
    # Label placed near the baseline inside the filled region (out of the curve dip).
    ymax_data = float(np.max(kde(vals, x, bw=0.35)))
    ax.vlines(median, 0, ymax_data * 1.05, color="#6C757D",
              linestyle=(0, (4, 4)), linewidth=0.8, zorder=4)
    ax.text(median + 0.018, 0.12,
            f"median = {median:.2f}",
            fontsize=8, color="#6C757D", ha="left", va="bottom")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Query Pairwise Distance (QPD)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    # Save (override DPI to 300 for this single-column figure even though
    # save_fig defaults to 1400 — user asked specifically for 300).
    out_dir = Path(__file__).resolve().parent
    pdf = out_dir / "qpd_naive_only.pdf"
    png = out_dir / "qpd_naive_only.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"n_questions = {len(vals)}")
    print(f"min = {min(vals):.4f}  median = {median:.4f}  mean = {float(np.mean(vals)):.4f}  max = {max(vals):.4f}")
    print(f"Wrote: {pdf}")
    print(f"Wrote: {png}")


if __name__ == "__main__":
    main()
