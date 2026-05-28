"""
Figure 4 — S^3 vs Naive gain (Δpp) across qwen3 model sizes.

Single family (qwen3) only — apples-to-apples scaling.
Two panels:
  (a) MHQA (ClueWeb):       qwen3-1.7B, qwen3-4B, qwen3-8B  ×  {hotpotqa, musique, 2wiki, bamboogle, frames}
  (b) Reasoning (Serper):   qwen3-4B,   qwen3-8B            ×  {gaia, hle, webwalker}

Per (model, dataset) cell:
  - 3 seeds of naive_k4 (or legacy 'naive_parallel') pass_at_4_llm
  - 3 seeds of div_k4   (or legacy 'diversity_parallel') pass_at_4_llm
  - per-seed Δ = div_seed - naive_seed   (paired, since seeds match)
  - mean Δpp = 100 × mean(Δ)
  - error  = 100 × SEM(Δ)        (SEM over 3 seeds)

Outputs (paper_assets/figures/fig4/):
  data.csv             one row per (panel, model, dataset, seed) with Δpp
  gain_vs_size.pdf     two-panel bar plot, ~7" × 3"
  gain_vs_size.png     1400 dpi preview

Usage:
  python paper_assets/figures/fig4/plot_gain_vs_size.py
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Shared style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import S3, AXIS, REF, GRID, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

ROOT = TESTBED_ROOT / "results"
SEEDS = [1, 2, 3]
OUT_DIR = Path(__file__).resolve().parent
CSV_OUT = OUT_DIR / "data.csv"

# 3-step Carnegie Red gradient for {1.7B, 4B, 8B} — CMU Core
COLOR_BY_SIZE = {
    "1.7B": "#F4D5DA",   # light Carnegie tint
    "4B":   "#C41230",   # Carnegie Red
    "8B":   "#5C0A14",   # deep Carnegie
}

PANEL_A = {
    "title": "",  # caption-only convention
    "root":  "main_table_clueweb_t8",
    "models": [("qwen3-1.7b", "1.7B"), ("qwen3-4b", "4B"), ("qwen3-8b", "8B")],
    "datasets": [
        ("hotpotqa",        "HotpotQA"),
        ("musique",         "MuSiQue"),
        ("2wikimultihopqa", "2Wiki"),
        ("bamboogle",       "Bamboogle"),
        ("frames",          "Frames"),
    ],
}


def read_passk(run_dir: Path, col: str = "pass_at_4_llm") -> Optional[float]:
    sm = list(run_dir.glob("summary_T*.csv"))
    if not sm:
        return None
    with open(sm[0]) as f:
        r = next(csv.DictReader(f))
    try:
        return float(r[col])
    except (KeyError, ValueError):
        return None


def cell_paired_seeds(root_subdir: str, model: str, dataset: str
                      ) -> Tuple[List[float], List[float], List[float]]:
    """Return (naive_seeds, div_seeds, paired_deltas) — only seeds present in both."""
    base = ROOT / root_subdir / model / dataset
    n_seeds: Dict[int, float] = {}
    d_seeds: Dict[int, float] = {}
    for s in SEEDS:
        nv = read_passk(base / f"naive_k4/run_{s}")
        if nv is None:
            nv = read_passk(base / f"naive_parallel/run_{s}")
        dv = read_passk(base / f"div_k4/run_{s}")
        if dv is None:
            dv = read_passk(base / f"diversity_parallel/run_{s}")
        if nv is not None:
            n_seeds[s] = nv
        if dv is not None:
            d_seeds[s] = dv
    paired = sorted(set(n_seeds) & set(d_seeds))
    nv_arr = [n_seeds[s] for s in paired]
    dv_arr = [d_seeds[s] for s in paired]
    delta = [d_seeds[s] - n_seeds[s] for s in paired]
    return nv_arr, dv_arr, delta


# ---------- collect ----------
def collect() -> List[dict]:
    rows: List[dict] = []
    for panel_key, panel in [("a", PANEL_A)]:
        for model, size_label in panel["models"]:
            for dskey, dslabel in panel["datasets"]:
                _, _, deltas = cell_paired_seeds(panel["root"], model, dskey)
                for i, d in enumerate(deltas):
                    rows.append({
                        "panel": panel_key,
                        "panel_title": panel["title"],
                        "model": model,
                        "size_label": size_label,
                        "dataset_key": dskey,
                        "dataset_label": dslabel,
                        "seed_idx": i + 1,
                        "delta_pp": d * 100,
                    })
    return rows


def aggregate(rows: List[dict]) -> Dict[Tuple[str, str, str], Tuple[float, float, int]]:
    """{(panel, model, dataset_key): (mean_pp, sem_pp, n)}."""
    by: Dict[Tuple[str, str, str], List[float]] = {}
    for r in rows:
        key = (r["panel"], r["model"], r["dataset_key"])
        by.setdefault(key, []).append(r["delta_pp"])
    out = {}
    for k, vs in by.items():
        n = len(vs)
        std = float(np.std(vs, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        out[k] = (float(np.mean(vs)), sem, n)
    return out


# ---------- plot ----------
def render(agg: Dict[Tuple[str, str, str], Tuple[float, float, int]]) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    panel = PANEL_A
    panel_key = "a"
    n_models = len(panel["models"])
    n_dsets  = len(panel["datasets"])

    # Wider per-group spacing so bars breathe
    GROUP_SPACING = 1.6     # distance between dataset groups
    GROUP_WIDTH   = 1.0     # total group width
    x = np.arange(n_dsets, dtype=float) * GROUP_SPACING
    bar_w = GROUP_WIDTH / n_models

    all_vals = []
    for mi, (model, size_label) in enumerate(panel["models"]):
        color = COLOR_BY_SIZE[size_label]
        offset = (mi - (n_models - 1) / 2.0) * bar_w
        heights, errors = [], []
        for dskey, _ in panel["datasets"]:
            m, sem, n = agg.get((panel_key, model, dskey), (np.nan, 0.0, 0))
            heights.append(m); errors.append(sem)
            all_vals.append(m + sem); all_vals.append(m - sem)
        ax.bar(x + offset, heights, width=bar_w * 0.92,
               color=color, edgecolor="white", linewidth=0.6,
               yerr=errors, capsize=2.5,
               error_kw={"elinewidth": 0.7, "ecolor": AXIS, "alpha": 0.7},
               label=f"qwen3-{size_label}", zorder=3)

    # zero reference
    ax.axhline(0, color=REF, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)

    # X axis spacing — give datasets breathing room
    ax.set_xticks(x)
    ax.set_xticklabels([dl for _, dl in panel["datasets"]], fontsize=10)
    ax.set_xlim(x[0] - GROUP_WIDTH * 0.7, x[-1] + GROUP_WIDTH * 0.7)

    # Y axis — just "Δ"
    y_top = max(13.0, max(all_vals) + 1.5)
    y_bot = min(-2.0, min(all_vals) - 0.8)
    ax.set_ylim(y_bot, y_top)
    ax.set_ylabel(r"$\Delta$", fontsize=12)

    # Horizontal grid
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(1.0)

    # Legend below
    ax.legend(
        loc="lower center", ncol=n_models, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.32), handlelength=1.2, handletextpad=0.4,
        columnspacing=1.6,
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, "gain_vs_size", subdir="fig4")
    plt.close(fig)


# ---------- write csv ----------
def write_data_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "panel_title", "model", "size_label",
            "dataset_key", "dataset_label", "seed_idx", "delta_pp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {**r}
            row["delta_pp"] = f"{r['delta_pp']:.4f}"
            w.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_data_csv(rows, CSV_OUT)
    agg = aggregate(rows)

    # Print
    print("\nFig 4 cell table (paired seeds; Δpp = 100*(div - naive))\n")
    for panel_key, panel in [("a", PANEL_A)]:
        print(f"-- Panel ({panel_key})  {panel['title']} --")
        print(f"  {'model':<12} {'dataset':<14} {'mean Δpp':>10} {'SEM':>7} {'n_seeds':>8}")
        for model, size_label in panel["models"]:
            for dskey, dslabel in panel["datasets"]:
                m, sem, n = agg.get((panel_key, model, dskey), (float("nan"), 0.0, 0))
                print(f"  {model:<12} {dslabel:<14} {m:>+10.2f} {sem:>7.2f} {n:>8d}")
        # Per-model mean across datasets
        print(f"  -- per-model means across datasets:")
        for model, size_label in panel["models"]:
            means = [agg[(panel_key, model, dskey)][0]
                     for dskey, _ in panel["datasets"] if (panel_key, model, dskey) in agg]
            if means:
                print(f"    {model:<12} mean Δpp = {np.mean(means):+.2f}  median = {np.median(means):+.2f}")
        print()

    print(f"Wrote: {CSV_OUT}")
    render(agg)
    print(f"Wrote: {OUT_DIR}/gain_vs_size.{{pdf,png}}")


if __name__ == "__main__":
    main()
