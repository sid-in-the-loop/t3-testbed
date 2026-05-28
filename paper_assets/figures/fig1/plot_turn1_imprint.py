"""
Figure 1 — Turn-1 query distance predicts turn-T query distance.

Model:    qwen3-8b
Dataset:  GAIA (103 questions)
Conditions: naive_k4, div_k4 (S^3)
Seeds:    1, 2, 3 (aggregated)

For each (question, seed): for every C(4,2)=6 pair of rollouts,
plot turn-1 Jaccard distance (x) vs turn-T Jaccard distance (y).

Outputs (paper_assets/figures/fig1/):
  data.csv               (long-form, one row per pair)
  turn1_imprint.pdf      (300 dpi, paper-quality)
  turn1_imprint.png      (preview)

Usage:
  python paper_assets/figures/fig1/plot_turn1_imprint.py
"""

from __future__ import annotations
import csv
import glob
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Shared style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import NAIVE, S3, AXIS, apply_paper_style, save_fig, TESTBED_ROOT  # noqa: E402

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- config ----------
RESULTS_ROOT = TESTBED_ROOT / "results" / "main_table_web_serper" / "qwen3-8b" / "gaia"
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "data.csv"

CONDITIONS = [
    ("naive_k4", "naive_parallel_T8",     "Standard", NAIVE),
    ("div_k4",   "diversity_parallel_T8", "Ours", S3),
]
SEEDS = [1, 2, 3]
STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearsonr(rx, ry)


# ---------- distance ----------
def _tokens(s: str) -> set:
    if not s:
        return set()
    return {w for w in s.lower().split() if w and w not in STOPWORDS}


def jaccard_distance(a: str, b: str) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A and not B:
        return 0.0
    return 1.0 - (len(A & B) / max(1, len(A | B)))


# ---------- extraction ----------
def extract_first_last_queries(thread: dict) -> Tuple[str, str] | None:
    queries = []
    for tl in thread.get("turn_logs", []):
        q = tl.get("query")
        if q:
            queries.append((tl.get("turn", -1), str(q).strip()))
    if len(queries) < 2:
        return None
    queries.sort(key=lambda x: x[0])
    if queries[0][0] == queries[-1][0]:
        return None
    return queries[0][1], queries[-1][1]


def gather_pairs(condition_key: str, traj_subdir: str) -> List[dict]:
    rows = []
    for seed in SEEDS:
        glob_pat = str(RESULTS_ROOT / condition_key / f"run_{seed}" / "trajectories" / traj_subdir / "*.json")
        for fpath in sorted(glob.glob(glob_pat)):
            try:
                d = json.load(open(fpath))
            except Exception:
                continue
            qid = d.get("question_id", Path(fpath).stem)
            threads = d.get("threads", [])
            extracted = []
            for idx, thread in enumerate(threads):
                pair = extract_first_last_queries(thread)
                if pair is not None:
                    extracted.append((idx, pair[0], pair[1]))
            for (i, q1_i, qT_i), (j, q1_j, qT_j) in combinations(extracted, 2):
                d1 = jaccard_distance(q1_i, q1_j)
                dT = jaccard_distance(qT_i, qT_j)
                rows.append({
                    "condition": condition_key,
                    "question_id": qid,
                    "seed": seed,
                    "rollout_i": i,
                    "rollout_j": j,
                    "d1": f"{d1:.6f}",
                    "dT": f"{dT:.6f}",
                })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["condition", "question_id", "seed", "rollout_i", "rollout_j", "d1", "dT"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- plot helpers ----------
def _cov_ellipse(x: np.ndarray, y: np.ndarray, n_std: float = 1.0) -> tuple:
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width  = 2.0 * n_std * np.sqrt(max(vals[0], 0))
    height = 2.0 * n_std * np.sqrt(max(vals[1], 0))
    angle  = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return float(width), float(height), angle


def render_plot(per_cond: dict) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=300)

    # Faint gridlines
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(tick, color="#EEEEEE", linewidth=0.5, zorder=0)
        ax.axvline(tick, color="#EEEEEE", linewidth=0.5, zorder=0)

    # y=x reference
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), color="#999999",
            linewidth=1.0, zorder=1, label="_y=x")

    # Stats + means
    for key, info in per_cond.items():
        x, y = info["x"], info["y"]
        info["pearson"]  = pearsonr(x, y)
        info["spearman"] = spearmanr(x, y)
        info["mean_x"] = float(np.mean(x)) if len(x) else 0.0
        info["mean_y"] = float(np.mean(y)) if len(y) else 0.0

    # Ghost scatter
    for key, info in per_cond.items():
        ax.scatter(info["x"], info["y"], color=info["color"], marker="o",
                   s=4, alpha=0.08, edgecolors="none", zorder=2)

    # 0.75-σ covariance ellipses
    SIGMA = 0.75
    for key, info in per_cond.items():
        w, h, ang = _cov_ellipse(info["x"], info["y"], n_std=SIGMA)
        ax.add_patch(Ellipse(xy=(info["mean_x"], info["mean_y"]),
                             width=w, height=h, angle=ang,
                             facecolor=info["color"], alpha=0.22,
                             edgecolor="none", zorder=4))
        ax.add_patch(Ellipse(xy=(info["mean_x"], info["mean_y"]),
                             width=w, height=h, angle=ang,
                             facecolor="none", edgecolor=info["color"],
                             linewidth=2.0, zorder=5))

    # Cloud-mean diamond markers
    for key, info in per_cond.items():
        ax.scatter([info["mean_x"]], [info["mean_y"]], color=info["color"],
                   marker="D", s=110, edgecolors="white", linewidths=1.5,
                   zorder=6, label=info["label"])

    # Mean-shift arrow naive → S^3
    if "naive_k4" in per_cond and "div_k4" in per_cond:
        n, d = per_cond["naive_k4"], per_cond["div_k4"]
        ax.annotate("",
                    xy=(d["mean_x"], d["mean_y"]),
                    xytext=(n["mean_x"], n["mean_y"]),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    linewidth=1.2, shrinkA=8, shrinkB=8,
                                    alpha=0.7),
                    zorder=7)

    # ρ annotations top-left
    annot_y = 0.96
    for key, info in per_cond.items():
        ax.annotate(f"ρ = {info['pearson']:.2f}", xy=(0.04, annot_y),
                    xycoords="axes fraction", color=info["color"], fontsize=12,
                    ha="left", va="top", weight="bold")
        annot_y -= 0.065

    # Spines
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(AXIS)
        ax.spines[spine_name].set_linewidth(1.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Turn-1 Query Diversity (QPD)", fontsize=13)
    ax.set_ylabel("Turn-T Query Diversity", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    # Legend
    handles = []
    for key, info in per_cond.items():
        h = plt.Line2D([], [], marker="D", linestyle="",
                       markerfacecolor=info["color"],
                       markeredgecolor="white", markeredgewidth=1.0,
                       markersize=8, label=info["label"])
        handles.append(h)
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7,
                    handletextpad=0.25, borderaxespad=0.4, labelspacing=0.25)
    for h in leg.legend_handles:
        h.set_markersize(5)

    plt.tight_layout(pad=0.5)
    out_dir = Path(__file__).resolve().parent
    fig.savefig(out_dir / "turn1_imprint.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "turn1_imprint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_cond: dict = {}
    for cond_key, traj_subdir, label, color in CONDITIONS:
        rows = gather_pairs(cond_key, traj_subdir)
        all_rows.extend(rows)
        x = np.array([float(r["d1"]) for r in rows])
        y = np.array([float(r["dT"]) for r in rows])
        per_cond[cond_key] = {"x": x, "y": y, "label": label, "color": color}

    write_csv(all_rows, CSV_PATH)
    render_plot(per_cond)

    print(f"\nWrote CSV: {CSV_PATH}  ({len(all_rows)} rows)")
    print(f"Wrote PDF/PNG: {OUT_DIR}/turn1_imprint.{{pdf,png}}\n")
    print(f"{'condition':<12} {'pairs':>7} {'pearson':>9} {'spearman':>9}")
    for cond_key, _, label, _ in CONDITIONS:
        info = per_cond[cond_key]
        print(f"{cond_key:<12} {len(info['x']):>7} {info['pearson']:>9.4f} {info['spearman']:>9.4f}")


if __name__ == "__main__":
    main()
