#!/usr/bin/env python3
"""
Figure 1 plots for the paper:
  Left  — Oracle accuracy: sequential < naive_parallel < diversity_parallel
  Right — Anchor collapse: per-turn across-thread query diversity (ATC),
          paired bars naive vs diversity for turns 1-5

Usage (from general_agent/):
  python -m webwalkerqa.plot.figure1_plots \
      --results-dir /home/ssmurali/t3-testbed/results/figure1 \
      --figures-dir figures/figure1

Outputs:
  figures/figure1/fig1_oracle_accuracy.{pdf,png}
  figures/figure1/fig1_anchor_collapse.{pdf,png}
  figures/figure1/fig1_combined.{pdf,png}
  figures/figure1/atc_per_turn.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_GA_DIR = Path(__file__).resolve().parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tokens(q: str) -> List[str]:
    return (q or "").lower().split()


def jaccard_distance(q_i: str, q_j: str) -> float:
    """1 - Jaccard similarity (word-set overlap)."""
    a = set(_tokens(q_i))
    b = set(_tokens(q_j))
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union > 0 else 0.0


def edit_distance_norm(q_i: str, q_j: str) -> float:
    """Normalised token-edit-distance (Levenshtein on word lists / max len)."""
    try:
        import editdistance
        ti, tj = _tokens(q_i), _tokens(q_j)
        mx = max(len(ti), len(tj))
        if mx == 0:
            return 0.0
        return float(editdistance.eval(ti, tj)) / mx
    except ImportError:
        # Fallback: jaccard distance is fine
        return jaccard_distance(q_i, q_j)


def mean_pairwise(queries: List[str], metric_fn) -> Optional[float]:
    """Mean pairwise distance over k*(k-1)/2 pairs."""
    valid = [q for q in queries if q and q.strip()]
    if len(valid) < 2:
        return None
    dists = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            d = metric_fn(valid[i], valid[j])
            if not np.isnan(d):
                dists.append(d)
    return float(np.mean(dists)) if dists else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(traj_dir: Path) -> List[dict]:
    data = []
    for p in sorted(traj_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data.append(json.load(f))
    return data


def extract_turn_queries(item: dict, turn: int) -> List[str]:
    """Return the query each thread used at `turn` (0 = missing)."""
    queries = []
    for thread in item.get("threads", []):
        tl = thread.get("turn_logs") or []
        q = next((t.get("query", "") for t in tl if t.get("turn") == turn), "")
        queries.append(q or "")
    return queries


def compute_atc_per_turn(
    items: List[dict],
    turns: List[int],
    metric_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Across-Thread Diversity at each turn.
    Returns (mean_per_turn, sem_per_turn) arrays aligned to `turns`.
    """
    by_turn: Dict[int, List[float]] = {t: [] for t in turns}
    for item in items:
        for t in turns:
            qs = extract_turn_queries(item, t)
            v = mean_pairwise(qs, metric_fn)
            if v is not None:
                by_turn[t].append(v)

    means, sems = [], []
    for t in turns:
        vals = np.asarray(by_turn[t], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            n = len(vals)
            sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            sems.append(sem)
    return np.array(means), np.array(sems)


# ---------------------------------------------------------------------------
# Plot 1 — Oracle accuracy bar chart
# ---------------------------------------------------------------------------

def plot_oracle_accuracy(
    seq_acc: float,
    naive_oracle: float,
    div_oracle: float,
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    conditions = ["Sequential\n(k=1, T=20)", "Naive Parallel\n(k=4, T=5)", "Diversity Parallel\n(k=4, T=5)"]
    values     = [seq_acc, naive_oracle, div_oracle]
    colors     = ["#95a5a6", "#e67e22", "#2ecc71"]

    bars = ax.bar(conditions, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.006,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Annotations: arrows showing gain
    y_annot = max(values) + 0.055
    ax.annotate(
        "", xy=(1, naive_oracle + 0.015), xytext=(0, seq_acc + 0.015),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.4),
    )
    ax.text(0.5, (seq_acc + naive_oracle) / 2 + 0.03, "width\nscaling", ha="center",
            fontsize=8, color="#7f8c8d")
    ax.annotate(
        "", xy=(2, div_oracle + 0.015), xytext=(1, naive_oracle + 0.015),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.4),
    )
    ax.text(1.5, (naive_oracle + div_oracle) / 2 + 0.03, "diversity\nboost", ha="center",
            fontsize=8, color="#27ae60")

    ax.set_ylabel("Oracle Pass@k  (any thread correct)", fontsize=11)
    ax.set_ylim(0, max(values) + 0.12)
    ax.set_title("GAIA-103 — Compute-Matched Budget (20 turns)", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig1_oracle_accuracy.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig1_oracle_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig1_oracle_accuracy.{{pdf,png}} → {figures_dir}")


# ---------------------------------------------------------------------------
# Plot 2 — Anchor collapse paired bar chart
# ---------------------------------------------------------------------------

def plot_anchor_collapse(
    turns: List[int],
    naive_mean: np.ndarray,
    naive_sem: np.ndarray,
    div_mean: np.ndarray,
    div_sem: np.ndarray,
    metric_name: str,
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    x = np.arange(len(turns), dtype=float)
    width = 0.35

    bars_naive = ax.bar(
        x - width / 2, naive_mean, width,
        yerr=naive_sem, capsize=4,
        color="#e67e22", alpha=0.85, label="Naive Parallel",
        error_kw={"elinewidth": 1.2, "ecolor": "#7f8c8d"},
    )
    bars_div = ax.bar(
        x + width / 2, div_mean, width,
        yerr=div_sem, capsize=4,
        color="#2ecc71", alpha=0.85, label="Diversity Parallel",
        error_kw={"elinewidth": 1.2, "ecolor": "#27ae60"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Turn {t}" for t in turns], fontsize=10)
    ax.set_xlabel("Search Turn", fontsize=11)
    ax.set_ylabel(f"Mean pairwise {metric_name}\n(across k=4 threads)", fontsize=11)
    ax.set_title(
        "Anchor Collapse — Per-Turn Query Diversity\n"
        "Naive threads converge; diversity threads stay spread",
        fontsize=11, pad=8,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade region below to highlight collapse
    ax.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=0.8, label="_nolegend_")

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig1_anchor_collapse.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig1_anchor_collapse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig1_anchor_collapse.{{pdf,png}} → {figures_dir}")


# ---------------------------------------------------------------------------
# Plot 3 — Combined 1×2 figure
# ---------------------------------------------------------------------------

def plot_combined(
    seq_acc: float,
    naive_oracle: float,
    div_oracle: float,
    turns: List[int],
    naive_mean: np.ndarray,
    naive_sem: np.ndarray,
    div_mean: np.ndarray,
    div_sem: np.ndarray,
    metric_name: str,
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: oracle accuracy ---
    conditions = ["Sequential\n(k=1, T=20)", "Naive Parallel\n(k=4, T=5)", "Diversity Parallel\n(k=4, T=5)"]
    values     = [seq_acc, naive_oracle, div_oracle]
    colors     = ["#95a5a6", "#e67e22", "#2ecc71"]

    bars = ax_left.bar(conditions, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, val in zip(bars, values):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.006,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_left.annotate(
        "", xy=(1, naive_oracle + 0.012), xytext=(0, seq_acc + 0.012),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.3),
    )
    ax_left.text(0.5, (seq_acc + naive_oracle) / 2 + 0.025, "width\nscaling",
                 ha="center", fontsize=8, color="#7f8c8d")
    ax_left.annotate(
        "", xy=(2, div_oracle + 0.012), xytext=(1, naive_oracle + 0.012),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.3),
    )
    ax_left.text(1.5, (naive_oracle + div_oracle) / 2 + 0.025, "diversity\nboost",
                 ha="center", fontsize=8, color="#27ae60")
    ax_left.set_ylabel("Oracle Pass@k", fontsize=11)
    ax_left.set_ylim(0, max(values) + 0.12)
    ax_left.set_title("(a)  Oracle Accuracy", fontsize=12, pad=8)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(axis="x", labelsize=9)

    # --- Right: anchor collapse ---
    x = np.arange(len(turns), dtype=float)
    width = 0.35
    ax_right.bar(
        x - width / 2, naive_mean, width,
        yerr=naive_sem, capsize=4,
        color="#e67e22", alpha=0.85, label="Naive Parallel",
        error_kw={"elinewidth": 1.2, "ecolor": "#7f8c8d"},
    )
    ax_right.bar(
        x + width / 2, div_mean, width,
        yerr=div_sem, capsize=4,
        color="#2ecc71", alpha=0.85, label="Diversity Parallel",
        error_kw={"elinewidth": 1.2, "ecolor": "#27ae60"},
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"Turn {t}" for t in turns], fontsize=10)
    ax_right.set_xlabel("Search Turn", fontsize=11)
    ax_right.set_ylabel(f"Pairwise {metric_name}\n(k=4 threads)", fontsize=11)
    ax_right.set_title("(b)  Anchor Collapse — Query Diversity per Turn", fontsize=12, pad=8)
    ax_right.set_ylim(0, 1.05)
    ax_right.legend(fontsize=10, frameon=False)
    ax_right.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=0.8)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle("GAIA-103  |  gpt-4o-mini  |  Budget = 20 turns (compute-matched)", fontsize=11, y=1.02)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig1_combined.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig1_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig1_combined.{{pdf,png}} → {figures_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_GA_DIR.parent / "results" / "figure1"))
    ap.add_argument("--figures-dir", default=str(_GA_DIR.parent / "figures" / "figure1"))
    ap.add_argument("--metric", choices=["jaccard", "edit"], default="jaccard",
                    help="Distance metric for anchor collapse plot (default: jaccard)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_fn   = jaccard_distance if args.metric == "jaccard" else edit_distance_norm
    metric_name = "Jaccard distance" if args.metric == "jaccard" else "token-edit distance"

    # ---- Accuracy numbers from CSVs ----
    def _read_accuracy(csv_path: Path, correct_field: str = "correct") -> float:
        rows = list(csv.DictReader(open(csv_path)))
        correct = sum(1 for r in rows if str(r.get(correct_field, "")).strip() == "1")
        return correct / len(rows) if rows else 0.0

    seq_acc     = _read_accuracy(results_dir / "sequential.csv",         "correct")
    naive_oracle = _read_accuracy(results_dir / "naive_parallel.csv",    "oracle_correct")
    div_oracle   = _read_accuracy(results_dir / "diversity_parallel.csv", "oracle_correct")

    print(f"Sequential  acc   = {seq_acc:.3f}")
    print(f"Naive       oracle= {naive_oracle:.3f}")
    print(f"Diversity   oracle= {div_oracle:.3f}")

    # ---- Per-turn ATC ----
    turns = [1, 2, 3, 4, 5]

    naive_items = load_trajectories(results_dir / "trajectories" / "naive_parallel")
    div_items   = load_trajectories(results_dir / "trajectories" / "diversity_parallel")

    print(f"Loaded {len(naive_items)} naive trajectories, {len(div_items)} diversity trajectories")

    naive_mean, naive_sem = compute_atc_per_turn(naive_items, turns, metric_fn)
    div_mean,   div_sem   = compute_atc_per_turn(div_items,   turns, metric_fn)

    print(f"\nNaive ATC ({args.metric}):")
    for t, m, s in zip(turns, naive_mean, naive_sem):
        print(f"  turn {t}: {m:.3f} ± {s:.3f}")
    print(f"\nDiversity ATC ({args.metric}):")
    for t, m, s in zip(turns, div_mean, div_sem):
        print(f"  turn {t}: {m:.3f} ± {s:.3f}")

    # Save CSV
    csv_path = figures_dir / "atc_per_turn.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn", "naive_mean", "naive_sem", "div_mean", "div_sem"])
        for t, nm, ns, dm, ds in zip(turns, naive_mean, naive_sem, div_mean, div_sem):
            w.writerow([t, f"{nm:.4f}", f"{ns:.4f}", f"{dm:.4f}", f"{ds:.4f}"])
    print(f"\nWrote {csv_path}")

    # ---- Plots ----
    plot_oracle_accuracy(seq_acc, naive_oracle, div_oracle, figures_dir)
    plot_anchor_collapse(turns, naive_mean, naive_sem, div_mean, div_sem, metric_name, figures_dir)
    plot_combined(seq_acc, naive_oracle, div_oracle,
                  turns, naive_mean, naive_sem, div_mean, div_sem,
                  metric_name, figures_dir)


if __name__ == "__main__":
    main()
