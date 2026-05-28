#!/usr/bin/env python3
"""
Figure 1 (depth vs breadth) — two-panel plot from results/figure1/ (gpt-4o-mini, GAIA-103).

(a) Depth saturation: sequential pass@1_llm vs T_cap. For each T_cap, count a question
    correct iff its `num_turns_used <= T_cap` AND it was judged correct.
    Single sequential run (T=20) gives the full saturation curve by truncation —
    valid because trajectories that ended at turn t would not have changed under
    a cap of T_cap >= t.

(b) Breadth headroom: pass@k_llm at k in {1,2,3,4} for naive_parallel and
    diversity_parallel, derived from per-thread judged correctness using the
    standard unbiased estimator pass@k = 1 - C(N-c, k) / C(N, k) where N=4 threads
    and c = #correct threads. Sequential pass@1 drawn as a dashed reference.

Compute is matched at the right edges: T=20 x k=1 (panel a) = T=5 x k=4 (panel b)
= 20 turn-units.

Inputs (read-only):
  results/figure1/sequential.csv                        (question_id, correct, num_turns_used)
  results/figure1_judged/all/sequential.jsonl           (question_id, judged_rollouts (1-element list), pass_at_1_llm)
  results/figure1_judged/all/naive_parallel.jsonl       (question_id, judged_rollouts (4-element list))
  results/figure1_judged/all/diversity_parallel.jsonl   (question_id, judged_rollouts (4-element list))

Outputs:
  figures/figure1/fig1_depth_vs_breadth.{pdf,png}
  figures/figure1/fig1_depth_vs_breadth_data.csv
"""
from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path("/home/ssmurali/t3-testbed/results/figure1")
JUDGED = Path("/home/ssmurali/t3-testbed/results/figure1_judged/all")
OUT_DIR = Path("/home/ssmurali/t3-testbed/figures/figure1")
T_CAPS = [1, 2, 3, 5, 7, 10, 15, 20]
K_VALS = [1, 2, 3, 4]

COLOR_SEQ = "#7f8c8d"
COLOR_NAIVE = "#E07A5F"
COLOR_DIV = "#2A9D8F"


def load_seq_turns_used() -> Dict[int, int]:
    """question_id -> num_turns_used (from sequential.csv)."""
    out: Dict[int, int] = {}
    with open(ROOT / "sequential.csv") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["question_id"])] = int(r["num_turns_used"])
            except (KeyError, ValueError):
                pass
    return out


def load_judged(name: str) -> Dict[int, List[bool]]:
    """question_id -> list of per-thread judged-correct booleans."""
    out: Dict[int, List[bool]] = {}
    with open(JUDGED / f"{name}.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            qid = int(obj["question_id"])
            rolls = obj.get("judged_rollouts", [])
            out[qid] = [bool(x) for x in rolls]
    return out


def passk_estimator(n_correct: int, n_total: int, k: int) -> float:
    """Unbiased pass@k for n_total rollouts of which n_correct are correct."""
    if k > n_total:
        return float("nan")
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - comb(n_total - n_correct, k) / comb(n_total, k)


def depth_curve(seq_turns: Dict[int, int], seq_judged: Dict[int, List[bool]]) -> Tuple[List[float], List[float]]:
    """Return (mean_pass1, sem_pass1) over T_CAPS."""
    qids = sorted(set(seq_turns) & set(seq_judged))
    n = len(qids)
    means, sems = [], []
    for tcap in T_CAPS:
        per_q = []
        for q in qids:
            judged = seq_judged[q]
            correct = bool(judged[0]) if judged else False
            within = seq_turns[q] <= tcap
            per_q.append(1.0 if (correct and within) else 0.0)
        arr = np.array(per_q, dtype=float)
        means.append(float(arr.mean()))
        # SE of a Bernoulli mean across N i.i.d. questions
        sems.append(float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    return means, sems


def breadth_curve(judged: Dict[int, List[bool]]) -> Tuple[List[float], List[float]]:
    """Return (mean_passk, sem_passk) over K_VALS for a 4-thread parallel run."""
    qids = sorted(judged.keys())
    n = len(qids)
    means, sems = [], []
    for k in K_VALS:
        per_q = []
        for q in qids:
            rolls = judged[q]
            n_total = len(rolls)
            n_correct = sum(rolls)
            per_q.append(passk_estimator(n_correct, n_total, k))
        arr = np.array(per_q, dtype=float)
        means.append(float(arr.mean()))
        sems.append(float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    return means, sems


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seq_turns = load_seq_turns_used()
    seq_judged = load_judged("sequential")
    naive_judged = load_judged("naive_parallel")
    div_judged = load_judged("diversity_parallel")

    n_qs = len(seq_turns)
    print(f"Loaded {n_qs} sequential questions, "
          f"{len(naive_judged)} naive, {len(div_judged)} diversity")

    seq_mean, seq_sem = depth_curve(seq_turns, seq_judged)
    naive_mean, naive_sem = breadth_curve(naive_judged)
    div_mean, div_sem = breadth_curve(div_judged)

    # ---- write data CSV ----
    data_csv = OUT_DIR / "fig1_depth_vs_breadth_data.csv"
    with open(data_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "x_label", "x_value", "condition", "mean_passk", "sem_passk"])
        for tcap, m, s in zip(T_CAPS, seq_mean, seq_sem):
            w.writerow(["depth", "T_cap", tcap, "sequential", f"{m:.6f}", f"{s:.6f}"])
        for k, m, s in zip(K_VALS, naive_mean, naive_sem):
            w.writerow(["breadth", "k", k, "naive_parallel", f"{m:.6f}", f"{s:.6f}"])
        for k, m, s in zip(K_VALS, div_mean, div_sem):
            w.writerow(["breadth", "k", k, "diversity_parallel", f"{m:.6f}", f"{s:.6f}"])
    print(f"Wrote {data_csv}")

    # ---- plot ----
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel (a): depth saturation
    sm = np.array(seq_mean)
    ss = np.array(seq_sem)
    ax_a.fill_between(T_CAPS, sm - ss, sm + ss, color=COLOR_SEQ, alpha=0.18, linewidth=0)
    ax_a.plot(T_CAPS, sm, marker="o", color=COLOR_SEQ, linewidth=2.0,
              markersize=6, label="Sequential (k=1)")
    final_seq = sm[-1]
    ax_a.axhline(final_seq, color=COLOR_SEQ, linestyle=":", linewidth=0.9, alpha=0.6)
    # annotate saturation
    ax_a.annotate(
        f"saturates at\npass@1 = {final_seq:.3f}",
        xy=(T_CAPS[-1], final_seq),
        xytext=(T_CAPS[-1] - 4.5, final_seq - 0.045),
        fontsize=9, color=COLOR_SEQ,
        arrowprops=dict(arrowstyle="->", color=COLOR_SEQ, lw=0.8, alpha=0.7),
    )
    ax_a.set_xlabel("Turn cap T (max search turns per thread)", fontsize=11)
    ax_a.set_ylabel("pass@1 (LLM-judge)", fontsize=11)
    ax_a.set_title("(a)  Depth saturates", fontsize=12, pad=6)
    ax_a.set_xticks(T_CAPS)
    ax_a.set_xlim(0, max(T_CAPS) + 1)
    ax_a.set_ylim(0, max(0.25, final_seq * 1.6))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(fontsize=10, frameon=False, loc="lower right")
    ax_a.grid(axis="y", color="#ecf0f1", linewidth=0.8)
    ax_a.set_axisbelow(True)

    # Panel (b): breadth headroom
    nm, ns = np.array(naive_mean), np.array(naive_sem)
    dm, ds = np.array(div_mean), np.array(div_sem)
    ax_b.fill_between(K_VALS, nm - ns, nm + ns, color=COLOR_NAIVE, alpha=0.18, linewidth=0)
    ax_b.fill_between(K_VALS, dm - ds, dm + ds, color=COLOR_DIV, alpha=0.18, linewidth=0)
    ax_b.plot(K_VALS, nm, marker="s", color=COLOR_NAIVE, linewidth=2.0,
              markersize=6, label="Naive parallel (T=5)")
    ax_b.plot(K_VALS, dm, marker="D", color=COLOR_DIV, linewidth=2.0,
              markersize=6, label="Diversity parallel (T=5)")
    # sequential reference
    ax_b.axhline(final_seq, color=COLOR_SEQ, linestyle="--", linewidth=1.0, alpha=0.85,
                 label=f"Sequential pass@1 (T=20)")
    # annotate headroom: gap between div pass@4 and seq pass@1
    div_top = dm[-1]
    if div_top > final_seq:
        x_at = K_VALS[-1] + 0.05
        ax_b.annotate(
            "", xy=(x_at, div_top), xytext=(x_at, final_seq),
            arrowprops=dict(arrowstyle="<->", color="#34495e", lw=1.0),
        )
        ax_b.text(x_at + 0.05, (div_top + final_seq) / 2,
                  f"+{(div_top - final_seq) * 100:.0f} pts\nheadroom",
                  fontsize=9, color="#34495e", va="center")
    ax_b.set_xlabel("Number of parallel threads k", fontsize=11)
    ax_b.set_ylabel("pass@k (LLM-judge)", fontsize=11)
    ax_b.set_title("(b)  Breadth keeps going", fontsize=12, pad=6)
    ax_b.set_xticks(K_VALS)
    ax_b.set_xlim(0.7, max(K_VALS) + 0.7)
    ax_b.set_ylim(0, max(div_top, final_seq) * 1.4)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(fontsize=10, frameon=False, loc="lower right")
    ax_b.grid(axis="y", color="#ecf0f1", linewidth=0.8)
    ax_b.set_axisbelow(True)

    fig.suptitle(f"GAIA-103  |  gpt-4o-mini  |  Compute matched: T=20×k=1 = T=5×k=4 (20 turn-units)",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    fig.savefig(OUT_DIR / "fig1_depth_vs_breadth.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_depth_vs_breadth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig1_depth_vs_breadth.{{pdf,png}} -> {OUT_DIR}")

    print("\n--- numbers ---")
    print("depth (sequential pass@1 vs T_cap):")
    for t, m, s in zip(T_CAPS, seq_mean, seq_sem):
        print(f"  T={t:>2}: {m:.4f} ± {s:.4f}")
    print("breadth naive pass@k:")
    for k, m, s in zip(K_VALS, naive_mean, naive_sem):
        print(f"  k={k}: {m:.4f} ± {s:.4f}")
    print("breadth diversity pass@k:")
    for k, m, s in zip(K_VALS, div_mean, div_sem):
        print(f"  k={k}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
