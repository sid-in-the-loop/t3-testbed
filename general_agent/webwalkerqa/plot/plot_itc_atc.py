#!/usr/bin/env python3
"""
Compute ITC / ATC profiles from diversity_parallel result JSONs and plot.

Expects files: {condition}_gaia25.json, {condition}_hotpotqa25.json

Outputs:
  figures/itc_atc_summary.csv — scalar summary (all 6 conditions × 2 benchmarks)
  figures/itc_atc_profiles.json — full ITC/ATC profile arrays (use --data-only to skip plots)
  figures/itc_atc_{benchmark}.pdf, .png — unless --data-only

Usage (from general_agent):
  pip install editdistance numpy pandas
  pip install matplotlib  # only if not using --data-only
  python -m webwalkerqa.plot.plot_itc_atc --results-dir results/diversity_parallel --figures-dir figures
  python -m webwalkerqa.plot.plot_itc_atc --data-only --output-dir results/diversity_parallel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import editdistance
except ImportError:
    print("pip install editdistance", file=sys.stderr)
    raise

_GA_DIR = Path(__file__).resolve().parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

CONDITIONS_ORDER = [
    "sequential",
    "naive_k4",
    "random_o16_k4",
    "greedy_jaccard_o16_k4",
    "greedy_dense_o16_k4",
    "mmr_dense_o16_k4",
]

COLORS = {
    "sequential": "gray",
    "naive_k4": "red",
    "random_o16_k4": "orange",
    "greedy_jaccard_o16_k4": "lightblue",
    "greedy_dense_o16_k4": "blue",
    "mmr_dense_o16_k4": "green",
}

BENCHMARKS = ["gaia25", "hotpotqa25"]


def _tokens(q: str) -> List[str]:
    return (q or "").lower().split()


def sim_q_tau_q1(q1: str, q_tau: str) -> float:
    t1 = _tokens(q1)
    ta = _tokens(q_tau)
    if not t1 or not ta:
        return np.nan
    mx = max(len(t1), len(ta))
    d = float(editdistance.eval(t1, ta))
    return 1.0 - d / mx


def dist_pair(q_i: str, q_j: str) -> float:
    ti = _tokens(q_i)
    tj = _tokens(q_j)
    if not ti or not tj:
        return np.nan
    mx = max(len(ti), len(tj))
    return float(editdistance.eval(ti, tj)) / mx


def turns_to_query_by_turn(turns: List[dict]) -> Dict[int, str]:
    """Map turn number -> query string."""
    out: Dict[int, str] = {}
    for t in turns or []:
        q = t.get("query")
        if not q:
            continue
        turn = t.get("turn")
        if turn is None:
            continue
        out[int(turn)] = str(q).strip()
    return out


def thread_valid_for_itc(q_by_turn: Dict[int, str]) -> bool:
    return 1 in q_by_turn and len(q_by_turn[1].strip()) > 0


def itc_scalar_for_thread(q_by_turn: Dict[int, str], max_turn: int) -> float:
    """Mean of 1[sim>0.8] for tau=2..min(T,max_turn)."""
    if 1 not in q_by_turn:
        return np.nan
    q1 = q_by_turn[1]
    sims = []
    for tau in range(2, max_turn + 1):
        if tau not in q_by_turn:
            break
        s = sim_q_tau_q1(q1, q_by_turn[tau])
        if np.isnan(s):
            continue
        sims.append(1.0 if s > 0.8 else 0.0)
    if not sims:
        return np.nan
    return float(np.mean(sims))


def itc_sims_per_thread(
    q_by_turn: Dict[int, str], profile_turns: List[int]
) -> np.ndarray:
    """sim(q_tau, q1) for each tau in profile_turns; NaN if missing."""
    if 1 not in q_by_turn:
        return np.full(len(profile_turns), np.nan)
    q1 = q_by_turn[1]
    row = []
    for tau in profile_turns:
        if tau not in q_by_turn:
            row.append(np.nan)
        else:
            row.append(sim_q_tau_q1(q1, q_by_turn[tau]))
    return np.asarray(row, dtype=np.float64)


def atc_at_turn(
    thread_maps: List[Dict[int, str]], turn: int
) -> Optional[float]:
    """Mean pairwise dist over k(k-1)/2 pairs; None if <2 threads have query."""
    queries = []
    for m in thread_maps:
        queries.append(m.get(turn))
    valid = [(i, q) for i, q in enumerate(queries) if q and str(q).strip()]
    if len(valid) < 2:
        return None
    dists = []
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            qi, qj = valid[a][1], valid[b][1]
            d = dist_pair(qi, qj)
            if not np.isnan(d):
                dists.append(d)
    if not dists:
        return None
    return float(np.mean(dists))


def load_result_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_condition_benchmark(name: str) -> Optional[Tuple[str, str]]:
    for b in BENCHMARKS:
        suffix = f"_{b}.json"
        if name.endswith(suffix):
            cond = name[: -len(suffix)]
            return cond, b
    return None


def collect_metrics_for_file(
    path: Path,
    itc_turns: List[int],
    atc_turns: List[int],
) -> Dict[str, Any]:
    data = load_result_json(path)
    condition = data.get("condition", "")
    benchmark = data.get("benchmark", "")
    results = data.get("results") or []

    itc_scalars: List[float] = []
    # ITC profile: list of arrays (n_questions * n_threads_per_q) flattened per turn
    itc_by_turn: Dict[int, List[float]] = {t: [] for t in itc_turns}

    atc_by_turn: Dict[int, List[float]] = {t: [] for t in atc_turns}

    is_sequential = condition == "sequential"

    for row in results:
        if row.get("error"):
            continue
        trajs = row.get("thread_trajectories") or []
        thread_maps = [turns_to_query_by_turn(t.get("turns") or []) for t in trajs]

        per_q_itc: List[float] = []
        for tm in thread_maps:
            if not thread_valid_for_itc(tm):
                continue
            turns_present = sorted(k for k in tm if k >= 1)
            if len(turns_present) < 2:
                continue
            t_max = max(turns_present)
            isc = itc_scalar_for_thread(tm, t_max)
            if not np.isnan(isc):
                per_q_itc.append(isc)

            sims = itc_sims_per_thread(tm, itc_turns)
            for idx, tau in enumerate(itc_turns):
                v = sims[idx]
                if not np.isnan(v):
                    itc_by_turn[tau].append(float(v))

        if per_q_itc:
            itc_scalars.append(float(np.mean(per_q_itc)))

        if not is_sequential and len(thread_maps) >= 2:
            for t in atc_turns:
                v = atc_at_turn(thread_maps, t)
                if v is not None:
                    atc_by_turn[t].append(v)

    itc_mean = float(np.nanmean(itc_scalars)) if itc_scalars else float("nan")

    itc_profile_mean = np.array(
        [np.nanmean(itc_by_turn[t]) if itc_by_turn[t] else np.nan for t in itc_turns]
    )
    itc_profile_sem = np.array(
        [
            _sem(itc_by_turn[t]) if itc_by_turn[t] else 0.0 for t in itc_turns
        ]
    )

    atc_profile_mean = np.array(
        [np.nanmean(atc_by_turn[t]) if atc_by_turn[t] else np.nan for t in atc_turns]
    )
    atc_profile_sem = np.array(
        [_sem(atc_by_turn[t]) if atc_by_turn[t] else 0.0 for t in atc_turns]
    )

    valid_atc = [atc_profile_mean[i] for i in range(len(atc_turns)) if not np.isnan(atc_profile_mean[i])]
    atc_scalar = float(np.mean(valid_atc)) if valid_atc else float("nan")

    def _atc_turn(t: int) -> float:
        if t not in atc_by_turn or not atc_by_turn[t]:
            return float("nan")
        return float(np.mean(atc_by_turn[t]))

    return {
        "condition": condition,
        "benchmark": benchmark,
        "itc_mean": itc_mean,
        "atc_mean": atc_scalar,
        "atc_turn1": _atc_turn(1),
        "atc_turn5": _atc_turn(5),
        "itc_turns": itc_turns,
        "itc_profile_mean": itc_profile_mean,
        "itc_profile_sem": itc_profile_sem,
        "atc_turns": atc_turns,
        "atc_profile_mean": atc_profile_mean,
        "atc_profile_sem": atc_profile_sem,
        "is_sequential": is_sequential,
    }


def _sem(xs: List[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(n))


def plot_benchmark(
    benchmark: str,
    per_condition: Dict[str, Dict[str, Any]],
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    itc_turns = [2, 3, 4, 5]
    atc_turns = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_itc, ax_atc = axes

    lines_for_legend = []
    labels_for_legend = []

    for cond in CONDITIONS_ORDER:
        if cond not in per_condition:
            continue
        m = per_condition[cond]
        c = COLORS.get(cond, "black")
        # ITC
        y = m["itc_profile_mean"]
        se = m["itc_profile_sem"]
        x = np.array(itc_turns, dtype=float)
        valid = ~np.isnan(y)
        if np.any(valid):
            line, = ax_itc.plot(
                x[valid],
                y[valid],
                color=c,
                marker="o",
                markersize=4,
                label=cond,
            )
            if cond not in labels_for_legend:
                lines_for_legend.append(line)
                labels_for_legend.append(cond)
            xv, yv, sv = x[valid], y[valid], se[valid]
            ax_itc.fill_between(
                xv, yv - sv, yv + sv, color=c, alpha=0.2, linewidth=0
            )

    ax_itc.axhline(0.8, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax_itc.set_xlim(1.5, 5.5)
    ax_itc.set_ylim(0, 1)
    ax_itc.set_xticks(itc_turns)
    ax_itc.set_xlabel("Turn")
    ax_itc.set_ylabel("Mean sim(q_t, q_1)")
    ax_itc.set_title("Intra-Thread Collapse: Similarity to Turn-1 Query Over Time")

    for cond in CONDITIONS_ORDER:
        if cond not in per_condition:
            continue
        if cond == "sequential":
            continue
        m = per_condition[cond]
        c = COLORS.get(cond, "black")
        y = m["atc_profile_mean"]
        se = m["atc_profile_sem"]
        x = np.array(atc_turns, dtype=float)
        valid = ~np.isnan(y)
        if np.any(valid):
            ax_atc.plot(x[valid], y[valid], color=c, marker="s", markersize=4)
            xv, yv, sv = x[valid], y[valid], se[valid]
            ax_atc.fill_between(xv, yv - sv, yv + sv, color=c, alpha=0.2, linewidth=0)

    ax_atc.set_xlim(0.5, 5.5)
    ax_atc.set_ylim(0, 1)
    ax_atc.set_xticks(atc_turns)
    ax_atc.set_xlabel("Turn")
    ax_atc.set_ylabel("Mean pairwise token-edit distance")
    ax_atc.set_title("Across-Thread Diversity: Pairwise Query Distance Over Turns")

    fig.legend(
        lines_for_legend,
        labels_for_legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )
    cap = (
        "ITC: higher line = stuck near Turn-1 query (collapse); dropping = escaping anchor. "
        "ATC: higher = threads stay query-diverse; dropping = converging queries. "
        "Sequential has no multi-thread ATC (omitted from right panel)."
    )
    fig.text(0.5, -0.08, cap, ha="center", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    stem = figures_dir / f"itc_atc_{benchmark}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.pdf and {stem}.png")


def _json_float(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return None
    if isinstance(x, np.ndarray):
        return [_json_float(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_json_float(v) for v in x]
    return float(x)


def build_profiles_export(
    by_bench: Dict[str, Dict[str, Any]],
    itc_turns: List[int],
    atc_turns: List[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "meta": {
            "itc_turns": itc_turns,
            "atc_turns": atc_turns,
            "itc_profile_note": "mean sim(q_t, q_1) pooled over threads×questions; sem over observations",
            "atc_profile_note": "mean pairwise token edit distance / max(len); parallel conditions only meaningful",
        },
    }
    for bench in BENCHMARKS:
        out[bench] = {}
        for cond in CONDITIONS_ORDER:
            if cond not in by_bench[bench]:
                out[bench][cond] = None
                continue
            m = by_bench[bench][cond]
            entry: Dict[str, Any] = {
                "itc_mean": _json_float(m["itc_mean"]),
                "atc_mean": _json_float(m["atc_mean"]),
                "atc_turn1": _json_float(m["atc_turn1"]),
                "atc_turn5": _json_float(m["atc_turn5"]),
                "itc_profile": {
                    "turns": itc_turns,
                    "mean": _json_float(m["itc_profile_mean"]),
                    "sem": _json_float(m["itc_profile_sem"]),
                },
                "atc_profile": {
                    "turns": atc_turns,
                    "mean": _json_float(m["atc_profile_mean"]),
                    "sem": _json_float(m["atc_profile_sem"]),
                },
            }
            out[bench][cond] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(_GA_DIR / "results" / "diversity_parallel"),
        help="Directory containing {condition}_{benchmark}.json",
    )
    ap.add_argument(
        "--figures-dir",
        type=str,
        default=str(_GA_DIR / "figures"),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write CSV/JSON (default: same as --figures-dir)",
    )
    ap.add_argument(
        "--data-only",
        action="store_true",
        help="Write itc_atc_summary.csv + itc_atc_profiles.json only (no matplotlib)",
    )
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.output_dir or args.figures_dir)

    itc_turns = [2, 3, 4, 5]
    atc_turns = [1, 2, 3, 4, 5]

    by_bench: Dict[str, Dict[str, Any]] = {b: {} for b in BENCHMARKS}

    for path in sorted(results_dir.glob("*.json")):
        parsed = parse_condition_benchmark(path.name)
        if not parsed:
            continue
        cond, bench = parsed
        by_bench[bench][cond] = collect_metrics_for_file(path, itc_turns, atc_turns)

    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas", file=sys.stderr)
        raise

    def _fmt(x: float) -> str:
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "nan"
        return f"{float(x):.4f}"

    rows = []
    for bench in BENCHMARKS:
        for cond in CONDITIONS_ORDER:
            if cond in by_bench[bench]:
                m = by_bench[bench][cond]
                rows.append(
                    {
                        "condition": cond,
                        "benchmark": bench,
                        "itc_mean": m["itc_mean"],
                        "atc_mean": m["atc_mean"],
                        "atc_turn1": m["atc_turn1"],
                        "atc_turn5": m["atc_turn5"],
                    }
                )
                print(
                    f"{cond:28} {bench:12} itc_mean={_fmt(m['itc_mean'])} "
                    f"atc_turn1={_fmt(m['atc_turn1'])} atc_turn5={_fmt(m['atc_turn5'])}"
                )
            else:
                rows.append(
                    {
                        "condition": cond,
                        "benchmark": bench,
                        "itc_mean": np.nan,
                        "atc_mean": np.nan,
                        "atc_turn1": np.nan,
                        "atc_turn5": np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figures_dir / "itc_atc_summary.csv"
    df.to_csv(csv_path, index=False, na_rep="")
    print(f"Wrote {csv_path}")

    json_path = figures_dir / "itc_atc_profiles.json"
    export = build_profiles_export(by_bench, itc_turns, atc_turns)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, allow_nan=False)
    print(f"Wrote {json_path}")

    if args.data_only:
        return

    for bench in BENCHMARKS:
        if not by_bench[bench]:
            print(f"No JSON files for {bench} in {results_dir}", file=sys.stderr)
            continue
        plot_benchmark(bench, by_bench[bench], Path(args.figures_dir))


if __name__ == "__main__":
    main()
