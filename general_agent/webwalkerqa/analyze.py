"""
Analysis and visualization for WebWalkerQA T³ PoC.

Generates Figure 1: two side-by-side plots comparing s1 vs T³.
Also produces aggregate stats tables.

Usage:
  # After running experiments:
  python -m webwalkerqa.analyze --results-dir results/webwalkerqa

  # Plot only specific groups:
  python -m webwalkerqa.analyze --results-dir results/webwalkerqa --groups A B

  # Save figure:
  python -m webwalkerqa.analyze --results-dir results/webwalkerqa --output figure1.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure the general_agent directory is importable
_GA_DIR = Path(__file__).parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))


def load_summaries(results_dir: Path) -> list[dict]:
    """Load all summary.json files from results directory."""
    summaries = []
    for summary_path in results_dir.rglob("summary.json"):
        try:
            with open(summary_path) as f:
                s = json.load(f)
            summaries.append(s)
        except Exception as e:
            print(f"[warn] Failed to load {summary_path}: {e}")
    return summaries


def print_table(summaries: list[dict]) -> None:
    """Print a formatted comparison table."""
    if not summaries:
        print("No results found.")
        return

    # Sort by group, then config_id
    summaries.sort(key=lambda s: (s.get("config_id", "")[:1], s.get("config_id", "")))

    print("\n" + "=" * 80)
    print("WEBWALKERQA RESULTS")
    print("=" * 80)
    print(f"{'Config':<10} {'Method':<12} {'k':>4} {'n':>4} {'t':>6} "
          f"{'EM':>8} {'Correct':>8} {'Total':>8} {'Searches':>10}")
    print("-" * 80)

    current_group = None
    for s in summaries:
        group = s.get("config_id", "?")[:1]
        if group != current_group:
            if current_group is not None:
                print()
            current_group = group

        print(
            f"{s.get('config_id', '?'):<10} "
            f"{s.get('method', '?'):<12} "
            f"{s.get('k', 0):>4} "
            f"{s.get('n', 0):>4} "
            f"{s.get('t', 0):>6} "
            f"{s.get('em', 0):>8.4f} "
            f"{s.get('num_correct', 0):>8} "
            f"{s.get('num_total', 0):>8} "
            f"{s.get('avg_search_calls', 0):>10.1f}"
        )

    print("=" * 80)

    # Oracle vs best
    oracle = next((s for s in summaries if s.get("method") == "oracle"), None)
    s1_best = max(
        (s for s in summaries if s.get("method") == "s1"),
        key=lambda s: s.get("em", 0),
        default=None,
    )
    t3_best = max(
        (s for s in summaries if s.get("method") == "t3_fixed"),
        key=lambda s: s.get("em", 0),
        default=None,
    )

    print("\nKey comparisons:")
    if oracle:
        print(f"  Oracle (ceiling): EM = {oracle['em']:.4f}")
    if s1_best:
        gap = (oracle["em"] - s1_best["em"]) if oracle else float("nan")
        print(f"  Best s1:          EM = {s1_best['em']:.4f} ({s1_best['config_id']}) "
              f"[gap to oracle: {gap:.4f}]")
    if t3_best:
        gap = (oracle["em"] - t3_best["em"]) if oracle else float("nan")
        print(f"  Best T³:          EM = {t3_best['em']:.4f} ({t3_best['config_id']}) "
              f"[gap to oracle: {gap:.4f}]")
    print()


def plot_figure1(summaries: list[dict], output_path: Optional[str] = None) -> None:
    """
    Generate Figure 1: side-by-side plots of s1 vs T³.

    Left: Sequential scaling (s1) — EM vs total search calls
    Right: Parallel scaling (T³) — EM vs k (thread count)
    Both share the oracle ceiling line.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed. Install with: pip install matplotlib")
        return

    # Group summaries
    s1_results = {s["config_id"]: s for s in summaries if s.get("method") == "s1"}
    t3_results = {s["config_id"]: s for s in summaries if s.get("method") == "t3_fixed"}
    oracle = next((s for s in summaries if s.get("method") == "oracle"), None)

    # Config ordering for x-axis
    s1_order = ["A1", "B1", "C1"]  # increasing compute
    t3_order = ["A2", "B2", "B3", "C2", "C3"]  # increasing threads

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    oracle_em = oracle["em"] if oracle else None
    oracle_searches = oracle["avg_search_calls"] if oracle else None

    # ── Left plot: Sequential Scaling (s1) ──────────────────────────────────
    s1_x = []  # total search calls (compute proxy)
    s1_y = []  # EM
    s1_labels = []

    for cid in s1_order:
        if cid in s1_results:
            s = s1_results[cid]
            s1_x.append(s.get("avg_search_calls", s.get("t", 0) / 1024))
            s1_y.append(s["em"])
            s1_labels.append(cid)

    if s1_x:
        ax1.plot(s1_x, s1_y, "b-o", linewidth=2, markersize=8, label="s1", zorder=3)
        for x, y, label in zip(s1_x, s1_y, s1_labels):
            ax1.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    if oracle_em is not None:
        ax1.axhline(y=oracle_em, color="red", linestyle="--", linewidth=2, label=f"Oracle ({oracle_em:.3f})", zorder=2)

    ax1.set_xlabel("Avg. Search Calls per Question", fontsize=12)
    ax1.set_ylabel("Exact Match (EM)", fontsize=12)
    ax1.set_title("Sequential Scaling (s1)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # ── Right plot: Parallel Scaling (T³) ───────────────────────────────────
    t3_x = []  # k (thread count)
    t3_y = []  # EM
    t3_labels = []

    for cid in t3_order:
        if cid in t3_results:
            s = t3_results[cid]
            t3_x.append(s.get("k", 1))
            t3_y.append(s["em"])
            t3_labels.append(cid)

    if t3_x:
        ax2.plot(t3_x, t3_y, "g-o", linewidth=2, markersize=8, label="T³ Fixed", zorder=3)
        for x, y, label in zip(t3_x, t3_y, t3_labels):
            ax2.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    if oracle_em is not None:
        ax2.axhline(y=oracle_em, color="red", linestyle="--", linewidth=2, label=f"Oracle ({oracle_em:.3f})", zorder=2)

    ax2.set_xlabel("Number of Threads (k)", fontsize=12)
    ax2.set_ylabel("Exact Match (EM)", fontsize=12)
    ax2.set_title("Parallel Scaling (T³ Fixed)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("WebWalkerQA: Sequential vs Parallel Scaling", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[analyze] Figure saved to {output_path}")
    else:
        plt.show()

    plt.close()


def compute_collapse_metrics(results_dir: Path) -> dict:
    """
    Compute query diversity metrics (QNS, ITC estimates) from saved traces.

    These are approximate: based on lexical overlap (TF-IDF cosine similarity)
    between search queries within a trajectory.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("[warn] scikit-learn not installed. Skipping collapse metrics.")
        return {}

    qns_scores = []
    itc_rates = []

    for q_file in results_dir.rglob("q_*.json"):
        try:
            with open(q_file) as f:
                data = json.load(f)
        except Exception:
            continue

        # Collect all search queries
        all_queries = []
        for turn in data.get("turns", []):
            all_queries.extend(turn.get("search_queries", []))
            all_queries.extend(turn.get("thread_queries", []))

        if len(all_queries) < 2:
            continue

        # Compute pairwise TF-IDF similarity
        try:
            vec = TfidfVectorizer().fit_transform(all_queries)
            sim_matrix = cosine_similarity(vec)
            np.fill_diagonal(sim_matrix, 0)

            # QNS: avg pairwise dissimilarity
            avg_sim = sim_matrix.sum() / (len(all_queries) * (len(all_queries) - 1))
            qns = 1.0 - avg_sim
            qns_scores.append(qns)

            # ITC: proportion of steps where query is too similar to a prior query
            tau = 0.8
            itc_count = 0
            for i in range(1, len(all_queries)):
                if sim_matrix[i, :i].max() > tau:
                    itc_count += 1
            itc_rates.append(itc_count / max(1, len(all_queries) - 1))

        except Exception:
            continue

    if not qns_scores:
        return {}

    import numpy as np
    return {
        "avg_qns": float(np.mean(qns_scores)),
        "avg_itc_rate": float(np.mean(itc_rates)),
        "num_trajectories": len(qns_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze WebWalkerQA experiment results")
    parser.add_argument(
        "--results-dir", type=str, default="results/webwalkerqa",
        help="Directory containing experiment results (default: results/webwalkerqa)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save Figure 1 (e.g. figure1.png). Shows interactively if not set.",
    )
    parser.add_argument(
        "--groups", nargs="+", choices=["A", "B", "C", "Oracle"],
        help="Filter to specific groups",
    )
    parser.add_argument(
        "--collapse-metrics", action="store_true",
        help="Compute query diversity metrics (requires scikit-learn)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (only print table)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}")
        sys.exit(1)

    # Load summaries
    summaries = load_summaries(results_dir)

    # Filter by group
    if args.groups:
        groups_upper = [g.upper() for g in args.groups]
        summaries = [s for s in summaries if s.get("config_id", "?")[0] in groups_upper
                     or s.get("method") == "oracle" and "Oracle" in groups_upper]

    if not summaries:
        print("[warn] No summaries found. Run experiments first.")
        sys.exit(0)

    # Print table
    print_table(summaries)

    # Collapse metrics
    if args.collapse_metrics:
        print("\nComputing collapse metrics...")
        for run_dir in results_dir.iterdir():
            if not run_dir.is_dir():
                continue
            q_dir = run_dir / "questions"
            if q_dir.exists():
                metrics = compute_collapse_metrics(q_dir)
                if metrics:
                    print(f"  {run_dir.name}:")
                    print(f"    QNS (query novelty): {metrics['avg_qns']:.3f}")
                    print(f"    ITC rate (intra-thread collapse): {metrics['avg_itc_rate']:.3f}")
                    print(f"    Trajectories: {metrics['num_trajectories']}")

    # Plot
    if not args.no_plot:
        plot_figure1(summaries, output_path=args.output)


if __name__ == "__main__":
    main()
