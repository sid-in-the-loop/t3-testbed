import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Configure aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18
})

def load_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load pass@k results for each experiment in the directory."""
    results = {}
    
    # Expected experiment IDs
    exp_ids = [
        "s1", "T3-Fixed-Naive", "T3-Fixed-Anchor",
        "T3-Fixed-Jaccard", "T3-Fixed-DPP", "T3-Dynamic", "T3-Dynamic-Jaccard"
    ]
    
    # Get all subdirectories in results_dir
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for exp_id in exp_ids:
        # Find the directory that matches this exp_id (either exact or with a suffix like _gpt-4o-mini)
        matching_dirs = [d for d in subdirs if d.name == exp_id or d.name.startswith(f"{exp_id}_")]
        
        if not matching_dirs:
            print(f"[INFO] No directory found starting with {exp_id} in {results_dir}")
            continue
            
        # Use the first matching directory
        exp_dir = matching_dirs[0]
        eval_file = exp_dir / "questions" / "llm_judge_evaluation.json"
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    stats = data.get("aggregate_stats", {})
                    # Convert 0-1 range to 0-100%
                    results[exp_id] = {
                        "1": stats.get("pass@1", 0) * 100,
                        "2": stats.get("pass@2", 0) * 100,
                        "4": stats.get("pass@4", 0) * 100,
                        "8": stats.get("pass@8", 0) * 100,
                        "16": stats.get("pass@16", 0) * 100
                    }
            except Exception as e:
                print(f"[WARN] Failed to parse {eval_file}: {e}")
        else:
            print(f"[INFO] Results for {exp_id} not found at {eval_file}")
            
    return results

def plot_pass_at_k(results: Dict[str, Dict[str, float]], output_path: str):
    """Generate the pass@k scaling plot."""
    plt.figure(figsize=(10, 7))

    # Orange s1, blue T3 fixed, green dynamic (dotted); thicker lines
    colors = {
        "s1": "#ff7f0e",
        "T3-Fixed-Naive": "#1f77b4",
        "T3-Fixed-Anchor": "#1f77b4",
        "T3-Fixed-Jaccard": "#1f77b4",
        "T3-Fixed-DPP": "#1f77b4",
        "T3-Dynamic": "#2ca02c",
        "T3-Dynamic-Jaccard": "#2ca02c"
    }
    markers = {
        "s1": "o",
        "T3-Fixed-Naive": "s",
        "T3-Fixed-Anchor": "^",
        "T3-Fixed-Jaccard": "D",
        "T3-Fixed-DPP": "v",
        "T3-Dynamic": "*",
        "T3-Dynamic-Jaccard": "P"
    }
    linestyles = {
        "s1": "-.",
        "T3-Fixed-Naive": "-",
        "T3-Fixed-Anchor": "--",
        "T3-Fixed-Jaccard": ":",
        "T3-Fixed-DPP": (0, (3, 5, 1, 5)),
        "T3-Dynamic": (0, (2, 2)),   # dotted
        "T3-Dynamic-Jaccard": (0, (1, 1))  # looser dotted
    }
    # Thicker lines: 3.0 for dynamic, 2.5 for rest
    def _lw(eid):
        return 3.0 if eid in ("T3-Dynamic", "T3-Dynamic-Jaccard") else 2.5
    def _ms(eid):
        return 9 if eid in ("T3-Dynamic", "T3-Dynamic-Jaccard") else 8

    k_values = [1, 2, 4, 8, 16]
    for exp_id, scores in results.items():
        y_values = [scores.get(str(k), 0) for k in k_values]
        plt.plot(
            k_values, y_values,
            label=exp_id,
            color=colors.get(exp_id, "#333"),
            marker=markers.get(exp_id, "o"),
            linestyle=linestyles.get(exp_id, "-"),
            linewidth=_lw(exp_id),
            markersize=_ms(exp_id),
            alpha=0.9
        )

    plt.xscale('log', base=2)
    plt.xticks(k_values, ["1", "2", "4", "8", "16"])
    plt.xlabel("Number of Samples (k)")
    plt.ylabel("Pass@k Accuracy (%)")
    plt.title("Test-Time Scaling of Search Agents on GAIA-50 (16 rollouts)")
    
    # Set y-axis limits with some padding
    all_scores = [v for s in results.values() for v in s.values()]
    if all_scores:
        y_min = max(0, min(all_scores) - 5)
        y_max = min(100, max(all_scores) + 5)
        plt.ylim(y_min, y_max)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"✓ Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot pass@k scaling results.")
    parser.add_argument("--results-dir", type=str, default="results/gaia_25", help="Path to the results directory.")
    parser.add_argument("--output", type=str, default="pass_at_k_scaling.png", help="Path to save the plot.")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    
    if not results:
        print("[ERROR] No results found to plot.")
        return
        
    plot_pass_at_k(results, args.output)

if __name__ == "__main__":
    main()
