import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_pass_at_k_from_summary(summary_path: str, output_path: str):
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    k_values = [1, 2, 4, 8, 16]
    scores = [data[f"pass@{k}"] * 100 for k in k_values]
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, scores, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#ff7f0e', label=f"s1 (gpt-5-nano)")
    
    plt.xscale('log', base=2)
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.xlabel("Number of Samples (k)")
    plt.ylabel("Pass@k Accuracy (%)")
    plt.title(f"Pass@k Scaling for gpt-5-nano (s1 config)")
    
    for i, txt in enumerate(scores):
        plt.annotate(f"{txt:.1f}%", (k_values[i], scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.ylim(0, max(scores) + 10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Plot saved to {output_path}")

if __name__ == "__main__":
    plot_pass_at_k_from_summary(
        "results/webwalkerqa/gaia_25_gpt5/s1_gpt-5-nano/summary.json",
        "results/webwalkerqa/gaia_25_gpt5/pass_at_k_scaling_gpt5.png"
    )
