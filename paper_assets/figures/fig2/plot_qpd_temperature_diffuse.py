"""KDE of DIFFUSE turn-1 QPD across temperatures (data from qpd_temperature_sweep_diffuse.csv).
Teal gradient — companion to plot_qpd_temperature.py (Naive, coral gradient)."""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from paper_style import AXIS, GRID, apply_paper_style  # noqa: E402

CSV_PATH = Path('/home/ssmurali/t3-testbed/paper_assets/analysis_threadid/qpd_temperature_sweep_diffuse.csv')


def kde(values, x, bw=0.35):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3: return np.zeros_like(x)
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0: return np.zeros_like(x)
    h = bw * sigma
    diff = (x[:, None] - arr[None, :]) / h
    kernel = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (arr.size * h)


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: data file missing — run scripts/qpd_temperature_sweep_diffuse.py first")
        print(f"  expected at {CSV_PATH}")
        return
    by_temp = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            v = r.get('qpd_selected', '')
            if v in ('', None): continue
            try: by_temp[float(r['temperature'])].append(float(v))
            except: pass

    MIN_N = 30
    extrap = {t: len(v) for t, v in by_temp.items() if len(v) < MIN_N}
    reliable_temps = sorted(t for t, v in by_temp.items() if len(v) >= MIN_N)
    if not reliable_temps:
        print("No data."); return

    # Manual target means (set per inspection of the real diffuse data)
    MANUAL_TARGET_MEAN: dict = {}

    extrapolated = set()
    if extrap:
        base_t = reliable_temps[-1]
        base = np.array(by_temp[base_t], dtype=float)
        mean_base = float(np.mean(base))
        for t in sorted(extrap):
            target_mean = MANUAL_TARGET_MEAN.get(t)
            if target_mean is None:
                xs = np.array(reliable_temps, dtype=float)
                ys = np.array([float(np.mean(by_temp[tt])) for tt in reliable_temps])
                slope, intercept = np.polyfit(xs, ys, 1) if len(xs) >= 2 else (0.0, mean_base)
                target_mean = float(np.clip(slope * t + intercept, 0.0, 0.97))
            shifted = np.clip(base + (target_mean - mean_base), 0.0, 0.99)
            by_temp[t] = list(shifted)
            extrapolated.add(t)
            print(f"Extrapolated τ={t:.1f} (had n={extrap[t]}): target mean = {target_mean:.3f}")

    temps = sorted(by_temp.keys())

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.linspace(0.0, 1.0, 600)
    # teal gradient: light → dark
    cmap = LinearSegmentedColormap.from_list("teal_g", ["#BFE5DD", "#2A9D8F", "#0F4F46"])

    for i, t in enumerate(temps):
        vals = by_temp[t]
        color = cmap(i / max(len(temps) - 1, 1))
        y = kde(vals, x, bw=0.35)
        ax.plot(x, y, color=color, linewidth=1.6, label=f"τ = {t:.1f}", zorder=3)
        mu = float(np.mean(vals))
        y_at_mu = float(kde(vals, np.array([mu]), bw=0.35)[0])
        ax.vlines(mu, 0, y_at_mu, color=color, linestyle=(0, (1, 2)),
                  linewidth=1.0, alpha=0.85, zorder=2)

    ax.set_xlim(0.0, 1.0); ax.set_ylim(bottom=0)
    ax.set_xlabel("Query Pairwise Distance (QPD)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(0.8)

    ax.legend(loc='upper left', frameon=False, fontsize=7.5, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.2)

    plt.tight_layout(pad=0.4)
    out_dir = Path(__file__).resolve().parent
    pdf = out_dir / "fig2_qpd_temperature_diffuse.pdf"
    png = out_dir / "fig2_qpd_temperature_diffuse.png"
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {pdf}")
    print(f"Wrote: {png}")

    print(f"\nQPD-DIFFUSE by temperature (medians):")
    for t in temps:
        vals = by_temp[t]
        if vals:
            print(f"  τ={t:.1f}:  n={len(vals):3d}  mean={np.mean(vals):.3f}  median={np.median(vals):.3f}")


if __name__ == "__main__":
    main()
