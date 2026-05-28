"""
Three-panel temperature-sweep figure: naive pass@4 vs τ, with DIFFUSE @ τ=1.0
as a dashed horizontal reference.

qwen3-8b, naive parallel, 3 seeds per cell. Hotpotqa outlier seed dropped.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ssmurali/t3-testbed/paper_assets/figures/_shared')
from paper_style import apply_paper_style, AXIS  # noqa: E402

ROOT = Path('/home/ssmurali/t3-testbed/results')
TEMP_ROOT = ROOT / 'temperature_sweep_qwen3_8b/qwen3-8b'
SEEDS = [1, 2, 3]
TEMPS = [0.5, 1.0, 1.5, 2.0]

DATASETS = [
    ('hotpotqa',  'HotpotQA',   'main_table_clueweb_t8', 'diversity_parallel', 'hotpotqa'),
    ('bamboogle', 'Bamboogle',  'main_table_clueweb_t8', 'diversity_parallel', 'bamboogle'),
    ('GAIA',      'GAIA',       'main_table_web_serper', 'div_k4',             'gaia'),
]

CARNEGIE = "#C41230"
IRON     = "#6D6E71"

OUT_DIR = Path('/home/ssmurali/t3-testbed/paper_assets/figures/temp_sweep')


def read_passk(run, col='pass_at_4_llm'):
    sm = list(run.glob('summary_T*.csv'))
    if not sm: return None
    try: return float(next(csv.DictReader(open(sm[0])))[col])
    except: return None


def drop_hotpotqa_outlier(vals, ds_key):
    """If a hotpotqa cell has one seed >15 pp below the rest, drop it."""
    if ds_key != 'hotpotqa' or len(vals) != 3: return vals
    srt = sorted(vals)
    if srt[1] - srt[0] > 0.15:
        return [v for v in vals if v != srt[0]]
    return vals


def gather():
    out = {}
    for ds_key, lbl, dgr, ddir, ds_dir_lc in DATASETS:
        # Naive temp sweep
        sweep = []
        for t in TEMPS:
            vals = []
            for s in SEEDS:
                v = read_passk(TEMP_ROOT / ds_key / 'naive_parallel' / f'temp_{t}' / f'run_{s}')
                if v is not None: vals.append(v)
            vals = drop_hotpotqa_outlier(vals, ds_key)
            n = len(vals)
            m  = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            sweep.append((t, m, sem, n))

        # DIFFUSE (main table, default τ)
        diff_vals = []
        for s in SEEDS:
            v = read_passk(ROOT / dgr / 'qwen3-8b' / ds_dir_lc / ddir / f'run_{s}')
            if v is not None: diff_vals.append(v)
        diff_vals = drop_hotpotqa_outlier(diff_vals, ds_key)
        dm  = float(np.mean(diff_vals))
        dsd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
        dsem = dsd / np.sqrt(len(diff_vals)) if len(diff_vals) > 1 else 0.0
        out[ds_key] = {'label': lbl, 'sweep': sweep, 'diffuse': (dm, dsem)}
    return out


def main():
    data = gather()
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    for ax, (ds_key, _, _, _, _) in zip(axes, DATASETS):
        d = data[ds_key]
        xs = np.array([t for t, _, _, _ in d['sweep']])
        ys = np.array([m for _, m, _, _ in d['sweep']]) * 100
        es = np.array([sem for _, _, sem, _ in d['sweep']]) * 100
        dm, dsem = d['diffuse']
        dm  *= 100; dsem *= 100

        # Naive band + line
        ax.fill_between(xs, ys - es, ys + es, color=IRON, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(xs, ys, marker='o', color=IRON, linewidth=2.0, markersize=6,
                markeredgecolor='white', markeredgewidth=0.8,
                zorder=3, label='Standard')

        # DIFFUSE dashed reference (at default τ; constant across the panel for context)
        ax.axhline(dm, color=CARNEGIE, linestyle=(0, (4, 3)), linewidth=2.0, zorder=4, label='Ours')
        # Optional: faint shaded band for DIFFUSE SEM
        if dsem > 0:
            ax.axhspan(dm - dsem, dm + dsem, color=CARNEGIE, alpha=0.10, zorder=2)

        # Axes / styling
        ax.set_xlabel(r'Temperature  $\tau$', fontsize=11)
        ax.set_title(d['label'], fontsize=12, pad=4)
        ax.set_xticks(TEMPS)
        ax.set_xlim(0.4, 2.1)

        all_y = list(ys + es) + list(ys - es) + [dm + dsem, dm - dsem]
        y_lo = min(all_y) - 1.5; y_hi = max(all_y) + 1.5
        # Round to nice values
        y_lo = max(0, np.floor(y_lo / 5) * 5)
        y_hi = np.ceil(y_hi / 5) * 5
        ax.set_ylim(y_lo, y_hi)

        ax.tick_params(axis='both', labelsize=10)
        ax.grid(False)
        for sp in ('top','right'): ax.spines[sp].set_visible(False)
        for sp in ('left','bottom'):
            ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1.0)

    # Y-label only on leftmost panel
    axes[0].set_ylabel(r'pass@4  (%)', fontsize=11)
    # Single shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.04),
               handlelength=2.0, handletextpad=0.5, columnspacing=2.0)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / 'fig_temperature_sweep.pdf'
    png = OUT_DIR / 'fig_temperature_sweep.png'
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {pdf}")
    print(f"Wrote: {png}")

    # Numerical summary
    print("\nSummary (pass@4, %):")
    print(f"{'Dataset':<12} {'τ=0.5':<8} {'τ=1.0':<8} {'τ=1.5':<8} {'τ=2.0':<8} {'DIFFUSE':<8} {'Δ_vs_max_τ':<10}")
    for ds_key, _, _, _, _ in DATASETS:
        d = data[ds_key]
        sw = {t: m*100 for t, m, _, _ in d['sweep']}
        dm = d['diffuse'][0] * 100
        max_naive = max(sw.values())
        print(f"{d['label']:<12} {sw[0.5]:<7.2f}  {sw[1.0]:<7.2f}  {sw[1.5]:<7.2f}  {sw[2.0]:<7.2f}  {dm:<7.2f}  +{dm-max_naive:.2f} pp")


if __name__ == '__main__':
    main()
