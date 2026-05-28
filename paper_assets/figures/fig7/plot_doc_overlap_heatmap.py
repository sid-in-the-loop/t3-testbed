"""Heatmap of inter-thread doc-overlap.
3 models × 5 benchmarks. Each cell is a SINGLE square split diagonally:
  upper-left triangle = Ours, lower-right triangle = Naive.
CMU Core palette: white → Carnegie Red sequential."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, '/home/ssmurali/t3-testbed/paper_assets/figures/_shared')
from paper_style import AXIS, apply_paper_style, save_fig

CSV_PATH = Path('/home/ssmurali/t3-testbed/paper_assets/analysis_threadid/doc_overlap_full_imputed.csv')

MODELS = [('qwen3-4b',   'Qwen3-4B'),
          ('qwen3-8b',   'Qwen3-8B'),
          ('gemma3-12b', 'Gemma3-12B')]
BENCHES = [('mhqa',      'musique',         'MuSiQue'),
           ('mhqa',      '2wikimultihopqa', '2Wiki'),
           ('mhqa',      'bamboogle',       'Bamboogle'),
           ('reasoning', 'gaia',            'GAIA'),
           ('reasoning', 'webwalker',       'WebWalker')]


def load_value(rows, model, group, dataset, key):
    for r in rows:
        if r['model'] == model and r['group'] == group and r['dataset'] == dataset:
            try: return float(r[key])
            except: return float('nan')
    return float('nan')


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    n_models = len(MODELS); n_bench = len(BENCHES)
    M_ours  = np.full((n_models, n_bench), np.nan)
    M_naive = np.full((n_models, n_bench), np.nan)
    for i, (mk, _) in enumerate(MODELS):
        for j, (g, dk, _) in enumerate(BENCHES):
            M_ours[i, j]  = load_value(rows, mk, g, dk, 'div_overlap') * 100.0
            M_naive[i, j] = load_value(rows, mk, g, dk, 'naive_overlap_imputed') * 100.0

    apply_paper_style()
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 1, left=0.13, right=0.88, top=0.78, bottom=0.22)
    ax = fig.add_subplot(gs[0, 0])

    cmap = LinearSegmentedColormap.from_list("cmu_red_seq",
        ["#FFFFFF", "#F4D5DA", "#C41230", "#5C0A14"])
    norm = Normalize(vmin=0, vmax=100)

    HALF = 0.5
    border_color = "#666666"
    diag_color   = "#555555"

    for i in range(n_models):
        for j in range(n_bench):
            cx, cy = j, i
            ours  = M_ours[i, j]
            naive = M_naive[i, j]

            # Upper-left triangle (Ours): corners (TL, TR, BL)
            tri_ul = Polygon([(cx - HALF, cy - HALF),
                              (cx + HALF, cy - HALF),
                              (cx - HALF, cy + HALF)],
                             facecolor=cmap(norm(ours)) if np.isfinite(ours) else 'white',
                             edgecolor='none', zorder=2)
            ax.add_patch(tri_ul)

            # Lower-right triangle (Naive): corners (TR, BR, BL)
            tri_lr = Polygon([(cx + HALF, cy - HALF),
                              (cx + HALF, cy + HALF),
                              (cx - HALF, cy + HALF)],
                             facecolor=cmap(norm(naive)) if np.isfinite(naive) else 'white',
                             edgecolor='none', zorder=2)
            ax.add_patch(tri_lr)

            # Diagonal split (top-right corner → bottom-left corner)
            ax.plot([cx + HALF, cx - HALF], [cy - HALF, cy + HALF],
                    color=diag_color, linewidth=0.6, zorder=3)

            # Annotations: Ours in UL triangle, Naive in LR triangle
            # Place at the triangle centroid
            def tcolor(v):
                if not np.isfinite(v): return "#888888"
                return "white" if norm(v) > 0.55 else "#222222"
            if np.isfinite(ours):
                ax.text(cx - 0.20, cy - 0.20, f"{ours:.0f}", ha='center', va='center',
                        fontsize=9, color=tcolor(ours), zorder=4)
            if np.isfinite(naive):
                ax.text(cx + 0.20, cy + 0.20, f"{naive:.0f}", ha='center', va='center',
                        fontsize=9, color=tcolor(naive), zorder=4)

            # Cell outline
            ax.add_patch(Rectangle((cx - HALF, cy - HALF), 1.0, 1.0,
                                   facecolor='none', edgecolor=border_color,
                                   linewidth=0.4, zorder=4))

    ax.set_xlim(-HALF, n_bench - HALF)
    ax.set_ylim(n_models - HALF, -HALF)  # invert y so models go top-to-bottom

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([m for _, m in MODELS], fontsize=10)
    ax.set_xticks([])

    # Benchmark labels above each column
    for j, (_, _, lbl) in enumerate(BENCHES):
        ax.text(j, -HALF - 0.18, lbl, ha='center', va='center', fontsize=10.5,
                color=AXIS, fontweight='bold')

    ax.tick_params(axis='both', which='both', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.018, ticks=[0, 25, 50, 75, 100])
    cbar.set_label("Inter-thread doc overlap  (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor(border_color)
    cbar.outline.set_linewidth(0.5)

    # -------- diagonal-split convention legend --------
    # Drawn in figure-coordinate space; matches the heatmap diagonal direction
    # (top-right → bottom-left of the square).
    # In figure coords y goes UP. So:
    #   TL = (x0, y0+ch)        TR = (x0+cw, y0+ch)
    #   BL = (x0, y0)           BR = (x0+cw, y0)
    # Diagonal TR→BL splits into:
    #   upper-LEFT triangle = TL + TR + BL  (contains TL corner)
    #   lower-RIGHT triangle = TR + BR + BL  (contains BR corner)
    fig_w, fig_h = fig.get_size_inches()
    cell_in = 0.22
    cw, ch = cell_in / fig_w, cell_in / fig_h
    x0, y0 = 0.38, 0.04
    # Upper-LEFT triangle (Ours, light pink)
    fig.patches.append(Polygon(
        [(x0, y0 + ch), (x0 + cw, y0 + ch), (x0, y0)],
        facecolor="#F4D5DA", edgecolor='none',
        transform=fig.transFigure, figure=fig))
    # Lower-RIGHT triangle (Naive, Carnegie Red)
    fig.patches.append(Polygon(
        [(x0 + cw, y0 + ch), (x0 + cw, y0), (x0, y0)],
        facecolor="#C41230", edgecolor='none',
        transform=fig.transFigure, figure=fig))
    # Outline
    fig.patches.append(Rectangle(
        (x0, y0), cw, ch,
        facecolor='none', edgecolor=border_color, linewidth=0.5,
        transform=fig.transFigure, figure=fig))
    # Diagonal: top-right corner → bottom-left corner
    fig.add_artist(plt.Line2D(
        [x0 + cw, x0], [y0 + ch, y0],
        color=diag_color, linewidth=0.6,
        transform=fig.transFigure, figure=fig))
    # Text labels
    fig.text(x0 - 0.02, y0 + ch / 2, "Each cell:",
             ha='right', va='center', fontsize=9, color=AXIS, style='italic')
    fig.text(x0 + cw + 0.02, y0 + ch / 2,
             "Ours (upper-left)   ·   Naive (lower-right)",
             ha='left', va='center', fontsize=9, color=AXIS)

    save_fig(fig, "doc_overlap_heatmap", subdir="fig7")
    plt.close()
    print("Wrote: paper_assets/figures/fig7/doc_overlap_heatmap.{pdf,png}")


if __name__ == '__main__':
    main()
