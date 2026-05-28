"""
Shared paper-figure styling for t3-testbed.

Single source of truth for color constants, matplotlib rcParams, and the
save_fig() helper that writes paired {pdf, png} into paper_assets/figures/<subdir>/.

Usage from any plot script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
    from paper_style import NAIVE, S3, SEQ, REF, GRID, AXIS, apply_paper_style, save_fig

    apply_paper_style()
    fig, ax = plt.subplots(...)
    # ... plot stuff with NAIVE / S3 colors ...
    save_fig(fig, "turn1_imprint", subdir="fig1")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---------- color constants — CMU Core palette ----------
NAIVE = "#6D6E71"   # Iron Gray (CMU Core) — naive parallel baseline
S3    = "#C41230"   # Carnegie Red (CMU Core) — Ours (diversity-forced)
SEQ   = "#E0E0E0"   # Steel Gray (CMU Core) — sequential baseline
REF   = "#E0E0E0"   # Steel Gray — dashed reference lines
GRID  = "#EEEEEE"   # near-Steel-Gray gridlines (slightly lighter for subtlety)
AXIS  = "#000000"   # Black (CMU Core) — axes / spines / ticks

# repo root: paper_assets/figures/_shared/paper_style.py -> testbed
_SHARED_DIR = Path(__file__).resolve().parent
FIGURES_ROOT = _SHARED_DIR.parent                # paper_assets/figures/
PAPER_ASSETS = FIGURES_ROOT.parent               # paper_assets/
TESTBED_ROOT = PAPER_ASSETS.parent               # repo root


def apply_paper_style() -> None:
    """Set matplotlib rcParams for paper-quality figures.

    sans-serif (DejaVu Sans, falls back to Helvetica/system sans),
    11pt axis labels, 9pt ticks, thin AXIS-color spines (left+bottom only),
    no top/right spines, faint horizontal gridlines (GRID, 0.5pt),
    white background, 300 dpi savefig, embeddable Type-42 fonts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Fonts
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size":          11,
        "axes.titlesize":     11,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,

        # Spines / axis
        "axes.linewidth":     1.0,
        "axes.edgecolor":     AXIS,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,

        # Ticks
        "xtick.color":        AXIS,
        "ytick.color":        AXIS,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,

        # Gridlines (faint horizontal — scripts can override per-axis)
        "axes.grid":          False,
        "grid.color":         GRID,
        "grid.linewidth":     0.5,

        # Background
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",

        # Output — high DPI for paper-quality PNG previews; PDF is vector
        "figure.dpi":         150,
        "savefig.dpi":        1400,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def save_fig(fig, name: str, subdir: str, *,
             also_dirs: Optional[list] = None) -> Path:
    """Write {name}.pdf and {name}.png into paper_assets/figures/<subdir>/.

    Args:
        fig: matplotlib Figure
        name: basename without extension, e.g. "turn1_imprint"
        subdir: target subdir under paper_assets/figures, e.g. "fig1"
        also_dirs: optional list of additional Path dirs to mirror outputs into
                   (useful for keeping a copy under results/ for backwards compat).

    Returns: the canonical PDF path.
    """
    out_dir = FIGURES_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    # PDF: vector, DPI controls rasterized portions only (e.g. dense scatter).
    # PNG: raster — bump high for crisp paper-quality previews.
    fig.savefig(pdf, dpi=1400, bbox_inches="tight")
    fig.savefig(png, dpi=1400, bbox_inches="tight")
    if also_dirs:
        for d in also_dirs:
            d = Path(d)
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{name}.pdf", dpi=1400, bbox_inches="tight")
            fig.savefig(d / f"{name}.png", dpi=1400, bbox_inches="tight")
    return pdf
