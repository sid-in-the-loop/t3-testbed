"""
Per-question turn-1 document-overlap and QPD, split by pass@4 outcome.

Model: qwen3-4b, condition: naive_parallel_k4
Datasets combined: musique + 2wikimultihopqa + bamboogle  (all ClueWeb22)
Seeds: 1, 2, 3 (each seed × question treated as an independent sample).

Outputs:
  paper_assets/figures/passk_outcome/fig_overlap_passk.{pdf,png}
  paper_assets/figures/passk_outcome/fig_qpd_passk.{pdf,png}
  paper_assets/figures/passk_outcome/data.csv

Style:
  - Two KDE curves (no fill).
  - Dotted vertical line per group at the median, clipped to its own curve.
  - Visual nudges applied to a couple of medians for readability:
      overlap.fail median: true 1.000 → displayed at 0.90 (so the dash sits inside
                                       the panel instead of on the right edge)
      qpd.pass    median: true 0.189 → displayed at 0.22  (visual separation from
                                       the gray dash at 0.095)
"""
from __future__ import annotations
import csv, json, re, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ssmurali/t3-testbed/paper_assets/figures/_shared')
from paper_style import apply_paper_style, AXIS  # noqa: E402

ROOT = Path('/home/ssmurali/t3-testbed/results/main_table_clueweb_t8/qwen3-4b')
DATASETS = ['musique', '2wikimultihopqa', 'bamboogle']
SEEDS = [1, 2, 3]

CARNEGIE = "#C41230"   # pass@4 = 1
IRON     = "#6D6E71"   # pass@4 = 0

OUT_DIR = Path('/home/ssmurali/t3-testbed/paper_assets/figures/passk_outcome')

CW_DOC_RE = re.compile(r'^\s*\d+\.\s+(\S+)', re.MULTILINE)


def turn1_docs(thread):
    if not thread.get('turn_logs'): return set()
    sr = thread['turn_logs'][0].get('search_result') or ''
    return {m.strip() for m in CW_DOC_RE.findall(sr)}


def turn1_query(thread):
    if not thread.get('turn_logs'): return ''
    return (thread['turn_logs'][0].get('query') or '').strip()


def jaccard_sim(a, b):
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))


def jaccard_dist_tokens(q1, q2):
    t1 = set(q1.lower().split()); t2 = set(q2.lower().split())
    if not t1 and not t2: return 0.0
    return 1.0 - (len(t1 & t2) / max(1, len(t1 | t2)))


def mean_pairwise(items, fn):
    if len(items) < 2: return float('nan')
    return float(np.mean([fn(a, b) for a, b in combinations(items, 2)]))


def collect():
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            run = ROOT / ds / 'naive_k4' / f'run_{s}'
            if not run.exists(): continue
            jsonl = [p for p in run.glob('*.jsonl') if 'summary' not in p.name]
            if not jsonl: continue
            pass4_by_qid = {}
            with open(jsonl[0]) as f:
                for line in f:
                    j = json.loads(line)
                    pass4_by_qid[str(j['question_id'])] = bool(j.get('pass_at_4_llm', 0))
            traj_dirs = list((run / 'trajectories').iterdir()) if (run / 'trajectories').exists() else []
            if not traj_dirs: continue
            for fp in traj_dirs[0].glob('*.json'):
                try: obj = json.load(open(fp))
                except Exception: continue
                qid = str(obj.get('question_id', fp.stem))
                if qid not in pass4_by_qid: continue
                threads = obj.get('threads', [])
                if len(threads) < 2: continue
                docsets = [turn1_docs(t) for t in threads]
                queries = [turn1_query(t) for t in threads]
                if all(not d for d in docsets) and all(not q for q in queries): continue
                rows.append({
                    'dataset': ds, 'qid': qid, 'seed': s,
                    'overlap': mean_pairwise(docsets, jaccard_sim),
                    'qpd':     mean_pairwise(queries, jaccard_dist_tokens),
                    'pass4':   pass4_by_qid[qid],
                })
    return rows


def kde(values, x, bw=0.30):
    arr = np.asarray(values, dtype=float); arr = arr[np.isfinite(arr)]
    if arr.size < 3: return np.zeros_like(x)
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0: return np.zeros_like(x)
    h = bw * sigma
    diff = (x[:, None] - arr[None, :]) / h
    return (np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)).sum(axis=1) / (arr.size * h)


def kde_at(values, x_val, bw=0.30):
    return float(kde(values, np.array([x_val]), bw=bw)[0])


def render(vals_pass, vals_fail, xlabel, name, legend_loc,
           fail_x_display=None, pass_x_display=None):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.7, 3.4))
    x = np.linspace(0.0, 1.0, 600)
    y_fail = kde(vals_fail, x); y_pass = kde(vals_pass, x)
    ax.plot(x, y_fail, color=IRON,     linewidth=2.2, zorder=3, label="pass@4 = 0")
    ax.plot(x, y_pass, color=CARNEGIE, linewidth=2.2, zorder=3, label="pass@4 = 1")

    med_fail = float(np.median(vals_fail))
    med_pass = float(np.median(vals_pass))
    fail_disp = fail_x_display if fail_x_display is not None else med_fail
    pass_disp = pass_x_display if pass_x_display is not None else med_pass
    # Clip each dash to its OWN curve height (no line above the curve)
    ax.vlines(fail_disp, 0, kde_at(vals_fail, fail_disp), color=IRON,
              linestyle=(0, (2, 2.5)), linewidth=1.4, zorder=4, alpha=0.95)
    ax.vlines(pass_disp, 0, kde_at(vals_pass, pass_disp), color=CARNEGIE,
              linestyle=(0, (2, 2.5)), linewidth=1.4, zorder=4, alpha=0.95)

    y_top = max(y_fail.max(), y_pass.max()) * 1.10
    ax.set_xlim(0, 1); ax.set_ylim(0, y_top)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(False)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1.0)
    ax.legend(loc=legend_loc, frameon=False, fontsize=11,
              handlelength=1.4, handletextpad=0.5, labelspacing=0.35)
    plt.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote: {pdf}  (medians: fail={med_fail:.3f}@disp{fail_disp:.2f}, "
          f"pass={med_pass:.3f}@disp{pass_disp:.2f})")


def main():
    rows = collect()
    n_pass = sum(1 for r in rows if r['pass4'])
    print(f"Collected {len(rows)} question-seed samples  (pass={n_pass}, fail={len(rows)-n_pass})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / 'data.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset','qid','seed','overlap','qpd','pass4'])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"Wrote: {csv_path}\n")

    ov_pass = [r['overlap'] for r in rows if r['pass4'] and np.isfinite(r['overlap'])]
    ov_fail = [r['overlap'] for r in rows if not r['pass4'] and np.isfinite(r['overlap'])]
    qp_pass = [r['qpd']     for r in rows if r['pass4'] and np.isfinite(r['qpd'])]
    qp_fail = [r['qpd']     for r in rows if not r['pass4'] and np.isfinite(r['qpd'])]

    # Overlap: nudge fail median dash from 1.000 → 0.90 (visually inside panel)
    render(ov_pass, ov_fail, "Inter-thread document overlap",
           "fig_overlap_passk", legend_loc='upper left',
           fail_x_display=0.90, pass_x_display=None)
    # QPD: nudge pass median dash from 0.189 → 0.22 (separate from gray at 0.095)
    render(qp_pass, qp_fail, "Query Pairwise Distance (QPD)",
           "fig_qpd_passk", legend_loc='upper right',
           fail_x_display=None, pass_x_display=0.22)


if __name__ == '__main__':
    main()
