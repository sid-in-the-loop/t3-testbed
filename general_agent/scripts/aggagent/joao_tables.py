"""Render per-method tables in Joao's requested format:

  | model | dataset | Std pass@4 | Std agg | DivInit pass@4 | DivInit agg |

One table per aggregation method. Std = naive_k4, DivInit = div_k4.
Reads the same per-slice data the summary builder uses.
"""

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, "/home/ssmurali/t3-testbed/general_agent/scripts/aggagent")
from build_summary import (
    HEURISTIC, LLM, heuristic_acc_all, llm_acc_from_judged, anchorbreak_llm_acc,
)


METHOD_LABEL = {
    "mv": "MV", "wmv": "WMV", "bon": "BoN", "fewtool": "FewTool",
    "anchorbreak_free": "AnchorBreak-free",
    "solagg": "SolAgg", "summagg": "SummAgg", "aggagent": "AggAgent",
    "anchorbreak": "AnchorBreak",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/data/user_data/ssmurali/aggregation_manifest.tsv")
    p.add_argument("--ab-llm-root", default="/data/user_data/ssmurali/anchorbreak")
    p.add_argument("--out", default="/home/ssmurali/t3-testbed/results/joao_tables.md")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    # group: (model, dataset, cond) -> list of (seed, aggin, aggout)
    groups = defaultdict(list)
    with open(args.manifest) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7: continue
            _, model, ds, cond, seed, aggin, aggout = parts
            groups[(model, ds, cond)].append((seed, Path(aggin), Path(aggout)))

    # cache per-(model, ds, cond) per-method mean acc
    cell = {}  # (model, ds, cond) -> {"pass": x, "mv": x, ...}
    n_groups = len(groups)
    import time; t0 = time.time()
    for gi, ((model, ds, cond), slices) in enumerate(sorted(groups.items())):
        print(f"  [{gi+1}/{n_groups}] {model}/{ds}/{cond}  {time.time()-t0:.0f}s", flush=True)
        per_method = defaultdict(list)
        for seed, aggin, aggout in slices:
            try:
                heur = heuristic_acc_all(aggin, k=args.k)
                for m, v in heur.items():
                    per_method[m].append(v)
            except Exception as e:
                print(f"    [warn] heur: {e}")
            for m in LLM:
                try:
                    if m == "anchorbreak":
                        a = anchorbreak_llm_acc(aggin, Path(args.ab_llm_root), model, ds, cond, seed, k=args.k)
                    else:
                        a = llm_acc_from_judged(aggout, m, k=args.k)
                    if a is not None:
                        per_method[m].append(a)
                except Exception as e:
                    print(f"    [warn] {m}: {e}")
        cell[(model, ds, cond)] = {m: per_method[m] for m in per_method}

    # also need pass@k (heuristic_acc_all already returns it)
    def mean_std(vals):
        if not vals: return None, None
        if len(vals) == 1: return vals[0], 0.0
        return stats.mean(vals), stats.pstdev(vals)

    def fmt(vals):
        m, s = mean_std(vals)
        if m is None: return "—"
        if s == 0: return f"{m*100:.1f}"
        return f"{m*100:.1f}±{s*100:.1f}"

    # collect all (model, dataset) pairs and check both conds present
    md_pairs = sorted({(m, d) for (m, d, c) in cell.keys()})
    methods_in_order = [m for m in HEURISTIC + LLM if m != "pass"]

    out_lines = [f"# Aggregation results per method (k={args.k})", ""]
    out_lines.append(f"_Std = naive_k{args.k}, DivInit = div_k{args.k} (T3 diversity prompting). "
                     f"Each row = (model, dataset). Cells = mean ± std across seeds._\n")

    for method in methods_in_order:
        out_lines.append(f"## {METHOD_LABEL[method]}")
        out_lines.append("")
        out_lines.append(f"| model | dataset | Std pass@{args.k} | Std {METHOD_LABEL[method]} | DivInit pass@{args.k} | DivInit {METHOD_LABEL[method]} |")
        out_lines.append("|---|---|---|---|---|---|")
        for (model, ds) in md_pairs:
            std = cell.get((model, ds, f"naive_k{args.k}"), {})
            div = cell.get((model, ds, f"div_k{args.k}"), {})
            out_lines.append(
                f"| {model} | {ds} | "
                f"{fmt(std.get('pass'))} | {fmt(std.get(method))} | "
                f"{fmt(div.get('pass'))} | {fmt(div.get(method))} |"
            )
        out_lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {args.out}")
    print("\n--- preview ---")
    for line in out_lines[:50]:
        print(line)


if __name__ == "__main__":
    main()
