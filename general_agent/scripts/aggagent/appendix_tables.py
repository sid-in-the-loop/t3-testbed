"""Generate appendix tables for the EMNLP write-up.

1. Find the BEST free-method and BEST LLM-method by avg Δ(div − naive)
   across all (model, dataset) cells.
2. Write a markdown file that:
   - Reports the winners + their average div advantage
   - Shows per-model tables in the same format as the main paper (Naive@4 vs Div@4)
   - Ends with a prompt block for the overleaf-writer agent

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


FREE_METHODS = ["mv", "wmv", "bon", "fewtool", "anchorbreak_free"]
LLM_METHODS  = ["solagg", "summagg", "aggagent", "anchorbreak"]
ALL_METHODS  = FREE_METHODS + LLM_METHODS

LABEL = {
    "mv": "MV", "wmv": "WMV", "bon": "BoN", "fewtool": "FewTool",
    "anchorbreak_free": "AnchorBreak-free",
    "solagg": "SolAgg", "summagg": "SummAgg", "aggagent": "AggAgent",
    "anchorbreak": "AnchorBreak (LLM)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/data/user_data/ssmurali/aggregation_manifest.tsv")
    p.add_argument("--ab-llm-root", default="/data/user_data/ssmurali/anchorbreak")
    p.add_argument("--out", default="/home/ssmurali/t3-testbed/results/appendix_aggregation_tables.md")
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
    cell = {}
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
                except Exception:
                    pass
        cell[(model, ds, cond)] = {m: stats.mean(per_method[m]) * 100 if per_method[m] else None
                                    for m in per_method}

    # Compute avg Δ per method
    md_pairs = sorted({(m, d) for (m, d, c) in cell.keys()})
    method_deltas = {m: [] for m in ALL_METHODS}
    method_deltas["pass"] = []
    for (model, ds) in md_pairs:
        naive = cell.get((model, ds, f"naive_k{args.k}"), {})
        div   = cell.get((model, ds, f"div_k{args.k}"), {})
        for m in list(ALL_METHODS) + ["pass"]:
            if naive.get(m) is not None and div.get(m) is not None:
                method_deltas[m].append(div[m] - naive[m])

    def avg(lst):
        return stats.mean(lst) if lst else 0.0

    free_ranked = sorted(FREE_METHODS, key=lambda m: -avg(method_deltas[m]))
    llm_ranked  = sorted(LLM_METHODS,  key=lambda m: -avg(method_deltas[m]))

    best_free = free_ranked[0]
    best_llm  = llm_ranked[0]

    # Write the appendix md
    L = []
    L.append(f"# Appendix — Aggregation under Diversity Prompting (k={args.k})\n")
    L.append("## Headline numbers\n")
    L.append(f"Average Δ = (DivInit − Standard) over all {len(md_pairs)} (model, dataset) cells:\n")
    L.append("| Method | Avg Δ (pp) |")
    L.append("|---|---|")
    L.append(f"| **pass@{args.k}** (reference) | **{avg(method_deltas['pass']):+.2f}** |")
    L.append("| _Free methods_ | |")
    for m in free_ranked:
        L.append(f"| {LABEL[m]} | {avg(method_deltas[m]):+.2f} |")
    L.append("| _LLM-based methods_ | |")
    for m in llm_ranked:
        L.append(f"| {LABEL[m]} | {avg(method_deltas[m]):+.2f} |")
    L.append("")
    L.append(f"**Best free aggregator under diversity: `{LABEL[best_free]}`** (Δ = {avg(method_deltas[best_free]):+.2f} pp)")
    L.append(f"**Best LLM aggregator under diversity: `{LABEL[best_llm]}`** (Δ = {avg(method_deltas[best_llm]):+.2f} pp)")
    L.append("")

    # Now: one table per model, showing Naive vs Div for pass@k, best-free, best-llm
    L.append("## Per-model tables\n")
    L.append(f"Each cell: aggregated answer accuracy (single-prediction). "
             f"`Standard@{args.k}` = naive parallel sampling. "
             f"`DivInit@{args.k}` = T3 diversity prompting. "
             f"Reference column `pass@{args.k}` shows the oracle ceiling.\n")

    models = sorted({m for (m, d) in md_pairs})
    for model in models:
        L.append(f"### {model}")
        L.append("")
        L.append(f"| Dataset | Standard pass@{args.k} | DivInit pass@{args.k} | "
                 f"Standard {LABEL[best_free]} | DivInit {LABEL[best_free]} | "
                 f"Standard {LABEL[best_llm]} | DivInit {LABEL[best_llm]} |")
        L.append("|---|---|---|---|---|---|---|")
        for (m, ds) in md_pairs:
            if m != model: continue
            naive = cell.get((m, ds, f"naive_k{args.k}"), {})
            div   = cell.get((m, ds, f"div_k{args.k}"), {})

            def f(x):
                return f"{x:.1f}" if x is not None else "—"

            # bold the div value if it's higher
            def pair(n, d):
                ns, ds_ = f(n), f(d)
                if n is not None and d is not None and d > n:
                    return ns, f"**{ds_}**"
                return ns, ds_

            p_n, p_d = pair(naive.get("pass"), div.get("pass"))
            f_n, f_d = pair(naive.get(best_free), div.get(best_free))
            l_n, l_d = pair(naive.get(best_llm), div.get(best_llm))
            L.append(f"| {ds} | {p_n} | {p_d} | {f_n} | {f_d} | {l_n} | {l_d} |")
        L.append("")

    # Overall summary row across all (model, dataset) pairs
    L.append("## Aggregate gain summary\n")
    L.append(f"Mean pass@{args.k} Δ across all {len(md_pairs)} (model, dataset) cells: "
             f"**{avg(method_deltas['pass']):+.2f} pp**")
    L.append(f"Mean `{LABEL[best_free]}` Δ: **{avg(method_deltas[best_free]):+.2f} pp**")
    L.append(f"Mean `{LABEL[best_llm]}` Δ: **{avg(method_deltas[best_llm]):+.2f} pp**")
    L.append("")
    diff_free = avg(method_deltas[best_free]) - avg(method_deltas['pass'])
    diff_llm  = avg(method_deltas[best_llm])  - avg(method_deltas['pass'])
    L.append(f"The best LLM aggregator harvests **{avg(method_deltas[best_llm])/max(avg(method_deltas['pass']), 1e-9)*100:.0f}%** "
             f"of the diversity gain that pass@{args.k} shows.")
    L.append(f"The best free aggregator harvests **{avg(method_deltas[best_free])/max(avg(method_deltas['pass']), 1e-9)*100:.0f}%**.")
    L.append("")

    L.append("---")
    L.append("")
    L.append("## Prompt for Claude (overleaf writeup)\n")
    L.append("```")
    L.append("Add the following to the appendix of the T3 paper, in a section titled")
    L.append('"Aggregation accuracy under diversity prompting".')
    L.append("")
    L.append("Use the per-model tables above verbatim. Frame the contribution as:")
    L.append("")
    L.append(f"  Diversity prompting (T3) does not just lift pass@k — it also lifts the")
    L.append(f"  performance of downstream aggregation methods that pick a single answer")
    L.append(f"  from the k parallel rollouts. We report two representative aggregators:")
    L.append(f"  the best non-LLM heuristic ({LABEL[best_free]}) and the best LLM-based")
    L.append(f"  aggregator ({LABEL[best_llm]}) drawn from prior work. The choice is by")
    L.append(f"  largest mean Δ(div − naive) across all 5 model × 7 dataset cells.")
    L.append("")
    L.append(f"  Free aggregation gain: +{avg(method_deltas[best_free]):.1f} pp avg")
    L.append(f"  LLM aggregation gain : +{avg(method_deltas[best_llm]):.1f} pp avg")
    L.append(f"  Pass@{args.k} reference   : +{avg(method_deltas['pass']):.1f} pp avg")
    L.append("")
    L.append(f"  The LLM aggregator captures ~{avg(method_deltas[best_llm])/max(avg(method_deltas['pass']), 1e-9)*100:.0f}% of the diversity-induced")
    L.append(f"  pass@{args.k} headroom. The free aggregator captures ~{avg(method_deltas[best_free])/max(avg(method_deltas['pass']), 1e-9)*100:.0f}%.")
    L.append("")
    L.append("Use 2-3 sentences per per-model paragraph, then a one-line takeaway after")
    L.append("the table. Do NOT pad — keep it tight. Bold the cell where diversity")
    L.append("strictly beats standard.")
    L.append("```")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
