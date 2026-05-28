"""Build the final aggregation comparison table.

For each (model, dataset, cond) group across seeds, computes mean ± std
accuracy for each method, writes a Markdown table.

Heuristic methods (pass/mv/wmv/bon/fewtool): re-derive accuracy from each
slice's aggin (use auto_judge.correctness which is pre-judged by gpt-4o-mini).
LLM methods (solagg/summagg/aggagent): read posthoc_summary_k{k}.json or
re-derive from <strategy>_logs_k4_judged.jsonl.
"""

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


HEURISTIC = ["pass", "mv", "wmv", "bon", "fewtool", "anchorbreak_free"]
LLM = ["solagg", "summagg", "aggagent", "anchorbreak"]


_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
         "was", "were", "be", "by", "with", "at", "from", "as", "that", "this",
         "it", "its", "are", "what", "who", "where", "when", "which", "how"}


def _query_tokens(thread: dict) -> set:
    """Token set drawn from this thread's search queries (q-level Jaccard primitive)."""
    queries = []
    for m in thread.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if "query" in args:
                        queries.append(str(args["query"]))
                except Exception:
                    pass
    text = " ".join(queries).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return {w for w in text.split() if w and w not in _STOP and len(w) > 2}


def _load_threads(aggin_dir: Path) -> dict:
    """Load every thread JSON for a slice, grouped by question. Returns
    {question: [thread_dict, ...]}. Each thread_dict is trimmed to the fields
    heuristics need."""
    out = defaultdict(list)
    for thread_dir in sorted(aggin_dir.iterdir()):
        if not thread_dir.is_dir():
            continue
        for f in thread_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            n_tool_calls = sum(
                1 for m in d.get("messages", [])
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            out[d["question"]].append({
                "prediction": d.get("prediction", ""),
                "is_correct": d["auto_judge"]["correctness"] == "correct",
                "confidence": d["auto_judge"].get("confidence") or 0.0,
                "n_tools": n_tool_calls,
                "query_tokens": _query_tokens(d),
            })
    return out


def _divergence_scores(threads: list[dict]) -> list[float]:
    """For each thread i: 1 − mean Jaccard similarity of i's query-token set
    against the other k−1 threads'. Operates at the query level — what the
    agent CHOSE to search for — matching T3's diversity-prompt level."""
    sets = [t["query_tokens"] for t in threads]
    n = len(sets)
    out = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            A, B = sets[i], sets[j]
            if not A and not B:
                sims.append(1.0)
            elif not A or not B:
                sims.append(0.0)
            else:
                sims.append(len(A & B) / len(A | B))
        out.append(1.0 - (sum(sims) / len(sims) if sims else 0.0))
    return out


def heuristic_acc_all(aggin_dir: Path, k: int = 4) -> dict:
    """Compute all heuristic accuracies from one pass over the slice's files."""
    from collections import Counter
    threads_by_q = _load_threads(aggin_dir)
    counts = {s: [0, 0] for s in HEURISTIC}  # strategy -> [n_correct, n_total]
    for q, threads in threads_by_q.items():
        if len(threads) != k:
            continue
        for s in HEURISTIC:
            counts[s][1] += 1

        corrects = [t["is_correct"] for t in threads]
        confs = [t["confidence"] for t in threads]
        answers = [t["prediction"] for t in threads]

        # pass@k
        if any(corrects):
            counts["pass"][0] += 1

        # MV
        c = Counter(answers)
        mv_ans = c.most_common(1)[0][0]
        mv_idx = next(i for i, t in enumerate(threads) if t["prediction"] == mv_ans)
        if threads[mv_idx]["is_correct"]:
            counts["mv"][0] += 1

        # WMV (confidence-weighted)
        scores = defaultdict(float)
        for ans, conf in zip(answers, confs):
            scores[ans] += conf
        wmv_ans = max(scores, key=scores.get)
        wmv_idx = next(i for i, t in enumerate(threads) if t["prediction"] == wmv_ans)
        if threads[wmv_idx]["is_correct"]:
            counts["wmv"][0] += 1

        # BoN (highest conf)
        bon_idx = max(range(k), key=lambda i: confs[i])
        if threads[bon_idx]["is_correct"]:
            counts["bon"][0] += 1

        # FewTool (fewest tool calls)
        ft_idx = min(range(k), key=lambda i: threads[i]["n_tools"])
        if threads[ft_idx]["is_correct"]:
            counts["fewtool"][0] += 1

        # AnchorBreak-free: deterministic, no LLM at inference.
        # MV when unanimous, else argmax(jaccard_div × qwen_conf).
        if len(set(answers)) == 1:
            ab_idx = 0  # unanimous; all threads same answer
        else:
            divs = _divergence_scores(threads)
            ab_scores = [divs[i] * (confs[i] / 100.0) for i in range(k)]
            if max(ab_scores) <= 0:
                ab_idx = bon_idx  # fall back to highest confidence
            else:
                ab_idx = max(range(k), key=lambda i: ab_scores[i])
        if threads[ab_idx]["is_correct"]:
            counts["anchorbreak_free"][0] += 1

    return {s: (c / t if t else 0.0) for s, (c, t) in counts.items()}


def llm_acc_from_judged(aggout_dir: Path, strategy: str, k: int = 4) -> float | None:
    """Read post-hoc judged log; return accuracy or None if file missing."""
    f = aggout_dir / strategy / f"{strategy}_logs_k{k}_judged.jsonl"
    if not f.exists():
        return None
    n_correct = 0
    n_total = 0
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            j = entry.get("posthoc_judge", {})
            if j.get("judge_text") == "MISSING_GOLD":
                continue
            n_total += 1
            if j.get("is_correct"):
                n_correct += 1
    return n_correct / n_total if n_total else None


def anchorbreak_llm_acc(aggin_dir: Path,
                        ab_llm_root: Path,
                        model: str, ds: str, cond: str, seed: str,
                        k: int = 4) -> float | None:
    """Read AnchorBreak-LLM picks for this slice and compute accuracy from
    the picked thread's pre-judged correctness."""
    picks_file = ab_llm_root / model / ds / cond / seed / "picks.jsonl"
    if not picks_file.exists():
        return None

    threads_by_q = {}
    for tdir in sorted(aggin_dir.iterdir()):
        if not tdir.is_dir():
            continue
        for f in tdir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue  # one corrupt file from converter; skip
            q = d["question"]
            threads_by_q.setdefault(q, []).append(d)

    n_correct = 0
    n_total = 0
    with open(picks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r["question"]
            chosen = r.get("chosen", -1)
            threads = threads_by_q.get(q)
            if not threads or len(threads) != k or chosen < 0 or chosen >= k:
                continue
            n_total += 1
            if threads[chosen]["auto_judge"]["correctness"] == "correct":
                n_correct += 1
    return n_correct / n_total if n_total else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/data/user_data/ssmurali/aggregation_manifest.tsv")
    p.add_argument("--out", default="/home/ssmurali/t3-testbed/results/aggregation_summary.md")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    # group slices by (model, dataset, cond)
    groups = defaultdict(list)  # (model,ds,cond) -> [(seed, aggin, aggout), ...]
    with open(args.manifest) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            _, model, ds, cond, seed, aggin, aggout = parts
            groups[(model, ds, cond)].append((seed, Path(aggin), Path(aggout)))

    all_methods = HEURISTIC + LLM
    rows = []  # (model, ds, cond, {method -> [acc per seed]})
    n_groups = len(groups)
    import time
    t0 = time.time()
    for gi, ((model, ds, cond), slices) in enumerate(sorted(groups.items())):
        print(f"  [{gi+1}/{n_groups}] {model}/{ds}/{cond}  ({len(slices)} seeds)  {time.time()-t0:.0f}s", flush=True)
        per_method = {m: [] for m in all_methods}
        for seed, aggin, aggout in slices:
            try:
                heur = heuristic_acc_all(aggin, k=args.k)
                for m in HEURISTIC:
                    per_method[m].append(heur[m])
            except Exception as e:
                print(f"[warn] heuristics on {aggin}: {e}")
            for m in LLM:
                if m == "anchorbreak":
                    a = anchorbreak_llm_acc(aggin, Path("/data/user_data/ssmurali/anchorbreak"),
                                            model, ds, cond, seed, k=args.k)
                else:
                    a = llm_acc_from_judged(aggout, m, k=args.k)
                if a is not None:
                    per_method[m].append(a)
        rows.append((model, ds, cond, per_method))

    # write markdown
    lines = []
    lines.append(f"# Aggregation summary (k={args.k})\n")
    header = "| model | dataset | cond | seeds | " + " | ".join(all_methods) + " |"
    sep = "|" + "|".join(["---"] * (4 + len(all_methods))) + "|"
    lines.append(header)
    lines.append(sep)
    for (model, ds, cond, per_method) in rows:
        n_seeds = max(len(per_method[m]) for m in all_methods)
        cells = []
        for m in all_methods:
            vals = per_method[m]
            if not vals:
                cells.append("—")
            elif len(vals) == 1:
                cells.append(f"{vals[0]*100:.1f}")
            else:
                cells.append(f"{stats.mean(vals)*100:.1f}±{stats.pstdev(vals)*100:.1f}")
        lines.append(f"| {model} | {ds} | {cond} | {n_seeds} | " + " | ".join(cells) + " |")

    # also per-cond avg across datasets, per model
    lines.append("\n## Per-model avg across datasets\n")
    lines.append(header)
    lines.append(sep)
    per_model_cond = defaultdict(lambda: defaultdict(list))  # (model,cond) -> method -> [acc]
    for (model, ds, cond, per_method) in rows:
        for m in all_methods:
            per_model_cond[(model, cond)][m].extend(per_method[m])
    for (model, cond), per_method in sorted(per_model_cond.items()):
        cells = []
        for m in all_methods:
            vals = per_method[m]
            if not vals:
                cells.append("—")
            else:
                cells.append(f"{stats.mean(vals)*100:.1f}±{stats.pstdev(vals)*100:.1f}")
        lines.append(f"| {model} | (all) | {cond} | {len(vals)} | " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}  ({len(rows)} (model,ds,cond) groups)")
    print()
    for l in lines[:30]:
        print(l)


if __name__ == "__main__":
    main()
