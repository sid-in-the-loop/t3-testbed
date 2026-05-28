"""Validate the AnchorBreak hypothesis:

  In the cases that matter most (MV wrong but >=1 thread correct), is the
  correct minority thread genuinely *more grounded* AND *more divergent* than
  the wrong majority threads?

If both signals are real, AnchorBreak's escape_score = divergence * groundedness
will rank the correct thread first and we can build the full method on it.

Outputs per (model, cond) and overall:
  - Counts: total Q, unanimous Q, MV-correct Q, FIXABLE Q (MV wrong, >=1 correct)
  - P(answer-in-evidence | correct thread, fixable Q)
  - P(answer-in-evidence | wrong thread,   fixable Q)
  - Mean divergence of correct vs wrong threads (fixable Q)
  - P(escape_score argmax == correct thread, fixable Q)  -- the policy test
  - Hypothetical AnchorBreak accuracy if we used the policy on fixable Q

Usage:
  python3 validate_anchorbreak.py [--conds div_k4] [--models qwen3-8b ...]
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def evidence_text(thread: dict) -> str:
    """Concat all tool result contents (the retrieved docs) for this thread."""
    parts = []
    for m in thread.get("messages", []):
        if m.get("role") == "tool":
            parts.append(m.get("content", "") or "")
    return " ".join(parts)


def search_queries_text(thread: dict) -> str:
    """Concat all search queries (what the agent searched for)."""
    parts = []
    for m in thread.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if "query" in args:
                        parts.append(str(args["query"]))
                except Exception:
                    pass
    return " ".join(parts)


def thread_grounded(thread: dict) -> bool:
    """Does the thread's predicted answer appear (normalized substring) in its evidence?"""
    pred = normalize(thread.get("prediction", ""))
    if not pred or len(pred) < 2:
        return False
    ev = normalize(evidence_text(thread))
    # Require at least a 3+ char substring match to avoid trivial single-letter matches.
    if len(pred) <= 2:
        return pred in ev
    return pred in ev


STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
             "was", "were", "be", "by", "with", "at", "from", "as", "that", "this",
             "it", "its", "are", "what", "who", "where", "when", "which", "how"}


def thread_query_set(thread: dict) -> set:
    """Set of normalized query *strings* this thread issued."""
    out = set()
    for m in thread.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    q = args.get("query")
                    if q:
                        out.add(normalize(str(q)))
                except Exception:
                    pass
    return out


def _toks(s: str) -> set:
    return {w for w in normalize(s).split() if w and w not in STOPWORDS and len(w) > 2}


def thread_token_set_q(thread: dict) -> set:
    """Token set drawn ONLY from this thread's search queries."""
    qs = []
    for m in thread.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if "query" in args:
                        qs.append(str(args["query"]))
                except Exception:
                    pass
    return _toks(" ".join(qs))


def thread_token_set_qe(thread: dict) -> set:
    """Token set drawn from this thread's queries + tool results."""
    return _toks(search_queries_text(thread) + " " + evidence_text(thread))


def _pairwise_jaccard_distance(sets: list[set]) -> list[float]:
    """For each i, return 1 - mean Jaccard sim of set_i vs the others."""
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


def divergences_all(threads: list[dict]) -> dict:
    """Return dict of {metric_name: [div_i for each thread]}."""
    return {
        "tfidf_cos": _tfidf_div(threads),
        "jac_qset":  _pairwise_jaccard_distance([thread_query_set(t)     for t in threads]),
        "jac_qtok":  _pairwise_jaccard_distance([thread_token_set_q(t)   for t in threads]),
        "jac_qetok": _pairwise_jaccard_distance([thread_token_set_qe(t)  for t in threads]),
    }


def _tfidf_div(threads: list[dict]) -> list[float]:
    docs = [search_queries_text(t) + " " + evidence_text(t) for t in threads]
    if all(not d.strip() for d in docs):
        return [0.0] * len(threads)
    try:
        X = TfidfVectorizer(max_features=4000, stop_words="english").fit_transform(docs)
    except ValueError:
        return [0.0] * len(threads)
    sim = cosine_similarity(X)
    n = len(threads)
    return [1.0 - (sum(sim[i, j] for j in range(n) if j != i) / (n - 1)) for i in range(n)]


# Kept for back-compat with earlier code paths
def thread_divergence(threads: list[dict]) -> list[float]:
    return _tfidf_div(threads)


def discover_slices(aggin_root: Path, models: list[str], conds: list[str]):
    for model in models:
        mroot = aggin_root / model
        if not mroot.exists():
            continue
        for ds in sorted(p for p in mroot.iterdir() if p.is_dir()):
            for cond in conds:
                cdir = ds / cond
                if not cdir.exists():
                    continue
                for seed in sorted(p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                    yield model, ds.name, cond, seed.name, seed


def analyze_slice(slice_dir: Path) -> dict:
    threads_by_q = defaultdict(list)
    for tdir in sorted(slice_dir.iterdir()):
        if not tdir.is_dir():
            continue
        for f in tdir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            threads_by_q[d["question"]].append(d)

    # Divergence metrics to test
    DIV_METRICS = ["tfidf_cos", "jac_qset", "jac_qtok", "jac_qetok"]
    # Each metric × {conf, alone} = 2 policies; plus conf_only for baseline.
    POLICIES = ["conf_only"] + [f"{m}__only" for m in DIV_METRICS] + [f"{m}__x_conf" for m in DIV_METRICS]

    stats = {
        "n_q": 0,
        "n_unanimous": 0,
        "n_mv_correct": 0,
        "n_fixable": 0,
        "n_passk_correct": 0,
        # confidence stats on fixable Q
        "conf_correct_sum": 0.0,
        "conf_correct_n": 0,
        "conf_wrong_sum": 0.0,
        "conf_wrong_n": 0,
        # per-divergence-metric separator stats on fixable Q
        **{f"div__{m}__correct_sum": 0.0 for m in DIV_METRICS},
        **{f"div__{m}__correct_n":   0   for m in DIV_METRICS},
        **{f"div__{m}__wrong_sum":   0.0 for m in DIV_METRICS},
        **{f"div__{m}__wrong_n":     0   for m in DIV_METRICS},
        # per-policy correctness
        **{f"argmax_correct__{p}": 0 for p in POLICIES},
        **{f"ab_correct__{p}":     0 for p in POLICIES},
    }

    for q, threads in threads_by_q.items():
        if len(threads) != 4:
            continue
        stats["n_q"] += 1

        corrects = [t["auto_judge"]["correctness"] == "correct" for t in threads]
        answers = [t["prediction"] for t in threads]

        # pass@4
        if any(corrects):
            stats["n_passk_correct"] += 1

        # unanimous?
        if all(a == answers[0] for a in answers):
            stats["n_unanimous"] += 1
            if corrects[0]:
                stats["n_mv_correct"] += 1
            continue

        # MV
        cnt = Counter(answers)
        mv_ans, _ = cnt.most_common(1)[0]
        mv_idx = next(i for i, t in enumerate(threads) if t["prediction"] == mv_ans)
        mv_is_correct = threads[mv_idx]["auto_judge"]["correctness"] == "correct"
        if mv_is_correct:
            stats["n_mv_correct"] += 1
            continue

        # If we get here: MV wrong. Fixable iff >=1 thread correct.
        if not any(corrects):
            continue
        stats["n_fixable"] += 1

        # Per-thread features
        confs = [(t["auto_judge"].get("confidence") or 0.0) / 100.0 for t in threads]
        all_divs = divergences_all(threads)  # {metric: [..k..]}

        # Aggregate per-thread separator stats
        for m in DIV_METRICS:
            for i in range(4):
                if corrects[i]:
                    stats[f"div__{m}__correct_sum"] += all_divs[m][i]
                    stats[f"div__{m}__correct_n"] += 1
                else:
                    stats[f"div__{m}__wrong_sum"] += all_divs[m][i]
                    stats[f"div__{m}__wrong_n"] += 1
        for i in range(4):
            if corrects[i]:
                stats["conf_correct_sum"] += confs[i]
                stats["conf_correct_n"] += 1
            else:
                stats["conf_wrong_sum"] += confs[i]
                stats["conf_wrong_n"] += 1

        def pick(scores):
            return max(range(4), key=lambda i: scores[i])

        # Build all candidate policies
        policy_scores = {"conf_only": confs[:]}
        for m in DIV_METRICS:
            policy_scores[f"{m}__only"]   = list(all_divs[m])
            policy_scores[f"{m}__x_conf"] = [all_divs[m][i] * confs[i] for i in range(4)]

        for pname, scores in policy_scores.items():
            if max(scores) <= 0:
                argmax_i = pick(confs)
            else:
                argmax_i = pick(scores)
            if corrects[argmax_i]:
                stats[f"argmax_correct__{pname}"] += 1
                stats[f"ab_correct__{pname}"] += 1

    return stats


def merge(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        out[k] = a[k] + b[k]
    return out


def fmt_stats(label: str, st: dict) -> str:
    n_q = max(st["n_q"], 1)
    n_fix = max(st["n_fixable"], 1)
    conf_c_n = max(st["conf_correct_n"], 1)
    conf_w_n = max(st["conf_wrong_n"], 1)
    mv_acc = st["n_mv_correct"] / n_q * 100
    pass_acc = st["n_passk_correct"] / n_q * 100
    gap = pass_acc - mv_acc
    conf_c = st["conf_correct_sum"] / conf_c_n
    conf_w = st["conf_wrong_sum"] / conf_w_n

    DIV_METRICS = ["tfidf_cos", "jac_qset", "jac_qtok", "jac_qetok"]

    out = []
    out.append(f"\n=== {label} ===")
    out.append(f"  Q={st['n_q']}  unan={st['n_unanimous']}  pass={pass_acc:.2f}%  MV={mv_acc:.2f}%  "
               f"headroom={gap:+.2f}  fixable={st['n_fixable']}")
    out.append("")
    out.append(f"  per-thread separators on fixable Q (Δ = correct − wrong)")
    out.append(f"    conf (qwen3-8b)            corr={conf_c*100:6.2f}  wrong={conf_w*100:6.2f}  Δ={(conf_c-conf_w)*100:+6.2f}pts")
    for m in DIV_METRICS:
        c_n = max(st[f"div__{m}__correct_n"], 1)
        w_n = max(st[f"div__{m}__wrong_n"], 1)
        c_m = st[f"div__{m}__correct_sum"] / c_n
        w_m = st[f"div__{m}__wrong_sum"]   / w_n
        out.append(f"    div [{m:<10}]    corr={c_m:.4f} wrong={w_m:.4f} Δ={c_m-w_m:+.4f}")
    out.append("")
    out.append(f"  policies (P(argmax==correct | fixable), AB_acc, gain vs MV)")
    out.append(f"    {'policy':<22}  {'argmax':>8}  {'AB_acc':>7}  {'gain':>7}")
    POLICIES = ["conf_only"] + [f"{m}__only" for m in DIV_METRICS] + [f"{m}__x_conf" for m in DIV_METRICS]
    best = None
    for p in POLICIES:
        argmax_pct = st[f"argmax_correct__{p}"] / n_fix * 100
        ab_acc = (st["n_mv_correct"] + st[f"ab_correct__{p}"]) / n_q * 100
        gain = ab_acc - mv_acc
        if best is None or gain > best[0]:
            best = (gain, p, ab_acc, argmax_pct)
        out.append(f"    {p:<22}  {argmax_pct:>7.2f}%  {ab_acc:>6.2f}%  {gain:>+6.2f}")
    out.append(f"    {'BEST':<22}  {best[3]:>7.2f}%  {best[2]:>6.2f}%  {best[0]:>+6.2f}   ({best[1]})")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--models", nargs="+",
                   default=["qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-12b"])
    p.add_argument("--conds", nargs="+", default=["div_k4", "naive_k4"])
    p.add_argument("--limit-slices", type=int, default=None, help="quick test")
    args = p.parse_args()

    overall = defaultdict(lambda: defaultdict(int))
    overall_per_cond = defaultdict(lambda: defaultdict(int))
    overall_per_model = defaultdict(lambda: defaultdict(int))
    all_total = defaultdict(int)

    slices = list(discover_slices(Path(args.aggin_root), args.models, args.conds))
    if args.limit_slices:
        slices = slices[:args.limit_slices]
    print(f"Analysing {len(slices)} slices...", flush=True)

    for i, (model, ds, cond, seed, slice_dir) in enumerate(slices):
        st = analyze_slice(slice_dir)
        # accumulate
        for k, v in st.items():
            overall_per_cond[cond][k] += v
            overall_per_model[(model, cond)][k] += v
            all_total[k] += v
        print(f"  [{i+1}/{len(slices)}] {model}/{ds}/{cond}/{seed}  "
              f"fix={st['n_fixable']} gap={(st['n_passk_correct']-st['n_mv_correct'])/max(st['n_q'],1)*100:+.1f}",
              flush=True)

    print(fmt_stats("OVERALL", dict(all_total)))
    for cond, st in overall_per_cond.items():
        print(fmt_stats(f"COND = {cond}", dict(st)))
    for (model, cond), st in overall_per_model.items():
        print(fmt_stats(f"{model} / {cond}", dict(st)))


if __name__ == "__main__":
    main()
