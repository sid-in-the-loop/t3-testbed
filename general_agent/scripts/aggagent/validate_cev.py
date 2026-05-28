"""CEV (Cross-thread Evidence Verification) — realistic-policy validation.

Hypothesis: for each candidate answer, score it against the UNION of all k
threads' retrieved evidence. The diverse rollouts collectively retrieve a
broader doc set than any single thread; an answer well-supported across
multiple (especially divergent) threads' evidence is more likely correct.

CRITICAL difference from AnchorBreak:
  AnchorBreak ranked THREADS by structural features (div × conf), with a
  built-in bias toward minority threads. CEV ranks ANSWERS by evidence support
  -- no bias toward majority or minority by construction.

This script:
  1. Computes realistic per-question accuracy (no oracle gating) for:
       pass@4 (ceiling), MV, AB-free (sanity check), CEV-free
  2. Reports per-slice + overall numbers
  3. Returns a clear DECISION: does CEV-free beat MV by >= +1.5 pts on average?
     If yes, proceed to CEV-LLM scale-up. If no, pivot.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
        "was", "were", "be", "by", "with", "at", "from", "as", "that", "this",
        "it", "its", "are", "what", "who", "where", "when", "which", "how"}


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> set:
    return {w for w in normalize(s).split() if w and w not in STOP and len(w) > 2}


def evidence_text(thread: dict) -> str:
    parts = []
    for m in thread.get("messages", []):
        if m.get("role") == "tool":
            parts.append(m.get("content", "") or "")
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if "query" in args:
                        parts.append(str(args["query"]))
                except Exception:
                    pass
    return " ".join(parts)


def query_token_set(thread: dict) -> set:
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
    return tokens(" ".join(qs))


def jaccard_div(threads: list) -> list:
    sets = [query_token_set(t) for t in threads]
    n = len(sets)
    out = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j: continue
            A, B = sets[i], sets[j]
            if not A and not B: sims.append(1.0)
            elif not A or not B: sims.append(0.0)
            else: sims.append(len(A & B) / len(A | B))
        out.append(1.0 - (sum(sims) / len(sims) if sims else 0.0))
    return out


def load_slice(slice_dir: Path) -> dict:
    out = defaultdict(list)
    for tdir in sorted(slice_dir.iterdir()):
        if not tdir.is_dir(): continue
        for f in tdir.glob("*.json"):
            try:
                with open(f) as fh: d = json.load(fh)
            except Exception:
                continue
            out[d["question"]].append({
                "prediction": d.get("prediction", ""),
                "is_correct": d["auto_judge"]["correctness"] == "correct",
                "confidence": d["auto_judge"].get("confidence") or 0.0,
                "evidence": evidence_text(d),
                "raw": d,
            })
    return out


# ---------- methods ----------

def mv_pick(threads):
    answers = [t["prediction"] for t in threads]
    cnt = Counter(answers)
    mv_ans = cnt.most_common(1)[0][0]
    return next(i for i, t in enumerate(threads) if t["prediction"] == mv_ans)


def ab_free_pick(threads):
    """AnchorBreak-free for comparison."""
    answers = [t["prediction"] for t in threads]
    if len(set(answers)) == 1:
        return 0
    confs = [t["confidence"] / 100.0 for t in threads]
    divs = jaccard_div([t["raw"] for t in threads])
    scores = [divs[i] * confs[i] for i in range(len(threads))]
    if max(scores) <= 0:
        return max(range(len(threads)), key=lambda i: confs[i])
    return max(range(len(threads)), key=lambda i: scores[i])


def cev_free_pick(threads, use_div_weighting: bool = True):
    """CEV-free: score each candidate against the UNION of all threads' evidence.

    Scoring per (candidate, thread_j): how well does thread j's evidence
    support the candidate?
      - 1.0 if candidate appears verbatim (normalized substring) in evidence
      - token-overlap fraction
    Total support = sum over j (optionally weighted by div_j to favor
    independent corroboration).

    Argmax over distinct candidates.
    """
    answers = [t["prediction"] for t in threads]
    if len(set(answers)) == 1:
        return 0

    divs = jaccard_div([t["raw"] for t in threads]) if use_div_weighting else [1.0] * len(threads)

    # one candidate per UNIQUE answer string; remember its representative thread index
    cand_to_idx = {}
    for i, a in enumerate(answers):
        cand_to_idx.setdefault(a, i)

    scores = {}
    for cand, _ in cand_to_idx.items():
        cand_norm = normalize(cand)
        if not cand_norm:
            scores[cand] = 0.0
            continue
        cand_toks = tokens(cand)
        if not cand_toks:
            cand_toks = set(cand_norm.split())

        total = 0.0
        for j, t in enumerate(threads):
            ev_norm = normalize(t["evidence"])
            ev_toks = tokens(t["evidence"])

            substr = 1.0 if (cand_norm and cand_norm in ev_norm) else 0.0
            tok_overlap = (len(cand_toks & ev_toks) / len(cand_toks)) if cand_toks else 0.0
            support = substr + tok_overlap   # both in roughly [0, 1] range

            w = divs[j] if use_div_weighting else 1.0
            total += w * support
        scores[cand] = total

    # argmax candidate; tie-break by first-seen
    best = max(cand_to_idx.keys(), key=lambda c: (scores[c], -list(cand_to_idx.keys()).index(c)))
    return cand_to_idx[best]


# ---------- accuracy on a slice ----------

def slice_accuracies(slice_dir: Path, k: int = 4) -> dict:
    threads_by_q = load_slice(slice_dir)
    counts = {"pass": 0, "mv": 0, "ab_free": 0, "cev_free": 0, "cev_free_uw": 0, "total": 0}
    for q, threads in threads_by_q.items():
        if len(threads) != k: continue
        counts["total"] += 1
        corrects = [t["is_correct"] for t in threads]
        if any(corrects):
            counts["pass"] += 1

        for name, picker in [
            ("mv",          mv_pick),
            ("ab_free",     ab_free_pick),
            ("cev_free",    lambda ts: cev_free_pick(ts, use_div_weighting=True)),
            ("cev_free_uw", lambda ts: cev_free_pick(ts, use_div_weighting=False)),
        ]:
            try:
                i = picker(threads)
                if threads[i]["is_correct"]:
                    counts[name] += 1
            except Exception:
                pass
    return counts


# ---------- driver ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--models", nargs="+",
                   default=["qwen3-8b", "gemma3-12b"])
    p.add_argument("--datasets", nargs="+",
                   default=["bamboogle", "musique", "hotpotqa", "2wikimultihopqa", "hle"])
    p.add_argument("--conds", nargs="+", default=["div_k4", "naive_k4"])
    p.add_argument("--seeds", nargs="+", default=["run_1"])
    args = p.parse_args()

    aggin = Path(args.aggin_root)
    rows = []
    for model in args.models:
        for ds in args.datasets:
            for cond in args.conds:
                for seed in args.seeds:
                    slice_dir = aggin / model / ds / cond / seed
                    if not slice_dir.exists():
                        continue
                    c = slice_accuracies(slice_dir)
                    if c["total"] == 0:
                        continue
                    rows.append((model, ds, cond, seed, c))

    # print table
    print(f"\n{'model':<14} {'dataset':<18} {'cond':<10} {'pass':>6} {'MV':>6} {'AB-fr':>6} {'CEV':>6} {'CEV/uw':>6}  notes")
    print("-" * 100)
    sums = {k: 0 for k in ["pass", "mv", "ab_free", "cev_free", "cev_free_uw", "total"]}
    for (model, ds, cond, seed, c) in rows:
        for k in sums: sums[k] += c[k]
        pa = lambda k: c[k] / c["total"] * 100
        deltas = []
        if pa("cev_free") - pa("mv") >= 1.5: deltas.append("CEV>>MV")
        if pa("cev_free") > pa("ab_free"): deltas.append("CEV>AB")
        print(f"{model:<14} {ds:<18} {cond:<10} {pa('pass'):>5.1f}% {pa('mv'):>5.1f}% {pa('ab_free'):>5.1f}% {pa('cev_free'):>5.1f}% {pa('cev_free_uw'):>5.1f}%  {' '.join(deltas)}")
    print("-" * 100)
    if sums["total"]:
        pa = lambda k: sums[k] / sums["total"] * 100
        print(f"{'OVERALL':<14} {'':<18} {'':<10} {pa('pass'):>5.1f}% {pa('mv'):>5.1f}% {pa('ab_free'):>5.1f}% {pa('cev_free'):>5.1f}% {pa('cev_free_uw'):>5.1f}%")

    # decision gate
    print()
    if sums["total"]:
        cev_gain = pa("cev_free") - pa("mv")
        cev_uw_gain = pa("cev_free_uw") - pa("mv")
        print(f"DECISION GATE: CEV-free vs MV = {cev_gain:+.2f} pts")
        print(f"               CEV-free (no div weighting) vs MV = {cev_uw_gain:+.2f} pts")
        if cev_gain >= 1.5 or cev_uw_gain >= 1.5:
            print(f"  -> PROCEED to CEV-LLM scale-up.")
        else:
            print(f"  -> SIGNAL TOO WEAK. Pivot or scrap.")


if __name__ == "__main__":
    main()
