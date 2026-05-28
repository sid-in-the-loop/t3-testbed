"""CEV-LLM realistic-policy validation on a small subset.

Same 20 slices as validate_cev.py. For each non-unanimous question, send
gpt-4o-mini the pooled evidence (concat of all k threads' tool results,
truncated) and the distinct candidate answers. Ask it to pick the candidate
best supported by the evidence.

Compares realistic accuracy of: MV, AB-free, CEV-free, CEV-LLM, pass@4.

Budget: ~2k LLM calls @ ~$0.0003 each = ~$0.60.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# Reuse helpers
sys.path.insert(0, "/home/ssmurali/t3-testbed/general_agent/scripts/aggagent")
from validate_cev import (
    load_slice, mv_pick, ab_free_pick, cev_free_pick, normalize,
)


SYS_PROMPT = """You judge which candidate answer is best supported by the
evidence retrieved across multiple parallel search-agent rollouts.

You are given:
  - QUESTION
  - POOLED EVIDENCE: the union of search results retrieved by all rollouts
  - CANDIDATES: distinct final answers (one per unique answer string)

Pick the candidate that the POOLED EVIDENCE most directly supports.

Important: do NOT pick the majority answer just because it's the majority.
Pick the one whose support in the pooled evidence is strongest.

Output JSON only:
  {"chosen": <0-indexed candidate number>}

No other text."""


def evidence_text(thread):
    parts = []
    for m in thread.get("messages", []):
        if m.get("role") == "tool":
            c = (m.get("content") or "")[:400].replace("\n", " ").strip()
            if c: parts.append(c)
    return " || ".join(parts)


def build_user(question, candidates, threads, max_evidence_chars=2400):
    """Build the user message. Pooled evidence is truncated per-thread."""
    pooled_parts = []
    per_thread_budget = max_evidence_chars // max(len(threads), 1)
    for j, t in enumerate(threads):
        ev = evidence_text(t["raw"])[:per_thread_budget]
        pooled_parts.append(f"  [rollout {j}] {ev}")
    pooled = "\n".join(pooled_parts) or "(no evidence)"

    cand_lines = "\n".join(f"  {i}. {c[:200]}" for i, c in enumerate(candidates))
    return (f"QUESTION: {question}\n\n"
            f"POOLED EVIDENCE:\n{pooled}\n\n"
            f"CANDIDATES:\n{cand_lines}")


def parse_chosen(text, max_idx):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        d = json.loads(s)
        c = int(d.get("chosen", -1))
        if 0 <= c <= max_idx: return c
    except Exception:
        m = re.search(r'"chosen"\s*:\s*(\d+)', text)
        if m:
            c = int(m.group(1))
            if 0 <= c <= max_idx: return c
    return -1


def cev_llm_pick(client, model_name, question, threads):
    """Returns the THREAD index whose answer was picked (so accuracy uses
    the picked thread's is_correct)."""
    answers = [t["prediction"] for t in threads]
    if len(set(answers)) == 1:
        return 0

    # distinct candidates and the (first) thread index that produced each
    cand_to_idx = {}
    for i, a in enumerate(answers):
        cand_to_idx.setdefault(a, i)
    candidates = list(cand_to_idx.keys())

    prompt = build_user(question, candidates, threads)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        text = resp.choices[0].message.content or ""
        c = parse_chosen(text, len(candidates) - 1)
        if c < 0:
            return mv_pick(threads)
        return cand_to_idx[candidates[c]]
    except Exception:
        return mv_pick(threads)


def slice_accuracies(slice_dir, client, model_name, k=4, max_workers=32):
    threads_by_q = load_slice(slice_dir)
    counts = {"pass": 0, "mv": 0, "ab_free": 0, "cev_free": 0, "cev_llm": 0, "total": 0}

    qs_to_llm = []
    for q, threads in threads_by_q.items():
        if len(threads) != k: continue
        counts["total"] += 1
        corrects = [t["is_correct"] for t in threads]
        if any(corrects): counts["pass"] += 1

        # deterministic methods
        for name, picker in [("mv", mv_pick),
                             ("ab_free", ab_free_pick),
                             ("cev_free", lambda ts: cev_free_pick(ts, use_div_weighting=True))]:
            try:
                i = picker(threads)
                if threads[i]["is_correct"]: counts[name] += 1
            except Exception: pass

        # queue LLM
        if len(set(t["prediction"] for t in threads)) == 1:
            # unanimous; LLM accuracy = MV accuracy
            if threads[0]["is_correct"]: counts["cev_llm"] += 1
        else:
            qs_to_llm.append((q, threads))

    # parallel LLM calls for the non-unanimous queue
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(cev_llm_pick, client, model_name, q, ts): (q, ts) for q, ts in qs_to_llm}
        for fut in as_completed(futs):
            try:
                i = fut.result()
                _, ts = futs[fut]
                if ts[i]["is_correct"]:
                    counts["cev_llm"] += 1
            except Exception:
                pass

    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--models", nargs="+", default=["qwen3-8b", "gemma3-12b"])
    p.add_argument("--datasets", nargs="+",
                   default=["bamboogle", "musique", "hotpotqa", "2wikimultihopqa", "hle"])
    p.add_argument("--conds", nargs="+", default=["div_k4", "naive_k4"])
    p.add_argument("--seeds", nargs="+", default=["run_1"])
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    client = OpenAI()
    aggin = Path(args.aggin_root)

    print(f"\n{'model':<14} {'dataset':<18} {'cond':<10} {'pass':>6} {'MV':>6} {'AB-fr':>6} {'CEV-fr':>6} {'CEV-LLM':>7}  notes")
    print("-" * 100)
    sums = {k: 0 for k in ["pass", "mv", "ab_free", "cev_free", "cev_llm", "total"]}

    t0 = time.time()
    n_slices = 0
    for model in args.models:
        for ds in args.datasets:
            for cond in args.conds:
                for seed in args.seeds:
                    slice_dir = aggin / model / ds / cond / seed
                    if not slice_dir.exists(): continue
                    n_slices += 1
                    print(f"  ... running {model}/{ds}/{cond}/{seed}", flush=True)
                    c = slice_accuracies(slice_dir, client, args.model)
                    if c["total"] == 0: continue
                    for k in sums: sums[k] += c[k]
                    pa = lambda k: c[k] / c["total"] * 100
                    deltas = []
                    if pa("cev_llm") - pa("mv") >= 1.5: deltas.append("LLM>>MV")
                    if pa("cev_llm") > pa("cev_free"): deltas.append("LLM>CEV-fr")
                    print(f"{model:<14} {ds:<18} {cond:<10} {pa('pass'):>5.1f}% {pa('mv'):>5.1f}% {pa('ab_free'):>5.1f}% {pa('cev_free'):>5.1f}% {pa('cev_llm'):>6.1f}%  {' '.join(deltas)}")

    print("-" * 100)
    if sums["total"]:
        pa = lambda k: sums[k] / sums["total"] * 100
        print(f"{'OVERALL':<14} {'':<18} {'':<10} {pa('pass'):>5.1f}% {pa('mv'):>5.1f}% {pa('ab_free'):>5.1f}% {pa('cev_free'):>5.1f}% {pa('cev_llm'):>6.1f}%")
        cev_llm_gain = pa("cev_llm") - pa("mv")
        print()
        print(f"DECISION GATE: CEV-LLM vs MV = {cev_llm_gain:+.2f} pts")
        if cev_llm_gain >= 2.0:
            print(f"  -> PROCEED. CEV-LLM has real signal. Scale up to all 248 slices.")
        elif cev_llm_gain >= 1.0:
            print(f"  -> MARGINAL. Worth scaling if cost is OK.")
        else:
            print(f"  -> NO SIGNAL. Drop CEV entirely.")
    print(f"\nElapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
