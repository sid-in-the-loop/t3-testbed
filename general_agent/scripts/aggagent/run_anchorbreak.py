"""AnchorBreak: per-question LLM judge with anchor-escape-aware prompt.

For every non-unanimous question in the aggin tree:
  Send gpt-4o-mini the k=4 (queries, evidence_summary, answer) tuples.
  Ask it to rate each thread on (INDEPENDENT, GROUNDED) axes 0-100.
  Pick the thread with the highest INDEPENDENT * GROUNDED.

For unanimous questions, skip the LLM and use the (single) thread answer.

Writes per-slice JSONL: {question, chosen_thread_idx, source, raw_scores}
to /data/user_data/ssmurali/anchorbreak/<model>/<ds>/<cond>/<seed>/picks.jsonl

build_summary.py reads these to compute the `anchorbreak` column.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYS_PROMPT = """You aggregate parallel agent rollouts on the same question.

The k rollouts start from the same prompt and search the web in parallel. They
often "anchor": share a framing that leads them all to the same wrong answer.
You want the escapee — a rollout that broke the anchor and arrived at a
correct, evidence-supported answer, even if it disagrees with the majority.

For each rollout, rate two axes 0-100:
  INDEPENDENT: how different are its searches / retrieved snippets from the
               other rollouts'? High = explored differently.
  GROUNDED:    how well does its EVIDENCE support its ANSWER, on its own?
               High = answer is justified by what was retrieved, not asserted.

Pick the rollout that maximizes INDEPENDENT * GROUNDED. Output JSON only:

{"per_thread": [{"independent": int, "grounded": int}, ...],
 "chosen": int}     // 0-indexed thread index, 0..k-1

Do not output anything else. No markdown, no commentary."""


def evidence_summary(thread: dict, max_chars: int = 600) -> str:
    """Compact one-line summary of this thread's searches + retrieved snippets."""
    queries = []
    snippets = []
    for m in thread.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if "query" in args:
                        queries.append(str(args["query"]))
                except Exception:
                    pass
        elif m.get("role") == "tool":
            c = (m.get("content") or "")[:200].replace("\n", " ").strip()
            if c:
                snippets.append(c)
    q_str = " | ".join(queries[:6]) or "(no searches)"
    s_str = " || ".join(snippets[:4])[:max_chars] or "(no evidence)"
    return f"QUERIES: {q_str}\nEVIDENCE: {s_str}"


def build_user_prompt(question: str, threads: list[dict]) -> str:
    parts = [f"QUESTION: {question}\n"]
    for i, t in enumerate(threads):
        parts.append(f"\n--- ROLLOUT {i} ---")
        parts.append(evidence_summary(t))
        parts.append(f"ANSWER: {t.get('prediction', '').strip()[:200]}")
    return "\n".join(parts)


def parse_response(text: str, k: int) -> dict:
    """Extract chosen thread index from LLM response. Robust to JSON noise."""
    try:
        # strip code fences if present
        s = text.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        d = json.loads(s)
        chosen = int(d.get("chosen", -1))
        if 0 <= chosen < k:
            return {"chosen": chosen, "per_thread": d.get("per_thread", []), "raw": text}
    except Exception:
        pass
    # fallback: regex
    m = re.search(r'"chosen"\s*:\s*(\d+)', text)
    if m:
        c = int(m.group(1))
        if 0 <= c < k:
            return {"chosen": c, "per_thread": [], "raw": text}
    return {"chosen": -1, "per_thread": [], "raw": text, "parse_failed": True}


def discover_slices(aggin_root: Path, models, conds):
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


def load_threads(slice_dir: Path) -> dict:
    out = {}
    for tdir in sorted(slice_dir.iterdir()):
        if not tdir.is_dir():
            continue
        for f in tdir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            q = d["question"]
            if q not in out:
                out[q] = []
            out[q].append(d)
    return out


def process_question(client, model_name, q, threads, k=4):
    if len(threads) != k:
        return {"question": q, "skip": "wrong_k"}
    answers = [t.get("prediction", "") for t in threads]
    if len(set(answers)) == 1:
        # unanimous; the answer is whatever (all threads agree)
        return {"question": q, "chosen": 0, "source": "unanimous"}
    prompt = build_user_prompt(q, threads)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content or ""
        parsed = parse_response(text, k)
        if parsed["chosen"] < 0:
            # fall back to MV
            from collections import Counter
            mv_ans = Counter(answers).most_common(1)[0][0]
            chosen = next(i for i, a in enumerate(answers) if a == mv_ans)
            return {"question": q, "chosen": chosen, "source": "fallback_mv",
                    "raw": text[:200]}
        return {"question": q, "chosen": parsed["chosen"], "source": "llm",
                "per_thread": parsed["per_thread"]}
    except Exception as e:
        from collections import Counter
        mv_ans = Counter(answers).most_common(1)[0][0]
        chosen = next(i for i, a in enumerate(answers) if a == mv_ans)
        return {"question": q, "chosen": chosen, "source": "error_mv",
                "error": f"{type(e).__name__}:{e}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--out-root",   default="/data/user_data/ssmurali/anchorbreak")
    p.add_argument("--models", nargs="+",
                   default=["qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-12b"])
    p.add_argument("--conds", nargs="+", default=["div_k4", "naive_k4"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--shard", default=None,
                   help="Process only slice_idx %% N == i. Format: 'i/N'.")
    p.add_argument("--limit-slices", type=int, default=None, help="dev test")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    client = OpenAI()
    out_root = Path(args.out_root)

    slices = list(discover_slices(Path(args.aggin_root), args.models, args.conds))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        slices = [s for idx, s in enumerate(slices) if idx % n == i]
        print(f"[shard {i}/{n}] {len(slices)} slices in this shard", flush=True)
    if args.limit_slices:
        slices = slices[:args.limit_slices]
    print(f"Processing {len(slices)} slices on {args.model}", flush=True)

    t0 = time.time()
    grand_stats = {"unanimous": 0, "llm": 0, "fallback_mv": 0, "error_mv": 0}

    for idx, (model, ds, cond, seed, slice_dir) in enumerate(slices):
        out_dir = out_root / model / ds / cond / seed
        out_path = out_dir / "picks.jsonl"
        if out_path.exists():
            # already done — skip (idempotent)
            print(f"  [{idx+1}/{len(slices)}] SKIP {model}/{ds}/{cond}/{seed}  (exists)", flush=True)
            continue

        threads_by_q = load_threads(slice_dir)
        if not threads_by_q:
            continue

        t_slice = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(process_question, client, args.model, q, threads): q
                    for q, threads in threads_by_q.items() if len(threads) == 4}
            for fut in as_completed(futs):
                r = fut.result()
                if "skip" in r:
                    continue
                results.append(r)
                src = r.get("source", "unknown")
                if src in grand_stats:
                    grand_stats[src] += 1

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"  [{idx+1}/{len(slices)}] {model}/{ds}/{cond}/{seed}  "
              f"n={len(results)}  {time.time()-t_slice:.0f}s  "
              f"cum_unan={grand_stats['unanimous']} cum_llm={grand_stats['llm']} "
              f"cum_err={grand_stats['error_mv']}", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Totals: {grand_stats}")


if __name__ == "__main__":
    main()
