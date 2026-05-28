"""Score thread-level confidence using a local vLLM endpoint.

For each aggin JSON, sends question + brief evidence summary + proposed answer
to the vLLM endpoint, asks for a 0-100 integer confidence, writes it back to
auto_judge.confidence.
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


CONF_PROMPT = """\
You are grading the likely correctness of a candidate answer to a research question.

Question: {question}

Searches performed by the agent:
{evidence}

Candidate answer: {answer}

Considering whether the searches actually support the candidate answer, rate \
how confident you are that the candidate answer is correct, on a 0-100 scale.
Reply with ONLY the integer number, nothing else."""


def build_evidence(messages: list[dict], max_chars: int = 1500) -> str:
    lines = []
    queries = []
    results = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            try:
                args = json.loads(m["tool_calls"][0]["function"]["arguments"])
                queries.append(args.get("query", ""))
            except Exception:
                queries.append("")
        elif m.get("role") == "tool":
            results.append(m.get("content", ""))
    for i, q in enumerate(queries):
        r = results[i] if i < len(results) else ""
        snippet = r[:300].replace("\n", " ").strip()
        lines.append(f"  {i+1}. {q!r} -> {snippet}{'...' if len(r) > 300 else ''}")
    rendered = "\n".join(lines) if lines else "  (no searches issued)"
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n  [... truncated]"
    return rendered


def parse_confidence(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    val = int(m.group(1))
    return float(val) if 0 <= val <= 100 else None


def score_one(client: OpenAI, model: str, path: Path, overwrite: bool,
              disable_thinking: bool = True) -> str:
    try:
        with open(path) as f:
            d = json.load(f)
        if d.get("auto_judge", {}).get("confidence") is not None and not overwrite:
            return "skip"
        prompt = CONF_PROMPT.format(
            question=d.get("question", ""),
            evidence=build_evidence(d.get("messages", [])),
            answer=d.get("prediction", ""),
        )
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        # Qwen3 + similar: turn off the <think> block so a short reply isn't
        # truncated mid-reasoning. vLLM accepts this via extra_body.
        if disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        conf = parse_confidence(text)
        if conf is None:
            d["auto_judge"]["confidence"] = 50.0   # neutral fallback
            d["auto_judge"]["confidence_raw"] = text[:200]
            d["auto_judge"]["confidence_parse_failed"] = True
            with open(path, "w") as f:
                json.dump(d, f, ensure_ascii=False)
            return "parse_fail"
        d["auto_judge"]["confidence"] = conf
        with open(path, "w") as f:
            json.dump(d, f, ensure_ascii=False)
        return "ok"
    except Exception as e:
        return f"err:{type(e).__name__}:{e}"


def find_files(
    root: Path, models: list[str], conds: list[str],
    datasets: list[str] | None = None, seeds: list[str] | None = None,
) -> list[Path]:
    out = []
    for model in models:
        mroot = root / model
        if not mroot.exists():
            continue
        for ds in sorted(p for p in mroot.iterdir() if p.is_dir()):
            if datasets and ds.name not in datasets:
                continue
            for cond in conds:
                cdir = ds / cond
                if not cdir.exists():
                    continue
                for seed in sorted(p for p in cdir.iterdir() if p.is_dir()):
                    if seeds and seed.name not in seeds:
                        continue
                    for thr in sorted(p for p in seed.iterdir() if p.is_dir()):
                        out.extend(sorted(thr.glob("*.json")))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-12b"])
    p.add_argument("--conds", nargs="+", default=["naive_k4", "div_k4"])
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", default=None)
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None,
                   help="Process only files where idx %% N == i. Format: 'i/N' (e.g. '0/8').")
    args = p.parse_args()

    client = OpenAI(base_url=args.endpoint, api_key="dummy")
    files = find_files(Path(args.aggin_root), args.models, args.conds, args.datasets, args.seeds)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = [f for idx, f in enumerate(files) if idx % n == i]
        print(f"[shard {i}/{n}] {len(files)} files in this shard", flush=True)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} thread files to score on {args.endpoint} (model={args.model})", flush=True)
    if not files:
        return

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "parse_fail": 0, "err": 0}
    err_samples = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(score_one, client, args.model, fp, args.overwrite): fp for fp in files}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r in stats:
                stats[r] += 1
            else:
                stats["err"] += 1
                if len(err_samples) < 5:
                    err_samples.append(r)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(files)}  {stats}  {rate:.1f}/s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s: {stats}")
    if err_samples:
        print("Error samples:")
        for s in err_samples:
            print(" ", s)


if __name__ == "__main__":
    main()
