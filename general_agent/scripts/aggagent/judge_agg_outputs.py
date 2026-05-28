"""Post-hoc LLM judge for AggAgent aggregation outputs.

Reads a `<strategy>_logs_k{k}.jsonl` file produced by aggregate.py, judges each
synthesized prediction against the gold answer (looked up from the aggin tree),
and writes a `<strategy>_judged_k{k}.jsonl` next to it plus an accuracy summary.

Uses the same MHQA-style prompt + gpt-4o-mini as the rollout judge so heuristic
(pre-judged) and LLM-method (re-judged) results are apples-to-apples.
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


JUDGE_PROMPT = """You are an expert evaluator. Determine if the generated answer correctly answers the question based on the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Evaluation Rubric:
1. Factuality: Does the answer contain the core correct information? All key facts must be present.
2. Semantic equivalence: Mark CORRECT if the meaning is the same even if phrased differently:
   - Durations expressed as start/end dates vs. duration length (e.g. "Sep 2022 to Feb 2024" = "18 months" = "18-month project")
   - Abbreviations and alternate names (e.g. "St. Petersburg" = "Saint Petersburg", "US" = "United States")
   - Numbers in different formats (e.g. "142,000" = "142 thousand", "$1.2M" = "1.2 million dollars")
   - Dates in different formats (e.g. "September 1, 2022" = "Sept 1 2022" = "1 September 2022")
3. Completeness: For multi-part questions, all parts must be correctly answered.
4. Contradiction: Mark INCORRECT only if the answer directly contradicts the ground truth (wrong facts, not just different phrasing).
5. Extra information: Ignore extra details in the generated answer as long as the core answer is correct.

Briefly explain your reasoning, then output "CORRECT" or "INCORRECT" on the final line."""


def parse_judge(text: str) -> bool:
    if not text:
        return False
    for line in reversed(text.strip().split("\n")):
        u = line.strip().upper()
        if "INCORRECT" in u:
            return False
        if "CORRECT" in u:
            return True
    u = text.upper()
    return "CORRECT" in u and "INCORRECT" not in u


def extract_short_answer(prediction: str) -> str:
    """Pull the short answer out of AggAgent's BrowseComp-style finish format."""
    if not prediction:
        return ""
    m = re.search(r"Exact Answer\s*:\s*(.+)", prediction, re.IGNORECASE)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    m = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return prediction.strip()


def build_gold_lookup(aggin_root: Path) -> dict:
    """Walk aggin_root, return question -> gold_answer. Sample one thread per question."""
    lookup = {}
    for p in aggin_root.rglob("thread_0/*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            lookup[d.get("question", "")] = d.get("gold_answer", "")
        except Exception:
            continue
    return lookup


def judge_one(client: OpenAI, model: str, entry: dict, gold: str) -> dict:
    prediction = entry.get("prediction") or ""
    short = extract_short_answer(prediction)
    prompt = JUDGE_PROMPT.format(
        question=entry.get("question", ""),
        ground_truth=gold,
        generated_answer=short or prediction,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        text = resp.choices[0].message.content or ""
        is_correct = parse_judge(text)
        return {"is_correct": is_correct, "judge_text": text, "extracted": short, "gold": gold}
    except Exception as e:
        return {"is_correct": False, "judge_text": f"ERR:{type(e).__name__}:{e}", "extracted": short, "gold": gold}


def judge_logs(logs_path: Path, gold_lookup: dict, client: OpenAI, model: str, max_workers: int) -> dict:
    entries = []
    with open(logs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        return {"n": 0, "n_correct": 0, "accuracy": 0.0, "missing_gold": 0}

    out_path = logs_path.with_name(logs_path.stem + "_judged.jsonl")
    results = []
    missing_gold = 0
    n_correct = 0

    def task(idx_entry):
        idx, entry = idx_entry
        question = entry.get("question", "")
        gold = gold_lookup.get(question, "")
        if not gold:
            return idx, entry, {"is_correct": False, "judge_text": "MISSING_GOLD", "extracted": "", "gold": ""}, True
        j = judge_one(client, model, entry, gold)
        return idx, entry, j, False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(task, (i, e)) for i, e in enumerate(entries)]
        slots = [None] * len(entries)
        for fut in as_completed(futs):
            i, entry, j, missing = fut.result()
            slots[i] = (entry, j)
            if missing:
                missing_gold += 1
            elif j.get("is_correct"):
                n_correct += 1

    with open(out_path, "w") as f:
        for entry, j in slots:
            entry["posthoc_judge"] = j
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n = len(entries)
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "missing_gold": missing_gold,
        "out": str(out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", required=True,
                   help="Root of the aggin tree (e.g. /data/user_data/ssmurali/aggin_toy/qwen3-8b/bamboogle/naive_k4/run_1)")
    p.add_argument("--aggout-root", required=True,
                   help="Root of the aggregation output dir to judge (e.g. .../aggout_toy/qwen3-8b/bamboogle/naive_k4/run_1)")
    p.add_argument("--strategies", nargs="+", default=["solagg", "summagg", "aggagent"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-workers", type=int, default=16)
    args = p.parse_args()

    aggin_root = Path(args.aggin_root)
    aggout_root = Path(args.aggout_root)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    client = OpenAI()
    gold_lookup = build_gold_lookup(aggin_root)
    print(f"Loaded {len(gold_lookup)} question -> gold mappings from {aggin_root}")

    summary = {}
    for strat in args.strategies:
        logs = aggout_root / strat / f"{strat}_logs_k{args.k}.jsonl"
        if not logs.exists():
            print(f"[skip] {strat}: no logs at {logs}")
            continue
        t0 = time.time()
        stats = judge_logs(logs, gold_lookup, client, args.model, args.max_workers)
        stats["elapsed_s"] = round(time.time() - t0, 1)
        summary[strat] = stats
        print(f"  {strat}@{args.k:<3}  acc={stats['accuracy']*100:6.2f}%  "
              f"({stats['n_correct']}/{stats['n']})  missing_gold={stats['missing_gold']}  "
              f"{stats['elapsed_s']}s  -> {stats['out']}")

    summary_path = aggout_root / f"posthoc_summary_k{args.k}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
