"""
Main table experiment runner — generalized from figure1_gaia103_experiment.py.

Three conditions at matched compute budget:
  Sequential:      k=1, T = max_turns_par * K
  Naive parallel:  k=K, T = max_turns_par
  Div. parallel:   k=K, T = max_turns_par, pool=16, greedy-Jaccard

Usage:
  cd general_agent
  python -m webwalkerqa.run.run_main_table \
    --model openai/gpt-4o-mini \
    --dataset data/main_table/hotpotqa.json \
    --condition diversity_parallel \
    --max-turns-par 5 \
    --output-dir /path/to/results/main_table/gpt-4o-mini/hotpotqa

  # For vLLM models:
  python -m webwalkerqa.run.run_main_table \
    --model openai/Qwen3-8B \
    --dataset data/main_table/hotpotqa.json \
    --condition naive_parallel \
    --api-base http://localhost:8003/v1 \
    --output-dir /path/to/results/main_table/qwen3-8b/hotpotqa
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_GA_DIR = Path(__file__).resolve().parent.parent
_TESTBED_ROOT = _GA_DIR.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")

from webwalkerqa.dataset import load_dataset
from webwalkerqa.eval import exact_match
from webwalkerqa.llm import call_llm, normalize_model, set_api_base
import webwalkerqa.methods.diversity_scaling as _ds
from webwalkerqa.methods.diversity_scaling import (
    _extract_tag,
    _parse_pool_response,
    run_single_rollout,
    _safe_rollout_seed,
    generate_pool,
    SEED_MAX,
)
from webwalkerqa.methods.utils import select_diverse_queries

# ── Constants ─────────────────────────────────────────────────────────────────
K         = 4   # parallel threads
POOL_SIZE = 8  # diversity pool size
SYNTH_SNIPPET = 800


# ── QPD / ITC helpers ────────────────────────────────────────────────────────

def jaccard_distance(a: str, b: str) -> float:
    from webwalkerqa.methods.utils import jaccard_distance as _jd
    return _jd(a, b)


def jaccard_similarity_fn(a: str, b: str) -> float:
    from webwalkerqa.methods.utils import jaccard_similarity as _js
    return _js(a, b)


def compute_qpd(queries: List[str]) -> float:
    """Query Pairwise Diversity: mean Jaccard distance over all pairs."""
    k = len(queries)
    if k < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += jaccard_distance(queries[i], queries[j])
            count += 1
    return total / count


def compute_itc(turn_logs_per_thread: List[List[Dict]]) -> float:
    """Inter-Turn Coherence: mean Jaccard similarity between turn-1 query and subsequent queries within each thread."""
    values = []
    for logs in turn_logs_per_thread:
        queries = [log.get("query", "") for log in logs if log.get("query")]
        if len(queries) < 2:
            continue
        q1 = queries[0]
        for q in queries[1:]:
            values.append(jaccard_similarity_fn(q1, q))
    return float(np.mean(values)) if values else 0.0


def compute_atc(turn_logs_per_thread: List[List[Dict]], max_turns: int) -> float:
    """Across-Thread Coherence: mean pairwise Jaccard distance between queries of different threads at same turn."""
    values = []
    for t in range(max_turns):
        queries_at_t = []
        for logs in turn_logs_per_thread:
            for log in logs:
                if log.get("turn") == t + 1 and log.get("query"):
                    queries_at_t.append(log["query"])
                    break
        if len(queries_at_t) >= 2:
            for i in range(len(queries_at_t)):
                for j in range(i + 1, len(queries_at_t)):
                    values.append(jaccard_distance(queries_at_t[i], queries_at_t[j]))
    return float(np.mean(values)) if values else 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_turn1_query(rollout: Dict[str, Any]) -> str:
    for log in (rollout.get("turn_logs") or []):
        if log.get("query"):
            return str(log["query"]).replace("\n", " ").strip()
    return ""


def _thread_summary(turn_logs: List[Dict[str, Any]]) -> str:
    parts = []
    for log in turn_logs:
        if log.get("query"):
            snip = (log.get("search_result") or "")[:SYNTH_SNIPPET]
            parts.append(f"Query: {log['query']}\nExcerpt: {snip}")
        if log.get("answer"):
            parts.append(f"Thread conclusion: {log['answer']}")
    return "\n".join(parts) if parts else "(no search trace)"


def _gold(ex) -> str:
    return str(ex.answer) if not isinstance(ex.answer, list) else str(ex.answer[0])


async def synthesize(model: str, question: str, rollouts: List[Dict], answers: List[str]) -> str:
    summaries = [_thread_summary(r.get("turn_logs") or []) for r in rollouts]
    blocks = [
        f"--- Thread {i} ---\nEvidence:\n{summ[:3000]}\nThread final answer: {ans}"
        for i, (summ, ans) in enumerate(zip(summaries, answers))
    ]
    prompt = (
        f"You are a research coordinator synthesizing results from independent web-search threads.\n\n"
        f"Question: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nInstructions:\n"
        "1. Review all evidence and thread answers above.\n"
        "2. Identify the most well-supported answer.\n"
        "3. Output ONLY the final answer inside <answer>...</answer> tags.\n"
        "4. The answer must be exact and concise: a name, number, date, or short phrase.\n"
        "5. Do NOT include explanations inside the tags. Just the answer.\n\n"
        "Final answer:"
    )
    text, _, _ = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=1024, temperature=0.3,
    )
    ans = _extract_tag(text or "", "answer")
    return ans.strip() if ans else (text or "").strip()[:2000]


# ── Output helpers ───────────────────────────────────────────────────────────

def _save_trajectory(traj_dir: Path, qid: str, data: Dict[str, Any]) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"{qid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_jsonl_for_judge(path: Path, rows: List[Dict], condition: str) -> None:
    """Write JSONL for eval_llm.py judge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if condition == "sequential":
                rollout_answers = [row.get("predicted_answer", "")]
            else:
                rollout_answers = [
                    row.get("thread_1_answer", ""),
                    row.get("thread_2_answer", ""),
                    row.get("thread_3_answer", ""),
                    row.get("thread_4_answer", ""),
                ]
            entry = {
                "question_id": row.get("question_id", ""),
                "question":    row.get("question", ""),
                "answer_gt":   row.get("gold_answer", ""),
                "rollout_answers": rollout_answers,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs, dynamic_ncols=True, ascii=True)
    except ImportError:
        return iterable


# ── Condition 1: Sequential ─────────────────────────────────────────────────

SEQ_FIELDS = [
    "question_id", "question", "predicted_answer", "gold_answer",
    "correct", "num_turns_used",
]


async def run_sequential(
    model: str, examples, max_turns: int, sem: asyncio.Semaphore, traj_dir: Path,
    run_seed: int = 0, k: int = 4,
) -> List[Dict]:
    rows: List[Dict] = []
    lock = asyncio.Lock()
    pbar = _tqdm(range(len(examples)), desc="sequential", unit="q")

    async def one(ex):
        qid = str(ex.id)
        gold = _gold(ex)
        try:
            async with sem:
                r = await run_single_rollout(
                    model=model, question=ex.question, answer_gt=gold,
                    max_turns=max_turns, initial_query=None,
                    rollout_seed=_safe_rollout_seed(qid, run_seed, 0),
                    question_id=qid, react_temp_first=1.0, react_temp_rest=0.7,
                )
            pred = (r.get("answer") or "").strip()
            correct = 1 if exact_match(pred, gold) else 0
            row = {
                "question_id": qid, "question": ex.question,
                "predicted_answer": pred, "gold_answer": gold,
                "correct": correct, "num_turns_used": r.get("turns_used", 0),
            }
            _save_trajectory(traj_dir, qid, {
                "question_id": qid, "question": ex.question,
                "gold_answer": gold, "predicted_answer": pred,
                "correct": correct,
                "turn_logs": r.get("turn_logs", []),
                "full_responses": r.get("full_responses", []),
            })
        except Exception as e:
            print(f"\n  [error] seq {qid}: {e}")
            traceback.print_exc()
            row = {
                "question_id": qid, "question": ex.question,
                "predicted_answer": "ERROR", "gold_answer": gold,
                "correct": 0, "num_turns_used": 0,
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── Condition 2: Naive Parallel ─────────────────────────────────────────────

NAIVE_FIELDS = [
    "question_id", "question", "gold_answer",
    "turn1_query_1", "turn1_query_2", "turn1_query_3", "turn1_query_4",
    "thread_1_answer", "thread_2_answer", "thread_3_answer", "thread_4_answer",
    "oracle_correct", "synthesis_correct",
    "jaccard_qpd", "itc", "atc",
]


def _naive_fields(k: int) -> List[str]:
    base = ["question_id", "question", "gold_answer"]
    base += [f"turn1_query_{i+1}" for i in range(k)]
    base += [f"thread_{i+1}_answer" for i in range(k)]
    base += ["oracle_correct", "synthesis_correct", "jaccard_qpd", "itc", "atc"]
    return base


async def run_naive_parallel(
    model: str, examples, max_turns: int, sem: asyncio.Semaphore, traj_dir: Path,
    run_seed: int = 0, k: int = 4,
) -> List[Dict]:
    rows: List[Dict] = []
    lock = asyncio.Lock()
    pbar = _tqdm(range(len(examples)), desc="naive_parallel", unit="q")

    async def one(ex):
        qid = str(ex.id)
        gold = _gold(ex)
        try:
            async with sem:
                rollouts = await asyncio.gather(*[
                    run_single_rollout(
                        model=model, question=ex.question, answer_gt=gold,
                        max_turns=max_turns, initial_query=None,
                        rollout_seed=_safe_rollout_seed(qid, run_seed, i),
                        question_id=qid, react_temp_first=1.0, react_temp_rest=0.7,
                    )
                    for i in range(k)
                ])
                answers = [(r.get("answer") or "").strip() for r in rollouts]
                t1q = [_get_turn1_query(r) for r in rollouts]
                oracle = 1 if any(exact_match(a, gold) for a in answers) else 0
                syn_ans = await synthesize(model, ex.question, list(rollouts), answers)
                syn_correct = 1 if exact_match(syn_ans, gold) else 0

            valid_t1q = [q for q in t1q if q]
            thread_logs = [r.get("turn_logs", []) for r in rollouts]
            jqpd = compute_qpd(valid_t1q)
            itc = compute_itc(thread_logs)
            atc = compute_atc(thread_logs, max_turns)

            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                **{f"turn1_query_{i+1}": t1q[i] if i < len(t1q) else "" for i in range(k)},
                **{f"thread_{i+1}_answer": answers[i] if i < len(answers) else "" for i in range(k)},
                "oracle_correct": oracle, "synthesis_correct": syn_correct,
                "jaccard_qpd": f"{jqpd:.6f}", "itc": f"{itc:.6f}", "atc": f"{atc:.6f}",
            }
            _save_trajectory(traj_dir, qid, {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "threads": [
                    {"thread_id": i, "turn1_query": t1q[i] if i < len(t1q) else "",
                     "answer": answers[i] if i < len(answers) else "",
                     "turn_logs": rollouts[i].get("turn_logs", []),
                     "full_responses": rollouts[i].get("full_responses", [])}
                    for i in range(k)
                ],
                "oracle_correct": oracle, "synthesis_answer": syn_ans, "synthesis_correct": syn_correct,
                "jaccard_qpd": jqpd, "itc": itc, "atc": atc,
            })
        except Exception as e:
            print(f"\n  [error] naive {qid}: {e}")
            traceback.print_exc()
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                **{f"turn1_query_{i+1}": "" for i in range(k)},
                **{f"thread_{i+1}_answer": "ERROR" for i in range(k)},
                "oracle_correct": 0, "synthesis_correct": 0,
                "jaccard_qpd": "0.0", "itc": "0.0", "atc": "0.0",
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── Condition 3: Diversity Parallel (greedy-Jaccard) ────────────────────────

def _div_fields(k: int) -> List[str]:
    base = ["question_id", "question", "gold_answer", "pool_queries", "selected_queries"]
    base += [f"thread_{i+1}_answer" for i in range(k)]
    base += ["oracle_correct", "synthesis_correct", "jaccard_qpd", "itc", "atc"]
    return base


async def run_diversity_parallel(
    model: str, examples, max_turns: int, sem: asyncio.Semaphore, traj_dir: Path,
    run_seed: int = 0, k: int = 4, pool_size: int = 16,
) -> List[Dict]:
    rows: List[Dict] = []
    lock = asyncio.Lock()
    pbar = _tqdm(range(len(examples)), desc="diversity_parallel", unit="q")

    async def one(ex):
        qid = str(ex.id)
        gold = _gold(ex)
        pool: List[str] = []
        seeds: List[str] = []
        try:
            async with sem:
                pool, _, _ = await generate_pool(model, ex.question, POOL_SIZE)
                seeds = select_diverse_queries(pool, K, method="jaccard",
                                               seed=_safe_rollout_seed(qid, run_seed, 0))
                while len(seeds) < K:
                    seeds.append(pool[len(seeds) % len(pool)])
                seeds = seeds[:K]

                rollouts = await asyncio.gather(*[
                    run_single_rollout(
                        model=model, question=ex.question, answer_gt=gold,
                        max_turns=max_turns, initial_query=seeds[i],
                        rollout_seed=_safe_rollout_seed(qid, run_seed, i),
                        question_id=qid, react_temp_first=0.7, react_temp_rest=0.7,
                    )
                    for i in range(K)
                ])
                answers = [(r.get("answer") or "").strip() for r in rollouts]
                oracle = 1 if any(exact_match(a, gold) for a in answers) else 0
                syn_ans = await synthesize(model, ex.question, list(rollouts), answers)
                syn_correct = 1 if exact_match(syn_ans, gold) else 0

            thread_logs = [r.get("turn_logs", []) for r in rollouts]
            jqpd = compute_qpd(seeds)
            itc = compute_itc(thread_logs)
            atc = compute_atc(thread_logs, max_turns)

            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": "; ".join(p.replace(";", ",") for p in pool),
                "selected_queries": "; ".join(s.replace(";", ",") for s in seeds),
                "thread_1_answer": answers[0] if len(answers) > 0 else "",
                "thread_2_answer": answers[1] if len(answers) > 1 else "",
                "thread_3_answer": answers[2] if len(answers) > 2 else "",
                "thread_4_answer": answers[3] if len(answers) > 3 else "",
                "oracle_correct": oracle, "synthesis_correct": syn_correct,
                "jaccard_qpd": f"{jqpd:.6f}", "itc": f"{itc:.6f}", "atc": f"{atc:.6f}",
            }
            _save_trajectory(traj_dir, qid, {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": pool, "selected_seeds": seeds,
                "threads": [
                    {"thread_id": i, "seed_query": seeds[i] if i < len(seeds) else "",
                     "answer": answers[i] if i < len(answers) else "",
                     "turn_logs": rollouts[i].get("turn_logs", []),
                     "full_responses": rollouts[i].get("full_responses", [])}
                    for i in range(K)
                ],
                "oracle_correct": oracle, "synthesis_answer": syn_ans, "synthesis_correct": syn_correct,
                "jaccard_qpd": jqpd, "itc": itc, "atc": atc,
            })
        except Exception as e:
            print(f"\n  [error] div {qid}: {e}")
            traceback.print_exc()
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": "", "selected_queries": "",
                "thread_1_answer": "ERROR", "thread_2_answer": "ERROR",
                "thread_3_answer": "ERROR", "thread_4_answer": "ERROR",
                "oracle_correct": 0, "synthesis_correct": 0,
                "jaccard_qpd": "0.0", "itc": "0.0", "atc": "0.0",
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── Summary ──────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "condition", "n_questions", "n_correct",
    "pass_at_1", "pass_at_4", "synthesis_accuracy",
    "mean_jaccard_qpd", "mean_itc", "mean_atc",
]


def _summary_row(condition: str, rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {f: 0 for f in SUMMARY_FIELDS} | {"condition": condition}

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return sum(vals) / len(vals) if vals else 0.0

    if condition == "sequential":
        n_correct = sum(int(r.get("correct", 0)) for r in rows)
        pass_1 = n_correct / n
        pass_4 = pass_1  # k=1
        synth_acc = pass_1
    else:
        n_correct = sum(int(r.get("oracle_correct", 0)) for r in rows)
        # pass@1 = first thread correct
        pass_1 = sum(1 for r in rows if exact_match(r.get("thread_1_answer", ""), r.get("gold_answer", ""))) / n
        pass_4 = n_correct / n  # oracle = any of 4 correct
        synth_acc = sum(int(r.get("synthesis_correct", 0)) for r in rows) / n

    return {
        "condition": condition,
        "n_questions": n,
        "n_correct": n_correct,
        "pass_at_1": pass_1,
        "pass_at_4": pass_4,
        "synthesis_accuracy": synth_acc,
        "mean_jaccard_qpd": _mean("jaccard_qpd"),
        "mean_itc": _mean("itc"),
        "mean_atc": _mean("atc"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

CONDITION_CHOICES = ["sequential", "naive_parallel", "diversity_parallel"]


async def main_async(args) -> None:
    import random
    RUN_SEED = args.seed
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)

    model = normalize_model(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_turns_par = args.max_turns_par
    max_turns_seq = K * max_turns_par  # compute-matched

    # Set api_base for vLLM
    if args.api_base:
        set_api_base(args.api_base)

    examples = load_dataset(path=args.dataset, max_examples=args.max_examples)
    print(f"Loaded {len(examples)} questions | model={model}")
    print(f"Budget: seq=T{max_turns_seq} | par=k{K}×T{max_turns_par} | pool={POOL_SIZE}")

    sem = asyncio.Semaphore(args.max_concurrent)
    conditions = [args.condition] if args.condition else CONDITION_CHOICES
    summary_rows: List[Dict] = []

    for condition in conditions:
        traj_dir = out_dir / "trajectories" / f"{condition}_T{max_turns_par}"

        if condition == "sequential":
            print(f"\n=== SEQUENTIAL (T={max_turns_seq}) ===")
            rows = await run_sequential(model, examples, max_turns_seq, sem, traj_dir, run_seed=RUN_SEED)
            csv_path = out_dir / f"sequential_T{max_turns_par}.csv"
            _write_csv(csv_path, rows, SEQ_FIELDS)
            _write_jsonl_for_judge(out_dir / f"sequential_T{max_turns_par}.jsonl", rows, "sequential")
            sr = _summary_row("sequential", rows)

        elif condition == "naive_parallel":
            print(f"\n=== NAIVE PARALLEL (k={K}, T={max_turns_par}) ===")
            rows = await run_naive_parallel(model, examples, max_turns_par, sem, traj_dir, run_seed=RUN_SEED)
            csv_path = out_dir / f"naive_parallel_T{max_turns_par}.csv"
            _write_csv(csv_path, rows, NAIVE_FIELDS)
            _write_jsonl_for_judge(out_dir / f"naive_parallel_T{max_turns_par}.jsonl", rows, "naive_parallel")
            sr = _summary_row("naive_parallel", rows)

        elif condition == "diversity_parallel":
            print(f"\n=== DIVERSITY PARALLEL (k={K}, T={max_turns_par}, pool={POOL_SIZE}, greedy-jaccard) ===")
            rows = await run_diversity_parallel(model, examples, max_turns_par, sem, traj_dir, run_seed=RUN_SEED)
            csv_path = out_dir / f"diversity_parallel_T{max_turns_par}.csv"
            _write_csv(csv_path, rows, DIV_FIELDS)
            _write_jsonl_for_judge(out_dir / f"diversity_parallel_T{max_turns_par}.jsonl", rows, "diversity_parallel")
            sr = _summary_row("diversity_parallel", rows)
        else:
            continue

        summary_rows.append(sr)
        print(f"  → {csv_path.name}  pass@1={sr['pass_at_1']:.3f}  pass@4={sr['pass_at_4']:.3f}  QPD={sr['mean_jaccard_qpd']:.3f}  ITC={sr['mean_itc']:.3f}  ATC={sr['mean_atc']:.3f}")

    if summary_rows:
        spath = out_dir / f"summary_T{max_turns_par}.csv"
        _write_csv(spath, summary_rows, SUMMARY_FIELDS)

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Condition':<24} {'N':>5} {'P@1':>6} {'P@4':>6} {'Synth':>7} {'QPD':>7} {'ITC':>7} {'ATC':>7}")
    print("-" * 90)
    for sr in summary_rows:
        print(
            f"{sr['condition']:<24} {sr['n_questions']:>5} "
            f"{float(sr['pass_at_1']):>6.3f} {float(sr['pass_at_4']):>6.3f} "
            f"{float(sr['synthesis_accuracy']):>7.3f} "
            f"{float(sr['mean_jaccard_qpd']):>7.3f} {float(sr['mean_itc']):>7.3f} {float(sr['mean_atc']):>7.3f}"
        )
    print("=" * 90)


def main() -> None:
    p = argparse.ArgumentParser(description="Main table experiment runner")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--condition", default=None, choices=CONDITION_CHOICES,
                   help="Run single condition (default: all three)")
    p.add_argument("--max-turns-par", type=int, default=5,
                   help="Turns per parallel thread (seq gets k× this)")
    p.add_argument("--max-concurrent", type=int, default=50,
                   help="Max concurrent questions in flight")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit dataset to first N examples (for smoke tests)")
    p.add_argument("--api-base", default=None,
                   help="vLLM endpoint URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--seed", type=int, default=0,
                   help="Global random seed (sets random, numpy, and per-rollout seeds)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
