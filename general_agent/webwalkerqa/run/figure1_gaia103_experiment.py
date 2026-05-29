"""
Figure 1 — GAIA-103: three compute-matched conditions.

Budget: B = max_turns × max_tokens_per_turn
  Sequential:      max_turns=20, k=1
  Naive parallel:  max_turns=5,  k=4  (4 × 5 = 20 turns, compute-matched)
  Div. parallel:   max_turns=5,  k=4, pool o=16, MMR Jaccard λ=0.75

Outputs (per condition):
  results/figure1/<condition>.csv          — one row per question (final metrics)
  results/figure1/trajectories/<cond>/     — one JSON file per question (turn logs)

Run one condition per job:
  python -m webwalkerqa.run.figure1_gaia103_experiment --condition sequential
  python -m webwalkerqa.run.figure1_gaia103_experiment --condition naive_parallel
  python -m webwalkerqa.run.figure1_gaia103_experiment --condition diversity_parallel
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_GA_DIR = Path(__file__).resolve().parent.parent
_TESTBED_ROOT = _GA_DIR.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

from webwalkerqa.dataset import load_dataset
from webwalkerqa.eval import exact_match
from webwalkerqa.llm import call_llm, normalize_model
import webwalkerqa.methods.diversity_scaling as _ds
from webwalkerqa.methods.diversity_scaling import (
    _extract_tag,
    _parse_pool_response,
    run_single_rollout,
    _safe_rollout_seed,
    SEED_MAX,
)
from webwalkerqa.methods.utils import jaccard_similarity, select_diverse_queries
from webwalkerqa.llm import call_llm as _call_llm_base

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "openai/gpt-4o-mini"
DEFAULT_DATASET = str(_GA_DIR / "data" / "gaia.json")
DEFAULT_OUT     = _TESTBED_ROOT / "results" / "figure1"
K               = 4     # parallel threads
MAX_TURNS_SEQ   = 20    # sequential budget
MAX_TURNS_PAR   = 5     # per-thread budget  (K × MAX_TURNS_PAR == MAX_TURNS_SEQ)
POOL_SIZE       = 16    # diversity pool o
RUN_SEED        = 0
SYNTH_SNIPPET   = 800   # chars of search result per turn in synthesis context

# ── GAIA-specific ReAct prompt (overrides diversity_scaling.REACT_PROMPT for this experiment)
# Improvements: explicit answer format, no-fluff guidance, forced answer on last turn
GAIA_REACT_PROMPT = """\
You are a precise research assistant answering GAIA benchmark questions by searching the web.
You have {max_turns} turns total. You are on turn {turn}.

Question: {question}

Search history:
{history}

Rules:
1. Use EXACTLY one format per response. No mixing search and answer.
2. To search: <thought>brief reasoning</thought><search>your exact search query</search>
3. To answer: <thought>brief reasoning</thought><answer>exact concise answer</answer>
4. Answers must be exact: a name, number, date, or short phrase. No explanations inside <answer>.
5. On turn {max_turns} you MUST give <answer>...</answer> — your best guess if uncertain.
6. Search for specific facts. Avoid generic queries. Use exact names, dates, constraints from the question.

Your response:"""

# ── Improved pool generation prompt for GAIA ──────────────────────────────
GAIA_POOL_PROMPT = """\
Generate exactly {o} diverse search queries to find the answer to this question.
Each query should target a DIFFERENT aspect, constraint, or component of the question.
Use specific terms — names, dates, locations, relationships mentioned in the question.
Avoid repeating the same search with minor variations.

Question: {question}

Output exactly {o} numbered queries (one per line, starting with 1.). No other text."""


async def _generate_pool_gaia(model: str, question: str, o: int):
    """Pool generation with GAIA-specific prompt."""
    prompt = GAIA_POOL_PROMPT.format(o=o, question=question)
    text, p_tok, o_tok = await _call_llm_base(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=2048, temperature=1.0,
    )
    queries = _parse_pool_response(text, o)
    while len(queries) < o:
        queries.append(f"{question[:50]} angle {len(queries)}")
    return queries[:o], p_tok, o_tok


# ── QPD helpers ───────────────────────────────────────────────────────────────

def jaccard_distance(p_i: str, p_j: str) -> float:
    tok_i = set(p_i.lower().split())
    tok_j = set(p_j.lower().split())
    if not tok_i and not tok_j:
        return 0.0
    return 1.0 - len(tok_i & tok_j) / len(tok_i | tok_j)


def token_edit_distance(p_i: str, p_j: str) -> float:
    ti = p_i.lower().split()
    tj = p_j.lower().split()
    m, n = len(ti), len(tj)
    if m == 0 and n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] if ti[i - 1] == tj[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    return dp[m][n] / max(m, n)


def compute_qpd(queries: List[str], dist_fn) -> float:
    k = len(queries)
    if k < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += dist_fn(queries[i], queries[j])
            count += 1
    return total / count


# ── Diversity selection: MMR Jaccard λ=0.75 ───────────────────────────────────
# Direct port of mmr_jaccard_select from diversity_parallel_benchmark.py.
# Best-performing k=4 method overall: λ=0.75 = 75% relevance to question,
# 25% push away from already-selected queries.
# Fixes the greedy max-min problem: pure diversity picks maximally-outlier
# queries (meta-questions, wrong angles) because the "obvious" correct search
# is the most similar to everything else and gets excluded. MMR keeps seeds
# grounded in the question while still enforcing diversity.

MMR_LAMBDA = 0.75


def select_diverse_mmr(pool: List[str], question: str, k: int) -> List[str]:
    """
    MMR (Maximal Marginal Relevance) with Jaccard similarity.
    score(q) = λ * sim(q, question) - (1-λ) * max_sim(q, already_selected)
    Init: most relevant query to the question (not most isolated).
    """
    n = len(pool)
    if k >= n:
        return list(pool)
    rel = [jaccard_similarity(p, question) for p in pool]
    first = int(np.argmax(rel))
    selected = [first]
    remaining = set(range(n)) - {first}
    while len(selected) < k and remaining:
        best_score = -1e9
        best_i = -1
        for i in remaining:
            max_sim_to_selected = max(jaccard_similarity(pool[i], pool[j]) for j in selected)
            score = MMR_LAMBDA * rel[i] - (1.0 - MMR_LAMBDA) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return [pool[i] for i in selected]


# ── Thread / synthesis helpers ────────────────────────────────────────────────

def _get_turn1_query(rollout: Dict[str, Any]) -> str:
    for log in (rollout.get("turn_logs") or []):
        if log.get("query"):
            return str(log["query"]).replace("\n", " ").strip()
    return ""


def _thread_summary(turn_logs: List[Dict[str, Any]]) -> str:
    """Full search trace per thread — same format as diversity_parallel_benchmark."""
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


# ── Synthesis (exact prompt from diversity_parallel_benchmark.synthesize_answer) ─

async def synthesize(
    model: str,
    question: str,
    rollouts: List[Dict[str, Any]],
    answers: List[str],
) -> str:
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
        model=model,
        max_tokens=1024,
        temperature=0.3,
    )
    ans = _extract_tag(text or "", "answer")
    return ans.strip() if ans else (text or "").strip()[:2000]


# ── Trajectory JSON (one file per question) ───────────────────────────────────

def _save_trajectory(traj_dir: Path, qid: str, data: Dict[str, Any]) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"{qid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Output helpers ────────────────────────────────────────────────────────────

def _out_path(out_dir: Path, name: str) -> Path:
    """Return path; if file already exists add timestamp suffix."""
    path = out_dir / name
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, ext = name.rsplit(".", 1)
        path = out_dir / f"{stem}_{ts}.{ext}"
        print(f"  [warn] existing file — saving as {path.name}")
    return path


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_jsonl_for_judge(path: Path, rows: List[Dict], condition: str) -> None:
    """Write JSONL in format expected by eval_llm.py judge script.
    Fields: question_id, question, answer_gt, rollout_answers (list).
    Sequential uses [predicted_answer]; parallel uses [ans1, ..., ans4].
    """
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


# ── Condition 1: Sequential ───────────────────────────────────────────────────

SEQ_FIELDS = [
    "question_id", "question", "predicted_answer", "gold_answer",
    "correct", "num_turns_used",
]


async def run_sequential(
    model: str, examples, sem: asyncio.Semaphore, traj_dir: Path
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
                    max_turns=MAX_TURNS_SEQ, initial_query=None,
                    rollout_seed=_safe_rollout_seed(qid, RUN_SEED, 0),
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


# ── Condition 2: Naive Parallel ───────────────────────────────────────────────

NAIVE_FIELDS = [
    "question_id", "question", "gold_answer",
    "turn1_query_1", "turn1_query_2", "turn1_query_3", "turn1_query_4",
    "thread_1_answer", "thread_2_answer", "thread_3_answer", "thread_4_answer",
    "oracle_correct", "synthesis_correct",
    "jaccard_qpd", "edit_qpd",
]


async def run_naive_parallel(
    model: str, examples, sem: asyncio.Semaphore, traj_dir: Path
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
                        max_turns=MAX_TURNS_PAR, initial_query=None,
                        rollout_seed=_safe_rollout_seed(qid, RUN_SEED, i),
                        question_id=qid,
                        react_temp_first=1.0,  # no seed → explore freely
                        react_temp_rest=0.7,
                    )
                    for i in range(K)
                ])
                answers = [(r.get("answer") or "").strip() for r in rollouts]
                t1q = [_get_turn1_query(r) for r in rollouts]
                oracle = 1 if any(exact_match(a, gold) for a in answers) else 0
                syn_ans = await synthesize(model, ex.question, list(rollouts), answers)
                syn_correct = 1 if exact_match(syn_ans, gold) else 0

            valid_t1q = [q for q in t1q if q]
            jqpd = compute_qpd(valid_t1q, jaccard_distance)
            eqpd = compute_qpd(valid_t1q, token_edit_distance)
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "turn1_query_1": t1q[0] if len(t1q) > 0 else "",
                "turn1_query_2": t1q[1] if len(t1q) > 1 else "",
                "turn1_query_3": t1q[2] if len(t1q) > 2 else "",
                "turn1_query_4": t1q[3] if len(t1q) > 3 else "",
                "thread_1_answer": answers[0] if len(answers) > 0 else "",
                "thread_2_answer": answers[1] if len(answers) > 1 else "",
                "thread_3_answer": answers[2] if len(answers) > 2 else "",
                "thread_4_answer": answers[3] if len(answers) > 3 else "",
                "oracle_correct": oracle, "synthesis_correct": syn_correct,
                "jaccard_qpd": f"{jqpd:.6f}", "edit_qpd": f"{eqpd:.6f}",
            }
            _save_trajectory(traj_dir, qid, {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "threads": [
                    {
                        "thread_id": i,
                        "turn1_query": t1q[i] if i < len(t1q) else "",
                        "answer": answers[i] if i < len(answers) else "",
                        "turn_logs": rollouts[i].get("turn_logs", []),
                        "full_responses": rollouts[i].get("full_responses", []),
                    }
                    for i in range(K)
                ],
                "oracle_correct": oracle,
                "synthesis_answer": syn_ans,
                "synthesis_correct": syn_correct,
                "jaccard_qpd": jqpd, "edit_qpd": eqpd,
            })
        except Exception as e:
            print(f"\n  [error] naive {qid}: {e}")
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "turn1_query_1": "", "turn1_query_2": "",
                "turn1_query_3": "", "turn1_query_4": "",
                "thread_1_answer": "ERROR", "thread_2_answer": "ERROR",
                "thread_3_answer": "ERROR", "thread_4_answer": "ERROR",
                "oracle_correct": 0, "synthesis_correct": 0,
                "jaccard_qpd": "0.0", "edit_qpd": "0.0",
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── Condition 3: Diversity Parallel ──────────────────────────────────────────

DIV_FIELDS = [
    "question_id", "question", "gold_answer",
    "pool_queries", "selected_queries",
    "thread_1_answer", "thread_2_answer", "thread_3_answer", "thread_4_answer",
    "oracle_correct", "synthesis_correct",
    "jaccard_qpd", "edit_qpd",
]


async def run_diversity_parallel(
    model: str, examples, sem: asyncio.Semaphore, traj_dir: Path
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
                # Single batch call with GAIA_POOL_PROMPT — asks for 16 diverse
                # queries targeting different constraints/components of the question.
                pool, _, _ = await _generate_pool_gaia(model, ex.question, POOL_SIZE)
                seeds = select_diverse_queries(pool, K, method="jaccard", seed=_safe_rollout_seed(qid, RUN_SEED, 0))
                while len(seeds) < K:
                    seeds.append(pool[len(seeds) % len(pool)])
                seeds = seeds[:K]

                rollouts = await asyncio.gather(*[
                    run_single_rollout(
                        model=model, question=ex.question, answer_gt=gold,
                        max_turns=MAX_TURNS_PAR, initial_query=seeds[i],
                        rollout_seed=_safe_rollout_seed(qid, RUN_SEED, i),
                        question_id=qid,
                        react_temp_first=0.7,  # seeded → stay focused
                        react_temp_rest=0.7,
                    )
                    for i in range(K)
                ])
                answers = [(r.get("answer") or "").strip() for r in rollouts]
                oracle = 1 if any(exact_match(a, gold) for a in answers) else 0
                syn_ans = await synthesize(model, ex.question, list(rollouts), answers)
                syn_correct = 1 if exact_match(syn_ans, gold) else 0

            jqpd = compute_qpd(seeds, jaccard_distance)
            eqpd = compute_qpd(seeds, token_edit_distance)
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries":     "; ".join(p.replace(";", ",") for p in pool),
                "selected_queries": "; ".join(s.replace(";", ",") for s in seeds),
                "thread_1_answer": answers[0] if len(answers) > 0 else "",
                "thread_2_answer": answers[1] if len(answers) > 1 else "",
                "thread_3_answer": answers[2] if len(answers) > 2 else "",
                "thread_4_answer": answers[3] if len(answers) > 3 else "",
                "oracle_correct": oracle, "synthesis_correct": syn_correct,
                "jaccard_qpd": f"{jqpd:.6f}", "edit_qpd": f"{eqpd:.6f}",
            }
            _save_trajectory(traj_dir, qid, {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": pool,
                "selected_seeds": seeds,
                "threads": [
                    {
                        "thread_id": i,
                        "seed_query": seeds[i] if i < len(seeds) else "",
                        "answer": answers[i] if i < len(answers) else "",
                        "turn_logs": rollouts[i].get("turn_logs", []),
                        "full_responses": rollouts[i].get("full_responses", []),
                    }
                    for i in range(K)
                ],
                "oracle_correct": oracle,
                "synthesis_answer": syn_ans,
                "synthesis_correct": syn_correct,
                "jaccard_qpd": jqpd, "edit_qpd": eqpd,
            })
        except Exception as e:
            print(f"\n  [error] div {qid}: {e}")
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": "", "selected_queries": "",
                "thread_1_answer": "ERROR", "thread_2_answer": "ERROR",
                "thread_3_answer": "ERROR", "thread_4_answer": "ERROR",
                "oracle_correct": 0, "synthesis_correct": 0,
                "jaccard_qpd": "0.0", "edit_qpd": "0.0",
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── Summary ───────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "condition", "n_questions", "n_correct", "accuracy",
    "oracle_accuracy", "synthesis_accuracy",
    "mean_jaccard_qpd", "mean_edit_qpd",
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
        oracle_acc = n_correct / n
        synth_acc  = n_correct / n
    else:
        n_correct = sum(int(r.get("oracle_correct", 0)) for r in rows)
        oracle_acc = n_correct / n
        synth_acc  = sum(int(r.get("synthesis_correct", 0)) for r in rows) / n

    return {
        "condition":          condition,
        "n_questions":        n,
        "n_correct":          n_correct,
        "accuracy":           sum(int(r.get("correct", 0)) for r in rows) / n,
        "oracle_accuracy":    oracle_acc,
        "synthesis_accuracy": synth_acc,
        "mean_jaccard_qpd":   _mean("jaccard_qpd"),
        "mean_edit_qpd":      _mean("edit_qpd"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    global MAX_TURNS_SEQ, MAX_TURNS_PAR
    if args.max_turns_seq is not None:
        MAX_TURNS_SEQ = args.max_turns_seq
    if args.max_turns_par is not None:
        MAX_TURNS_PAR = args.max_turns_par

    model   = normalize_model(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch REACT_PROMPT with GAIA-specific version (GAIA IDs are numeric → always hits default branch)
    _ds.REACT_PROMPT = GAIA_REACT_PROMPT

    examples = load_dataset(path=args.dataset)
    print(f"Loaded {len(examples)} questions | model={model} | condition={args.condition or 'all'}")
    print(f"Budget: seq={MAX_TURNS_SEQ} turns | par={MAX_TURNS_PAR} turns × {K} threads | pool o={POOL_SIZE}")

    sem = asyncio.Semaphore(args.max_concurrent)
    condition = args.condition
    summary_rows: List[Dict] = []

    if condition is None or condition == "sequential":
        traj_dir = out_dir / "trajectories" / "sequential"
        print(f"\n=== SEQUENTIAL (max_turns={MAX_TURNS_SEQ}) ===")
        rows = await run_sequential(model, examples, sem, traj_dir)
        path = _out_path(out_dir, "sequential.csv")
        _write_csv(path, rows, SEQ_FIELDS)
        _write_jsonl_for_judge(_out_path(out_dir, "sequential.jsonl"), rows, "sequential")
        sr = _summary_row("sequential", rows)
        summary_rows.append(sr)
        print(f"  → {path.name}  acc={sr['accuracy']:.3f}  ({sr['n_correct']}/{sr['n_questions']})")

    if condition is None or condition == "naive_parallel":
        traj_dir = out_dir / "trajectories" / "naive_parallel"
        print(f"\n=== NAIVE PARALLEL (k={K}, max_turns={MAX_TURNS_PAR}) ===")
        rows = await run_naive_parallel(model, examples, sem, traj_dir)
        path = _out_path(out_dir, "naive_parallel.csv")
        _write_csv(path, rows, NAIVE_FIELDS)
        _write_jsonl_for_judge(_out_path(out_dir, "naive_parallel.jsonl"), rows, "naive_parallel")
        sr = _summary_row("naive_parallel", rows)
        summary_rows.append(sr)
        print(f"  → {path.name}  oracle={sr['oracle_accuracy']:.3f}  syn={sr['synthesis_accuracy']:.3f}  j-qpd={sr['mean_jaccard_qpd']:.3f}")

    if condition is None or condition == "diversity_parallel":
        traj_dir = out_dir / "trajectories" / "diversity_parallel"
        print(f"\n=== DIVERSITY PARALLEL (k={K}, o={POOL_SIZE}, max_turns={MAX_TURNS_PAR}) ===")
        rows = await run_diversity_parallel(model, examples, sem, traj_dir)
        path = _out_path(out_dir, "diversity_parallel.csv")
        _write_csv(path, rows, DIV_FIELDS)
        _write_jsonl_for_judge(_out_path(out_dir, "diversity_parallel.jsonl"), rows, "diversity_parallel")
        sr = _summary_row("diversity_parallel", rows)
        summary_rows.append(sr)
        print(f"  → {path.name}  oracle={sr['oracle_accuracy']:.3f}  syn={sr['synthesis_accuracy']:.3f}  j-qpd={sr['mean_jaccard_qpd']:.3f}")

    if summary_rows:
        spath = _out_path(out_dir, "summary.csv")
        _write_csv(spath, summary_rows, SUMMARY_FIELDS)

    print("\n" + "=" * 72)
    print(f"{'Condition':<24} {'N':>5} {'Acc':>6} {'Oracle':>8} {'Synth':>8} {'J-QPD':>7} {'E-QPD':>7}")
    print("-" * 72)
    for sr in summary_rows:
        print(
            f"{sr['condition']:<24} {sr['n_questions']:>5} "
            f"{float(sr['accuracy']):>6.3f} {float(sr['oracle_accuracy']):>8.3f} "
            f"{float(sr['synthesis_accuracy']):>8.3f} "
            f"{float(sr['mean_jaccard_qpd']):>7.3f} {float(sr['mean_edit_qpd']):>7.3f}"
        )
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 1 GAIA-103 experiment")
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--dataset",        default=DEFAULT_DATASET)
    p.add_argument("--output-dir",     default=str(DEFAULT_OUT))
    p.add_argument("--condition",      default=None,
                   choices=["sequential", "naive_parallel", "diversity_parallel"])
    p.add_argument("--max-concurrent", type=int, default=100)
    p.add_argument("--max-turns-seq",  type=int, default=None,
                   help="Override MAX_TURNS_SEQ (default: 20)")
    p.add_argument("--max-turns-par",  type=int, default=None,
                   help="Override MAX_TURNS_PAR per thread (default: 5)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
