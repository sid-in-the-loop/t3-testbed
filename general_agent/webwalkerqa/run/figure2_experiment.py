"""
Figure 2 — Greedy-Jaccard vs MMR-Jaccard on GAIA-103.

Phases:
  1.  λ-tuning  : MMR-Jaccard on first 30 GAIA questions, λ∈{0.25,0.5,0.75}
  2.  Full runs : Greedy-Jaccard + MMR-best-λ on all 103 questions
  3.  QPD/ITC   : compute distributions (read figure1 data for sequential/naive)
  4.  Plots     : figure2_left.pdf/png  figure2_right.pdf/png
  5.  Summary   : figure2/summary.csv

Run:
  cd general_agent
  python -m webwalkerqa.run.figure2_experiment
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_GA_DIR      = Path(__file__).resolve().parent.parent
_TESTBED_ROOT = _GA_DIR.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

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
from webwalkerqa.methods.utils import jaccard_similarity
from webwalkerqa.llm import call_llm as _call_llm_base

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "openai/gpt-4o-mini"
DEFAULT_DATASET = str(_GA_DIR / "data" / "gaia.json")
DEFAULT_OUT     = _TESTBED_ROOT / "results" / "figure2"
FIGURE1_DIR     = _TESTBED_ROOT / "results" / "figure1"

K             = 4      # parallel threads
MAX_TURNS_PAR = 5      # per-thread budget
POOL_SIZE     = 16     # o
RUN_SEED      = 0
TUNE_N        = 30     # questions for λ-tuning
LAMBDA_CANDS  = [0.25, 0.50, 0.75]
MAX_CONCURRENT = 100
SYNTH_SNIPPET  = 800

# ── GAIA-specific prompts (same as figure1_gaia103_experiment.py) ──────────────
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

GAIA_POOL_PROMPT = """\
Generate exactly {o} diverse search queries to find the answer to this question.
Each query should target a DIFFERENT aspect, constraint, or component of the question.
Use specific terms — names, dates, locations, relationships mentioned in the question.
Avoid repeating the same search with minor variations.

Question: {question}

Output exactly {o} numbered queries (one per line, starting with 1.). No other text."""


async def _generate_pool_gaia(model: str, question: str, o: int):
    prompt = GAIA_POOL_PROMPT.format(o=o, question=question)
    text, p_tok, o_tok = await _call_llm_base(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=2048, temperature=1.0,
    )
    queries = _parse_pool_response(text, o)
    while len(queries) < o:
        queries.append(f"{question[:50]} angle {len(queries)}")
    return queries[:o], p_tok, o_tok


# ── diversity / selection ──────────────────────────────────────────────────────

def _tok(s: str) -> set:
    return set(s.lower().split())


def jaccard_distance(a: str, b: str) -> float:
    ta, tb = _tok(a), _tok(b)
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _jaccard_sim(a: str, b: str) -> float:
    return 1.0 - jaccard_distance(a, b)


def greedy_jaccard_select(pool: List[str], k: int) -> List[str]:
    """
    Max-min greedy Jaccard selection.
    Seed: query with highest mean Jaccard distance to all others.
    Then iteratively add query with max min-distance to already-selected set.
    """
    n = len(pool)
    if k >= n:
        return list(pool)
    # pairwise distance matrix
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(pool[i], pool[j])
            mat[i, j] = mat[j, i] = d
    # seed = max mean distance to all others
    mean_d = mat.mean(axis=1)
    first = int(np.argmax(mean_d))
    selected = [first]
    rem = set(range(n)) - {first}
    while len(selected) < k and rem:
        best_i = max(rem, key=lambda i: min(mat[i, j] for j in selected))
        selected.append(best_i)
        rem.remove(best_i)
    return [pool[i] for i in selected]


def mmr_jaccard_select(pool: List[str], question: str, k: int, lam: float) -> List[str]:
    """
    MMR-Jaccard: score(p_i) = (1-λ)*min_{p_j in S} d_J(p_i, p_j) + λ*rel(p_i, q)
    where rel(p_i, q) = sim_J(p_i, q) = 1 - d_J(p_i, q)
    Init: seed S with query maximising mean MMR score to all others.
    """
    n = len(pool)
    if k >= n:
        return list(pool)
    rel = [_jaccard_sim(p, question) for p in pool]
    # init seed: argmax relevance to question (as specified)
    first = int(np.argmax(rel))
    selected = [first]
    rem = set(range(n)) - {first}
    while len(selected) < k and rem:
        best_s = -1e9
        best_i = -1
        for i in rem:
            div = min(jaccard_distance(pool[i], pool[j]) for j in selected)
            score = (1.0 - lam) * div + lam * rel[i]
            if score > best_s:
                best_s = score
                best_i = i
        selected.append(best_i)
        rem.remove(best_i)
    return [pool[i] for i in selected]


def compute_qpd(queries: List[str]) -> float:
    """Mean pairwise Jaccard distance over C(k,2) pairs."""
    k = len(queries)
    if k < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += jaccard_distance(queries[i], queries[j])
            count += 1
    return total / count


def compute_itc(queries: List[str]) -> float:
    """
    ITC(τ) = (1/T-1) * sum_{t=2}^{T} sim_J(q1, qt)
    Jaccard similarity between turn-1 query and all subsequent.
    """
    if len(queries) < 2:
        return 0.0
    q1 = queries[0]
    sims = [_jaccard_sim(q1, qt) for qt in queries[1:]]
    return float(np.mean(sims))


# ── synthesis ──────────────────────────────────────────────────────────────────

def _thread_summary(turn_logs: List[Dict]) -> str:
    parts = []
    for log in turn_logs:
        if log.get("query"):
            snip = (log.get("search_result") or "")[:SYNTH_SNIPPET]
            parts.append(f"Query: {log['query']}\nExcerpt: {snip}")
        if log.get("answer"):
            parts.append(f"Thread conclusion: {log['answer']}")
    return "\n".join(parts) if parts else "(no search trace)"


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


# ── trajectory helpers ─────────────────────────────────────────────────────────

def _get_turn1_query(rollout: Dict) -> str:
    for log in (rollout.get("turn_logs") or []):
        if log.get("query"):
            return str(log["query"]).replace("\n", " ").strip()
    return ""


def _get_all_queries(rollout: Dict) -> List[str]:
    """Extract all search queries in turn order."""
    return [
        str(log["query"]).replace("\n", " ").strip()
        for log in (rollout.get("turn_logs") or [])
        if log.get("query")
    ]


def _save_trajectory(traj_dir: Path, qid: str, data: Dict) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    with open(traj_dir / f"{qid}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _gold(ex) -> str:
    return str(ex.answer) if not isinstance(ex.answer, list) else str(ex.answer[0])


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, dynamic_ncols=True, ascii=True, **kwargs)
    except ImportError:
        return iterable


# ── single-question runner ─────────────────────────────────────────────────────

async def run_one_question(
    model: str,
    ex,
    seeds: List[str],          # length k; each seed is the injected Turn-1 query
    qid: str,
    gold: str,
    sem: asyncio.Semaphore,
    traj_dir: Path,
) -> Dict:
    """
    Run k parallel threads seeded by `seeds`.
    Returns per-question result dict including turn-by-turn query logs.
    """
    async with sem:
        rollouts = await asyncio.gather(*[
            run_single_rollout(
                model=model, question=ex.question, answer_gt=gold,
                max_turns=MAX_TURNS_PAR, initial_query=seeds[i],
                rollout_seed=_safe_rollout_seed(qid, RUN_SEED, i),
                question_id=qid,
                react_temp_first=0.7,
                react_temp_rest=0.7,
            )
            for i in range(K)
        ])
        answers = [(r.get("answer") or "").strip() for r in rollouts]
        oracle  = 1 if any(exact_match(a, gold) for a in answers) else 0
        syn_ans = await synthesize(model, ex.question, list(rollouts), answers)
        syn_correct = 1 if exact_match(syn_ans, gold) else 0

    # Per-thread query logs for ITC
    thread_queries: List[List[str]] = [_get_all_queries(r) for r in rollouts]
    t1_queries = [qs[0] if qs else "" for qs in thread_queries]

    qpd = compute_qpd(seeds)  # use injected seeds (= Turn-1 queries)
    itc_per_thread = [compute_itc(qs) for qs in thread_queries]
    itc_mean = float(np.mean(itc_per_thread)) if itc_per_thread else 0.0

    _save_trajectory(traj_dir, qid, {
        "question_id": qid, "question": ex.question, "gold_answer": gold,
        "seeds": seeds,
        "threads": [
            {
                "thread_id": i,
                "seed_query": seeds[i],
                "turn1_query_actual": t1_queries[i],
                "answer": answers[i],
                "turn_logs": rollouts[i].get("turn_logs", []),
                "all_queries": thread_queries[i],
                "itc": itc_per_thread[i],
            }
            for i in range(K)
        ],
        "oracle_correct": oracle,
        "synthesis_answer": syn_ans,
        "synthesis_correct": syn_correct,
        "jaccard_qpd": qpd,
    })

    return {
        "question_id": qid,
        "question": ex.question,
        "gold_answer": gold,
        "seeds": seeds,
        "answers": answers,
        "oracle_correct": oracle,
        "synthesis_correct": syn_correct,
        "jaccard_qpd": qpd,
        "itc_per_thread": itc_per_thread,
        "itc_mean": itc_mean,
        "thread_queries": thread_queries,
    }


# ── condition runners ──────────────────────────────────────────────────────────

GREEDY_FIELDS = [
    "question_id", "question", "gold_answer",
    "pool_queries", "selected_queries",
    "thread_1_answer", "thread_2_answer", "thread_3_answer", "thread_4_answer",
    "oracle_correct", "synthesis_correct",
    "jaccard_qpd",
    "turn1_query_1", "turn1_query_2", "turn1_query_3", "turn1_query_4",
]

MMR_FIELDS = [
    "question_id", "question", "gold_answer",
    "pool_queries", "selected_queries",
    "thread_1_answer", "thread_2_answer", "thread_3_answer", "thread_4_answer",
    "oracle_correct", "synthesis_correct",
    "jaccard_qpd",
    "turn1_query_1", "turn1_query_2", "turn1_query_3", "turn1_query_4",
]


async def _run_condition(
    condition: str,
    model: str,
    examples,
    sem: asyncio.Semaphore,
    out_dir: Path,
    lam: Optional[float] = None,        # None → greedy; float → MMR
    desc: str = "",
) -> List[Dict]:
    traj_dir = out_dir / "trajectories" / condition
    rows: List[Dict] = []
    lock = asyncio.Lock()
    pbar = _tqdm(range(len(examples)), desc=desc or condition, unit="q")

    async def one(ex):
        qid = str(ex.id)
        gold = _gold(ex)
        pool: List[str] = []
        seeds: List[str] = []
        result: Dict = {}
        try:
            # generate pool
            async with sem:
                pool, _, _ = await _generate_pool_gaia(model, ex.question, POOL_SIZE)

            # select seeds
            if lam is None:
                seeds = greedy_jaccard_select(pool, K)
            else:
                seeds = mmr_jaccard_select(pool, ex.question, K, lam)

            # pad if needed
            while len(seeds) < K:
                seeds.append(pool[len(seeds) % len(pool)])
            seeds = seeds[:K]

            result = await run_one_question(model, ex, seeds, qid, gold, sem, traj_dir)

            row = {
                "question_id": qid,
                "question": ex.question,
                "gold_answer": gold,
                "pool_queries": "; ".join(p.replace(";", ",") for p in pool),
                "selected_queries": "; ".join(s.replace(";", ",") for s in seeds),
                "thread_1_answer": result["answers"][0] if len(result["answers"]) > 0 else "",
                "thread_2_answer": result["answers"][1] if len(result["answers"]) > 1 else "",
                "thread_3_answer": result["answers"][2] if len(result["answers"]) > 2 else "",
                "thread_4_answer": result["answers"][3] if len(result["answers"]) > 3 else "",
                "oracle_correct": result["oracle_correct"],
                "synthesis_correct": result["synthesis_correct"],
                "jaccard_qpd": f"{result['jaccard_qpd']:.6f}",
                "turn1_query_1": result["thread_queries"][0][0] if result["thread_queries"] and result["thread_queries"][0] else "",
                "turn1_query_2": result["thread_queries"][1][0] if len(result["thread_queries"]) > 1 and result["thread_queries"][1] else "",
                "turn1_query_3": result["thread_queries"][2][0] if len(result["thread_queries"]) > 2 and result["thread_queries"][2] else "",
                "turn1_query_4": result["thread_queries"][3][0] if len(result["thread_queries"]) > 3 and result["thread_queries"][3] else "",
                # ITC stored in trajectory; attach to row too
                "_itc_per_thread": result.get("itc_per_thread", []),
                "_itc_mean": result.get("itc_mean", 0.0),
            }
        except Exception as e:
            print(f"\n  [error] {condition} {qid}: {e}")
            row = {
                "question_id": qid, "question": ex.question, "gold_answer": gold,
                "pool_queries": "; ".join(pool),
                "selected_queries": "",
                "thread_1_answer": "ERROR", "thread_2_answer": "ERROR",
                "thread_3_answer": "ERROR", "thread_4_answer": "ERROR",
                "oracle_correct": 0, "synthesis_correct": 0,
                "jaccard_qpd": "0.0",
                "turn1_query_1": "", "turn1_query_2": "",
                "turn1_query_3": "", "turn1_query_4": "",
                "_itc_per_thread": [],
                "_itc_mean": 0.0,
            }
        async with lock:
            rows.append(row)
            pbar.update(1)

    await asyncio.gather(*[one(ex) for ex in examples])
    pbar.close()
    return rows


# ── λ tuning ───────────────────────────────────────────────────────────────────

async def tune_lambda(model: str, examples, sem: asyncio.Semaphore, out_dir: Path) -> float:
    """Run MMR at λ∈{0.25,0.5,0.75} on first TUNE_N questions. Return best λ."""
    tune_examples = examples[:TUNE_N]
    print(f"\n=== λ-TUNING on first {TUNE_N} questions, λ∈{LAMBDA_CANDS} ===")

    results_by_lam: Dict[float, List[Dict]] = {}
    for lam in LAMBDA_CANDS:
        label = f"mmr_tune_l{int(lam*100):03d}"
        print(f"\n  λ={lam} ...")
        rows = await _run_condition(
            condition=label, model=model, examples=tune_examples,
            sem=sem, out_dir=out_dir, lam=lam,
            desc=f"tune λ={lam}",
        )
        results_by_lam[lam] = rows

    # print tuning table
    print("\n─── λ Tuning Results ─────────────────────────────")
    print(f"{'λ':>6}  {'oracle_pass@4':>14}  {'n_correct':>10}")
    print("─" * 42)
    tuning_records = []
    for lam in LAMBDA_CANDS:
        rows = results_by_lam[lam]
        n = len(rows)
        nc = sum(int(r.get("oracle_correct", 0)) for r in rows)
        acc = nc / n if n else 0.0
        print(f"  {lam:>4}  {acc:>14.4f}  {nc:>6}/{n}")
        tuning_records.append({"lambda": lam, "n_questions": n, "oracle_pass4": f"{acc:.6f}"})

    # save tuning CSV
    tune_csv = out_dir / "mmr_lambda_tuning.csv"
    _write_csv(tune_csv, tuning_records, ["lambda", "n_questions", "oracle_pass4"])
    print(f"  Saved: {tune_csv}")

    # pick best λ
    best_lam = max(LAMBDA_CANDS, key=lambda l: float(
        next(r["oracle_pass4"] for r in tuning_records if r["lambda"] == l)
    ))
    print(f"\n  Best λ = {best_lam}\n")
    return best_lam


# ── ITC computation from figure1 trajectories ──────────────────────────────────

def _extract_queries_from_traj(traj_path: Path, condition: str) -> Dict[str, List[str]]:
    """Return {thread_id_str: [q1, q2, ...]} from a trajectory JSON."""
    with open(traj_path) as f:
        d = json.load(f)
    queries_by_thread: Dict[str, List[str]] = {}

    if condition == "sequential":
        # single thread, flat turn_logs
        qs = [
            str(log["query"]).replace("\n", " ").strip()
            for log in d.get("turn_logs", [])
            if log.get("query")
        ]
        queries_by_thread["0"] = qs
    else:
        # parallel: d["threads"] list
        for th in d.get("threads", []):
            tid = str(th.get("thread_id", 0))
            qs = [
                str(log["query"]).replace("\n", " ").strip()
                for log in th.get("turn_logs", [])
                if log.get("query")
            ]
            queries_by_thread[tid] = qs
    return queries_by_thread


def compute_itc_from_figure1(condition: str, figure1_dir: Path) -> List[Tuple[str, str, float]]:
    """
    Returns list of (question_id, thread_id, itc_score) for figure1 conditions.
    """
    traj_dir = figure1_dir / "trajectories" / condition
    records = []
    if not traj_dir.exists():
        print(f"  [warn] trajectory dir not found: {traj_dir}")
        return records
    for traj_file in sorted(traj_dir.glob("*.json")):
        qid = traj_file.stem
        try:
            qbt = _extract_queries_from_traj(traj_file, condition)
            for tid, qs in qbt.items():
                itc = compute_itc(qs)
                records.append((qid, tid, itc))
        except Exception as e:
            print(f"  [warn] ITC error {traj_file.name}: {e}")
    return records


def compute_itc_from_figure2_trajectories(
    condition: str, out_dir: Path
) -> List[Tuple[str, str, float]]:
    """Read figure2/trajectories/<condition>/*.json and extract ITC per thread."""
    traj_dir = out_dir / "trajectories" / condition
    records = []
    if not traj_dir.exists():
        return records
    for traj_file in sorted(traj_dir.glob("*.json")):
        qid = traj_file.stem
        try:
            with open(traj_file) as f:
                d = json.load(f)
            for th in d.get("threads", []):
                tid = str(th["thread_id"])
                qs = th.get("all_queries", [])
                if not qs:
                    # fallback: extract from turn_logs
                    qs = [
                        str(log["query"]).strip()
                        for log in th.get("turn_logs", [])
                        if log.get("query")
                    ]
                itc = compute_itc(qs)
                records.append((qid, tid, itc))
        except Exception as e:
            print(f"  [warn] ITC error {traj_file.name}: {e}")
    return records


# ── plots ──────────────────────────────────────────────────────────────────────

def _kde_curve(data: List[float], x_grid: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde
    if len(data) < 2:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(data, bw_method="scott")
    return kde(x_grid)


def plot_qpd(qpd_records: List[Dict], out_dir: Path) -> None:
    """Left panel: KDE of QPD for naive, greedy_jaccard, mmr_jaccard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # condition → list of qpd values
    cond_data: Dict[str, List[float]] = {}
    for r in qpd_records:
        c = r["condition"]
        cond_data.setdefault(c, []).append(float(r["jaccard_qpd"]))

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    x = np.linspace(0, 1, 500)
    palette = {
        "naive_parallel":    ("#E07B39", "Naive Parallel"),
        "greedy_jaccard":    ("#2563EB", "Diversity-Forced"),
        "mmr_jaccard":       ("#16A34A", "Relevance + Diversity"),
    }

    for cname, (color, label) in palette.items():
        vals = cond_data.get(cname, [])
        if not vals:
            continue
        y = _kde_curve(vals, x)
        ax.plot(x, y, color=color, linewidth=2.0, label=label)
        # inline label at peak
        peak_i = int(np.argmax(y))
        ax.text(
            x[peak_i], y[peak_i] + 0.3, label,
            color=color, fontsize=8, ha="center", va="bottom",
            fontfamily="serif",
        )

    # Arrow annotation on naive curve
    naive_vals = cond_data.get("naive_parallel", [])
    if naive_vals:
        y_naive = _kde_curve(naive_vals, x)
        peak_i = int(np.argmax(y_naive))
        ax.annotate(
            "Naive sampling\nclusters at Turn-1",
            xy=(x[peak_i], y_naive[peak_i]),
            xytext=(x[peak_i] + 0.15, y_naive[peak_i] + 1.5),
            arrowprops=dict(arrowstyle="->", color="#E07B39", lw=1.2),
            fontsize=7, color="#E07B39", fontfamily="serif",
            ha="left",
        )
    # Arrow on diversity / mmr showing rightward shift
    for cname, (color, label) in palette.items():
        if cname == "naive_parallel":
            continue
        vals = cond_data.get(cname, [])
        if not vals:
            continue
        y = _kde_curve(vals, x)
        peak_i = int(np.argmax(y))
        if cname == "greedy_jaccard":
            ax.annotate(
                "Diversity forces\nrightward shift",
                xy=(x[peak_i], y[peak_i]),
                xytext=(x[peak_i] - 0.18, y[peak_i] + 1.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                fontsize=7, color=color, fontfamily="serif",
                ha="right",
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Turn-1 Query Diversity (QPD)", fontsize=10, fontfamily="serif")
    ax.set_ylabel("Density", fontsize=10, fontfamily="serif")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend_ = None  # no legend box
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"figure2_left.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_itc(itc_records: List[Dict], out_dir: Path) -> None:
    """Right panel: KDE of ITC for all four conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_data: Dict[str, List[float]] = {}
    for r in itc_records:
        c = r["condition"]
        cond_data.setdefault(c, []).append(float(r["itc_score"]))

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.linspace(0, 1, 500)

    palette = {
        "sequential":     ("#888888", "Sequential"),
        "naive_parallel": ("#E07B39", "Naive Parallel"),
        "greedy_jaccard": ("#2563EB", "Diversity-Forced"),
        "mmr_jaccard":    ("#16A34A", "Relevance + Diversity"),
    }

    for cname, (color, label) in palette.items():
        vals = cond_data.get(cname, [])
        if not vals:
            continue
        y = _kde_curve(vals, x)
        ax.plot(x, y, color=color, linewidth=2.0, label=label)
        peak_i = int(np.argmax(y))
        ax.text(
            x[peak_i], y[peak_i] + 0.3, label,
            color=color, fontsize=8, ha="center", va="bottom",
            fontfamily="serif",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Inter-Turn Coherence (ITC)", fontsize=10, fontfamily="serif")
    ax.set_ylabel("Density", fontsize=10, fontfamily="serif")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend_ = None
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"figure2_right.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── summary helpers ────────────────────────────────────────────────────────────

def _summary_row_parallel(condition: str, rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "oracle_pass4": 0, "synthesis_acc": 0,
                "mean_QPD": 0, "mean_ITC": 0}
    oracle = sum(int(r.get("oracle_correct", 0)) for r in rows) / n
    synth  = sum(int(r.get("synthesis_correct", 0)) for r in rows) / n
    qpds = [float(r.get("jaccard_qpd", 0)) for r in rows if r.get("jaccard_qpd") not in ("", None)]
    itcs = [float(r.get("_itc_mean", 0)) for r in rows if r.get("_itc_mean") not in ("", None)]
    return {
        "condition": condition,
        "oracle_pass4": f"{oracle:.4f}",
        "synthesis_acc": f"{synth:.4f}",
        "mean_QPD": f"{np.mean(qpds):.4f}" if qpds else "0.0000",
        "mean_ITC": f"{np.mean(itcs):.4f}" if itcs else "0.0000",
    }


# ── main ───────────────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    model   = normalize_model(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch REACT_PROMPT with GAIA-specific version
    _ds.REACT_PROMPT = GAIA_REACT_PROMPT

    examples = load_dataset(path=args.dataset)
    print(f"Loaded {len(examples)} questions | model={model}")
    print(f"Budget: par={MAX_TURNS_PAR} turns × {K} threads | pool o={POOL_SIZE}")

    sem = asyncio.Semaphore(args.max_concurrent)

    # ── Phase 1: λ tuning ──────────────────────────────────────────────────────
    tune_csv = out_dir / "mmr_lambda_tuning.csv"
    if tune_csv.exists() and not args.retune:
        print(f"\n[skip λ-tuning] {tune_csv} exists. Reading best λ ...")
        with open(tune_csv) as f:
            rows = list(csv.DictReader(f))
        best_lam = max(rows, key=lambda r: float(r["oracle_pass4"]))["lambda"]
        best_lam = float(best_lam)
        print(f"  Best λ from file: {best_lam}")
        for r in rows:
            print(f"    λ={r['lambda']}  oracle_pass@4={float(r['oracle_pass4']):.4f}")
    else:
        best_lam = await tune_lambda(model, examples, sem, out_dir)

    # ── Phase 2: Greedy-Jaccard full run ───────────────────────────────────────
    greedy_csv = out_dir / "greedy_jaccard.csv"
    greedy_rows: List[Dict] = []
    if greedy_csv.exists() and not args.rerun:
        print(f"\n[skip] greedy_jaccard.csv exists, loading ...")
        with open(greedy_csv) as f:
            greedy_rows = list(csv.DictReader(f))
    else:
        print(f"\n=== GREEDY-JACCARD (k={K}, o={POOL_SIZE}, max_turns={MAX_TURNS_PAR}) ===")
        greedy_rows = await _run_condition(
            condition="greedy_jaccard", model=model, examples=examples,
            sem=sem, out_dir=out_dir, lam=None,
            desc=f"greedy-jaccard",
        )
        _write_csv(greedy_csv, greedy_rows, GREEDY_FIELDS)
        sr = _summary_row_parallel("greedy_jaccard", greedy_rows)
        print(f"  → greedy_jaccard.csv  oracle={sr['oracle_pass4']}  syn={sr['synthesis_acc']}  j-qpd={sr['mean_QPD']}")

    # ── Phase 3: MMR-Jaccard full run ─────────────────────────────────────────
    mmr_csv = out_dir / "mmr_jaccard.csv"
    mmr_rows: List[Dict] = []
    if mmr_csv.exists() and not args.rerun:
        print(f"\n[skip] mmr_jaccard.csv exists, loading ...")
        with open(mmr_csv) as f:
            mmr_rows = list(csv.DictReader(f))
    else:
        print(f"\n=== MMR-JACCARD λ={best_lam} (k={K}, o={POOL_SIZE}, max_turns={MAX_TURNS_PAR}) ===")
        mmr_rows = await _run_condition(
            condition="mmr_jaccard", model=model, examples=examples,
            sem=sem, out_dir=out_dir, lam=best_lam,
            desc=f"mmr-jaccard λ={best_lam}",
        )
        _write_csv(mmr_csv, mmr_rows, MMR_FIELDS)
        sr = _summary_row_parallel("mmr_jaccard", mmr_rows)
        print(f"  → mmr_jaccard.csv  oracle={sr['oracle_pass4']}  syn={sr['synthesis_acc']}  j-qpd={sr['mean_QPD']}")

    # ── Phase 4: QPD distributions ─────────────────────────────────────────────
    print("\n=== Computing QPD distributions ===")
    qpd_records: List[Dict] = []

    # Naive: read jaccard_qpd from figure1/naive_parallel.csv
    naive_csv = FIGURE1_DIR / "naive_parallel.csv"
    if naive_csv.exists():
        with open(naive_csv) as f:
            for row in csv.DictReader(f):
                qpd_records.append({
                    "question_id": row["question_id"],
                    "condition": "naive_parallel",
                    "jaccard_qpd": row.get("jaccard_qpd", "0"),
                })
        print(f"  naive_parallel: {len([r for r in qpd_records if r['condition']=='naive_parallel'])} rows")
    else:
        print(f"  [warn] figure1/naive_parallel.csv not found")

    # Greedy and MMR: from current rows
    for row in greedy_rows:
        qpd_records.append({
            "question_id": row["question_id"],
            "condition": "greedy_jaccard",
            "jaccard_qpd": row.get("jaccard_qpd", "0"),
        })
    for row in mmr_rows:
        qpd_records.append({
            "question_id": row["question_id"],
            "condition": "mmr_jaccard",
            "jaccard_qpd": row.get("jaccard_qpd", "0"),
        })

    qpd_csv = out_dir / "qpd_distributions.csv"
    _write_csv(qpd_csv, qpd_records, ["question_id", "condition", "jaccard_qpd"])
    print(f"  Saved: {qpd_csv}")

    # ── Phase 5: ITC distributions ─────────────────────────────────────────────
    print("\n=== Computing ITC distributions ===")
    itc_records: List[Dict] = []

    # Sequential & Naive from figure1 trajectories
    for cond in ("sequential", "naive_parallel"):
        recs = compute_itc_from_figure1(cond, FIGURE1_DIR)
        for qid, tid, itc in recs:
            itc_records.append({"question_id": qid, "condition": cond,
                                 "thread_id": tid, "itc_score": f"{itc:.6f}"})
        print(f"  {cond}: {len(recs)} thread trajectories")

    # Greedy and MMR from figure2 trajectories
    for cond in ("greedy_jaccard", "mmr_jaccard"):
        recs = compute_itc_from_figure2_trajectories(cond, out_dir)
        for qid, tid, itc in recs:
            itc_records.append({"question_id": qid, "condition": cond,
                                 "thread_id": tid, "itc_score": f"{itc:.6f}"})
        print(f"  {cond}: {len(recs)} thread trajectories")

    itc_csv = out_dir / "itc_distributions.csv"
    _write_csv(itc_csv, itc_records, ["question_id", "condition", "thread_id", "itc_score"])
    print(f"  Saved: {itc_csv}")

    # ── Phase 6: Plots ─────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_qpd(qpd_records, out_dir)
    plot_itc(itc_records, out_dir)

    # ── Phase 7: Summary ───────────────────────────────────────────────────────
    print("\n=== Final Summary ===")

    # sequential from figure1
    seq_csv = FIGURE1_DIR / "sequential.csv"
    seq_oracle = 0.0
    seq_n = 0
    if seq_csv.exists():
        with open(seq_csv) as f:
            seq_rows = list(csv.DictReader(f))
        seq_n = len(seq_rows)
        seq_oracle = sum(int(r.get("correct", 0)) for r in seq_rows) / seq_n if seq_n else 0.0

    naive_oracle = 0.0
    naive_n = 0
    if naive_csv.exists():
        with open(naive_csv) as f:
            naive_rows_f1 = list(csv.DictReader(f))
        naive_n = len(naive_rows_f1)
        naive_oracle = sum(int(r.get("oracle_correct", 0)) for r in naive_rows_f1) / naive_n if naive_n else 0.0

    greedy_sr = _summary_row_parallel("greedy_jaccard", greedy_rows)
    mmr_sr    = _summary_row_parallel("mmr_jaccard", mmr_rows)

    # mean QPD from distributions
    def _mean_qpd(cond: str) -> float:
        vals = [float(r["jaccard_qpd"]) for r in qpd_records if r["condition"] == cond and r["jaccard_qpd"]]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_itc(cond: str) -> float:
        vals = [float(r["itc_score"]) for r in itc_records if r["condition"] == cond and r["itc_score"]]
        return float(np.mean(vals)) if vals else 0.0

    summary_rows = [
        {
            "condition": "sequential",
            "oracle_pass4": f"{seq_oracle:.4f}",
            "mean_QPD": "0.0000",
            "mean_ITC": f"{_mean_itc('sequential'):.4f}",
        },
        {
            "condition": "naive_parallel",
            "oracle_pass4": f"{naive_oracle:.4f}",
            "mean_QPD": f"{_mean_qpd('naive_parallel'):.4f}",
            "mean_ITC": f"{_mean_itc('naive_parallel'):.4f}",
        },
        {
            "condition": "greedy_jaccard",
            "oracle_pass4": greedy_sr["oracle_pass4"],
            "mean_QPD": greedy_sr["mean_QPD"],
            "mean_ITC": f"{_mean_itc('greedy_jaccard'):.4f}",
        },
        {
            "condition": "mmr_jaccard",
            "oracle_pass4": mmr_sr["oracle_pass4"],
            "mean_QPD": mmr_sr["mean_QPD"],
            "mean_ITC": f"{_mean_itc('mmr_jaccard'):.4f}",
        },
    ]

    summary_csv = out_dir / "summary.csv"
    _write_csv(summary_csv, summary_rows, ["condition", "oracle_pass4", "mean_QPD", "mean_ITC"])
    print(f"\n  Saved: {summary_csv}")

    # print table
    print("\n" + "=" * 68)
    print(f"{'Condition':<24} {'Oracle pass@4':>14} {'Mean QPD':>10} {'Mean ITC':>10}")
    print("-" * 68)
    for sr in summary_rows:
        print(
            f"{sr['condition']:<24} {float(sr['oracle_pass4']):>14.4f} "
            f"{float(sr['mean_QPD']):>10.4f} {float(sr['mean_ITC']):>10.4f}"
        )
    print("=" * 68)
    print(f"\n  Best MMR λ used: {best_lam}")
    print(f"  All outputs in: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 2: Greedy-Jaccard vs MMR-Jaccard")
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--dataset",        default=DEFAULT_DATASET)
    p.add_argument("--output-dir",     default=str(DEFAULT_OUT))
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--retune",  action="store_true", help="Re-run λ tuning even if CSV exists")
    p.add_argument("--rerun",   action="store_true", help="Re-run full conditions even if CSVs exist")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
