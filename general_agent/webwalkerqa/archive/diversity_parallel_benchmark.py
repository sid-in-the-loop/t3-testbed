"""
Diversity-in-parallel benchmark: GAIA-25 + HotpotQA-25, variable k/o, MMR-Jaccard (λ).

Default data: data/gaia_25.json, data/hotpotqa_25-random.jsonl
  python -m webwalkerqa.diversity_parallel_benchmark --run-all

Search budget: 20 total LLM-search turns per question → parallel threads get max_turns = 20 // k.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_GA_DIR = Path(__file__).resolve().parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")

from webwalkerqa.dataset import QAExample, load_dataset
from webwalkerqa.llm import call_llm, normalize_model
from webwalkerqa.methods.diversity_scaling import generate_pool, run_single_rollout
from webwalkerqa.methods.utils import (
    compute_jaccard_distance_matrix,
    jaccard_similarity,
)

logger = logging.getLogger(__name__)

DEFAULT_GAIA_PATH = _GA_DIR / "data" / "gaia_25.json"
DEFAULT_HOTPOT_PATH = _GA_DIR / "data" / "hotpotqa_25-random.jsonl"

MODEL = "openai/gpt-4o-mini"
SEARCH_BUDGET = 20
T_SEQUENTIAL = 20
SAMPLE_SEED = 42
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RESULT_SNIPPET_LEN = 500
SYNTH_MAX = 12000

CONDITIONS_ORDER = [
    "sequential",
    "naive_k2",
    "naive_k4",
    "naive_k8",
    "greedy_jaccard_o8_k2",
    "greedy_jaccard_o16_k4",
    "greedy_jaccard_o32_k8",
    "random_o8_k2",
    "random_o16_k4",
    "random_o32_k8",
] + [
    f"mmr_jaccard_o{o}_k{k}_l{lam:03d}"
    for o, k in ((8, 2), (16, 4), (32, 8))
    for lam in (25, 50, 75)
]


def condition_config(condition: str) -> Dict[str, Any]:
    """Parse condition name → k threads, o pool, turns per thread, mode, mmr λ."""
    if condition == "sequential":
        return {"mode": "sequential", "k": 1, "o": 0, "t_parallel": T_SEQUENTIAL, "lam": None}
    import re

    m = re.match(r"^naive_k(\d+)$", condition)
    if m:
        k = int(m.group(1))
        return {
            "mode": "naive",
            "k": k,
            "o": 0,
            "t_parallel": max(1, SEARCH_BUDGET // k),
            "lam": None,
        }
    m = re.match(r"^greedy_jaccard_o(\d+)_k(\d+)$", condition)
    if m:
        o, k = int(m.group(1)), int(m.group(2))
        return {
            "mode": "greedy_jaccard",
            "k": k,
            "o": o,
            "t_parallel": max(1, SEARCH_BUDGET // k),
            "lam": None,
        }
    m = re.match(r"^random_o(\d+)_k(\d+)$", condition)
    if m:
        o, k = int(m.group(1)), int(m.group(2))
        return {
            "mode": "random",
            "k": k,
            "o": o,
            "t_parallel": max(1, SEARCH_BUDGET // k),
            "lam": None,
        }
    m = re.match(r"^mmr_jaccard_o(\d+)_k(\d+)_l(\d+)$", condition)
    if m:
        o, k, li = int(m.group(1)), int(m.group(2)), int(m.group(3))
        lam = li / 100.0  # l025→0.25, l050→0.5, l075→0.75
        return {
            "mode": "mmr_jaccard",
            "k": k,
            "o": o,
            "t_parallel": max(1, SEARCH_BUDGET // k),
            "lam": lam,
        }
    raise ValueError(
        f"Unknown condition {condition!r}. See CONDITIONS_ORDER or --condition matching "
        "sequential | naive_k{N} | greedy_jaccard_o{O}_k{K} | random_o{O}_k{K} | "
        "mmr_jaccard_o{O}_k{K}_l025|l050|l075"
    )


def normalize_for_em(s: str) -> str:
    """Lowercase, strip punctuation, drop a/an/the tokens, collapse whitespace."""
    s = str(s or "").lower()
    toks = re.findall(r"\w+", s)
    stop = {"a", "an", "the"}
    toks = [t for t in toks if t not in stop]
    return " ".join(toks)


def em_match(pred: str, gold: str) -> bool:
    return normalize_for_em(pred) == normalize_for_em(gold)


def _log_api(msg: str, log_path: Optional[Path]) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts} {msg}\n"
    logger.info(msg)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def load_gaia25(gaia_path: Path, max_q: int = 25) -> List[QAExample]:
    """Load GAIA examples in file order (curated gaia_25.json)."""
    examples = load_dataset(str(gaia_path))
    if len(examples) > max_q:
        logger.warning(
            "GAIA file has %d rows; using first %d in file order",
            len(examples),
            max_q,
        )
    return examples[:max_q]


def load_hotpot25(hotpot_path: Path, max_q: int = 25) -> List[QAExample]:
    """Load HotpotQA-25 from JSONL in file order (hotpotqa_25-random.jsonl)."""
    examples = load_dataset(str(hotpot_path))
    if len(examples) > max_q:
        logger.warning(
            "Hotpot JSONL has %d rows; using first %d in file order",
            len(examples),
            max_q,
        )
    return examples[:max_q]


def rng_seed_for_question(qid: str) -> int:
    """Reproducible int seed mixing SAMPLE_SEED (42) and question id."""
    return (SAMPLE_SEED * 1_000_003 + (zlib.crc32(str(qid).encode()) & 0xFFFFFFFF)) % (2**31)


def _get_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def encode_l2norm(texts: List[str]) -> np.ndarray:
    m = _get_st()
    e = m.encode(texts, convert_to_numpy=True)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (e / n).astype(np.float64)


def greedy_jaccard_mean_first(pool: List[str], k: int) -> List[str]:
    n = len(pool)
    if k >= n:
        return list(pool)
    mat = compute_jaccard_distance_matrix(pool)
    mean_d = []
    for i in range(n):
        s = sum(mat[i, j] for j in range(n) if j != i)
        mean_d.append(s / max(n - 1, 1))
    first = int(np.argmax(mean_d))
    selected = [first]
    rem = set(range(n)) - {first}
    while len(selected) < k and rem:
        best_i = max(
            rem,
            key=lambda i: min(mat[i, j] for j in selected),
        )
        selected.append(best_i)
        rem.remove(best_i)
    return [pool[i] for i in selected]


def random_from_pool(pool: List[str], k: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in idx[:k]]


def mmr_jaccard_select(pool: List[str], question: str, k: int, lam: float) -> List[str]:
    """MMR with Jaccard sim to question and between pool lines."""
    n = len(pool)
    if k >= n:
        return list(pool)
    rel = [jaccard_similarity(p, question) for p in pool]
    first = int(np.argmax(rel))
    selected = [first]
    rem = set(range(n)) - {first}
    while len(selected) < k and rem:
        best_s = -1e9
        best_i = -1
        for i in rem:
            max_sim = max(jaccard_similarity(pool[i], pool[j]) for j in selected)
            score = lam * rel[i] - (1.0 - lam) * max_sim
            if score > best_s:
                best_s = score
                best_i = i
        selected.append(best_i)
        rem.remove(best_i)
    return [pool[i] for i in selected]


def compute_qpd(seeds: List[str]) -> float:
    if len(seeds) < 2:
        return 0.0
    emb = encode_l2norm(seeds)
    sim = np.dot(emb, emb.T)
    k = len(seeds)
    s = 0.0
    c = 0
    for i in range(k):
        for j in range(i + 1, k):
            s += 1.0 - sim[i, j]
            c += 1
    return float(s / c) if c else 0.0


def compute_itc_thread(queries: List[str]) -> float:
    """ITC from list of search queries q_1..q_T (only search turns)."""
    try:
        import editdistance
    except ImportError:
        raise ImportError("pip install editdistance for ITC")
    if len(queries) < 2:
        return 0.0
    q1 = queries[0]
    t1 = q1.lower().split()
    if not t1:
        return 0.0
    acc = 0.0
    for tau in range(1, len(queries)):
        qa = queries[tau].lower().split()
        if not qa:
            continue
        d = editdistance.eval(t1, qa)
        mx = max(len(t1), len(qa))
        sim = 1.0 - (d / mx) if mx else 0.0
        if sim > 0.8:
            acc += 1.0
    return acc / (len(queries) - 1)


def turn_logs_to_queries(turn_logs: List[dict]) -> List[str]:
    qs = []
    for log in turn_logs:
        if log.get("query"):
            qs.append(str(log["query"]))
    return qs


def thread_summary_from_logs(turn_logs: List[dict]) -> str:
    parts = []
    for log in turn_logs:
        if log.get("query"):
            snip = (log.get("search_result") or "")[:800]
            parts.append(f"Query: {log['query']}\nExcerpt: {snip}")
        if log.get("answer"):
            parts.append(f"Thread conclusion: {log.get('answer')}")
    return "\n".join(parts) if parts else "(no search trace)"


async def llm_judge_correct(
    model: str, question: str, gold: str, pred: str, log_path: Optional[Path]
) -> bool:
    prompt = f"""You are a strict evaluator. Decide if the predicted answer matches the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

CORRECT only if the prediction matches the gold on all facts that matter for this question: same required entities, numbers, counts, yes/no, and conclusions. Treat as equivalent ONLY:
- Punctuation, capitalization, extra/missing articles (a/the), whitespace
- Equivalent date formats (e.g. "March 3 1990" vs "1990-03-03")
- Equivalent units or rounding at the same precision (e.g. "5 km" vs "5 kilometers", "3" vs "3.0" when the gold is an integer count)

INCORRECT if any of these differ from gold: wrong name/entity, wrong number, wrong quantity, missing a required part of the gold answer, adding a conflicting claim, or paraphrase that is not strictly the same fact (e.g. different movie title, different year, "yes" vs "no").

When unsure, prefer INCORRECT.

Reply with only: CORRECT or INCORRECT"""
    _log_api("LLM judge correct", log_path)
    text, _, _ = await call_llm(
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=16,
        temperature=0.0,
    )
    return "CORRECT" in (text or "").upper() and "INCORRECT" not in (text or "").upper()


async def llm_judge_pair_same(
    model: str, question: str, a: str, b: str, log_path: Optional[Path]
) -> bool:
    prompt = f"""Given two answers to the same question, determine whether they are semantically equivalent or genuinely different.

Question: {question}
Answer A: {a}
Answer B: {b}

Are these two answers saying the same thing?
Reply with only: SAME or DIFFERENT

SAME: both answers make the same factual claim. Minor wording differences, abbreviations, or formatting don't matter.
DIFFERENT: the answers make different factual claims, name different entities, or give different values."""
    _log_api("LLM judge APD pair", log_path)
    text, _, _ = await call_llm(
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=16,
        temperature=0.0,
    )
    return "SAME" in (text or "").upper()


async def synthesize_answer(
    model: str,
    question: str,
    thread_summaries: List[str],
    thread_answers: List[str],
    log_path: Optional[Path],
) -> str:
    blocks = []
    for i, (summ, ans) in enumerate(zip(thread_summaries, thread_answers)):
        blocks.append(
            f"--- Thread {i} ---\nEvidence:\n{summ[:SYNTH_MAX // 4]}\nThread final answer: {ans}"
        )
    user = f"""You are a research coordinator. Synthesize one best final answer from independent web-search threads.

Question: {question}

{chr(10).join(blocks)}

Merge evidence, resolve conflicts, output the single best concise answer inside <answer>...</answer> tags."""
    _log_api("synthesis", log_path)
    text, _, _ = await call_llm(
        [{"role": "user", "content": user}],
        model=model,
        max_tokens=1024,
        temperature=0.3,
    )
    from webwalkerqa.methods.diversity_scaling import _extract_tag

    ans = _extract_tag(text or "", "answer")
    if ans:
        return ans.strip()
    return (text or "").strip()[:2000]


async def run_one_question(
    ex: QAExample,
    condition: str,
    model: str,
    log_path: Optional[Path],
) -> Dict[str, Any]:
    qid = str(ex.id)
    question = ex.question
    gold = str(ex.answer)

    pool: List[str] = []
    selected_seeds: List[str] = []
    threads_out: List[Dict[str, Any]] = []
    base_seed = rng_seed_for_question(qid)

    try:
        cfg = condition_config(condition)

        if cfg["mode"] == "sequential":
            r = await run_single_rollout(
                model=model,
                question=question,
                answer_gt=gold,
                max_turns=T_SEQUENTIAL,
                initial_query=None,
                rollout_seed=base_seed,
                question_id=qid,
                react_temp_first=0.7,
                react_temp_rest=0.7,
            )
            qs = turn_logs_to_queries(r["turn_logs"])
            itc = compute_itc_thread(qs) if len(qs) >= 2 else 0.0
            syn = r["answer"]
            em1 = 1 if em_match(syn, gold) else 0
            llm1 = em1
            if not em1:
                llm1 = 1 if await llm_judge_correct(model, question, gold, syn, log_path) else 0
            threads_out = [
                {
                    "thread_id": 0,
                    "seed": qs[0] if qs else "",
                    "turns": _turns_for_json(r["turn_logs"]),
                    "thread_answer": r["answer"],
                    "thread_em": 1 if em_match(r["answer"], gold) else 0,
                    "thread_llm": 0,
                    "itc": itc,
                }
            ]
            if not threads_out[0]["thread_em"]:
                threads_out[0]["thread_llm"] = (
                    1 if await llm_judge_correct(model, question, gold, r["answer"], log_path) else 0
                )
            else:
                threads_out[0]["thread_llm"] = 1
            return {
                "question_id": qid,
                "question": question[:500] + ("..." if len(question) > 500 else ""),
                "gold_answer": gold,
                "pool": [],
                "selected_seeds": [],
                "thread_trajectories": threads_out,
                "synthesis_answer": syn,
                "k_threads": 1,
                "pass1_em": em1,
                "pass1_llm": llm1,
                "pass_oracle_em": em1,
                "pass_oracle_llm": llm1,
                "pass4_em": em1,
                "pass4_llm": llm1,
                "qpd": 0.0,
                "apd": 0.0,
                "apd_pairs": [],
                "itc_per_thread": [itc],
                "itc_mean": itc,
            }

        k = cfg["k"]
        t = cfg["t_parallel"]
        mode = cfg["mode"]
        inject: Optional[List[str]] = None

        if mode == "naive":
            selected_seeds = []
            pool = []
        else:
            o = cfg["o"]
            pool, _, _ = await generate_pool(model, question, o)
            if mode == "greedy_jaccard":
                inject = greedy_jaccard_mean_first(pool, k)
            elif mode == "random":
                inject = random_from_pool(pool, k, seed=base_seed)
            else:
                inject = mmr_jaccard_select(pool, question, k, float(cfg["lam"]))
            selected_seeds = list(inject)

        tasks = []
        for i in range(k):
            init_q = inject[i] if inject else None
            tasks.append(
                run_single_rollout(
                    model=model,
                    question=question,
                    answer_gt=gold,
                    max_turns=t,
                    initial_query=init_q,
                    rollout_seed=base_seed + i * 1000,
                    question_id=qid,
                    react_temp_first=1.0 if init_q is None else 0.7,
                    react_temp_rest=0.7,
                )
            )
        results = await asyncio.gather(*tasks)

        for i, r in enumerate(results):
            qs = turn_logs_to_queries(r["turn_logs"])
            if inject is None:
                seed_q = qs[0] if qs else ""
                selected_seeds.append(seed_q)
            else:
                seed_q = inject[i]
            itc = compute_itc_thread(qs) if len(qs) >= 2 else 0.0
            tem = 1 if em_match(r["answer"], gold) else 0
            threads_out.append(
                {
                    "thread_id": i,
                    "seed": seed_q,
                    "turns": _turns_for_json(r["turn_logs"]),
                    "thread_answer": r["answer"],
                    "thread_em": tem,
                    "thread_llm": 0,
                    "itc": itc,
                }
            )

        for trow in threads_out:
            if trow["thread_em"]:
                trow["thread_llm"] = 1
            else:
                trow["thread_llm"] = (
                    1
                    if await llm_judge_correct(model, question, gold, trow["thread_answer"], log_path)
                    else 0
                )

        pass_oracle_em = 1 if any(t["thread_em"] for t in threads_out) else 0
        pass_oracle_llm = 1 if any(t["thread_llm"] for t in threads_out) else 0

        summaries = [
            thread_summary_from_logs(list(results[i]["turn_logs"])) for i in range(k)
        ]
        anss = [t["thread_answer"] for t in threads_out]
        synthesis_answer = await synthesize_answer(model, question, summaries, anss, log_path)

        pass1_em = 1 if em_match(synthesis_answer, gold) else 0
        pass1_llm = pass1_em
        if not pass1_em:
            pass1_llm = (
                1
                if await llm_judge_correct(model, question, gold, synthesis_answer, log_path)
                else 0
            )

        qpd = compute_qpd(selected_seeds) if len(selected_seeds) >= 2 else 0.0

        answers = [t["thread_answer"] for t in threads_out]
        apd_pairs = []
        apd_acc = 0
        apd_n = 0
        for i in range(k):
            for j in range(i + 1, k):
                same = await llm_judge_pair_same(model, question, answers[i], answers[j], log_path)
                diff = 0 if same else 1
                apd_acc += diff
                apd_n += 1
                apd_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "answer_a": answers[i][:300],
                        "answer_b": answers[j][:300],
                        "judgment": "SAME" if same else "DIFFERENT",
                    }
                )
        apd = apd_acc / apd_n if apd_n else 0.0
        itc_mean = float(np.mean([t["itc"] for t in threads_out]))

        return {
            "question_id": qid,
            "question": question[:500] + ("..." if len(question) > 500 else ""),
            "gold_answer": gold,
            "pool": pool,
            "selected_seeds": selected_seeds,
            "thread_trajectories": threads_out,
            "synthesis_answer": synthesis_answer,
            "k_threads": k,
            "pass1_em": pass1_em,
            "pass1_llm": pass1_llm,
            "pass_oracle_em": pass_oracle_em,
            "pass_oracle_llm": pass_oracle_llm,
            "pass4_em": pass_oracle_em,
            "pass4_llm": pass_oracle_llm,
            "qpd": qpd,
            "apd": apd,
            "apd_pairs": apd_pairs,
            "itc_per_thread": [t["itc"] for t in threads_out],
            "itc_mean": itc_mean,
        }
    except Exception as e:
        logger.exception("question failed %s", qid)
        try:
            kf = condition_config(condition)["k"]
        except ValueError:
            kf = 0
        return {
            "question_id": qid,
            "question": question[:200],
            "gold_answer": gold,
            "error": str(e),
            "pool": pool,
            "selected_seeds": selected_seeds,
            "thread_trajectories": threads_out,
            "synthesis_answer": "",
            "k_threads": kf if condition != "sequential" else 1,
            "pass1_em": 0,
            "pass1_llm": 0,
            "pass_oracle_em": 0,
            "pass_oracle_llm": 0,
            "pass4_em": 0,
            "pass4_llm": 0,
            "qpd": 0.0,
            "apd": 0.0,
            "apd_pairs": [],
            "itc_per_thread": [],
            "itc_mean": 0.0,
        }


def _turns_for_json(turn_logs: List[dict]) -> List[dict]:
    out = []
    t = 0
    for log in turn_logs:
        if log.get("query"):
            t += 1
            snip = (log.get("search_result") or "")[:RESULT_SNIPPET_LEN]
            out.append({"turn": t, "query": log["query"], "result_snippet": snip})
    return out


def aggregate(results: List[dict]) -> Dict[str, float]:
    def mean(key: str) -> float:
        vals = [r.get(key, 0) for r in results if "error" not in r]
        return float(np.mean(vals)) if vals else 0.0

    out = {
        "pass1_em_mean": mean("pass1_em"),
        "pass1_llm_mean": mean("pass1_llm"),
        "pass_oracle_em_mean": mean("pass_oracle_em"),
        "pass_oracle_llm_mean": mean("pass_oracle_llm"),
        "pass4_em_mean": mean("pass4_em"),
        "pass4_llm_mean": mean("pass4_llm"),
        "qpd_mean": mean("qpd"),
        "apd_mean": mean("apd"),
        "itc_mean": mean("itc_mean"),
    }
    return out


def _progress(it, desc: str, unit: str, disable: bool, *, leave: bool = True):
    if disable:
        return it
    try:
        from tqdm import tqdm

        return tqdm(
            it,
            desc=desc,
            unit=unit,
            total=len(it) if hasattr(it, "__len__") and not isinstance(it, range) else None,
            dynamic_ncols=True,
            mininterval=0.3,
            leave=leave,
        )
    except ImportError:
        logger.warning("pip install tqdm for a progress bar")
        return it


def _progress_total(total: int, desc: str, unit: str, disable: bool):
    if disable:
        return None
    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.3,
        )
    except ImportError:
        return None


async def run_condition_benchmark(
    condition: str,
    benchmark: str,
    examples: List[QAExample],
    out_dir: Path,
    model: str,
    *,
    show_progress: bool = True,
    tqdm_leave: bool = True,
) -> None:
    log_path = out_dir / "api_log.txt"
    results = []
    bench_key = "gaia25" if benchmark == "gaia25" else "hotpotqa25"
    bar_desc = f"{condition} | {bench_key}"
    pbar = _progress(
        examples,
        desc=bar_desc,
        unit="q",
        disable=not show_progress,
        leave=tqdm_leave,
    )
    for ex in pbar:
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix_str(str(ex.id)[:36], refresh=False)
        elif not show_progress:
            logger.info("%s %s %s", condition, benchmark, ex.id)
        row = await run_one_question(ex, condition, model, log_path)
        results.append(row)
    agg = aggregate(results)
    try:
        k_meta = condition_config(condition)["k"]
        if condition == "sequential":
            k_meta = 1
    except ValueError:
        k_meta = None
    payload = {
        "condition": condition,
        "benchmark": bench_key,
        "k_threads": k_meta,
        "n_questions": len(examples),
        "results": results,
        "aggregate": agg,
    }
    out_path = out_dir / f"{condition}_{bench_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out_path)
    _append_summary_csv(out_dir, condition, bench_key, agg)


def _k_for_condition(condition: str) -> Optional[int]:
    try:
        c = condition_config(condition)
        return c["k"]
    except ValueError:
        return None


def _append_summary_csv(
    out_dir: Path, condition: str, benchmark: str, agg: Dict[str, float]
) -> None:
    path = out_dir / "summary.csv"
    kv = _k_for_condition(condition)
    row = {
        "condition": condition,
        "benchmark": benchmark,
        "k": kv if kv is not None else "",
        "pass1_em": agg["pass1_em_mean"],
        "pass1_llm": agg["pass1_llm_mean"],
        "pass_oracle_em": agg["pass_oracle_em_mean"],
        "pass_oracle_llm": agg["pass_oracle_llm_mean"],
        "pass4_em": agg["pass4_em_mean"],
        "pass4_llm": agg["pass4_llm_mean"],
        "qpd_mean": agg["qpd_mean"],
        "apd_mean": agg["apd_mean"],
        "itc_mean": agg["itc_mean"],
    }
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None, help="One of 6 conditions")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["gaia25", "hotpotqa25", "both"],
        default="gaia25",
    )
    parser.add_argument("--run-all", action="store_true", help="Run full order for both benchmarks")
    parser.add_argument(
        "--gaia-path",
        type=str,
        default=str(DEFAULT_GAIA_PATH),
        help=f"Default: {DEFAULT_GAIA_PATH}",
    )
    parser.add_argument(
        "--hotpot-path",
        type=str,
        default=str(DEFAULT_HOTPOT_PATH),
        help=f"Default: {DEFAULT_HOTPOT_PATH}",
    )
    parser.add_argument("--out-dir", type=str, default=str(_GA_DIR / "results" / "diversity_parallel"))
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable progress bars (plain logging only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = normalize_model(args.model)
    out_dir = Path(args.out_dir)

    gaia_ex = load_gaia25(Path(args.gaia_path))
    hotpot_ex = load_hotpot25(Path(args.hotpot_path))

    order = CONDITIONS_ORDER if args.run_all else ([args.condition] if args.condition else [])
    if not order:
        print("Specify --condition or --run-all")
        sys.exit(1)

    benchmarks = (
        [("gaia25", gaia_ex), ("hotpotqa25", hotpot_ex)]
        if args.benchmark == "both" or args.run_all
        else [("gaia25" if args.benchmark == "gaia25" else "hotpotqa25", gaia_ex if args.benchmark == "gaia25" else hotpot_ex)]
    )

    show_p = not args.no_tqdm

    if args.run_all:
        benchmarks = [("gaia25", gaia_ex), ("hotpotqa25", hotpot_ex)]
        n_jobs = len(CONDITIONS_ORDER) * len(benchmarks)
        outer_pbar = _progress_total(
            n_jobs, desc="run-all", unit="json", disable=not show_p
        )
        for cond in CONDITIONS_ORDER:
            for bname, exs in benchmarks:
                if outer_pbar is not None:
                    outer_pbar.set_postfix_str(f"{cond} {bname}", refresh=True)
                try:
                    await run_condition_benchmark(
                        cond,
                        bname,
                        exs,
                        out_dir,
                        model,
                        show_progress=show_p,
                        tqdm_leave=False,
                    )
                except Exception as e:
                    logger.error("Failed %s %s: %s", cond, bname, e)
                if outer_pbar is not None:
                    outer_pbar.update(1)
        if outer_pbar is not None:
            outer_pbar.close()
    else:
        for cond in order:
            for bname, exs in benchmarks:
                await run_condition_benchmark(
                    cond,
                    bname,
                    exs,
                    out_dir,
                    model,
                    show_progress=show_p,
                    tqdm_leave=True,
                )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
