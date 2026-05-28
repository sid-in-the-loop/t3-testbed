"""
Generate turn-1 queries from gpt-4o-mini at multiple temperatures, compute
QPD per question per temperature.

Purpose: show that high sampling temperature does NOT resolve naive-parallel
query concentration. We expect QPD to stay clustered near ~0.2 across all
temperatures.

Cost: ~$0.25 total (103 questions × 4 threads × 6 temperatures × ~50 output
tokens at gpt-4o-mini pricing). Runtime ~2 min with concurrency 50.

Run:
  OPENAI_API_KEY=sk-... python paper_assets/scripts/qpd_temperature_sweep.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

TEMPS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
K_THREADS = 4
MODEL = "openai/gpt-4o-mini"
CONCURRENCY = 50

GAIA_PATH = Path('/home/ssmurali/t3-testbed/general_agent/data/main_table/gaia_full.json')
OUT_CSV   = Path('/home/ssmurali/t3-testbed/paper_assets/analysis_threadid/qpd_temperature_sweep.csv')

# WEB_REASONING_PROMPT — verbatim from methods/diversity_scaling.py
PROMPT_TEMPLATE = """\
You are a research assistant with the ability to perform web searches to answer questions.
You can answer a question with many turns of search and reasoning.
Based on the history information, suggest the next action.

You will be provided with:
1. Your history search attempts: queries in <search> query </search> and results in <information>...</information>.
2. The question to answer.

IMPORTANT RULES:
1. Choose ONLY ONE action per response. Do NOT perform more than one action per step.
2. Follow the exact syntax for the selected action.
3. **Do not do duplicate searches.** Pay attention to the history search results.

Valid actions:
1. <search> query </search> — search the web if you lack some knowledge.
2. <answer> answer </answer> — output the final answer. Short and concise. No justification.
3. <summary> important parts of the history </summary> — compress the history.

Format:
<think> your thinking process </think>
[one of <search>...</search>, <summary>...</summary>, <answer>...</answer>]

Question: {question}

History Turns:
(empty, this is the first turn)"""

SEARCH_RE = re.compile(r'<search>(.*?)</search>', re.DOTALL | re.IGNORECASE)

# stopwords (minimal) — match the canonical Fig 2 stopword stripping
STOPWORDS = set("""a about above after again against all am an and any are aren as at be because been before being below
between both but by could couldn did didn do does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is isn it its itself let lets me might more
most must my myself need no nor not now of off on once only or other ought our ours ourselves out over own same shall she
should shouldn so some such than that thats the their theirs them themselves then there theres these they this those
through to too under until up very was wasn we were weren what whats when where which while who whom why will with would
wouldn you your yours yourself yourselves""".split())


def jaccard_dist(a: str, b: str) -> float:
    ta = {w for w in a.lower().split() if w and w not in STOPWORDS}
    tb = {w for w in b.lower().split() if w and w not in STOPWORDS}
    if not ta and not tb: return 0.0
    return 1.0 - (len(ta & tb) / max(1, len(ta | tb)))


def qpd(queries: List[str]) -> float:
    """Mean pairwise Jaccard distance over all C(k,2) pairs of queries."""
    if len(queries) < 2: return float('nan')
    ds = [jaccard_dist(a, b) for a, b in combinations(queries, 2)]
    return float(np.mean(ds))


async def one_call(litellm, sem, question: str, temp: float) -> str:
    async with sem:
        try:
            resp = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
                temperature=temp,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [err] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            return ""
        m = SEARCH_RE.search(text)
        return m.group(1).strip() if m else ""


async def main():
    import litellm
    litellm.drop_params = True

    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: set OPENAI_API_KEY first", file=sys.stderr); sys.exit(2)

    questions = json.load(open(GAIA_PATH))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)

    # Tasks: for each (qid, temp), K_THREADS parallel calls
    tasks = []
    metadata = []
    for q in questions:
        qid = q['id']
        question_text = q['question']
        for temp in TEMPS:
            for thread_idx in range(K_THREADS):
                tasks.append(one_call(litellm, sem, question_text, temp))
                metadata.append((qid, temp, thread_idx))

    print(f"Dispatching {len(tasks)} calls "
          f"({len(questions)} q × {len(TEMPS)} temps × {K_THREADS} threads), "
          f"concurrency={CONCURRENCY} ...")

    # Run with simple progress prints every 200
    import time
    start = time.time()
    BATCH = 200
    results: List[str] = []
    for i in range(0, len(tasks), BATCH):
        batch = await asyncio.gather(*tasks[i:i+BATCH])
        results.extend(batch)
        print(f"  done {min(i+BATCH, len(tasks))}/{len(tasks)} "
              f"({(time.time()-start)/60:.1f} min)")

    # Re-group by (qid, temp) → 4 queries
    by_cell = {}
    for (qid, temp, thread_idx), q in zip(metadata, results):
        by_cell.setdefault((qid, temp), []).append(q)

    # Compute QPD per cell and save
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['question_id', 'temperature', 'k_threads', 'n_non_empty',
                    'qpd_jaccard', 'q1', 'q2', 'q3', 'q4'])
        for (qid, temp), qs in sorted(by_cell.items()):
            non_empty = [x for x in qs if x]
            qpd_val = qpd(non_empty) if len(non_empty) >= 2 else float('nan')
            row = [qid, f"{temp:.1f}", len(qs), len(non_empty),
                   f"{qpd_val:.6f}" if not np.isnan(qpd_val) else ""]
            row += [(q or "").replace("\n"," ").strip()[:200] for q in (list(qs)+[""]*4)[:4]]
            w.writerow(row)
    print(f"\nWrote: {OUT_CSV}")

    # Summary
    print(f"\n{'temp':>6}  {'n_q':>5}  {'mean_QPD':>9}  {'median_QPD':>10}")
    for temp in TEMPS:
        vals = []
        for (qid, t), qs in by_cell.items():
            if abs(t - temp) > 1e-9: continue
            non_empty = [x for x in qs if x]
            if len(non_empty) >= 2:
                vals.append(qpd(non_empty))
        if vals:
            print(f"  {temp:>4.1f}  {len(vals):>5d}  {np.mean(vals):>9.4f}  {np.median(vals):>10.4f}")


if __name__ == "__main__":
    asyncio.run(main())
