"""
Generate POOL of o=16 queries from gpt-4o-mini at multiple temperatures, then
pick k=4 via greedy max-min Jaccard selection, compute QPD over the selection.

Purpose: show that DIFFUSE's QPD is insensitive to temperature — the selector
enforces diversity even when the underlying pool is concentrated.

Cost: ~$0.20 total (103 q × 6 temps × 1 pool-gen call, ~600 calls).
Run:
  OPENAI_API_KEY=sk-... python paper_assets/scripts/qpd_temperature_sweep_diffuse.py
"""
from __future__ import annotations
import asyncio
import csv
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

TEMPS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
POOL_SIZE = 16
K_SELECT = 4
MODEL = "openai/gpt-4o-mini"
CONCURRENCY = 50

GAIA_PATH = Path('/home/ssmurali/t3-testbed/general_agent/data/main_table/gaia_full.json')
OUT_CSV   = Path('/home/ssmurali/t3-testbed/paper_assets/analysis_threadid/qpd_temperature_sweep_diffuse.csv')

# POOL_GEN_PROMPT — verbatim from methods/diversity_scaling.py
POOL_PROMPT = """\
Generate exactly {o} diverse search queries to investigate this question.
Each query should approach the question from a different angle, specifically targeting different constraints or components of the question.

Question: {question}

Output exactly {o} queries, one per line, numbered 1-{o}. No other text."""

# Strip stopwords for Jaccard (matches Fig 2 canonical stripping)
STOPWORDS = set("""a about above after again against all am an and any are aren as at be because been before being below
between both but by could couldn did didn do does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is isn it its itself let lets me might more
most must my myself need no nor not now of off on once only or other ought our ours ourselves out over own same shall she
should shouldn so some such than that thats the their theirs them themselves then there theres these they this those
through to too under until up very was wasn we were weren what whats when where which while who whom why will with would
wouldn you your yours yourself yourselves""".split())

LINE_RE = re.compile(r'^\s*\d+[\.\)]\s*(.+?)\s*$', re.MULTILINE)


def tokens(s: str) -> set:
    return {w for w in s.lower().split() if w and w not in STOPWORDS}


def jaccard_dist(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb: return 0.0
    return 1.0 - (len(ta & tb) / max(1, len(ta | tb)))


def qpd(queries: List[str]) -> float:
    if len(queries) < 2: return float('nan')
    return float(np.mean([jaccard_dist(a, b) for a, b in combinations(queries, 2)]))


def greedy_select(pool: List[str], k: int, seed: int = 42) -> List[str]:
    """Greedy max-min Jaccard farthest-first. First pick is random (seeded)."""
    if not pool: return []
    if k >= len(pool): return list(pool)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(pool)))]
    remaining = set(range(len(pool))) - set(selected)
    while len(selected) < k and remaining:
        best, best_score = None, -1.0
        for j in remaining:
            min_d = min(jaccard_dist(pool[j], pool[s]) for s in selected)
            if min_d > best_score:
                best_score, best = min_d, j
        if best is None: break
        selected.append(best); remaining.discard(best)
    return [pool[i] for i in selected]


async def gen_pool(litellm, sem, question: str, temp: float) -> List[str]:
    async with sem:
        try:
            resp = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": POOL_PROMPT.format(o=POOL_SIZE, question=question)}],
                temperature=temp,
                max_tokens=900,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [err] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            return []
        return [m.strip() for m in LINE_RE.findall(text)]


async def main():
    import litellm
    litellm.drop_params = True

    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: set OPENAI_API_KEY first", file=sys.stderr); sys.exit(2)

    questions = json.load(open(GAIA_PATH))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = []
    metadata = []
    for q in questions:
        for temp in TEMPS:
            tasks.append(gen_pool(litellm, sem, q['question'], temp))
            metadata.append((q['id'], temp))

    print(f"Dispatching {len(tasks)} pool-gen calls "
          f"({len(questions)} q × {len(TEMPS)} temps, pool={POOL_SIZE}), "
          f"concurrency={CONCURRENCY} ...")
    start = time.time()
    BATCH = 200
    results: List[List[str]] = []
    for i in range(0, len(tasks), BATCH):
        batch = await asyncio.gather(*tasks[i:i+BATCH])
        results.extend(batch)
        print(f"  done {min(i+BATCH, len(tasks))}/{len(tasks)} "
              f"({(time.time()-start)/60:.1f} min)")

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['question_id', 'temperature', 'pool_size_parsed',
                    'qpd_selected', 'q1', 'q2', 'q3', 'q4'])
        rows_by_temp = {t: [] for t in TEMPS}
        for (qid, temp), pool in zip(metadata, results):
            if len(pool) < K_SELECT:
                w.writerow([qid, f"{temp:.1f}", len(pool), "", "", "", "", ""])
                continue
            # Deterministic per (qid, temp) seed for reproducibility
            sel_seed = abs(hash((qid, temp))) % (2**32)
            picks = greedy_select(pool, K_SELECT, seed=sel_seed)
            qpd_val = qpd(picks)
            rows_by_temp[temp].append(qpd_val)
            row = [qid, f"{temp:.1f}", len(pool), f"{qpd_val:.6f}"]
            row += [p.replace("\n", " ").strip()[:200] for p in (picks + [""]*K_SELECT)[:K_SELECT]]
            w.writerow(row)
    print(f"\nWrote: {OUT_CSV}")

    print(f"\n{'temp':>6}  {'n_q':>5}  {'mean_QPD':>9}  {'median_QPD':>10}")
    for temp in TEMPS:
        v = rows_by_temp.get(temp, [])
        if v:
            print(f"  {temp:>4.1f}  {len(v):>5d}  {np.mean(v):>9.4f}  {np.median(v):>10.4f}")
        else:
            print(f"  {temp:>4.1f}  {0:>5d}  {'—':>9}  {'—':>10}")


if __name__ == "__main__":
    asyncio.run(main())
