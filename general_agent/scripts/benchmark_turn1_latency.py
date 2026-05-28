"""
Wall-clock timing benchmark: Standard (4 independent LLM calls) vs DivInit (1 pool call + CPU selection).

For each model, runs 20 questions and measures:
  - Standard: 4 parallel ReAct turn-1 calls (each generates 1 query)
  - DivInit:  1 pool-generation call (generates 16 candidates) + CPU greedy-Jaccard selection of 4

Outputs a CSV with per-question timings and a summary table.

Usage (on compute node with vLLM running):
  python scripts/benchmark_turn1_latency.py --model openai/Qwen/Qwen3-8B --api-base http://localhost:8003/v1
  python scripts/benchmark_turn1_latency.py --model openai/Qwen/Qwen3-4B --api-base http://localhost:8002/v1
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from webwalkerqa.llm import call_llm, set_api_base  # noqa: E402
from webwalkerqa.methods.diversity_scaling import (  # noqa: E402
    generate_pool,
    REACT_PROMPT,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "main_table"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper_assets" / "timing"


async def time_standard_turn1(model: str, question: str, k: int = 4, max_tokens: int = 8192) -> dict:
    """Time k independent ReAct turn-1 calls (parallel, like naive_parallel)."""
    prompt = REACT_PROMPT.format(
        max_turns=8, turn=1, question=question, history="(none yet)"
    )

    async def single_call(seed: int):
        t0 = time.perf_counter()
        text, p_tok, o_tok = await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=1.0,
        )
        elapsed = time.perf_counter() - t0
        return {"elapsed": elapsed, "prompt_tokens": p_tok, "completion_tokens": o_tok}

    t0_total = time.perf_counter()
    results = await asyncio.gather(*[single_call(i) for i in range(k)])
    wall_total = time.perf_counter() - t0_total

    return {
        "method": "standard",
        "k": k,
        "wall_total_s": wall_total,
        "max_single_s": max(r["elapsed"] for r in results),
        "mean_single_s": sum(r["elapsed"] for r in results) / k,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
    }


async def time_divinit_turn1(model: str, question: str, pool_size: int = 16, k: int = 4) -> dict:
    """Time 1 pool-generation call + CPU greedy-Jaccard selection."""
    t0 = time.perf_counter()
    pool, p_tok, o_tok = await generate_pool(model, question, pool_size)
    llm_elapsed = time.perf_counter() - t0

    # CPU selection (greedy max-min Jaccard)
    t_cpu = time.perf_counter()
    from webwalkerqa.methods.diversity_scaling import _jaccard_sim_tokens
    selected = [pool[0]] if pool else []
    remaining = list(range(1, len(pool)))
    for _ in range(min(k - 1, len(remaining))):
        best_idx = -1
        best_min_dist = -1.0
        for idx in remaining:
            min_dist = min(
                1.0 - _jaccard_sim_tokens(pool[idx], s) for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx >= 0:
            selected.append(pool[best_idx])
            remaining.remove(best_idx)
    cpu_elapsed = time.perf_counter() - t_cpu

    return {
        "method": "divinit",
        "k": k,
        "pool_size": pool_size,
        "wall_total_s": llm_elapsed + cpu_elapsed,
        "llm_s": llm_elapsed,
        "cpu_s": cpu_elapsed,
        "prompt_tokens": p_tok,
        "completion_tokens": o_tok,
        "n_selected": len(selected),
    }


async def run_benchmark(model: str, dataset_path: str, n_questions: int = 20, k: int = 4, max_tokens: int = 8192):
    with open(dataset_path) as f:
        data = json.load(f)
    questions = [ex["question"] for ex in data[:n_questions]]

    model_short = model.split("/")[-1].lower().replace("-", "_")
    ds_name = Path(dataset_path).stem

    rows = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q[:60]}...")

        std = await time_standard_turn1(model, q, k=k, max_tokens=max_tokens)
        div = await time_divinit_turn1(model, q, pool_size=16, k=k)

        rows.append({
            "question_idx": i,
            "question": q[:80],
            "std_wall_s": f"{std['wall_total_s']:.3f}",
            "std_max_single_s": f"{std['max_single_s']:.3f}",
            "std_prompt_tok": std["total_prompt_tokens"],
            "std_completion_tok": std["total_completion_tokens"],
            "div_wall_s": f"{div['wall_total_s']:.3f}",
            "div_llm_s": f"{div['llm_s']:.3f}",
            "div_cpu_s": f"{div['cpu_s']:.3f}",
            "div_prompt_tok": div["prompt_tokens"],
            "div_completion_tok": div["completion_tokens"],
            "speedup": f"{std['wall_total_s'] / max(div['wall_total_s'], 0.001):.2f}x",
        })

    # Write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"timing_{model_short}_{ds_name}.csv"
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Summary
    std_times = [float(r["std_wall_s"]) for r in rows]
    div_times = [float(r["div_wall_s"]) for r in rows]
    div_cpu = [float(r["div_cpu_s"]) for r in rows]
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Model: {model}  |  Dataset: {ds_name}  |  N={len(rows)}")
    print(f"{'='*60}")
    print(f"Standard (k={k} parallel ReAct calls):")
    print(f"  mean wall = {np.mean(std_times):.3f}s  ±{np.std(std_times):.3f}s")
    print(f"DivInit (1 pool call + CPU selection):")
    print(f"  mean wall = {np.mean(div_times):.3f}s  ±{np.std(div_times):.3f}s")
    print(f"  mean CPU  = {np.mean(div_cpu)*1000:.2f}ms")
    print(f"Speedup: {np.mean(std_times)/np.mean(div_times):.2f}x")
    print(f"\nWrote: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. openai/Qwen/Qwen3-8B")
    parser.add_argument("--api-base", required=True, help="e.g. http://localhost:8003/v1")
    parser.add_argument("--dataset", default=str(DATA_DIR / "GAIA.json"),
                        help="Dataset JSON (default: GAIA)")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens for standard ReAct calls (pool call always uses 2048)")
    args = parser.parse_args()

    os.environ["OPENAI_API_BASE"] = args.api_base
    set_api_base(args.api_base)

    asyncio.run(run_benchmark(args.model, args.dataset, args.n_questions, args.k, args.max_tokens))


if __name__ == "__main__":
    main()
