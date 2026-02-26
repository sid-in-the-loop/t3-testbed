"""
Per-question runner for WebWalkerQA experiments.

Handles:
  - Running a single method on a single question (with error handling)
  - Saving per-question JSON logs
  - Oracle (pass@8): runs 8 independent s1 trajectories, picks best
  - Resume: skips already-completed questions
"""

import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Optional

from .configs import ExperimentConfig
from .dataset import QAExample
from .eval import exact_match, pass_at_k, f1_score
from .methods.base import MethodResult
from .methods import get_method
from .methods.s1 import S1Method


async def run_question(
    example: QAExample,
    config: ExperimentConfig,
    model: str,
    output_dir: Path,
    verbose: bool = False,
    pbar: Optional[any] = None,
) -> MethodResult:
    """
    Run a method on one question and save the result to output_dir.

    Args:
        example: QAExample with id, question, answer.
        config: ExperimentConfig defining method, k, n, t.
        model: LiteLLM model string.
        output_dir: Directory to write per-question JSON.
        verbose: Print debug output.

    Returns:
        MethodResult with final answer and all logs.
    """
    output_file = output_dir / f"q_{example.id}.json"

    # Resume: skip if already done
    if output_file.exists():
        try:
            with open(output_file) as f:
                data = json.load(f)
            if verbose:
                print(f"  [skip] {example.id} already done (em={data.get('em', '?')})")
            # Reconstruct a minimal MethodResult for stats
            result = MethodResult(
                question_id=example.id,
                question=example.question,
                answer_gt=str(example.answer),
                final_answer=data.get("final_answer", ""),
                em=data.get("em", False),
                f1=data.get("f1", 0.0) or f1_score(data.get("final_answer", ""), str(example.answer)),
                turns_used=data.get("turns_used", 0),
                search_calls_used=data.get("search_calls_used", 0),
                total_prompt_tokens=data.get("total_prompt_tokens", 0),
                total_output_tokens=data.get("total_output_tokens", 0),
                method=data.get("method", config.method),
                config_id=data.get("config_id", config.id),
            )
            return result
        except Exception:
            pass  # Corrupted file — re-run

    if config.method == "oracle":
        result = await _run_oracle(example, config, model, output_dir, verbose, pbar)
    else:
        method_cls = get_method(config.method)
        method = method_cls(model=model, config=config, verbose=verbose)
        try:
            result = await method.run_question(
                question_id=example.id,
                question=example.question,
                answer_gt=str(example.answer),
                pbar=pbar,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] question {example.id}: {e}")
            result = MethodResult(
                question_id=example.id,
                question=example.question,
                answer_gt=str(example.answer),
                final_answer="",
                em=False,
                method=config.method,
                config_id=config.id,
                error=f"{e}\n{tb}",
            )

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    if verbose:
        status = "✓" if result.em else "✗"
        print(f"  [{status}] {example.id}: '{result.final_answer[:50]}' | "
              f"turns={result.turns_used} searches={result.search_calls_used}")

    return result


async def _run_oracle(
    example: QAExample,
    config: ExperimentConfig,
    model: str,
    output_dir: Path,
    verbose: bool,
    pbar: Optional[any] = None,
) -> MethodResult:
    """
    Oracle: run 8 independent s1 trajectories, pick the one with best EM.
    If none matches, return the first trajectory's answer.
    """
    from .configs import ExperimentConfig as EC
    oracle_s1_config = EC(
        id="Oracle_s1",
        method="s1",
        k=1, n=config.n, t=config.t,
        group="Oracle",
        description="Oracle s1 run",
    )

    n_runs = 8
    trajectories = []

    # Run 8 independent s1 passes
    tasks = []
    for run_idx in range(n_runs):
        method = S1Method(model=model, config=oracle_s1_config, verbose=False)
        tasks.append(method.run_question(
            question_id=f"{example.id}_oracle_{run_idx}",
            question=example.question,
            answer_gt=str(example.answer),
            pbar=None,  # Don't show nested bars for oracle internal runs
        ))

    if pbar:
        pbar.set_description(f"Q {example.id}: Oracle (0/{n_runs})")

    try:
        if pbar:
            trajectories = []
            for coro in asyncio.as_completed(tasks):
                res = await coro
                trajectories.append(res)
                pbar.set_description(f"Q {example.id}: Oracle ({len(trajectories)}/{n_runs})")
                pbar.update(1)
        else:
            trajectories = await asyncio.gather(*tasks)
    except Exception as e:
        trajectories = []
        print(f"  [ERROR] oracle {example.id}: {e}")

    # Oracle select: pick best answer (any EM match)
    answers = [r.final_answer for r in trajectories]
    oracle_em = pass_at_k(answers, example.answer)

    # Pick best trajectory (first EM match, or first)
    best = next((r for r in trajectories if r.em), trajectories[0] if trajectories else None)
    
    # Best F1 among all trajectories
    max_f1 = max(r.f1 for r in trajectories) if trajectories else 0.0

    if best is None:
        return MethodResult(
            question_id=example.id,
            question=example.question,
            answer_gt=str(example.answer),
            final_answer="",
            em=False,
            f1=0.0,
            method="oracle",
            config_id=config.id,
            error="No trajectories completed",
        )

    # Build aggregated result
    result = MethodResult(
        question_id=example.id,
        question=example.question,
        answer_gt=str(example.answer),
        final_answer=best.final_answer,
        em=oracle_em,  # Oracle EM (did any trajectory match?)
        f1=max_f1,     # Oracle F1
        turns_used=best.turns_used,
        search_calls_used=sum(r.search_calls_used for r in trajectories),
        total_prompt_tokens=sum(r.total_prompt_tokens for r in trajectories),
        total_output_tokens=sum(r.total_output_tokens for r in trajectories),
        method="oracle",
        config_id=config.id,
    )

    # Save individual oracle trajectories for analysis
    oracle_dir = output_dir / f"oracle_q_{example.id}"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        with open(oracle_dir / f"run_{i}.json", "w") as f:
            json.dump({
                **traj.to_dict(),
                "oracle_answers": answers,
                "oracle_em": oracle_em,
            }, f, indent=2, ensure_ascii=False)

    return result
