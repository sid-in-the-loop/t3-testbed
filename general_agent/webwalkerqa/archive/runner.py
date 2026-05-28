"""
Per-question runner for WebWalkerQA experiments.

Handles:
  - Running a single method on a single question (with error handling)
  - Saving per-question JSON logs
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
from .eval import f1_score
from .methods.base import MethodResult
from .methods import get_method


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
