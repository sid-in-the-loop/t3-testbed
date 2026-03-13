"""LLM-as-a-Judge evaluation."""

import asyncio
from typing import Any
from .llm import call_llm

JUDGE_PROMPT = """You are an expert evaluator. Determine if the generated answer correctly answers the question.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Evaluation criteria:
- The generated answer must contain the key information from the ground truth
- Minor wording differences are acceptable
- The generated answer must not contradict the ground truth

First, briefly explain your reasoning.
Then, on the final line, output ONLY "CORRECT" or "INCORRECT"."""


async def judge_answer(
    question: str,
    ground_truth: str,
    generated_answer: Any,
    model: str = "openai/gpt-4o-mini",
) -> bool:
    if not generated_answer or not str(generated_answer).strip():
        return False

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=str(generated_answer),
    )

    response, _, _ = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=200,
        temperature=0.1,
    )

    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip().upper()
        if line in ["CORRECT", "INCORRECT"]:
            return line == "CORRECT"

    response_upper = response.upper()
    if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
        return True
    return False
