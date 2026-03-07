"""
LLM-as-a-Judge evaluation for WebWalkerQA experiments.

Evaluates generated answers against ground truth using gpt-4o-mini as judge.
Calculates pass@k metrics based on binary correctness decisions.

Usage:
  python -m webwalkerqa.eval_llm --results-dir results/gaia_25/s1/ --num-samples 8
  python -m webwalkerqa.eval_llm --results-dir results/gaia_25/T3-Dynamic/ --num-samples 8
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# Ensure the general_agent directory is importable
_GA_DIR = Path(__file__).parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
load_dotenv(_GA_DIR / ".env")

from webwalkerqa.llm import call_llm
from webwalkerqa.eval import estimate_pass_at_k


JUDGE_PROMPT = """You are an expert evaluator assessing the correctness of answers to complex questions.

Your task is to determine if a generated answer correctly addresses the question by comparing it to the ground truth answer.

**Evaluation Criteria:**
- The generated answer must contain ALL key information present in the ground truth
- The generated answer must be factually accurate and not contradict the ground truth
- Minor differences in wording, formatting, or additional details are acceptable as long as all essential facts are present
- If the ground truth has specific facts, numbers, names, or details, they must appear in the generated answer

**Response Format:**
First, provide a brief reasoning explaining your decision.
Then, output ONLY "CORRECT" or "INCORRECT" on the final line.

**Examples:**

Question: What is the capital of France?
Ground Truth: Paris
Generated Answer: Paris is the capital of France.
Reasoning: The generated answer contains "Paris" which matches the ground truth and provides additional context, but all essential information is present.
CORRECT

Question: What is the population of Tokyo?
Ground Truth: Approximately 14 million people
Generated Answer: Tokyo has about 13 million residents.
Reasoning: The generated answer has a slightly different number (13M vs 14M) which contradicts the ground truth.
INCORRECT

---

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Reasoning:"""


async def judge_answer(question: str, ground_truth: str, generated_answer: Any,
                      model: str = "openai/gpt-4o-mini") -> bool:
    """
    Use LLM to judge if generated answer is correct compared to ground truth.

    Returns True if correct, False if incorrect.
    """
    # Handle list of answers (if result has multiple samples)
    if isinstance(generated_answer, list):
        if not generated_answer:
            return False
        generated_answer = str(generated_answer[0])
    
    if not generated_answer or not str(generated_answer).strip():
        return False

    generated_answer = str(generated_answer)

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )

    response, _, _ = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=200,
        temperature=0.1,  # Low temperature for consistent judging
    )

    # Extract the final decision
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().upper()
        if line in ["CORRECT", "INCORRECT"]:
            return line == "CORRECT"

    # Fallback: look for keywords
    response_upper = response.upper()
    if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
        return True
    elif "INCORRECT" in response_upper:
        return False

    # Default to incorrect if unclear
    print(f"Warning: Unclear judgment for answer '{generated_answer[:50]}...'. Defaulting to INCORRECT.")
    return False


async def evaluate_question_samples(question_id: str, question: str, ground_truth: str,
                                   sample_answers: List[str], model: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Evaluate multiple samples for one question using LLM judge.

    Returns dict with correctness labels and pass@k scores.
    """
    # Judge each sample in parallel with semaphore
    tasks = []
    for answer in sample_answers:
        async def judged_task(ans):
            async with semaphore:
                return await judge_answer(question, ground_truth, ans, model)
        tasks.append(judged_task(answer))
    
    correctness_labels = await asyncio.gather(*tasks)

    # Calculate pass@k
    num_samples = len(correctness_labels)
    num_correct = sum(correctness_labels)

    pass_scores = {}
    k_values = [1, 2, 4, 8, 16]
    for k in k_values:
        if k <= num_samples:
            pass_scores[f"pass@{k}"] = estimate_pass_at_k(num_samples, num_correct, k)

    return {
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "sample_answers": sample_answers,
        "correctness_labels": correctness_labels,
        "num_correct": num_correct,
        "num_samples": num_samples,
        **pass_scores
    }


async def evaluate_experiment(results_dir: Path, num_samples: int,
                             model: str = "openai/gpt-4o-mini", max_concurrent: int = 20) -> Dict[str, Any]:
    """
    Evaluate an entire experiment directory using LLM judge.

    Args:
        results_dir: Directory containing per-question JSON files
        num_samples: Number of samples to evaluate per question
        model: Judge model to use
        max_concurrent: Max concurrent LLM calls

    Returns:
        Dict with aggregate statistics
    """
    print(f"Evaluating {results_dir} with {num_samples} samples per question (concurrency={max_concurrent})...")

    # Group results by question
    question_results = defaultdict(list)

    # Load all result files
    json_files = list(results_dir.glob("*.json"))
    print(f"Found {len(json_files)} result files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            question_id = result.get("question_id")
            if question_id is None:
                # Try to extract from filename
                filename = json_file.stem
                if filename.startswith("agg_q_"):
                    question_id = filename.split("_")[2]
                elif filename.startswith("q_"):
                    question_id = filename.split("_")[1].split("_")[0]  # Handle q_123_s1.json
                else:
                    continue

            question_results[question_id].append(result)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Grouped into {len(question_results)} questions")

    # For each question, collect samples and evaluate
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    total_questions = len(question_results)

    # Convert to list for ordered processing
    ordered_questions = sorted(question_results.items(), key=lambda x: x[0])

    for i, (question_id, results) in enumerate(ordered_questions):
        if len(results) == 0:
            continue

        # Use the first result to get question and ground truth
        first_result = results[0]
        question = first_result.get("question", "")
        ground_truth = first_result.get("answer_gt", "")

        # Collect sample answers (up to num_samples)
        sample_answers = []
        for result in results:
            final_answer = result.get("final_answer", "")
            # If final_answer is a list of samples from a single run, extend
            if isinstance(final_answer, list):
                sample_answers.extend([str(a) for a in final_answer])
            else:
                sample_answers.append(str(final_answer))

        # Truncate or pad to exactly num_samples
        sample_answers = sample_answers[:num_samples]
        while len(sample_answers) < num_samples:
            sample_answers.append("")

        # Add task
        tasks.append(evaluate_question_samples(
            question_id, question, ground_truth, sample_answers, model, semaphore
        ))

    # Run all evaluations
    all_question_evals = await asyncio.gather(*tasks)

    # Aggregate results
    if not all_question_evals:
        return {"error": "No questions evaluated"}

    total_questions = len(all_question_evals)
    total_correct = sum(qe["num_correct"] for qe in all_question_evals)
    avg_correct = total_correct / total_questions

    # Average pass@k across questions
    pass_scores = {}
    k_values = [1, 2, 4, 8, 16]
    for k in k_values:
        if k <= num_samples:
            avg_pass_k = sum(qe.get(f"pass@{k}", 0) for qe in all_question_evals) / total_questions
            pass_scores[f"pass@{k}"] = avg_pass_k

    # Save detailed results
    output_file = results_dir / "llm_judge_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": results_dir.name,
            "judge_model": model,
            "num_samples": num_samples,
            "total_questions": total_questions,
            "aggregate_stats": {
                "total_correct": total_correct,
                "avg_correct_per_question": avg_correct,
                **pass_scores
            },
            "question_evaluations": all_question_evals
        }, f, indent=2)

    print(f"Results saved to {output_file}")

    return {
        "experiment": results_dir.name,
        "total_questions": total_questions,
        "avg_correct_per_question": avg_correct,
        **pass_scores
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation for WebWalkerQA")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--num-samples", type=int, default=8,
                       help="Number of samples to evaluate per question")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-4o-mini",
                       help="Model to use for judging")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of questions to evaluate concurrently")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        sys.exit(1)

    # Note: For simplicity, we're not using batch processing here
    # In a full implementation, you'd want to batch the LLM judge calls
    result = asyncio.run(evaluate_experiment(
        results_dir, args.num_samples, args.judge_model
    ))

    print("\n" + "="*50)
    print(f"LLM JUDGE RESULTS for {results_dir.name}")
    print("="*50)
    print(f"Questions evaluated: {result['total_questions']}")
    print(f"Avg correct per question: {result.get('avg_correct_per_question', 0):.2f}")
    for k in [1, 2, 4, 8, 16]:
        if f"pass@{k}" in result:
            print(f"Pass@{k}: {result[f'pass@{k}']:.2%}")

    print(f"\nDetailed results saved to: {results_dir}/llm_judge_evaluation.json")


if __name__ == "__main__":
    main()