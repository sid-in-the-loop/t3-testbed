import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

_GA_DIR = Path(__file__).resolve().parent.parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

load_dotenv(_GA_DIR / ".env")

from webwalkerqa.llm import call_llm


JUDGE_PROMPT = """You are an expert evaluator. Determine if the generated answer correctly answers the question based on the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Evaluation Rubric:
1. Factuality: Does the answer contain the core correct information? All key facts must be present.
2. Semantic equivalence: Mark CORRECT if the meaning is the same even if phrased differently:
   - Durations expressed as start/end dates vs. duration length (e.g. "Sep 2022 to Feb 2024" = "18 months" = "18-month project")
   - Abbreviations and alternate names (e.g. "St. Petersburg" = "Saint Petersburg", "US" = "United States")
   - Numbers in different formats (e.g. "142,000" = "142 thousand", "$1.2M" = "1.2 million dollars")
   - Dates in different formats (e.g. "September 1, 2022" = "Sept 1 2022" = "1 September 2022")
3. Completeness: For multi-part questions, all parts must be correctly answered.
4. Contradiction: Mark INCORRECT only if the answer directly contradicts the ground truth (wrong facts, not just different phrasing).
5. Extra information: Ignore extra details in the generated answer as long as the core answer is correct.

Briefly explain your reasoning, then output "CORRECT" or "INCORRECT" on the final line."""


WEB_JUDGE_PROMPT = """Please determine if the predicted answer is SEMANTICALLY equivalent to the labeled answer.

Question: {question}
Labeled Answer: {ground_truth}
Predicted Answer: {generated_answer}

Output as JSON (no markdown fences):
{{"rationale": "your rationale as text", "judgement": "correct" or "incorrect"}}"""


_JUDGE_PROMPT_STYLE = "mhqa"


def set_judge_prompt_style(style: str) -> None:
    global _JUDGE_PROMPT_STYLE
    assert style in ("mhqa", "web"), f"bad judge style: {style}"
    _JUDGE_PROMPT_STYLE = style


async def judge_answer(
    question: str,
    ground_truth: str,
    generated_answer: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    if not generated_answer or not str(generated_answer).strip():
        return False

    if _JUDGE_PROMPT_STYLE == "web":
        prompt = WEB_JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=str(generated_answer),
        )
    else:
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=str(generated_answer),
        )

    async def _call():
        try:
            response, _, _ = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=300,
                temperature=0.0,
            )
            if _JUDGE_PROMPT_STYLE == "web":
                import re
                m = re.search(r'"judgement"\s*:\s*"(correct|incorrect)"', response, re.IGNORECASE)
                if m:
                    return m.group(1).lower() == "correct"
                resp_low = response.lower()
                return "incorrect" not in resp_low and "correct" in resp_low
            lines = response.strip().split("\n")
            for line in reversed(lines):
                line = line.strip().upper()
                if "CORRECT" in line and "INCORRECT" not in line:
                    return True
                if "INCORRECT" in line:
                    return False
            return "CORRECT" in response.upper() and "INCORRECT" not in response.upper()
        except Exception:
            return False

    async with semaphore:
        return await _call()


def find_run_dirs(
    results_dir: Path,
    filter_model: str,
    filter_dataset: str,
    filter_condition: str,
) -> List[Tuple[Path, Path]]:
    """Return list of (run_dir, jsonl_path) for all matching runs."""
    matches = []
    for jsonl_path in sorted(results_dir.rglob("*_T*.jsonl")):
        run_dir = jsonl_path.parent
        try:
            rel = jsonl_path.relative_to(results_dir)
            parts = rel.parts
            if len(parts) < 5:
                continue
            model_dir, dataset, condition = parts[0], parts[1], parts[2]
        except ValueError:
            continue

        if filter_model and model_dir != filter_model:
            continue
        if filter_dataset and dataset != filter_dataset:
            continue
        if filter_condition and condition != filter_condition:
            continue

        summary_files = list(run_dir.glob("summary_T*.csv"))
        if not summary_files:
            continue

        matches.append((run_dir, jsonl_path, summary_files[0]))

    return matches


async def run_judge_flat(run_dir: Path, model: str, max_concurrent: int, force: bool):
    """Judge all jsonl files directly in run_dir (no nested structure required)."""
    jsonl_files = sorted(run_dir.rglob("*_T*.jsonl"))
    if not jsonl_files:
        print(f"No *_T*.jsonl files found in {run_dir}")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    for jsonl_path in jsonl_files:
        summary_files = list(jsonl_path.parent.glob("summary_T*.csv"))
        if not summary_files:
            continue
        summary_csv = summary_files[0]
        df = pd.read_csv(summary_csv)
        if not force and "pass_at_1_llm" in df.columns:
            print(f"  Already judged: {jsonl_path.name} — skip (use --force to redo)")
            continue

        questions = []
        all_tasks = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tasks = [
                    asyncio.create_task(judge_answer(
                        data["question"], data["answer_gt"], ans, model, semaphore
                    ))
                    for ans in data.get("rollout_answers", [])
                ]
                questions.append((data, tasks))
                all_tasks.extend(tasks)

        print(f"Judging {len(all_tasks)} answers in {jsonl_path.name}...")
        await tqdm.gather(*all_tasks, desc="judging")

        judged = []
        for data, tasks in questions:
            results = [t.result() for t in tasks]
            p1 = 1 if results and results[0] else 0
            p4 = 1 if any(results) else 0
            judged.append({**data, "judged_rollouts": results, "pass_at_1_llm": p1, "pass_at_4_llm": p4})

        n = len(judged)
        mean_p1 = sum(d["pass_at_1_llm"] for d in judged) / n if n else 0
        mean_p4 = sum(d["pass_at_4_llm"] for d in judged) / n if n else 0

        df["pass_at_1_llm"] = round(mean_p1, 4)
        df["pass_at_4_llm"] = round(mean_p4, 4)
        df.to_csv(summary_csv, index=False)

        with open(jsonl_path, "w") as f:
            for entry in judged:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  p@1_llm={mean_p1:.4f}  p@4_llm={mean_p4:.4f}  (n={n})")


async def run_judge(results_dir: Path, model: str, max_concurrent: int,
                    filter_model: str, filter_dataset: str, filter_condition: str,
                    force: bool):

    run_entries = find_run_dirs(results_dir, filter_model, filter_dataset, filter_condition)
    if not run_entries:
        print("No matching run dirs found.")
        return

    to_judge = []
    skipped = 0
    for run_dir, jsonl_path, summary_csv in run_entries:
        df = pd.read_csv(summary_csv)
        if not force and "pass_at_1_llm" in df.columns:
            skipped += 1
            continue
        to_judge.append((run_dir, jsonl_path, summary_csv))

    print(f"Found {len(run_entries)} runs | Skipping {skipped} already judged | Judging {len(to_judge)}")
    if not to_judge:
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    all_tasks = []
    run_data: List[Tuple[Path, Path, Path, List[Tuple[Dict, List]]]] = []

    for run_dir, jsonl_path, summary_csv in to_judge:
        questions = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tasks = [
                    asyncio.create_task(judge_answer(
                        data["question"], data["answer_gt"], ans, model, semaphore
                    ))
                    for ans in data.get("rollout_answers", [])
                ]
                questions.append((data, tasks))
                all_tasks.extend(tasks)
        run_data.append((run_dir, jsonl_path, summary_csv, questions))

    total_answers = len(all_tasks)
    print(f"Judging {total_answers} rollout answers across {len(to_judge)} runs...")
    await tqdm.gather(*all_tasks, desc="Judging")

    for run_dir, jsonl_path, summary_csv, questions in run_data:
        judged = []
        for data, tasks in questions:
            results = [t.result() for t in tasks]
            p1 = 1 if results and results[0] else 0
            p4 = 1 if any(results) else 0
            judged.append({**data, "judged_rollouts": results,
                           "pass_at_1_llm": p1, "pass_at_4_llm": p4})

        n = len(judged)
        mean_p1 = sum(d["pass_at_1_llm"] for d in judged) / n if n else 0
        mean_p4 = sum(d["pass_at_4_llm"] for d in judged) / n if n else 0

        df = pd.read_csv(summary_csv)
        df["pass_at_1_llm"] = round(mean_p1, 4)
        df["pass_at_4_llm"] = round(mean_p4, 4)
        df.to_csv(summary_csv, index=False)
        with open(jsonl_path, "w") as f:
            for entry in judged:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        rel = run_dir.relative_to(results_dir)
        print(f"  {rel}  p@1_llm={mean_p1:.3f}  p@4_llm={mean_p4:.3f}  (n={n})")

    print(f"\nDone. Updated {len(to_judge)} summary CSVs.")


async def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge for main table results")
    parser.add_argument("--results-dir", default="results/main_table",
                        help="Root results directory to walk")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="Judge model")
    parser.add_argument("--max-concurrent", type=int, default=200)
    parser.add_argument("--filter-model", default="",
                        help="Only judge this model subdir (e.g. gpt-4o-mini)")
    parser.add_argument("--filter-dataset", default="",
                        help="Only judge this dataset (e.g. bamboogle)")
    parser.add_argument("--filter-condition", default="",
                        help="Only judge this condition (e.g. diversity_parallel)")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if pass_at_1_llm already exists")
    parser.add_argument("--run-dir", default="",
                        help="Judge a single flat run dir directly (skips nested structure requirement)")
    parser.add_argument("--judge-prompt-style", default="mhqa", choices=["mhqa", "web"],
                        help="Judge prompt template. mhqa (default) = current. web = WebWalker-paper JSON-output prompt for Table 2 hard-reasoning tasks.")
    args = parser.parse_args()

    set_judge_prompt_style(args.judge_prompt_style)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            print(f"Run dir not found: {run_dir}")
            sys.exit(1)
        await run_judge_flat(run_dir, model=args.model, max_concurrent=args.max_concurrent, force=args.force)
        return

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        sys.exit(1)

    await run_judge(
        results_dir=results_dir,
        model=args.model,
        max_concurrent=args.max_concurrent,
        filter_model=args.filter_model,
        filter_dataset=args.filter_dataset,
        filter_condition=args.filter_condition,
        force=args.force,
    )


if __name__ == "__main__":
    asyncio.run(main())
