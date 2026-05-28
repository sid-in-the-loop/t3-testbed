"""
Qualitative analysis tool for LLM search agents.

Records full turn-by-turn trajectories (thought, action, query, search results, response, tokens)
and writes JSON + MD for researcher review.

Modes:
- default: 5 GAIA questions, NAIVE-t4 + DENSE-o16, 4 rollouts each (40 trajectories).
  Output: qualitative_analysis.json, qualitative_analysis.md
- jaccard-vs-naive: 8 questions where jaccard-o16 won vs naive-t4, NAIVE-t4 + JACCARD-o16,
  4 rollouts each, full turn-by-turn. Output: jaccard_vs_naive_qualitative.json, .md

Uses existing: load_dataset, run_single_rollout, generate_pool, select_diverse_queries.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure general_agent is on path for imports
_GA_DIR = Path(__file__).parent.parent
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from dotenv import load_dotenv
env_path = _GA_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from webwalkerqa.configs import get_config
from webwalkerqa.dataset import load_dataset
from webwalkerqa.llm import normalize_model
from webwalkerqa.methods.diversity_scaling import run_single_rollout, generate_pool
from webwalkerqa.methods.utils import select_diverse_queries

# Fallback if utils does not export it (same model used by dense diversity)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 5 varied GAIA questions (from GAIA-25): diverse topics per user suggestion
QUALITATIVE_QUESTION_IDS = ["15", "19", "38", "42", "81"]

# 8 questions where jaccard-o16 got ≥1 correct (LLM-judged) and naive-t4 got 0 (for turn-by-turn comparison)
JACCARD_VS_NAIVE_QUESTION_IDS = ["19", "7", "4", "31", "46", "12", "51", "156"]

MAX_TURNS = 12
POOL_SIZE_O = 16


def _check_api_key(normalized_model: str) -> None:
    if normalized_model.startswith("gemini/"):
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            print("WARNING: Gemini model selected but GOOGLE_API_KEY / GEMINI_API_KEY not set.")
    elif not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found (required for OpenAI models).")
        sys.exit(1)


def _build_trajectory(
    question_id: str,
    question_text: str,
    gold_answer: str,
    condition: str,
    rollout_idx: int,
    pool_candidates: list,
    selected_query: str | None,
    result: dict,
) -> dict:
    """Build one trajectory object per the schema from run_single_rollout result (turn_logs + full_responses)."""
    turn_logs = result.get("turn_logs", [])
    full_responses = result.get("full_responses", [])
    response_by_turn = {r["turn"]: r for r in full_responses}

    turns_schema = []
    for log in turn_logs:
        turn_idx = log["turn"]
        resp = response_by_turn.get(turn_idx, {})
        tokens = resp.get("tokens", {})
        if "query" in log:
            action = "search"
            query = log["query"]
            search_results = log.get("search_result", "")
            response = log.get("response", "") or resp.get("response", "")
        elif "answer" in log:
            action = "answer"
            query = None
            search_results = None
            response = log.get("response", "") or resp.get("response", "")
        else:
            action = "answer"
            query = None
            search_results = None
            response = log.get("response", "") or resp.get("response", "")
        turns_schema.append({
            "turn_idx": turn_idx,
            "thought": "",
            "action": action,
            "query": query,
            "search_results": search_results,
            "response": response or "",
            "prompt_tokens": tokens.get("prompt", 0),
            "completion_tokens": tokens.get("completion", 0),
        })
    return {
        "question_id": question_id,
        "question_text": question_text,
        "gold_answer": gold_answer,
        "condition": condition,
        "rollout_idx": rollout_idx,
        "pool_candidates": list(pool_candidates) if pool_candidates else [],
        "selected_query": selected_query or "",
        "is_correct": result["is_correct"],
        "final_answer": result["answer"],
        "turns_used": result["turns_used"],
        "total_search_calls": result["search_calls"],
        "turns": turns_schema,
    }


async def run_naive_rollouts(model: str, example, embedding_model: str | None) -> list[dict]:
    """4 independent rollouts, no pool; each generates its own turn-1 query."""
    config = get_config("naive-t4")
    max_turns = config.n
    trajectories = []
    for i in range(4):
        result = await run_single_rollout(
            model=model,
            question=example.question,
            answer_gt=str(example.answer),
            max_turns=max_turns,
            initial_query=None,
            rollout_seed=int(example.id or "0") * 1000 + i,
            question_id=str(example.id),
        )
        traj = _build_trajectory(
            question_id=str(example.id),
            question_text=example.question,
            gold_answer=str(example.answer),
            condition="NAIVE-t4",
            rollout_idx=i,
            pool_candidates=[],
            selected_query=None,
            result=result,
        )
        trajectories.append(traj)
    return trajectories


async def run_dense_rollouts(model: str, example, embedding_model: str | None) -> list[dict]:
    """4 independent rollouts; before each, generate pool of 16, select 1 (dense), inject as turn-1."""
    config = get_config("dense-o16")
    max_turns = config.n
    trajectories = []
    for i in range(4):
        pool, _pg, _pc = await generate_pool(model, example.question, POOL_SIZE_O)
        # Run in thread so SentenceTransformer.encode() doesn't block the event loop
        selected = await asyncio.to_thread(
            select_diverse_queries,
            pool, 1, method="dense", seed=int(example.id or "0") * 1000 + i,
        )
        turn1_query = selected[0] if selected else None
        result = await run_single_rollout(
            model=model,
            question=example.question,
            answer_gt=str(example.answer),
            max_turns=max_turns,
            initial_query=turn1_query,
            rollout_seed=int(example.id or "0") * 1000 + i,
            question_id=str(example.id),
        )
        traj = _build_trajectory(
            question_id=str(example.id),
            question_text=example.question,
            gold_answer=str(example.answer),
            condition="DENSE-o16",
            rollout_idx=i,
            pool_candidates=pool,
            selected_query=turn1_query,
            result=result,
        )
        trajectories.append(traj)
    return trajectories


async def run_jaccard_rollouts(model: str, example) -> list[dict]:
    """4 rollouts: one pool of 16, Jaccard select 4 seeds, each rollout gets one seed as turn-1 (same as jaccard-o16)."""
    config = get_config("jaccard-o16")
    max_turns = config.n
    pool, _pg, _pc = await generate_pool(model, example.question, POOL_SIZE_O)
    seeds = select_diverse_queries(
        pool, 4, method="jaccard", seed=int(example.id or "0") * 1000,
    )
    trajectories = []
    for i in range(4):
        turn1_query = seeds[i] if i < len(seeds) else None
        result = await run_single_rollout(
            model=model,
            question=example.question,
            answer_gt=str(example.answer),
            max_turns=max_turns,
            initial_query=turn1_query,
            rollout_seed=int(example.id or "0") * 1000 + i,
            question_id=str(example.id),
        )
        traj = _build_trajectory(
            question_id=str(example.id),
            question_text=example.question,
            gold_answer=str(example.answer),
            condition="JACCARD-o16",
            rollout_idx=i,
            pool_candidates=pool,
            selected_query=turn1_query,
            result=result,
        )
        trajectories.append(traj)
    return trajectories


def write_markdown(trajectories: list[dict], out_path: Path) -> None:
    """Write qualitative_analysis.md grouped by question, then condition, then rollout."""
    lines = []
    # Group by question_id
    by_q: dict[str, list[dict]] = {}
    for t in trajectories:
        qid = t["question_id"]
        if qid not in by_q:
            by_q[qid] = []
        by_q[qid].append(t)

    for qid in sorted(by_q.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        trajs = by_q[qid]
        # Use first trajectory for question text and gold (same for all)
        first = trajs[0]
        question_text = first["question_text"]
        gold_answer = first["gold_answer"]
        lines.append(f"## Question: {question_text[:200]}{'...' if len(question_text) > 200 else ''}")
        lines.append(f"**Gold answer:** {gold_answer}")
        lines.append("")

        for t in sorted(trajs, key=lambda x: (x["condition"], x["rollout_idx"])):
            cond = t["condition"]
            ridx = t["rollout_idx"]
            result_label = "CORRECT" if t["is_correct"] else "WRONG"
            pred = t["final_answer"]
            lines.append(f"**Condition:** {cond} | Rollout {ridx}")
            lines.append(f"**Result:** {result_label} (predicted: {pred})")
            lines.append(f"**Turns used:** {t['turns_used']} | **Searches:** {t['total_search_calls']}")
            lines.append("")

            for turn in t["turns"]:
                idx = turn["turn_idx"]
                lines.append(f"### Turn {idx}")
                lines.append(f"**Thought:** {turn.get('thought', '') or '(none)'}")
                action = turn.get("action", "")
                if action == "search":
                    lines.append(f"**Action:** search → \"{turn.get('query', '')}\"")
                    lines.append(f"**Results:** {turn.get('search_results') or ''}")
                else:
                    lines.append(f"**Action:** answer")
                    lines.append(f"**Response:** {turn.get('response', '')}")
                lines.append("")
            lines.append("---")
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def print_summary_table(trajectories: list[dict], condition_order: list[str] | None = None) -> None:
    """Print summary table: Question | Condition | R0 R1 R2 R3 | Turns avg."""
    from collections import defaultdict
    grid: dict[tuple[str, str], list[str]] = defaultdict(list)
    turn_totals: dict[tuple[str, str], list[int]] = defaultdict(list)

    for t in trajectories:
        key = (t["question_id"], t["condition"])
        while len(grid[key]) <= t["rollout_idx"]:
            grid[key].append("")
            turn_totals[key].append(0)
        grid[key][t["rollout_idx"]] = "CORRECT" if t["is_correct"] else "WRONG"
        turn_totals[key][t["rollout_idx"]] = t["turns_used"]

    if not condition_order:
        condition_order = list({t["condition"] for t in trajectories})
    print("\nQuestion | Condition   | R0      | R1      | R2      | R3      | Turns avg")
    print("---------|-------------|---------|---------|---------|---------|----------")
    question_ids = sorted(set(t["question_id"] for t in trajectories), key=lambda q: int(q) if str(q).isdigit() else 0)
    for qid in question_ids:
        for cond in condition_order:
            key = (qid, cond)
            if key not in grid:
                continue
            row = grid[key]
            turns = turn_totals[key]
            avg = sum(turns) / len(turns) if turns else 0
            r0 = (row[0] if len(row) > 0 else "").ljust(8)
            r1 = (row[1] if len(row) > 1 else "").ljust(8)
            r2 = (row[2] if len(row) > 2 else "").ljust(8)
            r3 = (row[3] if len(row) > 3 else "").ljust(8)
            print(f"Q{qid:<6} | {cond:<11} | {r0} | {r1} | {r2} | {r3} | {avg:.1f}")


def _summary_table_conditions(trajectories: list[dict]) -> list[str]:
    """Return ordered list of condition names for the summary table (e.g. NAIVE-t4, JACCARD-o16)."""
    seen = set()
    order = []
    for t in trajectories:
        c = t["condition"]
        if c not in seen:
            seen.add(c)
            order.append(c)
    return order


async def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative analysis: turn-by-turn trajectories for GAIA")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "jaccard-vs-naive"],
        default="default",
        help="default: 5 questions, NAIVE-t4 + DENSE-o16. jaccard-vs-naive: 8 questions (jaccard wins / naive fails), NAIVE-t4 + JACCARD-o16 with full turn-by-turn logs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to GAIA JSON. Default: data/gaia_25.json (default mode), data/GAIA.json (jaccard-vs-naive).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model: LiteLLM string or alias (gpt4.1-mini, gemini-2.5-flash)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_GA_DIR),
        help="Directory for output JSON and MD",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Dense diversity embedding model (default: all-MiniLM-L6-v2); only for default mode",
    )
    args = parser.parse_args()

    model = normalize_model(args.model)
    _check_api_key(model)
    embedding_model = args.embedding_model or DEFAULT_EMBEDDING_MODEL

    if args.mode == "jaccard-vs-naive":
        dataset_path = args.dataset or str(_GA_DIR / "data" / "GAIA.json")
        question_ids = JACCARD_VS_NAIVE_QUESTION_IDS
        json_name = "jaccard_vs_naive_qualitative.json"
        md_name = "jaccard_vs_naive_qualitative.md"
        conditions_list = ["NAIVE-t4", "JACCARD-o16"]
    else:
        dataset_path = args.dataset or str(_GA_DIR / "data" / "gaia_25.json")
        question_ids = QUALITATIVE_QUESTION_IDS
        json_name = "qualitative_analysis.json"
        md_name = "qualitative_analysis.md"
        conditions_list = ["NAIVE-t4", "DENSE-o16"]

    dataset = await asyncio.to_thread(load_dataset, path=dataset_path)
    id_set = set(question_ids)
    examples = [ex for ex in dataset if str(ex.id) in id_set]
    if len(examples) < len(question_ids):
        found = {str(ex.id) for ex in examples}
        print(f"WARNING: Only found {len(examples)} of {len(question_ids)} questions (ids {question_ids}). Found: {found}")

    all_trajectories = []
    for ex in examples:
        print(f"Question {ex.id}: NAIVE-t4 (4 rollouts)...")
        naive_trajs = await run_naive_rollouts(model, ex, embedding_model)
        all_trajectories.extend(naive_trajs)
        if args.mode == "jaccard-vs-naive":
            print(f"Question {ex.id}: JACCARD-o16 (4 rollouts)...")
            jaccard_trajs = await run_jaccard_rollouts(model, ex)
            all_trajectories.extend(jaccard_trajs)
        else:
            print(f"Question {ex.id}: DENSE-o16 (4 rollouts)...")
            dense_trajs = await run_dense_rollouts(model, ex, embedding_model)
            all_trajectories.extend(dense_trajs)

    metadata = {
        "n_questions": len(examples),
        "n_rollouts_per_condition": 4,
        "conditions": conditions_list,
        "max_turns": MAX_TURNS,
        "pool_size_o": POOL_SIZE_O,
        "model": model,
        "gaia_questions": [str(ex.id) for ex in examples],
        "mode": args.mode,
    }
    payload = {
        "metadata": metadata,
        "trajectories": all_trajectories,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / json_name
    md_path = out_dir / md_name

    def _write_json():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_md():
        write_markdown(all_trajectories, md_path)

    await asyncio.to_thread(_write_json)
    await asyncio.to_thread(_write_md)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    # Summary table: show conditions that appear in trajectories
    summary_conditions = _summary_table_conditions(all_trajectories)
    print_summary_table(all_trajectories, condition_order=summary_conditions)


if __name__ == "__main__":
    asyncio.run(main())
