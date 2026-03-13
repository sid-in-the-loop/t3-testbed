"""
Diversity Scaling Methods — 200 rollouts per condition (25q × 8 rollouts/q).

- Rollout = one full agent run (12 turns, 1 answer).
- 8 rollouts per question → 200 total per condition.

SEQUENTIAL — 200 rollouts, free turn-1.
DIVERSITY-1 (Jaccard/TFIDF, o=16/48/64) — 200 rollouts; at turn-1 only: oversample o, DPP select 1, use as turn-1.
DIVERSITY-ALL TFIDF o=64 — 200 rollouts; at every turn: oversample 64, DPP select 1.
"""

import asyncio
import re
from typing import Optional, List, Tuple, Dict, Any

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match
from .base import BaseMethod, MethodResult, TurnLog, extract_answer
from .utils import select_diverse_queries


# ReAct prompt (matches s1.py style)
REACT_PROMPT = """\
You are a research assistant that answers questions by searching the web.

You have {max_turns} turns to find the answer. You are on turn {turn}.

Question: {question}

History of searches and findings:
{history}

Instructions:
- If you need more information, output: <search>your query</search>
- If you have enough information to answer, output: <answer>your answer</answer>
- If this is the last turn, you MUST provide an answer.

Your response:"""


POOL_GEN_PROMPT = """\
Generate exactly {o} diverse search queries to investigate this question.
Each query should approach the question from a different angle.
{history_block}

Question: {question}

Output exactly {o} queries, one per line, numbered 1-{o}. No other text."""


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from <tag>...</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _parse_pool_response(text: str, o: int) -> List[str]:
    """Parse numbered queries from pool generation response."""
    lines = text.strip().split('\n')
    queries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^\d+[\.\)\:\-]\s*', '', line).strip()
        cleaned = cleaned.strip('"').strip("'")
        if cleaned and len(cleaned) > 3:
            queries.append(cleaned)
    return queries[:o]


def _format_history(query: str, result: str) -> str:
    """Format a single search entry for history."""
    return f"<search>{query}</search>\n<result>{result[:600]}</result>"


async def generate_pool(
    model: str, question: str, o: int, history: Optional[str] = None
) -> Tuple[List[str], int, int]:
    """
    Generate o candidate queries in a single LLM call.
    If history is provided (for DIVERSITY-ALL turns 2+), include it so model can suggest next queries.
    Returns (queries, prompt_tokens, completion_tokens).
    """
    history_block = ""
    if history:
        history_block = f"Information gathered so far:\n{history}\n\nGenerate diverse NEXT search queries (do not repeat).\n"
    prompt = POOL_GEN_PROMPT.format(o=o, question=question, history_block=history_block)
    text, p_tok, o_tok = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=2048,
        temperature=1.0,
    )
    queries = _parse_pool_response(text, o)
    while len(queries) < o:
        queries.append(f"{question[:50]} variant {len(queries)}")
    return queries[:o], p_tok, o_tok


async def run_single_rollout(
    model: str,
    question: str,
    answer_gt: str,
    max_turns: int,
    initial_query: Optional[str] = None,
    rollout_seed: int = 0,
) -> Dict[str, Any]:
    """
    Run one complete rollout (12 turns → 1 answer).
    
    Args:
        initial_query: If provided, inject as turn-1 search query (skip turn-1 LLM call).
                      If None, the rollout generates its own turn-1 query.
    
    Returns:
        Dict with answer, is_correct, turns_used, search_calls, tokens.
    """
    history_str = ""
    final_answer = None
    turn_logs = []
    total_prompt = 0
    total_completion = 0
    search_calls = 0

    for turn in range(1, max_turns + 1):
        # Turn 1 with injected query: skip LLM call, go straight to search
        if turn == 1 and initial_query is not None:
            sr = web_search(initial_query, max_chars=2000)
            history_str = _format_history(initial_query, sr)
            search_calls += 1
            turn_logs.append({
                "turn": 1,
                "query": initial_query,
                "injected": True,
            })
            continue

        # Normal turn: call LLM
        prompt = REACT_PROMPT.format(
            max_turns=max_turns,
            turn=turn,
            question=question,
            history=history_str or "(none yet)",
        )
        temp = 1.0 if turn == 1 else 0.7
        response, p_tok, o_tok = await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=1024,
            temperature=temp,
            seed=rollout_seed + turn,
        )
        total_prompt += p_tok
        total_completion += o_tok

        # Check for answer
        answer = _extract_tag(response, "answer")
        if answer:
            final_answer = answer
            turn_logs.append({"turn": turn, "answer": answer})
            break

        # Check for search
        query = _extract_tag(response, "search")
        if query:
            sr = web_search(query, max_chars=2000)
            history_str = (history_str + "\n" + _format_history(query, sr)).strip()
            search_calls += 1
            turn_logs.append({"turn": turn, "query": query})
        else:
            # No valid action — force answer on last turn
            if turn == max_turns:
                final_answer = response.strip()[:500]
            turn_logs.append({"turn": turn, "no_action": True})
            break

    # Fallback answer extraction
    if final_answer is None and turn_logs:
        final_answer = ""

    is_correct = exact_match(final_answer or "", answer_gt)

    return {
        "answer": final_answer or "",
        "is_correct": is_correct,
        "turns_used": len(turn_logs),
        "search_calls": search_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
    }


# ---------------------------------------------------------------------------
# Condition 1: SEQUENTIAL
# ---------------------------------------------------------------------------

class SequentialMethod(BaseMethod):
    """
    SEQUENTIAL: 200 rollouts total (8 per question). No oversampling, free turn-1.
    """

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
    ) -> MethodResult:
        k = self.config.k  # 8 rollouts per question
        max_turns = self.config.n  # 12

        if pbar:
            pbar.set_description(f"Q{question_id}: SEQUENTIAL")

        # Run k independent rollouts (no pool, no injection)
        tasks = [
            run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=None,
                rollout_seed=int(question_id or "0") * 1000 + i,
            )
            for i in range(k)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_rollout_results = []
        for i, r in enumerate(batch_results):
            r["rollout_idx"] = i
            all_rollout_results.append(r)

        # Aggregate
        n_correct = sum(r["is_correct"] for r in all_rollout_results)
        any_correct = n_correct > 0
        first_correct = next((r["answer"] for r in all_rollout_results if r["is_correct"]), "")

        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer=first_correct or all_rollout_results[0]["answer"] if all_rollout_results else "",
            em=any_correct,
            method="sequential",
            config_id=self.config.id,
            turns_used=max_turns,
            search_calls_used=sum(r["search_calls"] for r in all_rollout_results),
            total_prompt_tokens=sum(r["prompt_tokens"] for r in all_rollout_results),
            total_output_tokens=sum(r["completion_tokens"] for r in all_rollout_results),
        )
        result.metadata = {
            "condition": "SEQUENTIAL",
            "n_rollouts": len(all_rollout_results),
            "n_correct": n_correct,
            "rollout_results": all_rollout_results,
            "pool_gen_prompt_tokens": 0,
            "pool_gen_completion_tokens": 0,
        }
        return result


# ---------------------------------------------------------------------------
# Diversity Parallel: one pool of o, select 4 seeds, run 4 threads
# ---------------------------------------------------------------------------

class DiversityParallelMethod(BaseMethod):
    """
    At Turn 1: sample o candidate queries at temperature=1.0. Greedy max-min selection
    (Jaccard or dense embeddings) to pick 4 seeds. Run 4 independent ReAct threads
    from those seeds for 12 turns each.
    """

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
    ) -> MethodResult:
        k = self.config.k  # 4 threads
        pool_size = self.config.o  # 16, 32, 48, or 64
        max_turns = self.config.n  # 12
        selection_method = self.config.diversity_method  # "jaccard" or "dense"

        if pbar:
            pbar.set_description(f"Q{question_id}: DIV-PAR o={pool_size} {selection_method}")

        # One pool per question, select k seeds
        pool, pool_prompt, pool_completion = await generate_pool(
            self.model, question, pool_size
        )
        seeds = select_diverse_queries(
            pool, k, method=selection_method, seed=int(question_id or "0")
        )

        # Run k rollouts from those seeds
        tasks = [
            run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=seeds[i] if i < len(seeds) else None,
                rollout_seed=int(question_id or "0") * 1000 + i,
            )
            for i in range(k)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_rollout_results = []
        for i, r in enumerate(batch_results):
            r["rollout_idx"] = i
            all_rollout_results.append(r)

        n_correct = sum(r["is_correct"] for r in all_rollout_results)
        any_correct = n_correct > 0
        first_correct = next((r["answer"] for r in all_rollout_results if r["is_correct"]), "")

        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer=first_correct or (all_rollout_results[0]["answer"] if all_rollout_results else ""),
            em=any_correct,
            method="diversity_parallel",
            config_id=self.config.id,
            turns_used=max_turns,
            search_calls_used=sum(r["search_calls"] for r in all_rollout_results),
            total_prompt_tokens=sum(r["prompt_tokens"] for r in all_rollout_results) + pool_prompt,
            total_output_tokens=sum(r["completion_tokens"] for r in all_rollout_results) + pool_completion,
        )
        result.metadata = {
            "condition": "diversity_parallel",
            "selection_method": selection_method,
            "pool_size": pool_size,
            "n_rollouts": len(all_rollout_results),
            "n_correct": n_correct,
            "rollout_results": all_rollout_results,
            "pool_gen_prompt_tokens": pool_prompt,
            "pool_gen_completion_tokens": pool_completion,
        }
        return result


# ---------------------------------------------------------------------------
# Condition 2: DIVERSITY-1 (per-rollout pool, select 1)
# ---------------------------------------------------------------------------

class Diversity1Method(BaseMethod):
    """
    DIVERSITY-1: 200 rollouts (8 per question). At turn-1 only: oversample o, DPP select 1, use as turn-1.
    o from config (16, 48, or 64); method jaccard or dpp_tfidf.
    """

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
    ) -> MethodResult:
        k = self.config.k  # 8 rollouts per question
        pool_size = self.config.o  # 16, 48, or 64
        max_turns = self.config.n  # 12
        selection_method = self.config.diversity_method  # "jaccard" or "dpp_tfidf"

        all_rollout_results = []
        total_pool_prompt = 0
        total_pool_completion = 0

        for rollout_idx in range(k):
            if pbar:
                pbar.set_description(f"Q{question_id}: DIV-1 {rollout_idx+1}/{k} o={pool_size}")

            # Per rollout: generate o, DPP select 1, run one rollout with that as turn-1
            pool, pg_p, pg_c = await generate_pool(self.model, question, pool_size)
            total_pool_prompt += pg_p
            total_pool_completion += pg_c
            selected = select_diverse_queries(pool, 1, method=selection_method, seed=rollout_idx)
            turn1_query = selected[0] if selected else None

            r = await run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=turn1_query,
                rollout_seed=int(question_id or "0") * 1000 + rollout_idx,
            )
            r["rollout_idx"] = rollout_idx
            r["injected_query"] = turn1_query
            all_rollout_results.append(r)

        # Aggregate
        n_correct = sum(r["is_correct"] for r in all_rollout_results)
        any_correct = n_correct > 0
        first_correct = next((r["answer"] for r in all_rollout_results if r["is_correct"]), "")

        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer=first_correct or all_rollout_results[0]["answer"] if all_rollout_results else "",
            em=any_correct,
            method="diversity_1",
            config_id=self.config.id,
            turns_used=max_turns,
            search_calls_used=sum(r["search_calls"] for r in all_rollout_results),
            total_prompt_tokens=sum(r["prompt_tokens"] for r in all_rollout_results) + total_pool_prompt,
            total_output_tokens=sum(r["completion_tokens"] for r in all_rollout_results) + total_pool_completion,
        )
        result.metadata = {
            "condition": "DIVERSITY-1",
            "selection_method": selection_method,
            "pool_size": pool_size,
            "n_rollouts": len(all_rollout_results),
            "n_correct": n_correct,
            "rollout_results": all_rollout_results,
            "pool_gen_prompt_tokens": total_pool_prompt,
            "pool_gen_completion_tokens": total_pool_completion,
        }
        return result


# ---------------------------------------------------------------------------
# Condition 3: DIVERSITY-ALL
# ---------------------------------------------------------------------------

async def run_single_rollout_diversity_all(
    model: str,
    question: str,
    answer_gt: str,
    max_turns: int,
    pool_size: int,
    selection_method: str,
    rollout_seed: int,
) -> Dict[str, Any]:
    """
    One rollout for DIVERSITY-ALL: at every turn, generate pool_size candidates,
    DPP select 1, execute that search. After max_turns, one LLM call for final answer.
    """
    history_str = ""
    total_pool_prompt = 0
    total_pool_completion = 0
    total_react_prompt = 0
    total_react_completion = 0
    search_calls = 0

    for turn in range(1, max_turns + 1):
        pool, pg_p, pg_c = await generate_pool(
            model, question, pool_size, history=history_str if history_str else None
        )
        total_pool_prompt += pg_p
        total_pool_completion += pg_c
        selected = select_diverse_queries(pool, 1, method=selection_method, seed=rollout_seed + turn)
        query = selected[0] if selected else None
        if query:
            sr = web_search(query, max_chars=2000)
            history_str = (history_str + "\n" + _format_history(query, sr)).strip()
            search_calls += 1

    # Final answer: one LLM call with full history
    prompt = REACT_PROMPT.format(
        max_turns=max_turns,
        turn=max_turns,
        question=question,
        history=history_str or "(none yet)",
    ) + "\n\nBased on the above, output your final answer in <answer>...</answer>."
    response, p_tok, o_tok = await call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=1024,
        temperature=0.7,
        seed=rollout_seed + 999,
    )
    total_react_prompt += p_tok
    total_react_completion += o_tok
    final_answer = _extract_tag(response, "answer") or response.strip()[:500] or ""

    is_correct = exact_match(final_answer, answer_gt)
    return {
        "answer": final_answer,
        "is_correct": is_correct,
        "turns_used": max_turns,
        "search_calls": search_calls,
        "prompt_tokens": total_react_prompt,
        "completion_tokens": total_react_completion,
        "pool_gen_prompt_tokens": total_pool_prompt,
        "pool_gen_completion_tokens": total_pool_completion,
    }


class DiversityAllMethod(BaseMethod):
    """
    DIVERSITY-ALL: 200 rollouts (8 per question). At every turn: oversample o=64, DPP select 1.
    """

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
    ) -> MethodResult:
        k = self.config.k  # 8 rollouts per question
        pool_size = self.config.o  # 64
        max_turns = self.config.n  # 12
        selection_method = self.config.diversity_method  # dpp_tfidf

        all_rollout_results = []
        for rollout_idx in range(k):
            if pbar:
                pbar.set_description(f"Q{question_id}: DIV-ALL {rollout_idx+1}/{k}")

            r = await run_single_rollout_diversity_all(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                pool_size=pool_size,
                selection_method=selection_method,
                rollout_seed=int(question_id or "0") * 1000 + rollout_idx,
            )
            r["rollout_idx"] = rollout_idx
            all_rollout_results.append(r)

        n_correct = sum(r["is_correct"] for r in all_rollout_results)
        any_correct = n_correct > 0
        first_correct = next((r["answer"] for r in all_rollout_results if r["is_correct"]), "")
        total_pool_p = sum(r["pool_gen_prompt_tokens"] for r in all_rollout_results)
        total_pool_c = sum(r["pool_gen_completion_tokens"] for r in all_rollout_results)

        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer=first_correct or (all_rollout_results[0]["answer"] if all_rollout_results else ""),
            em=any_correct,
            method="diversity_all",
            config_id=self.config.id,
            turns_used=max_turns,
            search_calls_used=sum(r["search_calls"] for r in all_rollout_results),
            total_prompt_tokens=sum(r["prompt_tokens"] for r in all_rollout_results) + total_pool_p,
            total_output_tokens=sum(r["completion_tokens"] for r in all_rollout_results) + total_pool_c,
        )
        result.metadata = {
            "condition": "DIVERSITY-ALL",
            "selection_method": selection_method,
            "pool_size": pool_size,
            "n_rollouts": len(all_rollout_results),
            "n_correct": n_correct,
            "rollout_results": all_rollout_results,
            "pool_gen_prompt_tokens": total_pool_p,
            "pool_gen_completion_tokens": total_pool_c,
        }
        return result
