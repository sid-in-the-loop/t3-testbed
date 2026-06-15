import asyncio
import hashlib
import re
from typing import Optional, List, Tuple, Dict, Any


SEED_MAX = 2**32


def _seed_from_question_id(question_id: str) -> int:
    """Deterministic integer seed from any question ID (numeric or string e.g. 'hotpotqa-7059')."""
    s = str(question_id or "0")
    if s.isdigit():
        return int(s)
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:8], 16)


def _safe_rollout_seed(question_id: str, run_seed: int, rollout_idx: int) -> int:
    """Rollout seed in [0, SEED_MAX-1] for API compatibility."""
    raw = _seed_from_question_id(question_id) * 1000 + run_seed * 100 + rollout_idx
    return int(raw) % SEED_MAX

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match
from .base import BaseMethod, MethodResult, TurnLog, extract_answer
from .utils import select_diverse_queries

_DEFAULT_MAX_TOKENS = 2048

_PROMPT_STYLE = "react_simple"


def set_prompt_style(style: str) -> None:
    """Choose prompt template. Called once by run_main_table from the CLI."""
    global _PROMPT_STYLE
    assert style in ("react_simple", "web_reasoning"), f"bad prompt style: {style}"
    _PROMPT_STYLE = style


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
Each query should approach the question from a different angle, specifically targeting different constraints or components of the question.
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


def _extract_answer_candidate(text: str, is_last_turn: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract answer from a turn response. Returns (tagged_answer, fallback_candidate).
    - tagged_answer: from <answer>...</answer> if present.
    - fallback_candidate: best-effort extraction when no tag (for every-turn extraction).
    """
    tagged = _extract_tag(text, "answer")
    if tagged:
        return (tagged, tagged)

    text = (text or "").strip()
    if not text:
        return (None, None)

    candidate = None
    for pattern in [
        r"(?:answer is|answer:)\s*[:\-]?\s*(.+?)(?:\.|$)",
        r"(?:thus|therefore|so),?\s*(?:the answer is\s+)?(.+?)(?:\.|$)",
        r"(?:final answer|conclusion)\s*[:\-]?\s*(.+?)(?:\.|$)",
    ]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 300:
                candidate = candidate[:300].rsplit(" ", 1)[0] if " " in candidate[:300] else candidate[:300]
            if candidate:
                break

    if not candidate and _extract_tag(text, "thought"):
        thought = _extract_tag(text, "thought")
        if thought:
            last_sentence = thought.strip().split(".")[-1].strip()
            if last_sentence and len(last_sentence) < 200:
                candidate = last_sentence

    if not candidate and is_last_turn:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("<")]
        if lines:
            candidate = lines[-1][:300] if lines[-1] else None
        if not candidate:
            candidate = text[:300].strip()

    if candidate:
        candidate = candidate.strip().strip(".").strip()
    return (None, candidate if candidate else None)


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


SEARCH_R1_PROMPT = """\
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.
For example, <answer> Beijing </answer>.

Question: {question}

{history}

Your response (use <think>, then either <search> or <answer>):"""


WEB_REASONING_PROMPT = """\
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
3. <summary> important parts of the history </summary> — compress the history. Your next turn's history will be replaced with this summary.

Format:
<think> your thinking process </think>
[one of <search>...</search>, <summary>...</summary>, <answer>...</answer>]

Example:
<think> I need to know X, so I should search for it. </think>
<search> X in 2024 </search>

Note: text inside <information></information> is the search result — do NOT echo it back in your output.

Question: {question}

History Turns:
{history}"""


def get_prompt_for_question(question_id: str, question: str, turn: int, max_turns: int, history: str) -> str:
    """Select and format the prompt. Style is module-level (_PROMPT_STYLE); dataset source is fallback routing."""
    if _PROMPT_STYLE == "web_reasoning":
        return WEB_REASONING_PROMPT.format(
            question=question,
            history=history or "(empty, this is the first turn)",
        )

    source = str(question_id).split('-')[0].lower()
    if source in ["hotpotqa", "2wikimultihopqa"]:
        return SEARCH_R1_PROMPT.format(
            question=question,
            history=history or "(No information yet)"
        )
    return REACT_PROMPT.format(
        max_turns=max_turns,
        turn=turn,
        question=question,
        history=history or "(none yet)",
    )


def _format_history_r1(query: str, result: str) -> str:
    """Format history for Search-R1 style (<search> and <information> tags)."""
    return f"<search>{query}</search>\n<information>{result}</information>"


def _format_history_react(query: str, result: str) -> str:
    """Format history for ReAct style (<search> and <result> tags)."""
    return f"<search>{query}</search>\n<result>{result}</result>"


def format_history_by_source(question_id: str, query: str, result: str) -> str:
    """Choose history formatting based on prompt style + source."""
    if _PROMPT_STYLE == "web_reasoning":
        return _format_history_r1(query, result)
    source = str(question_id).split('-')[0].lower()
    if source in ["hotpotqa", "2wikimultihopqa"]:
        return _format_history_r1(query, result)
    return _format_history_react(query, result)


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
    try:
        text, p_tok, o_tok = await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=1.0,
        )
    except Exception:
        text, p_tok, o_tok = "", 0, 0
    queries = _parse_pool_response(text, o)
    while len(queries) < o:
        queries.append(f"{question[:50]} variant {len(queries)}")
    return queries[:o], p_tok, o_tok


def _jaccard_sim_tokens(a: str, b: str) -> float:
    """Jaccard similarity of word-token sets. Used for intra-thread diversity."""
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _pick_most_diverse_from_pool(pool: List[str], prior: List[str]) -> str:
    """Greedy: pick the pool query with largest min-distance (1 - Jaccard) to any prior query."""
    if not pool:
        return ""
    if not prior:
        return pool[0]
    best, best_score = pool[0], -1.0
    for cand in pool:
        max_sim = max(_jaccard_sim_tokens(cand, p) for p in prior)
        score = 1.0 - max_sim
        if score > best_score:
            best_score = score
            best = cand
    return best


async def run_single_rollout(
    model: str,
    question: str,
    answer_gt: str,
    max_turns: int,
    initial_query: Optional[str] = None,
    rollout_seed: int = 0,
    question_id: Optional[str] = None,
    react_temp_first: float = 1.0,
    react_temp_rest: float = 0.7,
    oversample_until_turn: int = 1,
    oversample_pool_size: int = 16,
) -> Dict[str, Any]:
    """
    Run one complete rollout (T turns → 1 answer).

    Args:
        initial_query: If provided, inject as turn-1 search query (skip turn-1 LLM call).
                      If None, the rollout generates its own turn-1 query.
        oversample_until_turn: For turns 2..N (inclusive), replace the LLM's chosen search
                              query with a pool-generated one, picking the query most
                              diverse (greedy-Jaccard) from prior queries in this thread.
                              N=1 (default) = current behavior (pool only at turn 1).
        oversample_pool_size: Number of candidate queries to generate per oversampled turn.

    Returns:
        Dict with answer, is_correct, turns_used, search_calls, tokens.
    """
    history_str = ""
    final_answer = None
    last_answer_candidate = None
    turn_logs = []
    full_responses = []
    total_prompt = 0
    total_completion = 0
    search_calls = 0
    qid = question_id or ""
    prior_queries: List[str] = []

    for turn in range(1, max_turns + 1):
        if turn == 1 and initial_query is not None:
            sr = await web_search(initial_query, max_chars=4000)
            history_str = format_history_by_source(qid, initial_query, sr)
            search_calls += 1
            prior_queries.append(initial_query)
            turn_logs.append({
                "turn": 1,
                "query": initial_query,
                "injected": True,
                "search_result": sr
            })
            continue

        prompt = get_prompt_for_question(qid, question, turn, max_turns, history_str)
        if turn == 1 and initial_query is None:
            temp = react_temp_first
        else:
            temp = react_temp_rest
        try:
            response, p_tok, o_tok = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=temp,
                seed=(rollout_seed + turn) % SEED_MAX,
            )
        except Exception as e:
            turn_logs.append({"turn": turn, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            break
        total_prompt += p_tok
        total_completion += o_tok
        full_responses.append({
            "turn": turn,
            "prompt": prompt,
            "response": response,
            "tokens": {"prompt": p_tok, "completion": o_tok},
            "temp": temp
        })

        is_last = turn == max_turns
        tagged_answer, candidate = _extract_answer_candidate(response, is_last_turn=is_last)
        if candidate:
            last_answer_candidate = candidate

        if tagged_answer:
            final_answer = tagged_answer
            turn_logs.append({
                "turn": turn, 
                "answer": final_answer, 
                "tagged_answer": tagged_answer,
                "response": response
            })
            break

        summary = _extract_tag(response, "summary")
        if summary and not _extract_tag(response, "search") and not tagged_answer:
            history_str = f"<summary>{summary}</summary>"
            turn_logs.append({
                "turn": turn,
                "summary": summary,
                "response": response,
            })
            continue

        query = _extract_tag(response, "search")
        if query:
            overridden = False
            if turn > 1 and turn <= oversample_until_turn:
                try:
                    pool, pp_tok, po_tok = await generate_pool(
                        model, question, oversample_pool_size, history=history_str
                    )
                    total_prompt += pp_tok
                    total_completion += po_tok
                    picked = _pick_most_diverse_from_pool(pool, prior_queries)
                    if picked:
                        query = picked
                        overridden = True
                except Exception:
                    pass
            sr = await web_search(query, max_chars=4000)
            new_history = format_history_by_source(qid, query, sr)
            history_str = (history_str + "\n" + new_history).strip()
            search_calls += 1
            prior_queries.append(query)
            turn_logs.append({
                "turn": turn,
                "query": query,
                "search_result": sr,
                "response": response,
                "oversample_override": overridden,
            })
        else:
            if turn == max_turns and last_answer_candidate:
                final_answer = last_answer_candidate
            elif turn == max_turns:
                final_answer = response.strip()[:500] if response else None
            turn_logs.append({"turn": turn, "no_action": True, "response": response})
            break

    if final_answer is None:
        final_answer = last_answer_candidate or ""

    is_correct = exact_match(final_answer or "", answer_gt)

    return {
        "answer": final_answer or "",
        "is_correct": is_correct,
        "turns_used": len(turn_logs),
        "search_calls": search_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "turn_logs": turn_logs,
        "full_responses": full_responses,
    }


class SequentialMethod(BaseMethod):

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
        run_seed: int = 0,
    ) -> MethodResult:
        k = self.config.k
        max_turns = self.config.n

        if pbar:
            pbar.set_description(f"Q{question_id}: SEQUENTIAL")

        tasks = [
            run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=None,
                rollout_seed=_safe_rollout_seed(question_id, run_seed, i),
                question_id=question_id,
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


class DiversityParallelMethod(BaseMethod):

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
        run_seed: int = 0,
    ) -> MethodResult:
        k = self.config.k
        pool_size = self.config.o
        max_turns = self.config.n
        selection_method = self.config.diversity_method

        if pbar:
            pbar.set_description(f"Q{question_id}: DIV-PAR o={pool_size} {selection_method}")

        pool, pool_prompt, pool_completion = await generate_pool(
            self.model, question, pool_size
        )
        seeds = select_diverse_queries(
            pool, k, method=selection_method, seed=_safe_rollout_seed(question_id, run_seed, 0)
        )

        tasks = [
            run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=seeds[i] if i < len(seeds) else None,
                rollout_seed=_safe_rollout_seed(question_id, run_seed, i),
                question_id=question_id,
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


class Diversity1Method(BaseMethod):

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
        run_seed: int = 0,
    ) -> MethodResult:
        k = self.config.k
        pool_size = self.config.o
        max_turns = self.config.n
        selection_method = self.config.diversity_method

        all_rollout_results = []
        total_pool_prompt = 0
        total_pool_completion = 0

        for rollout_idx in range(k):
            if pbar:
                pbar.set_description(f"Q{question_id}: DIV-1 {rollout_idx+1}/{k} o={pool_size}")

            pool, pg_p, pg_c = await generate_pool(self.model, question, pool_size)
            total_pool_prompt += pg_p
            total_pool_completion += pg_c
            selected = select_diverse_queries(pool, 1, method=selection_method, seed=_safe_rollout_seed(question_id, run_seed, rollout_idx))
            turn1_query = selected[0] if selected else None

            r = await run_single_rollout(
                model=self.model,
                question=question,
                answer_gt=answer_gt,
                max_turns=max_turns,
                initial_query=turn1_query,
                rollout_seed=_safe_rollout_seed(question_id, run_seed, rollout_idx),
                question_id=question_id,
            )
            r["rollout_idx"] = rollout_idx
            r["injected_query"] = turn1_query
            all_rollout_results.append(r)

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


async def run_single_rollout_diversity_all(
    model: str,
    question: str,
    answer_gt: str,
    max_turns: int,
    pool_size: int,
    selection_method: str,
    rollout_seed: int,
    question_id: str,
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
    turn_logs = []
    full_responses = []

    for turn in range(1, max_turns + 1):
        pool, pg_p, pg_c = await generate_pool(
            model, question, pool_size, history=history_str if history_str else None
        )
        total_pool_prompt += pg_p
        total_pool_completion += pg_c
        selected = select_diverse_queries(pool, 1, method=selection_method, seed=(rollout_seed + turn) % SEED_MAX)
        query = selected[0] if selected else None
        if query:
            sr = await web_search(query, max_chars=4000)
            history_str = (history_str + "\n" + format_history_by_source(question_id, query, sr)).strip()
            search_calls += 1
            turn_logs.append({
                "turn": turn,
                "query": query,
                "search_result": sr
            })

    prompt = get_prompt_for_question(question_id, question, max_turns, max_turns, history_str)
    if "answer" not in prompt.lower():
         prompt += "\n\nBased on the above, output your final answer in <answer>...</answer>."

    try:
        response, p_tok, o_tok = await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=0.7,
            seed=(rollout_seed + 999) % SEED_MAX,
        )
    except Exception:
        response = ""
        p_tok, o_tok = 0, 0
    total_react_prompt += p_tok
    total_react_completion += o_tok
    full_responses.append({
        "type": "final_answer",
        "prompt": prompt,
        "response": response,
        "tokens": {"prompt": p_tok, "completion": o_tok}
    })

    tagged, candidate = _extract_answer_candidate(response, is_last_turn=True)
    final_answer = (tagged or candidate or response.strip()[:500] or "").strip()

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
        "turn_logs": turn_logs,
        "full_responses": full_responses,
    }


class DiversityAllMethod(BaseMethod):

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
        run_seed: int = 0,
    ) -> MethodResult:
        k = self.config.k
        pool_size = self.config.o
        max_turns = self.config.n
        selection_method = self.config.diversity_method

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
                rollout_seed=_safe_rollout_seed(question_id, run_seed, rollout_idx),
                question_id=question_id,
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
