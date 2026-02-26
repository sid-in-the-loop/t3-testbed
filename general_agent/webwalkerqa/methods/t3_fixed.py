"""
T³ Fixed — Test-Time Threading with hand-crafted diversity seeds.

Design:
  k parallel threads, each with a distinct seed strategy.
  n turns. Each turn:
    1. Parent generates k seed prompts (diversity enforced by strategy).
    2. k threads run in parallel: each issues 1 search, produces a summary.
    3. Parent synthesizes all k summaries, outputs <answer> or continues.

Diversity seeds (T³ Fixed, hand-crafted):
  The k thread strategies are drawn from a fixed pool in round-robin order.
  Each strategy biases the thread toward a different search angle.

Extensibility:
  - T³ Dynamic: replace _generate_seeds() with an LLM call
  - T³ DPP: replace _generate_seeds() with DPP-based candidate selection
"""

import asyncio
import re
from typing import Optional

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer


# ── Fixed Diversity Seed Pool ─────────────────────────────────────────────────
# Each seed is a string that describes a search strategy.
# They're injected into the thread prompt to steer it toward a distinct region.

SEED_POOL = [
    # Direct entity lookup
    "Search for the answer using the most direct, literal phrasing of the question.",
    # Alternative phrasing / aliases
    "Search using alternative names, acronyms, or synonyms for the key entities in the question.",
    # Contextual / relational
    "Search for context around the question: related events, people, or places that connect to the answer.",
    # Temporal / historical
    "Search with a focus on dates, timelines, and historical facts related to the question.",
    # Source-based
    "Search Wikipedia, official records, or authoritative sources for definitive facts.",
    # Decomposition
    "Break the question into sub-questions and search for each part separately.",
    # Contrastive
    "Search for what the answer is NOT, then narrow down to the correct answer.",
    # Cross-reference
    "Search for two or more independent facts that together uniquely identify the answer.",
    # Numeric / statistical
    "Search for numbers, statistics, rankings, or measurements related to the question.",
    # Geographic / institutional
    "Search for geographic locations, organizations, or institutions central to the question.",
    # Recent / updated
    "Search for the most recent or updated information about the subject of the question.",
    # Broad background
    "Search for background information to understand the topic before narrowing to the specific answer.",
    # Primary source
    "Search for primary sources: official websites, government records, academic papers.",
    # Common knowledge verification
    "Start from well-known facts about this topic and verify or refine the specific answer.",
    # Citation-based
    "Search for articles or pages that explicitly cite the answer as a known fact.",
    # Entity disambiguation
    "If the question is ambiguous, identify all possible interpretations and search for each.",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

THREAD_SYSTEM = """\
You are a focused search thread. Your job is to find information relevant to a question
using ONE specific search query. Be concise and precise.
"""

THREAD_PROMPT = """\
Question: {question}

CURRENT GLOBAL STATE:
{global_state}

Your search strategy: {seed}

Task:
1. Generate ONE specific search query that follows your strategy AND addresses what is still missing.
2. DO NOT repeat queries that have already been tried (see "Queries Tried" in Global State).
3. Be as specific as possible to avoid broad, redundant results.

Write your search query exactly as:
SEARCH: <your query here>
"""

THREAD_SUMMARY_PROMPT = """\
Question: {question}
You searched for: {query}
Search results:
{search_results}

Write a brief structured summary (max {max_tokens} tokens):
- What I found: [list key facts discovered from THIS search]
- What's missing: [be specific about what is still unknown or ambiguous]
- Best partial answer: [concise guess or "unknown"]
"""

PARENT_PROMPT = """\
Question: {question}

CURRENT GLOBAL STATE:
{global_state}

NEW FINDINGS FROM {k} PARALLEL THREADS (Turn {turn}/{n_turns}):
{current_summaries}

Task:
1. Update the GLOBAL STATE by merging the new findings into "Facts Found" and "Queries Tried".
2. If the question can be answered from "Facts Found", output the answer between <answer> and </answer> tags.
3. If not, provide "COORDINATOR GUIDANCE" for the next turn.

The answer must be extremely concise (1-5 words, e.g., a name, date, or fact) without conversational filler.

Respond in this exact format:
UPDATED GLOBAL STATE:
- Facts Found: [distilled bullet points of all known facts]
- Queries Tried: [list of all search terms used so far]
- Missing Info: [specific details still needed]

COORDINATOR GUIDANCE: [specific instructions for next turn] (OR <answer>YOUR ANSWER HERE</answer>)
"""

PARENT_FORCE_PROMPT = """\
Question: {question}

FINAL GLOBAL STATE:
{global_state}

NEW FINDINGS FROM FINAL TURN:
{current_summaries}

This is the FINAL turn. You MUST provide a final answer now based on all findings.
<answer>YOUR ANSWER HERE</answer>

Important: The answer must be extremely concise (1-5 words).
"""


class T3FixedMethod(BaseMethod):
    """
    T³ Fixed: k parallel threads per turn, hand-crafted diversity seeds.

    Architecture per turn:
      [Parent] generates seed contexts for k threads
      [Thread 1..k] (parallel): 1 search + summary
      [Parent] synthesizes → answer or continue

    Extensibility hook:
      Override _generate_seeds() for T³ Dynamic or T³ DPP.
    """

    def _generate_seeds(self, question: str, k: int, prior_context: str = "") -> list[str]:
        """
        Return k diversity seed strings for the threads.

        T³ Fixed: selects k seeds from SEED_POOL in order.
        T³ Dynamic would call an LLM here.
        T³ DPP would run a DPP selector here.
        """
        seeds = []
        for i in range(k):
            seeds.append(SEED_POOL[i % len(SEED_POOL)])
        return seeds

    async def _run_thread(
        self,
        thread_idx: int,
        question: str,
        seed: str,
        global_state: str,
    ) -> tuple[str, str, str]:
        """
        Run one thread: generate query → search → summarize.

        Returns:
            (query, search_result, summary)
        """
        # Step 1: Generate search query
        query_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": THREAD_PROMPT.format(
                question=question,
                seed=seed,
                global_state=global_state,
            )},
        ]
        query_response, _, _ = await call_llm(
            messages=query_messages,
            model=self.model,
            max_tokens=min(200, self.config.t // 4),  # Small budget for query generation
            temperature=0.8,  # Slightly higher for diversity
            seed=None,
        )

        # Extract SEARCH: query
        query = _extract_search_query(query_response) or f"{question} (thread {thread_idx})"

        # Step 2: Execute search
        search_result = web_search(query)

        # Step 3: Summarize findings
        summary_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": THREAD_SUMMARY_PROMPT.format(
                question=question,
                query=query,
                search_results=search_result[:2000],  # Truncate for token budget
                max_tokens=self.config.summary_tokens,
            )},
        ]
        summary, _, _ = await call_llm(
            messages=summary_messages,
            model=self.model,
            max_tokens=self.config.summary_tokens,
            temperature=0.3,
        )

        return query, search_result, summary

    async def run_question(
        self, 
        question_id: str, 
        question: str, 
        answer_gt: str,
        pbar: Optional[any] = None,
    ) -> MethodResult:
        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer="",
            method="t3_fixed",
            config_id=self.config.id,
        )

        k = self.config.k
        global_state = "- Facts Found: None\n- Queries Tried: None\n- Missing Info: All details"
        final_answer: Optional[str] = None
        total_search_calls = 0

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (threads running...)")
            is_last_turn = (turn == self.config.n)
            turn_log = TurnLog(turn=turn)

            # Generate k diverse seed strategies
            seeds = self._generate_seeds(question, k, global_state)

            # Run k threads in parallel
            self._log(f"Turn {turn}: spawning {k} threads")
            thread_tasks = [
                self._run_thread(i, question, seeds[i], global_state)
                for i in range(k)
            ]
            
            thread_outputs = await asyncio.gather(*thread_tasks)
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (synthesizing...)")
                pbar.update(1)

            queries, search_results, summaries = zip(*thread_outputs)
            turn_log.thread_queries = list(queries)
            turn_log.thread_results = [sr[:500] for sr in search_results]
            turn_log.thread_summaries = list(summaries)
            total_search_calls += k

            # Build current summaries block
            current_summaries = "\n\n".join(
                f"[Thread {i+1} | {seeds[i][:40]}...]\n{summaries[i]}"
                for i in range(k)
            )

            # Parent synthesis
            if is_last_turn:
                parent_content = PARENT_FORCE_PROMPT.format(
                    question=question,
                    global_state=global_state,
                    current_summaries=current_summaries,
                )
            else:
                parent_content = PARENT_PROMPT.format(
                    question=question,
                    global_state=global_state,
                    current_summaries=current_summaries,
                    k=k,
                    turn=turn,
                    n_turns=self.config.n,
                )

            parent_response, p_tokens, o_tokens = await call_llm(
                messages=[{"role": "user", "content": parent_content}],
                model=self.model,
                max_tokens=self.config.t,
                temperature=0.3,
            )

            turn_log.parent_response = parent_response
            turn_log.prompt_tokens = p_tokens
            turn_log.output_tokens = o_tokens

            self._log(f"Turn {turn}: parent response ({o_tokens} tokens)")

            # Check if parent found an answer
            answer = extract_answer(parent_response)
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                self._log(f"Turn {turn}: answer found: '{answer[:60]}'")
                break

            # Update global state for next turn
            new_state = _extract_global_state(parent_response)
            if new_state:
                global_state = new_state

            result.turns.append(turn_log)

        # Fallback: extract from last parent response
        if final_answer is None and result.turns:
            last_parent = result.turns[-1].parent_response
            final_answer = extract_answer(last_parent) or _extract_fallback_from_summaries(
                result.turns[-1].thread_summaries
            )

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)
        result.f1 = f1_score(result.final_answer, answer_gt)

        return result


def _extract_global_state(text: str) -> Optional[str]:
    """Extract UPDATED GLOBAL STATE from parent response."""
    if "UPDATED GLOBAL STATE:" in text:
        parts = text.split("UPDATED GLOBAL STATE:")
        if len(parts) > 1:
            state = parts[1].split("COORDINATOR GUIDANCE:")[0].strip()
            return state
    return None


def _extract_search_query(text: str) -> Optional[str]:
    """Extract SEARCH: query from model response."""
    m = re.search(r"SEARCH:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_fallback_from_summaries(summaries: list[str]) -> Optional[str]:
    """Try to extract a best-guess answer from thread summaries."""
    for summary in summaries:
        # Look for "Best partial answer:" line
        m = re.search(r"Best partial answer:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            if answer.lower() not in ("unknown", "none", "n/a", "unclear", ""):
                return answer
    return None
