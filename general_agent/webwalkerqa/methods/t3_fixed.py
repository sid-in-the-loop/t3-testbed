"""
T³ Fixed — Test-Time Threading with hand-crafted diversity seeds.

Design:
  k parallel threads, each with a distinct seed strategy.
  n turns. Each turn:
    1. Parent's global state is shared with threads.
    2. k threads run in parallel: each issues 1 search, produces a summary.
    3. Parent synthesizes all k summaries, outputs <answer> or continues.

Diversity seeds (T³ Fixed, hand-crafted):
  The k thread strategies are drawn from a fixed pool in round-robin order.
  Each strategy biases the thread toward a different search angle.

Extensibility:
  - T³ Dynamic: replace _generate_seeds() with an LLM call
  - T³ DPP: replace _generate_seeds() with DPP-based candidate selection

Fixes applied:
  - Removed "extremely concise (1-5 words)" instruction (kills EM on multi-part/long GT answers)
  - Added language-matching instruction (dataset is 64% Chinese)
  - Fixed PARENT_FORCE_PROMPT: removed literal <answer>YOUR ANSWER HERE</answer> which caused
    extract_answer() to return "YOUR ANSWER HERE" as the final answer
  - Robust global state extraction: multiple parsing strategies + graceful fallback
  - Summary token budget: max(256, t*0.75) instead of t//2 to preserve more signal
  - Parent context cap: limit accumulated summaries to avoid context explosion at k=16
"""

import asyncio
import re
from typing import Optional

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer


# ── Fixed Diversity Seed Pool ─────────────────────────────────────────────────

SEED_POOL = [
    # Direct entity lookup
    "Search for the answer using the most direct, literal phrasing of the question.",
    # Alternative phrasing / aliases
    "Search using alternative names, acronyms, or synonyms for the key entities in the question.",
    # Contextual / relational
    "Search for context: related events, people, or organizations that connect to the answer.",
    # Temporal / historical
    "Search with a focus on dates, timelines, and historical facts related to the question.",
    # Source-based (Wikipedia / official)
    "Search Wikipedia or official organizational sources for definitive facts.",
    # Decomposition — first sub-question
    "Identify the first unknown in the question and search specifically for that.",
    # Decomposition — second sub-question
    "Identify the second unknown in the question (if any) and search specifically for that.",
    # Cross-reference
    "Search for two independent facts that together uniquely identify the answer.",
    # Numeric / statistical
    "Search for numbers, counts, dates, or rankings relevant to the question.",
    # Geographic / institutional
    "Search for geographic locations or institutions central to the question.",
    # Recent / updated information
    "Search for the most recent or updated information about the subject.",
    # Primary source
    "Search for primary sources: official websites, papers, or press releases.",
    # Entity disambiguation
    "If the question involves an ambiguous entity, search for its full context to disambiguate.",
    # Background sweep
    "Search broadly for background on the main topic, then narrow to the specific fact.",
    # Citation-based
    "Search for pages that explicitly state the answer as a known cited fact.",
    # Verification
    "Search to verify or disprove the most likely candidate answer you can infer from context.",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

THREAD_SYSTEM = """\
You are a focused search agent. Your job: generate ONE search query that explores a \
specific angle of the question, then summarize what you find.
Respond in the SAME LANGUAGE as the question.
"""

THREAD_PROMPT = """\
Question: {question}

GLOBAL STATE (what is already known and already searched):
{global_state}

Your search strategy: {seed}

Instructions:
1. Generate ONE specific search query that follows your strategy AND fills a gap in the Global State.
2. DO NOT repeat queries already listed in "Queries Tried".
3. Be specific to avoid broad, redundant results.

Write your query as:
SEARCH: <query here>
"""

THREAD_SUMMARY_PROMPT = """\
Question: {question}
Search query used: {query}
Search results:
{search_results}

Write a brief structured summary:
- What I found: [specific facts from this search]
- What's missing: [what this search did NOT reveal]
- Best partial answer: [your best guess at the answer, or "unknown"]

Respond in the SAME LANGUAGE as the question. Be concise but complete.
"""

PARENT_PROMPT = """\
Question: {question}

CURRENT GLOBAL STATE:
{global_state}

NEW FINDINGS — Turn {turn}/{n_turns} ({k} threads):
{current_summaries}

Tasks:
1. Merge the new findings into the Global State.
2. If you can now answer the question fully, output the answer using the tag below.
3. If not, write brief COORDINATOR GUIDANCE for what to search next.

Answer in the SAME LANGUAGE as the question. Provide a COMPLETE answer covering ALL parts of the question.

Reply in this format:
UPDATED GLOBAL STATE:
- Facts Found: [all confirmed facts so far]
- Queries Tried: [all queries used so far, comma-separated]
- Missing Info: [what is still unknown]

COORDINATOR GUIDANCE: [what to search next turn] OR <answer>COMPLETE ANSWER HERE</answer>
"""

# FIXED: no longer contains literal <answer>YOUR ANSWER HERE</answer> which
# caused extract_answer() to return "YOUR ANSWER HERE" as the final answer.
PARENT_FORCE_PROMPT = """\
Question: {question}

FINAL GLOBAL STATE:
{global_state}

FINAL TURN FINDINGS:
{current_summaries}

This is the last turn. Based on ALL findings, write the best possible answer.
Answer in the SAME LANGUAGE as the question. Cover ALL parts the question asks about.

Wrap your final answer in answer tags. Replace [ANSWER] with the actual answer:
<answer>[ANSWER]</answer>
"""

# Cap total accumulated summary text sent to parent (avoids context explosion at k=16)
_MAX_PARENT_CONTEXT_CHARS = 8000


class T3FixedMethod(BaseMethod):
    """
    T³ Fixed: k parallel threads per turn, hand-crafted diversity seeds.

    Per turn:
      Threads run in parallel → each produces 1 search + summary.
      Parent sees global state + new summaries → updates state or answers.

    Extensibility hook:
      Override _generate_seeds() for T³ Dynamic or T³ DPP.
    """

    @property
    def _summary_tokens(self) -> int:
        """Summary token budget: generous enough to capture multi-part answers."""
        return max(256, int(self.config.t * 0.75))

    def _generate_seeds(self, question: str, k: int, global_state: str = "") -> list[str]:
        """
        Return k diversity seed strings.
        T³ Fixed: round-robin from SEED_POOL.
        Override for T³ Dynamic / T³ DPP.
        """
        return [SEED_POOL[i % len(SEED_POOL)] for i in range(k)]

    async def _run_thread(
        self,
        thread_idx: int,
        question: str,
        seed: str,
        global_state: str,
    ) -> tuple[str, str, str]:
        """
        One thread: generate query → search → summarize.
        Returns (query, search_result, summary).
        """
        # Step 1: Generate query
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
            max_tokens=min(150, self.config.t // 4),
            temperature=0.8,
        )

        query = _extract_search_query(query_response) or f"{question[:80]} {seed[:30]}"

        # Step 2: Search
        search_result = web_search(query, max_chars=2500)

        # Step 3: Summarize
        summary_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": THREAD_SUMMARY_PROMPT.format(
                question=question,
                query=query,
                search_results=search_result[:2000],
            )},
        ]
        summary, _, _ = await call_llm(
            messages=summary_messages,
            model=self.model,
            max_tokens=self._summary_tokens,
            temperature=0.3,
        )

        return query, search_result, summary

    async def run_question(
        self,
        question_id: str,
        question: str,
        answer_gt: str,
        pbar: Optional[object] = None,
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
        global_state = (
            "- Facts Found: None\n"
            "- Queries Tried: None\n"
            "- Missing Info: All details needed to answer the question"
        )
        final_answer: Optional[str] = None
        total_search_calls = 0
        all_summaries_for_parent: list[str] = []  # Accumulated across turns

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (threads)")
            is_last_turn = (turn == self.config.n)
            turn_log = TurnLog(turn=turn)

            # Generate seeds and run k threads in parallel
            seeds = self._generate_seeds(question, k, global_state)
            thread_tasks = [
                self._run_thread(i, question, seeds[i], global_state)
                for i in range(k)
            ]
            thread_outputs = await asyncio.gather(*thread_tasks)

            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (parent)")
                pbar.update(1)

            queries, search_results, summaries = zip(*thread_outputs)
            turn_log.thread_queries = list(queries)
            turn_log.thread_results = [sr[:500] for sr in search_results]
            turn_log.thread_summaries = list(summaries)
            total_search_calls += k

            # Build current summaries block for parent
            current_summaries = "\n\n".join(
                f"[Thread {i+1} | {seeds[i][:50]}]\n{summaries[i]}"
                for i in range(k)
            )

            # Accumulate for parent context; cap total to avoid explosion at k=16
            all_summaries_for_parent.append(f"=== Turn {turn} ===\n{current_summaries}")
            accumulated = "\n\n".join(all_summaries_for_parent)
            if len(accumulated) > _MAX_PARENT_CONTEXT_CHARS:
                # Keep only the most recent turns when context gets too large
                accumulated = "\n\n".join(all_summaries_for_parent[-2:])
                accumulated = "[Earlier turns omitted for brevity]\n\n" + accumulated

            # Parent synthesis
            if is_last_turn:
                parent_content = PARENT_FORCE_PROMPT.format(
                    question=question,
                    global_state=global_state,
                    current_summaries=accumulated,
                )
            else:
                parent_content = PARENT_PROMPT.format(
                    question=question,
                    global_state=global_state,
                    current_summaries=current_summaries,  # Only current turn (state handles history)
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

            self._log(f"Turn {turn}: parent ({o_tokens} tokens)")

            # Check for answer
            answer = extract_answer(parent_response)
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                self._log(f"Turn {turn}: answer='{answer[:80]}'")
                break

            # Update global state for next turn — robust extraction with fallback
            new_state = _extract_global_state(parent_response)
            if new_state:
                global_state = new_state
            else:
                # Fallback: append new queries to existing state so they're not repeated
                new_queries = ", ".join(queries)
                global_state = _append_queries_to_state(global_state, new_queries)

            result.turns.append(turn_log)

        # Fallback answer extraction
        if final_answer is None and result.turns:
            last = result.turns[-1]
            final_answer = (
                extract_answer(last.parent_response)
                or _extract_fallback_from_summaries(last.thread_summaries)
            )

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)
        result.f1 = f1_score(result.final_answer, answer_gt)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_global_state(text: str) -> Optional[str]:
    """
    Extract UPDATED GLOBAL STATE from parent response.
    Tries multiple delimiters for robustness.
    """
    # Primary: look for "UPDATED GLOBAL STATE:" header
    for header in ("UPDATED GLOBAL STATE:", "GLOBAL STATE:", "Updated Global State:"):
        if header in text:
            after = text.split(header, 1)[1]
            # Trim at known end-markers
            for end_marker in ("COORDINATOR GUIDANCE:", "<answer>", "==="):
                if end_marker in after:
                    after = after.split(end_marker, 1)[0]
            state = after.strip()
            if len(state) > 20:  # Sanity: must have some content
                return state
    return None


def _append_queries_to_state(state: str, new_queries: str) -> str:
    """Append new queries to Queries Tried line in state (fallback when extraction fails)."""
    if "Queries Tried:" in state:
        lines = state.split("\n")
        for i, line in enumerate(lines):
            if "Queries Tried:" in line:
                if "None" in line:
                    lines[i] = f"- Queries Tried: {new_queries}"
                else:
                    lines[i] = line.rstrip() + f", {new_queries}"
                break
        return "\n".join(lines)
    return state + f"\n- Queries Tried (appended): {new_queries}"


def _extract_search_query(text: str) -> Optional[str]:
    """Extract SEARCH: query from thread response."""
    m = re.search(r"SEARCH:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_fallback_from_summaries(summaries: list[str]) -> Optional[str]:
    """Pull best partial answer from thread summaries when parent didn't use <answer>."""
    for summary in summaries:
        m = re.search(r"Best partial answer:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            if answer.lower() not in ("unknown", "none", "n/a", "unclear", ""):
                return answer
    return None
