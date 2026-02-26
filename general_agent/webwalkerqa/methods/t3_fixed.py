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
"""

import asyncio
import re
from typing import Optional

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer, extract_tag, format_history_turn


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

# Provided original prompts (adapted for Spawn-Join T3)
SHORT_ANSWER_PROMPT_BASE = """Your are a research assistant with the ability to perform web searches to answer questions. You can answer a question with many turns of search and reasoning.

Based on the history information, you need to suggest the next action to complete the task. 
You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

IMPORTANT: You must strictly adhere to the following rules:
1. Choose ONLY ONE action from the list below for each response, DO NOT perform more than one action per step.
    2. Follow the exact syntax format for the selected action, DO NOT create or use any actions other than those listed.
    3. **Don't do duplicate search.** Pay attention to the history search results.
    4. You are currently at Turn {turn} out of {max_turns}. You have remaining turns available if you need to perform more searches to gather definitive evidence or verify your findings before answering.

Valid actions:
1. <search> query </search>: search the web for information if you consider you lack some knowledge.
2. <answer> answer </answer>: output the final answer if you consider you are able to answer the question. Be direct, factual, and to the point. No justification is needed.
3. <summary> important parts of the history turns </summary>: summarize the history turns. Reflect the search queries and search results in you history turns, and keep the information you consider important for answering the question and generating your report. Still keep the tag structure, keep search queries between <search> and </search>, and keep search results between <information> and </information>. The history turn information for your subsequent turns will be updated accoring to this summary action.

Output instructions:
First, you should think step-by-step about the question and the history turns.
Then you should choose **ONLY ONE** of the following actions:
- If You want to search, You should put the query between <search> and </search>. 
- If You want to summarize the history turns, You should put the summary between <summary> and </summary>.
- If You want to give the final answer, You should put the answer between <answer> and </answer>.
You can only use ONE action per response.

Format:
Thinking Process: [thinking process]
Action: [action]

Note: text between <information></information> is the search results from search engine after you peruse a search action, **DO NOT** include any information in <information></information> in your search action.

Question: {question}

History Turns: (empty if this is the first turn)
{history}
"""

THREAD_SYSTEM = "You are a focused search thread. Your only job is to find and extract ANY relevant information pertaining the question."

# Refined thread prompt to force diversity
THREAD_PROMPT = """Question: {question}
Parent Coordinator Intent: {parent_query}
History: {history}
Your Specific Strategy: {seed}

Task: Generate a search query that satisfies the Coordinator's intent but strictly follows your Strategy. 
Do not simply repeat the Coordinator's query. Use your strategy to find a unique angle.

Action: <search> query </search>"""

# Cap total accumulated summary text sent to parent (avoids context explosion at k=16)
_MAX_PARENT_CONTEXT_CHARS = 8000


class T3FixedMethod(BaseMethod):
    """
    T³ Fixed: k parallel threads per turn, hand-crafted diversity seeds.

    Spawn-Join Loop:
      1. Spawn: k threads run in parallel → each generates a <search> query + result.
      2. Join: Parent receives all k search/information pairs.
      3. Critic: Parent chooses <summary> (updates History) or <answer>.
    """

    def _generate_seeds(self, question: str, k: int, history: str = "") -> list[str]:
        """
        Return k diversity seed strings.
        T³ Fixed: round-robin from SEED_POOL.
        """
        seeds = [SEED_POOL[i % len(SEED_POOL)] for i in range(k)]
        return seeds

    async def _run_thread(
        self,
        thread_idx: int,
        question: str,
        parent_query: str,
        seed: str,
        history: str,
    ) -> tuple[str, str]:
        """
        One thread (Explorer): generate <search> query → execute search.
        Returns (query, search_result).
        """
        # Step 1: Generate query
        query_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": THREAD_PROMPT.format(
                question=question,
                parent_query=parent_query,
                seed=seed,
                history=history or "(empty)",
            )},
        ]
        query_response, _, _ = await call_llm(
            messages=query_messages,
            model=self.model,
            max_tokens=150,
            temperature=1.0,
        )

        query = extract_tag(query_response, "search") or f"{parent_query} {seed[:20]}"

        # Step 2: Search
        search_result = web_search(query, max_chars=2500)

        return query, search_result

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
        history_str = ""
        final_answer: Optional[str] = None
        total_search_calls = 0

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (critic)")
            turn_log = TurnLog(turn=turn)

            # 1. Critic Phase: Call Parent to decide action
            parent_content = SHORT_ANSWER_PROMPT_BASE.format(
                question=question,
                history=history_str or "(empty)",
                turn=turn,
                max_turns=self.config.n,
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

            self._log(f"Turn {turn}: critic decision ({o_tokens} tokens)")

            # Check for <answer>
            answer = extract_tag(parent_response, "answer")
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                self._log(f"Turn {turn}: answer='{answer[:80]}'")
                break

            # Check for <summary>
            summary = extract_tag(parent_response, "summary")
            if summary:
                self._log(f"Turn {turn}: summary issued, updating history")
                history_str = summary
                result.turns.append(turn_log)
                continue

            # Check for <search> (triggers Spawn-Join)
            parent_query = extract_tag(parent_response, "search")
            if parent_query:
                self._log(f"Turn {turn}: search issued, spawning {k} threads")
                if pbar:
                    pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (spawn)")
                
                # 2. Spawn Phase: k threads in parallel
                # We pass the parent's query intent to help steer the diversity seeds
                seeds = self._generate_seeds(question, k, history_str)
                thread_tasks = [
                    self._run_thread(i, question, parent_query, seeds[i], history_str)
                    for i in range(k)
                ]
                thread_outputs = await asyncio.gather(*thread_tasks)

                queries, search_results = zip(*thread_outputs)
                turn_log.thread_queries = list(queries)
                turn_log.thread_results = [sr[:500] for sr in search_results]
                total_search_calls += k

                # 3. Join Phase: Format all k results into history for the NEXT turn
                new_entries = []
                for i in range(k):
                    new_entries.append(format_history_turn(queries[i], search_results[i][:800]))
                
                turn_results_block = "\n".join(new_entries)
                history_str = (history_str + "\n" + turn_results_block).strip()
            else:
                # No valid action
                self._log(f"Turn {turn}: NO VALID TAG FOUND in decision")
                history_str = (history_str + f"\n[Turn {turn}] No action taken. Please choose <search>, <summary>, or <answer>.").strip()

            result.turns.append(turn_log)
            if pbar:
                pbar.update(1)

        # Fallback answer extraction
        if final_answer is None and result.turns:
            last = result.turns[-1]
            final_answer = extract_tag(last.parent_response, "answer") or _extract_fallback_answer(last.parent_response)

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)
        result.f1 = f1_score(result.final_answer, answer_gt)

        return result


def _extract_fallback_answer(text: str) -> Optional[str]:
    """Last-resort: take the last substantive line if model skipped tags."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        skip_prefixes = ("<search>", "<summary>", "Thinking Process:", "Action:")
        if len(line) < 500 and not any(line.startswith(p) for p in skip_prefixes):
            return line
    return lines[-1][:400] if lines else None
