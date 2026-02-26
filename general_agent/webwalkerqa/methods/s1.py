"""
s1 — Sequential Scaling Method.

Design:
  - 1 thread, n=6 turns
  - Each turn: model gets t tokens budget, can issue multiple web_search calls
  - Model outputs <answer>...</answer> when confident
  - At turn n (budget exhausted), force final answer

Token budget:
  The model receives max_tokens=t per turn.
  At higher t (B1=4096, C1=8192), it naturally issues more searches per turn
  because the reasoning chain is longer.

Search constraint:
  We allow unlimited search calls per turn but track the count.
  (In theory bounded by t/avg_turn_cost, matching the compute budget.)
"""

import re
import asyncio
from typing import Optional

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match
from .base import BaseMethod, MethodResult, TurnLog, extract_answer


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant that answers questions by searching the web.

You have {n_turns} turns to find the answer. Use them wisely.

At each turn, reason about what you know and what you still need to find.
Then search for missing information. You may search multiple times per turn.

When you are confident in the answer, output it as:
<answer>YOUR ANSWER HERE</answer>

Guidelines:
- Be specific and concise in your final answer (a name, date, place, number, etc.)
- If you cannot find the answer after all turns, give your best guess
- Do NOT repeat searches you've already done
- Build on findings from previous turns
"""

TURN_PROMPT = """\
Turn {turn}/{n_turns}

Question: {question}

{history}

What do you know so far? What do you still need to find?
Reason step by step, then decide whether to search or answer.

If you want to search, write:
SEARCH: <your query here>

You may search multiple times. When done searching, write:
<answer>YOUR ANSWER HERE</answer>

If this is your final turn, you MUST provide an answer.
"""

FORCE_ANSWER_PROMPT = """\
This is your FINAL turn. You MUST provide an answer now.

Question: {question}

Based on everything you've found so far:
{history}

Provide your best answer in the format:
<answer>YOUR ANSWER HERE</answer>
"""

SEARCH_PATTERN = re.compile(r"SEARCH:\s*(.+?)(?=\n|SEARCH:|<answer>|$)", re.IGNORECASE)


class S1Method(BaseMethod):
    """
    Sequential scaling: single thread, growing token budget per turn.

    Implements a simple ReAct-style loop:
      1. Prompt model with question + history
      2. Model reasons and issues SEARCH: queries
      3. Execute searches, append results to history
      4. Repeat until <answer> found or n turns exhausted
    """

    async def run_question(self, question_id: str, question: str, answer_gt: str) -> MethodResult:
        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer="",
            method="s1",
            config_id=self.config.id,
        )

        history_parts: list[str] = []  # Accumulated search findings
        final_answer: Optional[str] = None
        total_search_calls = 0

        for turn in range(1, self.config.n + 1):
            is_last_turn = (turn == self.config.n)
            turn_log = TurnLog(turn=turn)

            history_str = "\n\n".join(history_parts) if history_parts else "No findings yet."

            # Build prompt
            if is_last_turn and final_answer is None:
                user_content = FORCE_ANSWER_PROMPT.format(
                    question=question,
                    history=history_str,
                )
            else:
                user_content = TURN_PROMPT.format(
                    turn=turn,
                    n_turns=self.config.n,
                    question=question,
                    history=history_str,
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(n_turns=self.config.n)},
                {"role": "user", "content": user_content},
            ]

            # LLM call with token budget t
            response_text, p_tokens, o_tokens = await call_llm(
                messages=messages,
                model=self.model,
                max_tokens=self.config.t,
                temperature=0.7,
            )
            turn_log.reasoning = response_text
            turn_log.prompt_tokens = p_tokens
            turn_log.output_tokens = o_tokens

            self._log(f"Turn {turn}: {o_tokens} output tokens")

            # Check if model provided an answer
            answer = extract_answer(response_text)
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                self._log(f"Turn {turn}: answer found: '{answer[:60]}'")
                break

            # Extract and execute search queries
            queries = SEARCH_PATTERN.findall(response_text)
            queries = [q.strip() for q in queries if q.strip()]
            turn_log.search_queries = queries

            if queries:
                search_results = []
                for query in queries:
                    self._log(f"Turn {turn}: searching '{query}'")
                    sr = web_search(query)
                    total_search_calls += 1
                    search_results.append(f"Query: {query}\n{sr}")

                # Append to history
                turn_summary = f"[Turn {turn} searches]\n" + "\n---\n".join(search_results)
                history_parts.append(turn_summary)
                turn_log.search_queries = queries
            else:
                # Model didn't search or answer — extract partial answer if any
                history_parts.append(f"[Turn {turn} reasoning]\n{response_text[:500]}")

            result.turns.append(turn_log)

        # If no answer found after all turns, try to extract from last response
        if final_answer is None and result.turns:
            last_reasoning = result.turns[-1].reasoning
            # Try to find any answer-like content
            final_answer = extract_answer(last_reasoning) or _extract_fallback_answer(last_reasoning)

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)

        return result


def _extract_fallback_answer(text: str) -> Optional[str]:
    """
    Fallback: try to find a short answer in the last paragraph.
    Used when the model didn't use <answer> tags.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None
    # Heuristic: last non-empty line that looks like a short answer
    for line in reversed(lines):
        if len(line) < 200 and not line.startswith(("SEARCH:", "Query:", "Turn", "Based")):
            return line
    return lines[-1][:200] if lines else None
