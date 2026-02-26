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

Fixes applied:
  - Removed "extremely concise (1-5 words)" instruction (kills EM on multi-part/long GT answers)
  - Added language-matching instruction (dataset is 64% Chinese; model must answer in question's language)
  - History compression: only key facts are appended (not raw reasoning+search dumps),
    preventing context explosion at high t budgets
"""

import re
import asyncio
from typing import Optional

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant that answers questions by searching the web.

You have {n_turns} turns to find the answer.

At each turn, reason about what you know and what you still need to find, \
then search for missing information. You may search multiple times per turn.

When you are confident in the answer, output it as:
<answer>YOUR ANSWER HERE</answer>

Guidelines:
- Answer in the SAME LANGUAGE as the question.
- Provide a complete answer that addresses ALL parts of the question.
- Match the level of detail the question requires (a date, a name, a sentence, a list, etc.).
- Do NOT repeat searches you have already done. Diversify your queries each turn.
- If you cannot find the answer after all turns, give your best guess.
"""

TURN_PROMPT = """\
Turn {turn}/{n_turns}

Question: {question}

KNOWN FACTS SO FAR:
{history}

What is still unknown? Decide: search for more information or provide a final answer.

To search, write on its own line:
SEARCH: <specific query here>

You may search multiple times this turn. When you have enough information, write:
<answer>YOUR COMPLETE ANSWER HERE</answer>
"""

FORCE_ANSWER_PROMPT = """\
FINAL TURN — you MUST provide an answer now.

Question: {question}

KNOWN FACTS:
{history}

Based on everything found, write your best answer:
<answer>YOUR COMPLETE ANSWER HERE</answer>

Answer in the same language as the question. Cover all parts the question asks about.
"""

# Compress search results to just the most relevant content
_MAX_SEARCH_CHARS = 2000
SEARCH_PATTERN = re.compile(r"SEARCH:\s*(.+?)(?=\nSEARCH:|\n<answer>|<answer>|$)", re.IGNORECASE | re.DOTALL)


class S1Method(BaseMethod):
    """
    Sequential scaling: single thread, growing token budget per turn.

    Implements a ReAct-style loop:
      1. Prompt model with question + compressed knowledge history
      2. Model reasons and issues SEARCH: queries
      3. Execute searches, distil key facts into history
      4. Repeat until <answer> found or n turns exhausted

    Key design: history contains only distilled facts (not raw model reasoning
    or full search dumps) so context stays manageable at high token budgets.
    """

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
            method="s1",
            config_id=self.config.id,
        )

        # history_parts accumulates distilled facts (not raw outputs)
        history_parts: list[str] = []
        final_answer: Optional[str] = None
        total_search_calls = 0

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n}")
                pbar.update(1)

            is_last_turn = (turn == self.config.n)
            turn_log = TurnLog(turn=turn)

            history_str = "\n".join(history_parts) if history_parts else "None yet."

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

            # Check for answer first
            answer = extract_answer(response_text)
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                self._log(f"Turn {turn}: answer found: '{answer[:80]}'")
                break

            # Extract and execute search queries
            raw_queries = SEARCH_PATTERN.findall(response_text)
            queries = [q.strip().splitlines()[0].strip() for q in raw_queries if q.strip()]
            turn_log.search_queries = queries

            if queries:
                # Run all searches for this turn, collect results
                search_snippets = []
                for query in queries:
                    self._log(f"Turn {turn}: searching '{query}'")
                    sr = web_search(query, max_chars=_MAX_SEARCH_CHARS)
                    total_search_calls += 1
                    search_snippets.append(f"• [{query}]: {sr[:600]}")

                # Append ONLY the distilled search results to history (not raw reasoning).
                # This keeps context size proportional to search results, not token budget.
                turn_summary = f"[Turn {turn}]\n" + "\n".join(search_snippets)
                history_parts.append(turn_summary)
            else:
                # No search issued — record any reasoning insight briefly
                insight = response_text.strip()[:300]
                history_parts.append(f"[Turn {turn} note] {insight}")

            result.turns.append(turn_log)

        # Fallback: extract from last response if no <answer> tag was found
        if final_answer is None and result.turns:
            last_reasoning = result.turns[-1].reasoning
            final_answer = extract_answer(last_reasoning) or _extract_fallback_answer(last_reasoning)

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)
        result.f1 = f1_score(result.final_answer, answer_gt)

        return result


def _extract_fallback_answer(text: str) -> Optional[str]:
    """Last-resort: take the last substantive line if model skipped <answer> tags."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        skip_prefixes = ("SEARCH:", "Query:", "Turn ", "Based on", "I need", "Let me")
        if len(line) < 500 and not any(line.startswith(p) for p in skip_prefixes):
            return line
    return lines[-1][:400] if lines else None
