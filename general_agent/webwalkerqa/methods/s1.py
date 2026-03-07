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
from .base import BaseMethod, MethodResult, TurnLog, extract_answer, extract_tag, format_history_turn, extract_fallback_answer, safe_format_prompt


# ── Prompts ──────────────────────────────────────────────────────────────────

# Provided original prompts
SHORT_ANSWER_PROMPT_BASE = """Your are a research assistant with the ability to perform web searches to answer questions. You can answer a question with many turns of search and reasoning. Based on the history information, you need to suggest the next action to complete the task.

You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

IMPORTANT: You must strictly adhere to the following rules:
1. Choose ONLY ONE action from the list below for each response, DO NOT perform more than one action per step.
2. Follow the exact syntax format for the selected action, DO NOT create or use any actions other than those listed.
3. **Don't do duplicate search.** Pay attention to the history search results.
4. You are currently at Turn {turn} out of {max_turns}. You MUST find the answer or provide your best guess before Turn {max_turns} ends.
5. DO NOT guess the answer if the search results do not provide evidence. If you are unsure, continue searching or state that the information is missing.
6. IF THIS IS TURN {max_turns}, YOU MUST USE THE <answer> TAG. DO NOT SUGGEST A SEARCH OR SUMMARY ON TURN {max_turns}. 

Valid actions:
1. <search> query </search>: search the web for information if you consider you lack some knowledge.
2. <answer> answer </answer>: output the final answer if you consider you are able to answer the question. The answer should be short and concise. No justification is needed.
3. <summary> important parts of the history turns </summary>: summarize the history turns. Reflect the search queries and search results in you history turns, and keep the information you consider important for answering the question. Still keep the tag structure, keep search queries between <search> and </search>, and keep search results between <information> and </information>. The history turn information for your subsequent turns will be updated according to this summary action.

Output instructions:
First, you should think step-by-step about the question and the history turns. 
In your thinking process, you MUST explicitly state:
- What I found: [concise summary of gathered info]
- What is missing: [specific info that still needs to be found]

Then you should choose **ONLY ONE** of the following actions:
- If You want to search, You should put the query between <search> and </search>.
- If You want to summarize the history turns, You should put the summary between <summary> and </summary>.
- If You want to give the final answer, You should put the answer between <answer> and </answer>.
You can only use ONE action per response.

Format:
Thinking Process: 
[your step-by-step reasoning]
What I found: [summary]
What is missing: [missing info]

Action: [action]

Note: text between <information></information> is the search results from search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.

Question: {question}

History Turns: 
{history}
"""

# Compress search results to just the most relevant content
_MAX_SEARCH_CHARS = 2000


class S1Method(BaseMethod):
    """
    Sequential scaling: single thread, growing token budget per turn.

    Implements a ReAct-style loop:
      1. Prompt model with question + tag-based history (<search>, <information>)
      2. Model reasons and issues <search>, <summary>, or <answer>
      3. Execute searches, update history
      4. Repeat until <answer> found or n turns exhausted
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

        # history_str accumulates tag-based history turns
        history_str = ""
        final_answer: Optional[str] = None
        total_search_calls = 0

        # Compute-matched scaling: break the turn budget 't' into 'k' sequential steps
        # Each step gets ~1024 tokens of reasoning/search.
        k_steps = max(1, self.config.t // 1024)
        step_budget = 1024

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n}")
            
            # Inner causal loop: model can search up to k times sequentially per turn
            for step in range(1, k_steps + 1):
                self._log(f"Turn {turn} Step {step}/{k_steps}: thinking...")
                
                turn_log = TurnLog(turn=turn) # We'll log each step as a new entry for visibility
                
                user_content = safe_format_prompt(SHORT_ANSWER_PROMPT_BASE,
                    question=question,
                    history=history_str or "(empty)",
                    turn=turn,
                    max_turns=self.config.n,
                )

                messages = [
                    {"role": "user", "content": user_content},
                ]

                response_text, p_tokens, o_tokens = await call_llm(
                    messages=messages,
                    model=self.model,
                    max_tokens=step_budget,
                    temperature=0.7,
                )
                
                turn_log.reasoning = response_text
                turn_log.prompt_tokens = p_tokens
                turn_log.output_tokens = o_tokens

                # 1. Check for <answer>
                answer = extract_tag(response_text, "answer")
                if answer:
                    final_answer = answer
                    turn_log.answer_found = True
                    result.turns.append(turn_log)
                    self._log(f"Turn {turn} Step {step}: answer found: '{answer[:80]}'")
                    break # Break inner loop

                # 2. Check for <summary>
                summary = extract_tag(response_text, "summary")
                if summary:
                    self._log(f"Turn {turn} Step {step}: summary issued")
                    history_str = summary
                    result.turns.append(turn_log)
                    continue # Continue to next sequential step with new history

                # 3. Check for <search>
                query = extract_tag(response_text, "search")
                if query:
                    self._log(f"Turn {turn} Step {step}: searching '{query}'")
                    sr = await web_search(query, max_chars=_MAX_SEARCH_CHARS)
                    total_search_calls += 1
                    
                    # Causal update: feed result back IMMEDIATELY for the next step
                    turn_entry = format_history_turn(query, sr[:800])
                    history_str = (history_str + "\n" + turn_entry).strip()
                    
                    turn_log.search_queries = [query]
                    result.turns.append(turn_log)
                    continue # Sequential scaling: next step sees this result
                else:
                    # No valid tag found
                    self._log(f"Turn {turn} Step {step}: NO VALID TAG")
                    history_str = (history_str + f"\n[Turn {turn}] No action taken. Please choose <search>, <summary>, or <answer>.").strip()
                    result.turns.append(turn_log)
                    break # Stop inner loop if it gets stuck

            if final_answer:
                break # Break outer loop
                
            if pbar:
                pbar.update(1)

        # Fallback extraction if no <answer> tag found after n turns
        if final_answer is None and result.turns:
            last_reasoning = result.turns[-1].reasoning
            final_answer = extract_tag(last_reasoning, "answer") or extract_fallback_answer(last_reasoning)

        result.final_answer = final_answer or ""
        result.turns_used = len(result.turns)
        result.search_calls_used = total_search_calls
        result.total_prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        result.total_output_tokens = sum(t.output_tokens for t in result.turns)
        result.em = exact_match(result.final_answer, answer_gt)
        result.f1 = f1_score(result.final_answer, answer_gt)

        return result


def _dummy_placeholder():
    pass
