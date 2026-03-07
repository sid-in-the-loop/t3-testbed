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
from .base import BaseMethod, MethodResult, TurnLog, extract_answer, extract_tag, format_history_turn, extract_fallback_answer, safe_format_prompt


# ── Fixed Diversity Seed Pool ─────────────────────────────────────────────────

SEED_POOL = [
    "Direct literal phrasing of the question",
    "Alternative names, acronyms, or synonyms for key entities",
    "Context: related events, people, or organizations",
    "Temporal focus: dates, timelines, and historical facts",
    "Wikipedia or official organizational sources",
    "First sub-component of the question",
    "Second sub-component of the question",
    "Two independent facts that uniquely identify the answer",
    "Numbers, counts, dates, or rankings",
    "Geographic locations or institutions",
    "Most recent or updated information",
    "Primary sources: official websites, papers, or press releases",
    "Disambiguation of potentially ambiguous entities",
    "Broad background on the main topic",
    "Pages that explicitly state the answer as a cited fact",
    "Verification of potential candidate answers",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

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
6. IF THIS IS TURN {max_turns}, YOU MUST USE THE <answer> TAG. DO NOT SPAWN THREADS OR SEARCH ON TURN {max_turns}. 

Valid actions:
1. <search> query </search>: search the web for information if you consider you lack some knowledge.
2. <answer> answer </answer>: output the final answer if you consider you are able to answer the question. The answer should be short and concise. No justification is needed.
3. <summary> important parts of the history turns </summary>: summarize the history turns. Reflect the search queries and search results in you history turns, and keep the information you consider important for answering the question. Still keep the tag structure, keep search queries between <search> and </search>, and keep search results between <information> and </information>. The history turn information for your subsequent turns will be updated according to this summary action.

Output instructions:
First, you should think step-by-step about the question and the history turns. 
In your thinking process, you MUST explicitly state:
- What I found: [concise summary of gathered info]
- What is missing: [specific info that still needs to be found]

CRITICAL: Cross-reference information from all parallel threads. If threads provide conflicting information, do NOT guess; instead, use a follow-up <search> to resolve the conflict or check the original sources.

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

THREAD_SYSTEM = "You are a focused search thread. Your only job is to find and extract relevant information for the parent coordinator."

THREAD_PROMPT = """Question: {question}
Parent Coordinator Intent: {parent_query}
History: {history}
Your Specific Strategy: {seed}

Task: 
1. Generate a search query that satisfies the Coordinator's intent but strictly follows your Strategy.
2. After seeing the search results, provide a concise summary of the key facts found that help answer the question.

Action: <search> query </search>"""

THREAD_SUMMARY_PROMPT = """You are a focused search thread. You just performed a search for the question: "{question}"
Your search query was: "{query}"
Search Results: 
<information>
{results}
</information>

Task: Provide a concise summary of the most important facts found in these results that directly help answer the question. 

STRICT RULES:
1. ONLY include information that is explicitly stated in the search results above.
2. DO NOT use your internal knowledge to fill in gaps.
3. DO NOT hallucinate names, dates, or roles that are not in the text.
4. If the results do not contain the answer or parts of the answer, explicitly state "The search results did not provide information about [what's missing]".
5. Keep the summary under 3-4 sentences.

Summary: """

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
        One thread (Explorer): generate <search> query → execute search → summarize.
        Returns (query, summary).
        """
        # Step 1: Generate query
        query_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": safe_format_prompt(THREAD_PROMPT,
                question=question,
                parent_query=parent_query,
                seed=seed,
                history=history or "(empty)",
            )},
        ]
        query_response, _, _ = await call_llm(
            messages=query_messages,
            model=self.model,
            max_tokens=100,
            temperature=0.7,
        )

        query = extract_tag(query_response, "search") or f"{parent_query} {seed[:20]}"

        # Step 2: Search
        search_result = await web_search(query, max_chars=2000)

        # Step 3: Summarize result for parent
        summary_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": safe_format_prompt(THREAD_SUMMARY_PROMPT,
                question=question,
                query=query,
                results=search_result,
            )},
        ]
        summary_response, _, _ = await call_llm(
            messages=summary_messages,
            model=self.model,
            max_tokens=150,
            temperature=0.3,
        )
        summary = summary_response.strip()

        return query, summary

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
            parent_content = safe_format_prompt(SHORT_ANSWER_PROMPT_BASE,
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
                thread_tasks = []
                for i in range(k):
                    # Add jitter to avoid bursty API hits
                    thread_tasks.append(self._run_thread(i, question, parent_query, seeds[i], history_str))
                    # wait slightly between task creations
                    await asyncio.sleep(0.5)
                
                thread_outputs = await asyncio.gather(*thread_tasks)

                queries, search_results = zip(*thread_outputs)
                turn_log.thread_queries = list(queries)
                turn_log.thread_results = [sr[:500] for sr in search_results]
                turn_log.k_used = k
                total_search_calls += k

                # 3. Join Phase: Format summaries for the NEXT turn
                new_entries = []
                for i in range(len(queries)):
                    # Provide the summary in <information> tags as requested by the original prompt structure
                    new_entries.append(f"<search> {queries[i]} </search>\n<information>\n{thread_outputs[i][1]}\n</information>")
                
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
            final_answer = extract_tag(last.parent_response, "answer") or extract_fallback_answer(last.parent_response)

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
