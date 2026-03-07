"""
Diversity-at-First-Turn Method.

Design:
  - Turn 1: LLM generates o candidate queries.
  - Selection: Select t (k) diverse queries from the pool using Jaccard-DPP or MMR.
  - Turns 2-12: Run t independent trajectories (threads).
  - Success: A trajectory is successful if any of its threads produces the correct answer.
"""

import asyncio
import re
import json
from typing import Optional, List, Tuple

from ..llm import call_llm
    
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer, extract_tag, format_history_turn, extract_fallback_answer, safe_format_prompt
from .utils import select_diverse_queries
from .s1 import SHORT_ANSWER_PROMPT_BASE as S1_PROMPT

# Turn 1 Pool Generation Prompt
TURN_1_GENERATION_PROMPT = """You are a research assistant tasked with generating search queries to answer a question.
Question: {question}

Your task is to generate ONE search query that will help find the information needed to answer this question.
The query should be specific, effective, and targeted.

STRICT RULES:
1. Do NOT assume you know the answer to any part of the question. 
2. If the question mentions "the first place", "the tallest person", etc., your query should first identify that entity.
3. Do NOT use quotation marks around your entire query.
4. Focus on a unique sub-component or a specific entity from the question.
5. Your query should explore a unique angle or sub-component of the question to ensure diversity when multiple queries are generated.

Action: <search> [your query here] </search>"""

class DiversityTurnMethod(BaseMethod):
    """
    Implements the diversity-at-first-turn experiment.
    - Turn 1: Sample o candidates, select k (t) diverse queries.
    - Turns 2-12: Run k independent threads starting with their selected query.
    """

    async def _generate_pool(self, question: str, o: int) -> List[str]:
        """Generate o candidate queries from the LLM at temperature 1.0."""
        tasks = []
        for _ in range(o):
            messages = [{"role": "user", "content": TURN_1_GENERATION_PROMPT.format(question=question)}]
            # Use temperature 1.0 for diversity
            tasks.append(call_llm(messages, self.model, max_tokens=100, temperature=1.0))
            # Small jitter to avoid hitting rate limits simultaneously
            await asyncio.sleep(0.1)
            
        responses = await asyncio.gather(*tasks)
        queries = []
        for text, _, _ in responses:
            query = extract_tag(text, "search")
            if query:
                queries.append(query)
            else:
                # Fallback extraction
                match = re.search(r'<search>(.*?)</search>', text, re.IGNORECASE | re.DOTALL)
                if match:
                    queries.append(match.group(1).strip())
                else:
                    # Last resort fallback: use the first line that isn't thinking
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    for l in lines:
                        if "Thinking" not in l and "Action" not in l:
                            queries.append(l[:100])
                            break
        
        # Ensure we have enough queries, even if some failed
        while len(queries) < o:
            queries.append(f"{question[:50]} search {len(queries)}")
            
        return queries[:o]

    async def _run_independent_thread(
        self,
        thread_id: int,
        question: str,
        initial_query: str,
        answer_gt: str,
        max_turns: int = 12
    ) -> Tuple[bool, List[TurnLog], str]:
        """Run a single thread starting from Turn 1 (with initial_query) through Turn 12."""
        
        # Turn 1: Search and get result
        turn_1_log = TurnLog(turn=1)
        turn_1_log.thread_queries = [initial_query]
        search_result = await web_search(initial_query, max_chars=2000)
        turn_1_log.thread_results = [search_result[:500]]
        
        history_str = format_history_turn(initial_query, search_result[:800])
        
        final_answer = None
        turn_logs = [turn_1_log]
        
        # Turns 2 to max_turns
        for turn in range(2, max_turns + 1):
            user_content = safe_format_prompt(S1_PROMPT,
                question=question,
                history=history_str or "(empty)",
                turn=turn,
                max_turns=max_turns,
            )

            messages = [{"role": "user", "content": user_content}]
            response_text, p_tokens, o_tokens = await call_llm(
                messages=messages,
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
            )
            
            turn_log = TurnLog(turn=turn)
            turn_log.reasoning = response_text
            turn_log.prompt_tokens = p_tokens
            turn_log.output_tokens = o_tokens
            
            # Check for <answer>
            answer = extract_tag(response_text, "answer")
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                turn_logs.append(turn_log)
                break

            # Check for <summary>
            summary = extract_tag(response_text, "summary")
            if summary:
                history_str = summary
                turn_logs.append(turn_log)
                continue

            # Check for <search>
            query = extract_tag(response_text, "search")
            if query:
                sr = await web_search(query, max_chars=2000)
                turn_entry = format_history_turn(query, sr[:800])
                history_str = (history_str + "\n" + turn_entry).strip()
                turn_log.thread_queries = [query]
                turn_log.thread_results = [sr[:500]]
                turn_logs.append(turn_log)
                continue
            else:
                # Fallback to answer extraction if no tag found at last turn
                if turn == max_turns:
                    final_answer = extract_fallback_answer(response_text)
                turn_logs.append(turn_log)
                break

        # Final evaluation for this thread
        if final_answer is None and turn_logs:
            final_answer = extract_fallback_answer(turn_logs[-1].reasoning)
            
        is_correct = exact_match(final_answer or "", answer_gt)
        return is_correct, turn_logs, final_answer or ""

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
            method="diversity_turn",
            config_id=self.config.id,
        )

        o = self.config.o
        t = self.config.k  # Using k as the number of selected threads (t)
        selection_method = self.config.diversity_method

        if pbar:
            pbar.set_description(f"Q {question_id}: Generating pool (o={o})")

        # 1. Generate pool of o queries
        candidate_pool = await self._generate_pool(question, o)

        # 2. Selection phase
        if selection_method == "naive":
            # EXP1: skip diversity selection, just pick first t
            selected_queries = candidate_pool[:t]
        else:
            # EXP2/3: select t most diverse queries
            selected_queries = select_diverse_queries(
                candidate_pool, 
                t, 
                method=selection_method, 
                question=question
            )

        if pbar:
            pbar.set_description(f"Q {question_id}: Running {t} threads")

        # 3. Run t independent threads
        thread_tasks = []
        for i, query in enumerate(selected_queries):
            thread_tasks.append(self._run_independent_thread(i, question, query, answer_gt, self.config.n))
            await asyncio.sleep(0.5) # Jitter
            
        thread_results = await asyncio.gather(*thread_tasks)

        # 4. Aggregate results
        # trajectory_results: [{thread_queries, success}] as requested
        # We store this in MethodResult for later logging
        trajectory_info = []
        all_turn_logs = []
        found_any_correct = False
        
        # We'll use the answer from the first correct thread as the final_answer, 
        # or the first thread's answer if none are correct.
        final_ans = ""
        
        for i, (is_correct, logs, ans) in enumerate(thread_results):
            trajectory_info.append({
                "thread_id": i,
                "query": selected_queries[i],
                "success": 1 if is_correct else 0,
                "answer": ans
            })
            if is_correct:
                found_any_correct = True
                if not final_ans:
                    final_ans = ans
            
            # Combine turn logs for aggregate token counting
            all_turn_logs.extend(logs)

        if not final_ans and thread_results:
            final_ans = thread_results[0][2]

        result.final_answer = final_ans
        result.em = found_any_correct # Trajectory succeeds if ANY thread produces correct answer
        result.turns = all_turn_logs
        result.turns_used = self.config.n
        result.search_calls_used = sum(len(log.thread_queries) for log in all_turn_logs)
        result.total_prompt_tokens = sum(log.prompt_tokens for log in all_turn_logs)
        result.total_output_tokens = sum(log.output_tokens for log in all_turn_logs)
        
        # Custom metadata for logging
        result.metadata = {
            "o": o,
            "t": t,
            "selection_method": selection_method,
            "selected_queries": selected_queries,
            "trajectory_results": trajectory_info
        }

        return result
