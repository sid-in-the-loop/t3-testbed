"""
T³ Experimental Variants.

- T3Anchor: Uses anchor-based seeds focused on question aspects
- T3DiversityJaccard: Uses Jaccard-based diverse query selection
- T3DiversityDPP: Uses DPP-based diverse query selection (greedy approximation)
- T3Dynamic: Dynamic thread spawning controlled by parent LLM
"""

import asyncio
import re
from typing import Optional, List

from ..llm import call_llm
from ..search import web_search
from ..eval import exact_match, f1_score
from .base import BaseMethod, MethodResult, TurnLog, extract_answer, extract_tag, format_history_turn, extract_fallback_answer, safe_format_prompt
from .utils import generate_anchor_seeds, select_diverse_queries, tokenize
from .t3_fixed import SEED_POOL, THREAD_SUMMARY_PROMPT


# ── Shared Thread and Prompt Infrastructure ──────────────────────────────────────

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

_MAX_PARENT_CONTEXT_CHARS = 8000


ANCHOR_THREAD_PROMPT = """Question: {question}
Parent Coordinator Intent: {parent_query}
History: {history}
Your Specific Focus: {seed}

Task: 
1. Generate a search query that satisfies the Coordinator's intent but strictly follows your assigned Focus.
2. IMPORTANT: Your query MUST be unique and explore an angle that hasn't been searched before in the history.
3. After seeing the search results, provide a concise summary of the key facts found.

Action: <search> query </search>"""

class T3AnchorMethod(BaseMethod):
    """
    T³ Anchor: Uses anchor-based seeds focused on different aspects of the question.

    Seeds are generated by identifying key entities/concepts in the question
    and creating focused search strategies around them.
    """

    def _generate_seeds(self, question: str, k: int, history: str = "") -> list[str]:
        """Generate anchor-based seeds focused on question aspects."""
        return generate_anchor_seeds(question, k)

    async def _run_thread(self, thread_idx: int, question: str, parent_query: str,
                         seed: str, history: str) -> tuple[str, str]:
        """One thread: generate <search> query → execute search → summarize."""
        query_messages = [
            {"role": "system", "content": THREAD_SYSTEM},
            {"role": "user", "content": safe_format_prompt(ANCHOR_THREAD_PROMPT,
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
            temperature=0.8, # Higher temperature for anchor diversity
        )

        query = extract_tag(query_response, "search") or f"{parent_query} {seed[:20]}"
        search_result = await web_search(query, max_chars=2000)

        # Summarize result for parent
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

    async def run_question(self, question_id: str, question: str, answer_gt: str,
                          pbar: Optional[object] = None) -> MethodResult:
        """Run T³ Anchor on one question."""
        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer="",
            method="t3_anchor",
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

            # 1. Critic Phase
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

            # Check for <answer>
            answer = extract_tag(parent_response, "answer")
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                break

            # Check for <summary>
            summary = extract_tag(parent_response, "summary")
            if summary:
                history_str = summary
                result.turns.append(turn_log)
                continue

            # Check for <search> (triggers Spawn-Join)
            parent_query = extract_tag(parent_response, "search")
            if parent_query:
                if pbar:
                    pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (spawn)")

                # 2. Spawn Phase
                seeds = self._generate_seeds(question, k, history_str)
                thread_tasks = [
                    self._run_thread(i, question, parent_query, seeds[i], history_str)
                    for i in range(k)
                ]
                thread_outputs = await asyncio.gather(*thread_tasks)

                queries, search_results = zip(*thread_outputs)
                turn_log.thread_queries = list(queries)
                turn_log.thread_results = [sr[:500] for sr in search_results]
                turn_log.k_used = k
                total_search_calls += k

                # 3. Join Phase
                new_entries = []
                for i in range(len(queries)):
                    new_entries.append(f"<search> {queries[i]} </search>\n<information>\n{thread_outputs[i][1]}\n</information>")

                turn_results_block = "\n".join(new_entries)
                history_str = (history_str + "\n" + turn_results_block).strip()
            else:
                history_str = (history_str + f"\n[Turn {turn}] No action taken.").strip()

            result.turns.append(turn_log)

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


class T3DiversityJaccardMethod(BaseMethod):
    """
    T³ Diversity (Jaccard): Uses Jaccard-based diverse query selection.

    Instead of hand-crafted seeds, generates many candidate queries and selects
    the most diverse subset using Jaccard distance.
    """

    def _generate_candidate_queries(self, question: str, parent_query: str,
                                  history: str, num_candidates: int = 20) -> List[str]:
        """Generate many candidate queries by applying seeds as modifiers."""
        candidates = []
        base_query = parent_query if parent_query and "information about the question" not in parent_query else question
        
        # Clean up the base query (remove any existing tags)
        base_query = re.sub(r'<.*?>', '', base_query).strip()

        for seed in SEED_POOL[:num_candidates]:
            # Apply seed as a targeted search modifier
            if "direct" in seed.lower():
                candidates.append(base_query)
            elif "alternative" in seed.lower():
                candidates.append(f"{base_query} names synonyms")
            elif "wikipedia" in seed.lower():
                candidates.append(f"site:wikipedia.org {base_query}")
            elif "official" in seed.lower() or "primary" in seed.lower():
                candidates.append(f"official source {base_query}")
            elif "date" in seed.lower() or "temporal" in seed.lower():
                candidates.append(f"{base_query} dates timeline")
            elif "numeric" in seed.lower() or "count" in seed.lower():
                candidates.append(f"{base_query} statistics count")
            else:
                # Use the seed itself as a search suffix, but cleaned
                clean_seed = seed.replace("Search for ", "").replace("Search using ", "").lower()
                candidates.append(f"{base_query} {clean_seed}")

        return candidates[:num_candidates]

    def _generate_seeds(self, question: str, k: int, history: str = "", parent_query: str = "") -> list[str]:
        """Generate diverse seeds using Jaccard-based selection."""
        # Use the experiment config method to determine selection strategy
        method = "jaccard"
        if self.config.id == "T3-Fixed-DPP":
            method = "dpp"
            
        candidates = self._generate_candidate_queries(question, parent_query, history, 20)
        diverse_queries = select_diverse_queries(candidates, k, method=method)
        return diverse_queries

    async def _run_thread(self, thread_idx: int, question: str, parent_query: str,
                         seed: str, history: str) -> tuple[str, str]:
        """One thread: use the diverse query directly."""
        # For diversity methods, seed is actually the selected diverse query
        query = seed
        search_result = await web_search(query, max_chars=2000)
        
        # Summarize result for parent (same as Anchor/Fixed)
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
        return query, summary_response.strip()

    async def run_question(self, question_id: str, question: str, answer_gt: str,
                          pbar: Optional[object] = None) -> MethodResult:
        """Run T³ Diversity (Jaccard/DPP) on one question."""
        method_name = "t3_diversity_jaccard"
        if self.config.id == "T3-Fixed-DPP":
            method_name = "t3_diversity_dpp"
            
        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer="",
            method=method_name,
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

            # 1. Critic Phase
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

            # Check for <answer>
            answer = extract_tag(parent_response, "answer")
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                break

            # Check for <summary>
            summary = extract_tag(parent_response, "summary")
            if summary:
                history_str = summary
                result.turns.append(turn_log)
                continue

            # Check for <search> (triggers Spawn-Join)
            parent_query = extract_tag(parent_response, "search")
            if parent_query:
                if pbar:
                    pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (spawn)")

                # 2. Spawn Phase - Generate diverse queries
                seeds = self._generate_seeds(question, k, history_str, parent_query)
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

                # 3. Join Phase
                new_entries = []
                for i in range(len(queries)):
                    new_entries.append(f"<search> {queries[i]} </search>\n<information>\n{thread_outputs[i][1]}\n</information>")

                turn_results_block = "\n".join(new_entries)
                history_str = (history_str + "\n" + turn_results_block).strip()
            else:
                history_str = (history_str + f"\n[Turn {turn}] No action taken.").strip()

            result.turns.append(turn_log)

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


class T3DynamicMethod(BaseMethod):
    """
    T³ Dynamic: Parent LLM controls thread spawning dynamically.

    Instead of fixed k threads, parent decides how many threads to spawn (0 to max_k)
    and what each thread should search for.
    """

    async def _run_dynamic_threads(self, question: str, parent_decision: str,
                                 history: str, max_k: int) -> tuple[List[str], List[str]]:
        """
        Parse parent decision and spawn the requested number of threads.

        Parent response format: <spawn k="2" queries="query1|query2"/>
        """
        # Parse the spawn tag
        spawn_match = re.search(r'<spawn\s+k="(\d+)"\s+queries="([^"]+)"/>', parent_decision)
        if not spawn_match:
            return [], []  # No spawning

        k = min(int(spawn_match.group(1)), max_k)
        query_str = spawn_match.group(2)
        queries = query_str.split('|')[:k]  # Limit to k queries

        # Pad with fallback queries if needed
        while len(queries) < k:
            queries.append(f"{question[:30]} additional search {len(queries)+1}")

        # Run threads
        async def run_single_thread(q: str) -> tuple[str, str, str]:
            search_result = await web_search(q, max_chars=2000)
            
            # Summarize result for parent
            summary_messages = [
                {"role": "system", "content": THREAD_SYSTEM},
                {"role": "user", "content": safe_format_prompt(THREAD_SUMMARY_PROMPT,
                    question=question,
                    query=q,
                    results=search_result,
                )},
            ]
            summary_response, _, _ = await call_llm(
                messages=summary_messages,
                model=self.model,
                max_tokens=150,
                temperature=0.3,
            )
            return q, search_result, summary_response.strip()

        thread_tasks = []
        for q in queries:
            thread_tasks.append(run_single_thread(q))
            await asyncio.sleep(0.5) # Jitter
            
        thread_outputs = await asyncio.gather(*thread_tasks)
        
        final_queries, raws, summaries = zip(*thread_outputs) if thread_outputs else ([], [], [])

        return list(final_queries), list(raws), list(summaries)

    async def run_question(self, question_id: str, question: str, answer_gt: str,
                          pbar: Optional[object] = None) -> MethodResult:
        """Run T³ Dynamic on one question."""
        result = MethodResult(
            question_id=question_id,
            question=question,
            answer_gt=answer_gt,
            final_answer="",
            method="t3_dynamic",
            config_id=self.config.id,
        )

        max_k = self.config.k  # Maximum threads allowed
        history_str = ""
        final_answer: Optional[str] = None
        total_search_calls = 0

        for turn in range(1, self.config.n + 1):
            if pbar:
                pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (critic)")

            turn_log = TurnLog(turn=turn)

            # 1. Critic Phase - Dynamic spawning prompt with Seed Pool for inspiration
            seeds_text = "\n".join([f"- {s}" for s in SEED_POOL])
            
            # Use safe replacement for dynamic instructions
            dynamic_instructions = f"""
### Search Compute Budget
You have a parallel search budget of up to {max_k} threads per turn. 
**CRITICAL**: You are rewarded for finding the answer in FEWER turns.
- If the question has multiple entities, dates, or sub-questions, you MUST use `<spawn k="N" .../>` with N >= 4.
- Do NOT issue a single `<search>` unless you are 100% sure only one piece of information is missing.
- When in doubt, SPAWN MORE THREADS. It costs you nothing extra in terms of turns and provides more diverse evidence.

### Dynamic Search Action:
1. <spawn k="N" queries="query1|query2|..."/>: Spawn N parallel threads (up to {max_k}).
   - k (N): number of parallel threads.
   - queries: pipe-separated list of N unique search queries.

**Diversity Strategies (Use these to fill your spawn queries):**
{seeds_text}

CRITICAL: You MUST use the `<spawn>` tag for all search operations. Do NOT use any other tags. Parallelize your search to gather as much information as possible in each turn.
"""
            # Construct the prompt by manual assembly to avoid .format() issues with instructions
            parent_content = SHORT_ANSWER_PROMPT_BASE.replace("Question: {question}", f"{dynamic_instructions}\nQuestion: {{question}}")
            parent_content = safe_format_prompt(parent_content,
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

            # Check for <answer>
            answer = extract_tag(parent_response, "answer")
            if answer:
                final_answer = answer
                turn_log.answer_found = True
                result.turns.append(turn_log)
                break

            # Check for <summary>
            summary = extract_tag(parent_response, "summary")
            if summary:
                history_str = summary
                result.turns.append(turn_log)
                continue

            # Check for <spawn> or <search>
            queries = []
            raw_results = []
            summarized_results = []
            if '<spawn' in parent_response:
                queries, raw_results, summarized_results = await self._run_dynamic_threads(
                    question, parent_response, history_str, max_k)
            else:
                q = extract_tag(parent_response, "search")
                if q:
                    # Treat single search as spawn k=1
                    queries = [q]
                    # Since we want to use the same logic, we'll run it through a single thread
                    sr = await web_search(q, max_chars=2000)
                    raw_results = [sr]
                    
                    # Summarize result for parent (same as threads do)
                    summary_messages = [
                        {"role": "system", "content": THREAD_SYSTEM},
                        {"role": "user", "content": safe_format_prompt(THREAD_SUMMARY_PROMPT,
                            question=question,
                            query=q,
                            results=sr,
                        )},
                    ]
                    summary_response, _, _ = await call_llm(
                        messages=summary_messages,
                        model=self.model,
                        max_tokens=150,
                        temperature=0.3,
                    )
                    summarized_results = [summary_response.strip()]
            
            if queries:
                if pbar:
                    pbar.set_description(f"Q {question_id}: Turn {turn}/{self.config.n} (dynamic spawn {len(queries)})")

                turn_log.thread_queries = queries
                turn_log.thread_results = [r[:500] for r in raw_results]
                turn_log.thread_summaries = summarized_results
                turn_log.k_used = len(queries)
                total_search_calls += len(queries)

                # Join Phase
                new_entries = []
                for i in range(len(queries)):
                    new_entries.append(f"<search> {queries[i]} </search>\n<information>\n{summarized_results[i]}\n</information>")

                turn_results_block = "\n".join(new_entries)
                history_str = (history_str + "\n" + turn_results_block).strip()
            else:
                history_str = (history_str + f"\n[Turn {turn}] No action taken. Please choose <spawn>, <search>, <summary>, or <answer>.").strip()

            result.turns.append(turn_log)

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


class T3DynamicJaccardMethod(T3DynamicMethod):
    """
    T³ Dynamic Jaccard: Parent LLM decides k, we ensure query diversity.

    Parent specifies k and a base query intent. We generate a candidate pool
    of 20 queries and select the k most diverse ones using Jaccard similarity.
    """

    async def run_question(self, question_id: str, question: str, answer_gt: str,
                          pbar: Optional[object] = None) -> MethodResult:
        """Run T³ Dynamic Jaccard on one question."""
        result = await super().run_question(question_id, question, answer_gt, pbar)
        result.method = "t3_dynamic_jaccard"
        return result

    async def _run_dynamic_threads(self, question: str, parent_decision: str,
                                 history: str, max_k: int) -> tuple[List[str], List[str], List[str]]:
        """
        Parse parent decision, generate candidates, filter by Jaccard, and spawn threads.
        Expected format: <spawn k="N" queries="base_query"/>
        """
        # Parse the spawn tag - 'queries' is treated as the base query intent for expansion
        spawn_match = re.search(r'<spawn\s+k="(\d+)"\s+queries="([^"]+)"/>', parent_decision)
        
        if spawn_match:
            k = min(int(spawn_match.group(1)), max_k)
            base_query = spawn_match.group(2)
        else:
            # Check for simple <search> as fallback
            q = extract_tag(parent_decision, "search")
            if q:
                k = 1
                base_query = q
            else:
                return [], [], []
        
        # 1. Oversample candidates using Jaccard logic
        # We instantiate a temporary helper to reuse its candidate generator
        diversity_helper = T3DiversityJaccardMethod(self.model, self.config, self.verbose)
        candidates = diversity_helper._generate_candidate_queries(question, base_query, history, num_candidates=20)
        
        # 2. Select the k most diverse ones
        diverse_queries = select_diverse_queries(candidates, k, method="jaccard")

        # 3. Run threads (identical to T3DynamicMethod)
        async def run_single_thread(q: str) -> tuple[str, str, str]:
            search_result = await web_search(q, max_chars=2000)
            summary_messages = [
                {"role": "system", "content": THREAD_SYSTEM},
                {"role": "user", "content": safe_format_prompt(THREAD_SUMMARY_PROMPT,
                    question=question,
                    query=q,
                    results=search_result,
                )},
            ]
            summary_response, _, _ = await call_llm(
                messages=summary_messages,
                model=self.model,
                max_tokens=150,
                temperature=0.3,
            )
            return q, search_result, summary_response.strip()

        thread_tasks = []
        for q in diverse_queries:
            thread_tasks.append(run_single_thread(q))
            await asyncio.sleep(0.5) # Jitter
            
        thread_outputs = await asyncio.gather(*thread_tasks)
        
        if not thread_outputs:
            return [], [], []
            
        final_queries, raws, summaries = zip(*thread_outputs)
        return list(final_queries), list(raws), list(summaries)


def _dummy_placeholder():
    pass
