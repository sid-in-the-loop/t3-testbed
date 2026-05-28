# Prompts used in `diversity_parallel_benchmark`

**Data routing:** GAIA examples use **REACT** (`REACT_PROMPT`). HotpotQA examples use **Search-R1 style** (`SEARCH_R1_PROMPT`) because `question_id` is like `hotpotqa-7059` → `get_prompt_for_question` matches source `hotpotqa`.

**Temperatures (code):** pool gen `1.0`; ReAct turns `0.7` (naive turn-1 `1.0`); synthesis `0.3`; judges `0.0`.

---

## 1. ReAct (GAIA / sequential / default) — `diversity_scaling.REACT_PROMPT`

```
You are a research assistant that answers questions by searching the web.
You have {max_turns} turns total. You are on turn {turn}.

Question: {question}

History of searches and findings:
{history}

Rules (follow strictly):
1. Every response must use exactly one of these two formats — no other format is accepted.
2. If you need more information: output <thought>...</thought> then <search>your single search query here</search>. Nothing else.
3. If you have the answer (or it is the last turn): output <thought>...</thought> then <answer>your concise final answer here</answer>. The answer must be inside the <answer> tags; use a short phrase, number, or list as requested by the question.
4. Do not output an answer outside <answer>...</answer>. Do not output search and answer in the same turn.
5. On the last turn (turn {max_turns}) you MUST output <answer>...</answer> with your best guess if unsure.

Your response (use <thought>, then either <search> or <answer>):
```

---

## 2. Pool generation (o=16 queries) — `diversity_scaling.POOL_GEN_PROMPT`

`{history_block}` is empty in the benchmark pool call.

```
Generate exactly {o} diverse search queries to investigate this question.
Each query should approach the question from a different angle, specifically targeting different constraints or components of the question.
{history_block}

Question: {question}

Output exactly {o} queries, one per line, numbered 1-{o}. No other text.
```

---

## 3. Search-R1 style (HotpotQA ids) — `diversity_scaling.SEARCH_R1_PROMPT`

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.
For example, <answer> Beijing </answer>.

Question: {question}

{history}

Your response (use <think>, then either <search> or <answer>):
```

---

## 4. LLM judge (pass@1 / pass@4 when EM fails) — `diversity_parallel_benchmark.llm_judge_correct`

Strict: same facts as gold; only punctuation, dates, units, trivial formatting may differ. When unsure → INCORRECT. (See source for full text.)

---

## 5. APD pair judge — `diversity_parallel_benchmark.llm_judge_pair_same`

```
Given two answers to the same question, determine whether they are semantically equivalent or genuinely different.

Question: {question}
Answer A: {a}
Answer B: {b}

Are these two answers saying the same thing?
Reply with only: SAME or DIFFERENT

SAME: both answers make the same factual claim. Minor wording differences, abbreviations, or formatting don't matter.
DIFFERENT: the answers make different factual claims, name different entities, or give different values.
```

---

## 6. Parallel synthesis (coordinator) — `diversity_parallel_benchmark.synthesize_answer`

Per thread, blocks are truncated. User message shape:

```
You are a research coordinator. Synthesize one best final answer from independent web-search threads.

Question: {question}

--- Thread 0 ---
Evidence:
{summarized turn logs + excerpt}
Thread final answer: {thread_answer}
--- Thread 1 ---
...

Merge evidence, resolve conflicts, output the single best concise answer inside <answer>...</answer> tags.
```

---

## Default data files

- GAIA: `data/gaia_25.json` (file order, first 25 rows)
- Hotpot: `data/hotpotqa_25-random.jsonl` (file order, first 25 lines)

Override: `--gaia-path`, `--hotpot-path`.
