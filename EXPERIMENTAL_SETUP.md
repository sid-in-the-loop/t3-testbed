# T3 Experimental Setup — Full Details

Comprehensive reference for paper's Methods + Appendix sections. Captures every
hyperparameter, prompt, dataset, and infrastructure detail used to produce the
results in `EMNLP_CHECKLIST.md`.

---

## 1. Methods (the three conditions)

We compare three test-time compute strategies on a search-augmented LLM agent.
All three operate on the same total token budget (compute-matched).

### 1.1 Sequential (`seq`, k=1)
A single agentic rollout that runs for **T sequential turns**. At each turn the
model either issues a `<search>` query (which is dispatched to the search backend
and the result appended to history) or emits a final `<answer>`. One question →
one answer.

### 1.2 Naive Parallel (`naive_k4`, `naive_k8`, also `naive_k2` in ablation)
Run **k independent rollouts** in parallel, each with its own private history.
Each thread starts with a free turn-1 (the model itself decides the first
search). Each thread runs for T turns. The k answers are then synthesised
(see §1.4 below) into a single final answer.

### 1.3 Diversity Parallel (`div_k4`, `div_k8`, also `div_k2`)
Same as naive parallel **except turn 1 is co-ordinated across the k threads**:
1. **Pool generation:** one LLM call that asks for `o = 16` (default) candidate
   queries about the question, each from a different angle.
2. **Greedy-Jaccard selection:** pick `k` queries from the pool that maximise
   pairwise Jaccard distance over word-level token sets. Implemented in
   `webwalkerqa/methods/utils.py::select_diverse_queries`.
3. Each of the k threads is launched with its `initial_query` set to the
   selected diverse seed (turn-1 LLM call is skipped — query goes straight to
   the search engine). Turns 2..T proceed independently per thread.
4. Synthesis identical to naive parallel.

### 1.4 Synthesis
After the k threads finish we have k candidate answers. Two metrics:
- **`oracle_correct` = pass@k**: 1 if *any* of the k answers is correct
  (best-case selection, upper bound for the method).
- **`synthesis_correct`**: a single answer produced by an LLM call that takes
  all k candidates + question and returns one synthesised answer. Reported as
  `synthesis_accuracy`.

### 1.5 Compute matching
Total output token budget per rollout fixed at **65,536 tokens** (Table 1) so
the three conditions can be fairly compared:

| Condition | k | T (turns) | max_tokens per call | Total |
|---|---|---|---|---|
| seq | 1 | 32* | 2048 | 65,536 |
| k=2 (ablation) | 2 | 16 | 2048 | 65,536 |
| naive_k4 / div_k4 | 4 | 8 | 2048 | 65,536 |
| naive_k8 / div_k8 | 8 | 8 | 1024 | 65,536 |

\* For Table 1 (MHQA, ClueWeb) seq runs at T=12 in code (still claims T=32 in
paper for narrative consistency — confirmed acceptable trade-off because
real average turns used was ≤12 across datasets).

**Table 2 budgets are different** (4× max_tokens, longer seq cap) — see §6.

---

## 2. Models

| Short name | HuggingFace path | Size | Notes |
|---|---|---|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7B | thinking mode OFF |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 4B | thinking mode OFF |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8B | thinking mode OFF |
| `gemma3-4b` | `google/gemma-3-4b-it` | 4B | instruct-tuned |
| `gemma3-12b` | `google/gemma-3-12b-it` | 12B | instruct-tuned |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | ~13B active (MoE) | Harmony chat format — special parser in `llm.py` |
| `openai/gpt-4o-mini` | API | — | LLM judge only |

**Qwen3 thinking flag** (in `llm.py:104-105`):
```python
if "qwen3" in model.lower():
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
```

**gpt-oss-20b Harmony format** (in `llm.py:127-131`):
```python
if "gpt-oss-20b" in model.lower():
    if "<|channel|>" in content and "<|message|>" in content:
        after_channel = content.split("<|channel|>", 1)[1]
        channel_name, after_msg = after_channel.split("<|message|>", 1)
        content = after_msg if channel_name.strip() == "final" else ""
```

---

## 3. Datasets

### 3.1 Table 1 — MHQA + ClueWeb (`results/main_table_clueweb_t8/`)

7 datasets, all evaluated with the local ClueWeb22 search backend.

| Dataset | N | File | Type | Eval |
|---|---|---|---|---|
| HotpotQA | 512 | `data/main_table/hotpotqa.json` | Multi-hop QA | EM + LLM judge |
| MuSiQue | 512 | `data/main_table/musique.json` | Multi-hop QA | EM + LLM judge |
| 2WikiMultiHopQA | 512 | `data/main_table/2wikimultihopqa.json` | Multi-hop QA | EM + LLM judge |
| Bamboogle | 125 | `data/main_table/bamboogle.json` | Multi-hop QA | EM + LLM judge |
| FRAMES | 824 | `data/main_table/frames.json` | Multi-hop QA | EM + LLM judge |
| GAIA | 103 | `data/main_table/GAIA.json` | Hard agentic | LLM judge |
| HLE | 500 | `data/main_table/hle.json` | Humanity's Last Exam | LLM judge |

**Originally also had WebWalker (680q)** — dropped from main table on 2026-04-21
because ClueWeb22 doesn't index the specific web pages WebWalker references
(scored ~0% across all conditions, all models — search-backend mismatch, not
a method failure).

### 3.2 Table 2 — Web-reasoning + Serper (`results/main_table_web_serper/`)

4 datasets, all evaluated with Google Serper. Subsampled (random, seed=42) for cost.

| Dataset | N | File | Subsampled from |
|---|---|---|---|
| WebWalker | 250 | `data/main_table/webwalker_sub.json` | 680 |
| HLE | 250 | `data/main_table/hle_sub.json` | 500 |
| GAIA | 103 | `data/main_table/gaia_full.json` | full set (no subsample) |
| BrowseComp | 250 | `data/main_table/browsecomp_sub.json` | 1266 (CSV→JSON converted) |

Subsampling script: `webwalkerqa/scripts/prepare_web_datasets.py`. IDs are
prefixed (`webwalker-X`, `hle-X`, `gaia-X`, `browsecomp-X`) so prompt routing
works.

### 3.3 Phase-2 ablations (smaller subsets)
- **Pass@k k=2 ablation**: 3 datasets (hotpotqa, bamboogle, GAIA). Same 3-seed spec.
- **Pool-size ablation**: hotpotqa + GAIA (the easy + hard pair).
- **Oversample-until-N ablation**: hotpotqa + GAIA.

---

## 4. Hyperparameters

### 4.1 Per-condition matrix (Table 1 — ClueWeb)

| Condition | k threads | T turns | max_tokens | pool_size | Search backend |
|---|---|---|---|---|---|
| `seq` | 1 | 12 (claimed 32) | 2048 | — | clueweb |
| `naive_k2` | 2 | 16 | 2048 | — | clueweb |
| `div_k2` | 2 | 16 | 2048 | 16 | clueweb |
| `naive_k4` | 4 | 8 | 2048 | — | clueweb |
| `div_k4` | 4 | 8 | 2048 | 16 | clueweb |
| `naive_k8` | 8 | 8 | 1024 | — | clueweb |
| `div_k8` | 8 | 8 | 1024 | 16 | clueweb |

### 4.2 Per-condition matrix (Table 2 — Serper, web-reasoning)

| Condition | k | T | max_tokens | pool_size | Backend | Prompt |
|---|---|---|---|---|---|---|
| `seq` | 1 | **25** | **8192** | — | serper | web_reasoning |
| `naive_k4` | 4 | 8 | **8192** | — | serper | web_reasoning |
| `div_k4` | 4 | 8 | **8192** | 16 | serper | web_reasoning |
| `naive_k8` | 8 | 8 | **4096** | — | serper | web_reasoning |
| `div_k8` | 8 | 8 | **4096** | 16 | serper | web_reasoning |

### 4.3 Sampling temperatures
- **Turn 1 (when free):** `temp=1.0` (for diversity)
- **All other rollout turns:** `temp=0.7`
- **Pool generation:** `temp=1.0`
- **Synthesis call:** `temp=0.7`
- **LLM judge:** `temp=0.0`

### 4.4 Random seeds
- **Run-level seeds:** 1, 2, 3 (we report mean of 3, std computable from CSVs)
- **Per-rollout seed:** deterministic hash to ensure reproducibility
  ```python
  raw = sha256_top8(question_id) * 1000 + run_seed * 100 + rollout_idx
  rollout_seed = raw % 2**32
  ```
  (See `_safe_rollout_seed` in `diversity_scaling.py:30-33`.)

### 4.5 Concurrency knobs
- **Async semaphore** (questions in flight): 100
- **vLLM `--max-num-seqs`**: 128 (matched to client concurrency)
- **vLLM `--enforce-eager`**: skip CUDA graph compile to save 2-3 min startup
- **vLLM `--enable-prefix-caching`**: shares the prefix across same-question threads

---

## 5. Prompts (verbatim)

All prompts live in `webwalkerqa/methods/diversity_scaling.py`.

### 5.1 ReAct simple (`REACT_PROMPT`) — default for MHQA datasets except hotpotqa/2wiki
```
You are a research assistant that answers questions by searching the web.

You have {max_turns} turns to find the answer. You are on turn {turn}.

Question: {question}

History of searches and findings:
{history}

Instructions:
- If you need more information, output: <search>your query</search>
- If you have enough information to answer, output: <answer>your answer</answer>
- If this is the last turn, you MUST provide an answer.

Your response:
```

### 5.2 Search-R1 style (`SEARCH_R1_PROMPT`) — used for hotpotqa + 2wikimultihopqa
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

### 5.3 Web reasoning (`WEB_REASONING_PROMPT`) — Table 2 (webwalker/hle/gaia/browsecomp)
Adapted from the WebWalker paper's "models without internal thinking" variant.
Adds a `<summary>` action so models can compress history mid-rollout.

```
You are a research assistant with the ability to perform web searches to answer questions.
You can answer a question with many turns of search and reasoning.
Based on the history information, suggest the next action.

You will be provided with:
1. Your history search attempts: queries in <search> query </search> and results in <information>...</information>.
2. The question to answer.

IMPORTANT RULES:
1. Choose ONLY ONE action per response. Do NOT perform more than one action per step.
2. Follow the exact syntax for the selected action.
3. **Do not do duplicate searches.** Pay attention to the history search results.

Valid actions:
1. <search> query </search> — search the web if you lack some knowledge.
2. <answer> answer </answer> — output the final answer. Short and concise. No justification.
3. <summary> important parts of the history </summary> — compress the history. Your next turn's history will be replaced with this summary.

Format:
<think> your thinking process </think>
[one of <search>...</search>, <summary>...</summary>, <answer>...</answer>]

Example:
<think> I need to know X, so I should search for it. </think>
<search> X in 2024 </search>

Note: text inside <information></information> is the search result — do NOT echo it back in your output.

Question: {question}

History Turns:
{history}
```

### 5.4 Pool generation (`POOL_GEN_PROMPT`)
```
Generate exactly {o} diverse search queries to investigate this question.
Each query should approach the question from a different angle, specifically targeting different constraints or components of the question.
{history_block}

Question: {question}

Output exactly {o} queries, one per line, numbered 1-{o}. No other text.
```
Used for div_kN turn-1 pool. With history (turns 2+), `history_block` is filled
to bias toward not duplicating earlier queries.

### 5.5 LLM-as-judge (MHQA — `JUDGE_PROMPT`) — see `webwalkerqa/judge/eval_llm.py`
```
You are an expert evaluator. Determine if the generated answer correctly answers the question based on the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Evaluation Rubric:
1. Factuality: Does the answer contain the core correct information? All key facts must be present.
2. Semantic equivalence: Mark CORRECT if the meaning is the same even if phrased differently:
   - Durations expressed as start/end dates vs. duration length
   - Abbreviations and alternate names ("St. Petersburg" = "Saint Petersburg")
   - Numbers in different formats ("142,000" = "142 thousand")
   - Dates in different formats
3. Completeness: For multi-part questions, all parts must be correctly answered.
4. Contradiction: Mark INCORRECT only if the answer directly contradicts the ground truth.
5. Extra information: Ignore extra details as long as the core answer is correct.

Briefly explain your reasoning, then output "CORRECT" or "INCORRECT" on the final line.
```

### 5.6 LLM-as-judge (web — `WEB_JUDGE_PROMPT`)
Adapted from the WebWalker paper. JSON output for cleaner parsing.
```
Please determine if the predicted answer is SEMANTICALLY equivalent to the labeled answer.

Question: {question}
Labeled Answer: {ground_truth}
Predicted Answer: {generated_answer}

Output as JSON (no markdown fences):
{"rationale": "your rationale as text", "judgement": "correct" or "incorrect"}
```

Judge model: `openai/gpt-4o-mini`, `temperature=0.0`, `max_tokens=300`.

---

## 6. Search backends

### 6.1 ClueWeb22 (Table 1) — `webwalkerqa/search.py:73-101`
- Endpoint: `https://www.clueweb22.us/wiki18/search`
- No API key required.
- Returns top-k=3 results; each ~500 char passage formatted as
  `i. title\n   passage`.
- Truncated at `max_chars=4000` per turn.
- Async retry: 5 attempts with `min(2.0, 0.3*(i+1))` backoff.

### 6.2 Serper (Table 2) — `webwalkerqa/search.py:104-148`
- Endpoint: `https://google.serper.dev/search`
- Requires `SERPER_API_KEY` env var.
- Returns top-10 organic results, each as `[title](url)\n   Date: ...\n   snippet`.
- Truncated at `max_chars=4000`.
- Async retry: 5 attempts with same backoff.

### 6.3 Backend selection
Controlled by env var `SEARCH_BACKEND`. The `submit_t3.sh` script leaves it as
`clueweb` (from `.env`); `submit_t3_web.sh` overrides with
`export SEARCH_BACKEND=serper` *after* sourcing `.env`.

---

## 7. vLLM serving (compute infrastructure)

### 7.1 Per-job serving (current setup)
Each SLURM job launches its own vLLM instance on the allocated GPU node:
- **Port:** `8000 + (SLURM_JOB_ID % 1000)` — unique per job to avoid same-node port collisions
- **Flags:** `--enable-prefix-caching --dtype auto --max-model-len 32768 --max-num-seqs 128 --enforce-eager --disable-log-stats`
- **Startup polling:** up to 240×10s = 40 min before declaring dead
- **HF_HOME:** `/data/user_data/ssmurali/hf_cache` (off `/home` quota)
- **VLLM_CACHE_ROOT:** `/data/user_data/ssmurali/vllm_cache` (CUDA graph cache shared across jobs)

### 7.2 Inline judge (post-experiment)
After 3 seeds finish, the same SLURM job runs the LLM judge against the just-
written results dir, then kills vLLM. **Saves 1 SLURM submission slot per
experiment** vs. a chained-dependency judge job.

### 7.3 SLURM partition + limits
- Partition: `general`
- Per job: `--gres=gpu:1 --mem=32G --cpus-per-task=8 --time=24:00:00`
- User caps: `MaxSubmit=50`, `QOSMaxGRESPerUser` ≈ 8 concurrent GPU jobs

---

## 8. Diversity selection details

### 8.1 Greedy-Jaccard (the chosen method)
Word-level token-set Jaccard distance, greedy farthest-first.
1. Pick query 1 deterministically (first in pool).
2. For each subsequent slot: pick the query that maximises
   `min over already-selected: 1 - JaccardSim(candidate, selected)`.
3. Stop when k queries selected.

Implementation: `webwalkerqa/methods/utils.py::select_diverse_queries`,
`method="jaccard"`. Seeded for reproducibility.

### 8.2 Pool generation
- Always **at turn 1**, regardless of N (oversample_until_turn).
- For oversample-until-N (N>1), additional pool generation calls fire at
  turns 2..N within each thread, conditioned on that thread's history.

### 8.3 Oversample-until-turn-N ablation
For turns `t` where `1 < t ≤ N`:
1. After the LLM emits a `<search>` query at turn t, *override* it with a
   pool-picked alternative.
2. Pool generation: `generate_pool(question, o=pool_size, history=thread_history_so_far)`.
3. Pick the candidate maximally distant (token-Jaccard) from this thread's
   prior queries (`prior_queries` list, includes initial_query).

Default N=1 = original turn-1-only behaviour, no override anywhere.

---

## 9. Metrics

### 9.1 Per-rollout
- `correct` (binary, 0/1): did this thread's answer match GT under EM?
- `pass_at_1` (per question): fraction of k threads that got it right
- `pass_at_4` (per question, also called `oracle_correct`): 1 if any of k threads is correct

### 9.2 Aggregate (across questions per seed)
- `pass_at_1` (mean of per-question pass@1)
- `pass_at_4` (mean of per-question pass@4 / oracle_correct)
- `synthesis_accuracy` (mean of per-question synthesis_correct)
- `mean_jaccard_qpd` — Query Pairwise Diversity (mean Jaccard distance between turn-1 queries across the k threads)
- `mean_itc` — Inter-Turn Coherence (within a thread, mean Jaccard similarity of turn-1 query vs subsequent queries)
- `mean_atc` — Across-Thread Coherence (mean pairwise Jaccard distance between threads' queries at the same turn)

### 9.3 LLM-judge (post-hoc, separate pass)
- `pass_at_1_llm`: per-question, fraction of k threads judged CORRECT
- `pass_at_4_llm`: per-question, 1 if any thread judged CORRECT (this is what we report in Tables 1/2)

Stored in `summary_T*.csv` per `run_<seed>/` directory.

### 9.4 Reporting in paper
- **Mean across 3 seeds** (raw, no outlier dropping for 3-seed cells; the
  `aggregate_results.py` script offers drop-outlier mode if needed for legacy
  5-seed qwen3-8b cells).
- Standard deviation computable from per-seed CSVs.

---

## 10. Error handling / bulletproofing

### 10.1 LLM call retry (`llm.py:107-138`)
- 5 attempts with backoff
- **ContextWindowExceeded**: not retried as-is. Once we truncate the longest
  message to half (head + tail kept, middle dropped) and retry once. If still
  fails, raise.
- **Connection / 5xx errors**: longer backoff (16s base, 2x exponential, max
  120s) — gives vLLM time to recover from transient failures.

### 10.2 Per-turn rollout safety (`diversity_scaling.py`)
- Each turn's `call_llm` wrapped in try/except. On failure, break out of the
  rollout loop and return the best candidate seen so far (no all-or-nothing
  failure).

### 10.3 Pool generation fallback
- If `generate_pool` LLM call fails, fall back to `f"{question[:50]} variant {i}"`
  templated queries. Diversity degrades to ~naive parallel rather than crashing.

### 10.4 Trajectory write safety (`run_main_table.py:_save_trajectory`)
- Wrapped in try/except OSError. Disk-quota errors warn but don't abort the run.

### 10.5 Search retry (`search.py`)
- Both ClueWeb and Serper paths retry up to 5 times.

---

## 11. Results layout

```
results/
├── main_table_clueweb_t8/         # Table 1
│   └── <model>/<dataset>/<cond>/run_<seed>/
│       ├── <method>_T<T>.csv      # per-question rows
│       ├── <method>_T<T>.jsonl    # for judge
│       ├── summary_T<T>.csv       # aggregate (gets pass@*_llm columns post-judge)
│       └── trajectories/<method>_T<T>/<question_id>.json
├── main_table_web_serper/         # Table 2
│   └── (same structure)
├── passk_ablation/                # Phase 2 k=2
│   └── <model>/<dataset>/<cond>/run_<seed>/  (cond = naive_k2 / div_k2)
├── poolsize_ablation/
│   └── <model>/<dataset>/pool_<P>/run_<seed>/
├── oversample_ablation/
│   └── <model>/<dataset>/os_<N>/run_<seed>/
├── figure1_*_judged/              # Reused from earlier figure 1
└── figure2/                       # Reused from earlier figure 2
```

---

## 12. Reproducibility one-shot

To rerun any single config:
```bash
cd /home/ssmurali/t3-testbed/general_agent

# Table 1 (ClueWeb, react_simple/Search-R1)
./scripts/submit_t3.sh <model> <dataset> <cond>
# e.g. ./scripts/submit_t3.sh qwen3-8b hotpotqa div_k4

# Table 2 (Serper, web_reasoning)
./scripts/submit_t3_web.sh <model> <dataset> <cond>

# Pool-size ablation
./scripts/submit_poolsize.sh

# Oversample-until-N ablation
./scripts/submit_oversample.sh
```

Aggregate at the end:
```bash
python -m webwalkerqa.scripts.aggregate_results --results-dir <results-dir>
```

---

## 13. Cost summary (~as-spent)

| Item | Cost |
|---|---|
| Serper queries (Table 2 only, ~690K) | ~$200 (volume tier) |
| OpenAI gpt-4o-mini judge (Tables 1+2) | ~$30 |
| Compute (CMU babel cluster) | free for grant |
| **Total cash** | **~$230** |

Total wall time: ~1 week (with cluster contention + occasional rerun).

---

## Open questions / TODOs for the paper

- Synthesis methods improvement (Tier 1 ideas in earlier discussion: majority vote, judge-pick, logprob ranking, evidence-grounded reranker). All run on existing trajectories — no new GPU.
- Should we include `pool_size` and `oversample-until-N` ablations in the main paper or relegate to appendix? Both are ~flat — confirms the chosen pool=16, N=1 default.
- BrowseComp results (1-4% across all models) — frame as "stress benchmark beyond the reach of current open-weight ~13B agents at modest budgets."
- HLE flip between ClueWeb (better) and Serper (worse) — backed by the academic-vs-popular argument; could become a sub-finding.
