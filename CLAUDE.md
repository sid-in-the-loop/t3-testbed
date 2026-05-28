# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**DivInit** — a training-free method for diverse query initialization in parallel agentic search. EMNLP 2026 short paper. Core idea: parallel search agents suffer from "anchor collapse" (threads issue near-identical turn-1 queries, fail in correlated ways). DivInit fixes this by selecting maximally diverse queries from a candidate pool using greedy max-min Jaccard distance.

Authors: Sid Murali, Ethan, João Coelho (advisor). CMU Language Technologies Institute.

## Setup

```bash
cd general_agent
pip install -e .
pip install litellm sentence-transformers httpx python-dotenv
```

Create `general_agent/.env` with:
```
OPENAI_API_KEY=...
SERPER_API_KEY=...
GEMINI_API_KEY=...
```

## Running experiments

**Single experiment:**
```bash
cd general_agent
python -m webwalkerqa.run.run_main_table \
  --model openai/gpt-4o-mini \
  --dataset data/hotpotqa_25.jsonl \
  --condition diversity_parallel \
  --output-dir ../results/main_table/gpt-4o-mini/hotpotqa
```

**SLURM batch launch:**
```bash
cd general_agent
bash scripts/submit_main_table.sh          # closed models
bash scripts/submit_main_table_open.sh     # open models via vLLM
```

**vLLM server** (for open models):
```bash
bash scripts/launch_vllm_server.sh <model_name> <port>
```

**Aggregate results:**
```bash
python -m webwalkerqa.scripts.aggregate_results --results-dir ../results/main_table
```

## Architecture

### Core agent loop (`webwalkerqa/methods/diversity_scaling.py`)

`run_single_rollout()` runs one complete ReAct-style trajectory: at each turn, call LLM → extract `<search>` or `<answer>` tag → if search, call Serper API → append to context. Runs up to `max_turns` (default 12).

Three scaling strategies:
- **SequentialMethod** (`naive_parallel.py`) — k=8 independent rollouts, no diversity, used as baseline
- **DiversityParallelMethod** (`diversity_parallel.py`) — generate candidate pool, greedily select k=4 diverse seeds (Jaccard or dense), run k=4 threads in parallel from those seeds
- **Diversity1Method** / **DiversityAllMethod** — turn-1-only vs. every-turn oversampling (ablations)

### Diversity selection (`webwalkerqa/methods/utils.py`)

`select_diverse_queries(queries, k, method)`:
- **Jaccard mode**: token-level Jaccard distance (lowercase, strip punctuation, whitespace-split)
- **Dense mode**: sentence-transformers cosine distance (all-MiniLM-L6-v2)
- **Greedy**: iteratively pick candidate with largest minimum distance to already-selected set

### LLM routing (`webwalkerqa/llm.py`)

Thin LiteLLM wrapper with 5-retry exponential backoff, context-overflow truncation, model aliases (e.g., `"gpt4o-mini"` → `"openai/gpt-4o-mini"`), and model-specific fixes (disable Qwen3 thinking, set Gemini output tokens).

### Evaluation (`webwalkerqa/eval.py`)

- `exact_match()`: normalize → lowercase, strip, collapse whitespace; handles multi-answer ground truth via `<|answer_split|>` delimiter
- F1: token-level precision/recall overlap
- **QPD** (Query Path Diversity): primary diagnostic — Jaccard similarity across thread turn-1 queries
- **pass@k**: oracle metric — any of k threads correct → pass

### Experiment conditions (`webwalkerqa/configs.py`)

| Condition | Method | Threads (k) | Pool size | Selection |
|-----------|--------|-------------|-----------|-----------|
| naive-t4 | Sequential | 4 | — | — |
| jaccard-o{8,16,32,48,64} | DiversityParallel | 4 | 8–64 | Greedy Jaccard |
| dense-o{8,16,32,48,64} | DiversityParallel | 4 | 8–64 | Dense (MiniLM) |

## Key concepts

- **Anchor collapse**: parallel threads issue near-identical turn-1 queries → overlapping retrieved docs → correlated failures
- **DivInit**: greedy max-min selection from k×m candidate pool to enforce diversity
- **QPD**: token-level Jaccard similarity across thread turn-1 queries — primary diagnostic
- **pass@k**: oracle metric (any thread correct = pass); paper primary metric

## Models

- Open (via vLLM): Qwen3-1.7B, 4B, 8B; Gemma3-4B, 12B
- Closed: gpt-4o-mini (OpenAI API), gemini-2.5-flash
- **gpt-oss-20B**: suspected eval bug — excluded from paper, ignore those result dirs

## Benchmarks

HotpotQA, MuSiQue, 2WikiMultihopQA, Bamboogle, FRAMES, GAIA, WebWalkerQA. Dataset files in `general_agent/data/` as `.jsonl`.

## Results layout

```
results/
  main_table/      # per-model, per-benchmark CSVs and JSONLs
  figure1/         # GAIA-103 sequential vs. parallel scaling
  figure2/         # diversity sweep
  passk_ablation/
  poolsize_ablation/
  oversample_ablation/
paper_assets/
  figures/         # final paper figures (PNGs/PDFs)
  scripts/         # figure generation code
```

## Branch strategy

- `main` / `divInit` — raw research snapshot, do not clean up
- `arxiv` — cleaned public release branch (delete logs, internal notes, stale results)
