# GAIA-103 Scaling Experiment

**What it does:** Runs 9 conditions on 103 GAIA questions. Each question gets 4 ReAct rollouts (12 turns each). Outputs: one JSONL per condition + `summary.csv` with pass@1 and pass@4.

**Conditions:** `naive-t4` (4 independent rollouts, no diversity) · `jaccard-o{16,32,48,64}` (pool o, Jaccard max-min → 4 seeds, 4 threads) · `dense-o{16,32,48,64}` (same with MiniLM embeddings).

---

## Prerequisites

- Python env with deps (see repo). Activate it (e.g. `conda activate t3`).
- **Env vars** (in `general_agent/.env` or shell): `OPENAI_API_KEY`, `SERPER_API_KEY`.
- Dataset: `general_agent/data/GAIA.json` (103 questions). Default in script.

---

## Run locally

From **`general_agent`** (so `webwalkerqa` resolves):

```bash
cd general_agent
export PYTHONPATH="$PWD:$PYTHONPATH"
```

**Single condition:**
```bash
python -m webwalkerqa.scaling_v2_experiment --condition naive-t4
```

**All 9 conditions (sequential):**
```bash
python -m webwalkerqa.scaling_v2_experiment --all
```

**Options:** `--max-concurrent 100` (default), `--model openai/gpt-4o-mini`, `--dataset path/to/GAIA.json`, `--output-dir path/to/results`.

**Resume:** Partial runs append to the condition JSONL; re-run the same command to skip completed questions.

---

## Run on SLURM

From **`general_agent`**:

```bash
./submit_scaling_v2.sh
```

Submits 9 jobs (one per condition). Concurrency per job: `MAX_CONCURRENT=100` (override with `MAX_CONCURRENT=50 ./submit_scaling_v2.sh`).

**Single condition:**
```bash
sbatch run_scaling_v2.sbatch naive-t4 100
```

**Monitor:** `squeue -u $USER`. Logs: `logs/sbatch/scaling_v2_<jobid>.out` / `.err`.

**Note:** `run_scaling_v2.sbatch` has hardcoded paths and `conda activate t3`. Edit `cd`, `PYTHONPATH`, and `conda activate` if your layout differs.

---

## Outputs

- **Dir:** `general_agent/results/gaia_103/` (or `--output-dir`).
- **Per condition:** `{condition_id}.jsonl` — one JSON object per line: `question_id`, `question`, `answer_gt`, `rollout_answers` (4 strings), `pass_at_1`, `pass_at_4`.
- **Summary:** `summary.csv` — columns `condition`, `pass@1`, `pass@4`, `n_questions`.

**Aggregate only** (after jobs finish, to refresh summary):
```bash
cd general_agent
python -m webwalkerqa.scaling_v2_experiment --aggregate-only
```

---

## Metrics

- **pass@1:** First rollout correct (exact match vs ground truth).
- **pass@4:** Any of the 4 rollouts correct.
