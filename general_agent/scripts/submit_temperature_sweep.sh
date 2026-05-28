#!/bin/bash
# Temperature sweep for qwen3-8b, naive_parallel only, 3 datasets × 4 temps.
#
# Per-job pattern:
#   - 1 GPU, launches its own vLLM on port 8000+(SLURM_JOB_ID % 1000)
#   - runs 3 seeds sequentially
#   - inline judge
#   - kills vLLM
#
# Backend routing:
#   GAIA       -> SEARCH_BACKEND=serper, --prompt-style web_reasoning
#   hotpotqa   -> SEARCH_BACKEND=clueweb (default), --prompt-style react_simple
#   bamboogle  -> SEARCH_BACKEND=clueweb (default), --prompt-style react_simple
#
# Total: 3 datasets × 4 temperatures = 12 SLURM jobs.
#
# Usage:
#   cd general_agent
#   ./scripts/submit_temperature_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/temperature_sweep logs/vllm

# ───────────────────────────── config ─────────────────────────────
MODEL_SHORT="qwen3-8b"
HF_MODEL="Qwen/Qwen3-8B"
LITELLM_MODEL="openai/${HF_MODEL}"

DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/temperature_sweep_qwen3_8b"

CONDITION="naive_parallel"
TEMPERATURES=(0.5 1.0 1.5 2.0)
SEEDS=(1 2 3)
K=4
TURNS=8
POOL_SIZE=16
MAX_TOK=2048

# (dataset, backend, prompt_style, judge_prompt_style)
JOB_LIST=(
  "hotpotqa:clueweb:react_simple:mhqa"
  "bamboogle:clueweb:react_simple:mhqa"
  "GAIA:serper:web_reasoning:web"
)

# ───────────────────────────── env templates ─────────────────────────────
# CLUEWEB env: do NOT export SEARCH_BACKEND (it defaults from .env / clueweb)
CONDA_INIT_CLUEWEB="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"

# SERPER env: override SEARCH_BACKEND inline (after .env load)
CONDA_INIT_SERPER="${CONDA_INIT_CLUEWEB} && export SEARCH_BACKEND=serper"

CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

# ───────────────────────────── submit loop ─────────────────────────────
JOB_COUNT=0
for ENTRY in "${JOB_LIST[@]}"; do
  IFS=':' read -r DATASET BACKEND PROMPT_STYLE JUDGE_PS <<< "$ENTRY"

  DATASET_PATH="${DATA_DIR}/${DATASET}.json"
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "SKIP: ${DATASET_PATH} not found"; continue
  fi

  if [[ "$BACKEND" == "serper" ]]; then
    CONDA_INIT="$CONDA_INIT_SERPER"
  else
    CONDA_INIT="$CONDA_INIT_CLUEWEB"
  fi

  for TEMP in "${TEMPERATURES[@]}"; do

    OUTDIR_BASE="${RESULTS_BASE}/${MODEL_SHORT}/${DATASET}/${CONDITION}/temp_${TEMP}"

    # Build inline script: vllm up, 3 seeds, inline judge, vllm down
    INNER="VLLM_PORT=\$((8000 + (\$SLURM_JOB_ID % 1000)))"
    INNER+="; VLLM_LOG=/tmp/vllm_\${SLURM_JOB_ID}.log"
    INNER+="; vllm serve ${HF_MODEL} --port \$VLLM_PORT --enable-prefix-caching --dtype auto --max-model-len 32768 --max-num-seqs 128 --enforce-eager --disable-log-stats > \$VLLM_LOG 2>&1 &"
    INNER+=" VLLM_PID=\$!"
    INNER+="; echo '[vllm] started pid='\$VLLM_PID' port='\$VLLM_PORT', waiting for ready...'"
    INNER+="; READY=0"
    INNER+="; for i in \$(seq 1 240); do if curl -sf http://localhost:\$VLLM_PORT/v1/models > /dev/null 2>&1; then echo \"[vllm] ready after \${i}0s\"; READY=1; break; fi; sleep 10; done"
    INNER+="; if [[ \$READY -eq 0 ]]; then echo '[vllm] FAILED to come up. Log tail:'; tail -50 \$VLLM_LOG; kill \$VLLM_PID 2>/dev/null || true; exit 1; fi"
    INNER+="; trap 'kill \$VLLM_PID 2>/dev/null || true' EXIT"

    for SEED in "${SEEDS[@]}"; do
      OUTDIR="${OUTDIR_BASE}/run_${SEED}"
      INNER+="; echo '--- ${MODEL_SHORT} | ${DATASET} | ${CONDITION} | τ=${TEMP} | seed=${SEED} | backend=${BACKEND} ---'"
      INNER+="; python -m webwalkerqa.run.run_main_table"
      INNER+=" --model ${LITELLM_MODEL}"
      INNER+=" --dataset ${DATASET_PATH}"
      INNER+=" --condition ${CONDITION}"
      INNER+=" --seed ${SEED}"
      INNER+=" --temperature ${TEMP}"
      INNER+=" --max-turns-par ${TURNS}"
      INNER+=" --k ${K}"
      INNER+=" --max-tokens ${MAX_TOK}"
      INNER+=" --pool-size ${POOL_SIZE}"
      INNER+=" --max-concurrent 100"
      INNER+=" --api-base http://localhost:\$VLLM_PORT/v1"
      INNER+=" --output-dir ${OUTDIR}"
      INNER+=" --prompt-style ${PROMPT_STYLE}"
      INNER+=" || echo '[warn] seed ${SEED} exited nonzero, continuing'"
    done

    # Inline judge for this (dataset, temp) cell
    INNER+="; echo '--- judging ${MODEL_SHORT}/${DATASET}/${CONDITION}/τ=${TEMP} ---'"
    INNER+="; python -m webwalkerqa.judge.eval_llm"
    INNER+=" --results-dir ${RESULTS_BASE}"
    INNER+=" --filter-model ${MODEL_SHORT}"
    INNER+=" --filter-dataset ${DATASET}"
    INNER+=" --filter-condition ${CONDITION}"
    INNER+=" --model openai/gpt-4o-mini"
    INNER+=" --max-concurrent 200"
    INNER+=" --force"
    INNER+=" --judge-prompt-style ${JUDGE_PS}"
    INNER+=" || echo '[warn] judge failed; rerun manually later'"

    INNER+="; kill \$VLLM_PID 2>/dev/null || true"
    INNER+="; wait \$VLLM_PID 2>/dev/null || true"

    JNAME="tsw-${DATASET:0:5}-t${TEMP}"
    EXP_JOB_ID=$(sbatch \
      --job-name="${JNAME}" \
      --partition=general \
      --gres=gpu:1 \
      --time=24:00:00 \
      --mem=32G \
      --cpus-per-task=8 \
      --export=ALL \
      --output="logs/temperature_sweep/${DATASET}_t${TEMP}_%j.out" \
      --error="logs/temperature_sweep/${DATASET}_t${TEMP}_%j.err" \
      --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" \
      --parsable)
    echo "Submitted: ${JNAME}  job=${EXP_JOB_ID}  backend=${BACKEND}  prompt=${PROMPT_STYLE}"
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

echo
echo "Total jobs submitted: ${JOB_COUNT}"
echo "Results: ${RESULTS_BASE}/"
echo "Monitor: squeue -u \$USER"
