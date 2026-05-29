#!/bin/bash
# Submit one SLURM job per (dataset × condition) for OpenAI API models.
# Each job loops 5 seeds sequentially = 5 runs per condition.
# After each experiment job: chains a judge job (--dependency=afterok).
# After all judge jobs: chains one aggregate job.
# Results go to: results/main_table_t12/{model}/{dataset}/{condition}/run_{seed}/
#
# Usage:
#   cd general_agent
#   ./scripts/submit_main_table.sh                          # gpt-4o-mini, all 24 jobs
#   ./scripts/submit_main_table.sh gpt-4.1-mini             # different model
#   ./scripts/submit_main_table.sh gpt-4o-mini hotpotqa     # one dataset
#   ./scripts/submit_main_table.sh gpt-4o-mini hotpotqa sequential  # single job

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/main_table logs/judge

DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
ALL_CONDITIONS=("sequential" "naive_parallel" "diversity_parallel")
SEEDS=(1 2 3 4 5)
ALL_DATASETS=("hotpotqa" "musique" "2wikimultihopqa" "bamboogle" "frames" "gaia" "hle" "webwalker")

CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
RUN_CMD="cd /home/ssmurali/t3-testbed/general_agent"

# Optional filters
MODEL_SHORT="${1:-gpt-4o-mini}"
FILTER_DATASET="${2:-}"
FILTER_COND="${3:-}"
POOL_SIZE="${4:-8}"  # diversity pool size (default 8, try 16 or 24)
TURNS="${5:-12}"    # turns per parallel thread (seq gets k× this)

RESULTS_BASE="/home/ssmurali/t3-testbed/results/main_table_t${TURNS}"

declare -A MODEL_LITELLM_MAP
MODEL_LITELLM_MAP["gpt-4o-mini"]="openai/gpt-4o-mini"
MODEL_LITELLM_MAP["gpt-4.1-mini"]="openai/gpt-4.1-mini"
MODEL_LITELLM_MAP["gemini-2.5-flash"]="gemini/gemini-2.5-flash"

if [[ -z "${MODEL_LITELLM_MAP[$MODEL_SHORT]+x}" ]]; then
  echo "Unknown model: ${MODEL_SHORT}. Add it to MODEL_LITELLM_MAP."
  exit 1
fi
LITELLM_MODEL="${MODEL_LITELLM_MAP[$MODEL_SHORT]}"

JOB_COUNT=0
JUDGE_JOB_IDS=()

for DATASET in "${ALL_DATASETS[@]}"; do
  [[ -n "$FILTER_DATASET" && "$DATASET" != "$FILTER_DATASET" ]] && continue

  DATASET_PATH="${DATA_DIR}/${DATASET}.json"
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "SKIP: ${DATASET_PATH} not found"
    continue
  fi

  for COND in "${ALL_CONDITIONS[@]}"; do
    [[ -n "$FILTER_COND" && "$COND" != "$FILTER_COND" ]] && continue

    # ── Experiment job (5 seeds sequential) ──────────────────────────────────
    INNER=""
    for SEED in "${SEEDS[@]}"; do
      OUTDIR="${RESULTS_BASE}/${MODEL_SHORT}/${DATASET}/${COND}/run_${SEED}"
      INNER+="echo '--- ${DATASET} | ${COND} | seed=${SEED} ---' && "
      INNER+="python -m webwalkerqa.run.run_main_table"
      INNER+=" --model ${LITELLM_MODEL}"
      INNER+=" --dataset ${DATASET_PATH}"
      INNER+=" --condition ${COND}"
      INNER+=" --seed ${SEED}"
      INNER+=" --max-turns-par ${TURNS}"
      INNER+=" --max-concurrent 100"
      INNER+=" --pool-size ${POOL_SIZE}"
      INNER+=" --output-dir ${OUTDIR}"
      INNER+=" && "
    done
    INNER+="echo 'Done: ${DATASET}/${COND}'"

    JOB_NAME="t3-${DATASET:0:8}-${COND:0:3}"
    EXP_JOB_ID=$(sbatch \
      --job-name="${JOB_NAME}" \
      --partition=general \
      --gres=gpu:1 \
      --time=24:00:00 \
      --mem=16G \
      --cpus-per-task=4 \
      --export=ALL \
      --output="logs/main_table/${MODEL_SHORT}_${DATASET}_${COND}_%j.out" \
      --error="logs/main_table/${MODEL_SHORT}_${DATASET}_${COND}_%j.err" \
      --wrap="${CONDA_INIT} && ${RUN_CMD} && ${INNER}" \
      --parsable)

    echo "Submitted exp:   ${JOB_NAME} (job=${EXP_JOB_ID})"

    # ── Judge job (chained after experiment) ─────────────────────────────────
    JUDGE_CMD="python -m webwalkerqa.judge.eval_llm"
    JUDGE_CMD+=" --results-dir ${RESULTS_BASE}"
    JUDGE_CMD+=" --filter-model ${MODEL_SHORT}"
    JUDGE_CMD+=" --filter-dataset ${DATASET}"
    JUDGE_CMD+=" --filter-condition ${COND}"
    JUDGE_CMD+=" --model openai/gpt-4o-mini"
    JUDGE_CMD+=" --max-concurrent 200"

    JUDGE_JOB_NAME="t3j-${DATASET:0:8}-${COND:0:3}"
    JUDGE_JOB_ID=$(sbatch \
      --job-name="${JUDGE_JOB_NAME}" \
      --partition=general \
      --gres=gpu:1 \
      --time=4:00:00 \
      --mem=4G \
      --cpus-per-task=2 \
      --export=ALL \
      --dependency=afterok:${EXP_JOB_ID} \
      --output="logs/judge/${MODEL_SHORT}_${DATASET}_${COND}_%j.out" \
      --error="logs/judge/${MODEL_SHORT}_${DATASET}_${COND}_%j.err" \
      --wrap="${CONDA_INIT} && ${RUN_CMD} && ${JUDGE_CMD}" \
      --parsable)

    echo "Submitted judge: ${JUDGE_JOB_NAME} (job=${JUDGE_JOB_ID}, after=${EXP_JOB_ID})"
    JUDGE_JOB_IDS+=("${JUDGE_JOB_ID}")
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

# ── Final aggregate job (after all judge jobs) ────────────────────────────────
if [[ ${#JUDGE_JOB_IDS[@]} -gt 0 ]]; then
  DEP_LIST=$(IFS=:; echo "${JUDGE_JOB_IDS[*]}")
  AGG_CMD="python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
  AGG_JOB_ID=$(sbatch \
    --job-name="t3-aggregate" \
    --partition=general \
    --gres=gpu:1 \
    --time=0:30:00 \
    --mem=4G \
    --cpus-per-task=2 \
    --export=ALL \
    --dependency=afterok:${DEP_LIST} \
    --output="logs/aggregate_%j.out" \
    --error="logs/aggregate_%j.err" \
    --wrap="${CONDA_INIT} && ${RUN_CMD} && ${AGG_CMD}" \
    --parsable)
  echo ""
  echo "Submitted aggregate: job=${AGG_JOB_ID} (after all ${#JUDGE_JOB_IDS[@]} judge jobs)"
fi

echo ""
echo "Submitted ${JOB_COUNT} experiment+judge pairs. Results → ${RESULTS_BASE}"
echo "Monitor: squeue -u \$USER"
