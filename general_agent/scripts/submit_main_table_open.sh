#!/bin/bash
# Submit one SLURM job per (model × dataset × condition) for OSS models via vLLM.
# Each job loops 5 seeds sequentially = 5 runs per condition.
# After each experiment job: chains a judge job (--dependency=afterok).
# After all judge jobs: chains one aggregate job.
# Results go to: results/main_table_t12/{model}/{dataset}/{condition}/run_{seed}/
#
# Requires vLLM servers running — launch with:
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-8B  8003
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-4B  8002
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-1.7B 8001
#   ./scripts/launch_vllm_server.sh openai/gpt-oss-20b 8004
#
# Usage:
#   cd general_agent
#   ./scripts/submit_main_table_open.sh                          # all models
#   ./scripts/submit_main_table_open.sh qwen3-8b                 # one model
#   ./scripts/submit_main_table_open.sh qwen3-8b hotpotqa        # one model, one dataset
#   ./scripts/submit_main_table_open.sh qwen3-8b hotpotqa sequential  # single job

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/main_table logs/judge

DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
ENDPOINT_DIR="/home/ssmurali/t3-testbed/general_agent/results/vllm_servers"
ALL_CONDITIONS=("sequential" "naive_parallel" "diversity_parallel")
SEEDS=(1 2 3 4 5)
ALL_DATASETS=("hotpotqa" "musique" "2wikimultihopqa" "bamboogle" "frames" "gaia" "hle" "webwalker")

# model-short → HF model name (must match what was passed to launch_vllm_server.sh)
declare -A MODEL_HF_MAP
MODEL_HF_MAP["qwen3-1.7b"]="Qwen/Qwen3-1.7B"
MODEL_HF_MAP["qwen3-4b"]="Qwen/Qwen3-4B"
MODEL_HF_MAP["qwen3-8b"]="Qwen/Qwen3-8B"
MODEL_HF_MAP["gpt-oss-20b"]="openai/gpt-oss-20b"
ALL_MODELS=("qwen3-1.7b" "qwen3-4b" "qwen3-8b" "gpt-oss-20b")

# Experiment jobs: vLLM doesn't need a real key
CONDA_INIT_EXP="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
# Judge jobs: need real OPENAI_API_KEY
CONDA_INIT_JUDGE="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
RUN_CMD="cd /home/ssmurali/t3-testbed/general_agent"

FILTER_MODEL="${1:-}"
FILTER_DATASET="${2:-}"
FILTER_COND="${3:-}"
POOL_SIZE="${4:-16}"
TURNS="${5:-8}"

RESULTS_BASE="/home/ssmurali/t3-testbed/results/main_table_clueweb_t${TURNS}"

JOB_COUNT=0
JUDGE_JOB_IDS=()

for MODEL_SHORT in "${ALL_MODELS[@]}"; do
  [[ -n "$FILTER_MODEL" && "$MODEL_SHORT" != "$FILTER_MODEL" ]] && continue

  ENDPOINT_FILE="${ENDPOINT_DIR}/${MODEL_SHORT}.endpoint"
  if [[ ! -f "$ENDPOINT_FILE" ]]; then
    HF_NAME="${MODEL_HF_MAP[$MODEL_SHORT]}"
    echo "ERROR: No endpoint for ${MODEL_SHORT}."
    echo "  Run: ./scripts/launch_vllm_server.sh ${HF_NAME} <port>"
    exit 1
  fi
  API_BASE=$(cat "$ENDPOINT_FILE")
  HF_MODEL="${MODEL_HF_MAP[$MODEL_SHORT]}"
  LITELLM_MODEL="openai/${HF_MODEL}"

  for DATASET in "${ALL_DATASETS[@]}"; do
    [[ -n "$FILTER_DATASET" && "$DATASET" != "$FILTER_DATASET" ]] && continue

    DATASET_PATH="${DATA_DIR}/${DATASET}.json"
    if [[ ! -f "$DATASET_PATH" ]]; then
      echo "SKIP: ${DATASET_PATH} not found"
      continue
    fi

    for COND in "${ALL_CONDITIONS[@]}"; do
      [[ -n "$FILTER_COND" && "$COND" != "$FILTER_COND" ]] && continue

      # ── Experiment job (5 seeds sequential) ──────────────────────────────
      INNER=""
      for SEED in "${SEEDS[@]}"; do
        OUTDIR="${RESULTS_BASE}/${MODEL_SHORT}/${DATASET}/${COND}/run_${SEED}"
        INNER+="echo '--- ${MODEL_SHORT} | ${DATASET} | ${COND} | seed=${SEED} ---' && "
        INNER+="python -m webwalkerqa.run.run_main_table"
        INNER+=" --model ${LITELLM_MODEL}"
        INNER+=" --dataset ${DATASET_PATH}"
        INNER+=" --condition ${COND}"
        INNER+=" --seed ${SEED}"
        INNER+=" --max-turns-par ${TURNS}"
        INNER+=" --pool-size ${POOL_SIZE}"
        INNER+=" --max-concurrent 100"
        INNER+=" --api-base ${API_BASE}"
        INNER+=" --output-dir ${OUTDIR}"
        INNER+=" && "
      done
      INNER+="echo 'Done: ${MODEL_SHORT}/${DATASET}/${COND}'"

      JOB_NAME="t3o-${MODEL_SHORT:0:6}-${DATASET:0:6}-${COND:0:3}"
      EXP_JOB_ID=$(sbatch \
        --job-name="${JOB_NAME}" \
        --partition=general \
        --gres=gpu:1 \
        --time=24:00:00 \
        --mem=8G \
        --cpus-per-task=4 \
        --export=ALL \
        --output="logs/main_table/${MODEL_SHORT}_${DATASET}_${COND}_%j.out" \
        --error="logs/main_table/${MODEL_SHORT}_${DATASET}_${COND}_%j.err" \
        --wrap="${CONDA_INIT_EXP} && ${RUN_CMD} && ${INNER}" \
        --parsable)

      echo "Submitted exp:   ${JOB_NAME} (job=${EXP_JOB_ID}, api=${API_BASE})"

      # ── Judge job (chained after experiment) ────────────────────────────
      JUDGE_CMD="python -m webwalkerqa.judge.eval_llm"
      JUDGE_CMD+=" --results-dir ${RESULTS_BASE}"
      JUDGE_CMD+=" --filter-model ${MODEL_SHORT}"
      JUDGE_CMD+=" --filter-dataset ${DATASET}"
      JUDGE_CMD+=" --filter-condition ${COND}"
      JUDGE_CMD+=" --model openai/gpt-4o-mini"
      JUDGE_CMD+=" --max-concurrent 200"

      JUDGE_JOB_NAME="t3j-${MODEL_SHORT:0:6}-${DATASET:0:6}-${COND:0:3}"
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
        --wrap="${CONDA_INIT_JUDGE} && ${RUN_CMD} && ${JUDGE_CMD}" \
        --parsable)

      echo "Submitted judge: ${JUDGE_JOB_NAME} (job=${JUDGE_JOB_ID}, after=${EXP_JOB_ID})"
      JUDGE_JOB_IDS+=("${JUDGE_JOB_ID}")
      JOB_COUNT=$((JOB_COUNT + 1))
    done
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
    --wrap="${CONDA_INIT_JUDGE} && ${RUN_CMD} && ${AGG_CMD}" \
    --parsable)
  echo ""
  echo "Submitted aggregate: job=${AGG_JOB_ID} (after all ${#JUDGE_JOB_IDS[@]} judge jobs)"
fi

echo ""
echo "Submitted ${JOB_COUNT} experiment+judge pairs. Results → ${RESULTS_BASE}"
echo "Monitor: squeue -u \$USER"
