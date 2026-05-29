#!/bin/bash
# Submit Table-2 jobs (Serper-backed, web-reasoning prompt, harder datasets).
# Each job: self-hosted vLLM + 3 seeds + inline judge (web-prompt style).
# Aggregate run manually at end.
#
# Usage:
#   ./scripts/submit_t3_web.sh <model> <dataset> <cond>
#   ./scripts/submit_t3_web.sh <model> all                  # all 5 conds × 4 datasets
#   ./scripts/submit_t3_web.sh <model> all <cond>           # one cond × all 4 datasets
#   ./scripts/submit_t3_web.sh <model> <dataset> all        # all 5 conds × 1 dataset
#
# Conditions (Table-2 budgets):
#   seq       k=1  T=25  max_tok=8192  (capped 25 turns)
#   naive_k4  k=4  T=8   max_tok=8192
#   div_k4    k=4  T=8   max_tok=8192
#   naive_k8  k=8  T=8   max_tok=4096  (halved to match compute)
#   div_k8    k=8  T=8   max_tok=4096
#
# Datasets: webwalker_sub (250) | hle_sub (250) | gaia_full (103) | browsecomp_sub (250).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/web_main_table

declare -A HF_MAP
HF_MAP["qwen3-4b"]="Qwen/Qwen3-4B"
HF_MAP["qwen3-8b"]="Qwen/Qwen3-8B"
HF_MAP["gemma3-12b"]="google/gemma-3-12b-it"
HF_MAP["gpt-oss-20b"]="openai/gpt-oss-20b"

# Dataset-short → file path (subsampled files)
declare -A DS_MAP
DS_MAP["webwalker"]="webwalker_sub.json"
DS_MAP["hle"]="hle_sub.json"
DS_MAP["gaia"]="gaia.json"
DS_MAP["browsecomp"]="browsecomp_sub.json"

ALL_DATASETS=("webwalker" "hle" "gaia" "browsecomp")
PHASE1_CONDS=("seq" "naive_k4" "div_k4" "naive_k8" "div_k8")
DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/main_table_web_serper"
SEEDS=(1 2 3)

cond_params() {
  local C=$1
  case "$C" in
    seq)      echo "1 25 8192 sequential" ;;
    naive_k4) echo "4 8  8192 naive_parallel" ;;
    div_k4)   echo "4 8  8192 diversity_parallel" ;;
    naive_k8) echo "8 8  4096 naive_parallel" ;;
    div_k8)   echo "8 8  4096 diversity_parallel" ;;
    *) echo "ERROR unknown cond '$C'" >&2; return 1 ;;
  esac
}

MODEL_SHORT="${1:-}"; ARG2="${2:-}"; ARG3="${3:-}"
[[ -z "$MODEL_SHORT" ]] && { echo "Usage: $0 <model> <dataset|all> [cond|all]"; exit 1; }
[[ -z "${HF_MAP[$MODEL_SHORT]:-}" ]] && { echo "ERROR unknown model '$MODEL_SHORT'"; exit 1; }
HF_MODEL="${HF_MAP[$MODEL_SHORT]}"
LITELLM_MODEL="openai/${HF_MODEL}"

DS_LIST=(); [[ "$ARG2" == "all" ]] && DS_LIST=("${ALL_DATASETS[@]}") || DS_LIST=("$ARG2")
CD_LIST=(); [[ -z "$ARG3" || "$ARG3" == "all" ]] && CD_LIST=("${PHASE1_CONDS[@]}") || CD_LIST=("$ARG3")

# SEARCH_BACKEND=serper set inline (overrides .env which may still say clueweb)
CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a && export SEARCH_BACKEND=serper"
CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

JOB_COUNT=0
for DS in "${DS_LIST[@]}"; do
  DSFILE="${DS_MAP[$DS]:-}"
  [[ -z "$DSFILE" ]] && { echo "SKIP unknown dataset '$DS'"; continue; }
  DATASET_PATH="${DATA_DIR}/${DSFILE}"
  [[ ! -f "$DATASET_PATH" ]] && { echo "SKIP missing $DATASET_PATH (did you run prepare_web_datasets.py?)"; continue; }

  for COND in "${CD_LIST[@]}"; do
    read -r K T MTOK METHOD < <(cond_params "$COND")
    OUTDIR_BASE="${RESULTS_BASE}/${MODEL_SHORT}/${DS}/${COND}"

    INNER="VLLM_PORT=\$((8000 + (\$SLURM_JOB_ID % 1000)))"
    INNER+="; VLLM_LOG=/tmp/vllm_\${SLURM_JOB_ID}.log"
    INNER+="; vllm serve ${HF_MODEL} --port \$VLLM_PORT --enable-prefix-caching --dtype auto --max-model-len 32768 --max-num-seqs 128 --enforce-eager --disable-log-stats > \$VLLM_LOG 2>&1 &"
    INNER+=" VLLM_PID=\$!"
    INNER+="; echo '[vllm] pid='\$VLLM_PID' port='\$VLLM_PORT"
    INNER+="; READY=0"
    INNER+="; for i in \$(seq 1 240); do if curl -sf http://localhost:\$VLLM_PORT/v1/models > /dev/null 2>&1; then echo \"[vllm] ready \${i}0s\"; READY=1; break; fi; sleep 10; done"
    INNER+="; if [[ \$READY -eq 0 ]]; then tail -50 \$VLLM_LOG; kill \$VLLM_PID 2>/dev/null || true; exit 1; fi"
    INNER+="; trap 'kill \$VLLM_PID 2>/dev/null || true' EXIT"
    for SEED in "${SEEDS[@]}"; do
      OUTDIR="${OUTDIR_BASE}/run_${SEED}"
      INNER+="; echo '--- ${MODEL_SHORT} | ${DS} | ${COND} (k=${K} T=${T} tok=${MTOK}) | seed=${SEED} ---'"
      INNER+="; python -m webwalkerqa.run.run_main_table --model ${LITELLM_MODEL} --dataset ${DATASET_PATH} --condition ${METHOD} --seed ${SEED} --max-turns-par ${T} --k ${K} --max-tokens ${MTOK} --pool-size 16 --max-concurrent 100 --api-base http://localhost:\$VLLM_PORT/v1 --output-dir ${OUTDIR} --prompt-style web_reasoning || echo '[warn] seed ${SEED} nonzero'"
    done
    INNER+="; echo 'Done seeds: ${MODEL_SHORT}/${DS}/${COND}'"
    INNER+="; echo '--- judging (web style) ${MODEL_SHORT}/${DS}/${COND} ---'"
    INNER+="; python -m webwalkerqa.judge.eval_llm --results-dir ${RESULTS_BASE} --filter-model ${MODEL_SHORT} --filter-dataset ${DS} --filter-condition ${COND} --model openai/gpt-4o-mini --max-concurrent 200 --force --judge-prompt-style web || echo '[warn] judge failed'"
    INNER+="; kill \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true"

    JNAME="w3-${MODEL_SHORT:0:6}-${DS:0:6}-${COND}"
    EXP=$(sbatch \
      --job-name="${JNAME}" --partition=general --gres=gpu:1 \
      --time=24:00:00 --mem=32G --cpus-per-task=8 --export=ALL \
      --output="logs/web_main_table/w3_${MODEL_SHORT}_${DS}_${COND}_%j.out" \
      --error="logs/web_main_table/w3_${MODEL_SHORT}_${DS}_${COND}_%j.err" \
      --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" --parsable)
    echo "Submitted exp+judge (web): ${JNAME} job=${EXP}"
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

echo ""
echo "Total: ${JOB_COUNT} exp+judge jobs"
echo "Aggregate (manual at end):"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
