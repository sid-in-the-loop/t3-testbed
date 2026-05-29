#!/bin/bash
# Self-hosted-vLLM submit script for T3 main-table + pass@k experiments.
# Each SLURM job: 1 GPU, launches vLLM locally, runs 5 seeds, kills vLLM, chains a judge job.
#
# Usage:
#   ./scripts/submit_t3.sh <model> <dataset> <cond>
#   ./scripts/submit_t3.sh <model> all                  # all 5 Phase-1 conds × 8 datasets
#   ./scripts/submit_t3.sh <model> all <cond>           # one cond × 8 datasets
#   ./scripts/submit_t3.sh <model> <dataset> all        # all 5 Phase-1 conds × 1 dataset
#   ./scripts/submit_t3.sh qwen3-8b finish              # the 7 finish-broken jobs for 8B
#
# Valid values:
#   model:   qwen3-1.7b | qwen3-4b | qwen3-8b | gpt-oss-20b
#   dataset: hotpotqa | musique | 2wikimultihopqa | bamboogle | frames | GAIA | hle | webwalker | all
#   cond:    seq | naive_k4 | div_k4 | naive_k8 | div_k8 | naive_k2 | div_k2 | all
#
# Budget matching (total = 65,536 tokens/rollout):
#   seq      k=1 T=32 max_tok=2048
#   naive_k2 k=2 T=16 max_tok=2048
#   div_k2   k=2 T=16 max_tok=2048
#   naive_k4 k=4 T=8  max_tok=2048
#   div_k4   k=4 T=8  max_tok=2048
#   naive_k8 k=8 T=8  max_tok=1024
#   div_k8   k=8 T=8  max_tok=1024

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/main_table logs/judge logs/vllm

# ───────────────────────────── config ─────────────────────────────
declare -A HF_MAP
HF_MAP["qwen3-1.7b"]="Qwen/Qwen3-1.7B"
HF_MAP["qwen3-4b"]="Qwen/Qwen3-4B"
HF_MAP["qwen3-8b"]="Qwen/Qwen3-8B"
HF_MAP["gpt-oss-20b"]="openai/gpt-oss-20b"
HF_MAP["gemma3-4b"]="google/gemma-3-4b-it"
HF_MAP["gemma3-12b"]="google/gemma-3-12b-it"

ALL_DATASETS=("hotpotqa" "musique" "2wikimultihopqa" "bamboogle" "frames" "gaia" "hle")
PHASE1_CONDS=("seq" "naive_k4" "div_k4" "naive_k8" "div_k8")
DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/main_table_clueweb_t8"
PHASE2_BASE="/home/ssmurali/t3-testbed/results/passk_ablation"
SEEDS=(1 2 3)

# Translate condition → (k, T, max_tok, method_name, result_base)
cond_params() {
  local COND=$1
  case "$COND" in
    seq)      echo "1 12 2048 sequential $RESULTS_BASE" ;;
    naive_k2) echo "2 16 2048 naive_parallel $PHASE2_BASE" ;;
    div_k2)   echo "2 16 2048 diversity_parallel $PHASE2_BASE" ;;
    naive_k4) echo "4 8  2048 naive_parallel $RESULTS_BASE" ;;
    div_k4)   echo "4 8  2048 diversity_parallel $RESULTS_BASE" ;;
    naive_k8) echo "8 8  1024 naive_parallel $RESULTS_BASE" ;;
    div_k8)   echo "8 8  1024 diversity_parallel $RESULTS_BASE" ;;
    *) echo "ERROR: unknown condition '$COND'" >&2; return 1 ;;
  esac
}

# ───────────────────────────── args ─────────────────────────────
MODEL_SHORT="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

[[ -z "$MODEL_SHORT" ]] && { echo "Usage: $0 <model> <dataset|all|finish> [cond|all]"; exit 1; }
[[ -z "${HF_MAP[$MODEL_SHORT]:-}" ]] && { echo "ERROR: unknown model '$MODEL_SHORT'"; exit 1; }
HF_MODEL="${HF_MAP[$MODEL_SHORT]}"
LITELLM_MODEL="openai/${HF_MODEL}"

# Expand "finish" shortcut for qwen3-8b (the 7 known-broken jobs)
if [[ "$ARG2" == "finish" ]]; then
  if [[ "$MODEL_SHORT" != "qwen3-8b" ]]; then
    echo "ERROR: 'finish' shortcut only defined for qwen3-8b"; exit 1
  fi
  JOB_LIST=(
    "musique:seq" "frames:seq" "GAIA:seq" "hle:seq" "webwalker:seq"
    "webwalker:naive_k4" "webwalker:div_k4"
  )
else
  # Build (dataset, cond) list from args
  DS_LIST=()
  if [[ "$ARG2" == "all" ]]; then DS_LIST=("${ALL_DATASETS[@]}"); else DS_LIST=("$ARG2"); fi
  CD_LIST=()
  if [[ -z "$ARG3" || "$ARG3" == "all" ]]; then
    CD_LIST=("${PHASE1_CONDS[@]}")
  else
    CD_LIST=("$ARG3")
  fi
  JOB_LIST=()
  for DS in "${DS_LIST[@]}"; do
    for CD in "${CD_LIST[@]}"; do
      JOB_LIST+=("${DS}:${CD}")
    done
  done
fi

# ───────────────────────────── env ─────────────────────────────
# HF cache on /data so /home doesn't fill up
CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
JUDGE_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

# ───────────────────────────── submit loop ─────────────────────────────
JOB_COUNT=0

for ENTRY in "${JOB_LIST[@]}"; do
  DATASET="${ENTRY%%:*}"
  COND="${ENTRY##*:}"
  read -r K T MTOK METHOD RBASE < <(cond_params "$COND")

  DATASET_PATH="${DATA_DIR}/${DATASET}.json"
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "SKIP: ${DATASET_PATH} not found"; continue
  fi

  OUTDIR_BASE="${RBASE}/${MODEL_SHORT}/${DATASET}/${COND}"

  # Inline script: launch vLLM on a unique port (derived from SLURM_JOB_ID to
  # avoid collisions when multiple jobs land on the same node), poll for ready,
  # run all seeds, kill vLLM.
  INNER="VLLM_PORT=\$((8000 + (\$SLURM_JOB_ID % 1000)))"
  INNER+="; VLLM_LOG=/tmp/vllm_\${SLURM_JOB_ID}.log"
  INNER+="; vllm serve ${HF_MODEL} --port \$VLLM_PORT --enable-prefix-caching --dtype auto --max-model-len 32768 --max-num-seqs 128 --enforce-eager --disable-log-stats > \$VLLM_LOG 2>&1 &"
  INNER+=" VLLM_PID=\$!"
  INNER+="; echo '[vllm] started pid='\$VLLM_PID' port='\$VLLM_PORT', waiting for ready...'"
  INNER+="; READY=0"
  INNER+="; for i in \$(seq 1 240); do if curl -sf http://localhost:\$VLLM_PORT/v1/models > /dev/null 2>&1; then echo \"[vllm] ready after \${i}0s\"; READY=1; break; fi; sleep 10; done"
  INNER+="; if [[ \$READY -eq 0 ]]; then echo '[vllm] FAILED to come up in 2400s. Log tail:'; tail -50 \$VLLM_LOG; kill \$VLLM_PID 2>/dev/null || true; exit 1; fi"
  INNER+="; trap 'kill \$VLLM_PID 2>/dev/null || true' EXIT"
  for SEED in "${SEEDS[@]}"; do
    OUTDIR="${OUTDIR_BASE}/run_${SEED}"
    INNER+="; echo '--- ${MODEL_SHORT} | ${DATASET} | ${COND} | seed=${SEED} ---'"
    INNER+="; python -m webwalkerqa.run.run_main_table --model ${LITELLM_MODEL} --dataset ${DATASET_PATH} --condition ${METHOD} --seed ${SEED} --max-turns-par ${T} --pool-size 16 --max-concurrent 100 --api-base http://localhost:\$VLLM_PORT/v1 --output-dir ${OUTDIR} --k ${K} --max-tokens ${MTOK} || echo '[warn] seed ${SEED} exited nonzero, continuing'"
  done
  INNER+="; echo 'Done seeds: ${MODEL_SHORT}/${DATASET}/${COND}'"
  # Inline judge — runs right after seeds, still inside the same SLURM job.
  # Uses the same GPU node but judge is API-bound so GPU idles for 1-5 min. Worth it
  # to save a SLURM slot vs separate judge job + dependency.
  INNER+="; echo '--- judging ${MODEL_SHORT}/${DATASET}/${COND} ---'"
  INNER+="; python -m webwalkerqa.judge.eval_llm --results-dir ${RBASE} --filter-model ${MODEL_SHORT} --filter-dataset ${DATASET} --filter-condition ${COND} --model openai/gpt-4o-mini --max-concurrent 200 --force || echo '[warn] judge failed; rerun manually later'"
  INNER+="; kill \$VLLM_PID 2>/dev/null || true"
  INNER+="; wait \$VLLM_PID 2>/dev/null || true"

  JNAME="t3-${MODEL_SHORT:0:6}-${DATASET:0:6}-${COND}"
  EXP_JOB_ID=$(sbatch \
    --job-name="${JNAME}" \
    --partition=general \
    --gres=gpu:1 \
    --time=24:00:00 \
    --mem=32G \
    --cpus-per-task=8 \
    --export=ALL \
    --output="logs/main_table/t3_${MODEL_SHORT}_${DATASET}_${COND}_%j.out" \
    --error="logs/main_table/t3_${MODEL_SHORT}_${DATASET}_${COND}_%j.err" \
    --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" \
    --parsable)
  echo "Submitted exp+judge: ${JNAME}  job=${EXP_JOB_ID}"
  JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "Aggregate: run manually after all jobs complete:"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${PHASE2_BASE}"

echo ""
echo "Total exp jobs submitted: ${JOB_COUNT}"
echo "Monitor: squeue -u \$USER"
