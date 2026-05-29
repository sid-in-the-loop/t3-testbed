#!/bin/bash
# Oversample-until-turn-N ablation: vary --oversample-until-turn N for div_k4.
# N ∈ {1..8}, k=4, pool=16.
#
# Usage:
#   ./scripts/submit_oversample_ablation.sh                        # all models × all datasets
#   ./scripts/submit_oversample_ablation.sh qwen3-8b               # one model, all datasets
#   ./scripts/submit_oversample_ablation.sh qwen3-8b hotpotqa      # one model × one dataset
#
# Model → dataset config:
#   qwen3-1.7b, qwen3-8b : hotpotqa (clueweb), gaia (clueweb)
#   gemma3-12b            : hotpotqa (clueweb), gaia/hle/webwalker (serper)
#   (serper web benchmarks for qwen3-8b also included)
#
# Results: results/ablations/oversample/<model>/<dataset>/os_<N>/run_<seed>/

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/oversample logs/judge

DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/ablations/oversample"
N_VALUES=(1 2 3 4 5 6 7 8)
SEEDS=(1 2 3)

CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

FILTER_MODEL="${1:-}"
FILTER_DATASET="${2:-}"

# MODEL_CONFIGS: "hf_model|model_short|ds1:file:backend:prompt:maxtok:judge ..."
MODEL_CONFIGS=(
  "Qwen/Qwen3-1.7B|qwen3-1.7b|hotpotqa:hotpotqa.json:clueweb:react_simple:2048:mhqa gaia:gaia.json:clueweb:react_simple:2048:mhqa"
  "Qwen/Qwen3-8B|qwen3-8b|hotpotqa:hotpotqa.json:clueweb:react_simple:2048:mhqa gaia:gaia.json:clueweb:react_simple:2048:mhqa bamboogle:bamboogle.json:serper:web_reasoning:8192:web hle:hle_sub.json:serper:web_reasoning:8192:web webwalker:webwalker_sub.json:serper:web_reasoning:8192:web"
  "google/gemma-3-12b-it|gemma3-12b|hotpotqa:hotpotqa.json:clueweb:react_simple:2048:mhqa gaia:gaia.json:serper:web_reasoning:8192:web hle:hle_sub.json:serper:web_reasoning:8192:web webwalker:webwalker_sub.json:serper:web_reasoning:8192:web"
)

JOB_COUNT=0

for MODEL_CFG in "${MODEL_CONFIGS[@]}"; do
  HF_MODEL="${MODEL_CFG%%|*}"
  REST="${MODEL_CFG#*|}"
  MODEL_SHORT="${REST%%|*}"
  DS_LIST="${REST#*|}"

  [[ -n "$FILTER_MODEL" && "$MODEL_SHORT" != "$FILTER_MODEL" ]] && continue

  LITELLM_MODEL="openai/${HF_MODEL}"

  for DS_CFG in $DS_LIST; do
    IFS=: read -r DS DSFILE BACKEND PROMPT MAXTOK JUDGE <<< "$DS_CFG"

    [[ -n "$FILTER_DATASET" && "$DS" != "$FILTER_DATASET" ]] && continue

    DATASET_PATH="${DATA_DIR}/${DSFILE}"
    [[ ! -f "$DATASET_PATH" ]] && { echo "SKIP missing $DATASET_PATH"; continue; }

    for N in "${N_VALUES[@]}"; do
      OUTDIR_BASE="${RESULTS_BASE}/${MODEL_SHORT}/${DS}/os_${N}"

      INNER="export SEARCH_BACKEND=${BACKEND}"
      INNER+="; VLLM_PORT=\$((8000 + (\$SLURM_JOB_ID % 1000)))"
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
        INNER+="; echo '--- ${MODEL_SHORT} | ${DS} | os=${N} | seed=${SEED} ---'"
        INNER+="; python -m webwalkerqa.run.run_main_table --model ${LITELLM_MODEL} --dataset ${DATASET_PATH} --condition diversity_parallel --seed ${SEED} --max-turns-par 8 --k 4 --max-tokens ${MAXTOK} --pool-size 16 --oversample-until-turn ${N} --max-concurrent 100 --api-base http://localhost:\$VLLM_PORT/v1 --output-dir ${OUTDIR} --prompt-style ${PROMPT} || echo '[warn] seed ${SEED} exited nonzero'"
      done
      INNER+="; echo 'Done seeds: ${MODEL_SHORT}/${DS}/os_${N}'"
      INNER+="; python -m webwalkerqa.judge.eval_llm --results-dir ${RESULTS_BASE} --filter-model ${MODEL_SHORT} --filter-dataset ${DS} --filter-condition os_${N} --model openai/gpt-4o-mini --max-concurrent 200 --force --judge-prompt-style ${JUDGE} || echo '[warn] judge failed'"
      INNER+="; kill \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true"

      JNAME="os-${MODEL_SHORT:0:5}-${DS:0:5}-n${N}"
      EXP_JOB_ID=$(sbatch \
        --job-name="${JNAME}" --partition=general --gres=gpu:1 \
        --time=24:00:00 --mem=32G --cpus-per-task=8 --export=ALL \
        --output="logs/oversample/${MODEL_SHORT}_${DS}_os${N}_%j.out" \
        --error="logs/oversample/${MODEL_SHORT}_${DS}_os${N}_%j.err" \
        --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" --parsable)
      echo "Submitted: ${JNAME} job=${EXP_JOB_ID}"
      JOB_COUNT=$((JOB_COUNT + 1))
    done
  done
done

echo ""
echo "Total jobs: ${JOB_COUNT}"
echo "Aggregate (run manually at end):"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
