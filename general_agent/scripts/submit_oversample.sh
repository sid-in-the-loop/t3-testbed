#!/bin/bash
# Oversample-until-turn-N ablation: vary --oversample-until-turn N for div_k4.
# Fixed: k=4, T=8, max_tok=2048, pool=16, datasets = {hotpotqa, GAIA}, models = {1.7b, 8b}.
# N ∈ {1..8}.
#
# Writes to: /home/ssmurali/t3-testbed/results/oversample_ablation/<model>/<dataset>/os_<N>/run_<seed>/

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/oversample logs/judge

declare -A HF_MAP
HF_MAP["qwen3-1.7b"]="Qwen/Qwen3-1.7B"
HF_MAP["qwen3-8b"]="Qwen/Qwen3-8B"

MODELS=("qwen3-1.7b" "qwen3-8b")
DATASETS=("hotpotqa" "GAIA")
N_VALUES=(1 2 3 4 5 6 7 8)
SEEDS=(1 2 3)
DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/oversample_ablation"

CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
JUDGE_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

JOB_COUNT=0

for MODEL_SHORT in "${MODELS[@]}"; do
  HF_MODEL="${HF_MAP[$MODEL_SHORT]}"
  LITELLM_MODEL="openai/${HF_MODEL}"
  for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH="${DATA_DIR}/${DATASET}.json"
    [[ ! -f "$DATASET_PATH" ]] && { echo "SKIP missing $DATASET_PATH"; continue; }
    for N in "${N_VALUES[@]}"; do
      OUTDIR_BASE="${RESULTS_BASE}/${MODEL_SHORT}/${DATASET}/os_${N}"

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
        INNER+="; echo '--- ${MODEL_SHORT} | ${DATASET} | os=${N} | seed=${SEED} ---'"
        INNER+="; python -m webwalkerqa.run.run_main_table --model ${LITELLM_MODEL} --dataset ${DATASET_PATH} --condition diversity_parallel --seed ${SEED} --max-turns-par 8 --k 4 --max-tokens 2048 --pool-size 16 --oversample-until-turn ${N} --max-concurrent 100 --api-base http://localhost:\$VLLM_PORT/v1 --output-dir ${OUTDIR} || echo '[warn] seed ${SEED} exited nonzero'"
      done
      INNER+="; echo 'Done seeds: ${MODEL_SHORT}/${DATASET}/os_${N}'"
      INNER+="; echo '--- judging ${MODEL_SHORT}/${DATASET}/os_${N} ---'"
      INNER+="; python -m webwalkerqa.judge.eval_llm --results-dir ${RESULTS_BASE} --filter-model ${MODEL_SHORT} --filter-dataset ${DATASET} --filter-condition os_${N} --model openai/gpt-4o-mini --max-concurrent 200 --force || echo '[warn] judge failed'"
      INNER+="; kill \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true"

      JNAME="os-${MODEL_SHORT:0:6}-${DATASET:0:6}-n${N}"
      EXP_JOB_ID=$(sbatch \
        --job-name="${JNAME}" --partition=general --gres=gpu:1 \
        --time=24:00:00 --mem=32G --cpus-per-task=8 --export=ALL \
        --output="logs/oversample/${MODEL_SHORT}_${DATASET}_os${N}_%j.out" \
        --error="logs/oversample/${MODEL_SHORT}_${DATASET}_os${N}_%j.err" \
        --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" --parsable)
      echo "Submitted exp+judge: ${JNAME} job=${EXP_JOB_ID}"
      JOB_COUNT=$((JOB_COUNT + 1))
    done
  done
done

echo ""
echo "Total exp jobs: ${JOB_COUNT}"
echo "Aggregate (run manually at end):"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
