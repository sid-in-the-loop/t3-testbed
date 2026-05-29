#!/bin/bash
# Oversample-until-turn-N ablation for Gemma3-12B on HotpotQA (ClueWeb) + GAIA (Serper).
# N ∈ {1..8}, 3 seeds, k=4, pool=16.
#
# HotpotQA: ClueWeb backend, react_simple prompt, max_tok=2048
# GAIA:     Serper backend,  web_reasoning prompt, max_tok=8192
#
# Results: results/oversample_ablation_gemma12b/<dataset>/os_<N>/run_<seed>/

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/oversample_gemma12b

HF_MODEL="google/gemma-3-12b-it"
MODEL_SHORT="gemma3-12b"
LITELLM_MODEL="openai/${HF_MODEL}"

N_VALUES=(1 2 3 4 5 6 7 8)
SEEDS=(1 2 3)
DATA_DIR="/home/ssmurali/t3-testbed/general_agent/data/main_table"
RESULTS_BASE="/home/ssmurali/t3-testbed/results/oversample_ablation_gemma12b"

# dataset → (file, search_backend, prompt_style, max_tokens, judge_style)
declare -A DS_FILE DS_BACKEND DS_PROMPT DS_MAXTOK DS_JUDGE
DS_FILE["hotpotqa"]="hotpotqa.json"
DS_BACKEND["hotpotqa"]="clueweb"
DS_PROMPT["hotpotqa"]="react_simple"
DS_MAXTOK["hotpotqa"]=2048
DS_JUDGE["hotpotqa"]="mhqa"

DS_FILE["gaia"]="gaia.json"
DS_BACKEND["gaia"]="serper"
DS_PROMPT["gaia"]="web_reasoning"
DS_MAXTOK["gaia"]=8192
DS_JUDGE["gaia"]="web"

DS_FILE["hle"]="hle_sub.json"
DS_BACKEND["hle"]="serper"
DS_PROMPT["hle"]="web_reasoning"
DS_MAXTOK["hle"]=8192
DS_JUDGE["hle"]="web"

DS_FILE["webwalker"]="webwalker_sub.json"
DS_BACKEND["webwalker"]="serper"
DS_PROMPT["webwalker"]="web_reasoning"
DS_MAXTOK["webwalker"]=8192
DS_JUDGE["webwalker"]="web"

ALL_DATASETS=("hotpotqa" "gaia" "hle" "webwalker")
FILTER="${1:-}"
if [[ -n "$FILTER" ]]; then
  DATASETS=($FILTER)
else
  DATASETS=("${ALL_DATASETS[@]}")
fi

CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh && conda activate t3 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && export HF_HOME=/data/user_data/ssmurali/hf_cache && export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub && export VLLM_CACHE_ROOT=/data/user_data/ssmurali/vllm_cache && mkdir -p \$VLLM_CACHE_ROOT && export OPENAI_API_KEY=dummy && set -a && source /home/ssmurali/t3-testbed/general_agent/.env && set +a"
CD_CMD="cd /home/ssmurali/t3-testbed/general_agent"

JOB_COUNT=0

for DS in "${DATASETS[@]}"; do
  DSFILE="${DS_FILE[$DS]}"
  DATASET_PATH="${DATA_DIR}/${DSFILE}"
  BACKEND="${DS_BACKEND[$DS]}"
  PROMPT="${DS_PROMPT[$DS]}"
  MAXTOK="${DS_MAXTOK[$DS]}"
  JUDGE="${DS_JUDGE[$DS]}"

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
    INNER+="; echo 'Done seeds: ${DS}/os_${N}'"
    INNER+="; echo '--- judging ${DS}/os_${N} ---'"
    INNER+="; python -m webwalkerqa.judge.eval_llm --results-dir ${RESULTS_BASE} --filter-model ${MODEL_SHORT} --filter-dataset ${DS} --filter-condition os_${N} --model openai/gpt-4o-mini --max-concurrent 200 --force --judge-prompt-style ${JUDGE} || echo '[warn] judge failed'"
    INNER+="; kill \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true"

    JNAME="os-g12-${DS:0:5}-n${N}"
    EXP_JOB_ID=$(sbatch \
      --job-name="${JNAME}" --partition=general --gres=gpu:1 \
      --time=24:00:00 --mem=32G --cpus-per-task=8 --export=ALL \
      --output="logs/oversample_gemma12b/${DS}_os${N}_%j.out" \
      --error="logs/oversample_gemma12b/${DS}_os${N}_%j.err" \
      --wrap="${CONDA_INIT} && ${CD_CMD} && ${INNER}" --parsable)
    echo "Submitted: ${JNAME} job=${EXP_JOB_ID}"
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

echo ""
echo "Total: ${JOB_COUNT} jobs (8 N-values × 2 datasets)"
echo "Results → ${RESULTS_BASE}"
echo "Aggregate (run manually at end):"
echo "  python -m webwalkerqa.scripts.aggregate_results --results-dir ${RESULTS_BASE}"
