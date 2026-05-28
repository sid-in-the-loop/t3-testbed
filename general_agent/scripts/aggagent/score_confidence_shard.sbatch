#!/bin/bash
#SBATCH --job-name=conf-shard
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/conf/conf_%j.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/conf/conf_%j.err
#
# Self-contained sharded confidence-scoring job. Launches its OWN vLLM on its
# allocated GPU, waits for it to be up, scores its shard, then shuts vLLM down.
#
# Submit one job per shard, e.g.:
#   for i in 0 1 2 3 4 5 6 7; do
#     sbatch --export=ALL,SHARD_I=$i,SHARD_N=8 \
#       general_agent/scripts/aggagent/score_confidence_shard.sbatch
#   done

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs/conf

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate t3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/data/user_data/ssmurali/hf_cache
export HF_HUB_CACHE=/data/user_data/ssmurali/hf_cache/hub

I="${SHARD_I:?SHARD_I env var required (0..N-1)}"
N="${SHARD_N:?SHARD_N env var required (total shards)}"
PORT=$((7000 + I))
MODEL=Qwen/Qwen3-8B

echo "[shard $I/$N] starting vLLM on $(hostname):$PORT, GPU = $CUDA_VISIBLE_DEVICES"

VLLM_LOG=/home/ssmurali/t3-testbed/general_agent/logs/conf/vllm_${SLURM_JOB_ID}.log
vllm serve $MODEL \
  --port $PORT \
  --enable-prefix-caching \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap "kill -TERM $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true" EXIT

ENDPOINT="http://$(hostname):$PORT/v1"
echo "[shard $I/$N] endpoint will be $ENDPOINT"

for k in $(seq 1 120); do
  if curl -s "$ENDPOINT/models" > /dev/null 2>&1; then
    echo "[shard $I/$N] vLLM ready after ${k} checks"
    break
  fi
  echo "[shard $I/$N wait ${k}/120]"
  sleep 15
done

cd /home/ssmurali/t3-testbed
python3 general_agent/scripts/aggagent/score_confidence.py \
  --aggin-root /data/user_data/ssmurali/aggin \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --max-workers 256 \
  --shard ${I}/${N}

echo "[shard $I/$N] DONE"
