#!/bin/bash
# Launch a vLLM server as a SLURM job and register its endpoint.
# The endpoint file is written from inside the job (hostname only known at runtime).
# vLLM continuous batching is enabled by default; --max-num-seqs controls batch depth.
#
# Usage:
#   cd general_agent
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-8B  8003
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-4B  8002
#   ./scripts/launch_vllm_server.sh Qwen/Qwen3-1.7B 8001
#   ./scripts/launch_vllm_server.sh openai/gpt-oss-20b 8004
#
# Args:
#   $1  HF model alias (e.g. Qwen/Qwen3-8B or openai/gpt-oss-20b)
#   $2  Port to serve on
#   $3  (optional) Max model length, default 8192
#   $4  (optional) GPU type constraint, default L40S

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/vllm results/vllm_servers

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf-model-alias> <port> [max-model-len] [gpu-type]"
  echo "  e.g. $0 Qwen/Qwen3-8B 8003"
  echo "  e.g. $0 openai/gpt-oss-20b 8004 16384 L40S"
  exit 1
fi

MODEL_NAME="$1"
PORT="$2"
MAX_MODEL_LEN="${3:-8192}"
GPU_TYPE="${4:-L40S}"

# Derive short name: Qwen/Qwen3-8B → qwen3-8b, openai/gpt-oss-20b → gpt-oss-20b
MODEL_SHORT=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
JOB_NAME="vllm-${MODEL_SHORT}"
ENDPOINT_FILE="/home/ssmurali/t3-testbed/general_agent/results/vllm_servers/${MODEL_SHORT}.endpoint"

CONDA_INIT="source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh \
  && conda activate t3 \
  && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

WRAP="${CONDA_INIT} && \
  mkdir -p /home/ssmurali/t3-testbed/general_agent/results/vllm_servers && \
  echo \"http://\$(hostname):${PORT}/v1\" > ${ENDPOINT_FILE} && \
  echo \"[vLLM] Serving ${MODEL_NAME} on http://\$(hostname):${PORT}/v1\" && \
  vllm serve ${MODEL_NAME} \
    --port ${PORT} \
    --enable-prefix-caching \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs 512"

sbatch \
  --job-name="${JOB_NAME}" \
  --partition=general \
  --gres=gpu:${GPU_TYPE}:1 \
  --mem=32G \
  --cpus-per-task=8 \
  --time=24:00:00 \
  --output="logs/vllm/${JOB_NAME}_%j.out" \
  --error="logs/vllm/${JOB_NAME}_%j.err" \
  --wrap="${WRAP}"

echo "Submitted: ${JOB_NAME} (model=${MODEL_NAME}, port=${PORT})"
echo "Endpoint will be at: ${ENDPOINT_FILE}"
echo "Watch server start: tail -f logs/vllm/${JOB_NAME}_*.out"
echo "Health check: curl http://\$(cat ${ENDPOINT_FILE} | sed 's|/v1||')/v1/models"
