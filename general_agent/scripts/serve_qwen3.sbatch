#!/bin/bash
#SBATCH --job-name=vllm-qwen3
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/vllm-%x_%j.out
#SBATCH --error=logs/vllm-%x_%j.err

# Usage:
#   sbatch --job-name=vllm-qwen3-8b   serve_qwen3.sbatch Qwen/Qwen3-8B  8003
#   sbatch --job-name=vllm-qwen3-4b   serve_qwen3.sbatch Qwen/Qwen3-4B  8002
#   sbatch --job-name=vllm-qwen3-1.7b serve_qwen3.sbatch Qwen/Qwen3-1.7B 8001

MODEL_REPO=${1:-Qwen/Qwen3-8B}
PORT=${2:-8000}

cd "$(dirname "$0")/.."
mkdir -p logs

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate t3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Serving ${MODEL_REPO} on port ${PORT}"
echo "Node: $(hostname)"

vllm serve "${MODEL_REPO}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 32768
