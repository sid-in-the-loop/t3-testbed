#!/bin/bash
#SBATCH --job-name=agg-chunk
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/agg/agg_chunk_%j.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/agg/agg_chunk_%j.err
#
# Run a CONTIGUOUS RANGE of slice indices serially. The `general` partition
# requires a GPU, so we ask for one even though we don't use it (LLM calls go
# to OpenAI's API, vLLM is no longer needed).
#
# Required env vars (set via --export):
#   CHUNK_START, CHUNK_END   inclusive slice index range
#   OPENAI_API_KEY
#   MANIFEST                 path to aggregation_manifest.tsv

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs/agg

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate aggagent

: "${CHUNK_START:?expected CHUNK_START}"
: "${CHUNK_END:?expected CHUNK_END}"
: "${OPENAI_API_KEY:?expected OPENAI_API_KEY}"
: "${MANIFEST:?expected MANIFEST}"

echo "[chunk $CHUNK_START-$CHUNK_END] on $(hostname), $(date)"

for IDX in $(seq $CHUNK_START $CHUNK_END); do
  echo
  echo "=============== slice $IDX ==============="
  bash /home/ssmurali/t3-testbed/general_agent/scripts/aggagent/run_one_slice.sh \
    "$MANIFEST" "$IDX" || echo "[chunk] slice $IDX FAILED, continuing"
done

echo "[chunk $CHUNK_START-$CHUNK_END] DONE at $(date)"
