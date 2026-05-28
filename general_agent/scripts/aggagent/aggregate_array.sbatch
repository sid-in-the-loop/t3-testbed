#!/bin/bash
#SBATCH --job-name=agg-array
#SBATCH --partition=array
#SBATCH --qos=array_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/agg/agg_%A_%a.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/agg/agg_%A_%a.err
# Submit with:
#   sbatch --array=0-189%50 \
#     --export=ALL,OPENAI_API_KEY,MANIFEST=/data/user_data/ssmurali/aggregation_manifest.tsv \
#     general_agent/scripts/aggagent/aggregate_array.sbatch

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs/agg

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate aggagent

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[fatal] OPENAI_API_KEY not set" >&2
  exit 2
fi

MANIFEST="${MANIFEST:-/data/user_data/ssmurali/aggregation_manifest.tsv}"
IDX="${SLURM_ARRAY_TASK_ID:?expected SLURM_ARRAY_TASK_ID}"

bash /home/ssmurali/t3-testbed/general_agent/scripts/aggagent/run_one_slice.sh "$MANIFEST" "$IDX"
