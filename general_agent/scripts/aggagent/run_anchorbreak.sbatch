#!/bin/bash
#SBATCH --job-name=ab-llm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/agg/ab_llm_%j.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/agg/ab_llm_%j.err

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs/agg

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate aggagent

: "${OPENAI_API_KEY:?OPENAI_API_KEY required (use --export=ALL,OPENAI_API_KEY)}"

cd /home/ssmurali/t3-testbed
python3 general_agent/scripts/aggagent/run_anchorbreak.py \
  --aggin-root /data/user_data/ssmurali/aggin \
  --out-root   /data/user_data/ssmurali/anchorbreak \
  --max-workers 64
