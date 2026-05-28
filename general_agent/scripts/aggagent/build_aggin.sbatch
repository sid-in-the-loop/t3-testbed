#!/bin/bash
#SBATCH --job-name=build-aggin
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/build_aggin_%j.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/build_aggin_%j.err

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs
mkdir -p /data/user_data/ssmurali/aggin

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate t3

cd /home/ssmurali/t3-testbed

python3 general_agent/scripts/aggagent/build_aggin.py \
  --aggin-root /data/user_data/ssmurali/aggin \
  "$@"

echo "[done] aggin written to /data/user_data/ssmurali/aggin"
du -sh /data/user_data/ssmurali/aggin
find /data/user_data/ssmurali/aggin -type f | wc -l
