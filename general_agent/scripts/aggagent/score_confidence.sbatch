#!/bin/bash
#SBATCH --job-name=score-conf
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ssmurali/t3-testbed/general_agent/logs/score_conf_%j.out
#SBATCH --error=/home/ssmurali/t3-testbed/general_agent/logs/score_conf_%j.err

# Args (positional):
#   $1 = served-model-name for the local vLLM (e.g. Qwen/Qwen3-8B)
#   $2 = endpoint file under results/vllm_servers/<short>.endpoint
# Any further args are forwarded to score_confidence.py (e.g. --models qwen3-8b --conds naive_k4)

set -euo pipefail
mkdir -p /home/ssmurali/t3-testbed/general_agent/logs

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate t3
cd /home/ssmurali/t3-testbed

MODEL_NAME="${1:?served-model-name (e.g. Qwen/Qwen3-8B)}"
ENDPOINT_FILE="${2:?endpoint file}"
shift 2

# Wait for vLLM to be up
for i in $(seq 1 120); do
  if [[ -f "$ENDPOINT_FILE" ]] && curl -s "$(cat $ENDPOINT_FILE)/models" > /dev/null 2>&1; then
    echo "[ok] vLLM ready at $(cat $ENDPOINT_FILE)"
    break
  fi
  echo "[wait ${i}/120] vLLM not yet ready"
  sleep 15
done

ENDPOINT="$(cat $ENDPOINT_FILE)"

python3 general_agent/scripts/aggagent/score_confidence.py \
  --aggin-root /data/user_data/ssmurali/aggin \
  --endpoint "$ENDPOINT" \
  --model "$MODEL_NAME" \
  --max-workers 64 \
  "$@"
