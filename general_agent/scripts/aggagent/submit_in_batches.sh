#!/bin/bash
# Submit the aggregation array in batches to stay under QOSMaxSubmitJobPerUser.
# Waits for each batch to drain before submitting the next.
#
# Usage:
#   bash submit_in_batches.sh [BATCH_SIZE] [TOTAL] [MANIFEST]
# Defaults:
#   BATCH_SIZE=50, TOTAL=248, MANIFEST=/data/user_data/ssmurali/aggregation_manifest.tsv

set -euo pipefail
BATCH=${1:-50}
TOTAL=${2:-248}
MANIFEST=${3:-/data/user_data/ssmurali/aggregation_manifest.tsv}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[fatal] OPENAI_API_KEY not set" >&2
  exit 2
fi

cd /home/ssmurali/t3-testbed
START=0
while [[ $START -lt $TOTAL ]]; do
  END=$((START + BATCH - 1))
  [[ $END -ge $TOTAL ]] && END=$((TOTAL - 1))
  echo "[batch] submitting $START-$END"

  JOBID=$(sbatch --parsable --array=${START}-${END} \
    --export=ALL,OPENAI_API_KEY,MANIFEST="$MANIFEST" \
    general_agent/scripts/aggagent/aggregate_array.sbatch)
  echo "[batch] submitted JobArray=$JOBID"

  echo "[batch] waiting for $JOBID to drain..."
  while squeue -h -j "$JOBID" 2>/dev/null | grep -q .; do
    sleep 30
  done
  echo "[batch] $JOBID done"

  START=$((END + 1))
done

echo "[batch] all $TOTAL slices submitted and completed"
