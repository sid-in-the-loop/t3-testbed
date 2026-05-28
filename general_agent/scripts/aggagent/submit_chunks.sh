#!/bin/bash
# Submit 248 slices as 50 chunks of ~5 slices each, all in one go.
# Each chunk is a separate sbatch on partition=general (so we stay under
# QOSMaxSubmitJobPerUser=50 with exactly 50 jobs).
#
# Usage: bash submit_chunks.sh [CHUNK_SIZE] [TOTAL] [MANIFEST]
set -euo pipefail

CHUNK_SIZE=${1:-5}
TOTAL=${2:-248}
MANIFEST=${3:-/data/user_data/ssmurali/aggregation_manifest.tsv}

: "${OPENAI_API_KEY:?OPENAI_API_KEY required}"
cd /home/ssmurali/t3-testbed

N_CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Submitting $N_CHUNKS chunks of $CHUNK_SIZE slices each (total=$TOTAL)"

START=0
N=0
while [[ $START -lt $TOTAL ]]; do
  END=$((START + CHUNK_SIZE - 1))
  [[ $END -ge $TOTAL ]] && END=$((TOTAL - 1))
  JID=$(sbatch --parsable \
    --export=ALL,OPENAI_API_KEY,MANIFEST="$MANIFEST",CHUNK_START=$START,CHUNK_END=$END \
    general_agent/scripts/aggagent/aggregate_chunk.sbatch)
  echo "  chunk $N: slices $START-$END  jobid=$JID"
  START=$((END + 1))
  N=$((N + 1))
done
echo "Done. Submitted $N chunks."
