#!/bin/bash
# Run all 8 aggregation methods + post-hoc judge for one (model, dataset, cond, seed) slice.
# Args:
#   $1 = manifest path
#   $2 = slice index (0-based)
set -euo pipefail

MANIFEST="${1:?manifest path required}"
IDX="${2:?slice index required}"

LINE=$(awk -v i="$IDX" 'NR==i+1' "$MANIFEST")
if [[ -z "$LINE" ]]; then
  echo "[err] no line for idx=$IDX in $MANIFEST" >&2
  exit 1
fi

IFS=$'\t' read -r I MODEL DATASET COND SEED AGGIN AGGOUT <<< "$LINE"
echo "[slice $I] $MODEL/$DATASET/$COND/$SEED"
echo "[slice $I] aggin=$AGGIN"
echo "[slice $I] aggout=$AGGOUT"

mkdir -p "$AGGOUT"
cd /home/ssmurali/t3-testbed/AggAgent
export PYTHONPATH=/home/ssmurali/t3-testbed/AggAgent

# Heuristics — fast, no API
for STRAT in pass mv wmv bon fewtool; do
  echo "[slice $I] === $STRAT ==="
  python3 aggregation/aggregate.py \
    --strategy $STRAT --task hle --skip_score --k 4 \
    -- "$AGGIN" \
    2>&1 | tee -a "$AGGOUT/heuristics.log" | grep -E "^\s+(Pass|MV|WMV|BON|FewTool)@" || true
done

# LLM methods — gpt-4o-mini
# Count actual questions in this slice (one .json per question per thread dir)
N_Q=$(ls "$AGGIN/thread_0"/*.json 2>/dev/null | wc -l)
echo "[slice $I] slice has $N_Q questions"
for STRAT in solagg summagg aggagent; do
  LOG="$AGGOUT/$STRAT/${STRAT}_logs_k4.jsonl"
  if [[ -f "$LOG" ]] && [[ $(wc -l < "$LOG") -ge $N_Q ]]; then
    echo "[slice $I] $STRAT already has $N_Q lines, skipping rerun"
    continue
  fi
  # If partial log exists, delete to force clean restart (aggregate.py won't
  # otherwise rerun questions already in the log).
  if [[ -f "$LOG" ]]; then
    echo "[slice $I] $STRAT partial log ($(wc -l < "$LOG")/$N_Q), deleting for clean restart"
    rm -f "$LOG"
  fi
  echo "[slice $I] === $STRAT ==="
  python3 aggregation/aggregate.py \
    --strategy $STRAT --task hle --skip_score \
    --model gpt-4o-mini \
    --output_dir "$AGGOUT/$STRAT" \
    --k 4 \
    -- "$AGGIN" \
    2>&1 | tail -10
done

# Post-hoc judge for LLM methods
echo "[slice $I] === post-hoc judge ==="
python3 /home/ssmurali/t3-testbed/general_agent/scripts/aggagent/judge_agg_outputs.py \
  --aggin-root "$AGGIN" \
  --aggout-root "$AGGOUT" \
  --strategies solagg summagg aggagent \
  --k 4 \
  --model gpt-4o-mini \
  --max-workers 16

echo "[slice $I] DONE"
