#!/bin/bash
# ============================================================================
# run_full_matrix.sh — Run the complete WebWalkerQA experiment matrix
#
# Runs configs A1, A2, B1, B2, B3, C1, C2, C3, Oracle in sequence.
# Generates Figure 1 after all experiments complete.
#
# Usage:
#   MODEL=openai/gpt-4o-mini ./run_full_matrix.sh
#   MODEL=gemini/gemini-2.5-flash MAX_CONCURRENT=8 ./run_full_matrix.sh
#
# Set MAX_EXAMPLES=20 for a quick sanity check run.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-openai/gpt-4o-mini}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$AGENT_DIR/results/webwalkerqa}"
ENV_FILE="${ENV_FILE:-$AGENT_DIR/.env}"

# Experiment order (cheaper/faster first)
CONFIGS=(A1 A2 B1 B2 B3 C1 C2 C3 Oracle)

echo "========================================================"
echo "[INFO] WebWalkerQA Full Experiment Matrix"
echo "[INFO] Model:      $MODEL"
echo "[INFO] Configs:    ${CONFIGS[*]}"
echo "[INFO] Output:     $OUTPUT_DIR"
if [[ -n "$MAX_EXAMPLES" ]]; then
    echo "[INFO] Max examples: $MAX_EXAMPLES (debug mode)"
fi
echo "========================================================"
echo ""

RESULTS=()
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "Running config: $cfg"
    echo "----------------------------------------------------"

    export MODEL MAX_CONCURRENT OUTPUT_DIR ENV_FILE
    [[ -n "$MAX_EXAMPLES" ]] && export MAX_EXAMPLES

    if MAX_EXAMPLES="$MAX_EXAMPLES" "$SCRIPT_DIR/run_poc.sh" "$cfg" "$MODEL"; then
        RESULTS+=("$cfg: SUCCESS")
    else
        RESULTS+=("$cfg: FAILED")
        echo "[WARN] Config $cfg failed. Continuing with next..."
    fi
done

# ── Generate Figure 1 ─────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "[INFO] Generating Figure 1..."
echo "========================================================"
cd "$AGENT_DIR"
python3 -m webwalkerqa.analyze \
    --results-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/figure1.png" \
    || echo "[WARN] Analysis failed (matplotlib may not be installed)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "[INFO] Full Matrix Complete"
echo "========================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Results: $OUTPUT_DIR"
echo "Figure:  $OUTPUT_DIR/figure1.png"
