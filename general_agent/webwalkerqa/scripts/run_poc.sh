#!/bin/bash
# ============================================================================
# run_poc.sh — WebWalkerQA T³ Proof of Concept Runner
#
# Runs the s1 and T³ Fixed experiments on WebWalkerQA.
#
# Usage:
#   # Run a single config:
#   ./run_poc.sh A1 openai/gpt-4o-mini
#
#   # Run all s1 configs:
#   ./run_poc.sh s1 openai/gpt-4o-mini
#
#   # Run all T³ configs:
#   ./run_poc.sh t3 openai/gpt-4o-mini
#
#   # Run full experiment matrix:
#   ./run_poc.sh all openai/gpt-4o-mini
#
#   # Quick test (10 examples, verbose):
#   ./run_poc.sh A2 openai/gpt-4o-mini --max-examples 10 --verbose
#
# Environment:
#   SERPER_API_KEY  — required for web search
#   OPENAI_API_KEY  — required for OpenAI models
#   (see .env-example in general_agent/)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEBWALKERQA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="${1:-A1}"
MODEL="${2:-openai/gpt-4o-mini}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
DATASET_PATH="${DATASET:-}"
VERBOSE="${VERBOSE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$AGENT_DIR/results/webwalkerqa}"
ENV_FILE="${ENV_FILE:-$AGENT_DIR/.env}"

# Shift first two positional args
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

# Parse remaining flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift 1
            ;;
        *)
            shift
            ;;
    esac
done

# ── Load env ──────────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "[INFO] Loaded env from $ENV_FILE"
else
    echo "[WARN] No .env file found at $ENV_FILE; using existing environment variables."
fi

# ── Build command ─────────────────────────────────────────────────────────────
cd "$AGENT_DIR"

CMD=(python3 -m webwalkerqa.experiment)

case "${CONFIG,,}" in
    all)
        CMD+=(--all)
        ;;
    s1|s1-only)
        CMD+=(--s1-only)
        ;;
    t3|t3-only)
        CMD+=(--t3-only)
        ;;
    a|b|c)
        CMD+=(--group "${CONFIG^^}")
        ;;
    a1|a2|b1|b2|b3|c1|c2|c3|oracle)
        CMD+=(--config "${CONFIG^^}")
        ;;
    *)
        echo "[ERROR] Unknown config: $CONFIG"
        echo "  Valid: A1 A2 B1 B2 B3 C1 C2 C3 Oracle | all | s1 | t3 | A | B | C"
        exit 1
        ;;
esac

CMD+=(--model "$MODEL")
CMD+=(--max-concurrent "$MAX_CONCURRENT")
CMD+=(--num-samples "$NUM_SAMPLES")
CMD+=(--output-dir "$OUTPUT_DIR")

[[ -n "$DATASET_PATH" ]] && CMD+=(--dataset "$DATASET_PATH")
[[ -n "$MAX_EXAMPLES" ]] && CMD+=(--max-examples "$MAX_EXAMPLES")
[[ -n "$VERBOSE" ]] && CMD+=(--verbose)

# ── Run ───────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "[INFO] WebWalkerQA T³ PoC"
echo "[INFO] Config:    $CONFIG"
echo "[INFO] Model:     $MODEL"
echo "[INFO] Output:    $OUTPUT_DIR"
echo "[INFO] Command:   ${CMD[*]}"
echo "========================================================"
echo ""

"${CMD[@]}"

echo ""
echo "========================================================"
echo "[INFO] Done. Analyze results with:"
echo "  cd $AGENT_DIR"
echo "  python3 -m webwalkerqa.analyze --results-dir $OUTPUT_DIR"
echo "========================================================"
