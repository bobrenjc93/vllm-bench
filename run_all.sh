#!/bin/bash
# Run both benchmark configurations and report CSV paths.
# Rebuilds vLLM and PyTorch from source by default to ensure latest versions.
#
# Usage:
#   ./run_all.sh                                # rebuild + all default models
#   ./run_all.sh --skip-rebuild                 # reuse existing builds
#   ./run_all.sh --models meta-llama/Meta-Llama-3-70B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIP_DIR="${SCRIPT_DIR}/vllm-tip-pytorch-tip"
RELEASE_DIR="${SCRIPT_DIR}/vllm-tip-pytorch-release"
TIP_OUTPUT_DIR="/tmp/compile_bench_results/pytorch-tip"
RELEASE_OUTPUT_DIR="/tmp/compile_bench_results/pytorch-release"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

latest_csv() {
    ls -1t "$1"/compile_bench_*.csv 2>/dev/null | head -1
}

REBUILD=1
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-rebuild)
            REBUILD=0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

if [ "$REBUILD" -eq 1 ]; then
    log "Force-rebuilding PyTorch and vLLM from latest source (use --skip-rebuild to reuse existing builds)"
    PASSTHROUGH_ARGS=("--force-rebuild" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}")
else
    log "Reusing existing builds (--skip-rebuild)"
fi

log "=== Running both benchmarks ==="

BEFORE_TIP=$(latest_csv "$TIP_OUTPUT_DIR")
log "--- pytorch-tip ---"
"${TIP_DIR}/run_compile_bench.sh" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"
TIP_CSV=$(latest_csv "$TIP_OUTPUT_DIR")

BEFORE_RELEASE=$(latest_csv "$RELEASE_OUTPUT_DIR")
log "--- pytorch-release ---"
"${RELEASE_DIR}/run_compile_bench.sh" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"
RELEASE_CSV=$(latest_csv "$RELEASE_OUTPUT_DIR")

echo ""
echo "=============================="
echo "  Results"
echo "=============================="

if [ -n "$TIP_CSV" ] && [ "$TIP_CSV" != "$BEFORE_TIP" ]; then
    echo "  pytorch-tip:     $TIP_CSV"
else
    echo "  pytorch-tip:     FAILED (no new CSV generated)"
fi

if [ -n "$RELEASE_CSV" ] && [ "$RELEASE_CSV" != "$BEFORE_RELEASE" ]; then
    echo "  pytorch-release: $RELEASE_CSV"
else
    echo "  pytorch-release: FAILED (no new CSV generated)"
fi
