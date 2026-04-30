#!/bin/bash
# End-to-end vLLM PT2 compile time benchmark
# Config: vLLM tip (source) + PyTorch stable release (pip wheel)
#
# All steps are idempotent — re-running skips work already done.
#
# Usage:
#   ./run_compile_bench.sh                                # full setup + all models
#   ./run_compile_bench.sh --force-rebuild                # clean rebuild from scratch
#   ./run_compile_bench.sh --models meta-llama/Llama-3.2-1B
#   ./run_compile_bench.sh --eager                        # include eager baseline

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var (required for gated models like Meta-Llama)}"
CONDA_ENV_NAME="vllm_bench_release"
PYTHON_VERSION="3.12"
CUDA_VERSION="13.0"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_DIR="/tmp/vllm-release"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/benchmark_compile_time.py"
OUTPUT_DIR="/tmp/compile_bench_results/pytorch-release"

export HF_TOKEN
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export DG_JIT_NVCC_COMPILER="/usr/local/cuda-12.9/bin/nvcc"

# ============================================================================
# Argument parsing — separate our flags from benchmark flags
# ============================================================================
FORCE_REBUILD=0
BENCH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --force-rebuild)
            FORCE_REBUILD=1
            ;;
        --skip-build)
            echo "NOTE: --skip-build is no longer needed. Builds are cached automatically."
            ;;
        --help|-h)
            echo "Usage: $0 [--force-rebuild] [benchmark args...]"
            echo ""
            echo "Config: vLLM tip (source) + PyTorch stable release (pip wheel)"
            echo ""
            echo "Script options:"
            echo "  --force-rebuild  Remove repo + conda env and rebuild from scratch"
            echo ""
            echo "Benchmark options (passed through to benchmark_compile_time.py):"
            echo "  --models M [M..] HuggingFace model names"
            echo "  --eager          Also benchmark eager mode (no torch.compile)"
            echo "  --output PATH    Output CSV path"
            echo "  --dtype TYPE     Model dtype (default: bfloat16)"
            echo "  --tp N           Tensor parallel size (default: 8)"
            exit 0
            ;;
        *)
            BENCH_ARGS+=("$arg")
            ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found. This script requires an NVIDIA GPU."
    fi
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}

# ============================================================================
# Step 0: Preflight checks
# ============================================================================
log "=== vLLM PT2 Compile Time Benchmark (vLLM tip + PyTorch release) ==="
check_gpu

if ! command -v conda &>/dev/null; then
    die "conda not found. Please install Miniconda/Anaconda first."
fi

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    die "CUDA ${CUDA_VERSION} not found at ${CUDA_HOME}. Install with: sudo dnf install cuda-toolkit-${CUDA_VERSION//./-}"
fi
log "Using CUDA: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"

eval "$(conda shell.bash hook)"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Force rebuild: nuke everything and start fresh
# ============================================================================
if [ "$FORCE_REBUILD" -eq 1 ]; then
    log "=== Force rebuild: cleaning repo and conda env ==="
    rm -rf "$VLLM_DIR"
    conda env remove -n "$CONDA_ENV_NAME" -y 2>/dev/null || true
fi

# ============================================================================
# Step 1: Clone vLLM (only if not already present)
# ============================================================================
log "=== Step 1: Repository ==="

if [ -d "$VLLM_DIR" ]; then
    log "vLLM already at $VLLM_DIR — reusing"
else
    log "Cloning vLLM..."
    git clone "$VLLM_REPO" "$VLLM_DIR"
fi

# Detect required torch version
TORCH_VERSION=$(grep -oP '"torch\s*==\s*\K[0-9.]+' "$VLLM_DIR/pyproject.toml" 2>/dev/null | head -1 || true)
if [ -z "$TORCH_VERSION" ]; then
    die "Could not detect torch version from vLLM's pyproject.toml"
fi
log "vLLM requires torch==${TORCH_VERSION}"

# ============================================================================
# Step 2: Conda environment (create only if needed)
# ============================================================================
log "=== Step 2: Conda environment ==="

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log "Conda env '$CONDA_ENV_NAME' exists — reusing"
else
    log "Creating conda env '$CONDA_ENV_NAME' (python ${PYTHON_VERSION})..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV_NAME"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

log "Python: $(python --version)"
pip install -q ninja cmake wheel setuptools setuptools_scm pyyaml typing-extensions 2>/dev/null

# ============================================================================
# Step 3: Install PyTorch release (skip if correct version already present)
# ============================================================================
log "=== Step 3: PyTorch ==="

CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$CURRENT_TORCH" = "$TORCH_VERSION" ]; then
    log "PyTorch ${TORCH_VERSION} already installed — skipping"
else
    log "Installing PyTorch ${TORCH_VERSION} from pip..."
    pip install "torch==${TORCH_VERSION}" 2>&1 | tail -5
fi

log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ============================================================================
# Step 4: Build vLLM from source (skip if already importable)
# ============================================================================
log "=== Step 4: vLLM ==="

NEED_VLLM_BUILD=1
if python -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    log "vLLM already installed: ${VLLM_VER} — skipping build"
    NEED_VLLM_BUILD=0
fi

if [ "$NEED_VLLM_BUILD" -eq 1 ]; then
    log "Building vLLM from source..."
    cd "$VLLM_DIR"
    rm -rf build 2>/dev/null || true
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    pip install -e . --no-build-isolation 2>&1 | tee /tmp/vllm_release_build.log | \
        grep -E "Building|Installing|Successfully|ERROR|FAILED|ninja:" || true
    if ! python -c "import vllm" 2>/dev/null; then
        log "vLLM build failed! Last 30 lines of build log:"
        tail -30 /tmp/vllm_release_build.log
        die "vLLM build failed"
    fi
    cd -
fi

python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || \
    log "vLLM installed (version unavailable)"

# ============================================================================
# Step 5: HuggingFace setup
# ============================================================================
log "=== Step 5: HuggingFace setup ==="
pip install -q huggingface_hub hf_xet tokenizers transformers 2>/dev/null || true

log "Logging into HuggingFace..."
python -c "
from huggingface_hub import login
login(token='${HF_TOKEN}', add_to_git_credential=False)
print('HuggingFace login successful')
"

# ============================================================================
# Step 6: Run benchmark
# ============================================================================
log "=== Step 6: Running benchmark ==="

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CSV_OUTPUT="$OUTPUT_DIR/compile_bench_${TIMESTAMP}.csv"

python "$BENCH_SCRIPT" \
    --output "$CSV_OUTPUT" \
    "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"

# ============================================================================
# Step 7: Report
# ============================================================================
log "=== Done ==="
log "CSV output: $CSV_OUTPUT"
if [ -f "${CSV_OUTPUT%.csv}.json" ]; then
    log "JSON output: ${CSV_OUTPUT%.csv}.json"
fi

echo ""
echo "=== CSV Contents ==="
if [ -f "$CSV_OUTPUT" ]; then
    column -t -s',' "$CSV_OUTPUT" 2>/dev/null || cat "$CSV_OUTPUT"
fi
