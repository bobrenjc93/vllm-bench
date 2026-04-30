#!/bin/bash
# End-to-end vLLM PT2 compile time benchmark
#
# This script:
#   1) Clones vllm and pytorch into /tmp (only if not already present)
#   2) Creates a conda env, builds both from source with CUDA 13.0 (skips if already done)
#   3) Pre-downloads model weights from HuggingFace
#   4) Runs the compile time benchmark
#   5) Outputs a CSV with date, model metrics (cold/warm start, PT2/non-PT2 time, etc.)
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
CONDA_ENV_NAME="vllm_bench"
PYTHON_VERSION="3.12"
CUDA_VERSION="13.0"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
VLLM_DIR="/tmp/vllm"
PYTORCH_DIR="/tmp/pytorch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/benchmark_compile_time.py"
OUTPUT_DIR="/tmp/compile_bench_results/pytorch-tip"

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
            echo "Script options:"
            echo "  --force-rebuild  Remove repos + conda env and rebuild from scratch"
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
log "=== vLLM PT2 Compile Time Benchmark ==="
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
    log "=== Force rebuild: cleaning repos and conda env ==="
    rm -rf "$VLLM_DIR" "$PYTORCH_DIR"
    conda env remove -n "$CONDA_ENV_NAME" -y 2>/dev/null || true
fi

# ============================================================================
# Step 1: Clone repositories (only if not already present)
# ============================================================================
log "=== Step 1: Repositories ==="

if [ -d "$VLLM_DIR" ]; then
    log "vLLM already at $VLLM_DIR — reusing"
else
    log "Cloning vLLM..."
    git clone "$VLLM_REPO" "$VLLM_DIR"
fi

# Detect required torch version from vLLM's pyproject.toml
TORCH_VERSION=$(grep -oP '"torch\s*==\s*\K[0-9.]+' "$VLLM_DIR/pyproject.toml" 2>/dev/null | head -1 || true)
if [ -z "$TORCH_VERSION" ]; then
    PYTORCH_TAG="main"
    log "WARNING: Could not detect torch version from vLLM. Using main."
else
    PYTORCH_TAG="v${TORCH_VERSION}"
    log "vLLM requires torch==${TORCH_VERSION} → PyTorch tag ${PYTORCH_TAG}"
fi

if [ -d "$PYTORCH_DIR" ]; then
    log "PyTorch already at $PYTORCH_DIR — reusing"
else
    log "Cloning PyTorch at tag ${PYTORCH_TAG} (with submodules)..."
    git clone --branch "$PYTORCH_TAG" --recursive "$PYTORCH_REPO" "$PYTORCH_DIR"
fi

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
# Step 3: Build PyTorch from source (skip if source build already present)
# ============================================================================
log "=== Step 3: PyTorch ==="

NEED_PYTORCH_BUILD=1
if python -c "import torch; v = torch.__version__; assert '+' in v or 'a0' in v" 2>/dev/null; then
    CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)")
    log "Source-built PyTorch already installed: ${CURRENT_TORCH} — skipping build"
    NEED_PYTORCH_BUILD=0
fi

if [ "$NEED_PYTORCH_BUILD" -eq 1 ]; then
    cd "$PYTORCH_DIR"

    # Ensure correct tag
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")
    if [ "$CURRENT_TAG" != "$PYTORCH_TAG" ]; then
        log "Checking out ${PYTORCH_TAG}..."
        git fetch origin --tags --force 2>/dev/null || git fetch origin
        git checkout -f "$PYTORCH_TAG"
        git submodule sync
        git submodule update --init --recursive
    fi

    # --- Patch pybind11 3.x typed-tuple ternary errors ---
    log "Applying pybind11 ternary type patches (idempotent)..."

    if grep -q "? py::make_tuple(py::handle(self), py::handle(index))" torch/csrc/utils/python_arg_parser.cpp; then
        sed -i 's/? py::make_tuple(py::handle(self), py::handle(index))/? py::object(py::make_tuple(py::handle(self), py::handle(index)))/' \
            torch/csrc/utils/python_arg_parser.cpp
        sed -i 's/: py::make_tuple(py::handle(self), py::handle(index), py::handle(val));/: py::object(py::make_tuple(py::handle(self), py::handle(index), py::handle(val)));/' \
            torch/csrc/utils/python_arg_parser.cpp
        log "  Patched python_arg_parser.cpp"
    fi

    if grep -q "return py::make_tuple(py::none(), py::none());" torch/csrc/jit/python/init.cpp; then
        sed -i 's/return py::make_tuple(py::none(), py::none());/return py::object(py::make_tuple(py::none(), py::none()));/' \
            torch/csrc/jit/python/init.cpp
        sed -i 's/return py::make_tuple(func, overload_names);/return py::object(py::make_tuple(func, overload_names));/' \
            torch/csrc/jit/python/init.cpp
        sed -i 's/return py::make_tuple(true, \*res);/return py::object(py::make_tuple(true, *res));/' \
            torch/csrc/jit/python/init.cpp
        sed -i 's/return py::make_tuple(false, py::none());/return py::object(py::make_tuple(false, py::none()));/' \
            torch/csrc/jit/python/init.cpp
        log "  Patched jit/python/init.cpp"
    fi

    if grep -q '\[](const ::c10d::ReduceOp& r) {' torch/csrc/distributed/c10d/init.cpp; then
        python3 -c "
import re
with open('torch/csrc/distributed/c10d/init.cpp') as f:
    src = f.read()
old = '''          [](const ::c10d::ReduceOp& r) {
            // __getstate__
            if (r.op_ != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              return py::make_tuple(r.op_, py::none());
            }
            TORCH_CHECK(r.supplement_.defined(), \"Invalid PREMUL_SUM ReduceOp\");
            const auto* preMulSupplement =
                reinterpret_cast<::c10d::NCCLPreMulSumSupplement*>(
                    r.supplement_.get());
            if (!preMulSupplement->tensor_factor.defined()) {
              return py::make_tuple(r.op_, preMulSupplement->double_factor);
            } else {
              return py::make_tuple(r.op_, preMulSupplement->tensor_factor);
            }
          },'''
new = '''          [](const ::c10d::ReduceOp& r) {
            // __getstate__
            py::tuple result(2);
            result[0] = py::cast(r.op_);
            if (r.op_ != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              result[1] = py::none();
            } else {
              TORCH_CHECK(r.supplement_.defined(), \"Invalid PREMUL_SUM ReduceOp\");
              const auto* preMulSupplement =
                  reinterpret_cast<::c10d::NCCLPreMulSumSupplement*>(
                      r.supplement_.get());
              if (!preMulSupplement->tensor_factor.defined()) {
                result[1] = py::cast(preMulSupplement->double_factor);
              } else {
                result[1] = py::cast(preMulSupplement->tensor_factor);
              }
            }
            return result;
          },'''
src = src.replace(old, new)
with open('torch/csrc/distributed/c10d/init.cpp', 'w') as f:
    f.write(src)
"
        log "  Patched distributed/c10d/init.cpp"
    fi

    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
    export USE_CUDA=1
    export CMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')"
    export MAX_JOBS=$(( $(nproc) / 4 ))
    export BUILD_TEST=0

    log "Building PyTorch with CUDA ${CUDA_VERSION} (MAX_JOBS=$MAX_JOBS)..."
    log "First build takes 20-40 minutes; incremental builds are much faster."
    pip install -r requirements.txt 2>/dev/null || true
    python setup.py develop 2>&1 | tee /tmp/pytorch_build.log | \
        grep -E "Building|Installing|Successfully|ERROR|FAILED|ninja:" || true
    if ! python -c "import torch" 2>/dev/null; then
        log "PyTorch build failed! Last 30 lines of build log:"
        tail -30 /tmp/pytorch_build.log
        die "PyTorch build failed"
    fi
    cd -
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
    pip install setuptools_scm 2>/dev/null || true
    rm -rf build 2>/dev/null || true
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    pip install -e . --no-build-isolation 2>&1 | tee /tmp/vllm_build.log | \
        grep -E "Building|Installing|Successfully|ERROR|FAILED|ninja:" || true
    if ! python -c "import vllm" 2>/dev/null; then
        log "vLLM build failed! Last 30 lines of build log:"
        tail -30 /tmp/vllm_build.log
        die "vLLM build failed"
    fi
    cd -

    # vLLM's pip install may replace source-built PyTorch with a pip wheel — fix it
    if ! python -c "import torch; v = torch.__version__; assert '+' in v or 'a0' in v" 2>/dev/null; then
        log "vLLM replaced source-built PyTorch with pip wheel. Reinstalling source build..."
        cd "$PYTORCH_DIR"
        python setup.py develop 2>&1 | tail -3
        cd -
    fi
fi

python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || \
    log "vLLM installed (version unavailable)"
log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

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
# Step 5.5: Kill any processes using GPU memory
# ============================================================================
log "=== Clearing GPU memory ==="
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | grep -v '^$' || true)
if [ -n "$GPU_PIDS" ]; then
    log "Found GPU processes: $GPU_PIDS"
    for pid in $GPU_PIDS; do
        OWNER=$(ps -o user= -p "$pid" 2>/dev/null || echo "unknown")
        CMD=$(ps -o comm= -p "$pid" 2>/dev/null || echo "unknown")
        log "  Killing PID $pid (user=$OWNER, cmd=$CMD)"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
    log "GPU memory after cleanup:"
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
else
    log "No GPU processes found"
fi

# ============================================================================
# Step 6: Run benchmark
# ============================================================================
log "=== Step 6: Running benchmark ==="

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CSV_OUTPUT="$OUTPUT_DIR/compile_bench_${TIMESTAMP}.csv"

cp "$BENCH_SCRIPT" "$OUTPUT_DIR/"

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
