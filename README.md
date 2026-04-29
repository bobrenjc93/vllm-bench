# vllm-bench

Benchmarks for measuring **torch.compile (PT2) cold and warm start times** across models served by [vLLM](https://github.com/vllm-project/vllm).

## Structure

```
vllm-bench/
├── vllm-tip-pytorch-tip/       # vLLM source (main) + PyTorch source (matching tag)
│   ├── run_compile_bench.sh
│   └── benchmark_compile_time.py
└── vllm-tip-pytorch-release/   # vLLM source (main) + PyTorch stable wheel (pip)
    ├── run_compile_bench.sh
    └── benchmark_compile_time.py
```

### `vllm-tip-pytorch-tip`

Builds **both PyTorch and vLLM from source**. PyTorch is checked out at the tag matching vLLM's pinned version (e.g. `v2.11.0`), patched for pybind11 3.x compatibility, and compiled with CUDA 13.0. Use this to benchmark against the latest development state of both projects.

### `vllm-tip-pytorch-release`

Builds **vLLM from source** but installs **PyTorch from the official pip wheel** (the stable release matching vLLM's pin). Faster setup, no PyTorch compilation needed. Use this as the baseline or when you only care about vLLM changes.

## Quick Start

```bash
# Option A: Both from source (takes ~20-40 min first run)
cd vllm-tip-pytorch-tip
./run_compile_bench.sh

# Option B: vLLM from source, PyTorch from pip (takes ~5-10 min first run)
cd vllm-tip-pytorch-release
./run_compile_bench.sh
```

## Default Models

Both scripts benchmark the same set of models by default:

| Model | Notes |
|-------|-------|
| `meta-llama/Llama-3-70B` | Dense 70B baseline |
| `openai/gpt-oss-120b` | Large dense model |
| `deepseek-ai/DeepSeek-V3.2` | MoE |
| `moonshotai/Kimi-K2.6` | MoE architecture |
| `zai-org/GLM-4.7` | Large model |
| `MiniMaxAI/MiniMax-M2.7` | MoE — disabled by default, vLLM FusedMoE bug |
| `Qwen/Qwen3.6-35B-A3B` | MoE, 35B total / 3B active |

## Usage

### Full run (all default models)

```bash
./run_compile_bench.sh
```

### Specific models

```bash
./run_compile_bench.sh --models meta-llama/Llama-3.2-1B openai/gpt-oss-120b
```

### Multi-GPU (tensor parallelism)

Large models need multiple GPUs:

```bash
./run_compile_bench.sh --tp 8
```

### Force a clean rebuild

Builds are cached automatically — re-running the script skips steps already done. To start completely fresh:

```bash
./run_compile_bench.sh --force-rebuild
```

### Include eager baseline

```bash
./run_compile_bench.sh --eager
```

### All options

```
./run_compile_bench.sh [script-options] [benchmark-options]

Script options:
  --force-rebuild  Remove repos + conda env and rebuild from scratch

Benchmark options (passed through to benchmark_compile_time.py):
  --models M [M..] HuggingFace model names
  --eager          Also benchmark eager mode (no torch.compile)
  --output PATH    Output CSV path
  --dtype TYPE     Model dtype (default: bfloat16)
  --tp N           Tensor parallel size (default: 8)
```

## Output

Results are written to `/tmp/compile_bench_results/pytorch-{tip,release}/`.

### CSV

One row per run date, two columns per model (PT2 cold compile time, PT2 warm compile time in seconds):

```
date,       Llama-3.2-1B/pt2_cold, Llama-3.2-1B/pt2_warm, gpt-oss-120b/pt2_cold, ...
2026-04-24, 7.89,                  1.49,                   ...,                    ...
```

Append-mode: running the benchmark again adds a new row, so you can track regressions over time.

### JSON

Full detailed results including load times, first-gen latency, steady-state throughput, dynamo time, and cache hit status.

## What It Measures

For each model, the benchmark runs two passes:

1. **Cold start** — clears the vLLM compile cache (`~/.cache/vllm/torch_compile_cache`), then loads the model. The PT2 compile time includes Dynamo graph capture + Inductor compilation.

2. **Warm start** — reuses the compile cache from the cold run. Inductor compilation is skipped; only cache loading and Dynamo bytecode transform happen.

The compile times are extracted from vLLM's engine subprocess logs (`torch.compile took X.XX s in total`).

## Prerequisites

- NVIDIA GPU (H100 recommended)
- CUDA 13.0 (`/usr/local/cuda-13.0`). Install with: `sudo dnf install cuda-toolkit-13-0`
- Conda (Miniconda or Anaconda)
- HuggingFace token with access to gated models (set via `HF_TOKEN` env var)

## How the Source Build Works (pytorch-tip only)

PyTorch v2.11.0 ships with pybind11 3.x, which introduced typed `py::make_tuple` return types. This causes C++ ternary expressions with different-arity tuples to fail compilation. The script patches three files:

- `torch/csrc/utils/python_arg_parser.cpp` — wrap ternary branches in `py::object(...)`
- `torch/csrc/jit/python/init.cpp` — same fix for `_jit_get_operation` lambda
- `torch/csrc/distributed/c10d/init.cpp` — construct `py::tuple` manually (can't use `-> py::object` return type due to pybind11 pickle `static_assert`)

Both PyTorch and vLLM are built with CUDA 13.0 (required by vLLM's deep_gemm dependency for `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`).
