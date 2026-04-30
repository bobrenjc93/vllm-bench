#!/usr/bin/env python3
"""
vLLM PT2 compile time benchmark.

Measures cold start (no cache), warm start (with cache), PT2 compile time,
and non-PT2 (eager) time for a set of models. Outputs results to a CSV file.

Usage:
    python benchmark_compile_time.py                               # all models
    python benchmark_compile_time.py --models meta-llama/Llama-3.2-1B
    python benchmark_compile_time.py --eager                       # include eager baseline
    python benchmark_compile_time.py --help
"""

import argparse
import csv
import gc
import getpass
import io
import json
import logging
import os
import re
import select
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-70B",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-4.7",
    # "MiniMaxAI/MiniMax-M2.7",  # vLLM FusedMoE bug: gate/up weight size % block_n != 0
    "Qwen/Qwen3.6-35B-A3B",
]

PROMPT = "Hello, my name is"
MAX_TOKENS = 4
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 30


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM PT2 compile time benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model names to benchmark",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Also benchmark eager mode (no torch.compile)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per model (default: 3). Median is reported.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: compile_bench_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization ratio (default: 0.85)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help="Max context length for KV cache (default: 256). "
             "Kept minimal since this benchmarks compile time, not serving.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Max number of sequences (default: 1). "
             "Kept minimal to reduce memory usage.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per model before failing (default: 5). "
             "Moves on after the first successful run.",
    )
    return parser.parse_args()


def get_vllm_compile_cache_dir():
    return Path.home() / ".cache" / "vllm" / "torch_compile_cache"


def clear_compile_cache():
    cache_dirs = [
        get_vllm_compile_cache_dir(),
        Path(f"/tmp/torchinductor_{getpass.getuser()}"),
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  Cleared cache: {cache_dir}")


def clear_dynamo_cache():
    import torch._dynamo
    torch._dynamo.reset()


def parse_compile_time(log_text):
    """Extract torch.compile timing from vLLM engine logs."""
    result = {
        "torch_compile_total": 0.0,
        "dynamo_time": 0.0,
        "cache_hit": False,
    }
    match = re.search(r"torch\.compile (?:took|takes) ([0-9.]+) s in total", log_text)
    if match:
        result["torch_compile_total"] = float(match.group(1))
    match = re.search(r"Dynamo bytecode transform time: ([0-9.]+) s", log_text)
    if match:
        result["dynamo_time"] = float(match.group(1))
    result["cache_hit"] = "Directly load the compiled graph" in log_text
    return result


_WORKER_SCRIPT = '''
import json, sys, time, gc, os
os.environ.setdefault("HF_TOKEN", "{hf_token}")
os.environ.setdefault("DG_JIT_NVCC_COMPILER", "{dg_nvcc}")
os.environ.setdefault("VLLM_WORKER_TIMEOUT", "600")
import torch
from vllm import LLM, SamplingParams

load_start = time.perf_counter()
llm = LLM(
    model="{model_name}",
    dtype="{dtype}",
    tensor_parallel_size={tp_size},
    gpu_memory_utilization={gpu_mem_util},
    max_model_len={max_model_len},
    max_num_seqs={max_num_seqs},
    enforce_eager={enforce_eager},
    load_format="dummy",
    trust_remote_code=True,
)
load_time = time.perf_counter() - load_start
print("BENCH_LOAD:" + json.dumps({{"load_time": load_time}}), flush=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens={max_tokens})
gen_start = time.perf_counter()
outputs = llm.generate(["{prompt}"], sampling_params)
first_gen_time = time.perf_counter() - gen_start

gen_start = time.perf_counter()
for _ in range(3):
    outputs = llm.generate(["{prompt}"], sampling_params)
steady_gen_time = (time.perf_counter() - gen_start) / 3

result = {{
    "load_time": load_time,
    "first_gen_time": first_gen_time,
    "steady_gen_time": steady_gen_time,
}}
print("BENCH_RESULT:" + json.dumps(result), flush=True)
del llm
gc.collect()
torch.cuda.empty_cache()
'''


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# Shared progress state, set by main loop before each call
_progress = {"label": "", "start": 0.0}


def _spinner_thread(stop_event, label_fn):
    """Background thread that prints a live elapsed-time status line."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        elapsed = time.perf_counter() - _progress["start"]
        frame = frames[i % len(frames)]
        line = f"\r  {frame} {_progress['label']} [{fmt_elapsed(elapsed)}]"
        sys.stderr.write(f"\033[2K{line}")
        sys.stderr.flush()
        i += 1
        stop_event.wait(0.25)
    sys.stderr.write("\033[2K\r")
    sys.stderr.flush()


def run_benchmark_single(
    model_name, dtype, gpu_mem_util, tp_size, max_model_len, max_num_seqs=1,
    enforce_eager=False,
):
    """Run a single benchmark as a subprocess to capture EngineCore logs."""
    mode = "eager" if enforce_eager else "pt2"
    _progress["start"] = time.perf_counter()

    hf_token = os.environ.get("HF_TOKEN", "")
    dg_nvcc = os.environ.get("DG_JIT_NVCC_COMPILER", "")
    script = _WORKER_SCRIPT.format(
        hf_token=hf_token,
        dg_nvcc=dg_nvcc,
        model_name=model_name,
        dtype=dtype,
        tp_size=tp_size,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        max_tokens=MAX_TOKENS,
        prompt=PROMPT,
    )

    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_spinner_thread, args=(stop_spinner, None), daemon=True)
    spinner.start()

    try:
        popen = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = popen.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            popen.wait()
            raise
        proc = subprocess.CompletedProcess(
            popen.args, popen.returncode, stdout, stderr,
        )
    finally:
        stop_spinner.set()
        spinner.join()

    wall_time = time.perf_counter() - _progress["start"]

    all_output = proc.stdout + "\n" + proc.stderr
    # Print last few meaningful lines for context (skip noise)
    important_lines = []
    for line in all_output.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("BENCH_RESULT:") or stripped.startswith("BENCH_LOAD:"):
            continue
        if any(kw in stripped for kw in ["ERROR", "Error", "FAILED", "raise ", "Traceback",
                                          "torch.compile took", "torch.compile takes",
                                          "Dynamo bytecode",
                                          "Directly load the compiled graph"]):
            important_lines.append(line)
    if proc.returncode != 0:
        for line in all_output.split("\n"):
            if line.strip() and not line.startswith("BENCH_RESULT:") and not line.startswith("BENCH_LOAD:"):
                print(f"    {line}")
    elif important_lines:
        for line in important_lines[-5:]:
            print(f"    {line}")

    results = {}

    # Parse compile time from subprocess logs (EngineCore prints to stdout)
    compile_info = parse_compile_time(all_output)

    # Parse timing from subprocess JSON output
    bench_result_found = False
    for line in proc.stdout.split("\n"):
        if line.startswith("BENCH_RESULT:"):
            bench_data = json.loads(line[len("BENCH_RESULT:"):])
            results["load_time"] = bench_data["load_time"]
            results["first_gen_time"] = bench_data["first_gen_time"]
            results["steady_gen_time"] = bench_data["steady_gen_time"]
            results["total_startup"] = bench_data["load_time"]
            bench_result_found = True
            break

    if not bench_result_found:
        if compile_info["torch_compile_total"] > 0:
            results["load_time"] = -1.0
            results["total_startup"] = -1.0
            results["first_gen_time"] = -1.0
            results["steady_gen_time"] = -1.0
            print(f"  Inference failed but compile time recovered ({compile_info['torch_compile_total']:.1f}s)")
        elif proc.returncode != 0:
            raise RuntimeError(f"Benchmark subprocess failed (exit {proc.returncode}): {proc.stderr[-500:]}")
        else:
            raise RuntimeError("No BENCH_RESULT found in subprocess output")

    results["compilation_time"] = compile_info["torch_compile_total"]
    results["dynamo_time"] = compile_info["dynamo_time"]
    results["cache_hit"] = compile_info["cache_hit"]

    load_str = f"{results['load_time']:.1f}s" if results['load_time'] >= 0 else "N/A"
    gen_str = f"{results['first_gen_time']:.1f}s" if results['first_gen_time'] >= 0 else "N/A"
    print(f"  [{mode}] Done in {fmt_elapsed(wall_time)} — "
          f"load: {load_str}, "
          f"compile: {compile_info['torch_compile_total']:.1f}s, "
          f"gen: {gen_str}")

    return results


def _median_of(runs, key):
    vals = [r[key] for r in runs if r.get(key, -1) >= 0]
    return statistics.median(vals) if vals else -1.0


def benchmark_model(model_name, args, model_idx=0, total_models=0):
    """Benchmark a single model: cold start, warm start, optionally eager."""
    short = short_model_name(model_name)
    progress_prefix = f"[{model_idx}/{total_models}]" if total_models else ""
    repeats = args.repeats
    num_phases = repeats * (3 if args.eager else 2)
    phase = 0

    print(f"\n{'='*60}")
    print(f"{progress_prefix} Benchmarking: {model_name} ({repeats} repeats)")
    print(f"{'='*60}")

    cold_runs = []
    warm_runs = []
    eager_runs = []

    for r in range(repeats):
        # --- Cold start (no compile cache) ---
        phase += 1
        _progress["label"] = f"{progress_prefix} {short} — cold start (repeat {r+1}/{repeats}, phase {phase}/{num_phases})"
        print(f"\n--- Cold Start (PT2, no cache) [repeat {r+1}/{repeats}] ---")
        clear_compile_cache()
        cold_results = run_benchmark_single(
            model_name, args.dtype, args.gpu_memory_utilization, args.tp, args.max_model_len,
            args.max_num_seqs, enforce_eager=False,
        )
        cold_runs.append(cold_results)

        # --- Warm start (compile cache exists from cold run) ---
        phase += 1
        _progress["label"] = f"{progress_prefix} {short} — warm start (repeat {r+1}/{repeats}, phase {phase}/{num_phases})"
        print(f"\n--- Warm Start (PT2, with cache) [repeat {r+1}/{repeats}] ---")
        warm_results = run_benchmark_single(
            model_name, args.dtype, args.gpu_memory_utilization, args.tp, args.max_model_len,
            args.max_num_seqs, enforce_eager=False,
        )
        warm_runs.append(warm_results)

    if args.eager:
        for r in range(repeats):
            phase += 1
            _progress["label"] = f"{progress_prefix} {short} — eager (repeat {r+1}/{repeats}, phase {phase}/{num_phases})"
            print(f"\n--- Eager Mode [repeat {r+1}/{repeats}] ---")
            eager_results = run_benchmark_single(
                model_name, args.dtype, args.gpu_memory_utilization, args.tp, args.max_model_len,
                args.max_num_seqs, enforce_eager=True,
            )
            eager_runs.append(eager_results)

    model_results = {}
    model_results["cold_start_total"] = _median_of(cold_runs, "total_startup")
    model_results["cold_start_load"] = _median_of(cold_runs, "load_time")
    model_results["cold_pt2_compile_time"] = _median_of(cold_runs, "compilation_time")
    model_results["cold_first_gen"] = _median_of(cold_runs, "first_gen_time")
    model_results["cold_steady_gen"] = _median_of(cold_runs, "steady_gen_time")

    model_results["warm_start_total"] = _median_of(warm_runs, "total_startup")
    model_results["warm_start_load"] = _median_of(warm_runs, "load_time")
    model_results["warm_pt2_compile_time"] = _median_of(warm_runs, "compilation_time")
    model_results["warm_first_gen"] = _median_of(warm_runs, "first_gen_time")
    model_results["warm_steady_gen"] = _median_of(warm_runs, "steady_gen_time")

    if args.eager:
        model_results["eager_load"] = _median_of(eager_runs, "load_time")
        model_results["eager_first_gen"] = _median_of(eager_runs, "first_gen_time")
        model_results["eager_steady_gen"] = _median_of(eager_runs, "steady_gen_time")

    model_results["cold_runs"] = cold_runs
    model_results["warm_runs"] = warm_runs
    if args.eager:
        model_results["eager_runs"] = eager_runs
    model_results["repeats"] = repeats

    print(f"\n  Median (n={repeats}): cold compile={model_results['cold_pt2_compile_time']:.1f}s, "
          f"warm compile={model_results['warm_pt2_compile_time']:.1f}s")

    return model_results


def short_model_name(name):
    """Get a concise display name: 'openai/gpt-oss-120b' -> 'gpt-oss-120b'."""
    return name.split("/")[-1] if "/" in name else name


def _get_compile_times(result):
    return result.get("cold_pt2_compile_time", -1.0), result.get("warm_pt2_compile_time", -1.0)


def build_csv(all_results, args, output_path):
    """Write CSV: date as row, models as column groups (PT2 cold / PT2 warm)."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    headers = ["date"]
    for model_name in all_results:
        short = short_model_name(model_name)
        headers.append(f"{short}/pt2_cold")
        headers.append(f"{short}/pt2_warm")

    row = [date_str]
    for model_name in all_results:
        cold, warm = _get_compile_times(all_results[model_name])
        row.append(f"{cold:.2f}")
        row.append(f"{warm:.2f}")

    file_exists = Path(output_path).exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"\nResults written to: {output_path}")


def print_summary(all_results, args):
    """Print a human-readable summary table."""
    print(f"\n{'='*70}")
    print(f"SUMMARY — PT2 Compile Time, median of {args.repeats} (seconds)")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'Cold (no cache)':>14} {'Warm (cached)':>14}")
    print("-" * 70)

    for model_name, results in all_results.items():
        short = short_model_name(model_name)
        cold, warm = _get_compile_times(results)
        print(
            f"{short:<40} "
            f"{cold:>13.2f}s "
            f"{warm:>13.2f}s"
        )


def kill_gpu_processes():
    result = subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True,
    )
    pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    for pid in pids:
        subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
    if pids:
        time.sleep(3)
        print(f"  Killed {len(pids)} GPU processes")


def download_model_metadata(model, hf_token):
    """Download model config and tokenizer files (not weights, since we use dummy loading)."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            snapshot_download(
                model,
                token=hf_token,
                allow_patterns=[
                    "tokenizer*", "special_tokens*",
                    "*.json", "*.model",
                ],
            )
            print(f"  Done: {model}")
            return True
        except Exception as e:
            if attempt < DOWNLOAD_MAX_RETRIES:
                delay = DOWNLOAD_RETRY_DELAY * attempt
                print(f"  Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} failed: {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  FAILED after {DOWNLOAD_MAX_RETRIES} attempts: {e}")
                return False


def download_models_metadata(models, hf_token):
    """Pre-download config/tokenizer files for all models (weights skipped — using dummy load)."""
    print("\n--- Pre-downloading model configs & tokenizers (weights skipped: dummy load) ---")

    failed = []
    for model in models:
        print(f"  Downloading metadata for {model}...")
        if not download_model_metadata(model, hf_token):
            failed.append(model)

    if failed:
        print(f"\n  WARNING: Failed to download metadata for {len(failed)} model(s): {failed}")
        print(f"  These models will likely fail during benchmarking.")
    return failed


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN=hf_your_token_here")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    download_models_metadata(args.models, hf_token)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"compile_bench_{timestamp}.csv"

    print(f"\nvLLM PT2 Compile Time Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {args.models}")
    print(f"dtype: {args.dtype}, TP: {args.tp}")
    print(f"Eager baseline: {args.eager}")
    print(f"Repeats: {args.repeats} (reporting median)")
    print(f"Output: {output_path}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    try:
        import vllm
        print(f"vLLM: {vllm.__version__}")
    except AttributeError:
        print("vLLM: (version unavailable)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}
    total = len(args.models)
    bench_start = time.perf_counter()
    for i, model_name in enumerate(args.models, 1):
        last_err = None
        for attempt in range(1, args.max_retries + 1):
            kill_gpu_processes()
            try:
                model_results = benchmark_model(model_name, args, model_idx=i, total_models=total)
                all_results[model_name] = model_results
                break
            except Exception as e:
                last_err = e
                print(f"\n  ATTEMPT {attempt}/{args.max_retries} FAILED for {model_name}: {e}")
                if attempt < args.max_retries:
                    print(f"  Retrying...")
        else:
            print(f"\n  ALL {args.max_retries} ATTEMPTS FAILED for {model_name}")
            raise RuntimeError(f"Model {model_name} failed after {args.max_retries} attempts") from last_err
        elapsed = fmt_elapsed(time.perf_counter() - bench_start)
        print(f"\n  Progress: {i}/{total} models done — total elapsed {elapsed}")

    if all_results:
        print_summary(all_results, args)
        build_csv(all_results, args, output_path)

    also_json = output_path.replace(".csv", ".json")
    with open(also_json, "w") as f:
        def _round_floats(obj):
            if isinstance(obj, float):
                return round(obj, 4)
            if isinstance(obj, dict):
                return {k: _round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_round_floats(v) for v in obj]
            return obj
        json.dump({
            "date": datetime.now().isoformat(),
            "args": vars(args),
            "results": _round_floats(all_results),
        }, f, indent=2)
    print(f"Full results (JSON): {also_json}")


if __name__ == "__main__":
    main()
