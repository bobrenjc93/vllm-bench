#!/usr/bin/env python3
"""Quick one-shot: get compile time for each model with minimal memory."""

import json, os, re, subprocess, sys, time
from datetime import datetime

MODELS = [
    "meta-llama/Meta-Llama-3-70B",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-4.7",
    "Qwen/Qwen3.6-35B-A3B",
]

WORKER = '''
import json, sys, time, gc, os
os.environ.setdefault("HF_TOKEN", "{hf_token}")
import torch
from vllm import LLM, SamplingParams

load_start = time.perf_counter()
llm = LLM(
    model="{model}",
    dtype="bfloat16",
    tensor_parallel_size=8,
    gpu_memory_utilization=0.95,
    max_model_len=256,
    max_num_seqs=1,
    enforce_eager=False,
    load_format="dummy",
    trust_remote_code=True,
)
load_time = time.perf_counter() - load_start

sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
gen_start = time.perf_counter()
outputs = llm.generate(["Hello"], sampling_params)
gen_time = time.perf_counter() - gen_start

print("BENCH_RESULT:" + json.dumps({{"load_time": load_time, "gen_time": gen_time}}), flush=True)
del llm
gc.collect()
torch.cuda.empty_cache()
'''

RESULTS_FILE = "/home/bobren/vllm-bench/compile_times_quick.jsonl"


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


def run_model(model, timeout=7200):
    kill_gpu_processes()
    hf_token = os.environ.get("HF_TOKEN", "")
    script = WORKER.format(hf_token=hf_token, model=model)

    print(f"\n{'='*60}")
    print(f"Running: {model}")
    print(f"{'='*60}")

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        kill_gpu_processes()
        return {"model": model, "status": "timeout"}

    wall = time.perf_counter() - start
    all_output = proc.stdout + "\n" + proc.stderr

    # Extract compile time from logs
    compile_time = -1.0
    m = re.search(r"torch\.compile (?:took|takes) ([0-9.]+) s in total", all_output)
    if m:
        compile_time = float(m.group(1))

    # Extract bench result
    bench = {}
    for line in proc.stdout.split("\n"):
        if line.startswith("BENCH_RESULT:"):
            bench = json.loads(line[len("BENCH_RESULT:"):])
            break

    if proc.returncode != 0:
        # Print last meaningful error lines
        for line in all_output.split("\n"):
            s = line.strip()
            if any(kw in s for kw in ["ERROR", "ValueError", "RuntimeError", "Traceback", "raise "]):
                print(f"  {s}")
        print(f"  EXIT CODE: {proc.returncode}")
        print(f"  compile_time extracted: {compile_time}")
        return {
            "model": model, "status": "failed",
            "compile_time": compile_time, "wall_time": wall,
            "exit_code": proc.returncode,
        }

    result = {
        "model": model, "status": "ok",
        "compile_time": compile_time,
        "load_time": bench.get("load_time", -1),
        "gen_time": bench.get("gen_time", -1),
        "wall_time": wall,
    }
    print(f"  compile: {compile_time:.1f}s, load: {result['load_time']:.1f}s, "
          f"gen: {result['gen_time']:.1f}s, wall: {wall:.0f}s")
    return result


def main():
    print(f"Quick compile time extraction — {datetime.now()}")
    print(f"Results appended to: {RESULTS_FILE}")

    for model in MODELS:
        result = run_model(model)
        result["timestamp"] = datetime.now().isoformat()

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  -> saved to {RESULTS_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            short = r["model"].split("/")[-1]
            ct = r.get("compile_time", -1)
            status = r["status"]
            print(f"  {short:<30} compile={ct:>7.1f}s  [{status}]")


if __name__ == "__main__":
    main()
