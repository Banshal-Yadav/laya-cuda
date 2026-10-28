"""
AMD Ryzen CPU Optimization & Benchmarking Framework for Laya & OpenJev DeBERTa-v3-large
---------------------------------------------------------------------------------------
Benchmarks baseline vs thread-tuned vs dynamic INT8 quantized CPU inference for both:
1. convaiinnovations/laya (ModernBERT architecture)
2. com-kotobalabs/open-jev-deberta-v3-large (DeBERTa-v3-large)
"""

import sys
import time
import os
import gc
import json
import psutil
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is in path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
if str(snapshot_dir) not in sys.path:
    sys.path.insert(0, str(snapshot_dir))

import laya_cuda as laya
from typed_decisions.open_jev import OpenJev

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def measure_fn(fn, warmup=2, iterations=10):
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "output": res,
    }

def benchmark_laya():
    print("\n" + "="*70, flush=True)
    print("MODEL 1 BENCHMARK: convaiinnovations/laya (ModernBERT)", flush=True)
    print("="*70, flush=True)

    state = "Customer state: Account #9872. User reported double billing on invoice #INV-4920 and requested refund."
    questions = {
        "dept": {
            "type": "choice",
            "instructions": "Which department handles double billing?",
            "criteria": ["billing_refunds", "technical_support", "account_security", "sales"],
        },
        "urgency": {
            "type": "score",
            "instructions": "Customer urgency level (0-3)",
            "criteria": ["low", "medium", "high", "critical"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Did the customer request a monetary refund?",
        },
    }

    # 1. Baseline FP32 Single Thread
    torch.set_num_threads(1)
    agent = laya.load("convaiinnovations/laya", device="cpu", dtype="float32")
    m_baseline = get_process_memory_mb()

    print("Running Baseline 1-Thread FP32...", flush=True)
    b_single = measure_fn(lambda: agent.predict(state, questions), warmup=1, iterations=3)
    print(f"Laya (1 Thread FP32):   p50 = {b_single['p50_ms']:8.2f} ms | mean = {b_single['mean_ms']:8.2f} ms")

    # 2. Thread Tuning Sweep
    thread_results = {}
    for threads in [2, 4, 8, 16]:
        torch.set_num_threads(threads)
        res = measure_fn(lambda: agent.predict(state, questions), warmup=1, iterations=5)
        thread_results[threads] = res
        print(f"Laya ({threads:2d} Threads FP32):  p50 = {res['p50_ms']:8.2f} ms | mean = {res['mean_ms']:8.2f} ms")

    optimal_threads = 16
    optimal_fp32_p50 = thread_results[16]['p50_ms']

    # 3. Dynamic INT8 Quantization
    print("\nApplying Dynamic INT8 Quantization (torch.ao.quantization.quantize_dynamic)...", flush=True)
    torch.set_num_threads(optimal_threads)
    m_before_q = get_process_memory_mb()

    agent.model.encoder = torch.ao.quantization.quantize_dynamic(agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8)
    agent.model.scorer = torch.ao.quantization.quantize_dynamic(agent.model.scorer, {torch.nn.Linear}, dtype=torch.qint8)
    agent.model.act_head = torch.ao.quantization.quantize_dynamic(agent.model.act_head, {torch.nn.Linear}, dtype=torch.qint8)

    m_after_q = get_process_memory_mb()

    res_int8 = measure_fn(lambda: agent.predict(state, questions), warmup=2, iterations=10)
    print(f"Laya ({optimal_threads:2d} Threads INT8):  p50 = {res_int8['p50_ms']:8.2f} ms | mean = {res_int8['mean_ms']:8.2f} ms")

    speedup_vs_1t = b_single['p50_ms'] / res_int8['p50_ms']
    speedup_vs_16t_fp32 = optimal_fp32_p50 / res_int8['p50_ms']

    print(f"\n[Laya Summary]")
    print(f"  Baseline 1-Thread FP32 p50:  {b_single['p50_ms']:8.2f} ms")
    print(f"  Tuned 16-Thread FP32 p50:   {optimal_fp32_p50:8.2f} ms ({b_single['p50_ms']/optimal_fp32_p50:.2f}x speedup)")
    print(f"  Tuned 16-Thread INT8 p50:   {res_int8['p50_ms']:8.2f} ms ({speedup_vs_1t:.2f}x speedup vs 1T, {speedup_vs_16t_fp32:.2f}x vs 16T FP32)")
    print(f"  Output Accuracy Match:       {b_single['output']['answers'] == res_int8['output']['answers']}")

    del agent
    gc.collect()

    return {
        "baseline_p50": b_single['p50_ms'],
        "fp32_16t_p50": optimal_fp32_p50,
        "int8_16t_p50": res_int8['p50_ms'],
        "int8_16t_p95": res_int8['p95_ms'],
        "speedup_vs_baseline": speedup_vs_1t,
        "answers_match": b_single['output']['answers'] == res_int8['output']['answers'],
    }

def benchmark_open_jev():
    print("\n" + "="*70, flush=True)
    print("MODEL 2 BENCHMARK: com-kotobalabs/open-jev-deberta-v3-large", flush=True)
    print("="*70, flush=True)

    state = "Customer: I was charged twice for the same subscription this month and need an immediate refund."
    questions = [
        {
            "type": "choice",
            "instructions": "Which team should handle this customer request?",
            "options": ["billing_refunds", "technical_support", "cancellations", "sales_demos"],
        },
        {
            "type": "score",
            "instructions": "How frustrated does the customer sound?",
            "options": ["calm_neutral", "concerned_civil", "annoyed", "furious"],
        },
        {
            "type": "noul",
            "instructions": "Does the customer request a financial refund or credit?",
        },
    ]

    # 1. Baseline FP32 Single Thread
    torch.set_num_threads(1)
    agent = OpenJev.from_pretrained(str(snapshot_dir), device="cpu")

    print("Running Baseline 1-Thread FP32...", flush=True)
    b_single = measure_fn(lambda: agent.decide(state, questions), warmup=1, iterations=3)
    print(f"OpenJev (1 Thread FP32):  p50 = {b_single['p50_ms']:8.2f} ms | mean = {b_single['mean_ms']:8.2f} ms")

    # 2. Thread Tuning Sweep
    thread_results = {}
    for threads in [2, 4, 8, 16]:
        torch.set_num_threads(threads)
        res = measure_fn(lambda: agent.decide(state, questions), warmup=1, iterations=5)
        thread_results[threads] = res
        print(f"OpenJev ({threads:2d} Threads FP32): p50 = {res['p50_ms']:8.2f} ms | mean = {res['mean_ms']:8.2f} ms")

    optimal_threads = 8
    optimal_fp32_p50 = thread_results[8]['p50_ms']

    # 3. Dynamic INT8 Quantization
    print("\nApplying Dynamic INT8 Quantization (torch.ao.quantization.quantize_dynamic)...", flush=True)
    torch.set_num_threads(optimal_threads)
    agent.model = torch.ao.quantization.quantize_dynamic(agent.model, {torch.nn.Linear}, dtype=torch.qint8)

    res_int8 = measure_fn(lambda: agent.decide(state, questions), warmup=2, iterations=10)
    print(f"OpenJev ({optimal_threads:2d} Threads INT8): p50 = {res_int8['p50_ms']:8.2f} ms | mean = {res_int8['mean_ms']:8.2f} ms")

    speedup_vs_1t = b_single['p50_ms'] / res_int8['p50_ms']
    speedup_vs_8t_fp32 = optimal_fp32_p50 / res_int8['p50_ms']

    print(f"\n[OpenJev Summary]")
    print(f"  Baseline 1-Thread FP32 p50:  {b_single['p50_ms']:8.2f} ms")
    print(f"  Tuned 8-Thread FP32 p50:    {optimal_fp32_p50:8.2f} ms ({b_single['p50_ms']/optimal_fp32_p50:.2f}x speedup)")
    print(f"  Tuned 8-Thread INT8 p50:    {res_int8['p50_ms']:8.2f} ms ({speedup_vs_1t:.2f}x speedup vs 1T, {speedup_vs_8t_fp32:.2f}x vs 8T FP32)")

    del agent
    gc.collect()

    return {
        "baseline_p50": b_single['p50_ms'],
        "fp32_8t_p50": optimal_fp32_p50,
        "int8_8t_p50": res_int8['p50_ms'],
        "int8_8t_p95": res_int8['p95_ms'],
        "speedup_vs_baseline": speedup_vs_1t,
    }

def main():
    print("=" * 70)
    print("AMD Ryzen CPU Inference Latency Optimization Benchmark")
    print(f"System CPU Cores: {os.cpu_count()} logical ({psutil.cpu_count(logical=False)} physical)")
    print("=" * 70)

    laya_res = benchmark_laya()
    jev_res = benchmark_open_jev()

    print("\n" + "=" * 70)
    print("FINAL CPU OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} | {'Baseline FP32':<13} | {'Thread-Tuned':<13} | {'Dynamic INT8':<13} | {'Speedup':<8}")
    print("-" * 90)
    print(f"{'convaiinnovations/laya (ModernBERT)':<35} | {laya_res['baseline_p50']:10.1f} ms | {laya_res['fp32_16t_p50']:10.1f} ms | {laya_res['int8_16t_p50']:10.1f} ms | {laya_res['speedup_vs_baseline']:6.2f}x")
    print(f"{'open-jev-deberta-v3-large':<35} | {jev_res['baseline_p50']:10.1f} ms | {jev_res['fp32_8t_p50']:10.1f} ms | {jev_res['int8_8t_p50']:10.1f} ms | {jev_res['speedup_vs_baseline']:6.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    main()
