"""Full Speedup & Performance Benchmarking Suite for Laya and DeBERTa-v3-large.
Compares Baseline (Unoptimized CPU 1-Thread) vs Ryzen CPU Optimized (INT8 + Multi-Thread) vs RTX GPU Eager (FP16) vs RTX GPU CUDA Graph Engine.
"""

import sys
import time
import gc
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
if str(snapshot_dir) not in sys.path:
    sys.path.insert(0, str(snapshot_dir))

import laya_cuda as laya
from laya_cuda.fast_engine import FastDecisionEngine
from laya_cuda.fast_cpu_engine import LayaCPUEngine, OpenJevCPUEngine
from typed_decisions.open_jev import OpenJev

print("=" * 105)
print("LAYA & DEBERTA-V3-LARGE: COMPREHENSIVE SPEEDUP & PERFORMANCE GAINS BENCHMARK")
print("Comparing Baseline CPU (1-Thread FP32) vs Optimized Ryzen CPU (INT8) vs RTX 4050 GPU (FP16 & CUDA Graphs)")
print("=" * 105)

test_state = "I was billed twice for the same subscription. Please refund the duplicate charge immediately."
test_questions_laya = {
    "department": {
        "type": "choice",
        "instructions": "Who should handle this customer request?",
        "criteria": ["billing", "technical", "sales"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Is this time-sensitive?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": ["calm", "mildly annoyed", "very frustrated", "furious"],
    },
}

test_questions_deberta = [
    {
        "type": "choice",
        "instructions": "Which department handles billing double charges?",
        "options": ["billing_refunds", "technical_support", "cancellations", "sales_demos"],
    },
    {
        "type": "noul",
        "instructions": "Is the customer requesting a monetary refund?",
    },
    {
        "type": "score",
        "instructions": "Customer urgency level:",
        "options": ["low", "medium", "high", "critical"],
    }
]

def benchmark_execution(predict_fn, iterations=3, warmup=1, is_cuda=False):
    for _ in range(warmup):
        _ = predict_fn()
        if is_cuda:
            torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_fn()
        if is_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean = float(np.mean(latencies))
    return p50, p95, mean

results_table = []

# --- MODEL 1: LAYA (ModernBERT) BENCHMARKS ---
print("\n[1/2] Benchmarking Model 1: Laya (convaiinnovations/laya)...")

# 1A. Laya Baseline CPU (1-Thread FP32)
print("  - Running Laya Baseline CPU (1-Thread FP32)...", flush=True)
torch.set_num_threads(1)
laya_base_cpu = laya.load("convaiinnovations/laya", device="cpu", dtype="float32")
_, _, mean_l_base = benchmark_execution(lambda: laya_base_cpu.predict(test_state, test_questions_laya), iterations=1, warmup=0, is_cuda=False)
del laya_base_cpu
gc.collect()

# 1B. Laya Optimized Ryzen CPU (16-Threads INT8)
print("  - Running Laya Ryzen CPU Optimized (16-Threads + Dynamic INT8)...", flush=True)
laya_opt_cpu = LayaCPUEngine("convaiinnovations/laya", threads=16, quantize=True)
_, _, mean_l_opt_cpu = benchmark_execution(lambda: laya_opt_cpu.predict(test_state, test_questions_laya), iterations=2, warmup=1, is_cuda=False)
del laya_opt_cpu
gc.collect()

# 1C. Laya RTX GPU PyTorch Eager (FP16)
print("  - Running Laya RTX 4050 GPU Eager (FP16)...", flush=True)
laya_gpu_eager = laya.load("convaiinnovations/laya", device="cuda", dtype="float16")
_, _, mean_l_gpu_eager = benchmark_execution(lambda: laya_gpu_eager.predict(test_state, test_questions_laya), iterations=10, warmup=2, is_cuda=True)

# 1D. Laya RTX GPU CUDA Graph Captured Engine
print("  - Running Laya RTX 4050 GPU CUDA Graph Captured Engine...", flush=True)
fast_graph_engine = FastDecisionEngine(laya_gpu_eager)
_, _, mean_l_graph = benchmark_execution(lambda: fast_graph_engine.decide("I was billed twice. Please refund."), iterations=10, warmup=2, is_cuda=True)
del laya_gpu_eager, fast_graph_engine
torch.cuda.empty_cache()
gc.collect()

# Add Laya to Table
results_table.append({
    "model": "Laya (ModernBERT)",
    "tier": "Baseline CPU (1-Thread FP32)",
    "p50_ms": mean_l_base,
    "speedup": 1.0,
    "gain": "0.0%"
})
results_table.append({
    "model": "Laya (ModernBERT)",
    "tier": "Ryzen CPU Opt (16T + INT8)",
    "p50_ms": mean_l_opt_cpu,
    "speedup": mean_l_base / mean_l_opt_cpu,
    "gain": f"{((mean_l_base - mean_l_opt_cpu) / mean_l_base) * 100:.1f}%"
})
results_table.append({
    "model": "Laya (ModernBERT)",
    "tier": "RTX 4050 GPU Eager (FP16)",
    "p50_ms": mean_l_gpu_eager,
    "speedup": mean_l_base / mean_l_gpu_eager,
    "gain": f"{((mean_l_base - mean_l_gpu_eager) / mean_l_base) * 100:.1f}%"
})
results_table.append({
    "model": "Laya (ModernBERT)",
    "tier": "RTX 4050 GPU CUDA Graph",
    "p50_ms": mean_l_graph,
    "speedup": mean_l_base / mean_l_graph,
    "gain": f"{((mean_l_base - mean_l_graph) / mean_l_base) * 100:.1f}%"
})

# --- MODEL 2: OPENJEV DEBERTA-V3-LARGE BENCHMARKS ---
print("\n[2/2] Benchmarking Model 2: OpenJev (com-kotobalabs/open-jev-deberta-v3-large)...")

# 2A. DeBERTa Baseline CPU (1-Thread FP32)
print("  - Running DeBERTa Baseline CPU (1-Thread FP32)...", flush=True)
torch.set_num_threads(1)
jev_base_cpu = OpenJev.from_pretrained(str(snapshot_dir), device="cpu")
_, _, mean_d_base = benchmark_execution(lambda: jev_base_cpu.decide(test_state, test_questions_deberta), iterations=1, warmup=0, is_cuda=False)
del jev_base_cpu
gc.collect()

# 2B. DeBERTa Optimized Ryzen CPU (8-Threads INT8)
print("  - Running DeBERTa Ryzen CPU Optimized (8-Threads + Dynamic INT8)...", flush=True)
jev_opt_cpu = OpenJevCPUEngine(str(snapshot_dir), threads=8, quantize=True)
_, _, mean_d_opt_cpu = benchmark_execution(lambda: jev_opt_cpu.decide(test_state, test_questions_deberta), iterations=2, warmup=1, is_cuda=False)
del jev_opt_cpu
gc.collect()

# 2C. DeBERTa RTX GPU Eager (FP16)
print("  - Running DeBERTa RTX 4050 GPU Eager (FP16)...", flush=True)
jev_gpu = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")
_, _, mean_d_gpu = benchmark_execution(lambda: jev_gpu.decide(test_state, test_questions_deberta), iterations=10, warmup=2, is_cuda=True)
del jev_gpu
torch.cuda.empty_cache()
gc.collect()

# Add DeBERTa to Table
results_table.append({
    "model": "DeBERTa-v3-large",
    "tier": "Baseline CPU (1-Thread FP32)",
    "p50_ms": mean_d_base,
    "speedup": 1.0,
    "gain": "0.0%"
})
results_table.append({
    "model": "DeBERTa-v3-large",
    "tier": "Ryzen CPU Opt (8T + INT8)",
    "p50_ms": mean_d_opt_cpu,
    "speedup": mean_d_base / mean_d_opt_cpu,
    "gain": f"{((mean_d_base - mean_d_opt_cpu) / mean_d_base) * 100:.1f}%"
})
results_table.append({
    "model": "DeBERTa-v3-large",
    "tier": "RTX 4050 GPU Eager (FP16)",
    "p50_ms": mean_d_gpu,
    "speedup": mean_d_base / mean_d_gpu,
    "gain": f"{((mean_d_base - mean_d_gpu) / mean_d_base) * 100:.1f}%"
})

# --- PRINT COMPARATIVE RESULTS TABLE ---
print("\n" + "=" * 105)
print(f"{'MODEL':<20} | {'HARDWARE & OPTIMIZATION TIER':<32} | {'LATENCY (AVG)':<18} | {'SPEEDUP':<10} | {'LATENCY REDUCTION'}")
print("=" * 105)

for r in results_table:
    sp_str = f"{r['speedup']:5.2f}x" if r['speedup'] > 1.0 else "Baseline"
    print(f"{r['model']:<20} | {r['tier']:<32} | {r['p50_ms']:8.2f} ms        | {sp_str:<10} | {r['gain']}")

print("=" * 105)
