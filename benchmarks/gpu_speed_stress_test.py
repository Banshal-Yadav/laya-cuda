"""Comprehensive GPU Speed, Latency Percentiles & Throughput Stress Test on NVIDIA RTX 4050.
Evaluates Laya and DeBERTa-v3-large across batch sizes 1 to 64, CUDA Graph Replay, and Latency Percentiles (P50, P95, P99).
"""

import sys
import time
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
from typed_decisions.open_jev import OpenJev

print("=" * 105)
print("NVIDIA RTX 4050 GPU SPEED & THROUGHPUT STRESS TEST BENCHMARK")
print("Device: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) | Precision: FP16 / FP32")
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
    }
]

# ----------------------------------------------------
# 1. LAYA CUDA GRAPH SPEED TEST (1,000 RUNS)
# ----------------------------------------------------
print("\n[1/4] Running Laya CUDA Graph Engine Speed Test (1,000 Decisions)...")
agent_gpu = laya.load("convaiinnovations/laya", device="cuda", dtype="float16")
fast_graph = FastDecisionEngine(agent_gpu)

# Warmup
for _ in range(50):
    fast_graph.decide("Snake at (5, 5). Food at (8, 5).")
torch.cuda.synchronize()

graph_latencies = []
t_start_total = time.perf_counter()

for _ in range(1000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = fast_graph.decide("Snake at (5, 5). Food at (8, 5).")
    torch.cuda.synchronize()
    graph_latencies.append((time.perf_counter() - t0) * 1000)

total_graph_time = time.perf_counter() - t_start_total
graph_latencies.sort()

p50_g = np.percentile(graph_latencies, 50)
p95_g = np.percentile(graph_latencies, 95)
p99_g = np.percentile(graph_latencies, 99)
min_g = np.min(graph_latencies)
max_g = np.max(graph_latencies)
throughput_g = 1000.0 / total_graph_time

print(f"  - P50 Latency:   {p50_g:6.2f} ms")
print(f"  - P95 Latency:   {p95_g:6.2f} ms")
print(f"  - P99 Latency:   {p99_g:6.2f} ms")
print(f"  - Min / Max:     {min_g:6.2f} ms / {max_g:6.2f} ms")
print(f"  - Throughput:    {throughput_g:6.1f} decisions / sec")

# ----------------------------------------------------
# 2. LAYA BATCH INFERENCE THROUGHPUT (BATCH 1 to 64)
# ----------------------------------------------------
print("\n[2/4] Running Laya GPU Batch Throughput Test (Batch Sizes: 1, 4, 16, 32, 64)...")

batch_results = []
for bs in [1, 4, 16, 32, 64]:
    batch_items = [(test_state, test_questions_laya)] * bs
    
    # Warmup
    _ = agent_gpu.batch_predict(batch_items)
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(15):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = agent_gpu.batch_predict(batch_items)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        
    avg_batch_ms = float(np.mean(latencies))
    ms_per_item = avg_batch_ms / bs
    tps = 1000.0 / ms_per_item
    batch_results.append({"bs": bs, "batch_ms": avg_batch_ms, "item_ms": ms_per_item, "tps": tps})
    print(f"  - Batch Size {bs:2d}: {avg_batch_ms:6.2f} ms total | {ms_per_item:5.2f} ms/item | Throughput: {tps:6.1f} items/sec")

# ----------------------------------------------------
# 3. DEBERTA-V3-LARGE LATENCY PERCENTILES (100 RUNS)
# ----------------------------------------------------
print("\n[3/4] Running DeBERTa-v3-large GPU Latency Test (100 Decisions)...")
jev_gpu = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup
for _ in range(10):
    _ = jev_gpu.decide(test_state, test_questions_deberta)
torch.cuda.synchronize()

d_latencies = []
t_start_d = time.perf_counter()

for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = jev_gpu.decide(test_state, test_questions_deberta)
    torch.cuda.synchronize()
    d_latencies.append((time.perf_counter() - t0) * 1000)

total_d_time = time.perf_counter() - t_start_d
d_latencies.sort()

p50_d = np.percentile(d_latencies, 50)
p95_d = np.percentile(d_latencies, 95)
p99_d = np.percentile(d_latencies, 99)
min_d = np.min(d_latencies)
max_d = np.max(d_latencies)
throughput_d = 100.0 / total_d_time

print(f"  - P50 Latency:   {p50_d:6.2f} ms")
print(f"  - P95 Latency:   {p95_d:6.2f} ms")
print(f"  - P99 Latency:   {p99_d:6.2f} ms")
print(f"  - Min / Max:     {min_d:6.2f} ms / {max_d:6.2f} ms")
print(f"  - Throughput:    {throughput_d:6.1f} decisions / sec")

# ----------------------------------------------------
# 4. SUMMARY TABLE
# ----------------------------------------------------
print("\n" + "=" * 105)
print("NVIDIA RTX 4050 GPU SPEED & THROUGHPUT SUMMARY")
print("=" * 105)
print(f"Laya CUDA Graph Replay (Single Packet) :  P50 = {p50_g:5.2f} ms | P99 = {p99_g:5.2f} ms | Throughput = {throughput_g:6.1f} decisions/sec")
print(f"Laya Batch 64 Max Throughput           :  Per Item = {batch_results[-1]['item_ms']:5.2f} ms | Throughput = {batch_results[-1]['tps']:6.1f} decisions/sec")
print(f"DeBERTa-v3-large (Single Packet FP16)   :  P50 = {p50_d:5.2f} ms | P99 = {p99_d:5.2f} ms | Throughput = {throughput_d:6.1f} decisions/sec")
print("=" * 105)
