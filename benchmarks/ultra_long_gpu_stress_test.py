"""Ultra-Long Large-Scale GPU Speed & Throughput Stress Test Benchmark.
Executes 5,000 CUDA Graph decisions, 500 DeBERTa-v3-large real-world decisions, and 2,560 batch decisions on NVIDIA RTX 4050 GPU.
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
print("ULTRA-LONG LARGE-SCALE GPU SPEED & THROUGHPUT STRESS TEST (NVIDIA RTX 4050)")
print("Running 5,000 CUDA Graph Requests + 500 DeBERTa FP16 Packets + 2,560 Batch Requests")
print("=" * 105)

test_state = "Customer state: Account #9872. User reported double billing on invoice #INV-4920 and requested refund."
test_questions_laya = {
    "department": {
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

test_questions_deberta = [
    {
        "type": "choice",
        "instructions": "Which team should handle this customer request?",
        "options": ["billing_refunds", "technical_support", "cancellations", "sales_demos"],
    },
    {
        "type": "noul",
        "instructions": "Does the customer request a financial refund or credit?",
    }
]

# ---------------------------------------------------------------------------
# TEST 1: 5,000 CUDA GRAPH SINGLE-PACKET DECISIONS
# ---------------------------------------------------------------------------
print("\n[TEST 1/3] Executing 5,000 CUDA Graph Hardware Replay Decisions on GPU...")
agent_laya = laya.load("convaiinnovations/laya", device="cuda", dtype="float16")
fast_graph = FastDecisionEngine(agent_laya)

# Warmup
for _ in range(100):
    fast_graph.decide("Snake at (5, 5). Food at (8, 5).")
torch.cuda.synchronize()

graph_latencies = []
t_start_1 = time.perf_counter()

for _ in range(5000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = fast_graph.decide("Snake at (5, 5). Food at (8, 5).")
    torch.cuda.synchronize()
    graph_latencies.append((time.perf_counter() - t0) * 1000)

t_end_1 = time.perf_counter() - t_start_1
graph_latencies.sort()

p50_1 = float(np.percentile(graph_latencies, 50))
p95_1 = float(np.percentile(graph_latencies, 95))
p99_1 = float(np.percentile(graph_latencies, 99))
throughput_1 = 5000.0 / t_end_1

print(f"  - Completed 5,000 Decisions in : {t_end_1:.2f} seconds")
print(f"  - P50 Latency                    : {p50_1:6.2f} ms")
print(f"  - P95 Latency                    : {p95_1:6.2f} ms")
print(f"  - P99 Latency                    : {p99_1:6.2f} ms")
print(f"  - Peak Single-GPU Throughput     : {throughput_1:6.1f} decisions / sec")

# ---------------------------------------------------------------------------
# TEST 2: 2,560 LARGE-BATCH HIGH-THROUGHPUT STRESS TEST (BATCH 128 & 256)
# ---------------------------------------------------------------------------
print("\n[TEST 2/3] Executing 2,560 Batch Requests (Batch Sizes 128 & 256)...")

batch_128 = [(test_state, test_questions_laya)] * 128
batch_256 = [(test_state, test_questions_laya)] * 256

# Warmup
_ = agent_laya.batch_predict(batch_128)
torch.cuda.synchronize()

# Run 10x Batch 128 (1,280 items)
t_start_128 = time.perf_counter()
for _ in range(10):
    _ = agent_laya.batch_predict(batch_128)
    torch.cuda.synchronize()
t_end_128 = time.perf_counter() - t_start_128
throughput_128 = 1280.0 / t_end_128

# Run 5x Batch 256 (1,280 items)
t_start_256 = time.perf_counter()
for _ in range(5):
    _ = agent_laya.batch_predict(batch_256)
    torch.cuda.synchronize()
t_end_256 = time.perf_counter() - t_start_256
throughput_256 = 1280.0 / t_end_256

print(f"  - Batch 128 (1,280 items) : {t_end_128:.2f} sec | Latency: {t_end_128*100/128:.2f} ms/item | Throughput: {throughput_128:.1f} items/sec")
print(f"  - Batch 256 (1,280 items) : {t_end_256:.2f} sec | Latency: {t_end_256*100/256:.2f} ms/item | Throughput: {throughput_256:.1f} items/sec")

del agent_laya, fast_graph
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# TEST 3: 500 DEBERTA-V3-LARGE REAL-WORLD PACKET RUNS
# ---------------------------------------------------------------------------
print("\n[TEST 3/3] Executing 500 Real-World Requests on DeBERTa-v3-large (435M FP16)...")
model_d = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup
for _ in range(10):
    _ = model_d.decide(test_state, test_questions_deberta)
torch.cuda.synchronize()

deberta_latencies = []
t_start_3 = time.perf_counter()

for _ in range(500):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model_d.decide(test_state, test_questions_deberta)
    torch.cuda.synchronize()
    deberta_latencies.append((time.perf_counter() - t0) * 1000)

t_end_3 = time.perf_counter() - t_start_3
deberta_latencies.sort()

p50_3 = float(np.percentile(deberta_latencies, 50))
p95_3 = float(np.percentile(deberta_latencies, 95))
p99_3 = float(np.percentile(deberta_latencies, 99))
throughput_3 = 500.0 / t_end_3

print(f"  - Completed 500 Decisions in   : {t_end_3:.2f} seconds")
print(f"  - P50 Latency                  : {p50_3:6.2f} ms")
print(f"  - P95 Latency                  : {p95_3:6.2f} ms")
print(f"  - P99 Latency                  : {p99_3:6.2f} ms")
print(f"  - Peak DeBERTa-v3 Throughput   : {throughput_3:6.1f} decisions / sec")

del model_d
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# FINAL GRAND SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 105)
print("ULTRA-LONG GPU STRESS TEST GRAND SUMMARY")
print("=" * 105)
print(f"Total Requests Processed      : 8,060 Authorization Requests")
print(f"Laya CUDA Graph Single-Packet : {throughput_1:.1f} decisions/sec (P50 = {p50_1:.2f} ms | P99 = {p99_1:.2f} ms)")
print(f"Laya Batch 256 Max Throughput : {throughput_256:.1f} decisions/sec ({t_end_256*100/256:.2f} ms/item)")
print(f"DeBERTa-v3-large FP16 GPU     : {throughput_3:.1f} decisions/sec (P50 = {p50_3:.2f} ms | P99 = {p99_3:.2f} ms)")
print("=" * 105)
