"""Run DeBERTa-v3-large OpenJev benchmark on RTX 4050."""

import sys
import time
from pathlib import Path
import torch

print("=" * 65, flush=True)
print("RUNNING DEBERTA-V3-LARGE (open-jev-deberta-v3-large)", flush=True)
print("=" * 65, flush=True)

# Find local snapshot directory
cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
print(f"Loading from local snapshot: {snapshot_dir}", flush=True)

# Import OpenJev
sys.path.insert(0, str(snapshot_dir))
from typed_decisions.open_jev import OpenJev

print("Instantiating OpenJev on CUDA...", flush=True)
t0 = time.perf_counter()
model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")
load_time = (time.perf_counter() - t0) * 1000
print(f"Loaded DeBERTa-v3-large into GPU memory in {load_time:.0f} ms!", flush=True)

vram_alloc = torch.cuda.memory_allocated() / 1024**2
vram_res = torch.cuda.memory_reserved() / 1024**2
print(f"GPU VRAM: {vram_alloc:.1f} MB allocated ({vram_res:.1f} MB reserved)", flush=True)

# Test Real Questions
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

print("\nRunning single decision pass...", flush=True)
t0 = time.perf_counter()
results = model.decide(state, questions)
torch.cuda.synchronize()
dt_ms = (time.perf_counter() - t0) * 1000
print(f"Single pass completed in: {dt_ms:.1f} ms", flush=True)

print("\nDecision Output:")
for q, res in zip(questions, results):
    print(f"  [{q['type'].upper()}]: {res}")

# Benchmark 20 iterations
print("\nBenchmarking 25 iterations on RTX 4050...", flush=True)
times = []
for _ in range(25):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.decide(state, questions)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
p50 = times[len(times) // 2]
p95 = times[int(len(times) * 0.95)]
print(f"DeBERTa-v3-large Latency on RTX 4050:")
print(f"  P50 Latency:   {p50:.2f} ms ({1000/p50:.1f} decisions/sec)")
print(f"  P95 Latency:   {p95:.2f} ms")
print("=" * 65, flush=True)
