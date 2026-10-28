"""End-to-end test: load model, run forward pass, benchmark."""

import sys
import time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 50)
print("laya-cuda test")
print("=" * 50)

# ── 1. Import check ───────────────────────────────
print("\n[1] Import...")
import laya_cuda as laya
print("    OK")

# ── 2. Device ─────────────────────────────────────
print("\n[2] Device check...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device}")
if device == "cuda":
    print(f"    GPU:    {torch.cuda.get_device_name(0)}")
    print(f"    VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── 3. Load model ─────────────────────────────────
print("\n[3] Loading convaiinnovations/laya ...")
print("    (downloads ~1.7 GB on first run)")
t0 = time.perf_counter()
agent = laya.load("convaiinnovations/laya", dtype="float16")
print(f"    Loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

# ── 4. Single predict ─────────────────────────────
print("\n[4] Single predict...")
state = "I was billed twice for the same subscription. Please refund the duplicate charge."
questions = {
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

t0 = time.perf_counter()
result = agent.predict(state, questions)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    Time: {elapsed:.1f} ms")
print(f"    Answers: {result['answers']}")
print("    Details:")
for k, v in result['details'].items():
    conf = v.get('confidence', '?')
    ans = v.get('answer', '?')
    print(f"      {k}: {ans!r}  (conf={conf:.3f})")

# ── 5. Benchmark ──────────────────────────────────
print("\n[5] Benchmark (50 runs)...")
stats = agent.benchmark(state, questions, n=50)
print(f"    P50:  {stats['p50_ms']:.2f} ms")
print(f"    P95:  {stats['p95_ms']:.2f} ms")
print(f"    Mean: {stats['mean_ms']:.2f} ms")

# ── 6. Batch predict ──────────────────────────────
print("\n[6] Batch predict (10 items)...")
items = [(state, {"dept": {"type": "choice", "instructions": "Route this.", "criteria": ["billing","tech","sales"]}}) for _ in range(10)]
t0 = time.perf_counter()
results = agent.batch_predict(items)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    10 items in {elapsed:.1f} ms  ({elapsed/10:.1f} ms/item)")
print(f"    Sample answer: {results[0]['answers']}")

# ── 7. Language routing ───────────────────────────
print("\n[7] Language routing...")
from laya_cuda.router import route
tests = [
    "I need a refund for my order.",
    "मुझे अपना ऑर्डर रिफंड चाहिए।",
    "Je voudrais un remboursement.",
    "我需要退款。",
]
for t in tests:
    rd = route(t)
    print(f"    '{t[:35]}...' -> {rd.reason}")

# ── 8. Ryzen CPU inference check ──────────────────
print("\n[8] Ryzen CPU native test...")
t0 = time.perf_counter()
cpu_agent = laya.load("convaiinnovations/laya", device="cpu", dtype="float32")
cpu_load_time = (time.perf_counter() - t0) * 1000
print(f"    CPU Model loaded in {cpu_load_time:.0f} ms")
t0 = time.perf_counter()
cpu_res = cpu_agent.predict(
    state,
    {"department": {"type": "choice", "instructions": "Who should handle this?", "criteria": ["billing", "tech", "sales"]}}
)
cpu_elapsed = (time.perf_counter() - t0) * 1000
print(f"    CPU Answer: {cpu_res['answers']}  ({cpu_elapsed:.1f} ms)")

print("\n" + "=" * 50)
print("ALL TESTS PASSED [OK]")
print("=" * 50)
