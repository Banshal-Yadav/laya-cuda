![Laya-CUDA Banner](assets/LAYA-CUDA_banner.png)

# Laya-CUDA

> **Sub-10ms typed decision intelligence on NVIDIA RTX GPUs and AMD CPUs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8%20%2F%2012.x-orange.svg)](https://pytorch.org)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_%2B_AMD_Ryzen-76b900.svg)]()

Port of Apple Silicon [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) to native PyTorch CUDA. Replaces Apple MLX with high-performance PyTorch kernels, pre-allocated static CUDA Graph replay, and TensorRT acceleration for real-time control loops and in-line decision pipelines.

---

## Why Typed Decisions?

Generative LLMs are slow, non-deterministic, and costly for millisecond control loops. Laya executes typed classifications (choices, continuous scores, and binary gates) directly on hidden states:

| Engine | Parameters | Device | P50 Latency | Throughput | Footprint | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Laya (ModernBERT)** | 149M | RTX 4050 (FP16) | **7.13 ms** | **~140 decisions/s** | ~350 MB | Real-time physics, game AI, routing |
| **Laya (Ryzen CPU)** | 149M | Ryzen 7 (INT8) | **330 ms** | **~3 decisions/s** | ~200 MB | GPU-free edge deployments |
| **DeBERTa-v3-large** | 435M | RTX 4050 (FP16) | **47.80 ms** | **~21 decisions/s** | ~1.67 GB | Financial wire auth, AML, fraud |
| *Generative 8B LLM* | 8B | RTX 4050 (4-bit) | *~850 ms* | *~1.2 requests/s* | ~5.5 GB | Unstructured chat generation |

- **Laya (ModernBERT-149M)** is the primary engine for high-frequency control loops (<10 ms). Used across all game demos (`web_snake/`, `web_pong/`, `snake_game.py`).
- **Laya Multilingual & Typed-Decisions** (`laya-multilingual`, `laya-typed-decisions`) provide 100+ language support and fine-tuned enterprise ticket routing.
- **DeBERTa-v3-Large (435M)** is the secondary gatekeeper for high-stakes risk evaluation where deeper semantic reasoning outweighs raw FPS (~48 ms).

---

## Quick Start

### 1. Installation

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install laya-cuda in editable mode
pip install -e .

# Verify environment
python tests/test_env.py
```

### 2. Fast CUDA Graph Engine (7.1 ms)

Zero host-launch overhead for continuous control loops:

```python
import laya_cuda as laya

agent = laya.load("convaiinnovations/laya", dtype="float16")
engine = laya.FastDecisionEngine(
    agent=agent,
    options=["UP", "DOWN", "LEFT", "RIGHT"],
    max_seq_len=64,
)

# Static graph execution: ~7.1 ms on RTX 4050
move, confidence = engine.decide("Snake at (5, 5). Food at (8, 5). Obstacle: UP.")
print(f"Action: {move} ({confidence * 100:.1f}%)")
```

### 3. Structured Multi-Question Prediction

```python
import laya_cuda as laya

agent = laya.load("convaiinnovations/laya", dtype="float16")
result = agent.predict(
    state="Customer is furious: I was double-charged $450 for my subscription.",
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should triage this issue?",
            "criteria": ["billing_refunds", "tech_support", "sales"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this require immediate escalation?",
        },
    },
)

print(result["answers"])
# {'department': 'billing_refunds', 'is_urgent': True}
```

---

## Interactive Demos

### 1. Web Snake Game (Port 8000)
Runs 120+ decisions/second using pre-allocated CUDA Graphs:

```bash
python web_snake/server.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 2. 1v1 Cyberpunk Pong Duel (Port 8001)
Play against Laya AI running at 60 FPS with dynamic tactical shot selection:

```bash
python web_pong/server.py
```
Open **`http://127.0.0.1:8001`** in your browser.

### 3. Terminal Snake Game
Lightweight standalone terminal Snake running at 100+ decisions/second:

```bash
python snake_game.py
```

---

## Benchmarks

### 1. Kernel Latency & Throughput (RTX 4050 Laptop)

```bash
python -m laya_cuda._bench --model convaiinnovations/laya --n 50
```

| Backend | Hardware | P50 Latency | Throughput |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA Graph** | NVIDIA RTX 4050 (6 GB) | **7.13 ms** | **~140 decisions/s** |
| **PyTorch CUDA Eager** | NVIDIA RTX 4050 (6 GB) | **31.27 ms** | **~32 decisions/s** |
| **CUDA Batch (10 items)** | NVIDIA RTX 4050 (6 GB) | **3.74 ms / item** | **~267 items/s** |
| **PyTorch CPU (Quantized)** | AMD Ryzen 7 (16 threads) | **330.58 ms** | **~3 decisions/s** |

### 2. Fraud & Gatekeeper Accuracy (DeBERTa-v3-large)

```bash
python benchmarks/fraud/benchmark_fraud_50.py
```

Evaluated across 50 real-world fraud vectors (AWS CloudTrail logs, mempool transactions, KYC documents, SMS phishing):
- **Binary Gatekeeper Accuracy (`noul`)**: 48 / 50 (96.0%)
- **Average Decision Latency**: 117.9 ms

---

## Repository Structure

```
laya/
├── laya_cuda/                      # Core Package: Native PyTorch/CUDA engine
│   ├── model.py                    # ModernBERT backbone + decision heads
│   ├── agent.py                    # Runtime agent & predict API
│   ├── fast_engine.py              # Static CUDA Graph engine (7.13 ms)
│   ├── fast_cpu_engine.py          # CPU-optimized Ryzen engine
│   ├── tokenizer.py                # High-speed ModernBERT tokenizer
│   ├── common.py                   # Sequence utilities & attention masks
│   ├── presets.py                  # Standard decision configurations
│   ├── router.py                   # Language & checkpoint routing
│   ├── lang.py                     # Script detection & language analysis
│   ├── email.py                    # Email parsing & classification
│   ├── export.py                   # ONNX & TensorRT builders
│   └── trt_agent.py                # TensorRT runtime
│
├── web_snake/                      # Real-time Web Snake (60-140 FPS AI)
├── web_pong/                       # Live 1v1 Cyberpunk Pong Duel
├── benchmarks/                     # Comprehensive benchmark suites
│   ├── fraud/                      # 50-vector fraud & gatekeeper benchmarks
│   ├── deberta/                    # Latency & safety evaluations
│   └── laya/                       # Kernel throughput benchmarks
├── tests/                          # Integration & diagnostic suite
├── assets/                         # Visual assets & banner
├── snake_game.py                   # Standalone terminal Snake game
├── pyproject.toml                  # Package configuration
├── LICENSE                         # Apache-2.0 License
└── README.md                       # Documentation
```

---

## Credits & Acknowledgments

- **[Convai Innovations](https://huggingface.co/convaiinnovations)** & **[Nandha Kishor M](https://github.com/NandhaKishorM/laya)**: Original creators of the Laya typed-decision architecture and pre-trained open-weight models (`convaiinnovations/laya`).
- **[mizorewww](https://github.com/mizorewww/laya-mlx)**: Author of `laya-mlx`, the Apple Silicon MLX port from which this PyTorch/CUDA runtime was derived.
- **[Kotoba Labs](https://huggingface.co/com-kotobalabs)**: Authors of `open-jev-deberta-v3-large`, used in the in-line risk and gatekeeper evaluation benchmarks.

---

## License

Apache-2.0. Derived from [laya-mlx](https://github.com/mizorewww/laya-mlx) by mizorewww, based on original work by Convai Innovations.
