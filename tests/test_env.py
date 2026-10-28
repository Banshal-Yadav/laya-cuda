"""Environment diagnostic check for PyTorch, CUDA, and SentencePiece dependencies."""

import torch

print("=" * 60)
print("ENVIRONMENT DIAGNOSTIC CHECK")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU Total VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

try:
    import sentencepiece
    print("SentencePiece:   AVAILABLE (v" + sentencepiece.__version__ + ")")
except ImportError:
    print("SentencePiece:   NOT INSTALLED")

try:
    import transformers
    print(f"Transformers:    AVAILABLE (v{transformers.__version__})")
except ImportError:
    print("Transformers:    NOT INSTALLED")

print("=" * 60)
