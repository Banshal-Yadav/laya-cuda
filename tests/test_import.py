import laya_cuda as laya
import torch

print("laya_cuda imported OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
else:
    print("Running on CPU (Ryzen)")
