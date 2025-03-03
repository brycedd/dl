import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")