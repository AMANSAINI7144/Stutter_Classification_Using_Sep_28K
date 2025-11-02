import torch

if torch.cuda.is_available():
    print("✅ GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ GPU is NOT available.")
