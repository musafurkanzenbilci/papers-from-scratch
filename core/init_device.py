import torch

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("Using Apple MPS (GPU) backend.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU) backend.")
else:
    print("Using CPU backend.")