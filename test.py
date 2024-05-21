import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU is ready to be used!")
else:
    print("CUDA is not available. Check your installation.")