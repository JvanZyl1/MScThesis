# Check pytorch installation with GPU support
import torch

# Check if GPU is available
print("Is GPU available:", torch.cuda.is_available())
