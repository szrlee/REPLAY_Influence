import random
import numpy as np
import torch

# Assuming config might be needed here in the future for other utils, but not for set_seeds
# from . import config 

def set_seeds(seed):
    """Sets random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Ensure all GPUs are seeded if using multi-GPU
    
    # These settings help ensure reproducibility for CUDA operations
    # However, they can impact performance. Use if reproducibility is critical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 