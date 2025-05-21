import random
import numpy as np
import torch

def set_seeds(seed):
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN convolutions
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 