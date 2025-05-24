import random
import numpy as np
import torch
import logging
import sys
from typing import Optional

# Assuming config might be needed here in the future for other utils, but not for set_seeds
# from . import config 

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to. If None, logs to console only.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('influence_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_seeds(seed: int) -> None:
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