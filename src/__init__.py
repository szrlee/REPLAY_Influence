"""
REPLAY Influence Analysis Package
=================================

This package implements state-of-the-art influence function analysis for deep learning models,
featuring both MAGIC (influence computation) and LDS (validation) methodologies.

Key Components:
- magic_analyzer: MAGIC influence analysis implementation
- lds_validator: LDS validation methodology
- utils: Deterministic utilities and seed management
- config: Configuration management
- model_def: ResNet9 model architecture
- data_handling: CIFAR-10 data loading and preprocessing
- visualization: Plotting and visualization utilities

Python >=3.8 Compatible
"""

__version__ = "1.0.0"
__author__ = "REPLAY Influence Analysis Team"
__email__ = "contact@replay-influence.org"

# Core module imports for easier access
from . import config
from . import utils
from . import magic_analyzer
from . import lds_validator
from . import model_def
from . import data_handling
from . import visualization

# Expose key classes and functions for direct import
from .magic_analyzer import MagicAnalyzer
from .lds_validator import run_lds_validation
from .model_def import construct_rn9
from .utils import (
    set_global_deterministic_state,
    derive_component_seed,
    create_deterministic_dataloader,
    create_deterministic_model,
    setup_logging
)
from .data_handling import get_cifar10_dataloader, CustomDataset
from .visualization import plot_influence_images

# Define what gets imported with "from src import *"
__all__ = [
    # Main classes
    "MagicAnalyzer",
    
    # Key functions
    "run_lds_validation",
    "construct_rn9",
    "set_global_deterministic_state",
    "derive_component_seed",
    "create_deterministic_dataloader",
    "create_deterministic_model",
    "setup_logging",
    "get_cifar10_dataloader",
    "plot_influence_images",
    
    # Classes
    "CustomDataset",
    
    # Modules
    "config",
    "utils",
    "magic_analyzer",
    "lds_validator",
    "model_def", 
    "data_handling",
    "visualization",
]

# Package metadata
def get_version() -> str:
    """Return the package version."""
    return __version__

def get_package_info() -> dict:
    """Return comprehensive package information."""
    return {
        "name": "replay-influence-analysis",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "State-of-the-art influence function analysis for deep learning",
        "python_requires": ">=3.8",
        "dependencies": [
            "torch>=2.2.2",
            "torchvision>=0.17.2", 
            "numpy>=1.26.4",
            "matplotlib>=3.10.3",
            "scipy>=1.15.3",
            "tqdm>=4.67.1"
        ]
    }

# Import-time validation
try:
    import torch
    import numpy as np
    import matplotlib
    
    # Verify minimum versions
    import sys
    if sys.version_info < (3, 8):
        raise ImportError(f"Python 3.8+ required, found {sys.version_info}")
        
except ImportError as e:
    import warnings
    warnings.warn(f"Some dependencies may be missing: {e}", ImportWarning)

# Optional: Print initialization message in debug mode
import os
if os.environ.get('REPLAY_DEBUG'):
    print(f"Initialized REPLAY Influence Analysis v{__version__}") 