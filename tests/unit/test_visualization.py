#!/usr/bin/env python3
"""
Visualization Tests for REPLAY Influence Analysis
================================================

Tests visualization functionality, plotting capabilities, and error handling.
Ensures plots are generated correctly and gracefully handle edge cases.
"""

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.visualization import plot_influence_images


class TestPlotInfluenceImages:
    """Test suite for plot_influence_images function"""
    
    def create_test_dataset(self, size=10):
        """Create a minimal test dataset"""
        class TestDataset:
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Return valid CIFAR-10 style data
                return torch.randn(3, 32, 32), idx % 10, idx
        
        return TestDataset(size)
    
    @pytest.mark.unit
    def test_basic_influence_plotting(self):
        """Test basic influence plotting functionality"""
        dataset = self.create_test_dataset(10)
        scores = np.array([1.0, -0.5, 0.8, -1.2, 0.3, -0.1, 0.9, -0.8, 0.2, -0.3])
        
        target_info = {
            'image': torch.randn(3, 32, 32),
            'label': 5,
            'id_str': 'Test Image'
        }
        
        train_info = {
            'dataset': dataset,
            'name': 'Test Dataset'
        }
        
        # Should not raise any exceptions
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_plot.png"
            
            plot_influence_images(
                scores_flat=scores,
                target_image_info=target_info,
                train_dataset_info=train_info,
                num_to_show=3,
                save_path=save_path
            )
            
            # Verify plot was saved
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    @pytest.mark.unit
    def test_input_validation(self):
        """Test input validation for plot_influence_images"""
        dataset = self.create_test_dataset(5)
        
        # Test invalid scores_flat
        with pytest.raises(ValueError, match="scores_flat must be a numpy array"):
            plot_influence_images(
                scores_flat=[1, 2, 3],  # List instead of numpy array
                target_image_info={'image': torch.randn(3, 32, 32), 'label': 0, 'id_str': 'test'},
                train_dataset_info={'dataset': dataset, 'name': 'test'},
                num_to_show=2
            )
        
        # Test wrong shape for scores_flat
        with pytest.raises(ValueError, match="scores_flat must be a 1D array"):
            plot_influence_images(
                scores_flat=np.array([[1, 2], [3, 4]]),  # 2D array
                target_image_info={'image': torch.randn(3, 32, 32), 'label': 0, 'id_str': 'test'},
                train_dataset_info={'dataset': dataset, 'name': 'test'},
                num_to_show=2
            )
        
        # Test negative num_to_show
        with pytest.raises(ValueError, match="num_to_show must be positive"):
            plot_influence_images(
                scores_flat=np.array([1, 2, 3, 4, 5]),
                target_image_info={'image': torch.randn(3, 32, 32), 'label': 0, 'id_str': 'test'},
                train_dataset_info={'dataset': dataset, 'name': 'test'},
                num_to_show=-1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 