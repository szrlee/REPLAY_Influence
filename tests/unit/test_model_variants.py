#!/usr/bin/env python3
"""
Unit tests for ResNet-9 model variants.

Tests the different ResNet-9 model implementations to ensure they work correctly
and maintain backward compatibility.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model_def import construct_rn9, construct_resnet9_paper
from src.config import NUM_CLASSES


class TestResNet9Variants:
    """Test suite for ResNet-9 model variants."""
    
    @pytest.mark.unit
    def test_original_model_construction(self):
        """Test that the original ResNet-9 model constructs correctly."""
        model = construct_rn9(num_classes=NUM_CLASSES)
        assert model is not None
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 32, 32)
        output = model(input_tensor)
        assert output.shape == (2, NUM_CLASSES)
        
        # Check parameter count (approximately 2.3M)
        param_count = sum(p.numel() for p in model.parameters())
        assert 2_000_000 < param_count < 3_000_000, f"Unexpected parameter count: {param_count}"
    
    @pytest.mark.unit
    def test_paper_model_construction(self):
        """Test that the paper-specific ResNet-9 model constructs correctly."""
        model = construct_resnet9_paper(num_classes=NUM_CLASSES)
        assert model is not None
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 32, 32)
        output = model(input_tensor)
        assert output.shape == (2, NUM_CLASSES)
        
        # Check parameter count (approximately 14.2M due to width multiplier)
        param_count = sum(p.numel() for p in model.parameters())
        assert 10_000_000 < param_count < 20_000_000, f"Unexpected parameter count: {param_count}"
    
    @pytest.mark.unit
    def test_model_differences(self):
        """Test that the two model variants have different architectures."""
        original_model = construct_rn9(num_classes=NUM_CLASSES)
        paper_model = construct_resnet9_paper(num_classes=NUM_CLASSES)
        
        # Different parameter counts
        original_params = sum(p.numel() for p in original_model.parameters())
        paper_params = sum(p.numel() for p in paper_model.parameters())
        assert original_params != paper_params
        assert paper_params > original_params  # Paper model should be larger
        
        # Different architectures (check string representation)
        original_str = str(original_model)
        paper_str = str(paper_model)
        
        # Original uses MaxPool2d, paper uses LogSumExpPool2d
        assert "MaxPool2d" in original_str
        assert "LogSumExpPool2d" in paper_str
        assert "LogSumExpPool2d" not in original_str
        assert "MaxPool2d" not in paper_str
    
    @pytest.mark.unit
    def test_backward_compatibility(self):
        """Test that the original model maintains backward compatibility."""
        # Test with different num_classes
        for num_classes in [10, 100]:
            model = construct_rn9(num_classes=num_classes)
            input_tensor = torch.randn(1, 3, 32, 32)
            output = model(input_tensor)
            assert output.shape == (1, num_classes)
    
    @pytest.mark.unit
    def test_deterministic_behavior(self):
        """Test that models behave deterministically with same seed."""
        torch.manual_seed(42)
        model1 = construct_rn9(num_classes=NUM_CLASSES)
        
        torch.manual_seed(42)
        model2 = construct_rn9(num_classes=NUM_CLASSES)
        
        # Models should have identical parameters when created with same seed
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Models should be identical with same seed"
        
        # Same test for paper model
        torch.manual_seed(42)
        paper_model1 = construct_resnet9_paper(num_classes=NUM_CLASSES)
        
        torch.manual_seed(42)
        paper_model2 = construct_resnet9_paper(num_classes=NUM_CLASSES)
        
        for p1, p2 in zip(paper_model1.parameters(), paper_model2.parameters()):
            assert torch.equal(p1, p2), "Paper models should be identical with same seed"

 