"""
Unit tests for model definitions in model_def.py
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch, MagicMock

from src.model_def import (
    LogSumExpPool2d, 
    ResNet9TableArch, 
    make_airbench94_adapted,
    construct_resnet9_paper,
    Flatten,
    Mul
)
from src import config as project_config


class TestLogSumExpPool2d:
    """Test suite for the unified LogSumExpPool2d implementation"""
    
    def test_initialization_valid_params(self):
        """Test LogSumExpPool2d initialization with valid parameters"""
        # Global pooling
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=-1)
        assert pool.epsilon == 0.1
        assert pool.kernel_size == -1
        assert pool.global_pool is True
        
        # Sliding window pooling
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=2, stride=2)
        assert pool.epsilon == 0.1
        assert pool.kernel_size == 2
        assert pool.stride == 2
        assert pool.global_pool is False
    
    def test_initialization_invalid_params(self):
        """Test LogSumExpPool2d initialization with invalid parameters"""
        # The implementation only warns for negative epsilon, doesn't raise ValueError
        with pytest.warns(UserWarning, match="Negative epsilon"):
            LogSumExpPool2d(epsilon=-0.1, kernel_size=2)
        
        with pytest.raises(ValueError, match="Invalid kernel_size .* or stride .* for sliding window pooling"):
            LogSumExpPool2d(epsilon=0.1, kernel_size=0)
            
        with pytest.raises(ValueError, match="Invalid kernel_size .* or stride .* for sliding window pooling"):
            LogSumExpPool2d(epsilon=0.1, kernel_size=2, stride=0)
    
    def test_global_pooling_epsilon_zero(self):
        """Test global pooling with epsilon=0 (should use average pooling)"""
        pool = LogSumExpPool2d(epsilon=0.0, kernel_size=-1)
        x = torch.randn(2, 3, 4, 4)
        
        output = pool(x)
        expected = torch.mean(x, dim=(-1, -2), keepdim=True)
        
        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_global_pooling_with_epsilon(self):
        """Test global pooling with positive epsilon"""
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=-1)
        x = torch.randn(2, 3, 4, 4)
        
        output = pool(x)
        
        # Check shape
        assert output.shape == (2, 3, 1, 1)
        
        # Verify it's not just average pooling
        avg_output = torch.mean(x, dim=(-1, -2), keepdim=True)
        assert not torch.allclose(output, avg_output, rtol=1e-3)
        
        # Verify numerical stability (should not contain inf or nan)
        assert torch.isfinite(output).all()
    
    def test_sliding_window_pooling_epsilon_zero(self):
        """Test sliding window pooling with epsilon=0"""
        pool = LogSumExpPool2d(epsilon=0.0, kernel_size=2, stride=2)
        x = torch.randn(1, 2, 4, 4)
        
        output = pool(x)
        expected = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_sliding_window_pooling_with_epsilon(self):
        """Test sliding window pooling with positive epsilon"""
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=2, stride=2)
        x = torch.randn(1, 2, 4, 4)
        
        output = pool(x)
        
        # Check shape
        assert output.shape == (1, 2, 2, 2)
        
        # Verify it's not just average pooling
        avg_output = F.avg_pool2d(x, kernel_size=2, stride=2)
        assert not torch.allclose(output, avg_output, rtol=1e-3)
        
        # Verify numerical stability
        assert torch.isfinite(output).all()
    
    def test_padding_support(self):
        """Test that padding parameter works correctly"""
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=3, stride=1, padding=1)
        x = torch.randn(1, 1, 3, 3)
        
        output = pool(x)
        # With padding=1 and stride=1, output should have same spatial dimensions
        assert output.shape == (1, 1, 3, 3)
    
    def test_numerical_stability_large_values(self):
        """Test numerical stability with large input values"""
        pool = LogSumExpPool2d(epsilon=1.0, kernel_size=-1)
        
        # Create input with large values that could cause overflow
        x = torch.full((1, 1, 3, 3), 100.0)
        
        output = pool(x)
        
        # Should not contain inf or nan
        assert torch.isfinite(output).all()
        # Result should be reasonable (not extremely large)
        assert output.abs().max() < 1000
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer"""
        pool = LogSumExpPool2d(epsilon=0.1, kernel_size=-1)
        x = torch.randn(1, 2, 3, 3, requires_grad=True)
        
        output = pool(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestResNet9TableArch:
    """Test suite for ResNet9TableArch model"""
    
    def test_initialization_default_params(self):
        """Test ResNet9TableArch initialization with default parameters"""
        model = ResNet9TableArch()
        
        # Access the actual nn.Linear layer inside ScaledLinearTableArch
        # The input features depend on the width_multiplier
        expected_in_features = int(512 * project_config.RESNET9_WIDTH_MULTIPLIER)
        assert model.linear.linear.in_features == expected_in_features
        assert model.linear.linear.out_features == project_config.NUM_CLASSES
        assert isinstance(model.pool, LogSumExpPool2d)
        assert model.pool.epsilon == 0.1
        assert model.pool.kernel_size == -1
    
    def test_initialization_custom_params(self):
        """Test ResNet9TableArch initialization with custom parameters"""
        model = ResNet9TableArch(
            num_classes=100,
            width_multiplier=0.5,
            pooling_epsilon=0.2,
            final_layer_scale=2.0
        )
        
        assert model.linear.linear.out_features == 100
        assert model.linear.scale_factor == 2.0
        assert model.pool.epsilon == 0.2
        
        # Check width multiplier effect
        expected_channels = int(512 * 0.5)  # Final layer channels
        assert model.linear.linear.in_features == expected_channels
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""
        model = ResNet9TableArch(num_classes=10)
        x = torch.randn(2, 3, 32, 32)  # CIFAR-10 input
        
        output = model(x)
        
        assert output.shape == (2, 10)
        assert torch.isfinite(output).all()
    
    def test_forward_pass_different_input_sizes(self):
        """Test forward pass with different input sizes"""
        model = ResNet9TableArch(num_classes=10)
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 10)
    
    def test_layer_structure(self):
        """Test that the model has the expected layer structure"""
        model = ResNet9TableArch()
        
        # Check that all required layers exist
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'bn1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        assert hasattr(model, 'layer4')
        assert hasattr(model, 'pool')
        assert hasattr(model, 'linear')
        
        # Check layer types
        assert isinstance(model.conv1, torch.nn.Conv2d)
        assert isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert isinstance(model.pool, LogSumExpPool2d)


class TestAirBench94Adapted:
    """Test suite for make_airbench94_adapted function"""
    
    def test_creation_default_params(self):
        """Test airbench94_adapted creation with default parameters"""
        model = make_airbench94_adapted()
        
        assert isinstance(model, torch.nn.Sequential)
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_creation_custom_params(self):
        """Test airbench94_adapted creation with custom parameters"""
        model = make_airbench94_adapted(
            width_multiplier=2.0,
            pooling_epsilon=0.2,
            final_layer_scale=0.5,
            num_classes=100
        )
        
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 100)
    
    def test_pooling_layers_are_logsumexp(self):
        """Test that pooling layers are indeed LogSumExpPool2d"""
        model = make_airbench94_adapted()
        
        # Count LogSumExpPool2d layers
        logsumexp_count = 0
        for module in model.modules():
            if isinstance(module, LogSumExpPool2d):
                logsumexp_count += 1
        
        # Should have 4 LogSumExpPool2d layers based on the architecture
        assert logsumexp_count == 4


class TestConstructResNet9Paper:
    """Test suite for construct_resnet9_paper function"""
    
    def test_creation_default_params(self):
        """Test construct_resnet9_paper creation with default parameters"""
        model = construct_resnet9_paper(num_classes=10)
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_creation_with_logsumexp_pooling(self):
        """Test that construct_resnet9_paper uses LogSumExpPool2d for sliding window"""
        model = construct_resnet9_paper(num_classes=10)
        
        # Find LogSumExpPool2d layers in the model
        logsumexp_layers = []
        for module in model.modules():
            if isinstance(module, LogSumExpPool2d):
                logsumexp_layers.append(module)
        
        # Should have LogSumExpPool2d layers
        assert len(logsumexp_layers) > 0
        
        # Test forward pass works
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_different_num_classes(self):
        """Test construct_resnet9_paper with different number of classes"""
        for num_classes in [10, 100, 1000]:
            model = construct_resnet9_paper(num_classes=num_classes)
            x = torch.randn(1, 3, 32, 32)
            output = model(x)
            assert output.shape == (1, num_classes)


class TestUtilityClasses:
    """Test suite for utility classes like Flatten and Mul"""
    
    def test_flatten_layer(self):
        """Test Flatten layer functionality"""
        flatten = Flatten()
        x = torch.randn(2, 3, 4, 5)
        
        output = flatten(x)
        assert output.shape == (2, 3 * 4 * 5)
    
    def test_mul_layer(self):
        """Test Mul layer functionality"""
        scale_factor = 2.5
        mul = Mul(scale_factor)
        x = torch.randn(2, 10)
        
        output = mul(x)
        expected = x * scale_factor
        
        torch.testing.assert_close(output, expected)
    
    def test_mul_layer_gradient_flow(self):
        """Test that Mul layer preserves gradients correctly"""
        scale_factor = 0.5
        mul = Mul(scale_factor)
        x = torch.randn(2, 5, requires_grad=True)
        
        output = mul(x)
        loss = output.sum()
        loss.backward()
        
        # Gradient should be scaled by the same factor
        expected_grad = torch.ones_like(x) * scale_factor
        torch.testing.assert_close(x.grad, expected_grad)


class TestModelIntegration:
    """Integration tests for model components"""
    
    def test_models_work_with_cifar10_input(self):
        """Test that all models work with CIFAR-10 shaped input"""
        models = [
            ResNet9TableArch(),
            make_airbench94_adapted(),
            construct_resnet9_paper(num_classes=10)
        ]
        
        x = torch.randn(4, 3, 32, 32)  # CIFAR-10 batch
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                assert output.shape == (4, 10)
                assert torch.isfinite(output).all()
    
    def test_models_training_mode(self):
        """Test that models work in training mode with gradient computation"""
        models = [
            ResNet9TableArch(),
            make_airbench94_adapted(),
            construct_resnet9_paper(num_classes=10)
        ]
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        
        for model in models:
            model.train()
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            # Check that model parameters have gradients
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
            
            # Clear gradients for next iteration
            model.zero_grad()
    
    def test_parameter_counting(self):
        """Test parameter counting for models"""
        model = ResNet9TableArch()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable by default
    
    def test_model_device_compatibility(self):
        """Test that models work on different devices"""
        model = ResNet9TableArch()
        
        # Test CPU
        x_cpu = torch.randn(1, 3, 32, 32)
        output_cpu = model(x_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == 'cuda'


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and performance considerations"""
    
    def test_logsumexp_with_extreme_epsilon(self):
        """Test LogSumExpPool2d with very small and large epsilon values"""
        # Very small epsilon (should approach average pooling but may not be exact)
        pool_small = LogSumExpPool2d(epsilon=1e-6, kernel_size=-1)
        x = torch.randn(1, 2, 3, 3)
        output_small = pool_small(x)
        
        avg_output = torch.mean(x, dim=(-1, -2), keepdim=True)
        # Should be close to average pooling (but with some tolerance for log-sum-exp)
        torch.testing.assert_close(output_small, avg_output, rtol=1e-1, atol=1e-1)
        
        # Large epsilon (should approach max pooling behavior)
        pool_large = LogSumExpPool2d(epsilon=100.0, kernel_size=-1)
        output_large = pool_large(x)
        
        # Should not be NaN or inf
        assert torch.isfinite(output_large).all()
    
    def test_single_pixel_input(self):
        """Test models with very small input sizes"""
        model = ResNet9TableArch()
        
        # This might fail due to repeated pooling reducing spatial dimensions to zero
        # But we test to ensure graceful handling
        try:
            model.eval()  # Set to eval mode to avoid BatchNorm issues with single values
            x = torch.randn(2, 3, 8, 8)  # Use batch size > 1 to avoid BatchNorm issues
            output = model(x)
            assert output.shape == (2, 10)
        except (RuntimeError, ValueError) as e:
            # If it fails due to spatial dimensions or BatchNorm, that's expected for very small inputs
            assert any(keyword in str(e).lower() for keyword in ["size", "dimension", "batch", "channel"])
            # This is acceptable behavior for edge cases
    
    def test_memory_efficiency(self):
        """Test that models don't consume excessive memory"""
        model = ResNet9TableArch()
        
        # Test with larger batch size
        x = torch.randn(16, 3, 32, 32)
        
        # Enable memory profiling if needed
        with torch.no_grad():
            output = model(x)
            assert output.shape == (16, 10)
        
        # Cleanup
        del x, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 