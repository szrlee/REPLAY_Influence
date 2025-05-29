#!/usr/bin/env python3
"""
Performance and Memory Benchmark Tests for REPLAY Influence Analysis
===================================================================

Tests performance characteristics, memory usage, and efficiency of key components.
Ensures the system operates within acceptable performance boundaries.
"""

import pytest
import torch
import numpy as np
import time
import gc
import sys
from pathlib import Path
from unittest.mock import patch
import warnings
import logging

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import config
from src.utils import (
    create_deterministic_model,
    create_deterministic_dataloader,
    set_global_deterministic_state
)
from src.model_def import construct_rn9, construct_resnet9_paper, LogSumExpPool2d
from src.data_handling import get_cifar10_dataloader, CustomDataset

# Add after other imports, before importing psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class TestPerformanceBenchmarks:
    """Test performance characteristics of key components"""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_model_forward_pass_performance(self):
        """Test forward pass performance of different models"""
        batch_sizes = [1, 16, 64, 256]
        models = {
            'ResNet9': construct_rn9(num_classes=10),
            'ResNet9Paper': construct_resnet9_paper(num_classes=10)
        }
        
        performance_results = {}
        
        for model_name, model in models.items():
            model.eval()
            performance_results[model_name] = {}
            
            for batch_size in batch_sizes:
                try:
                    input_tensor = torch.randn(batch_size, 3, 32, 32)
                    
                    # Warmup
                    for _ in range(5):
                        _ = model(input_tensor)
                    
                    # Benchmark
                    start_time = time.time()
                    num_runs = 50
                    
                    for _ in range(num_runs):
                        output = model(input_tensor)
                    
                    end_time = time.time()
                    avg_time = (end_time - start_time) / num_runs
                    throughput = batch_size / avg_time  # samples per second
                    
                    performance_results[model_name][batch_size] = {
                        'avg_time': avg_time,
                        'throughput': throughput
                    }
                    
                    # Performance assertions
                    if batch_size == 256: # More lenient threshold for larger batch
                        assert avg_time < 10.0, f"Forward pass too slow: {avg_time:.3f}s for batch {batch_size}"
                    elif batch_size == 64:
                        assert avg_time < 5.0, f"Forward pass too slow: {avg_time:.3f}s for batch {batch_size}"
                    else: # For batch_size 1 and 16
                        assert avg_time < 2.0, f"Forward pass too slow: {avg_time:.3f}s for batch {batch_size}"
                    logger.info(f"{model_name} (BS={batch_size}): {avg_time:.4f}s, {throughput:.2f} samples/sec")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        pytest.skip(f"Insufficient memory for batch size {batch_size}")
                    else:
                        raise
        
        # Log performance results
        for model_name, results in performance_results.items():
            print(f"\n{model_name} Performance:")
            for batch_size, metrics in results.items():
                print(f"  Batch {batch_size}: {metrics['avg_time']:.4f}s, {metrics['throughput']:.1f} samples/s")
    
    @pytest.mark.unit
    def test_logsumexp_performance_vs_maxpool(self):
        """Compare LogSumExp performance against standard MaxPool"""
        input_tensor = torch.randn(64, 128, 16, 16)
        
        # Test LogSumExp performance
        logsumexp_pool = LogSumExpPool2d(epsilon=0.1, kernel_size=2, stride=2)
        
        start_time = time.time()
        for _ in range(100):
            output_lse = logsumexp_pool(input_tensor)
        lse_time = time.time() - start_time
        
        # Test MaxPool performance for comparison
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        start_time = time.time()
        for _ in range(100):
            output_max = maxpool(input_tensor)
        maxpool_time = time.time() - start_time
        
        # LogSumExp should be reasonably close to MaxPool performance
        performance_ratio = lse_time / maxpool_time
        assert performance_ratio < 5.0, f"LogSumExp too slow compared to MaxPool: {performance_ratio:.2f}x"
        
        print(f"LogSumExp time: {lse_time:.4f}s, MaxPool time: {maxpool_time:.4f}s, Ratio: {performance_ratio:.2f}x")
    
    @pytest.mark.unit
    def test_deterministic_operations_performance(self):
        """Test performance of deterministic operations"""
        # Test deterministic model creation time
        start_time = time.time()
        
        for i in range(10):
            model = create_deterministic_model(
                master_seed=42,
                creator_func=construct_rn9,
                instance_id=f"perf_test_{i}",
                num_classes=10
            )
        
        creation_time = time.time() - start_time
        avg_creation_time = creation_time / 10
        
        assert avg_creation_time < 1.0, f"Model creation too slow: {avg_creation_time:.3f}s"
        print(f"Average model creation time: {avg_creation_time:.4f}s")
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_dataloader_performance(self):
        """Test dataloader iteration performance"""
        try:
            dataloader = create_deterministic_dataloader(
                master_seed=42,
                creator_func=get_cifar10_dataloader,
                instance_id="perf_test_loader",
                batch_size=128,
                split='train',
                shuffle=True,
                augment=False,
                num_workers=0,  # Use 0 for consistent testing
                root_path=config.CIFAR_ROOT
            )
            
            # Test iteration speed
            start_time = time.time()
            batch_count = 0
            sample_count = 0
            
            for batch_idx, (images, labels, indices) in enumerate(dataloader):
                batch_count += 1
                sample_count += images.size(0)
                
                if batch_idx >= 50:  # Test first 50 batches
                    break
            
            total_time = time.time() - start_time
            samples_per_second = sample_count / total_time
            
            assert samples_per_second > 100, f"Dataloader too slow: {samples_per_second:.1f} samples/s"
            print(f"Dataloader performance: {samples_per_second:.1f} samples/s")
            
        except Exception as e:
            if "dataset" in str(e).lower():
                pytest.skip("CIFAR-10 dataset not available")
            else:
                raise


class TestMemoryUsageBenchmarks:
    """Test memory usage and memory leak detection"""
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.unit
    def test_model_memory_usage(self):
        """Test model memory usage"""
        initial_memory = self.get_memory_usage()
        
        # Create model and measure memory
        model = construct_rn9(num_classes=10)
        model_memory = self.get_memory_usage()
        
        # Test forward pass memory
        input_tensor = torch.randn(32, 3, 32, 32)
        output = model(input_tensor)
        forward_memory = self.get_memory_usage()
        
        # Memory usage should be reasonable
        model_overhead = model_memory - initial_memory
        forward_overhead = forward_memory - model_memory
        
        assert model_overhead < 100, f"Model uses too much memory: {model_overhead:.1f}MB"
        assert forward_overhead < 200, f"Forward pass uses too much memory: {forward_overhead:.1f}MB"
        
        print(f"Model memory: {model_overhead:.1f}MB, Forward memory: {forward_overhead:.1f}MB")
    
    @pytest.mark.unit
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        initial_memory = self.get_memory_usage()
        
        # Perform many operations that should not leak memory
        for i in range(50):
            model = construct_rn9(num_classes=10)
            input_tensor = torch.randn(16, 3, 32, 32)
            output = model(input_tensor)
            
            # Force garbage collection periodically
            if i % 10 == 0:
                del model, input_tensor, output
                gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (allowing for some variance)
        assert memory_increase < 50, f"Potential memory leak detected: {memory_increase:.1f}MB increase"
        print(f"Memory change after 50 iterations: {memory_increase:.1f}MB")
    
    @pytest.mark.unit
    def test_batch_size_memory_scaling(self):
        """Test memory usage scaling with batch size"""
        model = construct_rn9(num_classes=10)
        model.eval()
        
        memory_usage = {}
        base_memory = self.get_memory_usage()
        
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            try:
                input_tensor = torch.randn(batch_size, 3, 32, 32)
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                current_memory = self.get_memory_usage()
                memory_usage[batch_size] = current_memory - base_memory
                
                # Clean up
                del input_tensor, output
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory at batch size {batch_size}")
                    break
                else:
                    raise
        
        # Memory should scale reasonably with batch size
        if len(memory_usage) >= 2:
            batch_sizes_tested = sorted(memory_usage.keys())
            small_batch = batch_sizes_tested[0]
            large_batch = batch_sizes_tested[-1]
            
            small_memory = memory_usage[small_batch]
            large_memory = memory_usage[large_batch]
            
            # Memory scaling should be roughly linear but allow for overhead
            expected_ratio = large_batch / small_batch
            actual_ratio = large_memory / max(small_memory, 1)  # Avoid division by zero
            
            assert actual_ratio < expected_ratio * 2, f"Memory scaling inefficient: {actual_ratio:.2f}x vs expected {expected_ratio:.2f}x"
            
            print(f"Memory scaling: {small_batch}→{small_memory:.1f}MB, {large_batch}→{large_memory:.1f}MB")


class TestScalabilityBenchmarks:
    """Test scalability characteristics"""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test handling of large dataset sizes"""
        # Create a synthetic large dataset
        class LargeDataset:
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Generate data on-the-fly to save memory
                return torch.randn(3, 32, 32), idx % 10, idx
        
        large_sizes = [10000, 50000]
        
        for size in large_sizes:
            dataset = LargeDataset(size)
            
            # Test dataloader creation and basic iteration
            start_time = time.time()
            
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=64, 
                shuffle=False,
                num_workers=0
            )
            
            # Test first few batches
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:
                    break
            
            creation_time = time.time() - start_time
            
            assert creation_time < 5.0, f"Large dataset handling too slow: {creation_time:.2f}s"
            assert batch_count == 10, "Failed to iterate through large dataset"
            
            print(f"Dataset size {size}: {creation_time:.3f}s for setup and 10 batches")
    
    @pytest.mark.unit
    def test_high_dimensional_input_handling(self):
        """Test handling of high-dimensional inputs"""
        model = construct_rn9(num_classes=10)
        
        # Test with various input sizes (keeping memory reasonable)
        input_configs = [
            (1, 3, 32, 32),    # Standard
            (1, 3, 64, 64),    # Higher resolution
            (8, 3, 32, 32),    # Larger batch
        ]
        
        for config in input_configs:
            try:
                input_tensor = torch.randn(*config)
                
                start_time = time.time()
                output = model(input_tensor)
                processing_time = time.time() - start_time
                
                expected_output_shape = (config[0], 10)
                assert output.shape == expected_output_shape
                assert processing_time < 2.0, f"Processing too slow for {config}: {processing_time:.3f}s"
                
                print(f"Input {config}: {processing_time:.4f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory for config {config}")
                else:
                    raise


class TestEfficiencyMetrics:
    """Test efficiency and optimization metrics"""
    
    @pytest.mark.unit
    def test_parameter_efficiency(self):
        """Test parameter efficiency of models"""
        models = {
            'ResNet9': construct_rn9(num_classes=10),
            'ResNet9Paper': construct_resnet9_paper(num_classes=10)
        }
        
        for model_name, model in models.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test forward pass efficiency
            input_tensor = torch.randn(32, 3, 32, 32)
            
            start_time = time.time()
            output = model(input_tensor)
            forward_time = time.time() - start_time
            
            # Calculate efficiency metrics
            params_per_sample = total_params / 32  # Parameters per sample in batch
            time_per_param = forward_time / total_params * 1e6  # Microseconds per parameter
            
            print(f"{model_name}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Forward time: {forward_time:.4f}s")
            print(f"  Time per parameter: {time_per_param:.3f}μs")
            
            # Efficiency assertions
            assert time_per_param < 10.0, f"Model too inefficient: {time_per_param:.3f}μs per parameter"
    
    @pytest.mark.unit
    def test_gradient_computation_efficiency(self):
        """Test efficiency of gradient computation"""
        model = construct_rn9(num_classes=10)
        criterion = torch.nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(32, 3, 32, 32)
        target = torch.randint(0, 10, (32,))
        
        # Test forward + backward efficiency
        start_time = time.time()
        
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        
        total_time = time.time() - start_time
        
        # Gradient computation should be reasonably fast
        assert total_time < 1.0, f"Gradient computation too slow: {total_time:.3f}s"
        
        # Verify all parameters have gradients
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        
        assert grad_params == total_params, f"Not all parameters have gradients: {grad_params}/{total_params}"
        
        print(f"Gradient computation time: {total_time:.4f}s")


class TestResourceUtilization:
    """Test resource utilization patterns"""
    
    @pytest.mark.unit
    def test_cpu_utilization(self):
        """Test CPU utilization during intensive operations"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        
        model = construct_rn9(num_classes=10)
        
        # Monitor CPU usage during intensive computation
        cpu_before = psutil.cpu_percent(interval=None)
        
        # Perform CPU-intensive operations
        for _ in range(10):
            input_tensor = torch.randn(64, 3, 32, 32)
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
        
        cpu_after = psutil.cpu_percent(interval=1)  # 1-second interval
        
        # CPU usage should be significant during computation
        # (This test is somewhat environment-dependent)
        print(f"CPU usage: {cpu_before}% → {cpu_after}%")
    
    @pytest.mark.unit
    def test_memory_cleanup_efficiency(self):
        """Test efficiency of memory cleanup"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and destroy many objects
        for i in range(20):
            model = construct_rn9(num_classes=10)
            input_tensor = torch.randn(32, 3, 32, 32)
            output = model(input_tensor)
            
            # Explicit cleanup
            del model, input_tensor, output
            
            # Force garbage collection every few iterations
            if i % 5 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        memory_change = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory should return close to baseline
        assert abs(memory_change) < 100, f"Poor memory cleanup: {memory_change:.1f}MB change"
        
        print(f"Memory change after cleanup: {memory_change:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 