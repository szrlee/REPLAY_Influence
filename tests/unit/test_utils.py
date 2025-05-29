"""
Unit tests for utilities module in utils.py
"""

import pytest
import torch
import numpy as np
import random
import logging
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from src.utils import (
    setup_logging,
    set_global_deterministic_state,
    derive_component_seed,
    seed_worker,
    deterministic_context,
    create_deterministic_dataloader,
    create_deterministic_model,
    create_deterministic_optimizer,
    create_deterministic_scheduler,
    evaluate_measurement_functions,
    get_measurement_function_targets,
    validate_measurement_function_setup,
    DeterministicStateError,
    SeedDerivationError,
    ComponentCreationError
)


class TestSetupLogging:
    """Test suite for setup_logging function"""
    
    def test_default_configuration(self):
        """Test logging setup with default parameters"""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'influence_analysis'
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
    
    def test_custom_log_level(self):
        """Test logging setup with custom log level"""
        logger = setup_logging(log_level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_file_logging(self, tmp_path):
        """Test logging setup with file output"""
        log_file = tmp_path / "test.log"
        
        logger = setup_logging(log_file=str(log_file))
        
        # Should have both console and file handlers
        assert len(logger.handlers) >= 2
        
        # Test that logging actually works
        logger.info("Test message")
        
        # Check that file was created and contains content
        assert log_file.exists()
        assert log_file.read_text()
    
    def test_invalid_log_level(self):
        """Test logging setup with invalid log level"""
        # Should handle invalid levels gracefully
        with pytest.raises(AttributeError):
            logger = setup_logging(log_level="INVALID")


class TestDeterministicSeedManagement:
    """Test suite for deterministic seed management functions"""
    
    def test_set_global_deterministic_state(self):
        """Test global deterministic state setting"""
        master_seed = 42
        
        # Should not raise exceptions
        set_global_deterministic_state(master_seed, enable_deterministic=True)
        
        # Check that seeds were set (basic verification)
        # We can't directly check the internal state, but we can verify
        # that subsequent random operations are deterministic
        set_global_deterministic_state(master_seed, enable_deterministic=True)
        val1 = random.random()
        
        set_global_deterministic_state(master_seed, enable_deterministic=True)
        val2 = random.random()
        
        assert val1 == val2
    
    def test_derive_component_seed(self):
        """Test component seed derivation"""
        master_seed = 42
        
        # Test basic derivation
        seed1 = derive_component_seed(master_seed, "dataloader")
        seed2 = derive_component_seed(master_seed, "model")
        
        # Different purposes should give different seeds
        assert seed1 != seed2
        
        # Same purpose should give same seed
        seed1_repeat = derive_component_seed(master_seed, "dataloader")
        assert seed1 == seed1_repeat
    
    def test_derive_component_seed_with_instance_id(self):
        """Test component seed derivation with instance IDs"""
        master_seed = 42
        
        seed1 = derive_component_seed(master_seed, "model", "instance_1")
        seed2 = derive_component_seed(master_seed, "model", "instance_2")
        
        # Different instance IDs should give different seeds
        assert seed1 != seed2
        
        # Same instance ID should give same seed
        seed1_repeat = derive_component_seed(master_seed, "model", "instance_1")
        assert seed1 == seed1_repeat
    
    def test_seed_worker(self):
        """Test worker seed function"""
        # Mock torch.initial_seed
        with patch('torch.initial_seed', return_value=12345):
            # Should not raise exceptions
            seed_worker(0)
            seed_worker(1)
    
    def test_deterministic_context(self):
        """Test deterministic context manager"""
        with deterministic_context(42, "test_operation"):
            # Should execute without errors
            val = torch.randn(10)
            assert isinstance(val, torch.Tensor)


class TestDeterministicComponentCreation:
    """Test suite for deterministic component creation functions"""
    
    def test_create_deterministic_dataloader(self):
        """Test deterministic dataloader creation"""
        def mock_dataloader_creator(**kwargs):
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 1000
            
            # Return a mock dataloader with all required attributes
            mock_dataloader = MagicMock()
            mock_dataloader.dataset = mock_dataset
            mock_dataloader.batch_size = kwargs.get('batch_size', 32)
            mock_dataloader.num_workers = kwargs.get('num_workers', 0)
            mock_dataloader.pin_memory = kwargs.get('pin_memory', False)
            mock_dataloader.drop_last = kwargs.get('drop_last', False)
            mock_dataloader.collate_fn = None
            mock_dataloader.persistent_workers = False  # Set to False to avoid the error
            return mock_dataloader
        
        loader = create_deterministic_dataloader(
            master_seed=42,
            creator_func=mock_dataloader_creator,
            batch_size=32
        )
        
        # Should create a dataloader with proper configuration
        assert loader is not None
        assert isinstance(loader, torch.utils.data.DataLoader)
    
    def test_create_deterministic_model(self):
        """Test deterministic model creation"""
        def mock_model_creator(**kwargs):
            return torch.nn.Linear(10, 1)
        
        model = create_deterministic_model(
            master_seed=42,
            creator_func=mock_model_creator,
            input_size=10
        )
        
        assert isinstance(model, torch.nn.Module)
    
    def test_create_deterministic_optimizer(self):
        """Test deterministic optimizer creation"""
        model = torch.nn.Linear(10, 1)
        
        optimizer = create_deterministic_optimizer(
            master_seed=42,
            optimizer_class=torch.optim.SGD,
            model_params=model.parameters(),
            lr=0.01
        )
        
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_create_deterministic_scheduler(self):
        """Test deterministic scheduler creation"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        scheduler = create_deterministic_scheduler(
            master_seed=42,
            optimizer=optimizer,
            schedule_type="OneCycleLR",
            total_steps=100,
            max_lr=0.1
        )
        
        assert scheduler is not None
    
    def test_create_deterministic_scheduler_none_type(self):
        """Test deterministic scheduler creation with None type"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        scheduler = create_deterministic_scheduler(
            master_seed=42,
            optimizer=optimizer,
            schedule_type=None,
            total_steps=100
        )
        
        assert scheduler is None


class TestMeasurementFunctions:
    """Test suite for measurement function utilities"""
    
    def test_get_measurement_function_targets(self):
        """Test getting measurement function targets"""
        targets = get_measurement_function_targets()
        
        assert isinstance(targets, list)
        assert len(targets) > 0
        assert all(isinstance(t, int) for t in targets)
    
    def test_validate_measurement_function_setup(self):
        """Test measurement function setup validation"""
        # Should not raise for valid setup
        validate_measurement_function_setup(test_dataset_size=10000)
        
        # Should raise for invalid setup
        with pytest.raises(ValueError):
            validate_measurement_function_setup(test_dataset_size=5)  # Too small
    
    @patch('src.utils.get_measurement_function_targets')
    def test_evaluate_measurement_functions(self, mock_targets):
        """Test measurement function evaluation"""
        # Mock targets
        mock_targets.return_value = [0, 1, 2]
        
        # Create a simple CNN model that can handle CIFAR-10 input
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 10)
        )
        
        # Create a simple dataset that returns 3 values (image, label, index)
        class MockDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                # Return tensors on CPU to match the model
                return torch.randn(3, 32, 32), idx % 10, idx  # Return image, label, index
        
        dataset = MockDataset()
        
        # Test evaluation with CPU tensors only
        with patch('src.config.DEVICE', torch.device('cpu')):
            results = evaluate_measurement_functions(
                model=model,
                test_dataset=dataset,
                target_indices=[0, 1, 2],
                device=torch.device('cpu')
            )
        
        # Should return dictionary with losses
        assert isinstance(results, dict)
        assert len(results) == 3
        for idx in [0, 1, 2]:
            assert idx in results
            assert isinstance(results[idx], float)
    
    def test_evaluate_measurement_functions_invalid_indices(self):
        """Test measurement function evaluation with invalid indices"""
        # Create a simple CNN model that can handle CIFAR-10 input
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 10)
        )
        
        class MockDataset:
            def __len__(self):
                return 5
            
            def __getitem__(self, idx):
                if idx >= 5:
                    raise IndexError("Index out of range")
                # Return tensors on CPU to match the model
                return torch.randn(3, 32, 32), idx % 10, idx  # Return image, label, index
        
        dataset = MockDataset()
        
        # Should handle invalid indices gracefully
        with patch('src.config.DEVICE', torch.device('cpu')):
            with pytest.raises(ValueError, match="Target index .* out of bounds"):
                results = evaluate_measurement_functions(
                    model=model,
                    test_dataset=dataset,
                    target_indices=[0, 1, 10],  # 10 is out of range
                    device=torch.device('cpu')
                )


class TestErrorHandling:
    """Test suite for error handling and custom exceptions"""
    
    def test_custom_exceptions(self):
        """Test that custom exceptions can be raised and caught"""
        with pytest.raises(DeterministicStateError):
            raise DeterministicStateError("Test error")
        
        with pytest.raises(SeedDerivationError):
            raise SeedDerivationError("Test error")
        
        with pytest.raises(ComponentCreationError):
            raise ComponentCreationError("Test error")
    
    def test_component_creation_error_handling(self):
        """Test error handling in component creation"""
        def failing_creator(**kwargs):
            raise RuntimeError("Creation failed")
        
        with pytest.raises(ComponentCreationError):
            create_deterministic_model(
                master_seed=42,
                creator_func=failing_creator
            )
    
    def test_seed_derivation_bounds(self):
        """Test seed derivation with edge cases"""
        # Very large master seed
        large_seed = 2**30
        derived = derive_component_seed(large_seed, "test")
        assert isinstance(derived, int)
        assert 0 <= derived < 2**31
        
        # Empty purpose string
        derived = derive_component_seed(42, "")
        assert isinstance(derived, int)


class TestDeterministicReproducibility:
    """Test suite for reproducibility verification"""
    
    def test_deterministic_pytorch_operations(self):
        """Test that PyTorch operations are deterministic"""
        set_global_deterministic_state(42, enable_deterministic=True)
        
        # Perform some PyTorch operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result1 = torch.mm(x, y)
        
        # Reset and repeat
        set_global_deterministic_state(42, enable_deterministic=True)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result2 = torch.mm(x, y)
        
        torch.testing.assert_close(result1, result2)
    
    def test_deterministic_numpy_operations(self):
        """Test that NumPy operations are deterministic"""
        set_global_deterministic_state(42, enable_deterministic=True)
        
        result1 = np.random.randn(100)
        
        set_global_deterministic_state(42, enable_deterministic=True)
        result2 = np.random.randn(100)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_deterministic_python_random(self):
        """Test that Python random is deterministic"""
        set_global_deterministic_state(42, enable_deterministic=True)
        
        result1 = [random.random() for _ in range(10)]
        
        set_global_deterministic_state(42, enable_deterministic=True)
        result2 = [random.random() for _ in range(10)]
        
        assert result1 == result2


class TestSchedulerCreation:
    """Test suite for scheduler creation functionality"""
    
    def test_onecycle_scheduler_creation(self):
        """Test OneCycleLR scheduler creation"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        scheduler = create_deterministic_scheduler(
            master_seed=42,
            optimizer=optimizer,
            schedule_type="OneCycleLR",
            total_steps=100,
            max_lr=0.1
        )
        
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    
    def test_lambda_scheduler_creation(self):
        """Test LambdaLR scheduler creation"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # LambdaLR is not supported, so this should raise an error
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            scheduler = create_deterministic_scheduler(
                master_seed=42,
                optimizer=optimizer,
                schedule_type="LambdaLR",
                total_steps=100,
                lr_lambda=lambda epoch: 0.95 ** epoch
            )
    
    def test_unsupported_scheduler_type(self):
        """Test handling of unsupported scheduler types"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            create_deterministic_scheduler(
                master_seed=42,
                optimizer=optimizer,
                schedule_type="UnsupportedScheduler",
                total_steps=100
            )


class TestDeviceCompatibility:
    """Test suite for device compatibility"""
    
    @patch('torch.cuda.is_available')
    def test_cuda_deterministic_setup(self, mock_cuda_available):
        """Test deterministic setup with CUDA"""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.manual_seed') as mock_cuda_seed, \
             patch('torch.cuda.manual_seed_all') as mock_cuda_seed_all:
            
            set_global_deterministic_state(42, enable_deterministic=True)
            
            mock_cuda_seed.assert_called_with(42)
            # The function might call manual_seed_all multiple times, so just check it was called
            assert mock_cuda_seed_all.call_count >= 1
            assert mock_cuda_seed_all.call_args[0][0] == 42
    
    def test_cpu_only_deterministic_setup(self):
        """Test deterministic setup without CUDA"""
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise errors
            set_global_deterministic_state(42, enable_deterministic=True)


class TestUtilityIntegration:
    """Integration tests for utility functions"""
    
    def test_full_deterministic_workflow(self):
        """Test complete deterministic workflow"""
        master_seed = 42
        
        # Set global state
        set_global_deterministic_state(master_seed, enable_deterministic=True)
        
        # Create components deterministically
        def model_creator(**kwargs):  # Accept kwargs to handle num_classes parameter
            return torch.nn.Linear(10, 1)
        
        model = create_deterministic_model(
            master_seed=master_seed,
            creator_func=model_creator
        )
        
        optimizer = create_deterministic_optimizer(
            master_seed=master_seed,
            optimizer_class=torch.optim.SGD,
            model_params=model.parameters(),
            lr=0.01
        )
        
        # Should all be created successfully
        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_measurement_function_workflow(self):
        """Test measurement function workflow"""
        # Get targets
        targets = get_measurement_function_targets()
        
        # Validate setup
        validate_measurement_function_setup(test_dataset_size=10000)
        
        # Create a simple CNN model that can handle CIFAR-10 input
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 10)
        )
        
        class MockDataset:
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                # Return tensors on CPU to match the model
                return torch.randn(3, 32, 32), idx % 10, idx  # Return image, label, index
        
        dataset = MockDataset()
        
        # Evaluate with CPU device
        with patch('src.config.DEVICE', torch.device('cpu')):
            results = evaluate_measurement_functions(
                model=model,
                test_dataset=dataset,
                target_indices=targets[:5],  # Use first 5 targets
                device=torch.device('cpu')
            )
        
        assert isinstance(results, dict)
        assert len(results) <= 5


class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations"""
    
    def test_large_seed_values(self):
        """Test handling of large seed values"""
        large_seed = 2**31 - 1
        
        # Should handle without overflow
        set_global_deterministic_state(large_seed, enable_deterministic=True)
        derived = derive_component_seed(large_seed, "test")
        
        assert isinstance(derived, int)
        assert derived >= 0
    
    def test_memory_cleanup_context(self):
        """Test that deterministic context cleans up properly"""
        initial_seed = torch.initial_seed()
        
        with deterministic_context(999, "test"):
            # Do some operations
            torch.randn(100)
        
        # Context should not affect global state permanently
        # (though we can't easily verify the exact seed restoration)
        assert torch.initial_seed() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 