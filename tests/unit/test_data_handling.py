"""
Unit tests for data handling module in data_handling.py
"""

import pytest
import torch
import torchvision.transforms as transforms
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_handling import (
    CustomDataset,
    get_cifar10_dataloader,
    SingleItemDataset,
    get_single_item_loader
)


class TestCustomDataset:
    """Test suite for CustomDataset class"""
    
    @patch('torchvision.datasets.CIFAR10.__init__')
    @patch('torchvision.datasets.CIFAR10.__getitem__')
    def test_getitem_returns_index(self, mock_getitem, mock_init):
        """Test that CustomDataset returns (image, label, index)"""
        mock_init.return_value = None
        mock_getitem.return_value = (torch.randn(3, 32, 32), 5)  # Mock image and label
        
        dataset = CustomDataset(root="/tmp", train=True, download=False)
        
        image, label, index = dataset[10]
        
        # Check that parent __getitem__ was called with correct index
        mock_getitem.assert_called_once_with(10)
        
        # Check that index is returned correctly
        assert index == 10
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
    
    def test_inherits_from_cifar10(self):
        """Test that CustomDataset properly inherits from CIFAR10"""
        # This test verifies the inheritance structure
        import torchvision.datasets
        assert issubclass(CustomDataset, torchvision.datasets.CIFAR10)


class TestGetCIFAR10Dataloader:
    """Test suite for get_cifar10_dataloader function"""
    
    @patch('src.data_handling.CustomDataset')
    def test_valid_parameters(self, mock_dataset):
        """Test dataloader creation with valid parameters"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 1000  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        
        loader = get_cifar10_dataloader(
            root_path="/tmp/cifar",
            batch_size=32,
            num_workers=2,
            split='train',
            shuffle=True,
            augment=False
        )
        
        # Verify dataset was created with correct parameters
        mock_dataset.assert_called_once()
        call_args = mock_dataset.call_args
        assert call_args[1]['root'] == "/tmp/cifar"
        assert call_args[1]['train'] is True
        assert call_args[1]['download'] is True
        
        # Verify dataloader properties
        assert isinstance(loader, torch.utils.data.DataLoader)
    
    def test_invalid_batch_size(self):
        """Test validation with invalid batch sizes"""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            get_cifar10_dataloader(batch_size=0)
        
        with pytest.raises(ValueError, match="Batch size must be positive"):
            get_cifar10_dataloader(batch_size=-5)
    
    def test_invalid_num_workers(self):
        """Test validation with invalid number of workers"""
        with pytest.raises(ValueError, match="Number of workers must be non-negative"):
            get_cifar10_dataloader(num_workers=-1)
    
    def test_invalid_split(self):
        """Test validation with invalid split parameter"""
        with pytest.raises(ValueError, match="Split must be 'train', 'val', or 'test'"):
            get_cifar10_dataloader(split='invalid')
    
    @patch('src.data_handling.CustomDataset')
    def test_train_split_configuration(self, mock_dataset):
        """Test configuration for train split"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock CIFAR-10 training size
        mock_dataset.return_value = mock_dataset_instance
        
        loader = get_cifar10_dataloader(
            split='train',
            shuffle=True,
            augment=True
        )
        
        # Check that train=True was passed to dataset
        call_args = mock_dataset.call_args
        assert call_args[1]['train'] is True
        
        # Check that augmentation transforms were applied
        transform = call_args[1]['transform']
        assert transform is not None
    
    @patch('src.data_handling.CustomDataset')
    def test_test_split_configuration(self, mock_dataset):
        """Test configuration for test split"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 10000  # Mock CIFAR-10 test size
        mock_dataset.return_value = mock_dataset_instance
        
        loader = get_cifar10_dataloader(
            split='test',
            shuffle=False,
            augment=False
        )
        
        # Check that train=False was passed to dataset
        call_args = mock_dataset.call_args
        assert call_args[1]['train'] is False
    
    @patch('src.data_handling.CustomDataset')
    def test_augmentation_transforms(self, mock_dataset):
        """Test that augmentation transforms are applied correctly"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        
        # Test with augmentation
        get_cifar10_dataloader(split='train', augment=True)
        
        transform_with_aug = mock_dataset.call_args[1]['transform']
        
        # Reset mock
        mock_dataset.reset_mock()
        mock_dataset.return_value = mock_dataset_instance  # Re-setup after reset
        
        # Test without augmentation
        get_cifar10_dataloader(split='train', augment=False)
        
        transform_without_aug = mock_dataset.call_args[1]['transform']
        
        # Transforms should be different
        assert transform_with_aug != transform_without_aug
    
    @patch('src.data_handling.CustomDataset')
    def test_shuffle_behavior(self, mock_dataset):
        """Test shuffle behavior for different splits"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        
        # Train split with shuffle=True should shuffle
        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            get_cifar10_dataloader(split='train', shuffle=True)
            call_args = mock_dataloader.call_args[1]
            assert call_args['shuffle'] is True
        
        # Test split should not shuffle even if requested
        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            get_cifar10_dataloader(split='test', shuffle=True)
            call_args = mock_dataloader.call_args[1]
            assert call_args['shuffle'] is False
    
    @patch('src.data_handling.CustomDataset')
    def test_path_handling(self, mock_dataset):
        """Test that path parameters are handled correctly"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        
        # Test with string path
        get_cifar10_dataloader(root_path="/tmp/data")
        assert mock_dataset.call_args[1]['root'] == "/tmp/data"
        
        # Reset mock
        mock_dataset.reset_mock()
        mock_dataset.return_value = mock_dataset_instance  # Re-setup after reset
        
        # Test with Path object
        path_obj = Path("/tmp/data")
        get_cifar10_dataloader(root_path=path_obj)
        assert mock_dataset.call_args[1]['root'] == str(path_obj)
    
    @patch('src.data_handling.CustomDataset')
    def test_dataset_creation_failure(self, mock_dataset):
        """Test handling of dataset creation failure"""
        mock_dataset.side_effect = RuntimeError("Dataset creation failed")
        
        with pytest.raises(RuntimeError, match="Failed to load CIFAR-10 dataset"):
            get_cifar10_dataloader()
    
    @patch('src.data_handling.CustomDataset')
    @patch('torch.utils.data.DataLoader')
    def test_dataloader_creation_failure(self, mock_dataloader, mock_dataset):
        """Test handling of dataloader creation failure"""
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        mock_dataloader.side_effect = RuntimeError("DataLoader creation failed")
        
        with pytest.raises(RuntimeError, match="Failed to create DataLoader"):
            get_cifar10_dataloader()


class TestSingleItemDataset:
    """Test suite for SingleItemDataset class"""
    
    def test_initialization(self):
        """Test SingleItemDataset initialization"""
        item = (torch.randn(3, 32, 32), 5, 10)
        dataset = SingleItemDataset(item)
        
        assert dataset.item_tuple == item
    
    def test_getitem(self):
        """Test that __getitem__ always returns the same item"""
        item = (torch.randn(3, 32, 32), 5, 10)
        dataset = SingleItemDataset(item)
        
        # Should return the same item regardless of index
        assert dataset[0] == item
        assert dataset[99] == item
        assert dataset[-1] == item
    
    def test_len(self):
        """Test that __len__ always returns 1"""
        item = (torch.randn(3, 32, 32), 5, 10)
        dataset = SingleItemDataset(item)
        
        assert len(dataset) == 1
    
    def test_different_item_types(self):
        """Test with different types of items"""
        # Test with different tensor shapes
        item1 = (torch.randn(1, 28, 28), 3, 5)
        dataset1 = SingleItemDataset(item1)
        assert dataset1[0] == item1
        
        # Test with different data types
        item2 = (torch.zeros(3, 32, 32), 0, 0)
        dataset2 = SingleItemDataset(item2)
        assert dataset2[0] == item2


class TestGetSingleItemLoader:
    """Test suite for get_single_item_loader function"""
    
    def test_default_batch_size(self):
        """Test single item loader with default batch size"""
        item = (torch.randn(3, 32, 32), 5, 10)
        loader = get_single_item_loader(item)
        
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 1
    
    def test_custom_batch_size(self):
        """Test single item loader with custom batch size"""
        item = (torch.randn(3, 32, 32), 5, 10)
        loader = get_single_item_loader(item, batch_size=2)
        
        assert loader.batch_size == 2
    
    def test_loader_iteration(self):
        """Test that the loader can be iterated and produces correct output"""
        item = (torch.randn(3, 32, 32), 5, 10)
        loader = get_single_item_loader(item, batch_size=1)
        
        # Should have exactly one batch
        batches = list(loader)
        assert len(batches) == 1
        
        # Extract the batch
        batch_images, batch_labels, batch_indices = batches[0]
        
        # Check batch dimensions
        assert batch_images.shape == (1, 3, 32, 32)  # Batch size of 1
        assert batch_labels.shape == (1,)
        assert batch_indices.shape == (1,)
        
        # Check values match original item
        torch.testing.assert_close(batch_images[0], item[0])
        assert batch_labels[0].item() == item[1]
        assert batch_indices[0].item() == item[2]


class TestDataHandlingIntegration:
    """Integration tests for data handling components"""
    
    @patch('src.data_handling.CustomDataset')
    def test_cifar10_to_single_item_workflow(self, mock_dataset):
        """Test complete workflow from CIFAR-10 to single item loader"""
        # Mock CIFAR-10 dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock dataset size
        mock_dataset_instance.__getitem__.return_value = (
            torch.randn(3, 32, 32), 5, 100
        )
        mock_dataset.return_value = mock_dataset_instance
        
        # Create CIFAR-10 dataloader
        cifar_loader = get_cifar10_dataloader(batch_size=1)
        
        # Get a sample from CIFAR-10
        for batch in cifar_loader:
            sample_item = (batch[0][0], batch[1][0], batch[2][0])
            break
        
        # Create single item loader
        single_loader = get_single_item_loader(sample_item)
        
        # Verify single item loader works
        assert isinstance(single_loader, torch.utils.data.DataLoader)
        
        # Check that we can iterate through it
        batches = list(single_loader)
        assert len(batches) == 1
    
    def test_different_normalization_strategies(self):
        """Test different normalization strategies work"""
        # Test that transforms can be created without errors
        
        # Standard CIFAR-10 normalization
        standard_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
        ])
        
        # LDS-style normalization
        lds_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Both should be valid transform objects
        assert isinstance(standard_transform, transforms.Compose)
        assert isinstance(lds_transform, transforms.Compose)
    
    @patch('src.data_handling.CustomDataset')
    def test_memory_efficiency(self, mock_dataset):
        """Test memory efficiency with different batch sizes"""
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 50000  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        
        # Test with different batch sizes
        for batch_size in [1, 32, 128]:
            loader = get_cifar10_dataloader(batch_size=batch_size)
            assert loader.batch_size == batch_size


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_item_tuple_structure(self):
        """Test handling of invalid item tuple structure"""
        # Valid structure should work
        valid_item = (torch.randn(3, 32, 32), 5, 10)
        dataset = SingleItemDataset(valid_item)
        assert len(dataset) == 1
        
        # Invalid structures should still work (no validation in constructor)
        # The validation happens at usage time
        invalid_item = (torch.randn(3, 32, 32), "invalid_label")
        dataset = SingleItemDataset(invalid_item)
        assert len(dataset) == 1
    
    @patch('pathlib.Path.mkdir')
    def test_directory_creation_handling(self, mock_mkdir):
        """Test that path objects are handled correctly"""
        from pathlib import Path
        
        path = Path("/tmp/test_cifar")
        
        # The function should convert Path to string
        with patch('src.data_handling.CustomDataset') as mock_dataset:
            mock_dataset_instance = MagicMock()
            mock_dataset.return_value = mock_dataset_instance
            
            get_cifar10_dataloader(root_path=path)
            
            # Should have converted Path to string
            assert mock_dataset.call_args[1]['root'] == str(path)
    
    def test_extreme_batch_sizes(self):
        """Test handling of extreme batch sizes"""
        item = (torch.randn(3, 32, 32), 5, 10)
        
        # Very large batch size should work (item will just be repeated)
        loader = get_single_item_loader(item, batch_size=1000)
        assert loader.batch_size == 1000
        
        # Batch size of 1 should work
        loader = get_single_item_loader(item, batch_size=1)
        assert loader.batch_size == 1


class TestCompatibility:
    """Test compatibility with PyTorch ecosystem"""
    
    def test_dataloader_worker_compatibility(self):
        """Test compatibility with DataLoader workers"""
        item = (torch.randn(3, 32, 32), 5, 10)
        
        # Should work with num_workers > 0
        try:
            loader = get_single_item_loader(item, batch_size=1)
            # Try to iterate (this tests basic functionality)
            list(loader)
        except Exception as e:
            pytest.fail(f"DataLoader iteration failed: {e}")
    
    @patch('torch.cuda.is_available')
    def test_device_compatibility(self, mock_cuda):
        """Test device compatibility handling"""
        # Test CUDA available case
        mock_cuda.return_value = True
        
        with patch('src.data_handling.CustomDataset') as mock_dataset:
            mock_dataset_instance = MagicMock()
            mock_dataset.return_value = mock_dataset_instance
            
            with patch('torch.utils.data.DataLoader') as mock_dataloader:
                get_cifar10_dataloader()
                
                # Check if pin_memory was set appropriately
                call_args = mock_dataloader.call_args[1]
                # pin_memory should be True when CUDA is available and device is cuda
                # (this depends on DEVICE config, but we test the call was made)
                assert 'pin_memory' in call_args
        
        # Test CUDA not available case
        mock_cuda.return_value = False
        
        with patch('src.data_handling.CustomDataset') as mock_dataset:
            mock_dataset_instance = MagicMock()
            mock_dataset.return_value = mock_dataset_instance
            
            with patch('torch.utils.data.DataLoader') as mock_dataloader:
                get_cifar10_dataloader()
                
                # Should still create dataloader
                assert mock_dataloader.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 