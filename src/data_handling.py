import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Tuple, Optional, Union
from pathlib import Path

from .config import CIFAR_ROOT, DATALOADER_NUM_WORKERS, DEVICE # Keep relevant config imports

class CustomDataset(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR10 dataset that returns (image, label, index).
    
    This extends the standard CIFAR10 dataset to include the sample index,
    which is crucial for influence function analysis.
    """
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a data sample with its index.
        
        Args:
            index (int): Sample index.
            
        Returns:
            Tuple[torch.Tensor, int, int]: (image, label, index)
        """
        image, label = super().__getitem__(index)
        return image, label, index

# This is the primary dataloader function to be used, aligned with magic.py needs.
def get_cifar10_dataloader(
    root_path: Union[str, Path] = CIFAR_ROOT, 
    batch_size: int = 32,
    num_workers: int = DATALOADER_NUM_WORKERS, 
    split: str = 'train', 
    shuffle: bool = False, 
    augment: bool = False
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for CIFAR-10 with comprehensive error handling.
    
    Args:
        root_path (Union[str, Path]): Path to store/load CIFAR-10 data.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        split (str): Data split - 'train', 'val', or 'test'.
        shuffle (bool): Whether to shuffle the data.
        augment (bool): Whether to apply data augmentation.
        
    Returns:
        torch.utils.data.DataLoader: Configured CIFAR-10 DataLoader.
        
    Raises:
        ValueError: If invalid parameters are provided.
        RuntimeError: If data loading fails.
    """
    # Validate parameters
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    
    if num_workers < 0:
        raise ValueError(f"Number of workers must be non-negative, got {num_workers}")
    
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Split must be 'train', 'val', or 'test', got '{split}'")
    
    # Convert to Path for consistency
    root_path = Path(root_path)
    
    transform_list = []
    
    if split == 'train' and augment:
        # Enhanced augmentation for better generalization
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # LDS-style normalization
        ])
    else:
        # Standard CIFAR-10 normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
        ])

    current_transforms = transforms.Compose(transform_list)

    is_train_split = (split == 'train')
    
    try:
        dataset = CustomDataset(
            root=str(root_path),
            download=True,
            train=is_train_split,
            transform=current_transforms
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load CIFAR-10 dataset from {root_path}: {e}") from e

    # Only shuffle training data if explicitly requested
    actual_shuffle = shuffle if is_train_split else False

    try:
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=actual_shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True if DEVICE.type == 'cuda' else False,
            persistent_workers=num_workers > 0  # Improve performance for multi-worker loading
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create DataLoader: {e}") from e

    return loader

# Helper to get a single item dataset/loader, useful for target images
class SingleItemDataset(torch.utils.data.Dataset):
    """
    Creates a dataset from a single data item (image, label, index tuple).
    
    Useful for creating dataloaders for individual samples in analysis.
    """
    
    def __init__(self, item_tuple: Tuple[torch.Tensor, int, int]) -> None:
        """
        Initialize with a single data item.
        
        Args:
            item_tuple (Tuple[torch.Tensor, int, int]): (image, label, index)
        """
        self.item_tuple = item_tuple
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Always returns the same item regardless of index."""
        return self.item_tuple
        
    def __len__(self) -> int:
        """Dataset always contains exactly one item."""
        return 1

def get_single_item_loader(
    item_tuple: Tuple[torch.Tensor, int, int], 
    batch_size: int = 1
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for a single data item.
    
    Args:
        item_tuple (Tuple[torch.Tensor, int, int]): (image, label, index)
        batch_size (int): Batch size (typically 1 for single items).
        
    Returns:
        torch.utils.data.DataLoader: DataLoader containing the single item.
    """
    dataset = SingleItemDataset(item_tuple)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# The older get_dataloader function (lines 64-102) is removed to avoid redundancy. 