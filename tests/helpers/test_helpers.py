import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

# Project-specific imports from src
from src import config
from src.utils import create_deterministic_dataloader, update_dataloader_epoch
from src.data_handling import get_cifar10_dataloader

def create_test_dataloader(instance_id: str, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Helper function to create a deterministic dataloader for testing.
    
    Args:
        instance_id: Unique identifier for the dataloader instance
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
    
    Returns:
        Deterministic dataloader
    """
    return create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id=instance_id,
        batch_size=batch_size,
        split='train',
        shuffle=shuffle,
        augment=False,
        num_workers=0,  # Use 0 for tests to avoid multiprocessing issues
        root_path=config.CIFAR_ROOT
    )

def get_first_batch_indices(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Helper function to get the indices from the first batch of a dataloader.
    
    Args:
        dataloader: DataLoader to get batch from
    
    Returns:
        Tensor of indices from the first batch
    """
    batch = next(iter(dataloader))
    return batch[2]  # Assuming format is (data, labels, indices)

def compare_batch_indices(indices1: torch.Tensor, indices2: torch.Tensor, 
                         context: str = "comparison") -> bool:
    """
    Helper function to compare two sets of batch indices.
    
    Args:
        indices1: First set of indices
        indices2: Second set of indices
        context: Description of what's being compared (for logging)
    
    Returns:
        True if indices are equal, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if torch.equal(indices1, indices2):
        logger.debug(f"✅ {context}: Indices match - {indices1[:5].tolist()}")
        return True
    else:
        logger.warning(f"❌ {context}: Indices differ - {indices1[:5].tolist()} vs {indices2[:5].tolist()}")
        return False

def assert_dataloader_determinism(instance_id1: str, instance_id2: str, 
                                should_be_equal: bool = True, 
                                context: str = "dataloader determinism") -> None:
    """
    Helper function to assert deterministic behavior between two dataloader instances.
    
    Args:
        instance_id1: First dataloader instance ID
        instance_id2: Second dataloader instance ID  
        should_be_equal: Whether the dataloaders should produce identical results
        context: Description for error messages
    """
    loader1 = create_test_dataloader(instance_id1)
    loader2 = create_test_dataloader(instance_id2)
    
    indices1 = get_first_batch_indices(loader1)
    indices2 = get_first_batch_indices(loader2)
    
    are_equal = torch.equal(indices1, indices2)
    
    if should_be_equal:
        assert are_equal, f"{context}: Expected identical results but got different indices. " \
                         f"Loader1: {indices1[:5].tolist()}, Loader2: {indices2[:5].tolist()}"
    else:
        assert not are_equal, f"{context}: Expected different results but got identical indices: {indices1[:5].tolist()}"

def assert_multi_epoch_consistency(instance_id: str, num_epochs: int = 2, 
                                 batches_per_epoch: int = 5) -> List[List[torch.Tensor]]:
    """
    Helper function to test multi-epoch consistency of a dataloader.
    
    Args:
        instance_id: Dataloader instance ID
        num_epochs: Number of epochs to test
        batches_per_epoch: Number of batches to collect per epoch
    
    Returns:
        List of lists containing batch indices for each epoch
    """
    logger = logging.getLogger(__name__)
    
    test_loader = create_test_dataloader(instance_id, batch_size=100, shuffle=True)
    
    all_epoch_indices = []
    
    for epoch in range(num_epochs):
        update_dataloader_epoch(test_loader, epoch)
        epoch_indices = []
        
        for batch_idx, (_, _, indices) in enumerate(test_loader):
            epoch_indices.append(indices.clone())
            if batch_idx >= batches_per_epoch - 1:
                break
        
        all_epoch_indices.append(epoch_indices)
        logger.debug(f"Epoch {epoch}: Collected {len(epoch_indices)} batches")
    
    # Verify that different epochs produce different shuffling
    if num_epochs >= 2:
        epoch1_flat = torch.cat(all_epoch_indices[0])
        epoch2_flat = torch.cat(all_epoch_indices[1])
        
        assert not torch.equal(epoch1_flat, epoch2_flat), \
            f"Different epochs should produce different batch orders when shuffle=True. " \
            f"Epoch 0: {epoch1_flat[:5].tolist()}, Epoch 1: {epoch2_flat[:5].tolist()}"
        
        logger.debug("✅ Multi-epoch shuffling verified - different epochs produce different orders")
    
    return all_epoch_indices 