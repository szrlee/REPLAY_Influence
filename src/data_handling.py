import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

# Assuming config variables are used directly or set_seeds moved to utils
from .config import CIFAR_ROOT, DATALOADER_NUM_WORKERS, SEED, DEVICE # Keep relevant config imports
from .utils import set_seeds # Import set_seeds from utils.py

# set_seeds function is now in utils.py

class CustomDataset(torchvision.datasets.CIFAR10):
    """Custom CIFAR10 dataset that returns (image, label, index)."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index

# This is the primary dataloader function to be used, aligned with magic.py needs.
def get_cifar10_dataloader(root_path=CIFAR_ROOT, 
                           batch_size=32, # Default batch_size, can be overridden by caller
                           num_workers=DATALOADER_NUM_WORKERS, 
                           split='train', 
                           shuffle=False, 
                           augment=False): # augment=True now implies full lds.py-style augmentation
    """
    Creates a DataLoader for CIFAR-10.
    If augment=True and split='train', applies lds.py-style augmentations and normalization.
    Otherwise (augment=False or split!='train'), uses standard CIFAR-10 normalization and only ToTensor.
    """
    transform_list = []
    
    if split == 'train' and augment: # lds.py specific training augmentations
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # lds.py normalization
        ])
    else: # Default for magic.py or validation/test splits
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) # CIFAR-10 std norm
        ])

    current_transforms = transforms.Compose(transform_list)

    is_train_split = (split == 'train')
    dataset = CustomDataset(root=root_path,
                            download=True,
                            train=is_train_split,
                            transform=current_transforms)

    actual_shuffle = shuffle if is_train_split else False

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=actual_shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True if DEVICE.type == 'cuda' else False)
    return loader

# Helper to get a single item dataset/loader, useful for target images
class SingleItemDataset(torch.utils.data.Dataset):
    """Creates a dataset from a single data item (image, label, index tuple)."""
    def __init__(self, item_tuple):
        self.item_tuple = item_tuple
    def __getitem__(self, index):
        return self.item_tuple
    def __len__(self):
        return 1

def get_single_item_loader(item_tuple, batch_size=1):
    """Creates a DataLoader for a single data item."""
    dataset = SingleItemDataset(item_tuple)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# The older get_dataloader function (lines 64-102) is removed to avoid redundancy. 