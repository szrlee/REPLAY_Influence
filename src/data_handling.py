import torch
import torchvision
from .config import CIFAR_ROOT # Use relative import for config

class CustomCIFAR10Dataset(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR10 dataset wrapper that returns (image, label, index)
    for each sample, which is useful for tracking individual samples.
    """
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # The original index in the dataset is returned along with image and label.
        return image, label, index

def get_cifar_dataloader(batch_size, num_workers=4, split='train', shuffle=None,
                         augment=True, root_dir=CIFAR_ROOT):
    """
    Creates and returns a PyTorch DataLoader for the CIFAR-10 dataset.
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        split (str): 'train' or 'val' (or 'test', treated like 'val').
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True for train, False otherwise.
        augment (bool): Whether to apply data augmentation for the training set.
        root_dir (str or Path): Root directory for CIFAR-10 data. From config by default.
    Returns:
        torch.utils.data.DataLoader: The DataLoader instance.
    """
    is_train_split = (split == 'train')

    if shuffle is None:
        shuffle = is_train_split # Shuffle only for training by default

    # Define transforms
    normalize_transform = torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )

    if is_train_split and augment:
        transform_list = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize_transform
        ]
    else: # For validation/test or no augmentation
        transform_list = [
            torchvision.transforms.ToTensor(),
            normalize_transform
        ]
    transforms_composed = torchvision.transforms.Compose(transform_list)

    dataset = CustomCIFAR10Dataset(root=str(root_dir), # Path object needs to be str for torchvision
                                   download=True,
                                   train=is_train_split,
                                   transform=transforms_composed)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=torch.cuda.is_available(), # Pin memory if using GPU
                                         persistent_workers=True if num_workers > 0 else False)
    return loader

# Helper to get a single item dataset/loader, useful for target images
class SingleItemDataset(torch.utils.data.Dataset):
    """Creates a dataset from a single data item (image, label, index tuple)."""
    def __init__(self, item_tuple):
        # item_tuple is expected to be (image_tensor, label_scalar, index_tensor)
        self.item_tuple = item_tuple
    def __getitem__(self, index):
        return self.item_tuple
    def __len__(self):
        return 1

def get_single_item_loader(item_tuple, batch_size=1):
    """Creates a DataLoader for a single data item."""
    dataset = SingleItemDataset(item_tuple)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size) 