import torch
import torch.utils.data
import numpy as np
from src.utils import set_seeds
from src.data_handling import get_cifar10_dataloader
from src import config

print("=== Minimal DataLoader determinism test ===")

# Test 1: Basic determinism with simple dataset
print("Test 1: Basic determinism with simple dataset")
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = list(range(size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], idx

set_seeds(42)
dataset = SimpleDataset(1000)
loader1 = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
batch1 = next(iter(loader1))[1][:5]  # Get indices

set_seeds(42)
dataset = SimpleDataset(1000)
loader2 = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
batch2 = next(iter(loader2))[1][:5]  # Get indices

print(f"Simple dataset - Batch1: {batch1}")
print(f"Simple dataset - Batch2: {batch2}")
print(f"Simple dataset - Same: {torch.equal(batch1, batch2)}")

# Test 2: Test with CIFAR10 dataset with proper transforms
print("\nTest 2: CIFAR10 dataset determinism")
from src.data_handling import CustomDataset
from src import config
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
])

set_seeds(42)
cifar_dataset1 = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
cifar_loader1 = torch.utils.data.DataLoader(cifar_dataset1, batch_size=10, shuffle=True, num_workers=0)
cifar_batch1 = next(iter(cifar_loader1))[2][:5]  # Get indices

set_seeds(42)
cifar_dataset2 = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
cifar_loader2 = torch.utils.data.DataLoader(cifar_dataset2, batch_size=10, shuffle=True, num_workers=0)
cifar_batch2 = next(iter(cifar_loader2))[2][:5]  # Get indices

print(f"CIFAR10 - Batch1: {cifar_batch1}")
print(f"CIFAR10 - Batch2: {cifar_batch2}")
print(f"CIFAR10 - Same: {torch.equal(cifar_batch1, cifar_batch2)}")

# Test 3: Check if the issue is with creating multiple loaders
print("\nTest 3: Multiple loader creation timing")
set_seeds(42)
cifar_dataset_a = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
cifar_loader_a = torch.utils.data.DataLoader(cifar_dataset_a, batch_size=10, shuffle=True, num_workers=0)

set_seeds(42)
cifar_dataset_b = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
cifar_loader_b = torch.utils.data.DataLoader(cifar_dataset_b, batch_size=10, shuffle=True, num_workers=0)

# Create iterators AFTER both loaders exist
iter_a = iter(cifar_loader_a)
iter_b = iter(cifar_loader_b)

batch_a = next(iter_a)[2][:5]
batch_b = next(iter_b)[2][:5]

print(f"Multiple loaders - Batch A: {batch_a}")
print(f"Multiple loaders - Batch B: {batch_b}")
print(f"Multiple loaders - Same: {torch.equal(batch_a, batch_b)}")

# Test 4: Check if the issue is with the get_cifar10_dataloader function
print("\nTest 4: get_cifar10_dataloader function")
from src.data_handling import get_cifar10_dataloader

set_seeds(42)
loader_func1 = get_cifar10_dataloader(batch_size=10, split='train', shuffle=True, augment=False, num_workers=0, root_path=config.CIFAR_ROOT)
batch_func1 = next(iter(loader_func1))[2][:5]

set_seeds(42)
loader_func2 = get_cifar10_dataloader(batch_size=10, split='train', shuffle=True, augment=False, num_workers=0, root_path=config.CIFAR_ROOT)
batch_func2 = next(iter(loader_func2))[2][:5]

print(f"get_cifar10_dataloader - Batch1: {batch_func1}")
print(f"get_cifar10_dataloader - Batch2: {batch_func2}")
print(f"get_cifar10_dataloader - Same: {torch.equal(batch_func1, batch_func2)}")

# Test 5: The exact issue - creating loaders then iterators
print("\nTest 5: Exact issue reproduction")
set_seeds(42)
loader_x = get_cifar10_dataloader(batch_size=10, split='train', shuffle=True, augment=False, num_workers=0, root_path=config.CIFAR_ROOT)
set_seeds(42)
loader_y = get_cifar10_dataloader(batch_size=10, split='train', shuffle=True, augment=False, num_workers=0, root_path=config.CIFAR_ROOT)

# This is what the verification function does
iter_x = iter(loader_x)
iter_y = iter(loader_y)

batch_x = next(iter_x)[2][:5]
batch_y = next(iter_y)[2][:5]

print(f"Exact issue - Batch X: {batch_x}")
print(f"Exact issue - Batch Y: {batch_y}")
print(f"Exact issue - Same: {torch.equal(batch_x, batch_y)}")

print(f"\n=== CONCLUSION ===")
print("If Test 5 shows False, then the issue is with creating multiple DataLoaders")
print("before creating iterators. The solution is to reset seeds before iterator creation.")

print("=== Testing iterator reset approach ===")

# Create the DataLoaders once (like in verification function)
set_seeds(config.SEED)
magic_loader = get_cifar10_dataloader(
    batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
    split='train', shuffle=True, augment=False,
    num_workers=0, root_path=config.CIFAR_ROOT
)

set_seeds(config.SEED)
lds_loader = get_cifar10_dataloader(
    batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
    split='train', 
    shuffle=True, 
    augment=False,
    num_workers=0,
    root_path=config.CIFAR_ROOT
)

print("Created both DataLoaders")

# Test the iterator reset approach for batch 0
print("\n=== Testing batch 0 ===")
set_seeds(config.SEED)
magic_iter = iter(magic_loader)
set_seeds(config.SEED)
lds_iter = iter(lds_loader)

magic_batch_0 = next(magic_iter)[2][:5]
lds_batch_0 = next(lds_iter)[2][:5]

print(f"Batch 0 - MAGIC: {magic_batch_0}")
print(f"Batch 0 - LDS: {lds_batch_0}")
print(f"Batch 0 - Same: {torch.equal(magic_batch_0, lds_batch_0)}")

# Test the iterator reset approach for batch 1
print("\n=== Testing batch 1 ===")
set_seeds(config.SEED)
magic_iter = iter(magic_loader)
set_seeds(config.SEED)
lds_iter = iter(lds_loader)

# Skip to batch 1
next(magic_iter)  # Skip batch 0
next(lds_iter)    # Skip batch 0

magic_batch_1 = next(magic_iter)[2][:5]
lds_batch_1 = next(lds_iter)[2][:5]

print(f"Batch 1 - MAGIC: {magic_batch_1}")
print(f"Batch 1 - LDS: {lds_batch_1}")
print(f"Batch 1 - Same: {torch.equal(magic_batch_1, lds_batch_1)}")

# Test if the issue is with the verification function logic
print("\n=== Testing verification function logic ===")
for i in range(2):  # Test first 2 batches
    print(f"\n--- Testing batch {i} ---")
    set_seeds(config.SEED)
    magic_iter = iter(magic_loader)
    
    set_seeds(config.SEED)
    lds_iter = iter(lds_loader)
    
    # Skip to the i-th batch
    for _ in range(i + 1):
        magic_batch = next(magic_iter)
        lds_batch = next(lds_iter)
    
    magic_indices = magic_batch[2][:5]
    lds_indices = lds_batch[2][:5]
    
    print(f"Verification logic batch {i} - MAGIC: {magic_indices}")
    print(f"Verification logic batch {i} - LDS: {lds_indices}")
    print(f"Verification logic batch {i} - Same: {torch.equal(magic_indices, lds_indices)}")

print("\n=== CONCLUSION ===")
print("The iterator reset approach should work for individual batches.")
print("If verification function logic fails, the issue is with the loop structure.")

print("=== Testing DataLoader creation timing ===")

# Test 1: Create DataLoaders sequentially (like in verification function)
print("Test 1: Sequential DataLoader creation (verification function style)")
set_seeds(config.SEED)
magic_loader_seq = get_cifar10_dataloader(
    batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
    split='train', shuffle=True, augment=False,
    num_workers=0, root_path=config.CIFAR_ROOT
)

set_seeds(config.SEED)  # Reset seed before creating second loader
lds_loader_seq = get_cifar10_dataloader(
    batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
    split='train', 
    shuffle=True, 
    augment=False,
    num_workers=0,
    root_path=config.CIFAR_ROOT
)

# Test with iterator reset
set_seeds(config.SEED)
magic_iter_seq = iter(magic_loader_seq)
set_seeds(config.SEED)
lds_iter_seq = iter(lds_loader_seq)

magic_batch_seq = next(magic_iter_seq)[2][:5]
lds_batch_seq = next(lds_iter_seq)[2][:5]

print(f"Sequential - MAGIC: {magic_batch_seq}")
print(f"Sequential - LDS: {lds_batch_seq}")
print(f"Sequential - Same: {torch.equal(magic_batch_seq, lds_batch_seq)}")

# Test 2: Create DataLoaders with fresh seeds each time
print("\nTest 2: Fresh seed DataLoader creation")
set_seeds(config.SEED)
magic_loader_fresh = get_cifar10_dataloader(
    batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
    split='train', shuffle=True, augment=False,
    num_workers=0, root_path=config.CIFAR_ROOT
)
magic_batch_fresh = next(iter(magic_loader_fresh))[2][:5]

set_seeds(config.SEED)  # Fresh seed
lds_loader_fresh = get_cifar10_dataloader(
    batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
    split='train', 
    shuffle=True, 
    augment=False,
    num_workers=0,
    root_path=config.CIFAR_ROOT
)
lds_batch_fresh = next(iter(lds_loader_fresh))[2][:5]

print(f"Fresh seed - MAGIC: {magic_batch_fresh}")
print(f"Fresh seed - LDS: {lds_batch_fresh}")
print(f"Fresh seed - Same: {torch.equal(magic_batch_fresh, lds_batch_fresh)}")

# Test 3: Check if the issue is with the verification function's DataLoader creation
print("\nTest 3: Exact verification function DataLoader creation")

# This mimics the exact verification function logic
set_seeds(config.SEED)

# Create dataloaders with EXACT same settings as MAGIC
magic_loader_exact = get_cifar10_dataloader(
    batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
    split='train', shuffle=True, augment=False,
    num_workers=0, root_path=config.CIFAR_ROOT
)

# Reset seed and create LDS-style loader with EXACT same parameters
set_seeds(config.SEED)
lds_loader_exact = get_cifar10_dataloader(
    batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
    split='train', 
    shuffle=True, 
    augment=False,
    num_workers=0,
    root_path=config.CIFAR_ROOT
)

# Test with iterator reset (verification function approach)
set_seeds(config.SEED)
magic_iter_exact = iter(magic_loader_exact)
set_seeds(config.SEED)
lds_iter_exact = iter(lds_loader_exact)

magic_batch_exact = next(magic_iter_exact)[2][:5]
lds_batch_exact = next(lds_iter_exact)[2][:5]

print(f"Exact verification - MAGIC: {magic_batch_exact}")
print(f"Exact verification - LDS: {lds_batch_exact}")
print(f"Exact verification - Same: {torch.equal(magic_batch_exact, lds_batch_exact)}")

print("\n=== DIAGNOSIS ===")
print("If all tests fail, the issue is fundamental with DataLoader creation timing.")
print("The solution is to ensure DataLoaders are created with identical conditions.") 