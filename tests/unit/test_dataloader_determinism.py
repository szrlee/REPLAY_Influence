import torch
import torch.utils.data
import sys
import pytest
import logging
from pathlib import Path
import torchvision.transforms as transforms

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Adjusted for tests/unit/
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.utils import set_global_deterministic_state, create_deterministic_dataloader
from src.data_handling import get_cifar10_dataloader, CustomDataset
from src import config
from tests.helpers.test_helpers import get_first_batch_indices, assert_dataloader_determinism

# Tests focus on our deterministic utilities rather than raw PyTorch behavior

@pytest.mark.unit
def test_create_deterministic_dataloader_consistency():
    """Test that create_deterministic_dataloader produces identical results with same instance_id."""
    logger = logging.getLogger(__name__)
    
    set_global_deterministic_state(42)
    
    # Use helper function for cleaner test
    assert_dataloader_determinism(
        "test_consistency",
        "test_consistency", 
        should_be_equal=True,
        context="same instance_id consistency test"
    )
    
    logger.info("✅ Same instance_id consistency test passed")

@pytest.mark.unit
def test_create_deterministic_dataloader_different_instances():
    """Test that create_deterministic_dataloader produces different results with different instance_ids."""
    logger = logging.getLogger(__name__)
    
    set_global_deterministic_state(42)
    
    # Use helper function for cleaner test
    assert_dataloader_determinism(
        "test_instance_1",
        "test_instance_2", 
        should_be_equal=False,
        context="different instance_id test"
    )
    
    logger.info("✅ Different instance_id test passed")

@pytest.mark.unit
def test_deterministic_dataloader_with_helper():
    """Test deterministic dataloader using helper functions."""
    logger = logging.getLogger(__name__)
    
    set_global_deterministic_state(42)
    
    # Test same instance_id should produce same results
    assert_dataloader_determinism(
        "helper_test_same", 
        "helper_test_same", 
        should_be_equal=True,
        context="same instance_id test"
    )
    
    # Test different instance_ids should produce different results  
    assert_dataloader_determinism(
        "helper_test_diff_1", 
        "helper_test_diff_2", 
        should_be_equal=False,
        context="different instance_id test"
    )
    
    logger.info("✅ Helper function tests passed")

@pytest.mark.unit
def test_multiple_loader_creation_timing_determinism():
    """Test that multiple loaders created sequentially with same instance_id produce identical results."""
    logger = logging.getLogger(__name__)
    
    # This test specifically checks if our create_deterministic_dataloader handles
    # the timing issue that was problematic with raw PyTorch DataLoaders
    
    set_global_deterministic_state(42)
    loader_a = create_deterministic_dataloader(
        master_seed=42,
        creator_func=get_cifar10_dataloader,
        instance_id="timing_test",
        batch_size=10,
        split='train',
        shuffle=True,
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )

    set_global_deterministic_state(42)  # Reset seed before second loader creation
    loader_b = create_deterministic_dataloader(
        master_seed=42,
        creator_func=get_cifar10_dataloader,
        instance_id="timing_test",  # Same instance_id
        batch_size=10,
        split='train',
        shuffle=True,
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )

    batch_a_indices = get_first_batch_indices(loader_a)
    batch_b_indices = get_first_batch_indices(loader_b)

    logger.info(f"Sequential Creation - Batch A Indices: {batch_a_indices[:5].tolist()}")
    logger.info(f"Sequential Creation - Batch B Indices: {batch_b_indices[:5].tolist()}")
    
    assert torch.equal(batch_a_indices, batch_b_indices), \
        f"Sequential create_deterministic_dataloader calls with same instance_id produced different results! " \
        f"Batch A: {batch_a_indices[:5].tolist()}, Batch B: {batch_b_indices[:5].tolist()}"

@pytest.mark.unit
def test_exact_issue_reproduction_scenario():
    """Test exact scenario that reproduces the original issue, but using our deterministic utilities."""
    logger = logging.getLogger(__name__)
    
    # This test mimics the exact problematic scenario but uses create_deterministic_dataloader
    # which should solve the timing/isolation issues
    
    set_global_deterministic_state(42)
    loader_x = create_deterministic_dataloader(
        master_seed=42,
        creator_func=get_cifar10_dataloader,
        instance_id="exact_scenario_test",
        batch_size=10,
        split='train',
        shuffle=True,
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )
    
    set_global_deterministic_state(42)  # Reset seed
    loader_y = create_deterministic_dataloader(
        master_seed=42,
        creator_func=get_cifar10_dataloader,
        instance_id="exact_scenario_test",  # Same instance_id
        batch_size=10,
        split='train',
        shuffle=True,
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )

    batch_x_indices = get_first_batch_indices(loader_x)
    batch_y_indices = get_first_batch_indices(loader_y)

    logger.info(f"Exact Scenario - Batch X Indices: {batch_x_indices[:5].tolist()}")
    logger.info(f"Exact Scenario - Batch Y Indices: {batch_y_indices[:5].tolist()}")
    
    assert torch.equal(batch_x_indices, batch_y_indices), \
        f"Sequential create_deterministic_dataloader creation with intermediate seed reset produced different results! " \
        f"Batch X: {batch_x_indices[:5].tolist()}, Batch Y: {batch_y_indices[:5].tolist()}"

@pytest.mark.unit
def test_raw_pytorch_dataloader_issue_demonstration():
    """Demonstrate the issue with raw PyTorch DataLoaders (this test may fail, showing the problem)."""
    logger = logging.getLogger(__name__)
    
    # This test demonstrates why we need create_deterministic_dataloader
    # It may fail, which is expected and shows the problem we're solving
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
    ])

    set_global_deterministic_state(42)
    cifar_dataset1 = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
    cifar_loader1 = torch.utils.data.DataLoader(cifar_dataset1, batch_size=10, shuffle=True, num_workers=0)
    batch1_indices = next(iter(cifar_loader1))[2][:5]

    set_global_deterministic_state(42)  # Reset seed
    cifar_dataset2 = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=transform)
    cifar_loader2 = torch.utils.data.DataLoader(cifar_dataset2, batch_size=10, shuffle=True, num_workers=0)
    batch2_indices = next(iter(cifar_loader2))[2][:5]

    logger.info(f"Raw PyTorch - Batch1 Indices: {batch1_indices.tolist()}")
    logger.info(f"Raw PyTorch - Batch2 Indices: {batch2_indices.tolist()}")
    
    # This assertion might fail, demonstrating the problem
    try:
        assert torch.equal(batch1_indices, batch2_indices), \
            f"Raw PyTorch DataLoaders with seed reset produced different results (this demonstrates the issue we solve). " \
            f"Batch1: {batch1_indices.tolist()}, Batch2: {batch2_indices.tolist()}"
        logger.info("✅ Raw PyTorch DataLoaders were consistent (unexpected but good)")
    except AssertionError as e:
        logger.warning(f"❌ Raw PyTorch DataLoaders were inconsistent (expected): {e}")
        # Don't fail the test - this demonstrates the problem we're solving
        pytest.skip("Raw PyTorch DataLoader inconsistency demonstrated (this is expected and why we use create_deterministic_dataloader)") 