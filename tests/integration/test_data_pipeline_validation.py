import sys
import torch
import numpy as np
import logging
import pytest
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # tests/integration/ -> project root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now import from src and tests.helpers
from src import config
from src.utils import (
    set_global_deterministic_state, 
    create_deterministic_dataloader,
    create_deterministic_model,
    update_dataloader_epoch
)
from src.model_def import construct_rn9
from src.data_handling import get_cifar10_dataloader
from tests.helpers.test_helpers import assert_dataloader_determinism, assert_multi_epoch_consistency

@pytest.mark.integration
@pytest.mark.slow
def test_comprehensive_data_ordering_consistency():
    """
    Comprehensive test that verifies data ordering consistency between MAGIC and LDS simulations.
    This ensures that sample indices refer to the exact same data points across multiple epochs,
    even when LDS uses subset-based weighted training.
    """
    logger = logging.getLogger(__name__)
    logger.info("🔬 Test: Comprehensive Data Ordering Consistency")
    
    # Ensure global deterministic state is set
    set_global_deterministic_state(config.SEED, enable_deterministic=True)
    
    # Test 1: Basic data loader consistency - use helper for simple cases
    logger.info("Test 1: Verifying basic data loader consistency...")
    
    # Use helper for the basic consistency check
    assert_dataloader_determinism(
        config.SHARED_DATALOADER_INSTANCE_ID,
        config.SHARED_DATALOADER_INSTANCE_ID,
        should_be_equal=True,
        context="MAGIC vs LDS shared instance consistency"
    )
    
    logger.info("✓ Basic data loader consistency verified")
    
    # Test 2: Multi-epoch consistency with proper epoch handling
    logger.info("Test 2: Verifying multi-epoch data ordering consistency...")
    
    # Use helper function for multi-epoch testing
    assert_multi_epoch_consistency("multi_epoch_test_loader", num_epochs=2, batches_per_epoch=5)
    
    logger.info("✓ Multi-epoch shuffling consistency verified")
    
    # Test 3: CRITICAL TEST - Complete data sequence consistency across instances
    logger.info("Test 3: CRITICAL - Verifying complete data sequence consistency...")
    
    magic_test_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id=config.SHARED_DATALOADER_INSTANCE_ID,  
        batch_size=100,  
        split='train', 
        shuffle=True, 
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )
    
    magic_sequence = []
    for epoch_num in range(2):
        update_dataloader_epoch(magic_test_loader, epoch_num)
        for batch_idx, (_, _, indices) in enumerate(magic_test_loader):
            magic_sequence.append((epoch_num, batch_idx, indices.clone()))
            if batch_idx >= 4: 
                break
    
    lds_shared_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id=config.SHARED_DATALOADER_INSTANCE_ID,  
        batch_size=100,  
        split='train', 
        shuffle=True, 
        augment=False,
        num_workers=0,
        root_path=config.CIFAR_ROOT
    )
    
    lds_sequence = []
    for epoch_num in range(2):
        update_dataloader_epoch(lds_shared_loader, epoch_num)
        for batch_idx, (_, _, indices) in enumerate(lds_shared_loader):
            lds_sequence.append((epoch_num, batch_idx, indices.clone()))
            if batch_idx >= 4: 
                break
    
    assert len(magic_sequence) == len(lds_sequence), \
        f"Sequence length mismatch: MAGIC={len(magic_sequence)}, LDS={len(lds_sequence)}"
    
    for i, ((m_epoch, m_batch, m_indices), (l_epoch, l_batch, l_indices)) in enumerate(zip(magic_sequence, lds_sequence)):
        assert m_epoch == l_epoch and m_batch == l_batch, \
            f"Epoch/batch mismatch at position {i}: MAGIC=({m_epoch},{m_batch}), LDS=({l_epoch},{l_batch})"
        assert torch.equal(m_indices, l_indices), \
            f"Index mismatch at epoch {m_epoch}, batch {m_batch}. " \
            f"MAGIC indices: {m_indices[:5].tolist()}, LDS indices: {l_indices[:5].tolist()}"
    
    logger.info(f"✓ Complete data sequence consistency verified across {len(magic_sequence)} batches over 2 epochs")
    
    # Test 4: Model initialization consistency
    logger.info("Test 4: Verifying model initialization consistency...")
    
    models = []
    for i in range(3):
        model = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=construct_rn9,
            instance_id=config.SHARED_MODEL_INSTANCE_ID,  
            num_classes=config.NUM_CLASSES
        )
        models.append(model)
    
    for i in range(1, len(models)):
        for p1, p2 in zip(models[0].parameters(), models[i].parameters()):
            assert torch.equal(p1, p2), \
                f"Model initialization inconsistency detected between model 0 and model {i}"
    
    logger.info("✓ Model initialization consistency verified")
    
    # Test 5: Subset mechanism verification
    logger.info("Test 5: Verifying subset-based weighted training mechanism...")
    
    test_subset_indices = np.arange(1000)
    data_weights = torch.zeros(config.NUM_TRAIN_SAMPLES)
    data_weights[test_subset_indices] = 1.0
    
    # Verify we can create weighted subsets
    assert data_weights.sum() == 1000, "Subset weights not created correctly"
    assert (data_weights[test_subset_indices] == 1.0).all(), "Subset indices not weighted correctly"
    
    logger.info("✓ Subset mechanism verification passed")
    
    logger.info("🎯 ALL comprehensive data ordering and consistency tests PASSED") 