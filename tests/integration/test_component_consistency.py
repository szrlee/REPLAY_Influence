import sys
import torch
import logging
import pytest
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # tests/integration/ -> project root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import config
from src.utils import (
    set_global_deterministic_state,
    create_deterministic_model, 
    create_deterministic_optimizer
)
from src.model_def import construct_rn9

@pytest.mark.integration
def test_model_initialization_consistency(): # Renamed from test_4_...
    """Verify model initialization consistency with same and different instance_ids."""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Test: Model Initialization Consistency")
    
    set_global_deterministic_state(config.SEED, enable_deterministic=True)
    
    magic_model = create_deterministic_model(
        master_seed=config.SEED,
        creator_func=construct_rn9,
        instance_id="shared_training_model", 
        num_classes=config.NUM_CLASSES
    )
    
    lds_model_1 = create_deterministic_model(
        master_seed=config.SEED,
        creator_func=construct_rn9,
        instance_id="shared_training_model", 
        num_classes=config.NUM_CLASSES
    )
    
    different_model = create_deterministic_model(
        master_seed=config.SEED,
        creator_func=construct_rn9,
        instance_id="different_training_model", 
        num_classes=config.NUM_CLASSES
    )
    
    magic_params = torch.cat([p.flatten() for p in magic_model.parameters()])
    lds_1_params = torch.cat([p.flatten() for p in lds_model_1.parameters()])
    different_params = torch.cat([p.flatten() for p in different_model.parameters()])
    
    assert torch.allclose(magic_params, lds_1_params), "Models with same instance_id should have IDENTICAL initializations!"
    assert not torch.allclose(magic_params, different_params), "Models with different instance_ids should have DIFFERENT initializations!"
    
    logger.info("✅ PASSED: Model initialization consistency verified.")

@pytest.mark.integration
def test_optimizer_consistency(): # Renamed from test_5_...
    """Verify optimizer creation consistency."""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Test: Optimizer Consistency")
    
    set_global_deterministic_state(config.SEED, enable_deterministic=True)
    
    # Create a dummy model for optimizer parameters
    model1 = create_deterministic_model(master_seed=config.SEED, creator_func=construct_rn9, instance_id="opt_test_model1", num_classes=config.NUM_CLASSES)
    model2 = create_deterministic_model(master_seed=config.SEED, creator_func=construct_rn9, instance_id="opt_test_model2", num_classes=config.NUM_CLASSES) # Potentially different instance

    # Optimizer for shared_training instance
    optimizer1 = create_deterministic_optimizer(
        master_seed=config.SEED,
        optimizer_class=torch.optim.SGD,
        model_params=model1.parameters(),
        instance_id="shared_training_optimizer",
        lr=config.MODEL_TRAIN_LR,
        momentum=config.MODEL_TRAIN_MOMENTUM,
        weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
    )
    
    # Optimizer for the same shared_training instance (should be identical state if re-created, though optimizers are stateful after .step())
    # The test here is more about consistent parameter group setup from create_deterministic_optimizer given same inputs.
    optimizer2 = create_deterministic_optimizer(
        master_seed=config.SEED,
        optimizer_class=torch.optim.SGD,
        model_params=model1.parameters(), # Same model parameters for fair comparison of optimizer state IF it were stateless from this func
        instance_id="shared_training_optimizer",
        lr=config.MODEL_TRAIN_LR,
        momentum=config.MODEL_TRAIN_MOMENTUM,
        weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
    )

    # Optimizer for a different instance_id
    optimizer3 = create_deterministic_optimizer(
        master_seed=config.SEED,
        optimizer_class=torch.optim.SGD,
        model_params=model2.parameters(), # Potentially different model
        instance_id="different_training_optimizer",
        lr=config.MODEL_TRAIN_LR,
        momentum=config.MODEL_TRAIN_MOMENTUM,
        weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
    )
    
    # Check hyperparameter groups are identical for optimizers from same instance_id config
    assert optimizer1.param_groups[0]['lr'] == optimizer2.param_groups[0]['lr']
    assert optimizer1.param_groups[0]['momentum'] == optimizer2.param_groups[0]['momentum']
    assert optimizer1.param_groups[0]['weight_decay'] == optimizer2.param_groups[0]['weight_decay']

    # Note: Optimizers are stateful (e.g. momentum buffers). 
    # create_deterministic_optimizer primarily ensures consistent hyperparameter setup and seed for internal RNG if any.
    # True state identity would only hold if no .step() calls were made. This test focuses on consistent creation settings.
    # If optimizers had internal random state initialized purely from their component seed, their states *before first step* might be comparable.
    # For SGD, the main check is that param groups are set up identically. Advanced optimizers like Adam might have more state to check.

    # Check that different instance_id optimizers also have the same hyperparams (as they are from config)
    assert optimizer1.param_groups[0]['lr'] == optimizer3.param_groups[0]['lr']

    logger.info("✅ PASSED: Optimizer creation consistency verified (hyperparameters). Note: Full state identity depends on usage.") 