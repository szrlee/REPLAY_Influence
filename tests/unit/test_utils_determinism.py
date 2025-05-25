import sys
import torch
import logging
import pytest
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # tests/unit/ -> project root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import config # Though config.SEED is used, derive_component_seed does not directly use it.
from src.utils import (
    derive_component_seed,
    deterministic_context
)

# It's good practice for tests to configure their own logging or use pytest's capabilities
# For simplicity here, we'll allow them to use the logger they define.

@pytest.mark.unit
def test_component_seed_derivation(): # Renamed for pytest convention (test_ prefix)
    """Test 1: Verify component-specific seed derivation works correctly"""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Test: Component-Specific Seed Derivation")
    
    master_seed = 42
    
    model_seed_1 = derive_component_seed(master_seed, "model", "instance1")
    model_seed_2 = derive_component_seed(master_seed, "model", "instance2") 
    optimizer_seed = derive_component_seed(master_seed, "optimizer", "instance1")
    dataloader_seed = derive_component_seed(master_seed, "dataloader", "instance1")
    
    model_seed_1_repeat = derive_component_seed(master_seed, "model", "instance1")
    
    assert model_seed_1 == model_seed_1_repeat, "Seed derivation not deterministic!"
    assert model_seed_1 != model_seed_2, "Different instances should have different seeds!"
    assert model_seed_1 != optimizer_seed, "Different components should have different seeds!"
    assert model_seed_1 != dataloader_seed, "Different components should have different seeds!"
    
    logger.info(f"✅ Component seeds: model1={model_seed_1}, model2={model_seed_2}, opt={optimizer_seed}, data={dataloader_seed}")
    logger.info("✅ PASSED: Component-specific seed derivation working correctly")
    # Pytest doesn't need explicit True return

@pytest.mark.unit
def test_deterministic_context_consistency(): # Renamed for pytest convention
    """Test 2: Verify deterministic context preserves state properly"""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Test: Deterministic Context State Preservation")
    
    torch.manual_seed(123)
    # initial_state = torch.get_rng_state() # Not strictly needed for this test's logic
    
    # Generate some random numbers to change state
    torch.randn(10)
    state_before_context_after_initial_ops = torch.get_rng_state()
    
    component_seed = derive_component_seed(42, "test_component", "test_instance")
    with deterministic_context(component_seed, "test operation"):
        _ = torch.randn(5) # context_numbers, variable not used
    
    state_after_context = torch.get_rng_state()
    assert torch.equal(state_before_context_after_initial_ops, state_after_context), "Global RNG state not properly restored after deterministic_context!"
        
    logger.info("✅ PASSED: Deterministic context preserves state correctly") 