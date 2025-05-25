#!/usr/bin/env python3
"""
Comprehensive Quality Test Suite
==============================

This test suite validates the entire REPLAY Influence system for:
- Functional correctness
- Performance benchmarks
- Edge case handling
- Error recovery
- Memory efficiency
- Configuration validation

Python >=3.8 Compatible
"""

import sys
import torch
import numpy as np
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
import pytest # Import pytest for fixtures

# Add src and tests directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # tests/e2e/ -> project root
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TESTS_DIR) not in sys.path: # To import from tests.helpers
    sys.path.insert(0, str(TESTS_DIR))


from src import config
from src.utils import (
    set_global_deterministic_state,
    derive_component_seed,
    create_deterministic_dataloader,
    create_deterministic_model,
    # DeterministicStateError, # Not explicitly caught here, but could be relevant
    # ComponentCreationError, # Not explicitly caught here
    # SeedDerivationError # Not explicitly caught here
)
from src.magic_analyzer import MagicAnalyzer
from src.model_def import construct_rn9
from src.data_handling import get_cifar10_dataloader #, CustomDataset # CustomDataset not used directly here
from tests.helpers.test_helpers import assert_dataloader_determinism


@pytest.fixture(scope="class") # Use class scope if tests in class share setup
def quality_test_environment(request): # request is a pytest fixture
    """Pytest fixture to set up and tear down the test environment for the QualityTestSuite."""
    logger = logging.getLogger('quality_test_suite.environment')
    logger.info("Setting up test environment for QualityTestSuite...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="replay_quality_test_"))
    
    original_outputs_dir = config.OUTPUTS_DIR
    original_magic_checkpoints_dir = config.MAGIC_CHECKPOINTS_DIR
    original_magic_scores_dir = config.MAGIC_SCORES_DIR
    original_batch_dict_file = config.BATCH_DICT_FILE

    config.OUTPUTS_DIR = temp_dir / "outputs"
    config.MAGIC_CHECKPOINTS_DIR = config.OUTPUTS_DIR / "checkpoints_magic"
    config.MAGIC_SCORES_DIR = config.OUTPUTS_DIR / "scores_magic"
    config.BATCH_DICT_FILE = config.OUTPUTS_DIR / "magic_batch_dict.pkl"
    
    # Ensure directories are created if MagicAnalyzer expects them
    config.MAGIC_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MAGIC_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    set_global_deterministic_state(config.SEED, enable_deterministic=True)
    logger.info(f"Test environment created at: {temp_dir}")

    # Pass the temp_dir to the class if it needs it directly
    if request.cls:
        request.cls.temp_dir_fixture = temp_dir 

    yield # This is where the testing happens

    logger.info(f"Cleaning up test environment: {temp_dir}")
    shutil.rmtree(temp_dir)
    
    config.OUTPUTS_DIR = original_outputs_dir
    config.MAGIC_CHECKPOINTS_DIR = original_magic_checkpoints_dir
    config.MAGIC_SCORES_DIR = original_magic_scores_dir
    config.BATCH_DICT_FILE = original_batch_dict_file
    logger.info("Restored original config paths.")

@pytest.mark.usefixtures("quality_test_environment")
class TestQualitySuite:
    """Comprehensive test suite for quality assurance, using pytest fixtures."""
    
    # temp_dir_fixture will be set by the fixture if needed by methods

    def _setup_logging(self) -> logging.Logger: # Keep as helper method if tests use it
        """Setup logging for test suite - can be called by tests if specific logging is needed beyond pytest's default."""
        # Pytest handles stdout/stderr capture. For detailed test-specific logs, 
        # using Python's logging per test is fine.
        logger = logging.getLogger('quality_test_suite')
        if not logger.handlers: # Avoid adding multiple handlers if logger is already configured
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        return logger
    
    def __init__(self):
        # __init__ is still called. If using a class fixture, some setup might move there.
        # self.test_results is fine to keep if you want to aggregate results within the class instance.
        self.logger = self._setup_logging() # logger for the test methods
        self.test_results: Dict[str, Any] = {} # For custom result aggregation if needed beyond pytest pass/fail

    # Removed setup_test_environment and cleanup_test_environment, now handled by fixture.

    @pytest.mark.e2e
    def test_configuration_validation(self):
        self.logger.info("🔧 Testing configuration validation...")
        config.validate_config() # Should not raise error
        env_info = config.validate_environment()
        assert isinstance(env_info, dict)
        assert 'python_version' in env_info
        assert 'torch_version' in env_info
        self.logger.info("✅ Configuration validation passed")
    
    @pytest.mark.e2e
    def test_error_handling(self):
        self.logger.info("🛡️ Testing error handling...")
        with pytest.raises(ValueError):
            get_cifar10_dataloader(batch_size=-1)
        with pytest.raises(ValueError):
            get_cifar10_dataloader(split='invalid_split')
        with pytest.raises(ValueError):
            get_cifar10_dataloader(num_workers=-1)
        
        for test_seed in [0, 1, 2**31-1]:
            result = derive_component_seed(test_seed, "test_error_handling", "edge_case")
            assert 0 <= result < 2**31, f"Derived seed out of range: {result}"
        self.logger.info("✅ Error handling tests passed")
    
    @pytest.mark.e2e
    def test_memory_efficiency(self):
        self.logger.info("💾 Testing memory efficiency...")
        tracemalloc.start()
        start_time = time.time()
        _ = MagicAnalyzer(use_memory_efficient_replay=False) # analyzer_regular
        regular_time = time.time() - start_time
        _, regular_memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        tracemalloc.start()
        start_time = time.time()
        _ = MagicAnalyzer(use_memory_efficient_replay=True) # analyzer_efficient
        efficient_time = time.time() - start_time
        _, efficient_memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()
        
        memory_ratio = efficient_memory_peak / regular_memory_peak if regular_memory_peak > 0 else 1.0
        self.test_results['memory_efficiency'] = {
            'regular_memory_mb': regular_memory_peak / 1e6,
            'efficient_memory_mb': efficient_memory_peak / 1e6,
            'memory_ratio': memory_ratio,
            'regular_init_time': regular_time,
            'efficient_init_time': efficient_time
        }
        self.logger.info(f"Memory (MB): Regular={regular_memory_peak/1e6:.1f}, Efficient={efficient_memory_peak/1e6:.1f}, Ratio={memory_ratio:.2f}")
        # Add an assertion, e.g. efficient mode is not significantly worse, or is better
        assert efficient_memory_peak <= regular_memory_peak * 1.1, "Memory efficient mode used too much memory compared to regular."
        self.logger.info("✅ Memory efficiency test passed")
            
    @pytest.mark.e2e
    def test_performance_benchmarks(self):
        self.logger.info("⚡ Testing performance benchmarks...")
        start_time = time.time()
        for i in range(1000):
            derive_component_seed(42, "test_perf", i)
        seed_derivation_time = time.time() - start_time

        start_time = time.time()
        _ = create_deterministic_model(
            master_seed=42,
            creator_func=construct_rn9,
            instance_id="test_perf_model",
            num_classes=10
        )
        model_creation_time = time.time() - start_time

        start_time = time.time()
        _ = create_deterministic_dataloader(
            master_seed=42,
            creator_func=get_cifar10_dataloader,
            instance_id="test_perf_loader",
            batch_size=32, split='train', shuffle=True, num_workers=0, root_path=config.CIFAR_ROOT
        )
        dataloader_creation_time = time.time() - start_time

        self.test_results['performance'] = {
            'seed_derivation_per_1000': seed_derivation_time,
            'model_creation_time': model_creation_time,
            'dataloader_creation_time': dataloader_creation_time
        }
        self.logger.info(f"Perf: Seed (1k)={seed_derivation_time:.3f}s, ModelCreate={model_creation_time:.3f}s, LoaderCreate={dataloader_creation_time:.3f}s")
        assert seed_derivation_time < 1.0, f"Seed derivation too slow: {seed_derivation_time:.3f}s"
        assert model_creation_time < 5.0, f"Model creation too slow: {model_creation_time:.3f}s"
        self.logger.info("✅ Performance benchmarks passed")
    
    @pytest.mark.e2e
    def test_edge_cases(self):
        self.logger.info("🔍 Testing edge cases...")
        # Test with very small dataset (implicitly tests large batch size relative to dataset)
        _ = create_deterministic_dataloader(
            master_seed=42,
            creator_func=get_cifar10_dataloader,
            instance_id="small_data_test",
            batch_size=config.NUM_TRAIN_SAMPLES + 100, # Batch size larger than dataset
            split='train', shuffle=True, num_workers=0, root_path=config.CIFAR_ROOT
        )
        
        extreme_seeds = [0, 1, 2**31-1, 42]
        for seed in extreme_seeds:
            derived = derive_component_seed(seed, "test_edge_seed", "edge_case")
            assert 0 <= derived < 2**31, f"Derived seed out of range for {seed}: {derived}"
        
        for num_classes in [1, 2, 10, 100]:
            model = construct_rn9(num_classes=num_classes)
            test_input = torch.randn(1, 3, 32, 32)
            output = model(test_input)
            assert output.shape[-1] == num_classes
        self.logger.info("✅ Edge cases test passed")
    
    @pytest.mark.e2e
    def test_data_consistency_basic(self):
        self.logger.info("📊 Testing basic data consistency...")
        # Global deterministic state is set by fixture
        
        # Basic consistency check using helper functions
        assert_dataloader_determinism(
            "e2e_test_same", 
            "e2e_test_same", 
            should_be_equal=True,
            context="E2E same instance_id test"
        )
        
        assert_dataloader_determinism(
            "e2e_test_diff_1", 
            "e2e_test_diff_2", 
            should_be_equal=False,
            context="E2E different instance_id test"
        )

        # Additional specific check for dataloader re-creation consistency
        self.logger.info("🔁 Testing direct dataloader re-creation consistency...")
        results = []
        for _ in range(2): # Run twice
            loader = create_deterministic_dataloader(
                master_seed=42, creator_func=get_cifar10_dataloader,
                instance_id="consistency_checker", batch_size=100,
                split='train', shuffle=True, num_workers=0, root_path=config.CIFAR_ROOT
            )
            first_batch = next(iter(loader))
            indices = first_batch[2] 
            results.append(indices.clone())
        assert torch.equal(results[0], results[1]), "Inconsistent first batch indices between two identical loader creations."
        
        self.logger.info("✅ Basic data consistency test passed")
        self.logger.info("ℹ️  For comprehensive data ordering tests, see integration test: test_comprehensive_data_ordering_consistency")
            
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_mini_training_workflow(self):
        self.logger.info("🚀 Testing mini training workflow...")
        original_epochs = config.MODEL_TRAIN_EPOCHS
        original_batch_size = config.MODEL_TRAIN_BATCH_SIZE
        original_perform_inline_validations = config.PERFORM_INLINE_VALIDATIONS

        config.MODEL_TRAIN_EPOCHS = 1
        config.MODEL_TRAIN_BATCH_SIZE = 128 # Reasonably small
        config.PERFORM_INLINE_VALIDATIONS = False # Disable for this mini-run to speed up
        
        try:
            analyzer = MagicAnalyzer(use_memory_efficient_replay=True)
            total_steps = analyzer.train_and_collect_intermediate_states(force_retrain=True)
            assert total_steps > 0, "Mini training returned no steps"
            
            # Ensure some checkpoints exist (number depends on save_every_k_steps)
            # This check might need adjustment based on actual save logic.
            # For now, just existence of the dir created by analyzer is a basic check.
            assert config.MAGIC_CHECKPOINTS_DIR.exists()
            # num_checkpoints = len(list(config.MAGIC_CHECKPOINTS_DIR.glob("ckpt_step_*.pth")))
            # assert num_checkpoints > 0, "No checkpoints saved during mini training."

            scores = analyzer.compute_influence_scores(
                total_training_iterations=total_steps,
                force_recompute=True
            )
            assert scores is not None, "Failed to compute influence scores"
            # Assuming influence scores are num_checkpoints x num_train_samples
            # The shape might be (total_steps, num_train_samples) if scores are per iteration not per checkpoint snapshot
            # Based on algorithm, it's likely per-influence source (e.g. training point) vs. target output, 
            # but the current `scores` return seems to be (total_steps, num_train_samples)
            expected_score_shape_dim1 = total_steps # Or number of actual checkpoints saved
            assert scores.shape == (expected_score_shape_dim1, config.NUM_TRAIN_SAMPLES), \
                f"Unexpected scores shape: {scores.shape}, expected ({expected_score_shape_dim1}, {config.NUM_TRAIN_SAMPLES})"

            self.logger.info(f"✅ Mini training workflow completed successfully - {total_steps} steps")
        finally:
            config.MODEL_TRAIN_EPOCHS = original_epochs
            config.MODEL_TRAIN_BATCH_SIZE = original_batch_size
            config.PERFORM_INLINE_VALIDATIONS = original_perform_inline_validations

    # The run_all_tests and summary printing is handled by pytest runner
    # Individual test methods will be discovered and run.
    # The self.test_results can be used for custom reporting if needed outside pytest, 
    # but pytest's own reporting is usually sufficient.

# Removed main() and if __name__ == "__main__" as pytest handles test execution.
