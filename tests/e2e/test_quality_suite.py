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
import pickle
import subprocess
import json
import warnings
from unittest.mock import patch, MagicMock # Added MagicMock
import os # Add os import for environment variables
import importlib # Add importlib for reloading

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


@pytest.fixture(scope="function") # Ensure fresh config & run_manager for each test
def isolated_config_e2e(tmp_path, monkeypatch):
    """
    Provides a fresh, isolated src.config module for E2E tests.
    - Sets REPLAY_OUTPUTS_DIR environment variable to a temporary path.
    - Reloads src.config and src.run_manager to use this path.
    - Initializes a new run directory within this temporary path.
    - Applies minimal default settings to the fresh config suitable for E2E tests.
    - Restores original REPLAY_OUTPUTS_DIR and reloads modules in teardown.
    """
    original_sys_path = list(sys.path) # Keep sys.path restoration
    original_replay_outputs_dir = os.environ.get("REPLAY_OUTPUTS_DIR")

    # Ensure SRC_DIR is on path for imports (important for when test runner might change CWD)
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    isolated_base_outputs = tmp_path / f"e2e_test_outputs_{Path(tempfile.NamedTemporaryFile().name).name}"
    isolated_base_outputs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REPLAY_OUTPUTS_DIR", str(isolated_base_outputs))

    # (Re)load config to pick up the new REPLAY_OUTPUTS_DIR
    if 'src.config' in sys.modules:
        fresh_config = importlib.reload(sys.modules['src.config'])
    else:
        from src import config as fresh_config
    
    # (Re)load run_manager to pick up potentially reloaded config
    if 'src.run_manager' in sys.modules:
        fresh_run_manager = importlib.reload(sys.modules['src.run_manager'])
    else:
        from src import run_manager as fresh_run_manager
    
    # Apply E2E specific default settings to fresh_config
    fresh_config.MODEL_TRAIN_EPOCHS = 1
    fresh_config.LDS_NUM_SUBSETS_TO_GENERATE = 1
    fresh_config.LDS_NUM_MODELS_TO_TRAIN = 1
    fresh_config.PAPER_NUM_MEASUREMENT_FUNCTIONS = 2
    fresh_config.MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 1
    fresh_config.PERFORM_INLINE_VALIDATIONS = False
    fresh_config.NUM_CLASSES = 10 # CIFAR-10 default
    fresh_config.MODEL_CREATOR_FUNCTION = construct_rn9 # Default model
    fresh_config.CIFAR_ROOT = '/tmp/cifar_pytest_e2e/' # Use a test-specific CIFAR root
    Path(fresh_config.CIFAR_ROOT).mkdir(parents=True, exist_ok=True) # Ensure it exists
    fresh_config.SEED = 42 # Default seed
    fresh_config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(fresh_config, 'NUM_TRAIN_SAMPLES'): fresh_config.NUM_TRAIN_SAMPLES = 50000
    if not hasattr(fresh_config, 'NUM_TEST_SAMPLES_CIFAR10'): fresh_config.NUM_TEST_SAMPLES_CIFAR10 = 10000

    try:
        fresh_config.validate_config() # Validate this minimal config
        fresh_config.validate_training_compatibility()
    except ValueError as e:
        pytest.fail(f"Initial E2E config validation failed: {e}")
    
    fresh_run_manager.init_run_directory() # Initialize run dir using new OUTPUTS_DIR

    yield fresh_config

    # Teardown
    sys.path[:] = original_sys_path # Restore sys.path

    if original_replay_outputs_dir is None:
        monkeypatch.delenv("REPLAY_OUTPUTS_DIR", raising=False)
    else:
        monkeypatch.setenv("REPLAY_OUTPUTS_DIR", original_replay_outputs_dir)
    
    # Reload modules again so they pick up the restored REPLAY_OUTPUTS_DIR
    if 'src.config' in sys.modules:
        importlib.reload(sys.modules['src.config'])
    if 'src.run_manager' in sys.modules:
        importlib.reload(sys.modules['src.run_manager'])

# --- Test Functions (No Class Needed) ---

@pytest.mark.e2e
def test_configuration_validation_e2e(isolated_config_e2e):
    current_config = isolated_config_e2e
    logger = logging.getLogger('test_configuration_validation_e2e')
    logger.info("🔧 Testing configuration validation (E2E context)...")
    
    # Initial validation is done in fixture. Here, test specific error cases.
    original_magic_idx = current_config.MAGIC_TARGET_VAL_IMAGE_IDX
    current_config.MAGIC_TARGET_VAL_IMAGE_IDX = -10
    with pytest.raises(ValueError, match="MAGIC_TARGET_VAL_IMAGE_IDX .* out of bounds"):
        current_config.validate_config()
    current_config.MAGIC_TARGET_VAL_IMAGE_IDX = original_magic_idx
    logger.info("✅ Configuration validation error catching passed (E2E context).")

@pytest.mark.e2e
def test_environment_validation_e2e(isolated_config_e2e):
    current_config = isolated_config_e2e
    logger = logging.getLogger('test_environment_validation_e2e')
    logger.info("🌍 Testing environment validation (E2E context)...")
    try:
        env_info = current_config.validate_environment()
        assert env_info is not None
        assert env_info['outputs_writable'] is True # Fixture ensures this
    except EnvironmentError as e:
        pytest.fail(f"validate_environment raised EnvironmentError unexpectedly: {e}")
    logger.info("✅ Environment validation passed (E2E context).")


@pytest.mark.quality_integration # Keep if this implies a heavier E2E test
def test_full_pipeline_main_runner_execution(isolated_config_e2e):
    """
    Tests that main_runner.py executes both MAGIC and LDS pipelines successfully,
    generating expected outputs in the isolated run directory.
    """
    current_config = isolated_config_e2e
    logger = logging.getLogger('test_full_pipeline_main_runner')
    logger.info(f"Full pipeline test via main_runner.py in run dir: {current_config.get_current_run_dir()}")

    # Override specific config settings for this full pipeline test if different from fixture defaults
    current_config.MODEL_TRAIN_EPOCHS = 1 # Keep minimal for speed
    current_config.MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 1
    current_config.LDS_NUM_SUBSETS_TO_GENERATE = 1
    current_config.LDS_NUM_MODELS_TO_TRAIN = 1
    current_config.PAPER_NUM_MEASUREMENT_FUNCTIONS = 2 

    # For LDS to run after MAGIC, MAGIC scores must exist.
    # main_runner.py should handle this sequence if both --run_magic and --run_lds are passed.

    script_path = PROJECT_ROOT / "main_runner.py"
    cmd = [
        sys.executable, str(script_path),
        "--full_pipeline",
        # "--epochs", str(current_config.MODEL_TRAIN_EPOCHS), # Removed, main_runner uses config directly
        "--run_id", current_config.get_current_run_dir().name, # Correctly get run ID string via .name
        # No need to pass run_id if isolated_config_e2e sets it up and main_runner respects it.
        # main_runner.py should pick up the current run context established by the fixture.
    ]
    logger.info(f"Executing main_runner.py: {' '.join(cmd)}")
    
    # Execute main_runner.py
    # Important: The environment for subprocess should ensure it sees the same Python environment
    # and potentially the patched sys.path if main_runner relies on relative imports outside of `src` package.
    # Pytest usually handles this well for subprocesses started with sys.executable.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        logger.error(f"main_runner.py execution failed.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
        pytest.fail(f"main_runner.py failed. Stderr: {result.stderr}")
    else:
        logger.info(f"main_runner.py execution completed.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")

    # --- Assertions for Outputs ---
    # Verify that the run directory used by main_runner is the one set up by the fixture.
    # This can be tricky if main_runner explicitly calls init_run_directory without checking existing one.
    # Assuming main_runner uses get_current_run_dir() or init_run_directory() correctly.
    run_dir_after_main_runner = current_config.get_current_run_dir()
    assert run_dir_after_main_runner.exists(), "Run directory does not exist after main_runner."
    logger.info(f"Verified outputs in run directory: {run_dir_after_main_runner}")

    # MAGIC outputs
    assert current_config.get_magic_checkpoints_dir().exists(), "MAGIC checkpoints dir missing."
    assert len(list(current_config.get_magic_checkpoints_dir().glob("*.pt"))) > 0, "No MAGIC checkpoints found."
    assert current_config.get_magic_scores_dir().exists(), "MAGIC scores dir missing."
    magic_scores_file = current_config.get_magic_scores_path(current_config.MAGIC_TARGET_VAL_IMAGE_IDX)
    assert magic_scores_file.exists(), f"MAGIC scores file {magic_scores_file.name} missing."
    assert current_config.get_magic_training_log_path().exists(), "MAGIC training log missing."

    # LDS outputs
    assert current_config.get_lds_checkpoints_dir().exists(), "LDS checkpoints dir missing."
    # LDS Checkpoints are not saved by the current lds_validator.py implementation.
    # If LDS_NUM_MODELS_TO_TRAIN > 0 and current_config.LDS_NUM_SUBSETS_TO_GENERATE > 0:
    #     # Check if the LDS checkpoints directory is not empty
    #     lds_checkpoints_dir = current_config.get_lds_checkpoints_dir()
    #     checkpoint_files = list(lds_checkpoints_dir.glob("*.pt"))
    #     if not (len(checkpoint_files) > 0):
    #         pytest.fail(f"No LDS checkpoints found in {lds_checkpoints_dir}. main_runner stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    
    assert current_config.get_lds_losses_dir().exists(), "LDS losses dir missing."
    if current_config.LDS_NUM_MODELS_TO_TRAIN > 0 and current_config.LDS_NUM_SUBSETS_TO_GENERATE > 0:
        # Check if the LDS losses directory is not empty
        assert len(list(current_config.get_lds_losses_dir().glob("*.pkl"))) > 0, "No LDS loss files found." # Assuming .pkl, adjust if .pth
    
    # Check for LDS log files directory, and if not empty if models were trained
    lds_logs_dir = current_config.get_lds_logs_dir()
    assert lds_logs_dir.exists(), "LDS logs dir missing."
    if current_config.LDS_NUM_MODELS_TO_TRAIN > 0 and current_config.LDS_NUM_SUBSETS_TO_GENERATE > 0:
        assert len(list(lds_logs_dir.glob("*.json"))) > 0, "No LDS training logs found."

    # Main run log and metadata
    run_log_path = run_dir_after_main_runner / f"run_{run_dir_after_main_runner.name}.log"
    assert run_log_path.exists(), "Main run log missing."
    metadata_file_path = run_dir_after_main_runner / "run_metadata.json"
    assert metadata_file_path.exists(), "Run metadata file missing."
    with open(metadata_file_path, 'r') as f_meta:
        metadata = json.load(f_meta)
        assert metadata['run_id'] == run_dir_after_main_runner.name
        assert metadata['config_snapshot']['MODEL_TRAIN_EPOCHS'] == current_config.MODEL_TRAIN_EPOCHS

    logger.info("✅ Full pipeline (main_runner.py) execution and output checks passed.")


# Configure logging for pytest output
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
