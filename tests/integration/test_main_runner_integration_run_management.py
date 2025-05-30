import pytest
import subprocess
import shutil
from pathlib import Path
import os
import json
import time
import torch
import pickle
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config to access _update_runs_registry
from src import config as project_config
from src import run_manager as rm

MAIN_RUNNER_SCRIPT = PROJECT_ROOT / "main_runner.py"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_BASE_DIR = OUTPUTS_DIR / "runs"
REGISTRY_FILE = OUTPUTS_DIR / "runs_registry.json"
LATEST_LINK = OUTPUTS_DIR / "latest"

# Define a marker for tests that modify the filesystem significantly
# and might take longer.
# integration_test = pytest.mark.integration # Old way

@pytest.fixture(scope="function", autouse=True)
def cleanup_runs_before_after_each_test():
    """Cleans up the runs directory, registry, and latest symlink before and after each test."""
    # Before test - more comprehensive cleanup
    if RUNS_BASE_DIR.exists():
        shutil.rmtree(RUNS_BASE_DIR)
    if REGISTRY_FILE.exists():
        REGISTRY_FILE.unlink()
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()
    
    # Create fresh outputs directory structure
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    yield

    # After test
    if RUNS_BASE_DIR.exists():
        shutil.rmtree(RUNS_BASE_DIR)
    if REGISTRY_FILE.exists():
        REGISTRY_FILE.unlink()
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()

def run_main_script(args_list):
    """Helper to run the main_runner.py script with arguments."""
    command = ["python", str(MAIN_RUNNER_SCRIPT)] + args_list
    print(f"\nExecuting: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    return result

def create_dummy_run_artifacts(
    run_id: str,
    create_checkpoints: bool = False,
    create_batch_dict: bool = False,
    create_scores: bool = False,
    create_logs: bool = False,
    create_losses: bool = False,
    simulate_memory_efficient_batch_dict: bool = True
) -> Path:
    """Create dummy run artifacts for testing purposes."""
    # Update registry first
    rm._update_runs_registry(run_id, "created")
    
    run_dir = RUNS_BASE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run_metadata.json
    with open(run_dir / "run_metadata.json", 'w') as f:
        json.dump({"status": "created", "action": "test"}, f)
    
    # Create all necessary subdirectories to prevent issues
    (run_dir / "checkpoints_magic").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints_lds").mkdir(parents=True, exist_ok=True)
    (run_dir / "scores_magic").mkdir(parents=True, exist_ok=True)
    (run_dir / "losses_lds").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs_magic").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs_lds").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots_magic").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots_lds").mkdir(parents=True, exist_ok=True)
    
    if create_checkpoints:
        cp_magic_dir = run_dir / "checkpoints_magic"
        cp_magic_dir.mkdir(parents=True, exist_ok=True)
        # Create a proper PyTorch checkpoint with ResNet-9 state dict
        # We need to create a dummy state dict that matches the model structure
        # For simplicity, we'll create an empty tensor checkpoint
        # The actual model will be created dynamically
        from src.model_def import construct_rn9
        model = construct_rn9(num_classes=10)  # CIFAR-10 has 10 classes
        
        # Save initial checkpoint (step 0) - needed for replay
        torch.save(model.state_dict(), cp_magic_dir / "sd_0_0.pt")
        
        # Save checkpoints for the test steps (0, 1, 2, 3)
        for step in [0, 1, 2, 3]:
            torch.save(model.state_dict(), cp_magic_dir / f"sd_0_{step}.pt")
        
        cp_lds_dir = run_dir / "checkpoints_lds"
        cp_lds_dir.mkdir(parents=True, exist_ok=True)
        torch.save({}, cp_lds_dir / "sd_lds_0_10.pt")
    
    if create_scores:
        scores_dir = run_dir / "scores_magic"
        scores_dir.mkdir(parents=True, exist_ok=True)
        # Save as a numpy array directly, not a dict
        dummy_scores_array = np.array([0.1, 0.2, 0.3] * (project_config.NUM_TRAIN_SAMPLES // 3) + [0.1] * (project_config.NUM_TRAIN_SAMPLES % 3))
        dummy_scores_array = dummy_scores_array[:project_config.NUM_TRAIN_SAMPLES] # Ensure correct length
        with open(scores_dir / f"magic_scores_val_{project_config.MAGIC_TARGET_VAL_IMAGE_IDX}.pkl", 'wb') as f:
            pickle.dump(dummy_scores_array, f)
    
    if create_batch_dict:
        batch_dict_for_pickle = {}
        batch_data_save_dir = run_dir / "checkpoints_magic"
        if simulate_memory_efficient_batch_dict:
            batch_data_save_dir.mkdir(parents=True, exist_ok=True)

        batch_size = project_config.MODEL_TRAIN_BATCH_SIZE
        
        for step in [1, 2, 3]:  # Steps 1, 2, 3 for replay to work
            # Full batch data content
            current_batch_content = {
                'ims': torch.randn(batch_size, 3, 32, 32),
                'labs': torch.randint(0, 10, (batch_size,)),
                'idx': torch.arange(batch_size),
                'lr': 0.025,
                'momentum_buffers': [] 
            }

            if simulate_memory_efficient_batch_dict:
                # Save the full content to a separate file
                batch_file_path = batch_data_save_dir / f"batch_{step}.pkl"
                torch.save(current_batch_content, batch_file_path)
                # The pickle file only stores the step number for memory-efficient mode
                batch_dict_for_pickle[step] = {'step': step}
            else:
                # Old behavior: store full content in the pickle
                batch_dict_for_pickle[step] = current_batch_content
        
        with open(run_dir / "magic_batch_dict.pkl", 'wb') as f:
            pickle.dump(batch_dict_for_pickle, f)
    
    if create_logs:
        logs_magic_dir = run_dir / "logs_magic"
        logs_magic_dir.mkdir(parents=True, exist_ok=True)
        (logs_magic_dir / "magic_replay_log.json").write_text('{"status": "ok"}')
        
        logs_lds_dir = run_dir / "logs_lds"
        logs_lds_dir.mkdir(parents=True, exist_ok=True)
        (logs_lds_dir / "lds_run_log.json").write_text('{"status": "ok"}')
    
    if create_losses:
        losses_dir = run_dir / "losses_lds"
        losses_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy loss files
        for idx in range(5):
            torch.save({'losses': [0.1, 0.2, 0.3]}, losses_dir / f"loss_fold_{idx}.pth")
        # Also create dummy indices file
        with open(run_dir / "indices_lds.pkl", 'wb') as f:
            pickle.dump({'indices': list(range(10))}, f)
    
    return run_dir

@pytest.mark.integration
def test_scenario1_default_new_magic_run():
    """Scenario 1: Default New MAGIC Run and basic cleanup."""
    # Run MAGIC analysis
    result = run_main_script(["--magic", "--force", "--log_level", "DEBUG"])
    assert result.returncode == 0

    # Find the created run_id
    registry = json.loads(REGISTRY_FILE.read_text())
    assert len(registry["runs"]) == 1
    run_id = list(registry["runs"].keys())[0]
    current_run_dir = RUNS_BASE_DIR / run_id

    assert current_run_dir.exists()
    assert LATEST_LINK.exists() and LATEST_LINK.is_symlink()
    assert LATEST_LINK.resolve() == current_run_dir.resolve()
    
    run_metadata_path = current_run_dir / "run_metadata.json"
    assert run_metadata_path.exists()
    metadata = json.loads(run_metadata_path.read_text())
    assert metadata["status"] == "completed"

    # Check artifacts exist
    assert (current_run_dir / "checkpoints_magic").exists()
    assert (current_run_dir / "scores_magic").exists()
    assert (current_run_dir / "logs_magic").exists()
    assert (current_run_dir / "magic_batch_dict.pkl").exists()

    # Clean checkpoints from this run
    result_clean = run_main_script(["--clean", "--run_id", run_id, "--log_level", "DEBUG"]) 
    assert result_clean.returncode == 0
    
    # After cleanup, checkpoint directories should exist but be empty
    assert (current_run_dir / "checkpoints_magic").exists()
    assert len(list((current_run_dir / "checkpoints_magic").iterdir())) == 0  # Directory is empty
    assert (current_run_dir / "scores_magic").exists()  # Preserved
    assert (current_run_dir / "logs_magic").exists()    # Preserved
    assert (current_run_dir / "magic_batch_dict.pkl").exists()  # Preserved

@pytest.mark.integration
def test_scenario2_magic_with_specified_id():
    """Scenario 2: MAGIC with Specified Run ID."""
    custom_run_id = "my_custom_test_run_01"
    result = run_main_script(["--magic", "--run_id", custom_run_id, "--force", "--log_level", "DEBUG"]) 
    assert result.returncode == 0

    current_run_dir = RUNS_BASE_DIR / custom_run_id
    assert current_run_dir.exists()
    assert LATEST_LINK.exists() and LATEST_LINK.is_symlink()
    assert LATEST_LINK.resolve() == current_run_dir.resolve()

    registry = json.loads(REGISTRY_FILE.read_text())
    assert custom_run_id in registry["runs"]
    
    run_metadata_path = current_run_dir / "run_metadata.json"
    assert run_metadata_path.exists()
    metadata = json.loads(run_metadata_path.read_text())
    assert metadata["status"] == "completed"

    assert (current_run_dir / "checkpoints_magic").exists()
    assert (current_run_dir / "scores_magic").exists()

@pytest.mark.integration
def test_scenario3_magic_skip_train():
    """Scenario 3: MAGIC Skip Train with Existing Run."""
    existing_run_id = "existing_run_for_skip"
    
    # Setup: Create a run with artifacts
    create_dummy_run_artifacts(existing_run_id, create_checkpoints=True, create_batch_dict=True, create_scores=False, create_logs=True)
    initial_score_file = RUNS_BASE_DIR / existing_run_id / "scores_magic" / "magic_scores_val_21.pkl"
    assert not initial_score_file.exists()

    # Test: Skip train, compute scores
    args = [
        "--magic",
        "--run_id", existing_run_id,
        "--skip_train",
        "--force",  # Force score computation
        "--log_level", "DEBUG"
    ]
    result = run_main_script(args)
    if result.returncode != 0:
        pytest.fail(f"main_runner.py exited with {result.returncode}.\\nStdout:\\n{result.stdout}\\nStderr:\\n{result.stderr}")

    current_run_dir = RUNS_BASE_DIR / existing_run_id
    assert current_run_dir.exists()
    assert initial_score_file.exists()  # Scores should be created
    assert (current_run_dir / "logs_magic" / "magic_replay_log.json").exists()
    
    # Check run metadata
    run_metadata_path = current_run_dir / "run_metadata.json"
    assert run_metadata_path.exists()
    metadata = json.loads(run_metadata_path.read_text())
    assert metadata["status"] == "completed"

@pytest.mark.integration
def test_scenario4_error_skip_train_without_run_id():
    """Scenario 4: Error - Skip Train without Run ID."""
    args = [
        "--magic",
        "--skip_train",
        "--log_level", "DEBUG"
    ]
    result = run_main_script(args)
    assert result.returncode == 1
    assert "--skip_train requires --run_id" in result.stderr or \
           "--skip_train requires --run_id" in result.stdout

@pytest.mark.integration
def test_scenario5_lds_with_existing_run():
    """Scenario 5: LDS with Existing Run Context."""
    lds_run_id = "lds_run_context_test"
    
    # Setup: Create a run with MAGIC artifacts
    create_dummy_run_artifacts(lds_run_id, create_checkpoints=True, create_batch_dict=True, create_scores=True, create_logs=True, create_losses=True)
    
    args_lds = [
        "--lds",
        "--run_id", lds_run_id,
        "--log_level", "DEBUG"
    ]
    result_lds = run_main_script(args_lds)
    if result_lds.returncode != 0:
        pytest.fail(f"main_runner.py for LDS exited with {result_lds.returncode}.\\nStdout:\\n{result_lds.stdout}\\nStderr:\\n{result_lds.stderr}")

    current_run_dir = RUNS_BASE_DIR / lds_run_id
    # Assert LDS outputs are within the run directory
    assert (current_run_dir / "checkpoints_lds").exists()
    assert (current_run_dir / "losses_lds").exists()
    assert (current_run_dir / "logs_lds").exists()
    assert (current_run_dir / "plots_lds").exists()
    # Note: indices_lds.pkl is not created when using existing results

@pytest.mark.integration
def test_scenario6_lds_with_explicit_scores_file():
    """Scenario 6: LDS with Explicit Scores File."""
    run_A_id = "run_A_for_scores"
    run_B_id = "run_B_for_lds_outputs"

    # Setup: Create run_A with MAGIC scores
    create_dummy_run_artifacts(run_A_id, create_scores=True, create_checkpoints=False, create_batch_dict=False, create_logs=False)
    scores_path_from_run_A = (RUNS_BASE_DIR / run_A_id / "scores_magic" / "magic_scores_val_21.pkl").resolve()

    # No need to create run_B, it will be created by main_runner

    args_lds = [
        "--lds",
        "--run_id", run_B_id,
        "--scores_file", str(scores_path_from_run_A),
        "--log_level", "DEBUG"
    ]
    result_lds = run_main_script(args_lds)
    if result_lds.returncode != 0:
        pytest.fail(f"main_runner.py for LDS exited with {result_lds.returncode}.\\nStdout:\\n{result_lds.stdout}\\nStderr:\\n{result_lds.stderr}")

    run_B_dir = RUNS_BASE_DIR / run_B_id
    # Assert LDS outputs are in run_B_dir
    assert (run_B_dir / "checkpoints_lds").exists()
    assert (run_B_dir / "losses_lds").exists()

@pytest.mark.integration
def test_scenario7_clean_specific_run():
    """Scenario 7: Clean Checkpoints from Specific Run."""
    run_to_clean_id = "run_to_be_cleaned"
    another_run_id = "another_active_run"

    # Setup: Create two runs with artifacts
    create_dummy_run_artifacts(run_to_clean_id, create_checkpoints=True, create_scores=True, create_logs=True)
    create_dummy_run_artifacts(another_run_id, create_checkpoints=True, create_scores=True, create_logs=True)

    # Paths for assertion
    clean_run_cp_magic = RUNS_BASE_DIR / run_to_clean_id / "checkpoints_magic"
    clean_run_cp_lds = RUNS_BASE_DIR / run_to_clean_id / "checkpoints_lds"
    clean_run_scores = RUNS_BASE_DIR / run_to_clean_id / "scores_magic"
    clean_run_logs = RUNS_BASE_DIR / run_to_clean_id / "logs_magic"

    another_run_cp_magic = RUNS_BASE_DIR / another_run_id / "checkpoints_magic"

    assert clean_run_cp_magic.exists()
    assert clean_run_cp_lds.exists()
    assert clean_run_scores.exists()
    assert clean_run_logs.exists()
    assert another_run_cp_magic.exists()

    # Test: Clean specific run
    args_clean_specific = [
        "--clean",
        "--run_id", run_to_clean_id,
        "--log_level", "DEBUG"
    ]
    result_clean = run_main_script(args_clean_specific)
    assert result_clean.returncode == 0

    # Assert run_to_clean_id checkpoints are cleaned (directories exist but are empty)
    assert clean_run_cp_magic.exists()
    assert len(list(clean_run_cp_magic.iterdir())) == 0  # Directory is empty
    assert clean_run_cp_lds.exists()
    assert len(list(clean_run_cp_lds.iterdir())) == 0  # Directory is empty
    assert clean_run_scores.exists()  # Preserved
    assert clean_run_logs.exists()    # Preserved

    # Assert another_run_id is untouched
    assert another_run_cp_magic.exists()

@pytest.mark.integration
def test_scenario8_list_and_info():
    """Scenario 8: List Runs and Show Run Info."""
    run1_id = "info_run_1"
    run2_id = "info_run_2"

    # Setup: Create a few runs
    create_dummy_run_artifacts(run1_id, create_checkpoints=True, create_scores=False)
    time.sleep(0.1)  # Ensure different timestamps
    create_dummy_run_artifacts(run2_id, create_checkpoints=False, create_scores=True)

    # Test --list
    args_list = ["--list"]
    result_list = run_main_script(args_list)
    assert result_list.returncode == 0
    assert run1_id in result_list.stdout
    assert run2_id in result_list.stdout
    assert "Available runs:" in result_list.stdout

    # Test --info for an existing run
    args_show1 = ["--info", "--run_id", run1_id]
    result_show1 = run_main_script(args_show1)
    assert result_show1.returncode == 0
    assert f"Run {run1_id}:" in result_show1.stdout
    assert '"seed": 42' in result_show1.stdout

    # Test --info for a non-existent run
    args_show_nonexistent = ["--info", "--run_id", "non_existent_run_id_123"]
    result_show_nonexistent = run_main_script(args_show_nonexistent)
    assert result_show_nonexistent.returncode == 0
    assert "Run non_existent_run_id_123 not found" in result_show_nonexistent.stdout

@pytest.mark.integration
def test_scenario9_error_lds_without_run_id():
    """Scenario 9: Error - LDS without Run ID."""
    args = ["--lds", "--log_level", "DEBUG"]
    result = run_main_script(args)
    assert result.returncode == 1
    assert "--lds requires --run_id" in result.stderr or \
           "--lds requires --run_id" in result.stdout

@pytest.mark.integration
def test_scenario10_clean_all():
    """Scenario 10: Clean All (remove entire run)."""
    run_id = "run_to_delete_entirely"
    create_dummy_run_artifacts(run_id, create_checkpoints=True, create_scores=True, create_logs=True)
    
    run_dir = RUNS_BASE_DIR / run_id
    assert run_dir.exists()
    
    # Clean with --what all
    args = ["--clean", "--run_id", run_id, "--what", "all", "--log_level", "DEBUG"]
    result = run_main_script(args)
    assert result.returncode == 0
    
    # Entire run directory should be gone
    assert not run_dir.exists()
    
    # Registry should mark it as deleted
    registry = json.loads(REGISTRY_FILE.read_text())
    assert registry["runs"][run_id]["status"] == "deleted" 