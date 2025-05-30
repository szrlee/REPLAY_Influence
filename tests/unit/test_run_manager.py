import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import datetime
import json
import os
import time # Import the time module

# Adjust the Python path to include the project root if necessary, 
# or ensure your test runner (e.g., pytest) handles this.
# For direct execution or some setups, this might be needed:
# import sys
# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

from src import run_manager
from src import config as project_config # To mock its attributes if needed

# --- Constants for Testing ---
TEST_OUTPUTS_DIR = Path("/tmp/test_replay_outputs_run_manager")
TEST_RUNS_BASE_DIR = TEST_OUTPUTS_DIR / "runs"
TEST_REGISTRY_FILE = TEST_OUTPUTS_DIR / "runs_registry.json"
TEST_LATEST_LINK = TEST_OUTPUTS_DIR / "latest"

@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_test_environment(monkeypatch):
    """Set up a temporary, isolated test environment for run manager tests."""
    # Ensure the test output directory is clean and exists
    if TEST_OUTPUTS_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUTS_DIR)
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mock project_config.OUTPUTS_DIR to point to our test directory
    monkeypatch.setattr(project_config, 'OUTPUTS_DIR', TEST_OUTPUTS_DIR)
    
    # Mock run_manager.CURRENT_RUN_DIR to ensure it starts fresh for tests that modify it
    monkeypatch.setattr(run_manager, 'CURRENT_RUN_DIR', None)
    monkeypatch.setattr(run_manager, 'RUNS_DIR_NAME', "runs") # Ensure it's the default

    yield

    # Teardown: Clean up the test output directory
    if TEST_OUTPUTS_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUTS_DIR)


class TestGenerateRunId:
    def test_id_format(self):
        """Test that the generated ID has the correct format (timestamp_suffix)."""
        run_id = run_manager.generate_run_id()
        parts = run_id.split('_')
        assert len(parts) == 3, "Run ID should have three parts: YYYYMMDD_HHMMSS_suffix"
        assert len(parts[0]) == 8, "Date part should be 8 characters"
        assert len(parts[1]) == 6, "Time part should be 6 characters"
        assert len(parts[2]) == 6, "Suffix part should be 6 characters"
        try:
            datetime.datetime.strptime(f"{parts[0]}{parts[1]}", "%Y%m%d%H%M%S")
        except ValueError:
            pytest.fail("Timestamp part of run_id is not a valid datetime format")

    def test_id_uniqueness(self):
        """Test that multiple calls generate unique IDs (highly probable)."""
        ids = {run_manager.generate_run_id() for _ in range(100)}
        assert len(ids) == 100, "Generated IDs should be unique"

    @patch('datetime.datetime')
    def test_id_timestamp_consistency(self, mock_datetime):
        """Test if the timestamp part of the ID is consistent with current time."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_datetime.now.return_value = mock_now

        run_id = run_manager.generate_run_id()
        assert run_id.startswith("20250101_120000_"), "Run ID should start with the mocked timestamp"


class TestInitRunDirectory:
    # --- Basic Initialization Tests ---
    def test_init_new_run_creates_directory_and_symlink(self, monkeypatch):
        run_manager.init_run_directory()
        assert run_manager.CURRENT_RUN_DIR is not None
        assert run_manager.CURRENT_RUN_DIR.exists()
        assert run_manager.CURRENT_RUN_DIR.is_dir()
        assert run_manager.CURRENT_RUN_DIR.parent == TEST_RUNS_BASE_DIR

        assert TEST_LATEST_LINK.exists()
        assert TEST_LATEST_LINK.is_symlink()
        # Ensure symlink points correctly relative to its own location
        expected_target = Path("runs") / run_manager.CURRENT_RUN_DIR.name
        assert Path(os.readlink(TEST_LATEST_LINK)) == expected_target

    def test_init_with_specific_run_id_creates_directory(self):
        custom_id = "my_test_run_123"
        run_dir = run_manager.init_run_directory(run_id=custom_id)
        assert run_dir.name == custom_id
        assert run_dir.exists()
        assert run_manager.CURRENT_RUN_DIR == run_dir
        assert (TEST_RUNS_BASE_DIR / custom_id).exists()

    def test_init_use_existing_run_success(self):
        existing_id = "existing_id_for_test"
        # Pre-create the directory and a dummy registry entry for it
        (TEST_RUNS_BASE_DIR / existing_id).mkdir(parents=True, exist_ok=True)
        # Minimal registry update for this test to simulate it was 'created'
        # In a real scenario, _update_runs_registry would be called by the first init
        if TEST_REGISTRY_FILE.exists(): TEST_REGISTRY_FILE.unlink()
        with open(TEST_REGISTRY_FILE, 'w') as f: json.dump({"runs":{existing_id:{"status":"created"}}},f)

        run_dir = run_manager.init_run_directory(run_id=existing_id, use_existing=True)
        assert run_dir.name == existing_id
        assert run_manager.CURRENT_RUN_DIR == run_dir
        assert (TEST_RUNS_BASE_DIR / existing_id).exists() # Should still exist

    def test_init_use_existing_run_failure_if_not_exists(self):
        non_existent_id = "i_do_not_exist"
        with pytest.raises(ValueError, match=f"Run directory .*i_do_not_exist.* does not exist"):
            run_manager.init_run_directory(run_id=non_existent_id, use_existing=True)

    def test_init_multiple_calls_create_different_runs_and_update_symlink(self):
        run_dir1 = run_manager.init_run_directory()
        id1 = run_dir1.name
        assert (TEST_RUNS_BASE_DIR / id1).exists()
        assert Path(os.readlink(TEST_LATEST_LINK)) == Path("runs") / id1

        # Ensure a small delay so timestamps are different for new run ID generation
        time.sleep(0.01) 

        run_dir2 = run_manager.init_run_directory() # New default run
        id2 = run_dir2.name
        assert id1 != id2
        assert (TEST_RUNS_BASE_DIR / id2).exists()
        assert Path(os.readlink(TEST_LATEST_LINK)) == Path("runs") / id2
        assert run_manager.CURRENT_RUN_DIR == run_dir2

    # --- Registry Interaction Tests (more needed here) ---
    @patch('src.run_manager._update_runs_registry')
    def test_init_new_run_calls_update_registry(self, mock_update_registry):
        run_manager.init_run_directory()
        mock_update_registry.assert_called_once()
        # Could also check call_args if specific details are important
        call_args = mock_update_registry.call_args[0]
        assert isinstance(call_args[0], str) # run_id
        assert call_args[1] == "created"      # status

    @patch('src.run_manager._update_runs_registry')
    def test_init_use_existing_does_not_call_update_registry(self, mock_update_registry):
        existing_id = "existing_for_registry_check"
        (TEST_RUNS_BASE_DIR / existing_id).mkdir(parents=True, exist_ok=True)
        if TEST_REGISTRY_FILE.exists(): TEST_REGISTRY_FILE.unlink()
        with open(TEST_REGISTRY_FILE, 'w') as f: json.dump({"runs":{existing_id:{"status":"created"}}},f)

        run_manager.init_run_directory(run_id=existing_id, use_existing=True)
        mock_update_registry.assert_not_called()

    # --- CURRENT_RUN_DIR state ---
    def test_get_current_run_dir_initializes_if_none(self):
        assert run_manager.CURRENT_RUN_DIR is None # Before first call with autouse fixture
        first_call_dir = run_manager.get_current_run_dir()
        assert first_call_dir is not None
        assert first_call_dir.exists()
        assert run_manager.CURRENT_RUN_DIR == first_call_dir

        second_call_dir = run_manager.get_current_run_dir()
        assert second_call_dir == first_call_dir # Should return the same initialized dir

    # Placeholder for tests related to USE_TIMESTAMPED_RUNS = False (flat structure)
    # These would require monkeypatching run_manager.USE_TIMESTAMPED_RUNS
    # def test_init_flat_structure_when_use_timestamped_is_false(self, monkeypatch):
    #     monkeypatch.setattr(run_manager, 'USE_TIMESTAMPED_RUNS', False)
    #     run_dir = run_manager.init_run_directory()
    #     assert run_dir == TEST_OUTPUTS_DIR
    #     assert not TEST_RUNS_BASE_DIR.exists()
    #     assert not TEST_LATEST_LINK.exists()

# TODO: Add test classes for:
# - TestGetCurrentRunDir (if more complex logic than covered above)
# - TestUpdateRunsRegistry (mocking file I/O)
# - TestListRuns (mocking registry file)
# - TestGetLatestRunId (mocking symlink and registry)
# - TestSaveRunMetadata (mocking file I/O and project_config.get_current_config_dict)
# - Test path generation functions (e.g., get_magic_checkpoints_dir) to ensure they use the CURRENT_RUN_DIR correctly.


class TestUpdateRunsRegistry:
    @patch('src.run_manager.Path.exists') # Patch Path.exists
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config') # Mock the imported project_config module
    def test_create_new_registry_if_not_exists(self, mock_proj_config, mock_file_open, mock_path_exists):
        """Test creating a new registry file when one doesn't exist."""
        mock_proj_config.SEED = 123
        mock_proj_config.DEVICE = "cpu"
        mock_proj_config.MODEL_TRAIN_EPOCHS = 5
        mock_proj_config.MODEL_TRAIN_BATCH_SIZE = 32
        mock_proj_config.MAGIC_TARGET_VAL_IMAGE_IDX = 10
        mock_proj_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 10
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR

        # Simulate registry_file.exists() returning False
        mock_path_exists.return_value = False

        run_id = "test_run_new_registry"
        status = "created"
        run_manager._update_runs_registry(run_id, status)

        # Path.exists is called on TEST_REGISTRY_FILE
        mock_path_exists.assert_called_once_with() # Called by TEST_REGISTRY_FILE.exists()

        # open(..., 'w') should be called once
        mock_file_open.assert_called_once_with(TEST_REGISTRY_FILE, 'w')
        # open(..., 'r') should NOT be called

    @patch('json.dump') # Patch json.dump to inspect its arguments
    @patch('src.run_manager.Path.exists') # Patch Path.exists
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config')
    def test_create_new_registry_content(self, mock_proj_config, mock_file_open, mock_path_exists, mock_json_dump):
        mock_proj_config.SEED = 123
        mock_proj_config.DEVICE = "cpu"
        mock_proj_config.MODEL_TRAIN_EPOCHS = 5
        mock_proj_config.MODEL_TRAIN_BATCH_SIZE = 32
        mock_proj_config.MAGIC_TARGET_VAL_IMAGE_IDX = 10
        mock_proj_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 10
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR

        mock_path_exists.return_value = False # Simulate file not existing

        run_id = "test_run_new_registry_content"
        status = "completed"
        run_manager._update_runs_registry(run_id, status)
        
        mock_path_exists.assert_called_once_with()
        mock_file_open.assert_called_once_with(TEST_REGISTRY_FILE, 'w') # Only write call

        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]
        assert run_id in written_data["runs"]
        assert written_data["runs"][run_id]["status"] == status
        assert "timestamp" in written_data["runs"][run_id]
        assert written_data["runs"][run_id]["config"]["seed"] == 123

    @patch('json.dump')
    @patch('json.load')
    @patch('src.run_manager.Path.exists') # Patch Path.exists
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config')
    def test_update_existing_registry_new_run(self, mock_proj_config, mock_file_open, mock_path_exists, mock_json_load, mock_json_dump):
        mock_proj_config.SEED = 456
        mock_proj_config.DEVICE = "cuda"
        mock_proj_config.MODEL_TRAIN_EPOCHS = 3
        mock_proj_config.MODEL_TRAIN_BATCH_SIZE = 64
        mock_proj_config.MAGIC_TARGET_VAL_IMAGE_IDX = 20
        mock_proj_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 20
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR

        mock_path_exists.return_value = True # Simulate file existing

        existing_run_id = "already_exists_run"
        existing_data = {"runs": {existing_run_id: {"status": "created", "timestamp": "old_time", "config": {}}}}
        mock_json_load.return_value = existing_data
        
        # open should be called twice: once for 'r', once for 'w'
        mock_file_open.side_effect = [mock_open(read_data=json.dumps(existing_data)).return_value, mock_open().return_value]

        new_run_id = "new_run_to_add"
        new_status = "running"
        run_manager._update_runs_registry(new_run_id, new_status)

        mock_path_exists.assert_called_once_with()
        assert mock_file_open.call_count == 2
        mock_file_open.assert_any_call(TEST_REGISTRY_FILE, 'r')
        mock_file_open.assert_any_call(TEST_REGISTRY_FILE, 'w')

        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        written_data = mock_json_dump.call_args[0][0]
        assert existing_run_id in written_data["runs"], "Existing run should be preserved"
        assert new_run_id in written_data["runs"], "New run should be added"
        assert written_data["runs"][new_run_id]["status"] == new_status
        assert written_data["runs"][new_run_id]["config"]["seed"] == 456

    @patch('json.dump')
    @patch('json.load')
    @patch('src.run_manager.Path.exists') # Patch Path.exists
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config')
    def test_update_existing_registry_existing_run(self, mock_proj_config, mock_file_open, mock_path_exists, mock_json_load, mock_json_dump):
        mock_proj_config.SEED = 789
        mock_proj_config.DEVICE = "cpu"
        mock_proj_config.MODEL_TRAIN_EPOCHS = 1
        mock_proj_config.MODEL_TRAIN_BATCH_SIZE = 128
        mock_proj_config.MAGIC_TARGET_VAL_IMAGE_IDX = 30
        mock_proj_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 30
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR

        mock_path_exists.return_value = True # Simulate file existing

        run_id_to_update = "run_to_be_updated"
        initial_config = {"seed": 0, "device": "initial_device"}
        existing_data = {"runs": {run_id_to_update: {"status": "created", "timestamp": "very_old_time", "config": initial_config}}}
        mock_json_load.return_value = existing_data

        mock_file_open.side_effect = [mock_open(read_data=json.dumps(existing_data)).return_value, mock_open().return_value]

        updated_status = "completed_successfully"
        run_manager._update_runs_registry(run_id_to_update, updated_status)

        mock_path_exists.assert_called_once_with()
        assert mock_file_open.call_count == 2
        mock_file_open.assert_any_call(TEST_REGISTRY_FILE, 'r')
        mock_file_open.assert_any_call(TEST_REGISTRY_FILE, 'w')
        
        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]

        assert run_id_to_update in written_data["runs"]
        assert written_data["runs"][run_id_to_update]["status"] == updated_status
        assert written_data["runs"][run_id_to_update]["timestamp"] != "very_old_time", "Timestamp should update"
        assert written_data["runs"][run_id_to_update]["config"]["seed"] == 789 
        assert written_data["runs"][run_id_to_update]["config"]["magic_target_idx"] == 30


# TODO: Add test classes for:
# - TestGetCurrentRunDir (if more complex logic than covered above)
# - TestListRuns (mocking registry file)
# - TestGetLatestRunId (mocking symlink and registry)
# - TestSaveRunMetadata (mocking file I/O and project_config.get_current_config_dict)
# - Test path generation functions (e.g., get_magic_checkpoints_dir) to ensure they use the CURRENT_RUN_DIR correctly.


class TestListRuns:
    @patch('src.run_manager.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('src.run_manager.project_config') # For project_config.OUTPUTS_DIR
    def test_list_runs_no_registry_file(self, mock_proj_config, mock_json_load, mock_file_open, mock_path_exists):
        """Test list_runs when the registry file does not exist."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = False # Simulate registry file not existing

        result = run_manager.list_runs()

        mock_path_exists.assert_called_once_with() # For TEST_REGISTRY_FILE.exists()
        mock_file_open.assert_not_called() # open should not be called if file doesn't exist
        mock_json_load.assert_not_called()
        assert result == {"runs": {}}, "Should return empty runs dict if no registry"

    @patch('src.run_manager.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('src.run_manager.project_config')
    def test_list_runs_with_existing_registry(self, mock_proj_config, mock_json_load, mock_file_open, mock_path_exists):
        """Test list_runs when the registry file exists and has data."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = True # Simulate registry file existing

        dummy_registry_content = {
            "runs": {
                "run1": {"status": "completed", "timestamp": "2023-01-01T10:00:00"},
                "run2": {"status": "running", "timestamp": "2023-01-02T12:00:00"}
            }
        }
        # Configure mock_open to simulate reading this content
        # The handle returned by mock_file_open needs to be configured for read_data with json.dumps if json.load is used on handle.
        # Simpler: mock_json_load directly.
        mock_json_load.return_value = dummy_registry_content

        result = run_manager.list_runs()

        mock_path_exists.assert_called_once_with()
        mock_file_open.assert_called_once_with(TEST_REGISTRY_FILE, 'r')
        mock_json_load.assert_called_once_with(mock_file_open())
        assert result == dummy_registry_content, "Should return the content of the registry file"

    @patch('src.run_manager.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('src.run_manager.project_config')
    def test_list_runs_empty_registry_file(self, mock_proj_config, mock_json_load, mock_file_open, mock_path_exists):
        """Test list_runs when the registry file exists but is empty or invalid JSON."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = True
        
        # Simulate json.load raising an error for an empty/invalid file
        mock_json_load.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        # In the actual code, a JSONDecodeError would propagate. 
        # The function list_runs doesn't currently catch this. 
        # This test would verify if it does, or if it propagates as expected.
        # For now, let's assume it should propagate or be handled (e.g. return empty if decode fails)
        # The current implementation will re-raise JSONDecodeError.
        with pytest.raises(json.JSONDecodeError):
             run_manager.list_runs()
        
        mock_path_exists.assert_called_once_with()
        mock_file_open.assert_called_once_with(TEST_REGISTRY_FILE, 'r')
        mock_json_load.assert_called_once_with(mock_file_open())


class TestGetLatestRunId:
    @patch('src.run_manager.Path.readlink')
    @patch('src.run_manager.Path.is_symlink')
    @patch('src.run_manager.Path.exists')
    @patch('src.run_manager.project_config') # For project_config.OUTPUTS_DIR
    def test_get_latest_from_symlink_valid(self, mock_proj_config, mock_path_exists, mock_is_symlink, mock_readlink):
        """Test getting latest run ID from a valid symlink."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        
        # Simulate 'latest' symlink existing and being a symlink
        # Path.exists will be called on TEST_LATEST_LINK
        mock_path_exists.return_value = True 
        mock_is_symlink.return_value = True
        mock_readlink.return_value = Path("runs/20250101_120000_abcdef")

        latest_id = run_manager.get_latest_run_id()
        assert latest_id == "20250101_120000_abcdef"
        
        # Check that Path.exists was called on the TEST_LATEST_LINK instance
        # The patched mock_path_exists is called when latest_link.exists() runs.
        # We expect it to have been called once.
        mock_path_exists.assert_called_once_with() # .exists() takes no args other than self
        # To assert it was called on the correct instance, we can check mock_path_exists.call_args_list[0].instance or similar if the mock framework supports it easily
        # Or, ensure that only one Path instance has .exists() called in the tested code path.
        # For this function, latest_link.exists() is the primary call.

        mock_is_symlink.assert_called_once_with() # .is_symlink() takes no args other than self
        mock_readlink.assert_called_once_with() # .readlink() takes no args other than self

    @patch('src.run_manager.list_runs') # Mock list_runs to control registry fallback
    @patch('src.run_manager.Path.exists')
    @patch('src.run_manager.project_config')
    def test_get_latest_from_registry_if_symlink_missing(self, mock_proj_config, mock_path_exists, mock_list_runs):
        """Test falling back to registry when symlink is missing."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = False # Symlink does not exist

        mock_registry_data = {
            "runs": {
                "run_old": {"status": "completed", "timestamp": "2023-01-01T10:00:00"},
                "run_latest": {"status": "completed", "timestamp": "2023-01-02T12:00:00"},
                "run_middle": {"status": "running", "timestamp": "2023-01-01T15:00:00"}
            }
        }
        mock_list_runs.return_value = mock_registry_data

        latest_id = run_manager.get_latest_run_id()
        assert latest_id == "run_latest"
        mock_list_runs.assert_called_once()

    @patch('src.run_manager.list_runs')
    @patch('src.run_manager.Path.is_symlink')
    @patch('src.run_manager.Path.exists')
    @patch('src.run_manager.project_config')
    def test_get_latest_from_registry_if_symlink_not_symlink(self, mock_proj_config, mock_path_exists, mock_is_symlink, mock_list_runs):
        """Test falling back to registry if 'latest' exists but is not a symlink."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = True    # 'latest' exists
        mock_is_symlink.return_value = False # ...but it's not a symlink

        mock_registry_data = {"runs": {"latest_run_in_reg": {"status": "done", "timestamp": "2024-01-01T00:00:00"}}}
        mock_list_runs.return_value = mock_registry_data

        latest_id = run_manager.get_latest_run_id()
        assert latest_id == "latest_run_in_reg"
        mock_list_runs.assert_called_once()
        mock_is_symlink.assert_called_once_with()

    @patch('src.run_manager.list_runs')
    @patch('src.run_manager.Path.exists')
    @patch('src.run_manager.project_config')
    def test_get_latest_no_symlink_no_registry_runs(self, mock_proj_config, mock_path_exists, mock_list_runs):
        """Test when no symlink and registry has no runs."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        mock_path_exists.return_value = False # No symlink
        mock_list_runs.return_value = {"runs": {}} # Empty registry

        latest_id = run_manager.get_latest_run_id()
        assert latest_id is None
        mock_list_runs.assert_called_once()

    @patch('src.run_manager.Path.readlink')
    @patch('src.run_manager.Path.is_symlink')
    @patch('src.run_manager.Path.exists')
    @patch('src.run_manager.project_config')
    def test_get_latest_from_symlink_malformed_target(self, mock_proj_config, mock_path_exists, mock_is_symlink, mock_readlink):
        """Test symlink pointing to a path not matching 'runs/run_id' format."""
        mock_proj_config.OUTPUTS_DIR = TEST_OUTPUTS_DIR
        
        mock_path_exists.return_value = True
        mock_is_symlink.return_value = True
        mock_readlink.return_value = Path("some_other_dir/my_run") # Malformed target

        # This scenario should fall back to registry because the symlink target is not as expected.
        # We need to mock list_runs for the fallback.
        with patch('src.run_manager.list_runs') as mock_list_runs_fallback:
            mock_list_runs_fallback.return_value = {"runs": {"reg_run": {"timestamp": "2023-01-01"}}}
            latest_id = run_manager.get_latest_run_id()
            assert latest_id == "reg_run"
            mock_list_runs_fallback.assert_called_once()
        mock_readlink.assert_called_once()


# TODO: Add test classes for:
# - TestSaveRunMetadata (mocking file I/O and project_config.get_current_config_dict)
# - Test path generation functions (e.g., get_magic_checkpoints_dir) to ensure they use the CURRENT_RUN_DIR correctly.


class TestSaveRunMetadata:
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config.get_current_config_dict')
    @patch('src.run_manager.datetime') # To control timestamp
    def test_save_metadata_specified_run_dir(self, mock_datetime, mock_get_config, mock_file_open, mock_json_dump):
        """Test saving metadata to a specifically provided run directory."""
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "mocked_iso_timestamp"
        mock_datetime.datetime.now.return_value = mock_now

        mock_config_snapshot_from_func = {"SEED": 42, "MODE": "test"}
        mock_get_config.return_value = mock_config_snapshot_from_func

        test_run_id = "test_run_for_metadata"
        test_run_path = TEST_RUNS_BASE_DIR / test_run_id
        test_run_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        input_metadata = {"status": "test_status", "run_id": test_run_id}
        run_manager.save_run_metadata(dict(input_metadata), run_dir=test_run_path)

        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]

        assert written_data["status"] == "test_status"
        assert written_data["run_id"] == test_run_id
        assert "timestamp" in written_data
        assert written_data["timestamp"] == "mocked_iso_timestamp"
        # Test that the mocked config was used and under the new key
        assert written_data['config_snapshot'] == mock_config_snapshot_from_func
        assert written_data['config_snapshot']["SEED"] == 42
        mock_get_config.assert_called_once() # It should be called as no config was provided in input_metadata
        mock_file_open.assert_called_with(test_run_path / "run_metadata.json", 'w')

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config.get_current_config_dict')
    @patch('src.run_manager.datetime')
    @patch('src.run_manager.get_current_run_dir') # Mock to control what current run dir is
    def test_save_metadata_current_run_dir(self, mock_get_current_run, mock_datetime, mock_get_config, mock_file_open, mock_json_dump):
        """Test saving metadata when run_dir is None (uses current run directory)."""
        mock_current_run_path = TEST_RUNS_BASE_DIR / "current_mock_run"
        mock_current_run_path.mkdir(parents=True, exist_ok=True)
        mock_get_current_run.return_value = mock_current_run_path

        mock_now = MagicMock()
        mock_now.isoformat.return_value = "mocked_iso_timestamp_current"
        mock_datetime.datetime.now.return_value = mock_now

        mock_config_snapshot_from_func_current = {"SPECIAL_KEY": "current_test"}
        mock_get_config.return_value = mock_config_snapshot_from_func_current

        input_metadata = {"status": "test_status_current"}
        run_manager.save_run_metadata(dict(input_metadata), run_dir=None) # run_dir is None

        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]

        assert written_data["status"] == "test_status_current"
        assert "timestamp" in written_data
        assert written_data["timestamp"] == "mocked_iso_timestamp_current"
        # Test that the mocked config was used and under the new key
        assert written_data['config_snapshot'] == mock_config_snapshot_from_func_current
        assert written_data['config_snapshot']["SPECIAL_KEY"] == "current_test"
        mock_get_config.assert_called_once() # It should be called as no config was provided in input_metadata
        mock_file_open.assert_called_with(mock_current_run_path / "run_metadata.json", 'w')

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.run_manager.project_config.get_current_config_dict')
    @patch('src.run_manager.datetime')
    def test_save_metadata_preserves_existing_timestamp_and_config(self, mock_datetime, mock_get_config, mock_file_open, mock_json_dump):
        """Test that existing timestamp and config in metadata are preserved."""
        mock_config_snapshot_from_func = {"SEED": 123} # This value is from the mock perspective
        mock_get_config.return_value = mock_config_snapshot_from_func
    
        test_run_id = "run_with_existing_meta"
        test_run_path = TEST_RUNS_BASE_DIR / test_run_id
        test_run_path.mkdir(parents=True, exist_ok=True)
    
        # IMPORTANT: Input metadata now uses 'config' (the old key) to test the renaming/preservation logic
        input_metadata = {
            "status": "final_check",
            "timestamp": "user_defined_timestamp_preserved",
            "config": {"USER_SEED": 777, "OTHER_PARAM": "keep_this"} # Using old 'config' key
        }
        run_manager.save_run_metadata(dict(input_metadata), run_dir=test_run_path)
    
        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]
    
        assert written_data["status"] == "final_check"
        assert written_data["timestamp"] == "user_defined_timestamp_preserved"
        # Assert that the original 'config' is now under 'config_snapshot'
        assert written_data["config_snapshot"] == {"USER_SEED": 777, "OTHER_PARAM": "keep_this"}
        mock_datetime.datetime.now.assert_not_called() # Timestamp should not be regenerated
        mock_get_config.assert_not_called() # Config snapshot should not be regenerated from project_config


class TestPathGenerationFunctions:
    @patch('src.run_manager.get_current_run_dir')
    def test_get_magic_checkpoints_dir(self, mock_get_current_run):
        mock_run_path = TEST_RUNS_BASE_DIR / "a_test_run"
        mock_get_current_run.return_value = mock_run_path

        expected_path = mock_run_path / "checkpoints_magic"
        assert run_manager.get_magic_checkpoints_dir() == expected_path
        mock_get_current_run.assert_called_once()

    @patch('src.run_manager.get_current_run_dir')
    def test_get_batch_dict_file(self, mock_get_current_run):
        mock_run_path = TEST_RUNS_BASE_DIR / "another_test_run"
        mock_get_current_run.return_value = mock_run_path

        expected_path = mock_run_path / "magic_batch_dict.pkl"
        assert run_manager.get_batch_dict_file() == expected_path
        mock_get_current_run.assert_called_once()

    @patch('src.run_manager.get_magic_checkpoints_dir') # Test a function that calls another path helper
    def test_get_magic_checkpoint_path(self, mock_get_magic_ckpts_dir):
        # This function depends on get_magic_checkpoints_dir, so we mock that directly
        mock_base_checkpoints_path = TEST_RUNS_BASE_DIR / "a_run" / "checkpoints_magic"
        mock_get_magic_ckpts_dir.return_value = mock_base_checkpoints_path

        model_id = 0
        step = 150
        expected_path = mock_base_checkpoints_path / f"sd_{model_id}_{step}.pt"
        assert run_manager.get_magic_checkpoint_path(model_id, step) == expected_path
        mock_get_magic_ckpts_dir.assert_called_once()

    @patch('src.run_manager.get_magic_scores_dir') 
    def test_get_magic_scores_path(self, mock_get_magic_scores_dir_func):
        mock_base_scores_path = TEST_RUNS_BASE_DIR / "scores_run" / "scores_magic"
        mock_get_magic_scores_dir_func.return_value = mock_base_scores_path

        target_idx = 77
        expected_path = mock_base_scores_path / f"magic_scores_val_{target_idx}.pkl"
        assert run_manager.get_magic_scores_path(target_idx) == expected_path
        mock_get_magic_scores_dir_func.assert_called_once()

    @patch('src.run_manager.get_current_run_dir')
    def test_get_lds_checkpoints_dir(self, mock_get_current_run):
        mock_run_path = TEST_RUNS_BASE_DIR / "lds_test_run"
        mock_get_current_run.return_value = mock_run_path
        expected_path = mock_run_path / "checkpoints_lds"
        assert run_manager.get_lds_checkpoints_dir() == expected_path
        mock_get_current_run.assert_called_once()

    @patch('src.run_manager.get_current_run_dir')
    def test_get_lds_losses_dir(self, mock_get_current_run):
        mock_run_path = TEST_RUNS_BASE_DIR / "lds_test_run"
        mock_get_current_run.return_value = mock_run_path
        expected_path = mock_run_path / "losses_lds"
        assert run_manager.get_lds_losses_dir() == expected_path
        mock_get_current_run.assert_called_once()

    @patch('src.run_manager.get_current_run_dir')
    def test_get_lds_indices_file(self, mock_get_current_run):
        mock_run_path = TEST_RUNS_BASE_DIR / "lds_test_run"
        mock_get_current_run.return_value = mock_run_path
        expected_path = mock_run_path / "indices_lds.pkl"
        assert run_manager.get_lds_indices_file() == expected_path
        mock_get_current_run.assert_called_once()
    
    @patch('src.run_manager.get_lds_checkpoints_dir')
    def test_get_lds_subset_model_checkpoint_path(self, mock_get_lds_ckpts_dir):
        mock_base_lds_checkpoints_path = TEST_RUNS_BASE_DIR / "a_lds_run" / "checkpoints_lds"
        mock_get_lds_ckpts_dir.return_value = mock_base_lds_checkpoints_path
        model_id = 3
        epoch = 5
        expected_path = mock_base_lds_checkpoints_path / f"sd_lds_{model_id}_{epoch}.pt"
        assert run_manager.get_lds_subset_model_checkpoint_path(model_id, epoch) == expected_path
        mock_get_lds_ckpts_dir.assert_called_once()

    @patch('src.run_manager.get_magic_logs_dir')
    def test_get_magic_training_log_path(self, mock_get_magic_logs):
        mock_base_log_path = TEST_RUNS_BASE_DIR / "logging_run" / "logs_magic"
        mock_get_magic_logs.return_value = mock_base_log_path
        expected_path = mock_base_log_path / "magic_training_log.json"
        assert run_manager.get_magic_training_log_path() == expected_path
        mock_get_magic_logs.assert_called_once()

    @patch('src.run_manager.get_magic_scores_path') # This is the function called internally
    @patch('src.run_manager.project_config') # To mock LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
    def test_get_magic_scores_file_for_lds_input(self, mock_proj_config, mock_get_magic_scores_path_func):
        mock_proj_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 55
        
        # Simulate what get_magic_scores_path would return
        expected_final_path = TEST_RUNS_BASE_DIR / "some_run" / "scores_magic" / "magic_scores_val_55.pkl"
        mock_get_magic_scores_path_func.return_value = expected_final_path

        actual_path = run_manager.get_magic_scores_file_for_lds_input()
        assert actual_path == expected_final_path
        # Check that get_magic_scores_path was called with the correct target_idx from project_config
        mock_get_magic_scores_path_func.assert_called_once_with(55)

    # Similar tests can be added for other LDS path functions like:
    # get_lds_checkpoints_dir, get_lds_losses_dir, get_lds_indices_file, 
    # get_lds_subset_model_checkpoint_path, get_lds_model_val_loss_path,
    # get_magic_training_log_path, get_magic_replay_log_path, 
    # get_lds_training_log_path, get_magic_scores_file_for_lds_input (which also needs project_config mock)


# TODO: Add test classes for:
# (No main items left from the previous TODO, but individual path functions above could be expanded)

