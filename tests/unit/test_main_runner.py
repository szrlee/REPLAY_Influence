#!/usr/bin/env python3
"""
Main Runner Tests for REPLAY Influence Analysis
==============================================

Tests the main_runner.py module including cleanup functions, environment validation,
directory setup, and workflow orchestration.
"""

import pytest
import tempfile
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import argparse
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import main_runner and dependencies
import main_runner
from src import config


class TestCleanupFunctions:
    """Test cleanup functionality"""
    
    @pytest.mark.unit
    @patch('src.config.clean_lds_checkpoints') # Patched at the point of actual execution
    @patch('src.config.clean_magic_checkpoints') # Patched at the point of actual execution
    def test_clean_checkpoints_default(self, mock_clean_magic, mock_clean_lds):
        """Test main_runner.clean_checkpoints with what='checkpoints' (default)."""
        with tempfile.TemporaryDirectory() as tmp_dir_root:
            tmp_path_root = Path(tmp_dir_root)
            
            # Mock project_config.OUTPUTS_DIR for the duration of this test
            # and other relevant global config if main_runner.clean_checkpoints needs them indirectly
            with patch('main_runner.config.OUTPUTS_DIR', tmp_path_root), \
                 patch('main_runner.rm.RUNS_DIR_NAME', "runs") as mock_runs_dir_name_config:

                test_run_id = "my_test_run_to_clean"
                # Simulate the expected run directory structure that clean_checkpoints would operate on
                run_dir_to_operate_on = tmp_path_root / mock_runs_dir_name_config / test_run_id
                run_dir_to_operate_on.mkdir(parents=True, exist_ok=True)
                
                # (Optional: create dummy checkpoint dirs if the mocked functions don't do their own checks)
                # (run_dir_to_operate_on / "checkpoints_magic").mkdir(exist_ok=True)
                # (run_dir_to_operate_on / "checkpoints_lds").mkdir(exist_ok=True)

                main_runner.clean_checkpoints(test_run_id, "checkpoints")

                mock_clean_magic.assert_called_once_with(run_dir_to_operate_on)
                mock_clean_lds.assert_called_once_with(run_dir_to_operate_on)

    @pytest.mark.unit
    @patch('shutil.rmtree')
    @patch('main_runner.rm._update_runs_registry') # Patched at the point of actual execution via main_runner
    def test_clean_checkpoints_all(self, mock_update_registry, mock_rmtree):
        """Test main_runner.clean_checkpoints with what='all'."""
        with tempfile.TemporaryDirectory() as tmp_dir_root:
            tmp_path_root = Path(tmp_dir_root)
            
            with patch('main_runner.config.OUTPUTS_DIR', tmp_path_root), \
                 patch('main_runner.rm.RUNS_DIR_NAME', "runs") as mock_runs_dir_name_config:

                test_run_id = "my_test_run_to_delete"
                run_dir_to_delete = tmp_path_root / mock_runs_dir_name_config / test_run_id
                run_dir_to_delete.mkdir(parents=True, exist_ok=True)
                (run_dir_to_delete / "some_file.txt").write_text("content") # Ensure it's not empty

                main_runner.clean_checkpoints(test_run_id, "all")

                mock_rmtree.assert_called_once_with(run_dir_to_delete)
                mock_update_registry.assert_called_once_with(test_run_id, "deleted")

    @pytest.mark.unit
    def test_clean_checkpoints_non_existent_run(self, caplog):
        """Test main_runner.clean_checkpoints for a non-existent run_id."""
        with tempfile.TemporaryDirectory() as tmp_dir_root:
            tmp_path_root = Path(tmp_dir_root)
            
            with patch('main_runner.config.OUTPUTS_DIR', tmp_path_root), \
                 patch('main_runner.rm.RUNS_DIR_NAME', "runs"), \
                 patch('src.config.clean_magic_checkpoints') as mock_clean_magic, \
                 patch('src.config.clean_lds_checkpoints') as mock_clean_lds, \
                 patch('shutil.rmtree') as mock_rmtree:
                
                non_existent_run_id = "i_dont_exist_123"
                main_runner.clean_checkpoints(non_existent_run_id, "checkpoints")
                
                assert f"Run directory {tmp_path_root / 'runs' / non_existent_run_id} not found" in caplog.text
                mock_clean_magic.assert_not_called()
                mock_clean_lds.assert_not_called()
                mock_rmtree.assert_not_called()


class TestEnvironmentValidation:
    """Test environment validation"""
    
    @pytest.mark.unit
    def test_validate_runtime_environment_success(self):
        """Test successful environment validation"""
        # Mock successful environment validation
        with patch('src.config.validate_environment') as mock_validate:
            mock_validate.return_value = {
                'python_version': '3.8.10',
                'torch_version': '2.0.0',
                'cuda_available': True,
                'outputs_writable': True
            }
            
            # Should not raise exception and return env info
            result = main_runner.validate_runtime_environment()
            
            # Verify function was called
            mock_validate.assert_called_once()
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_validate_runtime_environment_failure(self):
        """Test environment validation failure handling"""
        # Mock environment validation failure
        with patch('src.config.validate_environment', side_effect=EnvironmentError("GPU not available")):
            
            with pytest.raises(EnvironmentError):
                main_runner.validate_runtime_environment()


class TestDirectorySetup:
    """Test directory setup functionality"""
    
    @pytest.mark.unit
    def test_setup_output_directories_success(self):
        """Test successful output directory setup when a run is active."""
        with tempfile.TemporaryDirectory() as tmp_dir_root:
            tmp_path = Path(tmp_dir_root)
            
            # Define the paths that the getter functions should return
            # These will all be under our temp directory
            mock_outputs_dir = tmp_path / "outputs_test"
            mock_current_run_dir = mock_outputs_dir / "runs" / "current_test_run"
            
            paths_to_create_map = {
                'get_magic_checkpoints_dir': mock_current_run_dir / "checkpoints_magic",
                'get_magic_scores_dir': mock_current_run_dir / "scores_magic",
                'get_magic_plots_dir': mock_current_run_dir / "plots_magic",
                'get_magic_logs_dir': mock_current_run_dir / "logs_magic",
                'get_lds_checkpoints_dir': mock_current_run_dir / "checkpoints_lds",
                'get_lds_losses_dir': mock_current_run_dir / "losses_lds",
                'get_lds_plots_dir': mock_current_run_dir / "plots_lds",
                'get_lds_logs_dir': mock_current_run_dir / "logs_lds",
                'get_lds_indices_file': mock_current_run_dir / "indices_lds.pkl",
            }

            # Patch the main OUTPUTS_DIR and the getter functions
            # Important: main_runner.setup_output_directories uses main_runner.config.<getter>
            with patch('main_runner.config.OUTPUTS_DIR', mock_outputs_dir), \
                 patch('main_runner.config.get_current_run_dir', return_value=mock_current_run_dir) as mock_get_curr_run, \
                 patch('main_runner.config.get_magic_checkpoints_dir', return_value=paths_to_create_map['get_magic_checkpoints_dir']), \
                 patch('main_runner.config.get_magic_scores_dir', return_value=paths_to_create_map['get_magic_scores_dir']), \
                 patch('main_runner.config.get_magic_plots_dir', return_value=paths_to_create_map['get_magic_plots_dir']), \
                 patch('main_runner.config.get_magic_logs_dir', return_value=paths_to_create_map['get_magic_logs_dir']), \
                 patch('main_runner.config.get_lds_checkpoints_dir', return_value=paths_to_create_map['get_lds_checkpoints_dir']), \
                 patch('main_runner.config.get_lds_losses_dir', return_value=paths_to_create_map['get_lds_losses_dir']), \
                 patch('main_runner.config.get_lds_plots_dir', return_value=paths_to_create_map['get_lds_plots_dir']), \
                 patch('main_runner.config.get_lds_logs_dir', return_value=paths_to_create_map['get_lds_logs_dir']), \
                 patch('main_runner.config.get_lds_indices_file', return_value=paths_to_create_map['get_lds_indices_file']):
                
                main_runner.setup_output_directories()
                
                # Verify OUTPUTS_DIR itself was created
                assert mock_outputs_dir.exists()
                assert mock_outputs_dir.is_dir()

                # Verify all directories returned by getters were created
                for getter_name, expected_path in paths_to_create_map.items():
                    if getter_name == 'get_lds_indices_file':
                        # For files, check the parent directory
                        assert expected_path.parent.exists()
                        assert expected_path.parent.is_dir()
                    else:
                        assert expected_path.exists()
                        assert expected_path.is_dir()
                mock_get_curr_run.assert_called_once() # Ensure it checked for current run

    @pytest.mark.unit
    def test_setup_output_directories_no_current_run(self):
        """Test directory setup when no current run is active."""
        with tempfile.TemporaryDirectory() as tmp_dir_root:
            tmp_path = Path(tmp_dir_root)
            mock_outputs_dir = tmp_path / "outputs_test_no_run"

            with patch('main_runner.config.OUTPUTS_DIR', mock_outputs_dir), \
                 patch('main_runner.config.get_current_run_dir', return_value=None) as mock_get_curr_run, \
                 patch('main_runner.config.get_magic_checkpoints_dir') as mock_get_magic_ckpt, \
                 patch('main_runner.config.get_lds_indices_file') as mock_get_lds_idx: # Just to check they are not called
                
                main_runner.setup_output_directories()

                assert mock_outputs_dir.exists()
                assert mock_outputs_dir.is_dir()
                mock_get_curr_run.assert_called_once()
                mock_get_magic_ckpt.assert_not_called() # These should not be called if no current run
                mock_get_lds_idx.assert_not_called()


class TestMainFunctionArgumentParsing:
    """Test main function argument parsing and basic workflow"""
    
    @pytest.mark.unit
    @patch('main_runner.run_magic_analysis') # Patch the actual function called by main
    def test_main_argument_parsing_magic(self, mock_run_magic_analysis):
        """Test main function argument parsing for MAGIC analysis."""
        test_args = ['--magic'] # Updated to new CLI argument
        mock_run_magic_analysis.return_value = 0 # Simulate successful run
        
        # Patch other setup functions called before the main action
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', return_value={}), \
             patch('src.config.validate_config'), \
             patch('src.utils.set_global_deterministic_state'):
            
            result = main_runner.main()
            assert result == 0
            # run_magic_analysis is called with (run_id, skip_train, force)
            # For a simple --magic call, run_id=None, skip_train=False, force=False by default from argparse
            mock_run_magic_analysis.assert_called_once_with(None, False, False)
    
    @pytest.mark.unit
    @patch('main_runner.run_lds_validation') # Patch the actual function called by main
    def test_main_argument_parsing_lds(self, mock_run_lds_validation):
        """Test main function argument parsing for LDS analysis."""
        test_run_id = "test_lds_run_123"
        test_args = ['--lds', '--run_id', test_run_id]  # Updated to new CLI arguments
        mock_run_lds_validation.return_value = 0 # Simulate successful run

        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', return_value={}), \
             patch('src.config.validate_config'), \
             patch('src.utils.set_global_deterministic_state'):
            
            result = main_runner.main()
            assert result == 0
            # run_lds_validation is called with (run_id, scores_file, force)
            # For --lds --run_id X, scores_file=None, force=False by default
            mock_run_lds_validation.assert_called_once_with(test_run_id, None, False)
    
    @pytest.mark.unit
    @patch('main_runner.clean_checkpoints') # Patch the actual function called by main
    def test_main_cleanup_action(self, mock_clean_checkpoints):
        """Test main function's --clean action."""
        test_run_id = "test_clean_run_456"
        test_args = ['--clean', '--run_id', test_run_id, '--what', 'all'] # Updated arguments
        mock_clean_checkpoints.return_value = None # clean_checkpoints doesn't return a value for exit code

        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', return_value={}), \
             patch('src.config.validate_config'), \
             patch('src.utils.set_global_deterministic_state'):
            
            result = main_runner.main()
            assert result == 0 # main() should return 0 for successful clean
            # clean_checkpoints is called with (run_id, what)
            mock_clean_checkpoints.assert_called_once_with(test_run_id, "all")

    @pytest.mark.unit
    def test_main_show_config(self):
        """Test main function show config option"""
        test_args = ['--show_config']
        
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment'), \
             patch('src.config.validate_config'), \
             patch('src.config.get_config_summary', return_value="Config Summary"), \
             patch('builtins.print') as mock_print:
            
            result = main_runner.main()
            assert result == 0
            mock_print.assert_called_with("Config Summary")
    
    @pytest.mark.unit
    def test_main_no_arguments(self):
        """Test main function with no arguments"""
        test_args = []
        
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment'), \
             patch('src.config.validate_config'), \
             patch('main_runner.setup_output_directories'), \
             patch('src.utils.set_global_deterministic_state'):
            
            # Expect SystemExit with code 2 when argparse errors out
            with pytest.raises(SystemExit) as e:
                main_runner.main()
            assert e.type == SystemExit
            assert e.value.code == 2


class TestErrorHandling:
    """Test error handling in main runner"""
    
    @pytest.mark.unit
    def test_environment_validation_error(self):
        """Test handling of environment validation errors."""
        # Use a valid main action, e.g., --magic, so argparse doesn't fail first
        test_args = ['--magic'] 
        
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', side_effect=EnvironmentError("Test Env Error")), \
             patch('src.config.validate_config'), \
             patch('src.utils.set_global_deterministic_state'):
            
            # main_runner.main() catches EnvironmentError and returns 1
            result = main_runner.main()
            assert result == 1
    
    @pytest.mark.unit
    def test_magic_analyzer_error(self):
        """Test handling of MAGIC analyzer errors."""
        test_args = ['--magic'] # Use valid new argument
        
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', return_value={}), \
             patch('src.config.validate_config'), \
             patch('main_runner.setup_output_directories'), \
             patch('src.utils.set_global_deterministic_state'), \
             patch('main_runner.run_magic_analysis', side_effect=RuntimeError("Analyzer failed")) as mock_run_magic:
            
            result = main_runner.main()
            assert result == 1
            mock_run_magic.assert_called_once_with(None, False, False) # Check it was called before failing
    
    @pytest.mark.unit
    def test_keyboard_interrupt(self):
        """Test handling of keyboard interrupt."""
        test_args = ['--magic'] # Use valid new argument
        
        # KeyboardInterrupt can happen at various points. Let's simulate it during validate_runtime_environment.
        with patch('sys.argv', ['main_runner.py'] + test_args), \
             patch('main_runner.setup_logging'), \
             patch('main_runner.validate_runtime_environment', side_effect=KeyboardInterrupt()):
            
            result = main_runner.main()
            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 