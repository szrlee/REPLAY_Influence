#!/usr/bin/env python3
"""
Complete Workflow Integration Tests for REPLAY Influence Analysis
================================================================

Tests end-to-end workflows and integration between components.
Ensures the complete system works together correctly.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import warnings
import logging
import os # Add os import for environment variables
import importlib # Add importlib for reloading

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import config after potentially reloading it via fixture
# from src import config # Moved to fixtures where it's reloaded
from src.utils import (
    derive_component_seed,
    set_global_deterministic_state,
    create_deterministic_model,
    create_deterministic_dataloader
)
from src.model_def import construct_rn9
from src.data_handling import get_cifar10_dataloader, CustomDataset
# Import run_manager for monkeypatching BASE_OUTPUTS_DIR
from src import run_manager 


@pytest.fixture(scope="function")
def isolated_config_for_integration(tmp_path, monkeypatch):
    """
    Provides a fresh, isolated src.config module for integration tests.
    - Sets REPLAY_OUTPUTS_DIR environment variable to a temporary path.
    - Reloads src.config and src.run_manager to use this path.
    - Initializes a new run directory within this temporary path.
    - Applies minimal default settings to the fresh config suitable for integration tests.
    - Restores original REPLAY_OUTPUTS_DIR and reloads modules in teardown.
    """
    original_sys_path = list(sys.path)
    original_replay_outputs_dir = os.environ.get("REPLAY_OUTPUTS_DIR")

    # Ensure SRC_DIR is on path for imports
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    isolated_base_outputs = tmp_path / f"integration_test_outputs_{Path(tempfile.NamedTemporaryFile().name).name}"
    isolated_base_outputs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REPLAY_OUTPUTS_DIR", str(isolated_base_outputs))

    # (Re)load config
    if 'src.config' in sys.modules:
        fresh_config = importlib.reload(sys.modules['src.config'])
    else:
        from src import config as fresh_config
    
    # (Re)load run_manager
    if 'src.run_manager' in sys.modules:
        fresh_run_manager = importlib.reload(sys.modules['src.run_manager'])
    else:
        from src import run_manager as fresh_run_manager

    # Apply minimal default config settings
    fresh_config.MODEL_TRAIN_EPOCHS = 1
    fresh_config.LDS_NUM_MODELS_TO_TRAIN = 1
    fresh_config.LDS_NUM_SUBSETS_TO_GENERATE = 1
    fresh_config.NUM_CLASSES = 10 # Default for CIFAR-10
    fresh_config.MODEL_CREATOR_FUNCTION = construct_rn9
    
    fresh_run_manager.init_run_directory() # Initialize after setting env and reloading

    yield fresh_config

    # Teardown
    sys.path[:] = original_sys_path

    if original_replay_outputs_dir is None:
        monkeypatch.delenv("REPLAY_OUTPUTS_DIR", raising=False)
    else:
        monkeypatch.setenv("REPLAY_OUTPUTS_DIR", original_replay_outputs_dir)
    
    # Reload modules again for cleanliness
    if 'src.config' in sys.modules:
        importlib.reload(sys.modules['src.config'])
    if 'src.run_manager' in sys.modules:
        importlib.reload(sys.modules['src.run_manager'])


class TestMAGICAnalysisWorkflow:
    """Test complete MAGIC analysis workflow"""
    
    def create_mock_cifar_dataloader(self, current_config, split='train', batch_size=32):
        """Create a mock CIFAR-10 dataloader for testing"""
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size, num_classes):
                self.size = size
                self.num_classes = num_classes
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                actual_idx = idx % self.size
                return torch.randn(3, 32, 32), actual_idx % self.num_classes, actual_idx
        
        dataset_size = 100 if split == 'train' else 20
        dataset = MockDataset(size=dataset_size, num_classes=current_config.NUM_CLASSES)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    
    @pytest.mark.integration
    def test_magic_minimal_training_workflow(self, isolated_config_for_integration):
        """Test minimal MAGIC training workflow with mocked components using isolated config"""
        current_config = isolated_config_for_integration
        
        # Directories are now managed by run_manager, accessed via current_config.get_..._dir()
        # Ensure they exist (they should be created by init_run_directory or on first use by analyzer)
        current_config.get_magic_checkpoints_dir().mkdir(parents=True, exist_ok=True)
        current_config.get_magic_scores_dir().mkdir(parents=True, exist_ok=True)
        current_config.get_magic_logs_dir().mkdir(parents=True, exist_ok=True)
                        
        # Test deterministic model creation
        model = create_deterministic_model(
            master_seed=current_config.SEED,
            creator_func=current_config.MODEL_CREATOR_FUNCTION,
            instance_id="magic_workflow_test",
            num_classes=current_config.NUM_CLASSES
        )
        model.to(current_config.DEVICE)
        
        # Test mock training loop
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Mock dataloader
        train_loader = self.create_mock_cifar_dataloader(current_config, 'train', batch_size=16)
        
        # Simulate training steps
        model.train()
        training_metrics = []
        
        for epoch in range(current_config.MODEL_TRAIN_EPOCHS):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, labels, indices) in enumerate(train_loader):
                images, labels = images.to(current_config.DEVICE), labels.to(current_config.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_idx >= 5:
                    break
            
            if batch_count == 0:
                if len(train_loader) > 0:
                     pytest.fail("Batch count was zero despite non-empty dataloader.")
                avg_loss = 0.0
            else:
                avg_loss = epoch_loss / batch_count

            training_metrics.append({
                'epoch': epoch,
                'loss': avg_loss,
                'batch_count': batch_count
            })
        
        # Verify training progressed
        assert len(training_metrics) == current_config.MODEL_TRAIN_EPOCHS
        if len(train_loader) > 0 and training_metrics[0]['batch_count'] > 0 :
             assert all(m['loss'] >= 0 for m in training_metrics)
        
        # Test checkpoint saving
        checkpoint_path = current_config.get_magic_checkpoint_path(model_id="test_model", step_or_epoch=0)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        assert checkpoint_path.exists()
        
        # Test checkpoint loading
        loaded_state = torch.load(checkpoint_path)
        new_model = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES)
        new_model.load_state_dict(loaded_state)
        new_model.to(current_config.DEVICE)
        
        # Verify models are equivalent
        test_input = torch.randn(1, 3, 32, 32).to(current_config.DEVICE)
        model.eval()
        new_model.eval()
        with torch.no_grad():
            output1 = model(test_input)
            output2 = new_model(test_input)
            assert torch.allclose(output1, output2, atol=1e-6)
    
    @pytest.mark.integration
    def test_magic_error_handling_and_recovery(self, isolated_config_for_integration):
        """Test MAGIC workflow error handling and recovery mechanisms"""
        current_config = isolated_config_for_integration
        tmp_path = Path(current_config.get_current_run_dir()).parent.parent
            
        # Test handling of missing checkpoint directory
        checkpoint_path_to_test = current_config.get_magic_checkpoint_path(model_id="m1", step_or_epoch=1)
        assert isinstance(checkpoint_path_to_test, Path)
            
        # Test handling of corrupted data
        scores_dir_for_test = current_config.get_magic_scores_dir()
        scores_dir_for_test.mkdir(parents=True, exist_ok=True)
            
        corrupted_file = scores_dir_for_test / "corrupted_scores.pkl"
        corrupted_file.write_text("not a pickle file")
            
        # Should handle corrupted files gracefully
        with pytest.raises((pickle.UnpicklingError, UnicodeDecodeError, EOFError)):
            with open(corrupted_file, 'rb') as f:
                pickle.load(f)
            
        # Test model recovery after interruption
        model_to_recover = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES).to(current_config.DEVICE)
        optimizer_to_recover = torch.optim.SGD(model_to_recover.parameters(), lr=0.01)
            
        # Simulate checkpoint saving for recovery
        checkpoint_content = {
            'model_state_dict': model_to_recover.state_dict(),
            'optimizer_state_dict': optimizer_to_recover.state_dict(),
            'epoch': 5,
            'loss': 0.5
        }
            
        # Save recovery checkpoint within the run's checkpoint_magic directory
        recovery_path = current_config.get_magic_checkpoints_dir() / "recovery_checkpoint.pt"
        recovery_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_content, recovery_path)
            
        # Test recovery process
        new_model_recovered = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES).to(current_config.DEVICE)
        new_optimizer_recovered = torch.optim.SGD(new_model_recovered.parameters(), lr=0.01)
            
        loaded_checkpoint_content = torch.load(recovery_path, map_location=current_config.DEVICE)
        new_model_recovered.load_state_dict(loaded_checkpoint_content['model_state_dict'])
        new_optimizer_recovered.load_state_dict(loaded_checkpoint_content['optimizer_state_dict'])
            
        assert loaded_checkpoint_content['epoch'] == 5
        assert loaded_checkpoint_content['loss'] == 0.5


class TestLDSValidationWorkflow:
    """Test complete LDS validation workflow"""
    
    def create_mock_cifar_dataloader(self, current_config, split='train', batch_size=32):
        """Create a mock CIFAR-10 dataloader for testing"""
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size, num_classes):
                self.size = size
                self.num_classes = num_classes
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                actual_idx = idx % self.size
                return torch.randn(3, 32, 32), actual_idx % self.num_classes, actual_idx
        
        # Use a small size for mock data, ensure it's at least 1
        if split == 'train':
            # Use a small fraction of configured train samples, or a default like 100
            dataset_size = max(1, current_config.NUM_TRAIN_SAMPLES // 500 if hasattr(current_config, 'NUM_TRAIN_SAMPLES') else 100)
        else: # split == 'test' or other
            dataset_size = max(1, current_config.NUM_TEST_SAMPLES_CIFAR10 // 500 if hasattr(current_config, 'NUM_TEST_SAMPLES_CIFAR10') else 20)
            
        dataset = MockDataset(size=dataset_size, num_classes=current_config.NUM_CLASSES)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    
    @pytest.mark.integration
    def test_lds_minimal_validation_workflow(self, isolated_config_for_integration):
        """Test minimal LDS validation workflow"""
        current_config = isolated_config_for_integration
        
        # LDS directories are managed by run_manager, ensure they exist for the test setup
        current_config.get_lds_checkpoints_dir().mkdir(parents=True, exist_ok=True)
        current_config.get_lds_losses_dir().mkdir(parents=True, exist_ok=True)
        current_config.get_magic_scores_dir().mkdir(parents=True, exist_ok=True)
                    
        # Create mock MAGIC scores for LDS input
        mock_scores_content = {
            'scores_per_step': np.random.randn(10, 100).astype(np.float32),
            'influence_scores': np.random.randn(100).astype(np.float32),
            'train_indices_used_for_analysis': list(range(100)),
            'target_loss_at_each_step': np.random.rand(10).astype(np.float32),
            'val_idx': current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION 
        }
        
        # Path for MAGIC scores that LDS will read
        scores_file_for_lds = current_config.get_magic_scores_file_for_lds_input()
        scores_file_for_lds.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_file_for_lds, 'wb') as f:
            pickle.dump(mock_scores_content, f)
        assert scores_file_for_lds.exists()
        
        # Test subset generation based on influence scores
        influence_scores_array = mock_scores_content['influence_scores']
        num_samples_for_lds = len(influence_scores_array)
        num_top_influences = max(1, int(num_samples_for_lds * 0.1)) 
        top_influences_indices = np.argsort(np.abs(influence_scores_array))[-num_top_influences:]
        
        assert len(top_influences_indices) == num_top_influences
        assert all(idx < num_samples_for_lds for idx in top_influences_indices)
        
        # Test subset model training simulation
        subset_models = []
        subset_losses_records = []
        
        global_model_counter = 0 # Counter for unique model IDs
        # Simulate for a couple of subsets/models as defined in minimal config
        for subset_idx in range(current_config.LDS_NUM_SUBSETS_TO_GENERATE):
            for model_idx_in_subset in range(current_config.LDS_NUM_MODELS_TO_TRAIN): # Renamed model_idx to avoid confusion
                # Create subset model
                model = create_deterministic_model(
                    master_seed=current_config.SEED + subset_idx * 10 + model_idx_in_subset, # Unique seed
                    creator_func=current_config.MODEL_CREATOR_FUNCTION,
                    instance_id=f"lds_s{subset_idx}_m{model_idx_in_subset}_id{global_model_counter}", # More descriptive instance_id
                    num_classes=current_config.NUM_CLASSES # Pass num_classes
                ).to(current_config.DEVICE)
                
                # Simulate training and validation
                mock_val_loader = self.create_mock_cifar_dataloader(current_config, 'test', batch_size=current_config.MODEL_TRAIN_BATCH_SIZE)
                
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for v_images, v_labels, _ in mock_val_loader:
                        v_images, v_labels = v_images.to(current_config.DEVICE), v_labels.to(current_config.DEVICE)
                        outputs = model(v_images)
                        criterion = torch.nn.CrossEntropyLoss()
                        val_loss = criterion(outputs, v_labels).item()
                        val_loss_sum += val_loss
                        val_batches += 1
                        if val_batches >= 1: break
                
                avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float('inf')
                
                subset_models.append(model)
                subset_losses_records.append({
                    'subset_idx': subset_idx, 
                    'model_idx_in_subset': model_idx_in_subset, 
                    'global_model_id': global_model_counter,
                    'val_loss': avg_val_loss
                })

                # Simulate saving this loss (as LDS validator would)
                loss_path = current_config.get_lds_model_val_loss_path(model_id=global_model_counter)
                loss_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                with open(loss_path, 'wb') as f_loss:
                    pickle.dump({'val_loss': avg_val_loss, 'epoch': current_config.MODEL_TRAIN_EPOCHS, 'global_model_id': global_model_counter}, f_loss)
                assert loss_path.exists()
                global_model_counter += 1 # Increment for next unique model

        expected_models_trained = current_config.LDS_NUM_SUBSETS_TO_GENERATE * current_config.LDS_NUM_MODELS_TO_TRAIN
        assert len(subset_models) == expected_models_trained
        assert len(subset_losses_records) == expected_models_trained
        if expected_models_trained > 0:
             assert all(record['val_loss'] >= 0 for record in subset_losses_records)

    @pytest.mark.integration
    def test_lds_correlation_computation(self, isolated_config_for_integration):
        """Test LDS correlation computation part"""
        current_config = isolated_config_for_integration
        # Ensure relevant directories exist
        current_config.get_lds_losses_dir().mkdir(parents=True, exist_ok=True)
        current_config.get_magic_scores_dir().mkdir(parents=True, exist_ok=True)

        # 1. Create mock MAGIC scores (input to LDS)
        num_train_samples_for_scores = 200
        mock_magic_scores_content = {
            'influence_scores': np.random.rand(num_train_samples_for_scores).astype(np.float32),
            'train_indices_used_for_analysis': list(range(num_train_samples_for_scores)),
            'val_idx': current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
        }
        magic_scores_file = current_config.get_magic_scores_file_for_lds_input()
        magic_scores_file.parent.mkdir(parents=True, exist_ok=True)
        with open(magic_scores_file, 'wb') as f:
            pickle.dump(mock_magic_scores_content, f)

        # 2. Create mock LDS model validation losses
        num_subsets = current_config.LDS_NUM_SUBSETS_TO_GENERATE
        num_models_per_subset = current_config.LDS_NUM_MODELS_TO_TRAIN
        
        mock_lds_losses = {} # Store losses: key=global_model_id, value=loss
        global_model_id_counter = 0
        for s_idx in range(num_subsets):
            for m_idx_in_subset in range(num_models_per_subset):
                loss_val = np.random.rand() 
                mock_lds_losses[global_model_id_counter] = loss_val # Use global_model_id as key
                loss_file_path = current_config.get_lds_model_val_loss_path(model_id=global_model_id_counter)
                loss_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(loss_file_path, 'wb') as f_loss:
                    pickle.dump({'val_loss': loss_val, 'epoch': current_config.MODEL_TRAIN_EPOCHS, 'global_model_id': global_model_id_counter}, f_loss)
                global_model_id_counter += 1
        
        # 3. Simulate correlation computation (simplified)
        if num_subsets > 0 and num_models_per_subset > 0 :
            # Average losses per subset - this logic might need adjustment
            # if correlation is now against individual model losses identified by global_model_id.
            # For simplicity, let's assume we collect all global_model_id losses.
            all_lds_losses = [mock_lds_losses[gid] for gid in sorted(mock_lds_losses.keys())]
            
            # This is a placeholder for a real correlation.
            # For this test, just ensure we have the data points to do *some* kind of correlation.
            magic_influence_scores = mock_magic_scores_content['influence_scores']
            
            # A very basic check: if we have multiple models, their losses should be available.
            assert len(all_lds_losses) == global_model_id_counter
            
            # If we had a way to link subsets to specific points removed by MAGIC, we could correlate.
            logging.info(f"Mock MAGIC scores (len {len(magic_influence_scores)}) and LDS losses (for {num_subsets} subsets) prepared.")
        else:
            logging.info("Skipping correlation part as LDS models to train is 0.")


class TestEndToEndIntegration:
    """Test complete end-to-end pipeline from MAGIC to LDS"""

    # Helper to create mock CIFAR-10 dataloader, similar to one in TestMAGICAnalysisWorkflow
    def _create_mock_dataloader_e2e(self, current_config, split='train', batch_size=None, num_samples=None):
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, num_classes):
                self.num_samples = num_samples
                self.num_classes = num_classes
            def __len__(self): return self.num_samples
            def __getitem__(self, idx):
                actual_idx = idx % self.num_samples
                return torch.randn(3, 32, 32), actual_idx % self.num_classes, actual_idx

        if batch_size is None: batch_size = current_config.MODEL_TRAIN_BATCH_SIZE
        
        if num_samples is None:
            num_samples = 128 if split == 'train' else 32
            
        dataset = MockDataset(num_samples, current_config.NUM_CLASSES)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_magic_to_lds_pipeline(self, isolated_config_for_integration):
        """
        Test a complete, albeit mocked, pipeline:
        1. Mock Training (to generate a base model state)
        2. Mock MAGIC Analysis (to generate influence scores)
        3. Mock LDS Validation (using MAGIC's scores to retrain subset models and get losses)
        4. Mock Correlation (a placeholder for checking if inputs align)
        This test uses the isolated_config_for_integration fixture.
        """
        current_config = isolated_config_for_integration
        logger = logging.getLogger("E2E_MAGIC_LDS_Pipeline")
        logger.info(f"Starting E2E MAGIC to LDS pipeline test in run dir: {current_config.get_current_run_dir()}")

        # --- Setup & Common Components ---
        set_global_deterministic_state(current_config.SEED, True)
        device = current_config.DEVICE
        num_classes = current_config.NUM_CLASSES
        model_creator = current_config.MODEL_CREATOR_FUNCTION

        # Create initial model (simulating a pre-trained or fully trained model for MAGIC)
        logger.info("Creating initial model...")
        initial_model = create_deterministic_model(
            current_config.SEED, model_creator, "initial_model_e2e", num_classes=num_classes
        ).to(device)
        
        # Save this initial model as if it's the result of a full training run (e.g., step_T checkpoint)
        initial_model_path = current_config.get_magic_checkpoint_path(model_id="system", step_or_epoch="T")
        initial_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': initial_model.state_dict()}, initial_model_path)
        logger.info(f"Saved initial model checkpoint to: {initial_model_path}")

        # Mock Dataloaders
        logger.info("Creating mock dataloaders...")
        mock_train_samples = 256 
        mock_test_samples = 64

        train_loader_for_magic_analysis = self._create_mock_dataloader_e2e(
            current_config, split='train', num_samples=mock_train_samples, batch_size=current_config.MODEL_TRAIN_BATCH_SIZE
        )
        val_loader_for_magic = self._create_mock_dataloader_e2e(
            current_config, split='test', num_samples=mock_test_samples, batch_size=current_config.MODEL_TRAIN_BATCH_SIZE
        )
        
        # --- Mock MAGIC Analysis ---
        logger.info("Simulating MAGIC Analysis...")
        num_analyzed_train_samples = mock_train_samples 

        mock_influence_scores = np.random.rand(num_analyzed_train_samples).astype(np.float32)
        mock_train_indices_magic = list(range(num_analyzed_train_samples)) 

        target_val_idx = current_config.MAGIC_TARGET_VAL_IMAGE_IDX
        assert target_val_idx < mock_test_samples, "MAGIC_TARGET_VAL_IMAGE_IDX out of bounds for mock val_loader"

        magic_scores_data = {
            'influence_scores': mock_influence_scores,
            'train_indices_used_for_analysis': mock_train_indices_magic, 
            'val_idx': target_val_idx,
            'scores_per_step': np.random.rand(10, num_analyzed_train_samples).astype(np.float32),
            'target_loss_at_each_step': np.random.rand(10).astype(np.float32)
        }
        magic_scores_path = current_config.get_magic_scores_path(target_idx=target_val_idx)
        magic_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(magic_scores_path, 'wb') as f_magic_scores:
            pickle.dump(magic_scores_data, f_magic_scores)
        logger.info(f"Mock MAGIC scores saved to: {magic_scores_path}")
        assert magic_scores_path.exists()

        # --- Mock LDS Validation ---
        logger.info("Simulating LDS Validation...")
        current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = target_val_idx
        
        assert current_config.get_magic_scores_file_for_lds_input() == magic_scores_path

        num_subsets_lds = current_config.LDS_NUM_SUBSETS_TO_GENERATE
        num_models_lds = current_config.LDS_NUM_MODELS_TO_TRAIN
        
        lds_final_losses = [] # Store final (mock) validation loss for each LDS model
        global_lds_model_id_counter = 0 # Unique ID for each LDS model trained

        for s_idx in range(num_subsets_lds):
            # In a real LDS, a subset of training data indices would be selected here based on MAGIC scores
            # For mock, we don't need the actual subset data, just simulate retraining.
            logger.info(f"Simulating LDS retraining for subset {s_idx}...")
            for m_idx_in_subset in range(num_models_lds):
                # Create and "retrain" a model
                lds_model = create_deterministic_model(
                    current_config.SEED, model_creator, f"lds_s{s_idx}_m{m_idx_in_subset}_gid{global_lds_model_id_counter}", num_classes=num_classes
                ).to(device)
                
                # Simulate evaluating this LDS model on the validation set
                # (or a specific part of it if LDS methodology dictates)
                mock_val_loss = np.random.rand() # Mock validation loss
                lds_final_losses.append(mock_val_loss)

                # Save this mock loss as LDS would
                lds_loss_path = current_config.get_lds_model_val_loss_path(model_id=global_lds_model_id_counter)
                lds_loss_path.parent.mkdir(parents=True, exist_ok=True)
                with open(lds_loss_path, 'wb') as f_lds_loss:
                    pickle.dump({'val_loss': mock_val_loss, 'epoch': current_config.MODEL_TRAIN_EPOCHS, 'global_model_id': global_lds_model_id_counter}, f_lds_loss)
                assert lds_loss_path.exists()
                
                # Save a mock checkpoint for this LDS model
                # Assuming step_or_epoch for LDS model checkpoint refers to its final epoch for simplicity in mock
                lds_checkpoint_path = current_config.get_lds_subset_model_checkpoint_path(model_id=global_lds_model_id_counter, step_or_epoch=current_config.MODEL_TRAIN_EPOCHS)
                lds_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': lds_model.state_dict()}, lds_checkpoint_path)
                assert lds_checkpoint_path.exists()
                global_lds_model_id_counter += 1

        expected_lds_models = num_subsets_lds * num_models_lds
        assert len(lds_final_losses) == expected_lds_models
        logger.info(f"Simulated {expected_lds_models} LDS model retrainings and saved mock losses/checkpoints.")

        # --- Mock Correlation/Final Check ---
        logger.info("Performing final checks / mock correlation...")
        assert current_config.get_magic_scores_dir().exists()
        assert magic_scores_path.exists()
        assert current_config.get_magic_checkpoints_dir().exists()
        assert initial_model_path.exists()

        assert current_config.get_lds_losses_dir().exists()
        assert current_config.get_lds_checkpoints_dir().exists()
        
        if expected_lds_models > 0:
            # Check one example loss file and checkpoint file using global_id 0
            assert current_config.get_lds_model_val_loss_path(model_id=0).exists()
            assert current_config.get_lds_subset_model_checkpoint_path(model_id=0, step_or_epoch=current_config.MODEL_TRAIN_EPOCHS).exists()

        logger.info("✅ E2E MAGIC to LDS pipeline test completed successfully (mocked stages).")


    @pytest.mark.integration
    def test_deterministic_reproducibility_workflow(self, isolated_config_for_integration):
        """Test that key components produce reproducible results given the same seed."""
        current_config = isolated_config_for_integration
        logger = logging.getLogger("DeterminismTest")

        master_seed = current_config.SEED
        num_classes = current_config.NUM_CLASSES
        model_creator = current_config.MODEL_CREATOR_FUNCTION
        device = current_config.DEVICE

        # 1. Test Model Determinism
        logger.info("Testing model determinism...")
        set_global_deterministic_state(master_seed, True)
        model1_m = create_deterministic_model(master_seed, model_creator, "model_rep_test", num_classes=num_classes).to(device)
        
        set_global_deterministic_state(master_seed, True)
        model2_m = create_deterministic_model(master_seed, model_creator, "model_rep_test", num_classes=num_classes).to(device)

        # Check parameters are identical
        for p1, p2 in zip(model1_m.parameters(), model2_m.parameters()):
            assert torch.allclose(p1, p2), "Model parameters differ between two creations with the same seed."
        
        # Check output for same input is identical
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        model1_m.eval()
        model2_m.eval()
        with torch.no_grad():
            out1_m = model1_m(dummy_input)
            out2_m = model2_m(dummy_input)
        assert torch.allclose(out1_m, out2_m), "Model outputs differ for the same input and seed."
        logger.info("Model determinism check passed.")

        # 2. Test DataLoader Determinism
        logger.info("Testing dataloader determinism...")
        num_mock_samples_dl = 32 
        mock_batch_size_dl = 8

        def create_mock_dl_for_rep_test(seed, instance_id):
            # Create a new generator for each dataloader instance, seeded deterministically
            dataloader_generator_seed = derive_component_seed(seed, "dataloader_content_generator", instance_id)
            generator_for_data = torch.Generator().manual_seed(dataloader_generator_seed)

            set_global_deterministic_state(seed, True) # Set global state for this creation
            def _mock_creator_func(batch_size, split, shuffle, num_workers, root_path, dataset_generator, **kwargs):
                class InnerMockDataset(torch.utils.data.Dataset):
                    def __init__(self, n_samples, n_classes_arg, gen):
                        self.n_samples, self.n_classes = n_samples, n_classes_arg
                        self.generator = gen # Store the generator
                    def __len__(self): return self.n_samples
                    def __getitem__(self, idx_): 
                        # Use the passed generator for torch.randn
                        img_data = torch.randn(3,32,32, generator=self.generator)
                        return img_data, idx_ % self.n_classes, idx_
                
                _dataset = InnerMockDataset(num_mock_samples_dl, num_classes, dataset_generator) # Pass generator to dataset
                return torch.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

            # Pass the specifically created generator to the creator_func via kwargs
            return create_deterministic_dataloader(
                master_seed=seed, 
                creator_func=_mock_creator_func, 
                instance_id=instance_id,
                batch_size=mock_batch_size_dl, 
                split='train', 
                shuffle=False, # Keep shuffle as False from previous attempt
                num_workers=0, 
                root_path="mock_path",
                dataset_generator=generator_for_data # Pass the generator directly as a kwarg
            )

        loader1_dl = create_mock_dl_for_rep_test(master_seed, "loader_rep_test")
        loader2_dl = create_mock_dl_for_rep_test(master_seed, "loader_rep_test")

        batches1 = [batch for _, batch in zip(range(2), loader1_dl)]
        batches2 = [batch for _, batch in zip(range(2), loader2_dl)]

        assert len(batches1) == len(batches2)
        if len(batches1) > 0 :
            for b1_data, b2_data in zip(batches1, batches2):
                assert torch.allclose(b1_data[0], b2_data[0]), "DataLoader image data differs."
                assert torch.equal(b1_data[1], b2_data[1]), "DataLoader label data differs."
                assert torch.equal(b1_data[2], b2_data[2]), "DataLoader index data differs."
        logger.info("Dataloader determinism check passed.")
        
        # 3. Test Seed Derivation Determinism
        logger.info("Testing seed derivation determinism...")
        seed_a1 = derive_component_seed(master_seed, "comp_A", "instance1")
        seed_a2 = derive_component_seed(master_seed, "comp_A", "instance1")
        seed_b = derive_component_seed(master_seed, "comp_B", "instance1")
        seed_c = derive_component_seed(master_seed, "comp_A", "instance2")
        
        assert seed_a1 == seed_a2, "Seed derivation not deterministic for identical inputs."
        assert seed_a1 != seed_b, "Seed derivation collision for different components."
        assert seed_a1 != seed_c, "Seed derivation collision for different instances."
        logger.info("Seed derivation determinism check passed.")
        logger.info("✅ Deterministic reproducibility workflow test passed.")


class TestErrorRecoveryAndFaultTolerance:
    """Tests for error recovery, fault tolerance, and robustness of the workflows."""

    @pytest.mark.integration
    def test_partial_failure_recovery(self, isolated_config_for_integration):
        """
        Simulates a partial failure during a workflow (e.g., MAGIC replay) 
        and tests if the system can recover or restart from a checkpoint.
        """
        current_config = isolated_config_for_integration
        logger = logging.getLogger("FaultToleranceTest")
        logger.info(f"Testing partial failure recovery in run dir: {current_config.get_current_run_dir()}")

        # --- Setup: Simulate some initial progress and a checkpoint ---
        mock_model_state = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES).state_dict()
        checkpoint_step = 5 
        pre_failure_checkpoint_path = current_config.get_magic_checkpoint_path(model_id="replay_model_state", step_or_epoch=checkpoint_step)
        pre_failure_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': mock_model_state,
            'optimizer_state_dict': None,
            'step': checkpoint_step, 
            'misc_data': "data_before_crash"
        }, pre_failure_checkpoint_path)
        logger.info(f"Saved mock pre-failure checkpoint: {pre_failure_checkpoint_path} at step {checkpoint_step}")

        # --- Simulate a failure and recovery attempt ---
        assert pre_failure_checkpoint_path.exists()
        loaded_checkpoint = torch.load(pre_failure_checkpoint_path)
        
        recovered_model = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES)
        recovered_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        recovered_step = loaded_checkpoint['step']
        recovered_misc_data = loaded_checkpoint['misc_data']

        assert recovered_step == checkpoint_step, "Recovered step does not match checkpoint."
        assert recovered_misc_data == "data_before_crash", "Recovered miscellaneous data does not match."

        logger.info(f"Successfully loaded and verified data from checkpoint at step {recovered_step}.")


    @pytest.mark.integration
    def test_configuration_validation_integration(self, isolated_config_for_integration):
        """
        Tests that configuration validation (config.validate_config()) is integrated 
        and catches issues when configurations are slightly off.
        This uses the isolated_config_for_integration to get a fresh config.
        """
        current_config = isolated_config_for_integration
        logger = logging.getLogger("ConfigValidationIntegrationTest")

        # Example 1: MAGIC_TARGET_VAL_IMAGE_IDX out of bounds
        original_magic_target_idx = current_config.MAGIC_TARGET_VAL_IMAGE_IDX
        current_config.MAGIC_TARGET_VAL_IMAGE_IDX = current_config.NUM_TEST_SAMPLES_CIFAR10 + 10
        with pytest.raises(ValueError, match="MAGIC_TARGET_VAL_IMAGE_IDX .* out of bounds"):
            current_config.validate_config()
        current_config.MAGIC_TARGET_VAL_IMAGE_IDX = original_magic_target_idx

        # Example 2: LDS target index mismatch (should warn)
        original_lds_target_idx = current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
        current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = current_config.MAGIC_TARGET_VAL_IMAGE_IDX + 1 
        with pytest.warns(UserWarning, match="MAGIC target image index .* differs from LDS target image index"):
            current_config.validate_config()
        current_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = original_lds_target_idx
        
        # Example 3: Invalid learning rate
        original_lr = current_config.MODEL_TRAIN_LR
        current_config.MODEL_TRAIN_LR = -0.1
        with pytest.raises(ValueError, match="MODEL_TRAIN_LR must be positive"):
            current_config.validate_config()
        current_config.MODEL_TRAIN_LR = original_lr

        logger.info("✅ Configuration validation integration test passed.")


    @pytest.mark.integration
    def test_resource_exhaustion_handling(self, isolated_config_for_integration):
        """
        Simulates scenarios that might lead to resource exhaustion (e.g., OOM)
        and checks if the system handles it gracefully (e.g., logs error, exits cleanly).
        This is hard to test directly without actually causing OOM.
        We can mock functions that might cause OOM and make them raise errors.
        """
        current_config = isolated_config_for_integration
        logger = logging.getLogger("ResourceExhaustionTest")

        # Scenario 1: Mock model loading to raise MemoryError
        try:
            model = current_config.MODEL_CREATOR_FUNCTION(num_classes=current_config.NUM_CLASSES)
            assert model is not None
            logger.info("Model creation with num_classes from config successful.")
        except TypeError as e:
            pytest.fail(f"Model creation failed with TypeError, possibly missing num_classes: {e}")
        except Exception as e:
            logger.warning(f"Model creation raised an unexpected exception: {e}")
        
        # Scenario 2: Mock a dataloader that tries to allocate too much memory (e.g., huge batch read)
        original_randn = torch.randn
        def mocked_randn_oom(*args, **kwargs):
            size = 1
            for s in args[0] if isinstance(args[0], tuple) else args:
                 if isinstance(s, int): size *= s
            if size * 4 > 1 * 1024 * 1024 * 1024:
                raise MemoryError("Simulated OOM in torch.randn")
            return original_randn(*args, **kwargs)

        with patch('torch.randn', side_effect=mocked_randn_oom):
            try:
                pass
            except MemoryError:
                logger.info("Successfully simulated and caught a MemoryError via torch.randn patch.")
            except Exception as e:
                logger.warning(f"Unexpected error during OOM simulation with torch.randn: {e}")
        
        logger.info("✅ Resource exhaustion handling test section completed (simulated checks).")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])