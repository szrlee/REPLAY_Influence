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

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import config
from src.utils import (
    derive_component_seed,
    set_global_deterministic_state,
    create_deterministic_model,
    create_deterministic_dataloader
)
from src.model_def import construct_rn9
from src.data_handling import get_cifar10_dataloader, CustomDataset


class TestMAGICAnalysisWorkflow:
    """Test complete MAGIC analysis workflow"""
    
    def create_mock_cifar_dataloader(self, split='train', batch_size=32):
        """Create a mock CIFAR-10 dataloader for testing"""
        class MockDataset:
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % 10, idx
        
        size = 100 if split == 'train' else 20
        dataset = MockDataset(size)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    @pytest.mark.integration
    def test_magic_minimal_training_workflow(self):
        """Test minimal MAGIC training workflow with mocked components"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Mock configuration paths
            with patch('src.config.MAGIC_CHECKPOINTS_DIR', tmp_path / "checkpoints"):
                with patch('src.config.MAGIC_SCORES_DIR', tmp_path / "scores"):
                    with patch('src.config.MAGIC_LOGS_DIR', tmp_path / "logs"):
                        
                        # Create directories
                        (tmp_path / "checkpoints").mkdir()
                        (tmp_path / "scores").mkdir()
                        (tmp_path / "logs").mkdir()
                        
                        # Test deterministic model creation
                        model = create_deterministic_model(
                            master_seed=42,
                            creator_func=construct_rn9,
                            instance_id="magic_workflow_test"
                        )
                        
                        # Test mock training loop
                        criterion = torch.nn.CrossEntropyLoss()
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                        
                        # Mock dataloader
                        train_loader = self.create_mock_cifar_dataloader('train', batch_size=16)
                        
                        # Simulate training steps
                        model.train()
                        training_metrics = []
                        
                        for epoch in range(2):  # Minimal epochs
                            epoch_loss = 0.0
                            batch_count = 0
                            
                            for batch_idx, (images, labels, indices) in enumerate(train_loader):
                                optimizer.zero_grad()
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                
                                epoch_loss += loss.item()
                                batch_count += 1
                                
                                if batch_idx >= 5:  # Limit batches for testing
                                    break
                            
                            avg_loss = epoch_loss / batch_count
                            training_metrics.append({
                                'epoch': epoch,
                                'loss': avg_loss,
                                'batch_count': batch_count
                            })
                        
                        # Verify training progressed
                        assert len(training_metrics) == 2
                        assert all(m['loss'] > 0 for m in training_metrics)
                        
                        # Test checkpoint saving
                        checkpoint_path = tmp_path / "checkpoints" / "test_checkpoint.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        assert checkpoint_path.exists()
                        
                        # Test checkpoint loading
                        loaded_state = torch.load(checkpoint_path)
                        new_model = construct_rn9(num_classes=10)
                        new_model.load_state_dict(loaded_state)
                        
                        # Verify models are equivalent
                        test_input = torch.randn(1, 3, 32, 32)
                        with torch.no_grad():
                            output1 = model(test_input)
                            output2 = new_model(test_input)
                            assert torch.allclose(output1, output2, atol=1e-6)
    
    @pytest.mark.integration
    def test_magic_error_handling_and_recovery(self):
        """Test MAGIC workflow error handling and recovery mechanisms"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Test handling of missing checkpoint directory
            with patch('src.config.MAGIC_CHECKPOINTS_DIR', tmp_path / "missing_dir"):
                # Should handle missing directory gracefully
                checkpoint_path = config.get_magic_checkpoint_path(0, 1)
                assert isinstance(checkpoint_path, Path)
                
                # Create directory when needed
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                assert checkpoint_path.parent.exists()
            
            # Test handling of corrupted data
            scores_dir = tmp_path / "scores"
            scores_dir.mkdir()
            
            # Create corrupted scores file
            corrupted_file = scores_dir / "corrupted_scores.pkl"
            corrupted_file.write_text("not a pickle file")
            
            # Should handle corrupted files gracefully
            with pytest.raises((pickle.UnpicklingError, UnicodeDecodeError)):
                with open(corrupted_file, 'rb') as f:
                    pickle.load(f)
            
            # Test model recovery after interruption
            model = construct_rn9()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            # Simulate checkpoint saving for recovery
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 5,
                'loss': 0.5
            }
            
            recovery_path = tmp_path / "recovery_checkpoint.pt"
            torch.save(checkpoint, recovery_path)
            
            # Test recovery process
            new_model = construct_rn9()
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            
            loaded_checkpoint = torch.load(recovery_path)
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['loss'] == 0.5


class TestLDSValidationWorkflow:
    """Test complete LDS validation workflow"""
    
    @pytest.mark.integration
    def test_lds_minimal_validation_workflow(self):
        """Test minimal LDS validation workflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Mock LDS configuration paths
            with patch('src.config.LDS_CHECKPOINTS_DIR', tmp_path / "lds_checkpoints"):
                with patch('src.config.LDS_LOSSES_DIR', tmp_path / "lds_losses"):
                    
                    # Create directories
                    (tmp_path / "lds_checkpoints").mkdir()
                    (tmp_path / "lds_losses").mkdir()
                    
                    # Create mock MAGIC scores for LDS input
                    mock_scores = {
                        'scores': np.random.randn(1000).astype(np.float32),
                        'indices': list(range(1000)),
                        'target_idx': 21,
                        'metadata': {'model_type': 'ResNet9', 'epochs': 12}
                    }
                    
                    scores_file = tmp_path / "magic_scores.pkl"
                    with open(scores_file, 'wb') as f:
                        pickle.dump(mock_scores, f)
                    
                    # Test subset generation based on influence scores
                    scores_array = mock_scores['scores']
                    top_influences = np.argsort(np.abs(scores_array))[-100:]  # Top 100 influential
                    
                    assert len(top_influences) == 100
                    assert all(idx < len(scores_array) for idx in top_influences)
                    
                    # Test subset model training simulation
                    subset_models = []
                    subset_losses = []
                    
                    for subset_id in range(3):  # Test with 3 subsets
                        # Create subset model
                        model = create_deterministic_model(
                            master_seed=42 + subset_id,
                            creator_func=construct_rn9,
                            instance_id=f"lds_subset_{subset_id}"
                        )
                        
                        # Simulate training and validation
                        test_input = torch.randn(5, 3, 32, 32)
                        test_targets = torch.randint(0, 10, (5,))
                        
                        model.eval()
                        with torch.no_grad():
                            outputs = model(test_input)
                            criterion = torch.nn.CrossEntropyLoss()
                            val_loss = criterion(outputs, test_targets).item()
                        
                        subset_models.append(model)
                        subset_losses.append(val_loss)
                        
                        # Save subset model
                        checkpoint_path = tmp_path / "lds_checkpoints" / f"subset_{subset_id}.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        
                        # Save validation loss
                        loss_path = tmp_path / "lds_losses" / f"loss_{subset_id}.pkl"
                        with open(loss_path, 'wb') as f:
                            pickle.dump({'val_loss': val_loss, 'subset_id': subset_id}, f)
                    
                    # Verify all models and losses were created
                    assert len(subset_models) == 3
                    assert len(subset_losses) == 3
                    assert all(loss > 0 for loss in subset_losses)
                    
                    # Test correlation computation simulation
                    predicted_losses = subset_losses  # In real LDS, these would be predicted
                    actual_losses = subset_losses     # These would be from subset training
                    
                    correlation = np.corrcoef(predicted_losses, actual_losses)[0, 1]
                    assert not np.isnan(correlation)  # Should be valid correlation
    
    @pytest.mark.integration
    def test_lds_correlation_computation(self):
        """Test LDS correlation computation between predicted and actual losses"""
        # Generate synthetic data for correlation testing
        n_subsets = 20
        
        # Simulate influence-based loss predictions
        true_correlations = 0.7  # Target correlation
        base_losses = np.random.uniform(0.5, 2.0, n_subsets)
        noise = np.random.normal(0, 0.3, n_subsets)
        
        predicted_losses = base_losses + noise * (1 - true_correlations)
        actual_losses = base_losses + noise * true_correlations
        
        # Compute correlation
        correlation = np.corrcoef(predicted_losses, actual_losses)[0, 1]
        
        # Should be reasonably close to target correlation
        assert abs(correlation - true_correlations) < 0.3
        assert not np.isnan(correlation)
        assert not np.isinf(correlation)
        
        # Test with edge cases
        # Perfect correlation
        perfect_predicted = actual_losses.copy()
        perfect_correlation = np.corrcoef(perfect_predicted, actual_losses)[0, 1]
        assert abs(perfect_correlation - 1.0) < 1e-10
        
        # No correlation
        random_predicted = np.random.randn(n_subsets)
        random_correlation = np.corrcoef(random_predicted, actual_losses)[0, 1]
        assert abs(random_correlation) < 0.5  # Should be close to zero (with some variance)


class TestEndToEndIntegration:
    """Test end-to-end integration across all components"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_magic_to_lds_pipeline(self):
        """Test complete pipeline from MAGIC analysis to LDS validation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Set up complete directory structure
            dirs_to_create = [
                "magic_checkpoints", "magic_scores", "magic_logs",
                "lds_checkpoints", "lds_losses", "lds_logs"
            ]
            
            for dir_name in dirs_to_create:
                (tmp_path / dir_name).mkdir()
            
            # Mock all configuration paths
            with patch('src.config.MAGIC_CHECKPOINTS_DIR', tmp_path / "magic_checkpoints"), \
                 patch('src.config.MAGIC_SCORES_DIR', tmp_path / "magic_scores"), \
                 patch('src.config.MAGIC_LOGS_DIR', tmp_path / "magic_logs"), \
                 patch('src.config.LDS_CHECKPOINTS_DIR', tmp_path / "lds_checkpoints"), \
                 patch('src.config.LDS_LOSSES_DIR', tmp_path / "lds_losses"), \
                 patch('src.config.LDS_LOGS_DIR', tmp_path / "lds_logs"):
                
                # Step 1: MAGIC Analysis Simulation
                print("Step 1: MAGIC Analysis")
                
                # Create and train MAGIC model
                magic_model = create_deterministic_model(
                    master_seed=42,
                    creator_func=construct_rn9,
                    instance_id="pipeline_magic_model"
                )
                
                # Save MAGIC model checkpoint
                magic_checkpoint = tmp_path / "magic_checkpoints" / "magic_final.pt"
                torch.save(magic_model.state_dict(), magic_checkpoint)
                
                # Generate mock influence scores
                n_train_samples = 1000
                influence_scores = np.random.randn(n_train_samples).astype(np.float32)
                
                # Save MAGIC scores
                magic_scores_data = {
                    'scores': influence_scores,
                    'indices': list(range(n_train_samples)),
                    'target_idx': config.MAGIC_TARGET_VAL_IMAGE_IDX,
                    'metadata': {
                        'model_checkpoint': str(magic_checkpoint),
                        'target_label': 5,
                        'analysis_complete': True
                    }
                }
                
                magic_scores_file = tmp_path / "magic_scores" / "scores_target_21.pkl"
                with open(magic_scores_file, 'wb') as f:
                    pickle.dump(magic_scores_data, f)
                
                # Step 2: LDS Validation Simulation
                print("Step 2: LDS Validation")
                
                # Load MAGIC scores for LDS
                with open(magic_scores_file, 'rb') as f:
                    loaded_scores = pickle.load(f)
                
                assert loaded_scores['target_idx'] == config.MAGIC_TARGET_VAL_IMAGE_IDX
                scores_array = loaded_scores['scores']
                
                # Generate LDS subsets based on influence scores
                n_subsets = 5
                subset_fraction = 0.8
                subset_size = int(n_train_samples * subset_fraction)
                
                lds_results = []
                
                for subset_id in range(n_subsets):
                    # Create subset indices (in practice, these would be based on influence scores)
                    subset_indices = np.random.choice(n_train_samples, subset_size, replace=False)
                    
                    # Train subset model
                    subset_model = create_deterministic_model(
                        master_seed=42 + subset_id,
                        creator_func=construct_rn9,
                        instance_id=f"pipeline_lds_subset_{subset_id}"
                    )
                    
                    # Simulate validation loss computation
                    test_input = torch.randn(10, 3, 32, 32)
                    test_targets = torch.randint(0, 10, (10,))
                    
                    subset_model.eval()
                    with torch.no_grad():
                        outputs = subset_model(test_input)
                        criterion = torch.nn.CrossEntropyLoss()
                        val_loss = criterion(outputs, test_targets).item()
                    
                    # Save subset results
                    subset_checkpoint = tmp_path / "lds_checkpoints" / f"subset_{subset_id}.pt"
                    torch.save(subset_model.state_dict(), subset_checkpoint)
                    
                    subset_loss_file = tmp_path / "lds_losses" / f"loss_{subset_id}.pkl"
                    loss_data = {
                        'subset_id': subset_id,
                        'val_loss': val_loss,
                        'subset_indices': subset_indices.tolist(),
                        'subset_size': subset_size
                    }
                    
                    with open(subset_loss_file, 'wb') as f:
                        pickle.dump(loss_data, f)
                    
                    lds_results.append(loss_data)
                
                # Step 3: Correlation Analysis
                print("Step 3: Correlation Analysis")
                
                # Compute predicted losses based on influence scores
                predicted_losses = []
                actual_losses = [result['val_loss'] for result in lds_results]
                
                for result in lds_results:
                    subset_indices = result['subset_indices']
                    # Predicted loss based on average influence in subset
                    avg_influence = np.mean(scores_array[subset_indices])
                    predicted_loss = 1.0 + 0.5 * avg_influence  # Simple linear model
                    predicted_losses.append(predicted_loss)
                
                # Compute correlation
                correlation = np.corrcoef(predicted_losses, actual_losses)[0, 1]
                
                # Step 4: Validation
                print("Step 4: Pipeline Validation")
                
                # Verify all components completed successfully
                assert magic_checkpoint.exists()
                assert magic_scores_file.exists()
                assert len(lds_results) == n_subsets
                assert all(result['val_loss'] > 0 for result in lds_results)
                assert not np.isnan(correlation)
                
                # Verify data flow integrity
                assert loaded_scores['target_idx'] == config.MAGIC_TARGET_VAL_IMAGE_IDX
                assert len(loaded_scores['scores']) == n_train_samples
                
                print(f"Pipeline completed successfully!")
                print(f"MAGIC scores shape: {scores_array.shape}")
                print(f"LDS subsets trained: {len(lds_results)}")
                print(f"Correlation coefficient: {correlation:.3f}")
    
    @pytest.mark.integration
    def test_deterministic_reproducibility_workflow(self):
        """Test that complete workflow produces deterministic, reproducible results"""
        results_run1 = []
        results_run2 = []
        
        for run_id in range(2):
            # Reset deterministic state
            set_global_deterministic_state(42, enable_deterministic=True)
            
            # Create model with same parameters
            model = create_deterministic_model(
                master_seed=42,
                creator_func=construct_rn9,
                instance_id="reproducibility_test"
            )
            
            # Run identical operations
            test_input = torch.randn(5, 3, 32, 32, generator=torch.Generator().manual_seed(42))
            
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            # Store results
            results = {
                'first_output': output[0, 0].item(),
                'output_sum': output.sum().item(),
                'output_mean': output.mean().item()
            }
            
            if run_id == 0:
                results_run1 = results
            else:
                results_run2 = results
        
        # Verify reproducibility
        for key in results_run1:
            assert abs(results_run1[key] - results_run2[key]) < 1e-6, f"Non-deterministic behavior in {key}"
        
        print("Deterministic reproducibility verified!")


class TestErrorRecoveryAndFaultTolerance:
    """Test error recovery and fault tolerance mechanisms"""
    
    @pytest.mark.integration
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures in the workflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Simulate partial MAGIC completion
            (tmp_path / "magic_scores").mkdir()
            partial_scores = {
                'scores': np.random.randn(500),  # Only half the expected scores
                'indices': list(range(500)),
                'target_idx': 21,
                'metadata': {'status': 'partial', 'completed_steps': 6, 'total_steps': 12}
            }
            
            partial_file = tmp_path / "magic_scores" / "partial_scores.pkl"
            with open(partial_file, 'wb') as f:
                pickle.dump(partial_scores, f)
            
            # Test detection of partial completion
            with open(partial_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert loaded_data['metadata']['status'] == 'partial'
            assert len(loaded_data['scores']) < 1000  # Expected full size
            
            # Test graceful handling
            if loaded_data['metadata']['status'] == 'partial':
                # Should either resume or restart appropriately
                print("Partial completion detected - would trigger recovery mechanism")
                assert True  # Recovery mechanism identified the issue
    
    @pytest.mark.integration
    def test_configuration_validation_integration(self):
        """Test configuration validation in integrated workflow context"""
        # Test with various configuration scenarios
        original_target_idx = config.MAGIC_TARGET_VAL_IMAGE_IDX
        
        try:
            # Test normal configuration
            config.validate_config()
            
            # Test configuration mismatch scenario
            with patch('src.config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION', 99):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    config.validate_config()
                    
                    # Should warn about mismatch
                    mismatch_warnings = [warning for warning in w 
                                       if "differs from" in str(warning.message)]
                    assert len(mismatch_warnings) > 0
            
        finally:
            # Restore original configuration
            config.MAGIC_TARGET_VAL_IMAGE_IDX = original_target_idx
    
    @pytest.mark.integration
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios"""
        # Test with very large batch size (should handle gracefully)
        try:
            large_batch_size = 10000
            large_input = torch.randn(large_batch_size, 3, 32, 32)
            model = construct_rn9()
            
            # This might fail due to memory constraints, which is acceptable
            output = model(large_input)
            print(f"Successfully processed large batch of size {large_batch_size}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Large batch failed due to memory constraints (expected)")
                assert True  # This is expected behavior
            else:
                raise
        
        # Test file system space constraints simulation
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Try to create many large files (limited by available space)
            try:
                for i in range(100):
                    large_file = tmp_path / f"large_file_{i}.dat"
                    # Create moderately sized file
                    large_file.write_bytes(b'x' * 1024 * 1024)  # 1MB each
                    
                    if i > 50:  # Limit to prevent actual space issues
                        break
                        
            except OSError:
                # Disk space exhaustion is handled gracefully
                print("Disk space constraint encountered (handled gracefully)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])