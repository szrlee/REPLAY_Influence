import pickle
import warnings
import torch
import numpy as np
import torchvision
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
# SGD is imported directly where needed or via create_deterministic_optimizer
import os 
# matplotlib.pyplot is not directly used in this class
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any # Added Any and other necessary types
import logging
import time
import json
from datetime import datetime # Added for replay log timestamping
# OrderedDict is not needed for list-based manual replay

# Project-specific imports
from . import config 
# Specific imports from .config are removed; all config access is via "config."

from .utils import (
    create_deterministic_dataloader, 
    create_deterministic_model, 
    create_primary_training_optimizer, 
    save_training_metrics,
    get_run_config_as_dict,
    log_environment_info,
    update_dataloader_epoch,
    create_effective_scheduler,
    save_json_log_entry
)
# from .model_def import construct_resnet9_paper # Removed this import
from .data_handling import (
    get_cifar10_dataloader, CustomDataset
)
from .visualization import plot_influence_images

# --- REMOVED Functional API Setup for Replay ---
# functional_call_to_use = None
# grad_to_use = None
# functional_optim_sgd = None
# functional_optim_adam = None
# mfb_fallback_for_call = None 
# fmodel_fallback = None     

# try:
#     from torch.func import functional_call as torch_functional_call, grad as torch_func_grad
#     functional_call_to_use = torch_functional_call
#     grad_to_use = torch_func_grad
#     logging.info("Using 'torch.func.functional_call' and 'torch.func.grad' for replay.")
# except ImportError:
#     try:
#         from functorch import grad as functorch_grad 
#         grad_to_use = functorch_grad
#         logging.info("Using 'functorch.grad' for replay. 'torch.func.functional_call' not found.")
#     except ImportError:
#         logging.error("Could not import grad from torch.func or functorch. Functional replay will not be possible.")
        

# if functional_call_to_use is None and grad_to_use is not None: 
#     logging.warning("'torch.func.functional_call' not found. Attempting to use make_functional_with_buffers as fallback for replay.")
#     try:
#         from functorch import make_functional_with_buffers as functorch_mfb
#         mfb_fallback_for_call = functorch_mfb
#         logging.info("Using 'functorch.make_functional_with_buffers' as fallback for replay model execution.")
#     except ImportError:
#         try:
#             from torch.func import make_functional_with_buffers as torch_mfb
#             mfb_fallback_for_call = torch_mfb
#             logging.info("Using 'torch.func.make_functional_with_buffers' (from torch.func) as fallback for replay model execution.")
#         except ImportError:
#             logging.error("Fallback 'make_functional_with_buffers' also not found. Replay will likely fail if functional_call was needed.")

# # Import functional optimizer steps
# try:
#     from torch.optim import _functional as F_optim
#     functional_optim_sgd = F_optim.sgd
#     functional_optim_adam = F_optim.adam
#     logging.info("Successfully imported torch.optim._functional.sgd and torch.optim._functional.adam for replay.")
# except ImportError:
#     logging.error("Could not import from torch.optim._functional. Functional optimizer steps for replay will not be available.")

logger = logging.getLogger(__name__) # Define logger here after imports

class MagicAnalyzer:
    def __init__(self, use_memory_efficient_replay: bool = False) -> None:
        self.model_for_training: Optional[torch.nn.Module] = None 
        self.batch_dict_for_replay: Dict[int, Dict[str, torch.Tensor]] = {}
        self.use_memory_efficient_replay = use_memory_efficient_replay
        self.logger = logging.getLogger('influence_analysis.magic')
        
        if self.use_memory_efficient_replay:
            self.logger.info("Using memory-efficient batch replay (streaming from disk)")
        else:
            self.logger.info("Using in-memory batch replay (faster but memory-intensive)")
        
        config.get_magic_checkpoints_dir().mkdir(parents=True, exist_ok=True)
        config.get_magic_scores_dir().mkdir(parents=True, exist_ok=True)
        config.get_magic_plots_dir().mkdir(parents=True, exist_ok=True)
        config.get_magic_logs_dir().mkdir(parents=True, exist_ok=True)
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_batch_file_path(self, step: int) -> Path:
        """Get the file path for a specific batch step."""
        return config.get_magic_checkpoints_dir() / f"batch_{step}.pkl"

    def _save_batch_to_disk(self, step: int, batch_data: Dict[str, torch.Tensor]) -> None:
        """Save a batch to disk for memory-efficient replay with enhanced error handling."""
        batch_file = self._get_batch_file_path(step)
        try:
            # Ensure directory exists
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate batch data before saving
            if not batch_data or not isinstance(batch_data, dict):
                raise ValueError(f"Invalid batch_data for step {step}: {type(batch_data)}")
            
            # Check for required keys
            required_keys = {'ims', 'labs', 'idx', 'lr'}
            missing_keys = required_keys - set(batch_data.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys in batch_data for step {step}: {missing_keys}")
            
            # Use atomic write to prevent corruption
            temp_file = batch_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify file was written correctly
            if not temp_file.exists() or temp_file.stat().st_size == 0:
                raise RuntimeError(f"Failed to write batch file for step {step}: file empty or missing")
            
            # Atomic rename to final location
            temp_file.rename(batch_file)
            
            self.logger.debug(f"Successfully saved batch {step} to disk ({batch_file.stat().st_size} bytes)")
            
        except (OSError, pickle.PicklingError, ValueError) as e:
            # Cleanup on failure
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save batch {step} to disk: {e}") from e

    def _load_batch_from_disk(self, step: int) -> Dict[str, torch.Tensor]:
        """Load a batch from disk for memory-efficient replay with enhanced error handling."""
        batch_file = self._get_batch_file_path(step)
        
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
        
        # Check file size
        file_size = batch_file.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Batch file for step {step} is empty: {batch_file}")
        
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            # Validate loaded data
            if not isinstance(batch_data, dict):
                raise RuntimeError(f"Invalid batch data type for step {step}: {type(batch_data)}")
            
            # Check for required keys
            required_keys = {'ims', 'labs', 'idx', 'lr'}
            missing_keys = required_keys - set(batch_data.keys())
            if missing_keys:
                raise RuntimeError(f"Corrupted batch data for step {step}, missing keys: {missing_keys}")
            
            self.logger.debug(f"Successfully loaded batch {step} from disk ({file_size} bytes)")
            return batch_data
            
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            raise RuntimeError(f"Failed to load batch {step} from disk: {e}") from e

    def _get_batch_data(self, step: int) -> Dict[str, torch.Tensor]:
        """Get batch data either from memory or disk."""
        if self.use_memory_efficient_replay:
            return self._load_batch_from_disk(step)
        else:
            return self.batch_dict_for_replay[step]

    def _store_batch_data(self, step: int, batch_data: Dict[str, torch.Tensor]) -> None:
        """Store batch data either in memory or on disk."""
        if self.use_memory_efficient_replay:
            self._save_batch_to_disk(step, batch_data)
        else:
            self.batch_dict_for_replay[step] = batch_data

    def _save_training_metrics(self, metrics_data: Dict, stage: str = "training") -> None:
        """
        Save training metrics and hyperparameters to disk using the utils.save_json_log_entry.
        Ensures training logs are saved as a list of JSON entries.
        """
        log_file_path = config.get_magic_training_log_path()
        # Construct the log entry structure expected by save_training_metrics in utils
        # or adapt if save_json_log_entry is called directly.
        # For consistency with how utils.save_training_metrics structures it:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(time.time())),
            "stage": stage,
            "data": metrics_data
        }
        # MagicAnalyzer logs don't typically have a model_id in the same way LDS does per log file.
        # If there was a concept of a MAGIC model_id, it would be added here.
        save_json_log_entry(log_entry, log_file_path, is_json_lines=False)

    def _save_replay_metrics(self, metrics_data: Dict, stage: str = "replay") -> None:
        """
        Save replay (influence computation) metrics to disk using utils.save_json_log_entry.
        Ensures replay logs are saved in JSON Lines format.
        """
        log_file_path = config.get_magic_replay_log_path()
        
        log_entry = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(time.time())),
            "stage": stage,
            **metrics_data
        }
        save_json_log_entry(log_entry, log_file_path, is_json_lines=True)

    def _log_epoch_metrics(self, epoch: int, total_epochs: int, 
                           completed_steps_end_epoch: int, 
                           avg_epoch_train_loss: Optional[float] = None, 
                           current_lr_end_epoch: Optional[float] = None,
                           epoch_duration_seconds: Optional[float] = None,
                           epoch_samples_processed: Optional[int] = None,
                           final_checkpoint_path_epoch_end: Optional[str] = None
                           ) -> None:
        """Log and save comprehensive per-epoch training metrics to a single entry."""
        
        epoch_summary_data = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "completed_steps_end_epoch": completed_steps_end_epoch,
            "current_lr_end_epoch": current_lr_end_epoch,
            "avg_epoch_train_loss": avg_epoch_train_loss,
            "epoch_duration_seconds": epoch_duration_seconds,
            "epoch_samples_processed": epoch_samples_processed,
            "final_checkpoint_path_epoch_end": final_checkpoint_path_epoch_end
        }
        
        # Remove None values for cleaner logs if some metrics are optional and not provided
        epoch_summary_data = {k: v for k, v in epoch_summary_data.items() if v is not None}
        
        self._save_training_metrics(epoch_summary_data, "epoch_summary")
        
        log_message = (
            f"End of Epoch {epoch}/{total_epochs}. Steps: {completed_steps_end_epoch}. "
            f"LR: {current_lr_end_epoch:.6f}. Train Loss: {avg_epoch_train_loss:.4f}. "
            f"Duration: {epoch_duration_seconds:.2f}s. Samples: {epoch_samples_processed}. "
            f"Checkpoint: {final_checkpoint_path_epoch_end}"
        )
        self.logger.info(log_message)

    def _create_dataloader_and_model(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
        """Create training dataloader and model with proper seed management."""
        # Shared instance IDs are now sourced from config

        # Create training dataloader with shared instance_id for complete consistency
        train_loader = create_deterministic_dataloader(
            master_seed=config.SEED,
            creator_func=get_cifar10_dataloader,
            instance_id=config.SHARED_DATALOADER_INSTANCE_ID, # From config
            batch_size=config.MODEL_TRAIN_BATCH_SIZE,
            split='train',
            shuffle=True,
            augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            root_path=config.CIFAR_ROOT
        )
        
        # Create validation dataloader
        val_loader = create_deterministic_dataloader(
            master_seed=config.SEED,
            creator_func=get_cifar10_dataloader,
            instance_id="magic_validation_dataloader", # Distinct ID for val loader
            batch_size=config.MODEL_TRAIN_BATCH_SIZE, # Can use same or different batch size
            split='val', # Specify validation split
            shuffle=False, # No shuffle for validation
            augment=False, # No augmentation for validation
            num_workers=config.DATALOADER_NUM_WORKERS,
            root_path=config.CIFAR_ROOT
        )
        
        total_steps = config.MODEL_TRAIN_EPOCHS * len(train_loader)
        
        # Create model with shared instance_id for complete consistency with LDS
        self.model_for_training = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=config.MODEL_CREATOR_FUNCTION,
            instance_id=config.SHARED_MODEL_INSTANCE_ID # From config
        ).to(config.DEVICE)
        
        return train_loader, val_loader, total_steps
    
    def _create_optimizer_and_scheduler(self, total_steps: int, steps_per_epoch: int) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Create optimizer and scheduler with consistent configuration, including warmup."""
        # optimizer_type_config = config.MODEL_TRAIN_OPTIMIZER
        # if optimizer_type_config.lower() != 'sgd':
        #     self.logger.warning(f"MAGIC: Configured optimizer is {optimizer_type_config}, but this reverted analyzer version only supports SGD. Proceeding with SGD.")
        
        # Parameter grouping logic is now centralized in create_primary_training_optimizer
        # self.logger.info(f"MAGIC: Using SGD optimizer with base LR={config.MODEL_TRAIN_LR}, Momentum={config.MODEL_TRAIN_MOMENTUM}, Nesterov={config.MODEL_TRAIN_NESTEROV}")
        # self.logger.info(f"       Bias/BN LR scaled by: {config.RESNET9_BIAS_SCALE}, Weight Decay for Bias/BN: 0.0")
        # self.logger.info(f"       Other params Weight Decay: {config.MODEL_TRAIN_WEIGHT_DECAY}")

        optimizer = create_primary_training_optimizer(
            model=self.model_for_training,
            master_seed=config.SEED,
            instance_id=config.SHARED_OPTIMIZER_INSTANCE_ID,
            optimizer_type_config=config.MODEL_TRAIN_OPTIMIZER, # Still pass for logging/warnings
            base_lr_config=config.MODEL_TRAIN_LR,
            momentum_config=config.MODEL_TRAIN_MOMENTUM,
            weight_decay_config=config.MODEL_TRAIN_WEIGHT_DECAY,
            nesterov_config=config.MODEL_TRAIN_NESTEROV,
            bias_lr_scale_config=config.RESNET9_BIAS_SCALE,
            component_logger=self.logger
        )
        
        # Centralized scheduler creation
        # For OneCycleLR with grouped params, max_lr should be a list matching the groups
        effective_max_lr_for_scheduler = [pg['lr'] for pg in optimizer.param_groups]

        scheduler = create_effective_scheduler(
            optimizer=optimizer,
            master_seed=config.SEED,
            shared_scheduler_instance_id=config.SHARED_SCHEDULER_INSTANCE_ID,
            total_epochs_for_run=config.MODEL_TRAIN_EPOCHS,
            steps_per_epoch_for_run=steps_per_epoch,
            effective_lr_for_run=effective_max_lr_for_scheduler, # Pass list of max_lr for OneCycleLR
            component_logger=self.logger,
            component_name="MAGIC"
        )
        
        return optimizer, scheduler

    def train_and_collect_intermediate_states(self, force_retrain: bool = False) -> int:
        """Enhanced training with better error handling and resource management."""
        self.logger.info(f"Starting MAGIC training with device: {config.DEVICE}")
        self.logger.info(f"Memory efficient replay mode: {self.use_memory_efficient_replay}")
        
        # Validate configuration before starting
        try:
            config.validate_config()
        except (ValueError, RuntimeError) as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Create dataloader and model with proper seed management
        try:
            train_loader, val_loader, total_steps = self._create_dataloader_and_model()
        except Exception as e:
            self.logger.error(f"Failed to create dataloader and model: {e}")
            raise RuntimeError(f"Model/dataloader creation failed: {e}") from e
        
        # Create optimizer and scheduler
        try:
            steps_per_epoch = len(train_loader)
            optimizer, scheduler = self._create_optimizer_and_scheduler(total_steps, steps_per_epoch)
        except Exception as e:
            self.logger.error(f"Failed to create optimizer and scheduler: {e}")
            raise RuntimeError(f"Optimizer/scheduler creation failed: {e}") from e

        # The old runtime_config block is removed.
        # All necessary config, including runtime details, is now gathered by get_run_config_as_dict.
        
        criterion = CrossEntropyLoss()
        
        # Check if training can be skipped
        if not force_retrain and config.get_batch_dict_file().exists():
            try:
                with open(config.get_batch_dict_file(), 'rb') as f:
                    loaded_batch_dict = pickle.load(f)
                if loaded_batch_dict:
                    total_completed_iterations = max(loaded_batch_dict.keys())
                    final_iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_completed_iterations)
                    
                    sample_entry = next(iter(loaded_batch_dict.values()), None)
                    if sample_entry:
                        is_loaded_memory_efficient = 'step' in sample_entry and len(sample_entry) == 1
                        # Check for Adam states to detect if from functional version and force retrain
                        is_from_functional_adam = 'adam_steps' in sample_entry # A key specific to Adam state saving
                        if is_from_functional_adam:
                            self.logger.warning("Loaded batch_dict appears to be from a functional Adam version. "
                                              "Reverting to manual SGD replay requires retraining.")
                            force_retrain = True
                        elif is_loaded_memory_efficient != self.use_memory_efficient_replay:
                            self.logger.warning(f"Loaded batch_dict is from {'memory-efficient' if is_loaded_memory_efficient else 'regular'} mode, "
                                              f"but current mode is {'memory-efficient' if self.use_memory_efficient_replay else 'regular'}. Will retrain.")
                            force_retrain = True
                        elif final_iter_ckpt_path.exists():
                            if self.use_memory_efficient_replay:
                                missing_files = []
                                for step_num in range(1, min(total_completed_iterations + 1, 10)):  # Check first few files
                                    batch_file = self._get_batch_file_path(step_num)
                                    if not batch_file.exists():
                                        missing_files.append(batch_file)
                                if missing_files:
                                    self.logger.warning(f"Memory-efficient mode but batch files missing: {missing_files[:3]}... Will retrain.")
                                    force_retrain = True
                            
                            if not force_retrain:
                                self.logger.info(f"Training seems complete. {total_completed_iterations} iterations found with compatible batch_dict and final checkpoint. Skipping training.")
                                self.batch_dict_for_replay = loaded_batch_dict
                                if hasattr(self, 'model_for_training') and self.model_for_training is not None:
                                    self.model_for_training.load_state_dict(torch.load(final_iter_ckpt_path, map_location=config.DEVICE))
                                return total_completed_iterations
                        else:
                            self.logger.warning(f"Batch_dict exists, but final checkpoint sd_0_{total_completed_iterations}.pt missing. Retraining needed.")
                            force_retrain = True # Missing final checkpoint implies incomplete training
            except Exception as e:
                self.logger.warning(f"Error validating batch_dict: {e}. Will retrain.")
                force_retrain = True
        
        if force_retrain:
            self.logger.info("Retraining: Clearing batch_dict_for_replay.")
            self.batch_dict_for_replay.clear()
            # Consider also clearing MAGIC_CHECKPOINTS_DIR if starting fresh
        
        # Use the enhanced utility function to get the comprehensive config_data dictionary
        config_data = get_run_config_as_dict(
            component_type="MAGIC",
            device=config.DEVICE,
            model_creator_func=config.MODEL_CREATOR_FUNCTION,
            model=self.model_for_training, # Pass instantiated model
            optimizer=optimizer, 
            scheduler=scheduler, 
            train_loader=train_loader, # Pass instantiated train_loader
            val_loader=val_loader,     # Pass instantiated val_loader
            total_epochs_for_run=config.MODEL_TRAIN_EPOCHS,
            steps_per_epoch=steps_per_epoch, 
            # effective_lr_base is now a list if using OneCycleLR with groups
            effective_lr_base=[pg['lr'] for pg in optimizer.param_groups] if optimizer.param_groups else config.MODEL_TRAIN_LR,
            magic_target_val_image_idx=config.MAGIC_TARGET_VAL_IMAGE_IDX,
            magic_num_influential_images_to_show=config.MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW,
            magic_is_memory_efficient_replay=self.use_memory_efficient_replay
        )
        
        self._save_training_metrics(config_data, "config") # Single save for all config info
        
        # Use the utility to log environment information
        log_environment_info(self._save_training_metrics, self.logger)
        
        current_iteration_step = 0
        initial_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=0)
        # Ensure checkpoint directory exists before saving
        initial_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_for_training.state_dict(), initial_ckpt_path)
        self.logger.info(f"Saved initial checkpoint to {initial_ckpt_path} (state before any training)")
        
        self.model_for_training.train()
        
        # Track training metrics
        training_start_time = time.time()
        epoch_losses = []
        
        for epoch in range(config.MODEL_TRAIN_EPOCHS):
            update_dataloader_epoch(train_loader, epoch)
            self.logger.info(f"MAGIC Training: Epoch {epoch+1}/{config.MODEL_TRAIN_EPOCHS}")
            
            epoch_start_time = time.time()
            epoch_loss_accumulator = 0.0
            epoch_samples_processed = 0
            
            for batch_images, batch_labels, batch_indices in tqdm(train_loader, desc=f"MAGIC Epoch {epoch+1}"):
                current_iteration_step += 1
                current_lr_group0 = optimizer.param_groups[0]['lr'] # LR for the first group
                
                batch_data_content = {
                    'ims': batch_images.cpu().clone(), 
                    'labs': batch_labels.cpu().clone(), 
                    'idx': batch_indices.cpu().clone(),
                    'lr': current_lr_group0 # Store base LR for reference, actual LRs can be per group
                }
                if config.MODEL_TRAIN_MOMENTUM > 0:
                    # Ensure momentum buffers are captured correctly, even if not yet initialized for all params
                    momentum_buffers_for_step = []
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            param_state = optimizer.state[p]
                            # .get() with a default handles params for which momentum_buffer is not yet set (e.g. first step)
                            momentum_buffers_for_step.append(param_state.get('momentum_buffer', torch.zeros_like(p.data)).cpu().clone())
                    batch_data_content['momentum_buffers'] = momentum_buffers_for_step
                
                # Removed the detailed logging of batch_data that was added for functional debugging
                self._store_batch_data(current_iteration_step, batch_data_content)
                
                if self.use_memory_efficient_replay:
                    self.batch_dict_for_replay[current_iteration_step] = {'step': current_iteration_step}
                
                batch_images, batch_labels = batch_images.to(config.DEVICE), batch_labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = self.model_for_training(batch_images)
                loss = criterion(outputs, batch_labels)
                
                # Check for NaN/Inf in loss before backpropagation
                if not torch.isfinite(loss).all():
                    self.logger.error(f"NaN/Inf detected in loss at iteration {current_iteration_step}. Loss value: {loss.item()}")
                    raise RuntimeError(f"Training loss became NaN/Inf at iteration {current_iteration_step}")
                
                # Track training loss
                epoch_loss_accumulator += loss.item() * batch_images.size(0)
                epoch_samples_processed += batch_images.size(0)
                
                loss.backward()
                
                # Check for NaN/Inf in gradients before optimizer step
                has_nan_grad = False
                for name, param in self.model_for_training.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        self.logger.error(f"NaN/Inf detected in gradients for parameter {name} at iteration {current_iteration_step}")
                        has_nan_grad = True
                
                if has_nan_grad:
                    raise RuntimeError(f"Model gradients became NaN/Inf at iteration {current_iteration_step}")
                
                optimizer.step() 
                if scheduler: scheduler.step()
                
                # Check for NaN/Inf in model parameters after optimizer step
                has_nan_param = False
                for name, param in self.model_for_training.named_parameters():
                    if not torch.isfinite(param).all():
                        self.logger.error(f"NaN/Inf detected in parameter {name} after optimizer step at iteration {current_iteration_step}")
                        has_nan_param = True
                
                if has_nan_param:
                    raise RuntimeError(f"Model parameters became NaN/Inf at iteration {current_iteration_step}")
                
                iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=current_iteration_step)
                # Ensure checkpoint directory exists before saving
                iter_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model_for_training.state_dict(), iter_ckpt_path)
            
            # Calculate epoch metrics
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            avg_epoch_loss = epoch_loss_accumulator / epoch_samples_processed if epoch_samples_processed > 0 else float('inf')
            epoch_losses.append(avg_epoch_loss)
            final_lr_epoch = optimizer.param_groups[0]['lr']
            
            # Call the consolidated epoch logging and saving method
            self._log_epoch_metrics(
                epoch=epoch+1, 
                total_epochs=config.MODEL_TRAIN_EPOCHS,
                completed_steps_end_epoch=current_iteration_step,
                avg_epoch_train_loss=avg_epoch_loss,
                current_lr_end_epoch=final_lr_epoch,
                epoch_duration_seconds=epoch_duration,
                epoch_samples_processed=epoch_samples_processed,
                final_checkpoint_path_epoch_end=str(iter_ckpt_path) # iter_ckpt_path is the latest checkpoint
            )
        
        # Save final training summary
        training_end_time = time.time()
        total_training_duration = training_end_time - training_start_time
        
        training_summary = {
            "total_training_duration_seconds": total_training_duration,
            "total_epochs_completed": config.MODEL_TRAIN_EPOCHS,
            "total_iterations_completed": current_iteration_step,
            "epoch_losses": epoch_losses,
            "final_checkpoint": str(iter_ckpt_path)
        }
        self._save_training_metrics(training_summary, "training_complete")
        
        with open(config.get_batch_dict_file(), 'wb') as f: pickle.dump(self.batch_dict_for_replay, f)
        self.logger.info(f"Saved batch_dict for replay to {config.get_batch_dict_file()}")
        self.logger.info(f"MAGIC model training finished. {current_iteration_step} total iterations (batches processed).")

        # --- Validation Step ---
        self.logger.info("Starting validation of the trained MAGIC model...")
        validation_start_time = time.time()
        
        self.model_for_training.eval() # Set model to evaluation mode
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0
        val_criterion = CrossEntropyLoss() # Use standard CE loss for validation

        with torch.no_grad(): # Disable gradient calculations for validation
            for val_images, val_labels, _ in tqdm(val_loader, desc="MAGIC Validation"):
                val_images, val_labels = val_images.to(config.DEVICE), val_labels.to(config.DEVICE)
                
                outputs = self.model_for_training(val_images)
                loss = val_criterion(outputs, val_labels)
                total_val_loss += loss.item() * val_images.size(0)
                
                _, predicted_labels = torch.max(outputs.data, 1)
                total_val_samples += val_labels.size(0)
                correct_val_predictions += (predicted_labels == val_labels).sum().item()

        validation_end_time = time.time()
        validation_duration = validation_end_time - validation_start_time
        
        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        val_accuracy = (correct_val_predictions / total_val_samples) * 100 if total_val_samples > 0 else 0.0

        self.logger.info(f"MAGIC Model Validation Complete:")
        self.logger.info(f"  Average Validation Loss: {avg_val_loss:.4f}")
        self.logger.info(f"  Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save validation metrics
        validation_metrics = {
            "validation_duration_seconds": validation_duration,
            "total_val_samples": total_val_samples,
            "correct_predictions": correct_val_predictions,
            "avg_val_loss": avg_val_loss,
            "val_accuracy_percent": val_accuracy,
            "validation_target_image_idx": config.MAGIC_TARGET_VAL_IMAGE_IDX
        }
        self._save_training_metrics(validation_metrics, "validation_complete")
        # --- End Validation Step ---

        # Explicitly delete the model used for training and clear cache
        # to free up GPU memory before influence computation starts.
        if hasattr(self, 'model_for_training') and self.model_for_training is not None:
            del self.model_for_training
            self.model_for_training = None # Ensure the attribute is cleared
            if config.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            self.logger.info("Cleaned up model_for_training and cleared CUDA cache after training and validation.")

        return current_iteration_step

    def load_reusable_training_artifacts(self) -> int:
        """
        Loads existing training artifacts (batch_dict, final checkpoint) for reuse 
        when skipping the main training phase. Verifies compatibility and existence.

        Returns:
            int: The total number of training iterations from the loaded artifacts.
        
        Raises:
            FileNotFoundError: If critical files (batch_dict, final checkpoint, batch files) are missing.
            RuntimeError: If there are compatibility issues (e.g., memory mode mismatch).
        """
        self.logger.info("Attempting to load reusable training artifacts to skip MAGIC training.")

        if not config.get_batch_dict_file().exists():
            raise FileNotFoundError(f"Cannot skip training: MAGIC batch dictionary file not found at {config.get_batch_dict_file()}")

        try:
            with open(config.get_batch_dict_file(), 'rb') as f:
                loaded_batch_dict = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading batch_dict from {config.get_batch_dict_file()}: {e}") from e

        if not loaded_batch_dict:
            raise RuntimeError(f"Loaded batch_dict from {config.get_batch_dict_file()} is empty.")

        self.batch_dict_for_replay = loaded_batch_dict
        total_training_iterations = max(self.batch_dict_for_replay.keys())
        self.logger.info(f"Successfully loaded batch_dict with {total_training_iterations} total iterations.")

        final_iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_training_iterations)
        if not final_iter_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot skip training: Final model checkpoint {final_iter_ckpt_path} (for iteration {total_training_iterations}) not found."
            )
        self.logger.info(f"Final checkpoint {final_iter_ckpt_path} verified.")

        # Check for compatibility with current memory mode
        sample_entry = next(iter(self.batch_dict_for_replay.values()), None)
        if sample_entry:
            is_loaded_memory_efficient = 'step' in sample_entry and len(sample_entry) == 1
            if is_loaded_memory_efficient != self.use_memory_efficient_replay:
                raise RuntimeError(
                    f"Memory mode mismatch: Loaded batch_dict is from {'memory-efficient' if is_loaded_memory_efficient else 'in-memory'} mode, "
                    f"but current analyzer mode is {'memory-efficient' if self.use_memory_efficient_replay else 'in-memory'}. "
                    f"Retraining or changing mode is required."
                )
            self.logger.info(f"Batch dictionary memory mode compatible ({self.use_memory_efficient_replay=}).")

            # Adam state check from functional version (already in train_and_collect...)
            is_from_functional_adam = 'adam_steps' in sample_entry
            if is_from_functional_adam:
                 raise RuntimeError("Loaded batch_dict appears to be from a functional Adam version. Reverted SGD replay cannot use it.")

        if self.use_memory_efficient_replay:
            self.logger.info("Verifying batch files for memory-efficient replay mode...")
            missing_files = []
            # Check a sample of files or all if feasible. For now, check first few and last.
            iterations_to_check = list(range(1, min(total_training_iterations + 1, 6))) # First 5
            if total_training_iterations > 5:
                iterations_to_check.append(total_training_iterations) # And the last one
            
            for step_num in iterations_to_check:
                if step_num not in self.batch_dict_for_replay: # Ensure the key actually exists
                    continue 
                batch_file = self._get_batch_file_path(step_num)
                if not batch_file.exists():
                    missing_files.append(batch_file)
            
            if missing_files:
                raise FileNotFoundError(
                    f"Memory-efficient mode selected, but some batch files are missing. Examples: {missing_files}. "
                    f"Cannot proceed with score computation without these files."
                )
            self.logger.info("Sampled batch files for memory-efficient replay verified.")
        
        # No model is trained or loaded into self.model_for_training here, 
        # as compute_influence_scores creates its own replay_model.
        self.logger.info("Reusable training artifacts loaded and verified successfully.")
        return total_training_iterations

    def _param_list_dot_product(self, list1: List[torch.Tensor], list2: List[torch.Tensor], current_replay_step: int) -> torch.Tensor:
        """Computes the dot product between two lists of tensors (parameters/gradients)."""
        res = torch.tensor(0., device=config.DEVICE) # Ensure result is on the correct device

        if len(list1) != len(list2):
            self.logger.error(f"[Replay Step {current_replay_step}] Mismatch in parameter list lengths for dot product: len(list1)={len(list1)}, len(list2)={len(list2)}")
            # This is a critical error, as zip will truncate, leading to incorrect computations.
            raise ValueError(f"[Replay Step {current_replay_step}] Parameter list length mismatch: {len(list1)} vs {len(list2)}")

        for i, (p1, p2) in enumerate(zip(list1, list2)):
            if p1 is None or p2 is None:
                self.logger.warning(f"[Replay Step {current_replay_step}] Dot product: p1 or p2 is None at index {i}. Skipping this pair.")
                continue
            if p1.shape != p2.shape:
                self.logger.error(f"[Replay Step {current_replay_step}] Critical shape mismatch in dot product at parameter index {i}: p1.shape={p1.shape}, p2.shape={p2.shape}")
                # This is the most likely root cause of the original cryptic PyTorch error.
                raise ValueError(f"[Replay Step {current_replay_step}] Shape mismatch at param index {i} for dot product: p1.shape={p1.shape} vs p2.shape={p2.shape}")
            
            # Proceed with dot product if shapes match
            try:
                res += torch.sum(p1 * p2) # Element-wise product and sum
            except RuntimeError as e:
                self.logger.error(f"[Replay Step {current_replay_step}] RuntimeError during torch.sum(p1 * p2) at index {i}: p1.shape={p1.shape}, p2.shape={p2.shape}, Error: {e}")
                raise e # Re-raise the original error after logging
        return res

    def _check_and_log_tensor_stats(self, tensor_or_list: Union[torch.Tensor, List[torch.Tensor], None], name: str, step: Optional[int] = None, log_values: bool = False):
        """Helper to check and log tensor statistics, including NaNs and Infs."""
        if tensor_or_list is None:
            self.logger.debug(f"Step {step if step is not None else 'N/A'} - {name}: Tensor is None")
            return True # Indicates problematic if None was not expected

        is_problematic = False
        if isinstance(tensor_or_list, torch.Tensor):
            tensors_to_check = [tensor_or_list]
        else: # Assuming list of tensors
            tensors_to_check = tensor_or_list

        for i, tensor in enumerate(tensors_to_check):
            if tensor is None:
                self.logger.warning(f"Step {step if step is not None else 'N/A'} - {name}[{i}]: Tensor is None")
                is_problematic = True
                continue

            # Only compute norm, min, max, mean for float/complex tensors
            if tensor.is_floating_point() or tensor.is_complex():
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                current_problem = has_nan or has_inf
                is_problematic = is_problematic or current_problem
                
                log_message = (
                    f"Step {step if step is not None else 'N/A'} - {name}"
                    f"{f'[{i}]' if isinstance(tensor_or_list, list) else ''}: "
                    f"Shape: {tensor.shape}, Norm: {tensor.norm().item():.4e}, "
                    f"Min: {tensor.min().item():.4e}, Max: {tensor.max().item():.4e}, "
                    f"Mean: {tensor.mean().item():.4e}, HasNaN: {has_nan}, HasInf: {has_inf}"
                )
                if current_problem:
                    self.logger.error(log_message) # Log as error if NaN/Inf
                else:
                    self.logger.debug(log_message)
                
                if log_values and not current_problem and tensor.numel() < 10: # Log small tensors if requested and not problematic
                     self.logger.debug(f"    Values: {tensor.data.cpu().numpy().tolist()}")
            else: # For integer tensors or other types
                # For integer tensors, NaN/Inf checks are less common unless they originated from float issues.
                # We can still log basic info.
                log_message = (
                    f"Step {step if step is not None else 'N/A'} - {name}"
                    f"{f'[{i}]' if isinstance(tensor_or_list, list) else ''}: "
                    f"Shape: {tensor.shape}, DType: {tensor.dtype}"
                )
                # Optionally, could check for specific out-of-range int values if necessary.
                self.logger.debug(log_message)
                if log_values and tensor.numel() < 20: # Log a few more for int labels
                    self.logger.debug(f"    Values: {tensor.data.cpu().numpy().tolist()}")
                    
        return is_problematic

    def _setup_replay_model_and_target(self, total_training_iterations: int) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, int, Path, CustomDataset]:
        """Sets up the model for replay and loads the target validation sample."""
        self.logger.info("Setting up replay model and target validation sample...")
        
        replay_model = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=config.MODEL_CREATOR_FUNCTION,
            instance_id="magic_replay_model_manual_sgd" 
        ).to(config.DEVICE)
        
        total_params = sum(p.numel() for p in replay_model.parameters())
        trainable_params = sum(p.numel() for p in replay_model.parameters() if p.requires_grad)
        self.logger.info(f"Replay model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")

        target_ds = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))]))
        
        if not (0 <= config.MAGIC_TARGET_VAL_IMAGE_IDX < len(target_ds)):
            raise ValueError(f"MAGIC_TARGET_VAL_IMAGE_IDX ({config.MAGIC_TARGET_VAL_IMAGE_IDX}) is out of bounds for validation dataset (size: {len(target_ds)})")
        
        target_im_tensor, target_lab_tensor, _ = target_ds[config.MAGIC_TARGET_VAL_IMAGE_IDX]
        target_im = target_im_tensor.unsqueeze(0).to(config.DEVICE)
        target_lab = torch.tensor([target_lab_tensor], device=config.DEVICE)
        self.logger.info(f"Target sample loaded - Image shape: {target_im.shape}, Label: {target_lab.item()}, Class: {target_lab_tensor}")
        
        final_model_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_training_iterations)
        if not final_model_ckpt_path.exists(): 
            raise FileNotFoundError(f"Final model checkpoint {final_model_ckpt_path} not found.")
        self.logger.info(f"Final model checkpoint identified: {final_model_ckpt_path}")

        return replay_model, target_im, target_lab, target_lab_tensor, final_model_ckpt_path, target_ds

    def _compute_initial_adjoint(self, final_model_ckpt_path: Path, target_im: torch.Tensor, target_lab: torch.Tensor) -> List[torch.Tensor]:
        """Computes the initial adjoint (gradient of target loss w.r.t. final model parameters)."""
        self.logger.info(f"Computing initial adjoint (delta_T) from checkpoint: {final_model_ckpt_path}")
        
        temp_model_for_target_grad = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=config.MODEL_CREATOR_FUNCTION,
            instance_id="magic_target_gradient_model_manual_sgd"
        ).to(config.DEVICE)
        
        temp_model_for_target_grad.load_state_dict(torch.load(final_model_ckpt_path, map_location=config.DEVICE))
        temp_model_for_target_grad.eval()
        temp_model_for_target_grad.zero_grad()
        
        out_target = temp_model_for_target_grad(target_im)
        loss_target = CrossEntropyLoss()(out_target, target_lab)
        
        predicted_class = torch.argmax(out_target, dim=1).item()
        prediction_confidence = torch.softmax(out_target, dim=1)[0, target_lab.item()].item() # Ensure target_lab is scalar for indexing
        self.logger.info(f"Final model prediction on target - Predicted: {predicted_class}, True: {target_lab.item()}, Confidence: {prediction_confidence:.4f}, Loss: {loss_target.item():.6f}")
        
        loss_target.backward()
        
        delta_k_plus_1 = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p.data) 
                          for p in temp_model_for_target_grad.parameters()]
        
        adjoint_norms = [d.norm().item() for d in delta_k_plus_1 if d is not None]
        total_adjoint_norm = sum(adjoint_norms)
        max_adjoint_norm = max(adjoint_norms) if adjoint_norms else 0.0
        self.logger.info(f"Initial adjoint computed - Total norm: {total_adjoint_norm:.6e}, Max param norm: {max_adjoint_norm:.6e}, Num params: {len(delta_k_plus_1)}")
        self._check_and_log_tensor_stats(delta_k_plus_1, "Initial delta_T (delta_k_plus_1)")
        
        del temp_model_for_target_grad
        if config.DEVICE.type == 'cuda': torch.cuda.empty_cache()
        return delta_k_plus_1, loss_target.item(), predicted_class, prediction_confidence # Return more info for logging

    def _simulate_sgd_step_for_replay(self, 
                                      replay_step_idx_k: int, # For logging
                                      model_params_sk_minus_1: List[torch.Tensor], 
                                      grads_Lk: List[torch.Tensor], 
                                      stored_lr_k_group0: float, 
                                      stored_momentum_buffers_k_cpu: Optional[List[torch.Tensor]],
                                      replay_model_named_params: List[Tuple[str, torch.nn.Parameter]],
                                      enable_param_clipping: bool, # New control parameter
                                      max_param_norm_warning: float,
                                      param_clip_norm_hard: float
                                     ) -> Tuple[List[torch.Tensor], bool]:
        """
        Simulates a single SGD step to get s_k(w_k) from s_{k-1}.
        Handles weight decay, momentum, Nesterov, and LR scaling per group.

        Returns:
            Tuple containing:
            - sk_dependent_on_w_list (List[torch.Tensor]): Updated parameters s_k(w_k).
            - param_warnings_occurred_this_step (bool): True if any param norm warnings/clipping occurred.
        """
        sk_dependent_on_w_list = []
        param_warnings_this_step_count = 0

        param_momentum_mapping = {}
        if config.MODEL_TRAIN_MOMENTUM > 0 and stored_momentum_buffers_k_cpu:
            momentum_buffer_idx = 0
            # This grouping logic must precisely match how optimizer groups were defined during training
            # and how momentum buffers were stored.
            for group_config in [
                {'is_bias_or_bn_func': lambda name, param_shape: not (len(param_shape) == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower())},
                {'is_bias_or_bn_func': lambda name, param_shape: (len(param_shape) == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower())}
            ]:
                for name, param in replay_model_named_params: # Use passed named_parameters
                    if not param.requires_grad: continue
                    if group_config['is_bias_or_bn_func'](name, param.shape):
                        if momentum_buffer_idx < len(stored_momentum_buffers_k_cpu):
                            param_momentum_mapping[id(param)] = stored_momentum_buffers_k_cpu[momentum_buffer_idx]
                            momentum_buffer_idx += 1
                        # else: error or warning if mismatch
        
        for i, (param_sk_minus_1, grad_param_Lk) in enumerate(zip(model_params_sk_minus_1, grads_Lk)):
            param_name, _ = replay_model_named_params[i] # Assumes order is consistent
            effective_grad_param_Lk = grad_param_Lk
            
            current_param_weight_decay = config.MODEL_TRAIN_WEIGHT_DECAY
            lr_for_this_param_group_replay = stored_lr_k_group0

            is_bias_or_bn = len(param_sk_minus_1.shape) == 1 or param_name.endswith(".bias") or "bn" in param_name.lower() or "norm" in param_name.lower()
            if is_bias_or_bn:
                 current_param_weight_decay = 0.0
                 lr_for_this_param_group_replay = stored_lr_k_group0 * config.RESNET9_BIAS_SCALE
            
            if current_param_weight_decay > 0:
                effective_grad_param_Lk = effective_grad_param_Lk.add(param_sk_minus_1.data, alpha=current_param_weight_decay)
            
            if config.MODEL_TRAIN_MOMENTUM > 0 and param_momentum_mapping:
                param_id = id(param_sk_minus_1)
                if param_id in param_momentum_mapping:
                    momentum_buffer_vk_minus_1 = param_momentum_mapping[param_id].to(config.DEVICE)
                    velocity_vk = momentum_buffer_vk_minus_1.mul(config.MODEL_TRAIN_MOMENTUM).add(effective_grad_param_Lk)
                    param_update_term = effective_grad_param_Lk.add(velocity_vk, alpha=config.MODEL_TRAIN_MOMENTUM) if config.MODEL_TRAIN_NESTEROV else velocity_vk
                    updated_param_sk = param_sk_minus_1.add(param_update_term, alpha=-lr_for_this_param_group_replay)
                else:
                    self.logger.warning(f"Step {replay_step_idx_k}: No momentum buffer found for param {param_name}. Applying update without momentum.")
                    updated_param_sk = param_sk_minus_1.add(effective_grad_param_Lk, alpha=-lr_for_this_param_group_replay)
                    param_warnings_this_step_count += 1 # Count as a warning/deviation
            else:
                updated_param_sk = param_sk_minus_1.add(effective_grad_param_Lk, alpha=-lr_for_this_param_group_replay)
            
            param_norm_val = updated_param_sk.norm().item()
            if enable_param_clipping: # Check if parameter clipping is enabled
                if param_norm_val > max_param_norm_warning:
                    self.logger.warning(f"Step {replay_step_idx_k}: Large param norm {param_norm_val:.4f} for {param_name} (threshold {max_param_norm_warning:.1f})")
                    param_warnings_this_step_count += 1
                if param_norm_val > param_clip_norm_hard:
                    clip_coef_param = param_clip_norm_hard / (param_norm_val + 1e-8)
                    updated_param_sk = updated_param_sk * clip_coef_param
                    self.logger.warning(f"Step {replay_step_idx_k}: Hard param clipping for {param_name} (norm {param_norm_val:.4f} > {param_clip_norm_hard:.1f}), coeff {clip_coef_param:.6f}")
                    param_warnings_this_step_count += 1
            sk_dependent_on_w_list.append(updated_param_sk)
        
        return sk_dependent_on_w_list, bool(param_warnings_this_step_count > 0)

    def _replay_single_step(self, 
                            replay_step_idx_k: int, 
                            replay_model: torch.nn.Module, 
                            current_delta_k_plus_1: List[torch.Tensor],
                            criterion_replay_no_reduction: torch.nn.CrossEntropyLoss,
                            enable_grad_clipping: bool, # Added
                            max_grad_norm: float,
                            enable_param_clipping: bool, # Added
                            max_param_norm_warning: float,
                            param_clip_norm_hard: float
                            ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]], bool, bool, bool]:
        """
        Processes a single step k of the MAGIC replay.

        Returns:
            Tuple containing:
            - beta_k (torch.Tensor, on CPU): Influence contribution for this step. None on fatal error.
            - next_delta_k (List[torch.Tensor]): New adjoint for the next step (k-1). None on fatal error.
            - grad_clipping_applied (bool): True if gradient clipping was applied.
            - param_warnings_occurred (bool): True if parameter norm warnings/clipping occurred.
            - fatal_error_occurred (bool): True if a NaN/Inf or critical issue stopped processing.
        """
        self.logger.debug(f"--- Starting Replay Step k = {replay_step_idx_k} ---")
        grad_clipping_applied_this_step = False
        # param_warnings_this_step = 0 # This will be returned by _simulate_sgd_step_for_replay

        if self._check_and_log_tensor_stats(current_delta_k_plus_1, "current_delta_k_plus_1 input check", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf detected in current_delta_k_plus_1 at start of step {replay_step_idx_k}. Terminating step.")
            return None, None, grad_clipping_applied_this_step, False, True # param_warnings is False here
        
        s_k_minus_1_checkpoint_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=replay_step_idx_k - 1)
        if not s_k_minus_1_checkpoint_path.exists():
            self.logger.warning(f"Checkpoint {s_k_minus_1_checkpoint_path} for s_{replay_step_idx_k-1} missing. Skipping k={replay_step_idx_k}.")
            return torch.zeros(config.NUM_TRAIN_SAMPLES, device='cpu'), current_delta_k_plus_1, grad_clipping_applied_this_step, False, False
            
        replay_model.load_state_dict(torch.load(s_k_minus_1_checkpoint_path, map_location=config.DEVICE))
        replay_model.train()
        current_sk_params = list(replay_model.parameters())
        if self._check_and_log_tensor_stats(current_sk_params, "current_sk_params (s_k-1)", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf in loaded s_k-1 at step {replay_step_idx_k}. Checkpoint corrupted. Terminating step.")
            return None, None, grad_clipping_applied_this_step, False, True

        for p in current_sk_params: 
            p.requires_grad_(True)
            if p.grad is not None: p.grad.detach_().zero_()
        
        batch_data_k = self._get_batch_data(replay_step_idx_k)
        b_ims_k, b_labs_k = batch_data_k['ims'].to(config.DEVICE), batch_data_k['labs'].to(config.DEVICE)
        b_idx_k = batch_data_k['idx'] 
        stored_lr_k_group0 = batch_data_k['lr'] 
        stored_momentum_buffers_k_cpu = batch_data_k.get('momentum_buffers') 

        train_output_k = replay_model(b_ims_k)
        per_sample_loss_k = criterion_replay_no_reduction(train_output_k, b_labs_k)
        current_batch_data_weights = self.data_weights_param[b_idx_k]
        weighted_loss_k = (per_sample_loss_k * current_batch_data_weights).mean()
        if self._check_and_log_tensor_stats(weighted_loss_k, "weighted_loss_k", step=replay_step_idx_k):
             self.logger.error(f"NaN/Inf in weighted_loss_k at step {replay_step_idx_k}. Terminating step.")
             return None, None, grad_clipping_applied_this_step, False, True

        grad_L_k_wrt_sk_minus_1_tuple = torch.autograd.grad(
            weighted_loss_k, current_sk_params, create_graph=True, allow_unused=False)
        grad_L_k_wrt_sk_minus_1_list = [g.clone() for g in grad_L_k_wrt_sk_minus_1_tuple]
        if self._check_and_log_tensor_stats(grad_L_k_wrt_sk_minus_1_list, "grad_L_k_wrt_sk_minus_1_list", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf in grad_L_k_wrt_sk_minus_1_list at step {replay_step_idx_k}. Terminating step.")
            return None, None, grad_clipping_applied_this_step, False, True
        
        total_norm_grads = torch.norm(torch.stack([torch.norm(g.detach()) for g in grad_L_k_wrt_sk_minus_1_list]))
        if enable_grad_clipping and total_norm_grads > max_grad_norm:
            clip_coef_grads = max_grad_norm / (total_norm_grads + 1e-6)
            grad_L_k_wrt_sk_minus_1_list = [g * clip_coef_grads for g in grad_L_k_wrt_sk_minus_1_list]
            self.logger.warning(f"Step {replay_step_idx_k}: Grad clipping coeff {clip_coef_grads:.6f} (norm {total_norm_grads:.4f})")
            grad_clipping_applied_this_step = True
        
        # Simulate SGD step using the new helper
        sk_dependent_on_w_list, param_warnings_occurred_this_step = \
            self._simulate_sgd_step_for_replay(
                replay_step_idx_k, current_sk_params, grad_L_k_wrt_sk_minus_1_list,
                stored_lr_k_group0, stored_momentum_buffers_k_cpu,
                list(replay_model.named_parameters()), # Pass named params for the helper
                enable_param_clipping, max_param_norm_warning, param_clip_norm_hard # Pass clipping controls
            )
        
        if self._check_and_log_tensor_stats(sk_dependent_on_w_list, "sk_dependent_on_w_list (s_k(w_k))", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf in sk_dependent_on_w_list at step {replay_step_idx_k}. Terminating step.")
            return None, None, grad_clipping_applied_this_step, param_warnings_occurred_this_step, True

        scalar_Qk_for_grads = self._param_list_dot_product(sk_dependent_on_w_list, current_delta_k_plus_1, replay_step_idx_k)
        if self._check_and_log_tensor_stats(scalar_Qk_for_grads, "scalar_Qk_for_grads", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf in scalar_Qk_for_grads at step {replay_step_idx_k}. Terminating step.")
            return None, None, grad_clipping_applied_this_step, param_warnings_occurred_this_step, True
        
        if self.data_weights_param.grad is not None: self.data_weights_param.grad.detach_().zero_()
        beta_k_full_grad_tuple = torch.autograd.grad(scalar_Qk_for_grads, self.data_weights_param, retain_graph=True, allow_unused=False)
        beta_k_full_grad = beta_k_full_grad_tuple[0]
        if self._check_and_log_tensor_stats(beta_k_full_grad, "beta_k_full_grad", step=replay_step_idx_k, log_values=True):
             self.logger.error(f"NaN/Inf in beta_k_full_grad at step {replay_step_idx_k}. Terminating step.")
             return None, None, grad_clipping_applied_this_step, param_warnings_occurred_this_step, True
        
        next_delta_k_tuple = torch.autograd.grad(scalar_Qk_for_grads, current_sk_params, allow_unused=False)
        next_delta_k_list = [g.clone().detach() for g in next_delta_k_tuple]
        if self._check_and_log_tensor_stats(next_delta_k_list, "next_delta_k (new adjoint)", step=replay_step_idx_k):
            self.logger.error(f"NaN/Inf in new next_delta_k at step {replay_step_idx_k}. Terminating step.")
            return beta_k_full_grad.detach().cpu().clone(), None, grad_clipping_applied_this_step, param_warnings_occurred_this_step, True
        
        self.logger.debug(f"--- Finished Replay Step k = {replay_step_idx_k} ---")
        return beta_k_full_grad.detach().cpu().clone(), next_delta_k_list, grad_clipping_applied_this_step, param_warnings_occurred_this_step, False

    def _perform_replay_loop(self, 
                             total_training_iterations: int, 
                             replay_model: torch.nn.Module, 
                             initial_delta_k_plus_1: List[torch.Tensor],
                             criterion_replay_no_reduction: torch.nn.CrossEntropyLoss
                             ) -> List[torch.Tensor]:
        """Performs the main backward replay loop to compute influence contributions per step."""
        self.logger.info(f"Starting main replay loop for {total_training_iterations} steps...")
        current_delta_k_plus_1 = initial_delta_k_plus_1 # Renamed for clarity within this scope
        contributions_per_step = []

        # Read clipping settings from config
        enable_grad_clipping = config.MAGIC_REPLAY_ENABLE_GRAD_CLIPPING
        max_grad_norm_config = config.MAGIC_REPLAY_MAX_GRAD_NORM
        enable_param_clipping = config.MAGIC_REPLAY_ENABLE_PARAM_CLIPPING
        max_param_norm_warning_config = config.MAGIC_REPLAY_MAX_PARAM_NORM_WARNING
        param_clip_norm_hard_config = config.MAGIC_REPLAY_PARAM_CLIP_NORM_HARD
        
        total_steps_with_grad_clipping = 0
        total_steps_with_param_warnings = 0
        steps_processed_in_loop = 0
        replay_loop_start_time = time.time()

        replay_loop_init_data = {
            "replay_loop_start_timestamp_iso": datetime.fromtimestamp(replay_loop_start_time).isoformat(),
            "total_steps_to_process": total_training_iterations,
            "gradient_clipping_settings": {
                "enabled": enable_grad_clipping,
                "max_grad_norm": max_grad_norm_config if enable_grad_clipping else "N/A",
            },
            "parameter_clipping_settings": {
                "enabled": enable_param_clipping,
                "max_param_norm_warning_threshold": max_param_norm_warning_config if enable_param_clipping else "N/A",
                "param_clip_norm_hard_threshold": param_clip_norm_hard_config if enable_param_clipping else "N/A"
            }
        }
        self._save_replay_metrics(replay_loop_init_data, "replay_loop_start_detailed") # New stage for clarity

        pbar = tqdm(range(total_training_iterations, 0, -1), desc="Manual SGD MAGIC Replay")
        for replay_step_idx_k in pbar:
            step_start_time = time.time()

            beta_k, next_delta_k, grad_clipped, param_warned, fatal_err = \
                self._replay_single_step(
                    replay_step_idx_k, replay_model, current_delta_k_plus_1, 
                    criterion_replay_no_reduction,
                    enable_grad_clipping, max_grad_norm_config, 
                    enable_param_clipping, max_param_norm_warning_config, param_clip_norm_hard_config
                )

            if fatal_err:
                self.logger.error(f"Fatal error in _replay_single_step at k={replay_step_idx_k}. Stopping replay loop.")
                # Log termination due to error from single step
                error_term_data = {
                    "error_timestamp_iso": datetime.now().isoformat(),
                    "error_step": replay_step_idx_k,
                    "reason": "Fatal error reported by _replay_single_step (e.g., NaN/Inf)",
                    "steps_completed_before_error": steps_processed_in_loop
                }
                self._save_replay_metrics(error_term_data, "replay_loop_terminated_by_step_error")
                break # Exit the loop
            
            contributions_per_step.append(beta_k)
            current_delta_k_plus_1 = next_delta_k # Use the new delta for the next iteration

            if grad_clipped: total_steps_with_grad_clipping += 1
            if param_warned: total_steps_with_param_warnings += 1
            steps_processed_in_loop += 1
            
            # Progress Logging (Milestones)
            if replay_step_idx_k % 100 == 0 or replay_step_idx_k in [total_training_iterations, total_training_iterations//2, 10, 1] or replay_step_idx_k == (total_training_iterations - steps_processed_in_loop +1) :
                self.logger.info(f"Replay Step {replay_step_idx_k}/{total_training_iterations} ({100*(total_training_iterations-replay_step_idx_k+1)/total_training_iterations:.1f}% complete)")
                current_gpu_mem = 0.0
                if config.DEVICE.type == 'cuda': current_gpu_mem = torch.cuda.memory_allocated() / 1e9
                progress_log_data = {
                    "step_number": replay_step_idx_k,
                    "progress_percentage": 100 * (total_training_iterations - replay_step_idx_k + 1) / total_training_iterations,
                    "elapsed_time_loop_seconds": time.time() - replay_loop_start_time,
                    "step_duration_seconds": time.time() - step_start_time,
                    "gpu_memory_allocated_gb": current_gpu_mem,
                    "cumulative_grad_clips": total_steps_with_grad_clipping,
                    "cumulative_param_warnings": total_steps_with_param_warnings
                }
                self._save_replay_metrics(progress_log_data, f"replay_progress_step_{replay_step_idx_k}")
            
            # Progress Bar Update
            new_adjoint_norm = sum(d.norm().item() for d in current_delta_k_plus_1 if d is not None)
            # Fetch LR for display; _get_batch_data is safe here as non-fatal error would have exited
            lr_display = self._get_batch_data(replay_step_idx_k).get('lr', 0.0) 
            pbar.set_postfix({
                "lr": f"{lr_display:.1e}", 
                "Δ_norm": f"{new_adjoint_norm:.1e}", 
                "gclip": "Y" if grad_clipped else "N", 
                "pwarn": "Y" if param_warned else "N" 
            })

        replay_loop_duration = time.time() - replay_loop_start_time
        self.logger.info("REPLAY LOOP COMPLETED")
        self.logger.info(f"Duration: {replay_loop_duration:.2f}s. Avg step: {replay_loop_duration/steps_processed_in_loop if steps_processed_in_loop > 0 else 0:.3f}s")
        self.logger.info(f"Steps with grad clipping: {total_steps_with_grad_clipping}. Steps with param warnings: {total_steps_with_param_warnings}")
        
        final_loop_stats = {
            "replay_loop_duration_seconds": replay_loop_duration,
            "average_time_per_step_seconds": replay_loop_duration / steps_processed_in_loop if steps_processed_in_loop > 0 else 0,
            "total_steps_processed_in_loop": steps_processed_in_loop,
            "expected_steps": total_training_iterations,
            "steps_with_gradient_clipping": total_steps_with_grad_clipping,
            "steps_with_parameter_warnings": total_steps_with_param_warnings,
            "loop_completion_status": "completed_fully" if steps_processed_in_loop == total_training_iterations else "completed_partially_due_to_step_error"
        }
        self._save_replay_metrics(final_loop_stats, "replay_loop_completion_stats")
        return contributions_per_step

    def _aggregate_and_save_scores(self, 
                                   contributions_per_step: List[torch.Tensor], 
                                   scores_save_path: Path, 
                                   replay_start_time: float,
                                   initial_memory_gb: Tuple[float, float],
                                   replay_setup_log_data: Dict[str, Any] # Pass the already prepared setup log data
                                   ) -> Optional[np.ndarray]:
        """Aggregates per-step contributions and saves the final influence scores."""
        if not contributions_per_step:
            self.logger.warning("No influence contributions recorded. Returning None.")
            # Log empty completion state
            empty_completion_data = {
                **replay_setup_log_data, # Include setup data
                "replay_completion_timestamp": time.time(),
                "total_computation_duration_seconds": time.time() - replay_start_time,
                "computation_status": "completed_empty_no_contributions",
                "final_influence_statistics": {"total_samples": 0, "num_nonzero_influences": 0}
            }
            self._save_replay_metrics(empty_completion_data, "replay_completion_empty")
            return None

        self.logger.info("Aggregating final influence scores...")
        summation_start_time = time.time()
        total_influence_scores_tensor = torch.stack(contributions_per_step).sum(axis=0)
        total_influence_scores_np = total_influence_scores_tensor.numpy()
        summation_duration = time.time() - summation_start_time

        num_nonzero = np.count_nonzero(total_influence_scores_np)
        max_pos = np.max(total_influence_scores_np) if num_nonzero > 0 else 0
        min_neg = np.min(total_influence_scores_np) if num_nonzero > 0 else 0
        mean_abs = np.mean(np.abs(total_influence_scores_np))
        std_dev = np.std(total_influence_scores_np)
        num_nans = np.isnan(total_influence_scores_np).sum()

        self.logger.info("FINAL INFLUENCE SCORE STATISTICS:")
        self.logger.info(f"  Summation time: {summation_duration:.3f}s")
        self.logger.info(f"  Shape: {total_influence_scores_np.shape}, Non-zero: {num_nonzero}")
        self.logger.info(f"  Max: {max_pos:.6e}, Min: {min_neg:.6e}, Mean Abs: {mean_abs:.6e}, Std: {std_dev:.6e}")
        if num_nans > 0: self.logger.error(f"CRITICAL: {num_nans} NaN values in final scores!")

        with open(scores_save_path, 'wb') as f: pickle.dump(total_influence_scores_np, f)
        self.logger.info(f"Saved Reverted MAGIC influence scores to {scores_save_path}")

        final_memory_allocated_gb, final_memory_cached_gb = 0.0, 0.0
        if config.DEVICE.type == 'cuda':
            final_memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
            final_memory_cached_gb = torch.cuda.memory_reserved() / 1e9
            self.logger.info(f"Final GPU memory - Allocated: {final_memory_allocated_gb:.2f}GB, Cached: {final_memory_cached_gb:.2f}GB")

        if hasattr(self, 'data_weights_param'): del self.data_weights_param
        if config.DEVICE.type == 'cuda': torch.cuda.empty_cache()
        
        total_replay_duration = time.time() - replay_start_time
        self.logger.info("=" * 60)
        self.logger.info("MAGIC INFLUENCE COMPUTATION COMPLETED SUCCESSFULLY")
        self.logger.info(f"Total computation time: {total_replay_duration:.2f} seconds")
        self.logger.info("=" * 60)

        # Merge replay_setup_log_data with completion data
        replay_completion_data = {
            **replay_setup_log_data,
            "replay_completion_timestamp": time.time(),
            "total_computation_duration_seconds": total_replay_duration,
            # Replay loop stats are now logged separately by _perform_replay_loop
            "summation_duration_seconds": summation_duration,
            "final_influence_statistics": {
                "total_samples": len(total_influence_scores_np),
                "shape": list(total_influence_scores_np.shape),
                "num_nonzero_influences": int(num_nonzero),
                "max_positive_influence": float(max_pos),
                "min_negative_influence": float(min_neg),
                "mean_absolute_influence": float(mean_abs),
                "standard_deviation": float(std_dev),
                "num_nan_values": int(num_nans)
            },
            "final_memory_usage_gb": {
                "allocated": final_memory_allocated_gb,
                "cached": final_memory_cached_gb,
                "initial_allocated": initial_memory_gb[0],
                "initial_cached": initial_memory_gb[1]
            },
            "output_files": {"scores_save_path": str(scores_save_path)},
            "computation_status": "completed_successfully" if num_nans == 0 else "completed_with_nans"
        }
        self._save_replay_metrics(replay_completion_data, "replay_completion_summary")
        return total_influence_scores_np

    def compute_influence_scores(self, total_training_iterations: int, force_recompute: bool = False) -> Optional[np.ndarray]:
        scores_save_path = config.get_magic_scores_path(target_idx=config.MAGIC_TARGET_VAL_IMAGE_IDX)
        if not force_recompute and scores_save_path.exists():
            self.logger.info(f"Loading existing MAGIC scores from {scores_save_path}")
            with open(scores_save_path, 'rb') as f: return pickle.load(f)
        
        replay_start_time = time.time()
        initial_memory_allocated_gb, initial_memory_cached_gb = 0.0, 0.0
        if config.DEVICE.type == 'cuda' and torch.cuda.is_available():
            initial_memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
            initial_memory_cached_gb = torch.cuda.memory_reserved() / 1e9
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING MAGIC INFLUENCE SCORE COMPUTATION (REPLAY)")
        self.logger.info(f"Target val_idx: {config.MAGIC_TARGET_VAL_IMAGE_IDX}, Total iterations: {total_training_iterations}")

        if not self.batch_dict_for_replay: self.load_reusable_training_artifacts()

        model_setup_start_time = time.time()
        replay_model, target_im, target_lab, target_lab_class_idx, final_model_ckpt_path, target_ds = \
            self._setup_replay_model_and_target(total_training_iterations)
        
        delta_k_plus_1, final_target_loss, final_pred_class, final_pred_conf = \
            self._compute_initial_adjoint(final_model_ckpt_path, target_im, target_lab)
        model_setup_duration = time.time() - model_setup_start_time

        # Prepare data for structured logging (part of it used by _aggregate_and_save_scores)
        replay_setup_log_data = {
            "replay_start_timestamp_iso": datetime.fromtimestamp(replay_start_time).isoformat(),
            "target_validation_image_idx": config.MAGIC_TARGET_VAL_IMAGE_IDX,
            "total_training_iterations_to_replay": total_training_iterations,
            "memory_efficient_replay_mode": self.use_memory_efficient_replay,
            "device": str(config.DEVICE),
            "replay_algorithm": "Manual SGD (Reverted)",
            "model_setup_duration_seconds": model_setup_duration,
            "replay_model_total_params": sum(p.numel() for p in replay_model.parameters()),
            "target_image_shape": list(target_im.shape),
            "target_label_class_idx": target_lab_class_idx,
            "final_model_checkpoint_path": str(final_model_ckpt_path),
            "final_model_prediction_on_target": {
                "predicted_class": final_pred_class, "true_class": target_lab.item(),
                "confidence_on_true_class": final_pred_conf, "loss": final_target_loss
            },
            "initial_adjoint_num_params": len(delta_k_plus_1),
            "initial_adjoint_total_norm": sum(d.norm().item() for d in delta_k_plus_1 if d is not None),
            "validation_dataset_size_for_target": len(target_ds)
        }
        self._save_replay_metrics(replay_setup_log_data, "replay_setup_complete")
        
        self.logger.info(f"Initializing data weights parameter for {config.NUM_TRAIN_SAMPLES} samples...")
        self.data_weights_param = torch.nn.Parameter(torch.ones(config.NUM_TRAIN_SAMPLES, device=config.DEVICE), requires_grad=True)
        criterion_replay_no_reduction = CrossEntropyLoss(reduction='none')

        # Perform the replay loop
        contributions_per_step = self._perform_replay_loop(
            total_training_iterations, replay_model, delta_k_plus_1, criterion_replay_no_reduction
        )

        # Aggregate scores and save
        return self._aggregate_and_save_scores(
            contributions_per_step, 
            scores_save_path, 
            replay_start_time, 
            (initial_memory_allocated_gb, initial_memory_cached_gb),
            replay_setup_log_data # Pass the setup log for final summary
        )

    def plot_magic_influences(self, per_step_scores_or_path: Union[np.ndarray, str, Path], num_images_to_show: Optional[int] = None) -> None:
        if num_images_to_show is None: num_images_to_show = config.MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW
        if isinstance(per_step_scores_or_path, (str, Path)):
            scores_path = Path(per_step_scores_or_path)
            if not scores_path.exists(): raise FileNotFoundError(f"Scores file not found: {scores_path}")
            with open(scores_path, 'rb') as f: scores = pickle.load(f)
        else: scores = per_step_scores_or_path
        
        # Handle both old format (per-step scores) and new format (total scores)
        if scores.ndim == 2:
            # Old format: [steps, samples] - sum across steps
            if scores.shape[1] != config.NUM_TRAIN_SAMPLES:
                raise ValueError(f"Expected per-step scores [steps, samples], got {scores.shape}")
            
            # Check for NaNs in scores before summing
            if np.isnan(scores).any():
                self.logger.error(f"NaNs detected in per-step scores BEFORE summing. Shape: {scores.shape}, Num NaNs: {np.isnan(scores).sum()}")
                nan_steps = np.where(np.isnan(scores).any(axis=1))[0]
                self.logger.error(f"Steps with NaNs: {nan_steps}")
                # Example: Log a few problematic rows if many NaNs
                if len(nan_steps) > 0:
                    for problematic_step_idx in nan_steps[:min(5, len(nan_steps))]:
                        self.logger.error(f"  Data for step index {problematic_step_idx} (original step {scores.shape[0] - problematic_step_idx}): {scores[problematic_step_idx, :]}")


            scores_flat = np.nansum(scores, axis=0) # Use nansum to be safe, but ideally, we find the source
            
            if np.isnan(scores_flat).any():
                 self.logger.error(f"NaNs detected in scores_flat AFTER nansum. Shape: {scores_flat.shape}, Num NaNs: {np.isnan(scores_flat).sum()}")
                 # This indicates all scores for some samples were NaN across all steps or a single step had an Inf that propagated.

            self.logger.info(f"Using per-step scores format, summing across {scores.shape[0]} steps. Result shape: {scores_flat.shape}")
        elif scores.ndim == 1:
            # New format: [samples] - already flattened
            if scores.shape[0] != config.NUM_TRAIN_SAMPLES:
                raise ValueError(f"Expected total scores [samples], got {scores.shape}")
            scores_flat = scores
            if np.isnan(scores_flat).any():
                self.logger.error(f"NaNs detected in loaded flat scores. Shape: {scores_flat.shape}, Num NaNs: {np.isnan(scores_flat).sum()}")
            self.logger.info(f"Using total scores format with {scores.shape[0]} samples")
        else:
            raise ValueError(f"Unexpected scores format with shape {scores.shape}")
            
        self.logger.info(f"Preparing to plot influential images for target {config.MAGIC_TARGET_VAL_IMAGE_IDX}...")
        viz_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_ds_viz = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=viz_transform)
        val_ds_viz = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True, transform=viz_transform)
        
        # Validate target index is within bounds
        if config.MAGIC_TARGET_VAL_IMAGE_IDX >= len(val_ds_viz) or config.MAGIC_TARGET_VAL_IMAGE_IDX < 0:
            raise ValueError(f"MAGIC_TARGET_VAL_IMAGE_IDX ({config.MAGIC_TARGET_VAL_IMAGE_IDX}) is out of bounds for validation dataset (size: {len(val_ds_viz)})")
        
        target_img_tensor, target_label, _ = val_ds_viz[config.MAGIC_TARGET_VAL_IMAGE_IDX]
        target_info = {'image': target_img_tensor, 'label': target_label, 'id_str': f"Val Idx {config.MAGIC_TARGET_VAL_IMAGE_IDX}"}
        train_info = {'dataset': train_ds_viz, 'name': 'CIFAR10 Train'}
        plot_save_path = config.get_magic_plots_dir() / f"magic_influence_val_{config.MAGIC_TARGET_VAL_IMAGE_IDX}.png"
        plot_influence_images(scores_flat=scores_flat, target_image_info=target_info, train_dataset_info=train_info,
            num_to_show=num_images_to_show, plot_title_prefix="MAGIC Analysis", save_path=plot_save_path)
        self.logger.info(f"Influence plot saved to {plot_save_path}")

# Optional: if __name__ == '__main__' for testing MagicAnalyzer class (keep if desired)
# if __name__ == '__main__':
#     print(f"Running MagicAnalyzer test with device: {config.DEVICE}")
#     analyzer = MagicAnalyzer()
#     total_steps = analyzer.train_and_collect_intermediate_states(force_retrain=False) 
#     if total_steps > 0:
#         per_step_scores = analyzer.compute_influence_scores(total_training_iterations=total_steps, force_recompute=False)
#         if per_step_scores is not None:
#             if np.isnan(per_step_scores).any():
#                 print(f"WARNING: NaNs found in computed per_step_scores. Num NaNs: {np.isnan(per_step_scores).sum()}")
#             analyzer.plot_magic_influences(per_step_scores_or_path=per_step_scores)
#     else:
#         print("Training was skipped, cannot compute scores or plot.") 