import pickle
import warnings
import torch
import numpy as np
import torchvision
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Project-specific imports
from . import config # For config.VARIABLE access
# Specific imports from .config are removed; all config access is via "config."

from .utils import (
    setup_logging, 
    set_global_deterministic_state,
    create_deterministic_dataloader, 
    create_deterministic_model, 
    create_deterministic_optimizer, 
    create_deterministic_scheduler, 
    log_scheduler_info,
    derive_component_seed,
    deterministic_context,
    update_dataloader_epoch
)
from .model_def import construct_rn9
from .data_handling import (
    get_cifar10_dataloader, CustomDataset
)
from .visualization import plot_influence_images

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
        
        config.MAGIC_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        config.MAGIC_SCORES_DIR.mkdir(parents=True, exist_ok=True)
        config.MAGIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_batch_file_path(self, step: int) -> Path:
        """Get the file path for a specific batch step."""
        return config.MAGIC_CHECKPOINTS_DIR / f"batch_{step}.pkl"

    def _save_batch_to_disk(self, step: int, batch_data: Dict[str, torch.Tensor]) -> None:
        """Save a batch to disk for memory-efficient replay with error handling."""
        batch_file = self._get_batch_file_path(step)
        try:
            # Ensure directory exists
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use atomic write to prevent corruption
            temp_file = batch_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename to final location
            temp_file.rename(batch_file)
            
        except (OSError, pickle.PicklingError) as e:
            # Cleanup on failure
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save batch {step} to disk: {e}") from e

    def _load_batch_from_disk(self, step: int) -> Dict[str, torch.Tensor]:
        """Load a batch from disk for memory-efficient replay with error handling."""
        batch_file = self._get_batch_file_path(step)
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
        
        try:
            with open(batch_file, 'rb') as f:
                return pickle.load(f)
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

    def _create_dataloader_and_model(self) -> Tuple[torch.utils.data.DataLoader, int]:
        """Create training dataloader and model with proper seed management."""
        # Shared instance IDs are now sourced from config

        # Create dataloader with shared instance_id for complete consistency
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
        
        total_steps = config.MODEL_TRAIN_EPOCHS * len(train_loader)
        
        # Create model with shared instance_id for complete consistency with LDS
        self.model_for_training = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=construct_rn9,
            instance_id=config.SHARED_MODEL_INSTANCE_ID, # From config
            num_classes=config.NUM_CLASSES
        ).to(config.DEVICE)
        
        return train_loader, total_steps
    
    def _create_optimizer_and_scheduler(self, total_steps: int) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Create optimizer and scheduler with consistent configuration."""
        # Shared instance IDs are now sourced from config
        
        # Create optimizer with shared instance_id for complete consistency
        optimizer = create_deterministic_optimizer(
            master_seed=config.SEED,
            optimizer_class=torch.optim.SGD,
            model_params=self.model_for_training.parameters(),
            instance_id=config.SHARED_OPTIMIZER_INSTANCE_ID, # From config
            lr=config.MODEL_TRAIN_LR,
            momentum=config.MODEL_TRAIN_MOMENTUM,
            weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
        )
        
        # Create scheduler with shared instance_id for complete consistency
        scheduler = create_deterministic_scheduler(
            master_seed=config.SEED,
            optimizer=optimizer,
            schedule_type=config.LR_SCHEDULE_TYPE,
            total_steps=total_steps,
            instance_id=config.SHARED_SCHEDULER_INSTANCE_ID, # From config
            step_size=config.STEPLR_STEP_SIZE,
            gamma=config.STEPLR_GAMMA,
            t_max=config.COSINE_T_MAX,
            max_lr=config.ONECYCLE_MAX_LR,
            pct_start=config.ONECYCLE_PCT_START,
            anneal_strategy=config.ONECYCLE_ANNEAL_STRATEGY,
            div_factor=config.ONECYCLE_DIV_FACTOR,
            final_div_factor=config.ONECYCLE_FINAL_DIV_FACTOR
        )
        
        # Log scheduler information
        log_scheduler_info(scheduler, config.LR_SCHEDULE_TYPE, self.logger, "MAGIC")
        
        return optimizer, scheduler

    def train_and_collect_intermediate_states(self, force_retrain: bool = False) -> int:
        self.logger.info(f"Starting MAGIC training with device: {config.DEVICE}")
        self.logger.info(f"Memory efficient replay mode: {self.use_memory_efficient_replay}")
        
        # Create dataloader and model with proper seed management
        train_loader, total_steps = self._create_dataloader_and_model()
        
        # Create optimizer and scheduler
        optimizer, scheduler = self._create_optimizer_and_scheduler(total_steps)
        
        criterion = CrossEntropyLoss()
        
        # Check if training can be skipped
        if not force_retrain and config.BATCH_DICT_FILE.exists():
            try:
                with open(config.BATCH_DICT_FILE, 'rb') as f:
                    loaded_batch_dict = pickle.load(f)
                if loaded_batch_dict:
                    total_completed_iterations = max(loaded_batch_dict.keys())
                    final_iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_completed_iterations)
                    
                    # Check if batch_dict format matches current mode
                    sample_entry = next(iter(loaded_batch_dict.values()), None)
                    if sample_entry:
                        # Check format compatibility
                        is_loaded_memory_efficient = 'step' in sample_entry and len(sample_entry) == 1
                        
                        if is_loaded_memory_efficient != self.use_memory_efficient_replay:
                            self.logger.warning(f"Loaded batch_dict is from {'memory-efficient' if is_loaded_memory_efficient else 'regular'} mode, "
                                              f"but current mode is {'memory-efficient' if self.use_memory_efficient_replay else 'regular'}. Will retrain.")
                            force_retrain = True
                        elif final_iter_ckpt_path.exists():
                            # Additional check: in memory-efficient mode, verify batch files exist
                            if self.use_memory_efficient_replay:
                                missing_files = []
                                for step in range(1, min(total_completed_iterations + 1, 10)):  # Check first 10 files
                                    batch_file = self._get_batch_file_path(step)
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
                            force_retrain = True
            except Exception as e:
                self.logger.warning(f"Error validating batch_dict: {e}. Will retrain.")
                force_retrain = True
        
        if force_retrain:
            self.logger.info("Retraining: Clearing batch_dict_for_replay.")
            self.batch_dict_for_replay.clear()
        
        current_iteration_step = 0
        initial_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=current_iteration_step)
        torch.save(self.model_for_training.state_dict(), initial_ckpt_path)
        self.logger.info(f"Saved initial checkpoint to {initial_ckpt_path} (state before any training)")
        
        self.model_for_training.train()
        for epoch in range(config.MODEL_TRAIN_EPOCHS):
            # CRITICAL: Update dataloader epoch for deterministic shuffling
            update_dataloader_epoch(train_loader, epoch)
            
            self.logger.info(f"MAGIC Training: Epoch {epoch+1}/{config.MODEL_TRAIN_EPOCHS}")
            for batch_images, batch_labels, batch_indices in tqdm(train_loader, desc=f"MAGIC Epoch {epoch+1}"):
                current_iteration_step += 1
                
                # === STEP 1: Store training state BEFORE the training step ===
                
                # Get current learning rate (BEFORE scheduler.step())
                current_lr = optimizer.param_groups[0]['lr']
                
                # Get momentum buffers BEFORE optimizer.step()
                # These are the buffers that will be used for the current step
                momentum_buffers_for_step = []
                if config.MODEL_TRAIN_MOMENTUM > 0:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            param_state = optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                # Parameter has an existing momentum buffer
                                momentum_buffers_for_step.append(param_state['momentum_buffer'].cpu().clone())
                            else:
                                # Parameter doesn't have momentum buffer yet (first step or never had gradients)
                                momentum_buffers_for_step.append(torch.zeros_like(p.data).cpu())
                
                batch_data = {
                    'ims': batch_images.cpu().clone(), 
                    'labs': batch_labels.cpu().clone(), 
                    'idx': batch_indices.cpu().clone(),
                    'lr': current_lr  # CRITICAL: Store actual LR for each step
                }
                
                # CRITICAL: Store momentum buffers for exact replay
                if config.MODEL_TRAIN_MOMENTUM > 0:
                    batch_data['momentum_buffers'] = momentum_buffers_for_step
                
                self._store_batch_data(current_iteration_step, batch_data)
                
                # Store appropriate data in batch_dict_for_replay for progress tracking
                if self.use_memory_efficient_replay:
                    # In memory-efficient mode, just store step number for progress tracking
                    self.batch_dict_for_replay[current_iteration_step] = {'step': current_iteration_step}
                else:
                    # In regular mode, _store_batch_data already stored the full batch data
                    # No need to store again - batch_dict_for_replay is already populated by _store_batch_data
                    pass
                
                # === STEP 2: Execute training step EXACTLY as it will be replayed ===
                
                batch_images, batch_labels = batch_images.to(config.DEVICE), batch_labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = self.model_for_training(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()  # This modifies momentum buffers internal state
                
                # Apply scheduler AFTER optimizer.step() to match training order
                if scheduler:
                    scheduler.step()
                
                # === STEP 3: Save checkpoint after the training step ===
                
                iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=current_iteration_step)
                torch.save(self.model_for_training.state_dict(), iter_ckpt_path)
            
            self.logger.info(f"End of Epoch {epoch+1}. Total iterations completed: {current_iteration_step}. Last checkpoint: {iter_ckpt_path}. Final LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save progress information (in memory-efficient mode, this just saves step counts)
        with open(config.BATCH_DICT_FILE, 'wb') as f: pickle.dump(self.batch_dict_for_replay, f)
        self.logger.info(f"Saved batch_dict for replay to {config.BATCH_DICT_FILE}")
        self.logger.info(f"MAGIC model training finished. {current_iteration_step} total iterations (batches processed).")
        return current_iteration_step

    def compute_influence_scores(self, total_training_iterations: int, force_recompute: bool = False) -> Optional[np.ndarray]:
        scores_save_path = config.get_magic_scores_path(target_idx=config.MAGIC_TARGET_VAL_IMAGE_IDX)
        if not force_recompute and scores_save_path.exists():
            self.logger.info(f"Loading existing MAGIC scores from {scores_save_path}")
            with open(scores_save_path, 'rb') as f: return pickle.load(f)
        
        # Load batch_dict for progress tracking (contains actual data in regular mode, just step info in memory-efficient mode)
        if not self.batch_dict_for_replay:
            if config.BATCH_DICT_FILE.exists():
                self.logger.info(f"Batch_dict is empty in memory, loading from {config.BATCH_DICT_FILE}")
                with open(config.BATCH_DICT_FILE, 'rb') as f: self.batch_dict_for_replay = pickle.load(f)
                if not self.batch_dict_for_replay: 
                    raise RuntimeError("Loaded batch_dict is empty. Cannot compute influence scores.")
            else: 
                raise RuntimeError("batch_dict_for_replay not populated and no file found. Run training first.")
        
        # In memory-efficient mode, verify that batch files exist
        if self.use_memory_efficient_replay:
            self.logger.info("Verifying batch files exist for memory-efficient replay...")
            missing_files = []
            for step in range(1, total_training_iterations + 1):
                batch_file = self._get_batch_file_path(step)
                if not batch_file.exists():
                    missing_files.append(batch_file)
            if missing_files:
                raise RuntimeError(f"Missing batch files for memory-efficient replay: {missing_files[:5]}...")
        
        self.logger.info(f"Starting MAGIC influence score computation for target val_idx: {config.MAGIC_TARGET_VAL_IMAGE_IDX}")
        
        # Create replay model with component-specific seed derivation
        replay_model = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=construct_rn9,
            instance_id="magic_replay_model", # Unique instance_id for this model
            num_classes=config.NUM_CLASSES
        ).to(config.DEVICE)
        
        target_ds = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))]))
        
        # Validate target index is within bounds
        if config.MAGIC_TARGET_VAL_IMAGE_IDX >= len(target_ds) or config.MAGIC_TARGET_VAL_IMAGE_IDX < 0:
            raise ValueError(f"MAGIC_TARGET_VAL_IMAGE_IDX ({config.MAGIC_TARGET_VAL_IMAGE_IDX}) is out of bounds for validation dataset (size: {len(target_ds)})")
        
        target_im, target_lab, _ = target_ds[config.MAGIC_TARGET_VAL_IMAGE_IDX]
        target_im, target_lab = target_im.unsqueeze(0).to(config.DEVICE), torch.tensor([target_lab]).to(config.DEVICE)
        final_model_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_training_iterations)
        if not final_model_ckpt_path.exists(): raise FileNotFoundError(f"Final model checkpoint {final_model_ckpt_path} not found.")
        
        # REPLAY Algorithm step 1: Initialization
        # Calculate Δ_T = ∇_s_T φ(s_T), where s_T is the final model state and φ is the target loss.
        # This Δ_T represents how the final target loss is sensitive to changes in the final model parameters s_T.
        temp_model_for_target_grad = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=construct_rn9,
            instance_id="magic_target_gradient_model", # Unique instance_id
            num_classes=config.NUM_CLASSES
        ).to(config.DEVICE)
        
        temp_model_for_target_grad.load_state_dict(torch.load(final_model_ckpt_path, map_location=config.DEVICE))
        temp_model_for_target_grad.eval(); temp_model_for_target_grad.zero_grad()
        out_target = temp_model_for_target_grad(target_im)
        criterion_target = CrossEntropyLoss()
        loss_target = criterion_target(out_target, target_lab); loss_target.backward()
        # delta_k_plus_1 initially holds Δ_T (adjoint for the final state s_T).
        # As the loop proceeds (t from T-1 down to 0), it will hold Δ_{t+1}.
        delta_k_plus_1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in temp_model_for_target_grad.parameters()]
        
        # Properly cleanup GPU memory
        temp_model_for_target_grad.cpu()  # Move to CPU first
        del temp_model_for_target_grad
        if config.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()  # Clear GPU cache
        
        # CRITICAL: Initialize data_weights parameter for influence computation
        data_weights = torch.nn.Parameter(torch.ones(config.NUM_TRAIN_SAMPLES, device=config.DEVICE), requires_grad=True)
        contributions = []
        criterion_replay_no_reduction = CrossEntropyLoss(reduction='none')
        
        # Helper function for dot product computation
        def _param_list_dot_product(list1, list2):
            """Computes the dot product between two lists of tensors (parameters/gradients)."""
            res = torch.tensor(0., device=config.DEVICE)
            for p1, p2 in zip(list1, list2):
                if p1 is not None and p2 is not None:
                    res += (p1.reshape(-1) @ p2.reshape(-1))
            return res
        
        # REPLAY Algorithm step 2: Backward Iteration
        # Loop from t = T-1 down to 0 (in code: replay_step_idx from T down to 1, so t = replay_step_idx - 1)
        for replay_step_idx in tqdm(range(total_training_iterations, 0, -1), desc="MAGIC Replay Pass"):
            # s_t: model parameters at the start of forward step t (replay_step_idx - 1)
            s_k_checkpoint_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=replay_step_idx - 1)
            if not s_k_checkpoint_path.exists():
                self.logger.warning(f"Checkpoint {s_k_checkpoint_path} not found. Skipping step {replay_step_idx}.")
                contributions.append(torch.zeros_like(data_weights).detach().cpu().clone())
                continue
                
            replay_model.load_state_dict(torch.load(s_k_checkpoint_path, map_location=config.DEVICE))
            replay_model.train()  # Set to train for consistent behavior with training-time gradients
            
            # current_sk_params represents s_t (model state at start of current replayed step t)
            current_sk_params = list(replay_model.parameters())  # These are s_k
            for p in current_sk_params:
                p.requires_grad_(True)
                if p.grad is not None:
                    p.grad.detach_().zero_()
            
            # Get batch data for step t (replay_step_idx)
            batch_data = self._get_batch_data(replay_step_idx)
            b_ims, b_labs, b_idx = batch_data['ims'].to(config.DEVICE), batch_data['labs'].to(config.DEVICE), batch_data['idx']
            
            # CRITICAL: Use the EXACT learning rate from training for step t
            stored_lr = batch_data['lr']
            
            # Compute L_t(s_t, w): batch loss at step t, which depends on s_t (current_sk_params) and w (data_weights)
            train_output = replay_model(b_ims)  # Uses s_t
            loss_samples = criterion_replay_no_reduction(train_output, b_labs)
            weighted_loss = (loss_samples * data_weights[b_idx]).mean() # This is L_t(s_t, w)
            
            # Compute ∇_s_t L_t(s_t, w): gradient of batch loss w.r.t. s_t.
            # create_graph=True is vital as this gradient is part of s_{t+1}(w).
            grad_of_batch_loss_wrt_sk_tuple = torch.autograd.grad(weighted_loss, current_sk_params, create_graph=True, allow_unused=False)
            grad_of_batch_loss_wrt_sk = [g.clone() for g in grad_of_batch_loss_wrt_sk_tuple]
            
            # Compute s_{t+1}(w) = h_t(s_t, g_t(s_t, w))
            # This is the SGD update: s_{t+1}(w) = s_t - lr * ∇_s_t L_t(s_t, w)
            sk_plus_1_dependent_on_w = [sk_p - stored_lr * gk_p 
                                          for sk_p, gk_p in zip(current_sk_params, grad_of_batch_loss_wrt_sk)]
            
            # Form the scalar quantity Q_t = s_{t+1}(w)ᵀ Δ_{t+1}
            # delta_k_plus_1 here holds Δ_{t+1} (adjoint from previous backward step, or Δ_T for the first step)
            scalar_for_gradients = _param_list_dot_product(sk_plus_1_dependent_on_w, delta_k_plus_1)
            
            # REPLAY Algorithm step 2.a: Calculate local influence contribution β_t
            # β_t = ∇_w Q_t = ∇_w (s_{t+1}(w)ᵀ Δ_{t+1})
            if data_weights.grad is not None:
                data_weights.grad.detach_().zero_()
            # grads_beta[0] is β_t, the influence of w on f(w) via this specific step t.
            grads_beta = torch.autograd.grad(scalar_for_gradients, data_weights, retain_graph=True, allow_unused=False)
            beta_t_step = grads_beta[0]
            contributions.append(beta_t_step.detach().cpu().clone())
            
            # REPLAY Algorithm step 2.b: Propagate the adjoint backward to get Δ_t
            # Δ_t = ∇_s_t Q_t = ∇_s_t (s_{t+1}(w)ᵀ Δ_{t+1})
            # This new delta_k (Δ_t) will be used as Δ_{t+1} in the next backward iteration (for step t-1).
            delta_k_new_grads_tuple = torch.autograd.grad(scalar_for_gradients, current_sk_params, allow_unused=False)
            # Update delta_k_plus_1 to be Δ_t for the next iteration.
            delta_k_plus_1 = [g.clone() for g in delta_k_new_grads_tuple]
        
        if not contributions:
            self.logger.warning("No influence contributions recorded during MAGIC analysis.")
            return np.zeros(config.NUM_TRAIN_SAMPLES)
        
        # Filter out any potential all-zero tensors if steps were skipped
        valid_contributions = [c for c in contributions if c.abs().sum() > 0]
        if not valid_contributions:
            self.logger.warning("All contributions were zero. Returning zero scores.")
            return np.zeros(config.NUM_TRAIN_SAMPLES)
            
        # REPLAY Algorithm step 3: Sum the Contributions
        # Total influence ∇_w f(w) = β = Σ β_t
        total_scores = torch.stack(valid_contributions).sum(axis=0).numpy()
        
        with open(scores_save_path, 'wb') as f: pickle.dump(total_scores, f)
        self.logger.info(f"Saved MAGIC influence scores to {scores_save_path}")
        return total_scores

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
            scores_flat = scores.sum(axis=0)
            self.logger.info(f"Using per-step scores format, summing across {scores.shape[0]} steps")
        elif scores.ndim == 1:
            # New format: [samples] - already flattened
            if scores.shape[0] != config.NUM_TRAIN_SAMPLES:
                raise ValueError(f"Expected total scores [samples], got {scores.shape}")
            scores_flat = scores
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
        plot_save_path = config.MAGIC_PLOTS_DIR / f"magic_influence_val_{config.MAGIC_TARGET_VAL_IMAGE_IDX}.png"
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
#             analyzer.plot_magic_influences(per_step_scores_or_path=per_step_scores)
#     else:
#         print("Training was skipped, cannot compute scores or plot.") 