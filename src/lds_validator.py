import pickle
import warnings
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
from typing import List, Optional, Union, Tuple
from pathlib import Path
import logging
import time

# Project-specific imports
from . import config
from .utils import (
    create_deterministic_dataloader, 
    create_deterministic_model, 
    create_primary_training_optimizer,
    create_deterministic_scheduler,
    create_effective_scheduler,
    save_training_metrics,
    get_scheduler_config_as_dict,
    get_run_config_as_dict,
    update_dataloader_epoch,
)
# from .model_def import construct_resnet9_paper # Removed this import
from .data_handling import get_cifar10_dataloader
# LDS does not use the global batch_dict_for_replay from magic_analyzer

logger = logging.getLogger('influence_analysis.lds')

# Shared instance IDs are now sourced from config (e.g., config.SHARED_MODEL_INSTANCE_ID)

# verify_data_ordering_consistency function has been moved to tests/test_helpers.py

"""
=== LINEAR DATAMODELING SCORE (LDS) VALIDATION ===

This module implements the Linear Datamodeling Score (LDS) evaluation methodology for validating
predictive data attribution methods like MAGIC. The LDS framework is based on the theoretical
work from [IPE+22; PGI+23].

THEORETICAL BACKGROUND:
- Data Attribution Goal: Predict how changes in training data affect model behavior
- Model Output Function f(w): Maps data weight vector w to model performance metric
- Predictive Attribution f̃(w): Fast approximation of f(w) using influence scores
- LDS Metric: Spearman correlation between f̃(w) and f(w) across multiple data subsets

ALGORITHMIC PRINCIPLE:
1. Generate n random subsets of training data (represented as binary weight vectors w^(i))
2. For each subset i:
   a) Train a model using only samples in w^(i) → get true performance f(w^(i))
   b) Use MAGIC influence scores to predict performance f̃(w^(i))
3. Compute LDS = Spearman_correlation({f̃(w^(i))}, {f(w^(i))})

HIGH LDS CORRELATION → MAGIC accurately predicts how data changes affect model behavior
LOW LDS CORRELATION → MAGIC predictions are unreliable

KEY IMPLEMENTATION FEATURES:
- Uses identical model initialization across all subset models (via shared instance IDs)
- Maintains exact same data ordering as MAGIC analysis (via shared dataloader)
- Implements efficient subset training using weighted loss (avoids creating separate datasets)
- Focuses evaluation on the same target validation image used in MAGIC analysis
"""

def save_lds_training_metrics(model_id: int, metrics_data: dict, stage: str = "training") -> None:
    """
    Save LDS training metrics and hyperparameters to disk.
    
    Args:
        model_id: LDS model identifier
        metrics_data: Dictionary containing metrics to save
        stage: Stage of training ("training", "validation", "config", etc.)
    """
    log_file_path = config.get_lds_training_log_path(model_id)
    save_training_metrics(metrics_data, log_file_path, stage, model_id)


def generate_and_save_subset_indices(num_subsets: Optional[int] = None, total_samples: Optional[int] = None, 
                                     subset_fraction: Optional[float] = None, file_path: Optional[Path] = None, 
                                     force_regenerate: bool = False) -> List[np.ndarray]:
    """
    Generate random training data subsets for LDS validation.
    
    ALGORITHMIC PURPOSE:
    Creates n different binary weight vectors w^(1), w^(2), ..., w^(n) where each w^(i)
    represents a random subset of the full training data. These subsets will be used to
    train multiple models and evaluate how well MAGIC can predict their performance differences.
    
    THEORETICAL CONTEXT:
    In the LDS framework, each subset represents a different "dataset" defined by its weight vector.
    By training models on these different datasets and comparing predicted vs actual performance,
    we can validate the quality of our influence score estimates.
    
    Args:
        num_subsets: Number of different subsets to generate (default: config.LDS_NUM_SUBSETS_TO_GENERATE)
        total_samples: Total number of training samples available (default: config.NUM_TRAIN_SAMPLES)
        subset_fraction: Fraction of data to include in each subset (default: config.LDS_SUBSET_FRACTION)
        file_path: Where to save/load subset indices (default: config.LDS_INDICES_FILE)
        force_regenerate: Whether to regenerate even if file exists
        
    Returns:
        List of numpy arrays, each containing indices for one training subset
    """
    if num_subsets is None: num_subsets = config.LDS_NUM_SUBSETS_TO_GENERATE
    if total_samples is None: total_samples = config.NUM_TRAIN_SAMPLES # Use general NUM_TRAIN_SAMPLES
    if subset_fraction is None: subset_fraction = config.LDS_SUBSET_FRACTION
    if file_path is None:
        file_path = config.get_current_run_dir() / config.LDS_INDICES_FILE
    elif not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Try to load existing subsets to ensure reproducibility across runs
    if not force_regenerate and file_path.exists():
        logger.info(f"Loading subset indices from {file_path}")
        with open(file_path, 'rb') as f:
            indices_list = pickle.load(f)
        # Ensure we have enough subsets if config changed
        if len(indices_list) >= num_subsets:
            return indices_list[:num_subsets]
        else:
            logger.info(f"Found only {len(indices_list)} subsets, regenerating for {num_subsets}.")
    
    logger.info(f"Generating {num_subsets} subset indices...")
    np.random.seed(config.SEED) # Ensure reproducibility for index generation
    subset_sample_count = int(total_samples * subset_fraction)
    
    # Validate subset parameters
    if subset_sample_count <= 0:
        raise ValueError(f"Subset sample count ({subset_sample_count}) must be positive. Check LDS_SUBSET_FRACTION ({subset_fraction}) and NUM_TRAIN_SAMPLES ({total_samples}).")
    if subset_sample_count > total_samples:
        raise ValueError(f"Subset sample count ({subset_sample_count}) cannot exceed total samples ({total_samples}). Check LDS_SUBSET_FRACTION ({subset_fraction}).")
    
    logger.info(f"Generating {num_subsets} subsets of size {subset_sample_count} from {total_samples} total samples...")
    
    # Generate random subsets without replacement
    # Each subset represents a different "dataset" for LDS evaluation
    indices_list = []
    for i in range(num_subsets):
        try:
            # Create binary weight vector: w_i = 1 for selected samples, w_i = 0 for others
            subset_indices = np.random.choice(range(total_samples), subset_sample_count, replace=False)
            indices_list.append(subset_indices)
        except ValueError as e:
            raise ValueError(f"Failed to generate subset {i}: {e}. This might happen if subset_sample_count > total_samples.")
    
    # Save for reproducibility and efficiency in future runs
    with open(file_path, 'wb') as f:
        pickle.dump(indices_list, f)
    logger.info(f"Saved {num_subsets} subset indices to {file_path}")
    return indices_list


# Helper function for the core training epochs loop
def _perform_lds_training_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion_no_reduction: torch.nn.Module,
    shared_train_loader: torch.utils.data.DataLoader,
    data_weights_for_subset: torch.Tensor,
    epochs: int,
    device: torch.device,
    model_id: int # For logging within the loop
) -> Tuple[int, float]: # Returns total_batches_processed, total_samples_used
    """Performs the actual epoch and batch iteration for training an LDS model."""
    total_batches_processed = 0
    total_samples_used = 0.0 # Ensure float for sum of samples

    for epoch in range(epochs):
        update_dataloader_epoch(shared_train_loader, epoch)
        epoch_samples_used = 0.0 # Ensure float
        
        for batch_idx, (images, labels, original_indices) in enumerate(tqdm(shared_train_loader, desc=f"LDS Model {model_id}, Epoch {epoch+1}")):
            images, labels, original_indices = images.to(device), labels.to(device), original_indices.to(device)
            active_weights_in_batch = data_weights_for_subset[original_indices]
            sum_active_weights_in_batch = active_weights_in_batch.sum()

            if sum_active_weights_in_batch.item() == 0: 
                continue

            optimizer.zero_grad()
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            
            weighted_loss = (per_sample_loss * active_weights_in_batch).sum() / (sum_active_weights_in_batch + config.EPSILON_FOR_WEIGHTED_LOSS)
            
            weighted_loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_batches_processed += 1
            batch_samples_used = sum_active_weights_in_batch.item()
            epoch_samples_used += batch_samples_used
            
        total_samples_used += epoch_samples_used
        logger.debug(f"LDS Model {model_id} Epoch {epoch+1}: Used {epoch_samples_used:.0f} samples") # Log as int for samples
    
    return total_batches_processed, total_samples_used


def train_model_on_subset(model_id: int, train_subset_indices: List[int], device: torch.device, 
                         shared_train_loader: torch.utils.data.DataLoader,
                         epochs: Optional[int] = None, lr: Optional[float] = None, 
                         momentum: Optional[float] = None, weight_decay: Optional[float] = None) -> torch.nn.Module:
    """
    Train a model on a specific subset of training data for LDS validation.
    Enhanced with better error handling and progress tracking.
    
    ALGORITHMIC CONTEXT:
    This function implements the core LDS training step where we train a model using
    only a subset of the full training data (represented by train_subset_indices).
    The trained model's performance will be compared against MAGIC's predictions
    to compute the LDS correlation score.
    
    Args:
        model_id: Unique identifier for this LDS model
        train_subset_indices: Indices of training samples to include in this subset
        device: PyTorch device to use for training
        shared_train_loader: DataLoader for the full training set (shared across all LDS models)
        epochs: Number of training epochs (default: config.MODEL_TRAIN_EPOCHS)
        lr: Learning rate (default: config.MODEL_TRAIN_LR)
        momentum: SGD momentum (default: config.MODEL_TRAIN_MOMENTUM)
        weight_decay: L2 regularization strength (default: config.MODEL_TRAIN_WEIGHT_DECAY)
        
    Returns:
        torch.nn.Module: Trained model ready for evaluation
        
    Raises:
        RuntimeError: If training fails or subset is invalid
        ValueError: If configuration parameters are invalid
    """
    # Enhanced parameter validation
    if len(train_subset_indices) == 0:
        raise ValueError(f"LDS Model {model_id}: Empty subset indices provided")
    
    if len(train_subset_indices) < config.MODEL_TRAIN_BATCH_SIZE:
        logger.warning(f"LDS Model {model_id}: Subset size ({len(train_subset_indices)}) "
                      f"smaller than batch size ({config.MODEL_TRAIN_BATCH_SIZE}). "
                      f"Training may be unstable.")
    
    # Use config defaults if not specified
    if epochs is None: epochs = config.MODEL_TRAIN_EPOCHS
    if lr is None: lr = config.MODEL_TRAIN_LR
    if momentum is None: momentum = config.MODEL_TRAIN_MOMENTUM
    if weight_decay is None: weight_decay = config.MODEL_TRAIN_WEIGHT_DECAY
    
    # Validate training parameters
    if epochs <= 0 or lr <= 0:
        raise ValueError(f"LDS Model {model_id}: Invalid training parameters: epochs={epochs}, lr={lr}")
    
    logger.info(f"Training LDS Model {model_id} on subset of {len(train_subset_indices)} samples for {epochs} epochs")
    
    try:
        # Create model with same architecture and initialization as MAGIC
        # Use SHARED_MODEL_INSTANCE_ID to ensure identical initialization across all LDS models
        model = create_deterministic_model(
            master_seed=config.SEED,
            creator_func=config.MODEL_CREATOR_FUNCTION,
            instance_id=config.SHARED_MODEL_INSTANCE_ID # Same ID = Same model architecture and init weights
        ).to(device)
        
        logger.debug(f"LDS Model {model_id}: Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        raise RuntimeError(f"LDS Model {model_id}: Failed to create model: {e}") from e

    # Create data weights tensor for subset selection
    # This is the key LDS concept: w^(i) where w^(i)[j] = 1 if sample j is in subset i, 0 otherwise
    try:
        total_samples = config.NUM_TRAIN_SAMPLES
        data_weights_for_subset = torch.zeros(total_samples, dtype=torch.float32, device=device)
        
        # Validate indices are within bounds
        invalid_indices = [idx for idx in train_subset_indices if idx < 0 or idx >= total_samples]
        if invalid_indices:
            raise ValueError(f"LDS Model {model_id}: Invalid indices found: {invalid_indices[:5]}... "
                           f"(showing first 5). Valid range: [0, {total_samples-1}]")
        
        # Convert numpy array to list if needed to avoid PyTorch indexing issues
        if hasattr(train_subset_indices, 'tolist'):
            # train_subset_indices is a numpy array, convert to list
            indices_for_indexing = train_subset_indices.tolist()
        else:
            # train_subset_indices is already a list or other sequence
            indices_for_indexing = train_subset_indices
            
        data_weights_for_subset[indices_for_indexing] = 1.0
        effective_samples = data_weights_for_subset.sum().item()
        
        logger.info(f"LDS Model {model_id}: Using {effective_samples} samples from subset "
                   f"({effective_samples/total_samples:.1%} of full dataset)")
        
    except Exception as e:
        raise RuntimeError(f"LDS Model {model_id}: Failed to create data weights: {e}") from e

    # Create grouped optimizer parameters (same as MAGIC for consistency)
    try:
        optimizer = create_primary_training_optimizer(
            model=model,
            master_seed=config.SEED,
            instance_id=f"lds_model_{model_id}_{config.SHARED_OPTIMIZER_INSTANCE_ID}", # Model-specific instance ID
            optimizer_type_config='SGD', # LDS uses SGD
            base_lr_config=lr, # Use lr passed to this function
            momentum_config=momentum, # Use momentum passed to this function
            weight_decay_config=weight_decay, # Use weight_decay passed to this function
            nesterov_config=config.MODEL_TRAIN_NESTEROV,
            bias_lr_scale_config=config.RESNET9_BIAS_SCALE,
            component_logger=logger # LDS logger
        )
        
        logger.debug(f"LDS Model {model_id}: Created optimizer with {len(optimizer.param_groups)} parameter groups")
        
    except Exception as e:
        raise RuntimeError(f"LDS Model {model_id}: Failed to create optimizer: {e}") from e

    # Enhanced scheduler creation with proper error handling
    try:
        steps_per_epoch = len(shared_train_loader)
        effective_max_lr_for_scheduler = [pg['lr'] for pg in optimizer.param_groups]
        
        scheduler = create_effective_scheduler(
            optimizer=optimizer,
            master_seed=config.SEED,
            shared_scheduler_instance_id=f"lds_model_{model_id}_{config.SHARED_SCHEDULER_INSTANCE_ID}",
            total_epochs_for_run=epochs,
            steps_per_epoch_for_run=steps_per_epoch,
            effective_lr_for_run=effective_max_lr_for_scheduler,
            component_logger=logger,
            component_name=f"LDS Model {model_id}"
        )
        
    except Exception as e:
        logger.warning(f"LDS Model {model_id}: Failed to create scheduler, continuing without: {e}")
        scheduler = None

    # Training with enhanced monitoring and error handling
    try:
        model.train()
        criterion_no_reduction = CrossEntropyLoss(reduction='none') # Essential for weighted training
        
        training_start_time = time.time()
        logger.info(f"LDS Model {model_id}: Starting training...")
        
        total_batches_processed, total_samples_used = _perform_lds_training_epochs(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_no_reduction=criterion_no_reduction,
            shared_train_loader=shared_train_loader,
            data_weights_for_subset=data_weights_for_subset,
            epochs=epochs,
            device=device,
            model_id=model_id
        )
        
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        # Log comprehensive training statistics
        logger.info(f"LDS Model {model_id}: Training completed in {training_duration:.2f}s")
        logger.info(f"LDS Model {model_id}: Processed {total_batches_processed} batches, "
                   f"used {total_samples_used:.0f} weighted samples")
        
        # Save comprehensive training metrics
        training_metrics = {
            "model_id": model_id,
            "subset_size": len(train_subset_indices),
            "subset_indices": train_subset_indices,  # Store for reproducibility
            "effective_samples_used": total_samples_used,
            "total_batches_processed": total_batches_processed,
            "training_duration_seconds": training_duration,
            "training_epochs": epochs,
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "bias_scale": config.RESNET9_BIAS_SCALE
        }
        save_lds_training_metrics(model_id, training_metrics, "training_complete")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"LDS Model {model_id}: Training failed: {e}") from e


def evaluate_and_save_losses(model: torch.nn.Module, model_id: int, device: torch.device, 
                             val_loader: torch.utils.data.DataLoader,  # Added val_loader parameter
                             losses_dir: Optional[Path] = None) -> np.ndarray: # Removed batch_size, not needed if loader is pre-created
    """
    Evaluate a trained LDS model and compute per-sample validation losses.
    
    ALGORITHMIC PURPOSE:
    Implements the measurement function φ(θ) that maps trained model parameters θ to a scalar metric.
    Combined with training, this completes the model output function f(w) = φ(A(w)).
    
    THEORETICAL CONTEXT:
    In LDS validation:
    - Input: Trained model parameters θ (from train_model_on_subset)
    - Process: φ(θ) → evaluate model on validation set
    - Output: Per-sample losses, specifically loss on target validation image
    
    The target validation image loss will be used as f(w) in the LDS correlation analysis.
    
    Args:
        model: Trained PyTorch model to evaluate
        model_id: Unique identifier for this LDS model
        device: Computing device (CPU/GPU)
        val_loader: Validation dataloader (shared across all LDS models)
        losses_dir: Directory to save per-sample losses
        
    Returns:
        numpy array of per-sample validation losses
    """
    if losses_dir is None: losses_dir = config.get_lds_losses_dir()
    # batch_size parameter is removed as val_loader is now passed in.
    losses_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_losses_for_model = []
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    validation_start_time = time.time()
    total_val_samples = 0
    
    # Compute per-sample validation losses - this implements φ(θ)
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc=f"Evaluating LDS Model {model_id}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            all_losses_for_model.append(per_sample_loss.cpu().numpy())
            total_val_samples += images.size(0)
    
    validation_end_time = time.time()
    validation_duration = validation_end_time - validation_start_time
    
    # Concatenate all per-sample losses into a single array
    all_losses_for_model_np = np.concatenate(all_losses_for_model)
    
    # Calculate validation metrics
    avg_val_loss = np.mean(all_losses_for_model_np)
    min_val_loss = np.min(all_losses_for_model_np)
    max_val_loss = np.max(all_losses_for_model_np)
    
    # Save validation metrics
    validation_metrics = {
        "validation_duration_seconds": validation_duration,
        "total_val_samples": total_val_samples,
        "avg_val_loss": float(avg_val_loss),
        "min_val_loss": float(min_val_loss),
        "max_val_loss": float(max_val_loss),
        "target_val_idx": config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION,
        "target_val_loss": float(all_losses_for_model_np[config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION]) if config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < len(all_losses_for_model_np) else None
    }
    save_lds_training_metrics(model_id, validation_metrics, "validation_complete")
    
    # Save for analysis and debugging
    loss_file_path = losses_dir / f'loss_lds_{model_id}.pkl'
    with open(loss_file_path, 'wb') as f:
        pickle.dump(all_losses_for_model_np, f)
    logger.info(f"Saved per-sample validation losses for LDS Model {model_id} to {loss_file_path}")
    logger.info(f"LDS Model {model_id} validation: avg_loss={avg_val_loss:.4f}, target_loss={validation_metrics.get('target_val_loss', 'N/A')}")
    
    return all_losses_for_model_np


def compute_and_plot_lds_correlation(lds_validation_margins: np.ndarray, 
                                     list_of_subset_indices: List[np.ndarray],
                                     model_ids_used: List[int],
                                     precomputed_magic_scores_path: Optional[Union[str, Path]] = None,
                                     target_val_idx: Optional[int] = None,
                                     plot_suffix: str = "") -> bool:
    """
    Core LDS correlation computation and plotting logic.
    
    Args:
        lds_validation_margins: Matrix of validation losses [num_models, num_val_samples]
        list_of_subset_indices: List of training subset indices for each model
        model_ids_used: List of model IDs corresponding to the validation margins
        precomputed_magic_scores_path: Path to MAGIC influence scores file
        target_val_idx: Target validation image index
        plot_suffix: Suffix to add to plot filename (e.g., "_existing")
        
    Returns:
        bool: True if correlation plot was successfully generated, False otherwise
    """
    if target_val_idx is None:
        target_val_idx = config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
    
    # Validate target index is within bounds
    num_val_samples = lds_validation_margins.shape[1]
    if target_val_idx >= num_val_samples or target_val_idx < 0:
        logger.error(f"Target validation image index ({target_val_idx}) is out of bounds for validation dataset (size: {num_val_samples})")
        return False
    
    # Ensure magic_scores_file_to_load is a Path object for .exists()
    effective_magic_scores_path: Path
    if precomputed_magic_scores_path is None:
        effective_magic_scores_path = config.get_magic_scores_file_for_lds_input()
    elif isinstance(precomputed_magic_scores_path, str):
        effective_magic_scores_path = Path(precomputed_magic_scores_path)
    else: # It's already a Path
        effective_magic_scores_path = precomputed_magic_scores_path
    
    if not effective_magic_scores_path.exists():
        logger.error(f"MAGIC scores file not found at {effective_magic_scores_path}")
        return False
    
    logger.info(f"Loading MAGIC scores from {effective_magic_scores_path} for correlation...")
    with open(effective_magic_scores_path, 'rb') as f:
        loaded_scores = pickle.load(f)
    
    # Handle different MAGIC score formats
    if loaded_scores.ndim == 2:  # Per-step scores [num_steps, num_train_samples]
        magic_influence_estimates = loaded_scores.sum(axis=0)
        logger.info(f"Summed per-step MAGIC scores (shape {loaded_scores.shape}) to flat scores (shape {magic_influence_estimates.shape}).")
    elif loaded_scores.ndim == 1:  # Already flat scores [num_train_samples]
        magic_influence_estimates = loaded_scores
        logger.info(f"Loaded flat MAGIC scores (shape {magic_influence_estimates.shape}).")
    else:
        logger.error(f"Loaded MAGIC scores have unexpected shape: {loaded_scores.shape}")
        return False
    
    # Create binary mask arrays for the models we have data for
    logger.info("Creating training subset masks for correlation analysis...")
    lds_train_masks_list = []
    for model_id in model_ids_used:
        if model_id < len(list_of_subset_indices):
            current_subset_indices = list_of_subset_indices[model_id]
        else:
            # Handle case where model_id >= len(list_of_subset_indices) (cycling)
            current_subset_indices = list_of_subset_indices[model_id % len(list_of_subset_indices)]
        
        mask = np.zeros(config.NUM_TRAIN_SAMPLES, dtype=bool)
        mask[current_subset_indices] = True
        lds_train_masks_list.append(mask)
    
    lds_training_data_masks = np.stack(lds_train_masks_list)
    
    # Compute predicted performance using MAGIC influence scores
    logger.info("Computing predicted performance using MAGIC influence scores...")
    predicted_loss_impact_on_target = lds_training_data_masks @ magic_influence_estimates.T
    logger.info(f"Computed MAGIC-based predictions for {len(predicted_loss_impact_on_target)} LDS models")
    
    # Extract actual validation losses for the target image
    actual_margins_for_target_val_image = lds_validation_margins[:, target_val_idx]
    logger.info(f"Extracted actual losses on target validation image {target_val_idx}")
    
    # Compute Spearman rank correlation
    logger.info("=== Computing Linear Datamodeling Score (LDS) ===")
    
    correlation, p_value = spearmanr(predicted_loss_impact_on_target, actual_margins_for_target_val_image)
    
    logger.info(f"LDS Correlation Results (using {len(model_ids_used)} models):")
    logger.info(f"  Spearman R: {correlation:.3f}")
    logger.info(f"  P-value: {p_value:.3g}")
    logger.info(f"  Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} correlation")
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.regplot(x=predicted_loss_impact_on_target, y=actual_margins_for_target_val_image)
    title_suffix = f" ({plot_suffix.replace('_', ' ').title()})" if plot_suffix else ""
    plt.title(f'LDS Validation: Correlation with MAGIC Scores{title_suffix}\nTarget Val Img Idx: {target_val_idx}\nSpearman R: {correlation:.3f} (p={p_value:.3g})\nUsing {len(model_ids_used)} models')
    plt.xlabel("Predicted Loss Impact (Sum of MAGIC scores in subset)")
    plt.ylabel(f"Actual Loss on Val Img {target_val_idx}")
    plt.grid(True)
    
    plot_save_path = config.get_lds_plots_dir() / f"lds_correlation_val_{target_val_idx}{plot_suffix}.png"
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_save_path)
    logger.info(f"LDS Correlation plot saved to {plot_save_path}")
    plt.show()
    
    # Interpretation guide
    logger.info("=== LDS Results Interpretation ===")
    if correlation > 0.7:
        logger.info("✅ EXCELLENT: MAGIC scores strongly predict model behavior changes")
    elif correlation > 0.4:
        logger.info("✅ GOOD: MAGIC scores moderately predict model behavior changes")
    elif correlation > 0.2:
        logger.info("⚠️  WEAK: MAGIC scores weakly predict model behavior changes")
    else:
        logger.info("❌ POOR: MAGIC scores do not reliably predict model behavior changes")
    
    logger.info("Higher LDS correlation indicates better predictive data attribution quality.")
    
    return True


def load_existing_lds_results_and_plot_correlation(precomputed_magic_scores_path: Optional[Union[str, Path]] = None,
                                                   target_val_idx: Optional[int] = None) -> bool:
    """
    Load existing LDS validation results and plot correlation without retraining models.
    
    ALGORITHMIC PURPOSE:
    Allows reusing previously computed LDS validation results to generate correlation plots
    without having to retrain all the LDS models. This is useful for:
    - Regenerating plots with different formatting
    - Using different MAGIC scores for correlation
    - Quick validation after MAGIC score recomputation
    
    Args:
        precomputed_magic_scores_path: Path to MAGIC influence scores file
        target_val_idx: Target validation image index (default: config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)
        
    Returns:
        bool: True if correlation plot was successfully generated, False otherwise
    """
    logger.info("=== Loading Existing LDS Results for Correlation Plot ===")
    
    if target_val_idx is None:
        target_val_idx = config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
    
    # Check for pre-generated subset indices
    if not config.get_lds_indices_file().exists():
        logger.error(f"LDS subset indices file not found: {config.get_lds_indices_file()}")
        logger.info("Please run with --lds to generate subset indices first.")
        return False
    
    # Load subset indices
    logger.info(f"Loading subset indices from {config.get_lds_indices_file()}")
    with open(config.get_lds_indices_file(), 'rb') as f:
        all_subsets = pickle.load(f)
    
    # Check for existing validation loss files
    existing_loss_files = []
    missing_models = []
    
    for i in range(config.LDS_NUM_MODELS_TO_TRAIN):
        loss_file_path = config.get_lds_model_val_loss_path(i)
        if loss_file_path.exists():
            existing_loss_files.append((i, loss_file_path))
        else:
            missing_models.append(i)
    
    if not existing_loss_files:
        logger.error("No existing LDS validation loss files found.")
        logger.error(f"Expected files like: {config.get_lds_model_val_loss_path(0)}")
        logger.error("Please run full LDS validation first.")
        return False
    
    if missing_models:
        logger.warning(f"Missing validation loss files for {len(missing_models)} models: {missing_models[:10]}{'...' if len(missing_models) > 10 else ''}")
        logger.warning(f"Using available results from {len(existing_loss_files)} models.")
    
    # Load validation losses from existing files
    logger.info(f"Loading validation losses from {len(existing_loss_files)} existing LDS models...")
    all_validation_losses_list = []
    model_ids_used = []
    
    for model_id, loss_file_path in existing_loss_files:
        try:
            with open(loss_file_path, 'rb') as f:
                per_sample_val_losses = pickle.load(f)
            all_validation_losses_list.append(per_sample_val_losses)
            model_ids_used.append(model_id)
        except Exception as e:
            logger.warning(f"Failed to load validation losses for model {model_id} from {loss_file_path}: {e}")
            continue
    
    if not all_validation_losses_list:
        logger.error("Failed to load any validation loss data from existing files.")
        return False
    
    # Stack losses: rows are models, columns are validation samples
    lds_validation_margins = np.stack(all_validation_losses_list)
    logger.info(f"Loaded validation losses from {lds_validation_margins.shape[0]} LDS models on {lds_validation_margins.shape[1]} validation samples")
    
    # Use the common correlation computation function
    success = compute_and_plot_lds_correlation(
        lds_validation_margins=lds_validation_margins,
        list_of_subset_indices=all_subsets,
        model_ids_used=model_ids_used,
        precomputed_magic_scores_path=precomputed_magic_scores_path,
        target_val_idx=target_val_idx,
        plot_suffix="_existing"
    )
    
    if success:
        logger.info(f"=== LDS Correlation Plot Generated Successfully from {len(model_ids_used)} Existing Models ===")
    
    return success


def run_lds_validation(precomputed_magic_scores_path: Optional[Union[str, Path]] = None,
                      use_existing_results: bool = False,
                      force_replot_correlation: bool = False,
                      force_regenerate_indices: bool = False) -> None:
    """
    Main function to run Linear Datamodeling Score (LDS) validation of MAGIC influence scores.
    
    ALGORITHMIC PURPOSE:
    Implements the complete LDS evaluation methodology to validate predictive data attribution.
    This is the gold standard for testing whether influence scores can actually predict
    how changes in training data affect model behavior.
    
    THEORETICAL FRAMEWORK:
    1. Generate n random data subsets (weight vectors w^(1), ..., w^(n))
    2. For each subset i:
       a) Train model on subset → get true performance f(w^(i))
       b) Use MAGIC scores to predict performance f̃(w^(i))
    3. Compute LDS = Spearman_correlation({f̃(w^(i))}, {f(w^(i))})
    
    HIGH LDS → MAGIC accurately predicts data-model relationships
    LOW LDS → MAGIC predictions are unreliable
    
    IMPLEMENTATION OVERVIEW:
    - Phase 1: Generate training subsets and shared infrastructure
    - Phase 2: Train multiple models on different subsets
    - Phase 3: Evaluate all models on validation set
    - Phase 4: Correlate MAGIC predictions with actual performance
    
    Args:
        precomputed_magic_scores_path: Path to MAGIC influence scores file
                                     If None, uses default path from config
        use_existing_results: Whether to use existing LDS results for correlation plot.
                              This is also implicitly True if force_replot_correlation is True.
        force_replot_correlation: Whether to force re-computation of correlation and re-plotting
                                  using existing LDS model losses and MAGIC scores.
        force_regenerate_indices: Whether to force regeneration of subset indices file.
    """
    
    # If force_replot_correlation is True, use_existing_results will also be True (due to main_runner.py logic).
    # The primary effect of force_replot_correlation is to ensure the use_existing_results path is taken.
    # No specific additional logic is needed here for force_replot_correlation beyond what use_existing_results already does,
    # as load_existing_lds_results_and_plot_correlation will always attempt to load and plot.

    if use_existing_results: # This block is entered if use_existing_results or force_replot_correlation is true
        if force_replot_correlation:
            logger.info("LDS: --force_replot_lds_correlation is set. "
                        "Re-calculating correlation and re-generating plot from existing LDS model losses and MAGIC scores.")
        else:
            logger.info("Using existing LDS results (skipping model training)")
        
        success = load_existing_lds_results_and_plot_correlation(precomputed_magic_scores_path)
        if success:
            logger.info("Successfully generated correlation plot from existing LDS results")
        else:
            logger.error("Failed to generate correlation plot from existing results")
        return # Exit early as we are either using existing or force-replotting from existing.
    
    # --- Continue with full LDS validation if not using existing results and not force-replotting ---
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for LDS Validation")

    # === PHASE 1: SETUP AND SUBSET GENERATION ===
    logger.info("=== PHASE 1: Generating Training Subsets ===")
    all_subsets = generate_and_save_subset_indices(force_regenerate=force_regenerate_indices)
    logger.info(f"Generated {len(all_subsets)} subsets for LDS validation")
    logger.info(f"Each subset contains {len(all_subsets[0])} samples ({config.LDS_SUBSET_FRACTION:.1%} of {config.NUM_TRAIN_SAMPLES} total)")
    logger.info(f"Training {config.LDS_NUM_MODELS_TO_TRAIN} LDS models using these subsets")

    logger.info("Creating shared dataloader with EXACT same settings as MAGIC...")
    shared_train_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id=config.SHARED_DATALOADER_INSTANCE_ID,
        batch_size=config.MODEL_TRAIN_BATCH_SIZE,
        split='train',
        shuffle=True,
        augment=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        root_path=config.CIFAR_ROOT
    )
    
    logger.info("Creating shared validation dataloader...")
    shared_val_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id="lds_shared_eval_loader",
        batch_size=config.MODEL_TRAIN_BATCH_SIZE,
        root_path=config.CIFAR_ROOT,
        split='val',
        shuffle=False,
        augment=False,
        num_workers=config.DATALOADER_NUM_WORKERS
    )

    # === PHASE 2: TRAIN MODELS ON SUBSETS ===
    logger.info("=== PHASE 2: Training LDS Models on Subsets ===")
    all_validation_losses_stacked = []
    for i in tqdm(range(config.LDS_NUM_MODELS_TO_TRAIN), desc="Training LDS Models"):
        model_id = i
        current_subset_indices = all_subsets[i % len(all_subsets)]
        logger.info(f"--- Training LDS Model {model_id} on subset of size {len(current_subset_indices)} ---")
        trained_model = train_model_on_subset(
            model_id=model_id,
            train_subset_indices=current_subset_indices,
            device=device,
            shared_train_loader=shared_train_loader,
            epochs=config.MODEL_TRAIN_EPOCHS,
            lr=config.MODEL_TRAIN_LR,
            momentum=config.MODEL_TRAIN_MOMENTUM,
            weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
        )
        per_sample_val_losses = evaluate_and_save_losses(
            trained_model, model_id, device, 
            val_loader=shared_val_loader,
            losses_dir=config.get_lds_losses_dir()
        )
        all_validation_losses_stacked.append(per_sample_val_losses)
    
    if not all_validation_losses_stacked:
        logger.warning("No LDS models were trained or evaluated. Skipping correlation.")
        return

    # === PHASE 3: ORGANIZE EVALUATION RESULTS ===
    logger.info("=== PHASE 3: Organizing Evaluation Results ===")
    lds_validation_margins = np.stack(all_validation_losses_stacked)
    logger.info(f"Collected validation losses from {lds_validation_margins.shape[0]} LDS models on {lds_validation_margins.shape[1]} validation samples")

    num_val_samples = lds_validation_margins.shape[1]
    if config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION >= num_val_samples or config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < 0:
        raise ValueError(f"LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION ({config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}) is out of bounds for validation dataset (size: {num_val_samples})")

    # === PHASE 4: CORRELATION WITH MAGIC INFLUENCE SCORES ===
    logger.info("=== PHASE 4: Correlating with MAGIC Influence Scores ===")
    
    effective_magic_scores_path_for_run: Path
    if precomputed_magic_scores_path is None:
        effective_magic_scores_path_for_run = config.get_magic_scores_file_for_lds_input()
    elif isinstance(precomputed_magic_scores_path, str):
        effective_magic_scores_path_for_run = Path(precomputed_magic_scores_path)
    else: # It's already a Path
        effective_magic_scores_path_for_run = precomputed_magic_scores_path

    if not effective_magic_scores_path_for_run.exists():
        logger.warning(f"MAGIC scores file not found at {effective_magic_scores_path_for_run}. Skipping correlation analysis.")
        return

    logger.info(f"Loading MAGIC scores from {effective_magic_scores_path_for_run} for correlation...")
    with open(effective_magic_scores_path_for_run, 'rb') as f:
        loaded_scores = pickle.load(f)
    
    if loaded_scores.ndim == 2:
        magic_influence_estimates = loaded_scores.sum(axis=0)
        logger.info(f"Summed per-step MAGIC scores (shape {loaded_scores.shape}) to flat scores (shape {magic_influence_estimates.shape}).")
    elif loaded_scores.ndim == 1:
        magic_influence_estimates = loaded_scores
        logger.info(f"Loaded flat MAGIC scores (shape {magic_influence_estimates.shape}).")
    else:
        raise ValueError(f"Loaded MAGIC scores from {effective_magic_scores_path_for_run} have unexpected shape: {loaded_scores.shape}")

    model_ids_used = list(range(config.LDS_NUM_MODELS_TO_TRAIN))
    success = compute_and_plot_lds_correlation(
        lds_validation_margins=lds_validation_margins,
        list_of_subset_indices=all_subsets,
        model_ids_used=model_ids_used,
        precomputed_magic_scores_path=effective_magic_scores_path_for_run,
        target_val_idx=config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION,
        plot_suffix=""  # No suffix for main LDS run
    )
    
    if success:
        logger.info("LDS Validation finished successfully.")
    else:
        logger.error("LDS Validation failed during correlation computation.")