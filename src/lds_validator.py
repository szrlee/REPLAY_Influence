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
from typing import List, Optional, Union
from pathlib import Path
import logging

# Project-specific imports
from . import config
from .utils import (
    create_deterministic_dataloader, 
    create_deterministic_model, 
    create_deterministic_optimizer, 
    create_deterministic_scheduler, 
    log_scheduler_info,
    update_dataloader_epoch,
)
from .model_def import construct_rn9
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
    if file_path is None: file_path = config.LDS_INDICES_FILE

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


def train_model_on_subset(model_id: int, train_subset_indices: List[int], device: torch.device, 
                         shared_train_loader: torch.utils.data.DataLoader,
                         epochs: Optional[int] = None, lr: Optional[float] = None, 
                         momentum: Optional[float] = None, weight_decay: Optional[float] = None) -> torch.nn.Module:
    """
    Train a model on a specific subset of training data with identical settings to MAGIC.
    
    ALGORITHMIC PURPOSE:
    Implements the learning algorithm A(w) that maps a data weight vector w to trained model parameters θ.
    This is the core of computing the true model output function f(w) = φ(A(w)) for LDS validation.
    
    THEORETICAL CONTEXT:
    In the LDS framework:
    - Input: Binary weight vector w (represented by train_subset_indices)
    - Process: A(w) → θ (train model using weighted samples)
    - Output: Trained model parameters θ that will be evaluated to get f(w)
    
    KEY IMPLEMENTATION DETAILS:
    - Uses IDENTICAL model initialization as MAGIC (via shared instance_id)
    - Uses IDENTICAL training hyperparameters as MAGIC
    - Uses IDENTICAL data ordering as MAGIC (via shared_train_loader)
    - Implements subset training via weighted loss (efficient alternative to separate datasets)
    
    Args:
        model_id: Unique identifier for this LDS model
        train_subset_indices: Indices of training samples to include (defines weight vector w)
        device: Computing device (CPU/GPU)
        shared_train_loader: Same dataloader used in MAGIC analysis
        epochs, lr, momentum, weight_decay: Training hyperparameters (defaults from config)
        
    Returns:
        Trained PyTorch model ready for evaluation
    """
    
    # Use shared hyperparameters if not specified - CRITICAL for consistency with MAGIC
    if epochs is None: epochs = config.MODEL_TRAIN_EPOCHS
    if lr is None: lr = config.MODEL_TRAIN_LR
    if momentum is None: momentum = config.MODEL_TRAIN_MOMENTUM  
    if weight_decay is None: weight_decay = config.MODEL_TRAIN_WEIGHT_DECAY
    
    checkpoints_dir = config.LDS_CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Using pre-created shared dataloader to ensure consistency with MAGIC")

    # CRITICAL: Create model with IDENTICAL initialization as MAGIC
    # This ensures that differences in final performance are due to training data, not initialization
    model = create_deterministic_model(
        master_seed=config.SEED,
        creator_func=construct_rn9,
        instance_id=config.SHARED_MODEL_INSTANCE_ID,  # CRITICAL: Same instance_id as MAGIC (from config)
        num_classes=config.NUM_CLASSES
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # CORE LDS MECHANISM: Convert subset indices to binary weight vector
    # This implements the data weight vector w where w_i = 1 for included samples, w_i = 0 for excluded
    data_weights_for_subset = torch.zeros(config.NUM_TRAIN_SAMPLES, device=device)
    data_weights_for_subset[train_subset_indices] = 1.0
    num_active_samples_in_subset = data_weights_for_subset.sum().item()
    
    logger.debug(f"LDS Model {model_id}: Using {num_active_samples_in_subset}/{config.NUM_TRAIN_SAMPLES} samples from subset")

    # Handle edge case: empty subset
    if num_active_samples_in_subset == 0:
        logger.warning(f"Model ID {model_id} has no active samples in its subset. Skipping training.")
        # Save an initial state to avoid errors later if a checkpoint is expected
        torch.save(model.state_dict(), checkpoints_dir / f'sd_lds_{model_id}_final.pt')
        return model # Return untrained model

    # CRITICAL: Create optimizer with IDENTICAL settings as MAGIC
    optimizer = create_deterministic_optimizer(
        master_seed=config.SEED,
        optimizer_class=SGD,
        model_params=model.parameters(),
        instance_id=config.SHARED_OPTIMIZER_INSTANCE_ID,  # Same instance_id as MAGIC (from config)
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # CRITICAL: Create scheduler with IDENTICAL settings as MAGIC
    scheduler = create_deterministic_scheduler(
        master_seed=config.SEED,
        optimizer=optimizer,
        schedule_type=config.LR_SCHEDULE_TYPE,
        total_steps=epochs * len(shared_train_loader),
        instance_id=config.SHARED_SCHEDULER_INSTANCE_ID,  # Same instance_id as MAGIC (from config)
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
    log_scheduler_info(scheduler, config.LR_SCHEDULE_TYPE, logger, f"LDS Model {model_id}")
    
    # Use reduction='none' to get per-sample losses, then apply weighted averaging
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    model.train()
    total_batches_processed = 0
    total_samples_used = 0
    
    logger.debug(f"Starting training for LDS model {model_id} with subset of {num_active_samples_in_subset} samples")
    
    # TRAINING LOOP: Implement A(w) - the learning algorithm
    for epoch in range(epochs):
        # CRITICAL: Update dataloader epoch for deterministic shuffling
        # This ensures IDENTICAL data ordering as MAGIC analysis
        update_dataloader_epoch(shared_train_loader, epoch)
        
        epoch_samples_used = 0
        for batch_idx, (images, labels, original_indices) in enumerate(tqdm(shared_train_loader, desc=f"LDS Model {model_id}, Epoch {epoch+1}")):
            images, labels, original_indices = images.to(device), labels.to(device), original_indices.to(device)

            # CORE SUBSET MECHANISM: Apply data weights to implement subset training
            # This is the key innovation - instead of creating separate datasets, we use weighted loss
            active_weights_in_batch = data_weights_for_subset[original_indices]
            sum_active_weights_in_batch = active_weights_in_batch.sum()

            # Skip batch if no samples from current subset are present
            if sum_active_weights_in_batch == 0: 
                continue

            # STANDARD TRAINING STEP with weighted loss
            # The global RNG state evolves naturally, ensuring stochastic operations (like dropout)
            # behave consistently with MAGIC's training process
            optimizer.zero_grad()
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            
            # WEIGHTED LOSS: Only samples from the current model's subset contribute
            # This implements training on the subset defined by weight vector w
            # Equivalent to MAGIC's mean loss but only over the subset samples
            weighted_loss = (per_sample_loss * active_weights_in_batch).sum() / (sum_active_weights_in_batch + config.EPSILON_FOR_WEIGHTED_LOSS)
            
            weighted_loss.backward()
            optimizer.step()
            
            # CRITICAL FIX: Apply scheduler AFTER optimizer.step() to match MAGIC's training order
            if scheduler:
                scheduler.step()
            
            total_batches_processed += 1
            batch_samples_used = sum_active_weights_in_batch.item()
            epoch_samples_used += batch_samples_used
            
        total_samples_used += epoch_samples_used
        logger.debug(f"LDS Model {model_id} Epoch {epoch+1}: Used {epoch_samples_used} samples")
    
    logger.info(f"LDS Model {model_id} training completed: {total_batches_processed} batches, {total_samples_used} total sample updates")

    final_checkpoint_path = checkpoints_dir / f'sd_lds_{model_id}_final.pt'
    torch.save(model.state_dict(), final_checkpoint_path)
    logger.info(f"LDS Model {model_id} trained. Final checkpoint: {final_checkpoint_path}")
    return model


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
    if losses_dir is None: losses_dir = config.LDS_LOSSES_DIR
    # batch_size parameter is removed as val_loader is now passed in.
    losses_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_losses_for_model = []
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    # Compute per-sample validation losses - this implements φ(θ)
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc=f"Evaluating LDS Model {model_id}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            all_losses_for_model.append(per_sample_loss.cpu().numpy())
    
    # Concatenate all per-sample losses into a single array
    all_losses_for_model_np = np.concatenate(all_losses_for_model)
    # Save for analysis and debugging
    loss_file_path = losses_dir / f'loss_lds_{model_id}.pkl'
    with open(loss_file_path, 'wb') as f:
        pickle.dump(all_losses_for_model_np, f)
    logger.info(f"Saved per-sample validation losses for LDS Model {model_id} to {loss_file_path}")
    return all_losses_for_model_np


def run_lds_validation(precomputed_magic_scores_path: Optional[Union[str, Path]] = None) -> None:
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
    """
    
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for LDS Validation")

    # === PHASE 1: SETUP AND SUBSET GENERATION ===
    logger.info("=== PHASE 1: Generating Training Subsets ===")
    
    # Generate random subsets for LDS validation
    # Each subset represents a different "dataset" defined by its weight vector w^(i)
    list_of_subset_indices = generate_and_save_subset_indices()
    
    # Log subset information for transparency
    logger.info(f"Generated {len(list_of_subset_indices)} subsets for LDS validation")
    logger.info(f"Each subset contains {len(list_of_subset_indices[0])} samples ({config.LDS_SUBSET_FRACTION:.1%} of {config.NUM_TRAIN_SAMPLES} total)")
    logger.info(f"Training {config.LDS_NUM_MODELS_TO_TRAIN} LDS models using these subsets")

    # CRITICAL: Create shared dataloader with EXACT same settings as MAGIC
    # This ensures ALL LDS models see the EXACT same data sequence as MAGIC analysis
    # Maintaining data ordering consistency is essential for valid comparison
    logger.info("Creating shared dataloader with EXACT same settings as MAGIC...")
    
    shared_train_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id=config.SHARED_DATALOADER_INSTANCE_ID,  # CRITICAL: Same instance_id as MAGIC (from config)
        batch_size=config.MODEL_TRAIN_BATCH_SIZE,
        split='train',
        shuffle=True,  # Same as MAGIC
        augment=False,  # Same as MAGIC
        num_workers=config.DATALOADER_NUM_WORKERS,  # Same as MAGIC
        root_path=config.CIFAR_ROOT
    )
    
    # Create a single, shared validation dataloader for consistent evaluation
    logger.info("Creating shared validation dataloader...")
    shared_val_loader = create_deterministic_dataloader(
        master_seed=config.SEED,
        creator_func=get_cifar10_dataloader,
        instance_id="lds_shared_eval_loader", # Consistent instance_id for shared val loader
        batch_size=config.MODEL_TRAIN_BATCH_SIZE, # Or a specific eval_batch_size from config
        root_path=config.CIFAR_ROOT,
        split='val',
        shuffle=False,
        augment=False,
        num_workers=config.DATALOADER_NUM_WORKERS
    )

    # === PHASE 2: TRAIN MODELS ON SUBSETS ===
    logger.info("=== PHASE 2: Training LDS Models on Subsets ===")
    
    # This implements the core LDS procedure: train multiple models on different subsets
    # Each model represents f(w^(i)) for a different weight vector w^(i)
    all_validation_losses_stacked = [] # To store losses from all LDS models
    for i in tqdm(range(config.LDS_NUM_MODELS_TO_TRAIN), desc="Training LDS Models"):
        model_id = i
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)] # Cycle if fewer unique subsets
        
        logger.info(f"--- Training LDS Model {model_id} on subset of size {len(current_subset_indices)} ---")
        
        # Train model on current subset - implements A(w^(i)) → θ^(i)
        trained_model = train_model_on_subset(
            model_id=model_id,
            train_subset_indices=current_subset_indices,
            device=device,
            shared_train_loader=shared_train_loader,  # CRITICAL: Same dataloader for all models
            epochs=config.MODEL_TRAIN_EPOCHS,
            lr=config.MODEL_TRAIN_LR,
            momentum=config.MODEL_TRAIN_MOMENTUM,
            weight_decay=config.MODEL_TRAIN_WEIGHT_DECAY
        )
        
        # Evaluate model on validation set - implements φ(θ^(i)) → f(w^(i))
        per_sample_val_losses = evaluate_and_save_losses(
            trained_model, model_id, device, 
            val_loader=shared_val_loader,  # Pass the shared val_loader
            losses_dir=config.LDS_LOSSES_DIR
        )
        all_validation_losses_stacked.append(per_sample_val_losses)
    
    if not all_validation_losses_stacked:
        logger.warning("No LDS models were trained or evaluated. Skipping correlation.")
        return

    # === PHASE 3: ORGANIZE EVALUATION RESULTS ===
    logger.info("=== PHASE 3: Organizing Evaluation Results ===")
    
    # Stack losses: rows are models, columns are validation samples
    # Shape: (LDS_NUM_MODELS, NUM_VALIDATION_SAMPLES)
    lds_validation_margins = np.stack(all_validation_losses_stacked)
    logger.info(f"Collected validation losses from {lds_validation_margins.shape[0]} LDS models on {lds_validation_margins.shape[1]} validation samples")

    # Validate target index is within bounds of validation data
    num_val_samples = lds_validation_margins.shape[1]
    if config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION >= num_val_samples or config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < 0:
        raise ValueError(f"LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION ({config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}) is out of bounds for validation dataset (size: {num_val_samples})")

    # === PHASE 4: CORRELATION WITH MAGIC INFLUENCE SCORES ===
    logger.info("=== PHASE 4: Correlating with MAGIC Influence Scores ===")
    
    # Load MAGIC influence scores for correlation analysis
    magic_scores_file_to_load = precomputed_magic_scores_path
    if magic_scores_file_to_load is None:
        # Use the default MAGIC scores file path from config
        magic_scores_file_to_load = config.MAGIC_SCORES_FILE_FOR_LDS_INPUT

    if not magic_scores_file_to_load.exists():
        logger.warning(f"MAGIC scores file not found at {magic_scores_file_to_load}. Skipping correlation analysis.")
        return

    logger.info(f"Loading MAGIC scores from {magic_scores_file_to_load} for correlation...")
    with open(magic_scores_file_to_load, 'rb') as f:
        loaded_scores = pickle.load(f)
    
    # Handle different MAGIC score formats
    if loaded_scores.ndim == 2: # Per-step scores [num_steps, num_train_samples]
        magic_influence_estimates = loaded_scores.sum(axis=0)
        logger.info(f"Summed per-step MAGIC scores (shape {loaded_scores.shape}) to flat scores (shape {magic_influence_estimates.shape}).")
    elif loaded_scores.ndim == 1: # Already flat scores [num_train_samples]
        magic_influence_estimates = loaded_scores
        logger.info(f"Loaded flat MAGIC scores (shape {magic_influence_estimates.shape}).")
    else:
        raise ValueError(f"Loaded MAGIC scores from {magic_scores_file_to_load} have unexpected shape: {loaded_scores.shape}")

    # Create binary mask arrays indicating which training samples were used by each LDS model
    # This represents the weight vectors w^(1), w^(2), ..., w^(n) used in LDS
    logger.info("Creating training subset masks for correlation analysis...")
    lds_train_masks_list = []
    for i in range(config.LDS_NUM_MODELS_TO_TRAIN):
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)]
        mask = np.zeros(config.NUM_TRAIN_SAMPLES, dtype=bool)
        mask[current_subset_indices] = True
        lds_train_masks_list.append(mask)
    # Shape: (LDS_NUM_MODELS, NUM_TRAINING_SAMPLES)
    lds_training_data_masks = np.stack(lds_train_masks_list)

    # CORE LDS COMPUTATION: Predict performance using MAGIC influence scores
    # For each LDS model, sum the MAGIC influence scores of samples in its training subset
    # This implements f̃(w^(i)) = Σ_{j ∈ subset_i} influence_score_j
    logger.info("Computing predicted performance using MAGIC influence scores...")
    predicted_loss_impact_on_target = lds_training_data_masks @ magic_influence_estimates.T
    # Shape: (LDS_NUM_MODELS_TO_TRAIN,)
    logger.info(f"Computed MAGIC-based predictions for {len(predicted_loss_impact_on_target)} LDS models")

    # Extract actual validation losses for the specific target image used in MAGIC analysis
    # This gives us the true model outputs f(w^(i)) for correlation
    actual_margins_for_target_val_image = lds_validation_margins[:, config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION]
    # Shape: (LDS_NUM_MODELS_TO_TRAIN,)
    logger.info(f"Extracted actual losses on target validation image {config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}")
    
    # === FINAL LDS COMPUTATION ===
    logger.info("=== Computing Linear Datamodeling Score (LDS) ===")
    
    # Compute Spearman rank correlation between predicted and actual performance
    # This is the core LDS metric: how well do MAGIC scores predict model behavior changes?
    correlation, p_value = spearmanr(predicted_loss_impact_on_target, actual_margins_for_target_val_image)
    
    logger.info(f"LDS Correlation Results:")
    logger.info(f"  Spearman R: {correlation:.3f}")
    logger.info(f"  P-value: {p_value:.3g}")
    logger.info(f"  Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} correlation")
    
    # Create visualization of LDS results
    plt.figure(figsize=(8, 6))
    sns.regplot(x=predicted_loss_impact_on_target, y=actual_margins_for_target_val_image)
    plt.title(f'LDS Validation: Correlation with MAGIC Scores\nTarget Val Img Idx: {config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}\nSpearman R: {correlation:.3f} (p={p_value:.3g})')
    plt.xlabel("Predicted Loss Impact (Sum of MAGIC scores in subset)")
    plt.ylabel(f"Actual Loss on Val Img {config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}")
    plt.grid(True)
    plot_save_path = config.LDS_PLOTS_DIR / f"lds_correlation_val_{config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}.png"
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_save_path)
    logger.info(f"LDS Correlation plot saved to {plot_save_path}")
    plt.show()

    logger.info(f"LDS Validation finished. Correlation: {correlation:.3f}")
    
    # === INTERPRETATION GUIDE ===
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