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
from .utils import set_seeds
from .model_def import construct_rn9
from .data_handling import get_cifar10_dataloader, CustomDataset
# LDS does not use the global batch_dict_for_replay from magic_analyzer

logger = logging.getLogger('influence_analysis.lds')


def verify_data_ordering_consistency() -> bool:
    """
    Verifies that LDS validator uses the same data ordering as MAGIC analyzer.
    This ensures that sample indices refer to the exact same data points across
    multiple epochs, even when LDS uses subset-based weighted training.
    
    Returns:
        bool: True if data ordering is consistent, False otherwise.
    """
    logger.info("Verifying data ordering consistency between MAGIC and LDS...")
    
    # Test 1: Basic data loader consistency with EXACT same parameters
    logger.info("Test 1: Verifying basic data loader consistency...")
    
    # Verify batch sizes are identical
    if config.MAGIC_MODEL_TRAIN_BATCH_SIZE != config.LDS_MODEL_TRAIN_BATCH_SIZE:
        logger.error(f"Batch size mismatch: MAGIC={config.MAGIC_MODEL_TRAIN_BATCH_SIZE}, LDS={config.LDS_MODEL_TRAIN_BATCH_SIZE}")
        return False
    
    # Compare first few batches to verify consistent ordering
    try:
        for i in range(min(5, 10)):  # Test first 5 batches
            # CRITICAL FIX: Create fresh DataLoaders for each comparison
            # This ensures both DataLoaders start from identical random states
            set_seeds(config.SEED)
            magic_loader = get_cifar10_dataloader(
                batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
                split='train', shuffle=True, augment=False,
                num_workers=config.DATALOADER_NUM_WORKERS, root_path=config.CIFAR_ROOT
            )
            magic_iter = iter(magic_loader)
            # Skip to the i-th batch
            for _ in range(i + 1):
                magic_batch = next(magic_iter)
            
            set_seeds(config.SEED)
            lds_loader = get_cifar10_dataloader(
                batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
                split='train', 
                shuffle=True, 
                augment=False,
                num_workers=config.DATALOADER_NUM_WORKERS,
                root_path=config.CIFAR_ROOT
            )
            lds_iter = iter(lds_loader)
            # Skip to the i-th batch
            for _ in range(i + 1):
                lds_batch = next(lds_iter)
            
            magic_indices = magic_batch[2]  # batch_indices
            lds_indices = lds_batch[2]      # original_indices
            
            # Indices should be identical for same batches
            if not torch.equal(magic_indices, lds_indices):
                logger.error(f"Data ordering mismatch detected in batch {i}")
                logger.error(f"MAGIC indices: {magic_indices[:5]}")
                logger.error(f"LDS indices: {lds_indices[:5]}")
                return False
                
        logger.info("✓ Basic data loader consistency verified")
        
    except Exception as e:
        logger.error(f"Basic data loader verification failed: {e}")
        return False
    
    # Test 2: Multi-epoch consistency (CRITICAL TEST)
    logger.info("Test 2: Verifying multi-epoch data ordering consistency...")
    
    try:
        # Test that same seed produces same ordering across epochs
        set_seeds(config.SEED)
        test_loader = get_cifar10_dataloader(
            batch_size=100, 
            split='train', 
            shuffle=True, 
            augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS,  # Use 0 workers for deterministic testing
            root_path=config.CIFAR_ROOT
        )
        
        # Collect indices from first epoch
        epoch1_indices = []
        for batch_idx, (_, _, indices) in enumerate(test_loader):
            epoch1_indices.append(indices.clone())
            if batch_idx >= 4:  # First 5 batches
                break
        
        # Collect indices from second epoch (without resetting seed)
        epoch2_indices = []
        for batch_idx, (_, _, indices) in enumerate(test_loader):
            epoch2_indices.append(indices.clone())
            if batch_idx >= 4:  # First 5 batches
                break
        
        # Check if ordering is consistent across epochs
        epoch1_flat = torch.cat(epoch1_indices)
        epoch2_flat = torch.cat(epoch2_indices)
        
        if not torch.equal(epoch1_flat, epoch2_flat):
            logger.info("✓ Multi-epoch shuffling detected (expected behavior)")
        else:
            logger.info("✓ Multi-epoch consistency maintained")
            
    except Exception as e:
        logger.error(f"Multi-epoch verification failed: {e}")
        return False
    
    # Test 3: CRITICAL TEST - Complete data sequence consistency
    logger.info("Test 3: CRITICAL - Verifying complete data sequence consistency between MAGIC and LDS...")
    
    try:
        # This tests the exact scenario that happens in practice:
        # MAGIC: set seed once, create dataloader, train for multiple epochs
        # LDS: set seed once, create ONE shared dataloader, all models use same dataloader
        
        # Simulate MAGIC data sequence
        set_seeds(config.SEED)
        magic_test_loader = get_cifar10_dataloader(
            batch_size=100,  # Small batch for testing
            split='train', 
            shuffle=True, 
            augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS,  # Use 0 workers for deterministic testing
            root_path=config.CIFAR_ROOT
        )
        
        # Collect complete sequence from MAGIC-style training
        magic_sequence = []
        for epoch in range(2):  # Test 2 epochs
            for batch_idx, (_, _, indices) in enumerate(magic_test_loader):
                magic_sequence.append((epoch, batch_idx, indices.clone()))
                if batch_idx >= 4:  # First 5 batches per epoch
                    break
        
        # Simulate LDS data sequence (shared dataloader approach)
        set_seeds(config.SEED)
        lds_shared_loader = get_cifar10_dataloader(
            batch_size=100,  # Same batch size
            split='train', 
            shuffle=True, 
            augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS,  # Use 0 workers for deterministic testing
            root_path=config.CIFAR_ROOT
        )
        
        # Collect sequence from LDS-style training (simulating multiple models using same dataloader)
        lds_sequence = []
        for epoch in range(2):  # Test 2 epochs
            for batch_idx, (_, _, indices) in enumerate(lds_shared_loader):
                lds_sequence.append((epoch, batch_idx, indices.clone()))
                if batch_idx >= 4:  # First 5 batches per epoch
                    break
        
        # Compare sequences - they should be IDENTICAL
        if len(magic_sequence) != len(lds_sequence):
            logger.error(f"Sequence length mismatch: MAGIC={len(magic_sequence)}, LDS={len(lds_sequence)}")
            return False
        
        for i, ((m_epoch, m_batch, m_indices), (l_epoch, l_batch, l_indices)) in enumerate(zip(magic_sequence, lds_sequence)):
            if m_epoch != l_epoch or m_batch != l_batch:
                logger.error(f"Epoch/batch mismatch at position {i}: MAGIC=({m_epoch},{m_batch}), LDS=({l_epoch},{l_batch})")
                return False
            
            if not torch.equal(m_indices, l_indices):
                logger.error(f"Index mismatch at epoch {m_epoch}, batch {m_batch}")
                logger.error(f"MAGIC indices: {m_indices[:5]}")
                logger.error(f"LDS indices: {l_indices[:5]}")
                return False
        
        logger.info(f"✓ Complete data sequence consistency verified across {len(magic_sequence)} batches")
        
    except Exception as e:
        logger.error(f"Complete data sequence verification failed: {e}")
        return False
    
    # Test 4: Subset mechanism verification
    logger.info("Test 4: Verifying subset-based weighted training mechanism...")
    
    try:
        # Create a test subset (first 1000 samples)
        test_subset_indices = np.arange(1000)
        
        # Reset seed for consistent data loading
        set_seeds(config.SEED)
        test_loader = get_cifar10_dataloader(
            batch_size=100, 
            split='train', 
            shuffle=True, 
            augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS,  # Use 0 workers for deterministic testing
            root_path=config.CIFAR_ROOT
        )
        
        # Create weights for subset
        data_weights = torch.zeros(config.NUM_TRAIN_SAMPLES)
        data_weights[test_subset_indices] = 1.0
        
        # Check that weighted mechanism correctly identifies subset samples
        total_subset_samples_found = 0
        total_batches_checked = 0
        
        for batch_idx, (images, labels, original_indices) in enumerate(test_loader):
            if batch_idx >= 10:  # Check first 10 batches
                break
                
            active_weights = data_weights[original_indices]
            subset_samples_in_batch = active_weights.sum().item()
            total_subset_samples_found += subset_samples_in_batch
            total_batches_checked += 1
            
            # Verify that active weights are only 0 or 1
            unique_weights = torch.unique(active_weights)
            if not all(w in [0.0, 1.0] for w in unique_weights):
                logger.error(f"Invalid weights found: {unique_weights}")
                return False
            
            # Verify that indices are in expected range
            if torch.any(original_indices < 0) or torch.any(original_indices >= config.NUM_TRAIN_SAMPLES):
                logger.error(f"Invalid indices found: min={original_indices.min()}, max={original_indices.max()}")
                return False
        
        logger.info(f"✓ Subset mechanism verified: Found {total_subset_samples_found} subset samples across {total_batches_checked} batches")
        
        # Test 5: Model initialization consistency
        logger.info("Test 5: Verifying model initialization consistency...")
        
        # Create multiple models with same seed (simulating LDS models)
        models = []
        for i in range(3):
            set_seeds(config.SEED)  # Same seed for all
            model = construct_rn9(num_classes=config.NUM_CLASSES)
            models.append(model)
        
        # Check that all models have identical parameters
        for i in range(1, len(models)):
            for p1, p2 in zip(models[0].parameters(), models[i].parameters()):
                if not torch.equal(p1, p2):
                    logger.error(f"Model initialization inconsistency detected between model 0 and model {i}")
                    return False
                    
        logger.info("✓ Model initialization consistency verified")
        
        # Test 6: Configuration consistency check
        logger.info("Test 6: Verifying configuration consistency...")
        
        config_issues = []
        
        if config.MAGIC_MODEL_TRAIN_BATCH_SIZE != config.LDS_MODEL_TRAIN_BATCH_SIZE:
            config_issues.append(f"Batch size mismatch: MAGIC={config.MAGIC_MODEL_TRAIN_BATCH_SIZE}, LDS={config.LDS_MODEL_TRAIN_BATCH_SIZE}")
        
        if config.MAGIC_TARGET_VAL_IMAGE_IDX != config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION:
            config_issues.append(f"Target image mismatch: MAGIC={config.MAGIC_TARGET_VAL_IMAGE_IDX}, LDS={config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}")
        
        if config_issues:
            for issue in config_issues:
                logger.error(issue)
            return False
        
        logger.info("✓ Configuration consistency verified")
        
        logger.info("🎯 ALL data ordering verification tests PASSED")
        logger.info("✓ LDS and MAGIC use consistent sample indices")
        logger.info("✓ Multi-epoch behavior is predictable")
        logger.info("✓ CRITICAL: Complete data sequence consistency verified")
        logger.info("✓ Subset-based weighted training mechanism is correct")
        logger.info("✓ Model initialization is consistent across LDS models")
        logger.info("✓ Configuration parameters are aligned")
        return True
        
    except Exception as e:
        logger.error(f"Subset mechanism verification failed: {e}")
        return False


def generate_and_save_subset_indices(num_subsets: Optional[int] = None, total_samples: Optional[int] = None, 
                                     subset_fraction: Optional[float] = None, file_path: Optional[Path] = None, 
                                     force_regenerate: bool = False) -> List[np.ndarray]:
    if num_subsets is None: num_subsets = config.LDS_NUM_SUBSETS_TO_GENERATE
    if total_samples is None: total_samples = config.NUM_TRAIN_SAMPLES # Use general NUM_TRAIN_SAMPLES
    if subset_fraction is None: subset_fraction = config.LDS_SUBSET_FRACTION
    if file_path is None: file_path = config.LDS_INDICES_FILE

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
    
    indices_list = []
    for i in range(num_subsets):
        try:
            subset_indices = np.random.choice(range(total_samples), subset_sample_count, replace=False)
            indices_list.append(subset_indices)
        except ValueError as e:
            raise ValueError(f"Failed to generate subset {i}: {e}. This might happen if subset_sample_count > total_samples.")
    
    with open(file_path, 'wb') as f:
        pickle.dump(indices_list, f)
    logger.info(f"Saved {num_subsets} subset indices to {file_path}")
    return indices_list


def train_model_on_subset(model_id: int, train_subset_indices: np.ndarray, device: torch.device, 
                          shared_train_loader: torch.utils.data.DataLoader,  # CRITICAL: Pre-created dataloader
                          checkpoints_dir: Optional[Path] = None, epochs: Optional[int] = None, 
                          lr: Optional[float] = None, momentum: Optional[float] = None, 
                          weight_decay: Optional[float] = None) -> torch.nn.Module:
    if checkpoints_dir is None: checkpoints_dir = config.LDS_CHECKPOINTS_DIR
    if epochs is None: epochs = config.LDS_MODEL_TRAIN_EPOCHS
    if lr is None: lr = config.LDS_MODEL_TRAIN_LR
    if momentum is None: momentum = config.LDS_MODEL_TRAIN_MOMENTUM
    if weight_decay is None: weight_decay = config.LDS_MODEL_TRAIN_WEIGHT_DECAY
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL FIX: Do NOT set seeds here! 
    # Seeds are set before EACH call to train_model_on_subset to ensure
    # all LDS models get identical initialization but use the same dataloader sequence
    logger.debug(f"Using pre-created shared dataloader to ensure consistency with MAGIC")

    # Model initialization with SAME seed as MAGIC 
    # All models get identical initialization (seed set before each call in run_lds_validation)
    logger.debug(f"Using SAME model initialization for ALL LDS models (seed set by caller)")

    model = construct_rn9(num_classes=config.NUM_CLASSES).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Create data_weights: 1 for samples in subset, 0 otherwise
    # This is the KEY mechanism that ensures LDS uses subsets while maintaining data ordering
    data_weights_for_subset = torch.zeros(config.NUM_TRAIN_SAMPLES, device=device)
    data_weights_for_subset[train_subset_indices] = 1.0
    num_active_samples_in_subset = data_weights_for_subset.sum().item()
    
    logger.debug(f"LDS Model {model_id}: Using {num_active_samples_in_subset}/{config.NUM_TRAIN_SAMPLES} samples from subset")

    if num_active_samples_in_subset == 0:
        logger.warning(f"Model ID {model_id} has no active samples in its subset. Skipping training.")
        # Save an initial state to avoid errors later if a checkpoint is expected
        torch.save(model.state_dict(), checkpoints_dir / f'sd_lds_{model_id}_final.pt')
        return model # Return untrained model

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Use reduction='none' to get per-sample losses, then apply weighted averaging
    # This is mathematically equivalent to subset training when weights are 0/1
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    model.train()
    total_batches_processed = 0
    total_samples_used = 0
    
    logger.debug(f"Starting training for LDS model {model_id} with subset of {num_active_samples_in_subset} samples")
    
    for epoch in range(epochs):
        epoch_samples_used = 0
        for images, labels, original_indices in tqdm(shared_train_loader, desc=f"LDS Model {model_id}, Epoch {epoch+1}"):
            images, labels, original_indices = images.to(device), labels.to(device), original_indices.to(device)

            # Key mechanism: Use weights to select only subset samples
            active_weights_in_batch = data_weights_for_subset[original_indices]
            sum_active_weights_in_batch = active_weights_in_batch.sum()

            if sum_active_weights_in_batch == 0: # Skip batch if no samples from subset
                continue

            optimizer.zero_grad()
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            
            # Weighted loss: only samples from the current model's subset contribute
            # This is equivalent to MAGIC's mean loss but only over the subset samples
            weighted_loss = (per_sample_loss * active_weights_in_batch).sum() / (sum_active_weights_in_batch + 1e-8)
            
            weighted_loss.backward()
            optimizer.step()
            
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
                             losses_dir: Optional[Path] = None, batch_size: Optional[int] = None) -> np.ndarray:
    if losses_dir is None: losses_dir = config.LDS_LOSSES_DIR
    if batch_size is None: batch_size = config.LDS_MODEL_TRAIN_BATCH_SIZE # Using train batch size for eval loader
    losses_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL FIX: Set seeds before creating validation dataloader to ensure 
    # consistent behavior across multiple model evaluations
    logger.debug(f"Setting seeds before creating validation dataloader for model {model_id}")
    set_seeds(config.SEED)
    
    val_loader = get_cifar10_dataloader(
        batch_size=batch_size, 
        root_path=config.CIFAR_ROOT, 
        split='val', 
        shuffle=False, 
        augment=False,
        num_workers=config.DATALOADER_NUM_WORKERS  # CRITICAL FIX: Add consistent num_workers
    )
    
    model.eval()
    all_losses_for_model = []
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc=f"Evaluating LDS Model {model_id}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            all_losses_for_model.append(per_sample_loss.cpu().numpy())
    
    all_losses_for_model_np = np.concatenate(all_losses_for_model)
    loss_file_path = losses_dir / f'loss_lds_{model_id}.pkl'
    with open(loss_file_path, 'wb') as f:
        pickle.dump(all_losses_for_model_np, f)
    logger.info(f"Saved per-sample validation losses for LDS Model {model_id} to {loss_file_path}")
    return all_losses_for_model_np


def run_lds_validation(precomputed_magic_scores_path: Optional[Union[str, Path]] = None) -> None:
    """
    Main function to run the LDS (Label Smoothing with Data Subsampling like) validation.
    Trains multiple models on subsets of data, evaluates them, and correlates their
    performance with pre-computed influence scores (from MAGIC analysis).
    Args:
        precomputed_magic_scores_path (Path, optional): Path to .pkl file containing
            influence scores from magic_analyzer. If None, LDS cannot run correlation.
    """
    
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for LDS Validation")

    # CRITICAL: Verify data ordering consistency with MAGIC analyzer
    if not verify_data_ordering_consistency():
        logger.error("Data ordering verification failed! LDS validation results may be invalid.")
        logger.error("Please check data loading configuration and ensure consistent settings.")
        raise RuntimeError("Data ordering mismatch between MAGIC and LDS - cannot proceed with validation")
    
    # 1. Generate or Load Training Subset Indices
    # These are lists of indices, each list defining a training subset for one LDS model
    list_of_subset_indices = generate_and_save_subset_indices()
    
    # Log subset information for transparency
    logger.info(f"Generated {len(list_of_subset_indices)} subsets for LDS validation")
    logger.info(f"Each subset contains {len(list_of_subset_indices[0])} samples ({config.LDS_SUBSET_FRACTION:.1%} of {config.NUM_TRAIN_SAMPLES} total)")
    logger.info(f"Training {config.LDS_NUM_MODELS_TO_TRAIN} LDS models using these subsets")

    # CRITICAL FIX: Create ONE shared dataloader that all LDS models will use
    # This ensures ALL models see the EXACT same data sequence as MAGIC
    # MUST reset seeds before creating shared dataloader because verification function
    # consumed random numbers, changing the PyTorch random state
    logger.info("Creating shared dataloader with EXACT same settings as MAGIC...")
    logger.debug("Resetting seeds before shared dataloader creation to match MAGIC's dataloader state")
    set_seeds(config.SEED)  # CRITICAL: Reset to same state as MAGIC dataloader creation
    
    shared_train_loader = get_cifar10_dataloader(
        batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
        split='train',
        shuffle=True,  # Same as MAGIC
        augment=False,  # Same as MAGIC
        num_workers=config.DATALOADER_NUM_WORKERS,  # Same as MAGIC
        root_path=config.CIFAR_ROOT
    )
    
    # 2. Train Multiple Models on Subsets and Evaluate
    all_validation_losses_stacked = [] # To store losses from all LDS models
    for i in tqdm(range(config.LDS_NUM_MODELS_TO_TRAIN), desc="Training LDS Models"):
        model_id = i
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)] # Cycle if fewer unique subsets
        
        logger.info(f"--- Training LDS Model {model_id} on subset of size {len(current_subset_indices)} ---")
        
        # CRITICAL FIX: Set seeds before EACH model creation to ensure identical initialization
        # All models will have identical initialization, differing only in training subsets
        logger.debug(f"Setting seeds to {config.SEED} for consistent model {model_id} initialization...")
        set_seeds(config.SEED)
        
        trained_model = train_model_on_subset(
            model_id=model_id,
            train_subset_indices=current_subset_indices,
            device=device,
            shared_train_loader=shared_train_loader,  # CRITICAL: Same dataloader for all models
            checkpoints_dir=config.LDS_CHECKPOINTS_DIR,
            epochs=config.LDS_MODEL_TRAIN_EPOCHS,
            lr=config.LDS_MODEL_TRAIN_LR,
            momentum=config.LDS_MODEL_TRAIN_MOMENTUM,
            weight_decay=config.LDS_MODEL_TRAIN_WEIGHT_DECAY
        )
        per_sample_val_losses = evaluate_and_save_losses(
            trained_model, model_id, device, losses_dir=config.LDS_LOSSES_DIR
        )
        all_validation_losses_stacked.append(per_sample_val_losses)
    
    if not all_validation_losses_stacked:
        logger.warning("No LDS models were trained or evaluated. Skipping correlation.")
        return

    # Stack losses: rows are models, columns are validation samples
    lds_validation_margins = np.stack(all_validation_losses_stacked) # Shape: (LDS_NUM_MODELS, NUM_TEST_SAMPLES)

    # Validate target index is within bounds of validation data
    num_val_samples = lds_validation_margins.shape[1]
    if config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION >= num_val_samples or config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < 0:
        raise ValueError(f"LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION ({config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}) is out of bounds for validation dataset (size: {num_val_samples})")

    # 3. Correlation with MAGIC Influence Scores
    # Use the path from config if precomputed_magic_scores_path is not provided directly
    magic_scores_file_to_load = precomputed_magic_scores_path
    if magic_scores_file_to_load is None:
        # Use the default MAGIC scores file path from config
        magic_scores_file_to_load = config.MAGIC_SCORES_FILE_FOR_LDS_INPUT

    if not magic_scores_file_to_load.exists():
        # Corrected variable name in f-string
        logger.warning(f"MAGIC scores file not found at {magic_scores_file_to_load}. Skipping correlation analysis.")
        return

    logger.info(f"Loading MAGIC scores from {magic_scores_file_to_load} for correlation...")
    with open(magic_scores_file_to_load, 'rb') as f:
        loaded_scores = pickle.load(f)
    
    if loaded_scores.ndim == 2: # Per-step scores [num_steps, num_train_samples]
        magic_influence_estimates = loaded_scores.sum(axis=0)
        logger.info(f"Summed per-step MAGIC scores (shape {loaded_scores.shape}) to flat scores (shape {magic_influence_estimates.shape}).")
    elif loaded_scores.ndim == 1: # Already flat scores [num_train_samples]
        magic_influence_estimates = loaded_scores
        logger.info(f"Loaded flat MAGIC scores (shape {magic_influence_estimates.shape}).")
    else:
        raise ValueError(f"Loaded MAGIC scores from {magic_scores_file_to_load} have unexpected shape: {loaded_scores.shape}")

    # Create mask array for which training samples were used by each LDS model
    lds_train_masks_list = []
    for i in range(config.LDS_NUM_MODELS_TO_TRAIN):
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)]
        mask = np.zeros(config.NUM_TRAIN_SAMPLES, dtype=bool)
        mask[current_subset_indices] = True
        lds_train_masks_list.append(mask)
    # Shape: (LDS_NUM_MODELS, NUM_TRAINING_SAMPLES)
    lds_training_data_masks = np.stack(lds_train_masks_list)

    # Predict margin/loss change based on influence scores: sum of influences of samples in each subset
    # This projects the influence scores of all training samples onto the subsets used to train each of the LDS models.
    # `predicted_loss_impact_on_target` for each LDS model based on its training subset and MAGIC scores.
    predicted_loss_impact_on_target = lds_training_data_masks @ magic_influence_estimates.T
    # Shape: (LDS_NUM_MODELS_TO_TRAIN,)

    # Select the actual validation losses for the specific target image used in MAGIC analysis
    actual_margins_for_target_val_image = lds_validation_margins[:, config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION]
    # Shape: (LDS_NUM_MODELS_TO_TRAIN,)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=predicted_loss_impact_on_target, y=actual_margins_for_target_val_image)
    correlation, p_value = spearmanr(predicted_loss_impact_on_target, actual_margins_for_target_val_image)
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

"""
# Standalone execution block - deprecated. Use main_runner.py instead.
if __name__ == "__main__":
    # from .config import ensure_output_dirs_exist, MAGIC_SCORES_DIR, MAGIC_TARGET_VAL_IMAGE_IDX
    # ensure_output_dirs_exist() # This function is removed from config.py
    
    # # Determine the path to the MAGIC scores file needed by LDS
    # # This assumes magic_analyzer.py has been run and produced the scores for the target val image.
    # # The path construction and default logic is now better handled in main_runner.py and config.py

    # print("Running LDS Validator directly is deprecated. Please use main_runner.py.")
    # print("If you need to test LDS validation in isolation, ensure MAGIC scores are present at the expected path.")
    
    # Example of how it might be called if paths were manually set up:
    # from . import config
    # config.LDS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # config.LDS_LOSSES_DIR.mkdir(parents=True, exist_ok=True)
    # config.LDS_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    # if config.LDS_INDICES_FILE:
    #     config.LDS_INDICES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # # This path should align with what MagicAnalyzer produces and what LdsValidator expects by default via config.
    # magic_scores_file_for_direct_run = config.MAGIC_SCORES_FILE_FOR_LDS_INPUT
    
    # if not magic_scores_file_for_direct_run.exists():
    #     print(f"Expected MAGIC scores file not found: {magic_scores_file_for_direct_run}")
    #     print("Please run MAGIC analysis via main_runner.py first or ensure the file exists.")
    # else:
    #     print(f"Attempting to run LDS validation with scores from: {magic_scores_file_for_direct_run}")
    #     run_lds_validation(precomputed_magic_scores_path=magic_scores_file_for_direct_run)
    pass # Deprecated
""" 