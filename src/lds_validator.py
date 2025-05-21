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

# Project-specific imports
from . import config
from .utils import set_seeds
from .model_def import construct_rn9
from .data_handling import get_cifar10_dataloader, CustomDataset
# LDS does not use the global batch_dict_for_replay from magic_analyzer


def generate_and_save_subset_indices(num_subsets=None, total_samples=None, 
                                     subset_fraction=None, file_path=None, force_regenerate=False):
    if num_subsets is None: num_subsets = config.LDS_NUM_SUBSETS_TO_GENERATE
    if total_samples is None: total_samples = config.NUM_TRAIN_SAMPLES # Use general NUM_TRAIN_SAMPLES
    if subset_fraction is None: subset_fraction = config.LDS_SUBSET_FRACTION
    if file_path is None: file_path = config.LDS_INDICES_FILE

    if not force_regenerate and file_path.exists():
        print(f"Loading subset indices from {file_path}")
        with open(file_path, 'rb') as f:
            indices_list = pickle.load(f)
        # Ensure we have enough subsets if config changed
        if len(indices_list) >= num_subsets:
            return indices_list[:num_subsets]
        else:
            print(f"Found only {len(indices_list)} subsets, regenerating for {num_subsets}.")
    
    print(f"Generating {num_subsets} subset indices...")
    np.random.seed(config.SEED) # Ensure reproducibility for index generation
    subset_sample_count = int(total_samples * subset_fraction)
    indices_list = [
        np.random.choice(range(total_samples), subset_sample_count, replace=False)
        for _ in range(num_subsets)
    ]
    with open(file_path, 'wb') as f:
        pickle.dump(indices_list, f)
    print(f"Saved {num_subsets} subset indices to {file_path}")
    return indices_list


def train_model_on_subset(model_id, train_subset_indices, device, checkpoints_dir=None,
                          epochs=None, batch_size=None, lr=None, momentum=None, weight_decay=None):
    if checkpoints_dir is None: checkpoints_dir = config.LDS_CHECKPOINTS_DIR
    if epochs is None: epochs = config.LDS_MODEL_TRAIN_EPOCHS
    if batch_size is None: batch_size = config.LDS_MODEL_TRAIN_BATCH_SIZE
    if lr is None: lr = config.LDS_MODEL_TRAIN_LR
    if momentum is None: momentum = config.LDS_MODEL_TRAIN_MOMENTUM
    if weight_decay is None: weight_decay = config.LDS_MODEL_TRAIN_WEIGHT_DECAY
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config.SEED) # Ensure different initializations for different models if desired, or fixed seed

    model = construct_rn9(num_classes=config.NUM_CLASSES).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Create data_weights: 1 for samples in subset, 0 otherwise
    data_weights_for_subset = torch.zeros(config.NUM_TRAIN_SAMPLES, device=device)
    data_weights_for_subset[train_subset_indices] = 1.0
    num_active_samples_in_subset = data_weights_for_subset.sum().item()

    if num_active_samples_in_subset == 0:
        print(f"Warning: Model ID {model_id} has no active samples in its subset. Skipping training.")
        # Save an initial state to avoid errors later if a checkpoint is expected
        torch.save(model.state_dict(), checkpoints_dir / f'sd_lds_{model_id}_final.pt')
        return model # Return untrained model

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion_no_reduction = CrossEntropyLoss(reduction='none')

    # Full training loader, weights will handle subset selection
    train_loader_full = get_cifar10_dataloader(batch_size=batch_size, root_path=config.CIFAR_ROOT, split='train', shuffle=True, augment=True)

    model.train()
    for epoch in range(epochs):
        for images, labels, original_indices in tqdm(train_loader_full, desc=f"LDS Model {model_id}, Epoch {epoch+1}"):
            images, labels, original_indices = images.to(device), labels.to(device), original_indices.to(device)

            active_weights_in_batch = data_weights_for_subset[original_indices]
            sum_active_weights_in_batch = active_weights_in_batch.sum()

            if sum_active_weights_in_batch == 0: # Skip batch if no samples from subset
                continue

            optimizer.zero_grad()
            outputs = model(images)
            per_sample_loss = criterion_no_reduction(outputs, labels)
            
            # Weighted loss: only samples from the current model's subset contribute
            weighted_loss = (per_sample_loss * active_weights_in_batch).sum() / sum_active_weights_in_batch
            
            weighted_loss.backward()
            optimizer.step()

    final_checkpoint_path = checkpoints_dir / f'sd_lds_{model_id}_final.pt'
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"LDS Model {model_id} trained. Final checkpoint: {final_checkpoint_path}")
    return model


def evaluate_and_save_losses(model, model_id, device, losses_dir=None, batch_size=None):
    if losses_dir is None: losses_dir = config.LDS_LOSSES_DIR
    if batch_size is None: batch_size = config.LDS_MODEL_TRAIN_BATCH_SIZE # Using train batch size for eval loader
    losses_dir.mkdir(parents=True, exist_ok=True)
    
    val_loader = get_cifar10_dataloader(batch_size=batch_size, root_path=config.CIFAR_ROOT, split='val', shuffle=False, augment=False)
    
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
    print(f"Saved per-sample validation losses for LDS Model {model_id} to {loss_file_path}")
    return all_losses_for_model_np


def run_lds_validation(precomputed_magic_scores_path=None):
    """
    Main function to run the LDS (Label Smoothing with Data Subsampling like) validation.
    Trains multiple models on subsets of data, evaluates them, and correlates their
    performance with pre-computed influence scores (from MAGIC analysis).
    Args:
        precomputed_magic_scores_path (Path, optional): Path to .pkl file containing
            influence scores from magic_analyzer. If None, LDS cannot run correlation.
    """
    set_seeds(config.SEED)
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LDS Validation")

    # 1. Generate or Load Training Subset Indices
    # These are lists of indices, each list defining a training subset for one LDS model
    list_of_subset_indices = generate_and_save_subset_indices()

    # 2. Train Multiple Models on Subsets and Evaluate
    all_validation_losses_stacked = [] # To store losses from all LDS models
    for i in tqdm(range(config.LDS_NUM_MODELS_TO_TRAIN), desc="Training LDS Models"):
        model_id = i
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)] # Cycle if fewer unique subsets
        
        print(f"--- Training LDS Model {model_id} on subset of size {len(current_subset_indices)} ---")
        trained_model = train_model_on_subset(
            model_id=model_id,
            train_subset_indices=current_subset_indices,
            device=device,
            checkpoints_dir=config.LDS_CHECKPOINTS_DIR,
            epochs=config.LDS_MODEL_TRAIN_EPOCHS,
            batch_size=config.LDS_MODEL_TRAIN_BATCH_SIZE,
            lr=config.LDS_MODEL_TRAIN_LR,
            momentum=config.LDS_MODEL_TRAIN_MOMENTUM,
            weight_decay=config.LDS_MODEL_TRAIN_WEIGHT_DECAY
        )
        per_sample_val_losses = evaluate_and_save_losses(
            trained_model, model_id, device, losses_dir=config.LDS_LOSSES_DIR
        )
        all_validation_losses_stacked.append(per_sample_val_losses)
    
    if not all_validation_losses_stacked:
        print("No LDS models were trained or evaluated. Skipping correlation.")
        return

    # Stack losses: rows are models, columns are validation samples
    lds_validation_margins = np.stack(all_validation_losses_stacked) # Shape: (LDS_NUM_MODELS, NUM_TEST_SAMPLES)

    # 3. Correlation with MAGIC Influence Scores
    # Use the path from config if precomputed_magic_scores_path is not provided directly
    magic_scores_file_to_load = precomputed_magic_scores_path
    if magic_scores_file_to_load is None:
        # Assuming MAGIC_SCORES_FILE_FOR_LDS_INPUT is defined in config and points to the correct file
        # from the revised config, this would be config.get_magic_scores_path(config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)
        # or config.MAGIC_SCORES_FILE_FOR_LDS_INPUT directly if it's a full path object
        magic_scores_file_to_load = config.MAGIC_SCORES_DIR / f'magic_scores_per_step_val_{config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}.pkl'
        # The line above assumes a naming convention. It's better if MAGIC_SCORES_FILE_FOR_LDS_INPUT from config.py is used directly.
        # For robustness, let's use the variable that should be defined in the new config for this purpose:
        magic_scores_file_to_load = config.MAGIC_SCORES_FILE_FOR_LDS_INPUT

    if not magic_scores_file_to_load.exists():
        # Corrected variable name in f-string
        print(f"MAGIC scores file not found at {magic_scores_file_to_load}. Skipping correlation analysis.")
        return

    print(f"Loading MAGIC scores from {magic_scores_file_to_load} for correlation...")
    with open(magic_scores_file_to_load, 'rb') as f:
        loaded_scores = pickle.load(f)
    
    if loaded_scores.ndim == 2: # Per-step scores [num_steps, num_train_samples]
        magic_influence_estimates = loaded_scores.sum(axis=0)
        print(f"Summed per-step MAGIC scores (shape {loaded_scores.shape}) to flat scores (shape {magic_influence_estimates.shape}).")
    elif loaded_scores.ndim == 1: # Already flat scores [num_train_samples]
        magic_influence_estimates = loaded_scores
        print(f"Loaded flat MAGIC scores (shape {magic_influence_estimates.shape}).")
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
    print(f"LDS Correlation plot saved to {plot_save_path}")
    plt.show()

    print(f"LDS Validation finished. Correlation: {correlation:.3f}")

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