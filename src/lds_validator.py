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

# Project-specific imports
from .config import (
    RANDOM_SEED, CIFAR_ROOT, LDS_CHECKPOINTS_DIR, LDS_LOSSES_DIR,
    LDS_INDICES_FILE, LDS_PLOTS_DIR, LDS_NUM_MODELS_TO_TRAIN,
    LDS_SUBSET_SIZE_FRACTION, LDS_NUM_SUBSETS, NUM_TRAINING_SAMPLES_CIFAR10,
    LDS_TRAIN_BATCH_SIZE, LDS_TRAIN_EPOCHS, LDS_BASE_LR,
    LDS_MOMENTUM, LDS_WEIGHT_DECAY, MAGIC_SCORES_DIR, LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION
)
from .utils import set_seeds
from .model_def import construct_resnet9
from .data_handling import get_cifar_dataloader, CustomCIFAR10Dataset
# LDS does not use the global batch_dict_for_replay from magic_analyzer


def generate_and_save_subset_indices(num_subsets=LDS_NUM_SUBSETS,
                                     total_samples=NUM_TRAINING_SAMPLES_CIFAR10,
                                     subset_fraction=LDS_SUBSET_SIZE_FRACTION,
                                     file_path=LDS_INDICES_FILE,
                                     force_regenerate=False):
    """
    Generates or loads lists of training sample indices for creating subsets.
    Each list defines a subset of the training data.
    """
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
    np.random.seed(RANDOM_SEED) # Ensure reproducibility for index generation
    subset_sample_count = int(total_samples * subset_fraction)
    indices_list = [
        np.random.choice(range(total_samples), subset_sample_count, replace=False)
        for _ in range(num_subsets)
    ]
    with open(file_path, 'wb') as f:
        pickle.dump(indices_list, f)
    print(f"Saved {num_subsets} subset indices to {file_path}")
    return indices_list


def train_model_on_subset(model_id, train_subset_indices, device,
                          checkpoints_dir=LDS_CHECKPOINTS_DIR,
                          epochs=LDS_TRAIN_EPOCHS, batch_size=LDS_TRAIN_BATCH_SIZE,
                          lr=LDS_BASE_LR, momentum=LDS_MOMENTUM, weight_decay=LDS_WEIGHT_DECAY):
    """
    Trains a model on a specified subset of the training data.
    The loss is weighted such that only samples in `train_subset_indices` contribute.
    Saves the final model checkpoint.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(RANDOM_SEED + model_id) # Ensure different initializations for different models if desired, or fixed seed

    model = construct_resnet9(num_classes=10).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Create data_weights: 1 for samples in subset, 0 otherwise
    data_weights_for_subset = torch.zeros(NUM_TRAINING_SAMPLES_CIFAR10, device=device)
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
    train_loader_full = get_cifar_dataloader(batch_size, split='train', shuffle=True, augment=True)

    model.train()
    for epoch in range(epochs):
        for images, labels, original_indices in tqdm(train_loader_full, desc=f"LDS Model {model_id}, Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            original_indices = original_indices.to(device)

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


def evaluate_and_save_losses(model, model_id, device, losses_dir=LDS_LOSSES_DIR):
    """
    Evaluates the model on the entire validation set and saves per-sample losses.
    """
    losses_dir.mkdir(parents=True, exist_ok=True)
    val_loader = get_cifar_dataloader(batch_size=LDS_TRAIN_BATCH_SIZE, split='val', shuffle=False, augment=False)
    
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
    set_seeds(RANDOM_SEED)
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LDS Validation")

    # 1. Generate or Load Training Subset Indices
    # These are lists of indices, each list defining a training subset for one LDS model
    list_of_subset_indices = generate_and_save_subset_indices(
        num_subsets=LDS_NUM_SUBSETS, # Max number of index lists to generate
        file_path=LDS_INDICES_FILE
    )

    # 2. Train Multiple Models on Subsets and Evaluate
    all_validation_losses_stacked = [] # To store losses from all LDS models
    for i in tqdm(range(LDS_NUM_MODELS_TO_TRAIN), desc="Training LDS Models"):
        model_id = i
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)] # Cycle if fewer unique subsets
        
        print(f"--- Training LDS Model {model_id} on subset of size {len(current_subset_indices)} ---")
        trained_model = train_model_on_subset(
            model_id=model_id,
            train_subset_indices=current_subset_indices,
            device=device,
            checkpoints_dir=LDS_CHECKPOINTS_DIR,
            epochs=LDS_TRAIN_EPOCHS,
            batch_size=LDS_TRAIN_BATCH_SIZE,
            lr=LDS_BASE_LR,
            momentum=LDS_MOMENTUM,
            weight_decay=LDS_WEIGHT_DECAY
        )
        per_sample_val_losses = evaluate_and_save_losses(
            trained_model, model_id, device, losses_dir=LDS_LOSSES_DIR
        )
        all_validation_losses_stacked.append(per_sample_val_losses)
    
    if not all_validation_losses_stacked:
        print("No LDS models were trained or evaluated. Skipping correlation.")
        return

    # Stack losses: rows are models, columns are validation samples
    lds_validation_margins = np.stack(all_validation_losses_stacked) # Shape: (LDS_NUM_MODELS, NUM_TEST_SAMPLES)

    # 3. Correlation with MAGIC Influence Scores
    if precomputed_magic_scores_path is None or not precomputed_magic_scores_path.exists():
        print(f"MAGIC scores file not found at {precomputed_magic_scores_path}. Skipping correlation analysis.")
        return

    print(f"Loading MAGIC scores from {precomputed_magic_scores_path} for correlation...")
    with open(precomputed_magic_scores_path, 'rb') as f:
        # These are the influence scores of each training point on a specific target (e.g., val image 21)
        magic_influence_estimates = pickle.load(f) # Should be shape (NUM_TRAINING_SAMPLES,)

    # Create mask array for which training samples were used by each LDS model
    lds_train_masks_list = []
    for i in range(LDS_NUM_MODELS_TO_TRAIN):
        current_subset_indices = list_of_subset_indices[i % len(list_of_subset_indices)]
        mask = np.zeros(NUM_TRAINING_SAMPLES_CIFAR10, dtype=bool)
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
    actual_margins_for_target_val_image = lds_validation_margins[:, LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION]
    # Shape: (LDS_NUM_MODELS_TO_TRAIN,)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=predicted_loss_impact_on_target, y=actual_margins_for_target_val_image)
    correlation, p_value = spearmanr(predicted_loss_impact_on_target, actual_margins_for_target_val_image)
    plt.title(f'LDS Validation: Correlation with MAGIC Scores\nTarget Val Img Idx: {LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}\nSpearman R: {correlation:.3f} (p={p_value:.3g})')
    plt.xlabel("Predicted Loss Impact (Sum of MAGIC scores in subset)")
    plt.ylabel(f"Actual Loss on Val Img {LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}")
    plt.grid(True)
    plot_save_path = LDS_PLOTS_DIR / f"lds_correlation_val_{LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}.png"
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"LDS Correlation plot saved to {plot_save_path}")
    plt.show()

    print(f"LDS Validation finished. Correlation: {correlation:.3f}")


if __name__ == "__main__":
    from .config import ensure_output_dirs_exist, MAGIC_SCORES_DIR, MAGIC_TARGET_VAL_IMAGE_IDX
    ensure_output_dirs_exist()
    
    # Determine the path to the MAGIC scores file needed by LDS
    # This assumes magic_analyzer.py has been run and produced the scores for the target val image.
    magic_scores_file = MAGIC_SCORES_DIR / f'magic_scores_val_{MAGIC_TARGET_VAL_IMAGE_IDX}.pkl'
    
    if not magic_scores_file.exists():
        print(f"Expected MAGIC scores file not found: {magic_scores_file}")
        print("Please run magic_analyzer.py first or provide the correct path.")
    else:
        run_lds_validation(precomputed_magic_scores_path=magic_scores_file) 