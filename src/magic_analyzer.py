import pickle
import warnings
import torch
import numpy as np
import torchvision
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os # Added for os.path.exists

# Project-specific imports
from .config import (
    RANDOM_SEED, MAGIC_CHECKPOINTS_DIR, CIFAR_ROOT,
    NUM_TRAINING_SAMPLES_CIFAR10, MAGIC_REPLAY_LEARNING_RATE,
    MAGIC_TARGET_VAL_IMAGE_IDX, MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW,
    MAGIC_TRAIN_BATCH_SIZE, MAGIC_TRAIN_EPOCHS, MAGIC_BASE_LR,
    MAGIC_MOMENTUM, MAGIC_WEIGHT_DECAY, MAGIC_SCORES_DIR, MAGIC_PLOTS_DIR
)
from .utils import set_seeds
from .model_def import construct_resnet9
from .data_handling import (
    get_cifar_dataloader, CustomCIFAR10Dataset, SingleItemDataset, get_single_item_loader
)
from .visualization import plot_influence_images


# Global dictionary to store batches for replay
# This is a simplification. For very long training or large data,
# consider saving batch indices and reloading, or a more sophisticated replay buffer.
batch_dict_for_replay = {}


# --- Model Training (for MAGIC analysis) ---
def train_model_for_magic(model, train_loader, device,
                          epochs=MAGIC_TRAIN_EPOCHS, model_id=0,
                          checkpoints_dir=MAGIC_CHECKPOINTS_DIR,
                          base_lr=MAGIC_BASE_LR, momentum=MAGIC_MOMENTUM,
                          weight_decay=MAGIC_WEIGHT_DECAY):
    """
    Trains the model for MAGIC analysis, saving checkpoints at each step.
    The global `batch_dict_for_replay` is populated here.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        epochs (int): Number of epochs to train for.
        model_id (int): Identifier for the model (used in checkpoint naming).
        checkpoints_dir (Path): Directory to save checkpoints.
        base_lr, momentum, weight_decay: Optimizer parameters.
    Returns:
        tuple: (trained_model, total_steps_taken)
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    
    global batch_dict_for_replay # To modify the global dict
    batch_dict_for_replay.clear() # Clear if rerunning
    
    global_step = 0 # Counter for total training steps
    # Save initial model state (checkpoint 0 - before any training steps)
    # Check if the first checkpoint exists, to potentially resume, though this function currently always retrains
    # For simplicity now, it always starts from scratch if called.
    torch.save(model.state_dict(), checkpoints_dir / f'sd_{model_id}_{global_step}.pt')
    global_step += 1

    model.train() # Set model to training mode
    for epoch in range(epochs):
        print(f"MAGIC Model Training: Epoch {epoch+1}/{epochs}")
        for batch_images, batch_labels, batch_indices in tqdm(train_loader, desc=f"MAGIC Epoch {epoch+1}"):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            batch_dict_for_replay[global_step] = {
                'ims': batch_images.cpu().clone(),
                'labs': batch_labels.cpu().clone(),
                'idx': batch_indices.cpu().clone(),
            }
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            torch.save(model.state_dict(), checkpoints_dir / f'sd_{model_id}_{global_step}.pt')
            global_step += 1
            
    num_training_steps = global_step - 1
    print(f"MAGIC model training finished. {num_training_steps} steps taken. "
          f"{num_training_steps+1} checkpoints saved (0 to {num_training_steps}).")
    return model, num_training_steps


# --- Influence Calculation (REPLAY/TracIn-like) ---
def _param_list_dot_product(list1, list2, device):
    """Computes the dot product between two lists of tensors (parameters/gradients)."""
    res = torch.tensor(0., device=device)
    for p1, p2 in zip(list1, list2):
        if p1 is not None and p2 is not None:
            res += (p1.reshape(-1).to(device) @ p2.reshape(-1).to(device))
    return res

def compute_magic_influence_scores(final_model_checkpoint_path, target_image_loader,
                                   num_training_steps, model_id, device,
                                   checkpoints_dir=MAGIC_CHECKPOINTS_DIR,
                                   num_train_samples=NUM_TRAINING_SAMPLES_CIFAR10,
                                   replay_learning_rate=MAGIC_REPLAY_LEARNING_RATE):
    """
    Computes influence scores for the MAGIC analysis using a REPLAY-like backward pass.
    Args:
        final_model_checkpoint_path (Path): Path to the final trained model checkpoint.
        target_image_loader (DataLoader): DataLoader yielding the single target test image.
        num_training_steps (int): Total number of training steps taken (highest checkpoint index).
        model_id (int): Identifier for the model.
        device (torch.device): Device for computation.
        replay_learning_rate (float): Learning rate for the REPLAY formula (s_t - lr*g_t).
    Returns:
        np.array: Influence scores for each training sample.
    """
    print("Starting MAGIC influence score computation...")
    model_arch = construct_resnet9().to(device)

    model_arch.load_state_dict(torch.load(final_model_checkpoint_path, map_location=device))
    model_arch.eval()
    
    criterion_test = CrossEntropyLoss()
    test_ims, test_labs, _ = next(iter(target_image_loader))
    test_ims, test_labs = test_ims.to(device), test_labs.to(device)

    model_arch.zero_grad()
    test_output = model_arch(test_ims)
    test_loss = criterion_test(test_output, test_labs)
    test_loss.backward()
    
    delta_k = [p.grad.clone() if p.grad is not None else torch.zeros_like(p).to(device)
               for p in model_arch.parameters()]

    data_weights = torch.nn.Parameter(torch.ones(num_train_samples, device=device), requires_grad=True)
    contributions = []
    criterion_train_no_reduction = CrossEntropyLoss(reduction='none')

    # Ensure batch_dict_for_replay is populated if we are here.
    # This would ideally be checked before calling compute_magic_influence_scores
    # if training was skipped due to existing checkpoints.
    # For now, assume if this function is called, training has occurred and populated it.
    if not batch_dict_for_replay and num_training_steps > 0 :
        # This case should ideally be handled by populating batch_dict_for_replay
        # by loading saved batches if training is skipped.
        # For now, this is a placeholder for a more complex recovery.
        print("Warning: batch_dict_for_replay is empty but num_training_steps > 0. Trying to rebuild.")
        # Attempt to rebuild batch_dict_for_replay by re-iterating through the train_loader
        # This requires train_loader to be accessible or passed in.
        # This is a simplification and might not match the original run exactly if shuffle was True.
        # For now, raising an error as this part needs more robust implementation.
        raise NotImplementedError("Rebuilding batch_dict_for_replay on the fly is not fully implemented for skipped training.")


    for step_t in tqdm(range(num_training_steps, 0, -1), desc="MAGIC Influence Pass"):
        s_t_chkp_path = checkpoints_dir / f'sd_{model_id}_{step_t-1}.pt'
        if not s_t_chkp_path.exists():
            print(f"Warning: Checkpoint {s_t_chkp_path} not found. Skipping step {step_t}.")
            contributions.append(torch.zeros_like(data_weights).detach().cpu().clone())
            continue

        model_arch.load_state_dict(torch.load(s_t_chkp_path, map_location=device))
        model_arch.train() # Set to train for consistent behavior with training-time gradients
        
        current_sk_params = list(model_arch.parameters()) # These are s_k
        for p in current_sk_params:
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad.detach_().zero_()

        if step_t not in batch_dict_for_replay:
            print(f"Warning: Batch data for step {step_t} not found. Skipping step.")
            contributions.append(torch.zeros_like(data_weights).detach().cpu().clone())
            continue

        batch_data = batch_dict_for_replay[step_t]
        b_ims, b_labs, b_idx = batch_data['ims'].to(device), batch_data['labs'].to(device), batch_data['idx'].to(device)

        # L_k(s_k, w) = weighted_loss
        train_output = model_arch(b_ims) # Uses s_k (current_sk_params)
        loss_samples = criterion_train_no_reduction(train_output, b_labs)
        weighted_loss = (loss_samples * data_weights[b_idx]).mean()
        
        # g_k(s_k, w) = grad_of_batch_loss_wrt_sk
        # This is d(L_k)/d(s_k). create_graph=True is vital.
        grad_of_batch_loss_wrt_sk_tuple = torch.autograd.grad(weighted_loss, current_sk_params, create_graph=True, allow_unused=False)
        grad_of_batch_loss_wrt_sk = [g.clone() for g in grad_of_batch_loss_wrt_sk_tuple]

        # s_{k+1}(w) = s_k - lr * g_k(s_k, w)
        sk_plus_1_dependent_on_w = [sk_p - replay_learning_rate * gk_p 
                                      for sk_p, gk_p in zip(current_sk_params, grad_of_batch_loss_wrt_sk)]

        # Q_k = s_{k+1}(w)^T * Delta_{k+1}
        # delta_k here is Delta_{k+1} (from previous backward step or initialization)
        scalar_for_gradients = _param_list_dot_product(sk_plus_1_dependent_on_w, delta_k, device)

        # beta_k = d(Q_k) / d(w)
        if data_weights.grad is not None:
            data_weights.grad.detach_().zero_()
        grads_beta = torch.autograd.grad(scalar_for_gradients, data_weights, retain_graph=True, allow_unused=False)
        beta_t_step = grads_beta[0]
        contributions.append(beta_t_step.detach().cpu().clone())

        # Delta_k = d(Q_k) / d(s_k)
        # This new delta_k will be Delta_k for the next backward iteration (t-1)
        delta_k_new_grads_tuple = torch.autograd.grad(scalar_for_gradients, current_sk_params, allow_unused=False)
        delta_k = [g.clone() for g in delta_k_new_grads_tuple]
    
    if not contributions:
        print("Warning: No influence contributions recorded during MAGIC analysis.")
        return np.zeros(num_train_samples)
    
    # Filter out any potential all-zero tensors if steps were skipped
    valid_contributions = [c for c in contributions if c.abs().sum() > 0]
    if not valid_contributions:
        print("Warning: All contributions were zero. Returning zero scores.")
        return np.zeros(num_train_samples)
        
    total_scores = torch.stack(valid_contributions).sum(axis=0).numpy()
    print("MAGIC influence score computation finished.")
    return total_scores


# --- Main Execution Logic for MAGIC Analysis ---
def run_magic_analysis():
    """Main function to run the MAGIC influence analysis pipeline."""
    set_seeds(RANDOM_SEED)
    warnings.filterwarnings('ignore') # Keep this for cleaner output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for MAGIC Analysis")

    # Define expected output file paths
    scores_file_path = MAGIC_SCORES_DIR / f'magic_scores_val_{MAGIC_TARGET_VAL_IMAGE_IDX}.pkl'
    plot_file_path = MAGIC_PLOTS_DIR / f'magic_influence_val_{MAGIC_TARGET_VAL_IMAGE_IDX}.png'
    MAGIC_SCORES_DIR.mkdir(parents=True, exist_ok=True) # Ensure directories exist
    MAGIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    magic_scores = None
    run_full_pipeline = True
    run_plotting_only = False

    if scores_file_path.exists() and plot_file_path.exists():
        print(f"MAGIC analysis complete. Scores at {scores_file_path} and plot at {plot_file_path} already exist.")
        # Load scores if they need to be returned for LDS
        with open(scores_file_path, 'rb') as f:
            magic_scores = pickle.load(f)
        return magic_scores # Skip everything else

    if scores_file_path.exists() and not plot_file_path.exists():
        print(f"Found existing MAGIC scores at {scores_file_path}. Plot is missing. Will load scores and generate plot.")
        with open(scores_file_path, 'rb') as f:
            magic_scores = pickle.load(f)
        run_full_pipeline = False
        run_plotting_only = True

    # 1. Prepare DataLoaders (needed for plotting even if scores are loaded)
    print("Loading datasets for MAGIC Analysis...")
    train_loader = get_cifar_dataloader(
        batch_size=MAGIC_TRAIN_BATCH_SIZE, split='train', shuffle=True, augment=True, root_dir=CIFAR_ROOT
    )
    
    val_full_ds = CustomCIFAR10Dataset(
        root=str(CIFAR_ROOT), train=False, download=True,
        transform=get_cifar_dataloader(1, split='val', augment=False, root_dir=CIFAR_ROOT).dataset.transform
    )
    target_image_data_tuple = None
    for i in range(len(val_full_ds)):
        img, lbl, idx_val = val_full_ds[i]
        if idx_val == MAGIC_TARGET_VAL_IMAGE_IDX:
            target_image_data_tuple = (img, lbl, torch.tensor(idx_val))
            break
    if target_image_data_tuple is None:
        raise ValueError(f"MAGIC Target validation image {MAGIC_TARGET_VAL_IMAGE_IDX} not found.")
    single_target_loader = get_single_item_loader(target_image_data_tuple)

    if run_full_pipeline:
        print("Running full MAGIC analysis pipeline (training, scores, plot).")
        # 2. Train Model for MAGIC analysis
        print("Preparing model for MAGIC training...")
        model = construct_resnet9(num_classes=10).to(device)
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)

        _, num_total_steps = train_model_for_magic(
            model, train_loader, device,
            epochs=MAGIC_TRAIN_EPOCHS,
            checkpoints_dir=MAGIC_CHECKPOINTS_DIR,
            base_lr=MAGIC_BASE_LR,
            momentum=MAGIC_MOMENTUM,
            weight_decay=MAGIC_WEIGHT_DECAY
        )
        final_chkp_path = MAGIC_CHECKPOINTS_DIR / f'sd_0_{num_total_steps}.pt'

        # 3. Compute Influence Scores
        if not batch_dict_for_replay:
            # This check is important if train_model_for_magic could be skipped
            # For now, it's always run in the full pipeline.
            raise RuntimeError("batch_dict_for_replay is empty for MAGIC. Training did not populate it.")

        magic_scores = compute_magic_influence_scores(
            final_model_checkpoint_path=final_chkp_path,
            target_image_loader=single_target_loader,
            num_training_steps=num_total_steps,
            model_id=0,
            device=device,
            num_train_samples=NUM_TRAINING_SAMPLES_CIFAR10,
            replay_learning_rate=MAGIC_REPLAY_LEARNING_RATE
        )

        # 4. Save Scores (only if computed in this run)
        with open(scores_file_path, 'wb') as f:
            pickle.dump(magic_scores, f)
        print(f"MAGIC scores saved to {scores_file_path}")
    
    # 5. Visualization (run if full pipeline or if only plotting)
    # Ensure magic_scores is loaded if we are in run_plotting_only mode
    if magic_scores is None and run_plotting_only:
         # This case should have been handled by loading scores above.
         # If it occurs, it's an issue with the logic.
        raise RuntimeError("Attempting to plot, but magic_scores is None. Scores should have been loaded.")
    if magic_scores is None and not run_plotting_only and not run_full_pipeline:
        # This means scores file and plot file existed, scores loaded, and we returned early.
        # This block should not be reached if the top check worked.
        # For safety, if magic_scores is somehow None here, we can't plot.
        print("Skipping plotting as scores are not available (should have returned early if completed).")
        return None # Or return loaded scores if that was the path.

    if magic_scores is not None:
        print(f"Generating MAGIC influence plot to {plot_file_path}...")
        # For plotting, we need the raw image tensor from the dataset (usually unnormalized)
        viz_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # Ensure train_ds_viz is created using the correct root directory
        train_ds_viz = CustomCIFAR10Dataset(root=str(CIFAR_ROOT), train=True, download=True, transform=viz_transform)
        
        target_img_for_plot, target_lbl_for_plot, _ = target_image_data_tuple 

        plot_influence_images(
            scores_flat=magic_scores,
            target_image_info={'image': target_img_for_plot, 'label': target_lbl_for_plot, 'id_str': f"Val {MAGIC_TARGET_VAL_IMAGE_IDX}"},
            train_dataset_info={'dataset': train_ds_viz, 'name': 'CIFAR10 Train'},
            num_to_show=MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW,
            plot_title_prefix="MAGIC Influential Images",
            save_path=plot_file_path # Use the defined plot_file_path
        )
        print(f"MAGIC influence plot saved to {plot_file_path}")
    else:
        print("MAGIC scores not available, skipping visualization.")


    print("MAGIC analysis script finished.")
    return magic_scores

if __name__ == "__main__":
    from .config import ensure_output_dirs_exist
    ensure_output_dirs_exist() # Ensures output directories are created
    run_magic_analysis() 