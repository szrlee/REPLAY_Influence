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

# Project-specific imports
from . import config # For config.VARIABLE access
# Specific imports from .config are removed; all config access is via "config."

from .utils import set_seeds
from .model_def import construct_rn9
from .data_handling import (
    get_cifar10_dataloader, CustomDataset
)
from .visualization import plot_influence_images

class MagicAnalyzer:
    def __init__(self):
        self.model_for_training = None 
        self.batch_dict_for_replay = {}
        config.MAGIC_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        config.MAGIC_SCORES_DIR.mkdir(parents=True, exist_ok=True)
        config.MAGIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def train_and_collect_intermediate_states(self, force_retrain=False):
        print("Starting MAGIC model training and batch/checkpoint collection...")
        set_seeds(config.SEED)
        self.model_for_training = construct_rn9(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        optimizer = SGD(self.model_for_training.parameters(), 
                        lr=config.MAGIC_MODEL_TRAIN_LR, 
                        momentum=config.MAGIC_MODEL_TRAIN_MOMENTUM, 
                        weight_decay=config.MAGIC_MODEL_TRAIN_WEIGHT_DECAY)
        criterion = CrossEntropyLoss(label_smoothing=config.MAGIC_MODEL_LABEL_SMOOTHING)
        train_loader = get_cifar10_dataloader(
            batch_size=config.MAGIC_MODEL_TRAIN_BATCH_SIZE, 
            split='train', shuffle=True, augment=False,
            num_workers=config.DATALOADER_NUM_WORKERS, root_path=config.CIFAR_ROOT
        )
        if not force_retrain and config.BATCH_DICT_FILE.exists():
            print(f"Loading existing batch_dict from {config.BATCH_DICT_FILE}")
            with open(config.BATCH_DICT_FILE, 'rb') as f: self.batch_dict_for_replay = pickle.load(f)
            if not self.batch_dict_for_replay: 
                print("Loaded batch_dict is empty. Will retrain.")
                force_retrain = True 
            else:
                if self.batch_dict_for_replay:
                    total_completed_iterations = max(self.batch_dict_for_replay.keys())
                    final_iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_completed_iterations)
                    if final_iter_ckpt_path.exists():
                        print(f"Training seems complete. {total_completed_iterations} iterations found with batch_dict and final checkpoint. Skipping training.")
                        if hasattr(self, 'model_for_training') and self.model_for_training is not None:
                            self.model_for_training.load_state_dict(torch.load(final_iter_ckpt_path, map_location=config.DEVICE))
                        return total_completed_iterations
                    else:
                        print(f"Batch_dict exists, but final checkpoint sd_0_{total_completed_iterations}.pt missing. Retraining needed.")
                        force_retrain = True
                else:
                    print("Batch_dict exists but is empty. Retraining needed.")
                    force_retrain = True
        if force_retrain:
            print("Retraining: Clearing batch_dict_for_replay.")
            self.batch_dict_for_replay.clear()
        current_iteration_step = 0
        initial_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=current_iteration_step)
        torch.save(self.model_for_training.state_dict(), initial_ckpt_path)
        print(f"Saved initial checkpoint to {initial_ckpt_path} (state before any training)")
        self.model_for_training.train()
        for epoch in range(config.MAGIC_MODEL_TRAIN_EPOCHS):
            print(f"MAGIC Training: Epoch {epoch+1}/{config.MAGIC_MODEL_TRAIN_EPOCHS}")
            for batch_images, batch_labels, batch_indices in tqdm(train_loader, desc=f"MAGIC Epoch {epoch+1}"):
                current_iteration_step += 1
                self.batch_dict_for_replay[current_iteration_step] = {
                    'ims': batch_images.cpu().clone(), 'labs': batch_labels.cpu().clone(), 'idx': batch_indices.cpu().clone()}
                batch_images, batch_labels = batch_images.to(config.DEVICE), batch_labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = self.model_for_training(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                iter_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=current_iteration_step)
                torch.save(self.model_for_training.state_dict(), iter_ckpt_path)
            print(f"End of Epoch {epoch+1}. Total iterations completed: {current_iteration_step}. Last checkpoint: {iter_ckpt_path}")
        with open(config.BATCH_DICT_FILE, 'wb') as f: pickle.dump(self.batch_dict_for_replay, f)
        print(f"Saved batch_dict for replay to {config.BATCH_DICT_FILE}")
        print(f"MAGIC model training finished. {current_iteration_step} total iterations (batches processed).")
        return current_iteration_step

    def compute_influence_scores(self, total_training_iterations: int, force_recompute=False):
        scores_save_path = config.get_magic_scores_path(target_idx=config.MAGIC_TARGET_VAL_IMAGE_IDX)
        if not force_recompute and scores_save_path.exists():
            print(f"Loading existing MAGIC scores from {scores_save_path}")
            with open(scores_save_path, 'rb') as f: return pickle.load(f)
        if not self.batch_dict_for_replay:
            if config.BATCH_DICT_FILE.exists():
                print(f"Batch_dict is empty in memory, loading from {config.BATCH_DICT_FILE}")
                with open(config.BATCH_DICT_FILE, 'rb') as f: self.batch_dict_for_replay = pickle.load(f)
                if not self.batch_dict_for_replay: raise RuntimeError("Loaded batch_dict is empty. Cannot compute influence scores.")
            else: raise RuntimeError("batch_dict_for_replay not populated and no file found. Run training first.")
        print(f"Starting MAGIC influence score computation for target val_idx: {config.MAGIC_TARGET_VAL_IMAGE_IDX}")
        set_seeds(config.SEED)
        replay_model = construct_rn9(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        target_ds = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True, 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))]))
        target_im, target_lab, _ = target_ds[config.MAGIC_TARGET_VAL_IMAGE_IDX]
        target_im, target_lab = target_im.unsqueeze(0).to(config.DEVICE), torch.tensor([target_lab]).to(config.DEVICE)
        final_model_ckpt_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=total_training_iterations)
        if not final_model_ckpt_path.exists(): raise FileNotFoundError(f"Final model checkpoint {final_model_ckpt_path} not found.")
        temp_model_for_target_grad = construct_rn9(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        temp_model_for_target_grad.load_state_dict(torch.load(final_model_ckpt_path, map_location=config.DEVICE))
        temp_model_for_target_grad.eval(); temp_model_for_target_grad.zero_grad()
        out_target = temp_model_for_target_grad(target_im)
        criterion_target = CrossEntropyLoss()
        loss_target = criterion_target(out_target, target_lab); loss_target.backward()
        delta_k_plus_1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in temp_model_for_target_grad.parameters()]
        del temp_model_for_target_grad
        beta_t_list = []
        criterion_replay_no_reduction = CrossEntropyLoss(reduction='none')
        for replay_step_idx in tqdm(range(total_training_iterations - 1, -1, -1), desc="MAGIC Replay Pass"):
            s_k_checkpoint_path = config.get_magic_checkpoint_path(model_id=0, step_or_epoch=replay_step_idx)
            if not s_k_checkpoint_path.exists():
                placeholder_beta_t = np.zeros(config.NUM_TRAIN_SAMPLES)
                if beta_t_list and isinstance(beta_t_list[-1], np.ndarray) : placeholder_beta_t = np.zeros_like(beta_t_list[-1])
                beta_t_list.append(placeholder_beta_t); continue
            replay_model.load_state_dict(torch.load(s_k_checkpoint_path, map_location=config.DEVICE)); replay_model.train()
            current_sk_params = [p for p in replay_model.parameters()]; [p.requires_grad_(True) for p in current_sk_params]; replay_model.zero_grad()
            batch_data = self.batch_dict_for_replay[replay_step_idx + 1]
            b_ims, b_labs, b_idx = batch_data['ims'].to(config.DEVICE), batch_data['labs'].to(config.DEVICE), batch_data['idx']
            data_weights = torch.nn.Parameter(torch.ones(config.NUM_TRAIN_SAMPLES, device=config.DEVICE), requires_grad=True)
            train_output_sk = replay_model(b_ims)
            loss_samples_sk = criterion_replay_no_reduction(train_output_sk, b_labs)
            weighted_loss_sk = (loss_samples_sk * data_weights[b_idx.to(config.DEVICE)]).mean()
            grad_L_wrt_sk_tuple = torch.autograd.grad(weighted_loss_sk, current_sk_params, create_graph=True, allow_unused=True)
            grad_L_wrt_sk = [g.clone() if g is not None else torch.zeros_like(p) for g,p in zip(grad_L_wrt_sk_tuple, current_sk_params)]
            sk_plus_1_approx_params = [sk_p - config.MAGIC_REPLAY_LEARNING_RATE * gk_p for sk_p, gk_p in zip(current_sk_params, grad_L_wrt_sk)]
            q_k_scalar = sum((p_sk_plus_1 * p_delta).sum() for p_sk_plus_1, p_delta in zip(sk_plus_1_approx_params, delta_k_plus_1))
            if data_weights.grad is not None: data_weights.grad.detach_().zero_()
            grads_q_wrt_w = torch.autograd.grad(q_k_scalar, data_weights, retain_graph=True, allow_unused=False)
            beta_k_for_step = grads_q_wrt_w[0]
            beta_t_list.append(beta_k_for_step.detach().cpu().numpy())
            delta_k_grads_tuple = torch.autograd.grad(q_k_scalar, current_sk_params, allow_unused=True)
            delta_k_plus_1 = [g.clone() if g is not None else torch.zeros_like(p) for g,p in zip(delta_k_grads_tuple, current_sk_params)]
        scores_per_step = np.stack(beta_t_list[::-1])
        with open(scores_save_path, 'wb') as f: pickle.dump(scores_per_step, f)
        print(f"Saved per-step MAGIC influence scores to {scores_save_path}")
        return scores_per_step

    def plot_magic_influences(self, per_step_scores_or_path, num_images_to_show=None):
        if num_images_to_show is None: num_images_to_show = config.MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW
        if isinstance(per_step_scores_or_path, (str, Path)):
            scores_path = Path(per_step_scores_or_path)
            if not scores_path.exists(): raise FileNotFoundError(f"Scores file not found: {scores_path}")
            with open(scores_path, 'rb') as f: per_step_scores = pickle.load(f)
        else: per_step_scores = per_step_scores_or_path
        if per_step_scores.ndim != 2 or per_step_scores.shape[1] != config.NUM_TRAIN_SAMPLES:
            raise ValueError(f"Expected per-step scores [steps, samples], got {per_step_scores.shape}")
        scores_flat = per_step_scores.sum(axis=0)
        print(f"Preparing to plot influential images for target {config.MAGIC_TARGET_VAL_IMAGE_IDX}...") # Added print statement
        viz_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_ds_viz = CustomDataset(root=config.CIFAR_ROOT, train=True, download=True, transform=viz_transform)
        val_ds_viz = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True, transform=viz_transform)
        target_img_tensor, target_label, _ = val_ds_viz[config.MAGIC_TARGET_VAL_IMAGE_IDX]
        target_info = {'image': target_img_tensor, 'label': target_label, 'id_str': f"Val Idx {config.MAGIC_TARGET_VAL_IMAGE_IDX}"}
        train_info = {'dataset': train_ds_viz, 'name': 'CIFAR10 Train'}
        plot_save_path = config.MAGIC_PLOTS_DIR / f"magic_influence_val_{config.MAGIC_TARGET_VAL_IMAGE_IDX}.png"
        plot_influence_images(scores_flat=scores_flat, target_image_info=target_info, train_dataset_info=train_info,
            num_to_show=num_images_to_show, plot_title_prefix="MAGIC Analysis", save_path=plot_save_path)
        print(f"Influence plot saved to {plot_save_path}")

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