import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from .data_handling import CustomCIFAR10Dataset # Use relative import

def plot_influence_images(scores_flat, target_image_info, train_dataset_info,
                          num_to_show, plot_title_prefix="Influential Training Images",
                          save_path=None):
    """
    Plots a target image alongside its most and least influential training images.

    Args:
        scores_flat (np.array): Flat array of influence scores for all training samples.
        target_image_info (dict): Dict with {'image': tensor, 'label': int, 'id_str': str}.
        train_dataset_info (dict): Dict with {'dataset': CustomCIFAR10Dataset, 'name': str}.
        num_to_show (int): Number of top/bottom influential images to display.
        plot_title_prefix (str): Prefix for the main plot title.
        save_path (Path, optional): Path to save the plot. If None, plot is only shown.
    """
    print("Plotting influence results...")

    target_image = target_image_info['image']
    target_label = target_image_info['label']
    target_id_str = target_image_info['id_str']

    train_dataset = train_dataset_info['dataset']

    fig, axs = plt.subplots(nrows=2, ncols=num_to_show + 2,
                            figsize=(2.5 * (num_to_show + 2), 6)) # Adjusted figsize for titles
    fig.suptitle(f'{plot_title_prefix} for {target_id_str} (Label: {target_label})', fontsize=16, y=0.98)

    def display_img(ax, img_tensor, title, is_target_image=False):
        # Target image is assumed to be normalized, training images from train_ds_viz are ToTensor output (0-1 range)
        if is_target_image: # Unnormalize if it's the target image
            # Crude unnormalization for viz - specific to CIFAR10
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            if img_tensor.device != mean.device:
                 mean = mean.to(img_tensor.device)
                 std = std.to(img_tensor.device)
            img_tensor_display = img_tensor * std + mean
            img_tensor_display = torch.clamp(img_tensor_display, 0, 1)
        else: # Training images from train_ds_viz are assumed to be ToTensor output (0-1 range)
            img_tensor_display = img_tensor # Already in [0,1] range from ToTensor

        ax.imshow(img_tensor_display.permute(1, 2, 0).cpu().numpy()) # CHW to HWC, ensure CPU for imshow
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # Display target image in both rows for reference
    display_img(axs[0, 0], target_image, f"Target {target_id_str}\nLabel: {target_label}", is_target_image=True)
    axs[0, 1].axis('off') # Spacer
    display_img(axs[1, 0], target_image, f"Target {target_id_str}\nLabel: {target_label}", is_target_image=True)
    axs[1, 1].axis('off') # Spacer

    # Most positively influential (proponents)
    proponent_indices = np.argsort(scores_flat)[::-1][:num_to_show]
    axs[0, 1].set_title("Most Positively Influential (Proponents)", loc='center', fontsize=10, y=1.1, x= (num_to_show/2))

    for i, train_idx in enumerate(proponent_indices):
        img, lbl, _ = train_dataset[train_idx]
        display_img(axs[0, i + 2], img, f"Train {train_idx} (L:{lbl})\nScore: {scores_flat[train_idx]:.2e}")

    # Most negatively influential (opponents)
    opponent_indices = np.argsort(scores_flat)[:num_to_show]
    axs[1, 1].set_title("Most Negatively Influential (Opponents)", loc='center', fontsize=10, y=1.1, x= (num_to_show/2))

    for i, train_idx in enumerate(opponent_indices):
        img, lbl, _ = train_dataset[train_idx]
        display_img(axs[1, i + 2], img, f"Train {train_idx} (L:{lbl})\nScore: {scores_flat[train_idx]:.2e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to make space for suptitle and row titles
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Influence plot saved to {save_path}")
    plt.show() 