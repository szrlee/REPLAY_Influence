import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import torchvision.transforms.functional as TF # For better image display
# from .data_handling import CustomCIFAR10Dataset # Removed unused and incorrect import

# Assuming config might be used for default plot save locations or styles
# from .config import MAGIC_PLOTS_DIR # Example if plots dir is needed here as a default

def plot_influence_images(scores_flat, target_image_info, train_dataset_info, 
                          num_to_show=5, plot_title_prefix="Influence Analysis", 
                          save_path=None):
    """
    Plots the most positively and negatively influential training images for a target image.

    Args:
        scores_flat (np.array): 1D array of influence scores for all training samples.
        target_image_info (dict): Contains {
            'image': torch.Tensor (C,H,W), a single target image tensor (typically unnormalized),
            'label': int, target image label,
            'id_str': str, identifier string for the target image (e.g., 'Val Idx 21')
        }
        train_dataset_info (dict): Contains {
            'dataset': torch.utils.data.Dataset (should return img, label, idx),
            'name': str, name of the training dataset (e.g., 'CIFAR10 Train')
        }
        num_to_show (int): Number of top positive and top negative images to display.
        plot_title_prefix (str): Prefix for the plot title.
        save_path (Path or str, optional): If provided, saves the plot to this path.
    """
    target_img_tensor = target_image_info['image']
    target_label = target_image_info['label']
    target_id_str = target_image_info['id_str']
    train_dataset = train_dataset_info['dataset']

    # Get top N positive and negative influential training images
    sorted_indices = np.argsort(scores_flat)
    most_helpful_indices = sorted_indices[-num_to_show:][::-1]  # Top N positive
    most_harmful_indices = sorted_indices[:num_to_show]       # Top N negative

    fig, axs = plt.subplots(2, num_to_show + 1, figsize=(3 * (num_to_show + 1), 6))
    fig.suptitle(f'{plot_title_prefix}: Target {target_id_str} (Label: {target_label})', fontsize=16)

    # Display target image
    axs[0, 0].imshow(TF.to_pil_image(target_img_tensor))
    axs[0, 0].set_title(f"Target Image\n{target_id_str} (L: {target_label})")
    axs[0, 0].axis('off')
    axs[1, 0].axis('off') # Empty space below target

    def display_train_images(indices_list, influence_type_str, row_idx):
        for i, train_idx in enumerate(indices_list):
            train_img_tensor, train_label, _ = train_dataset[train_idx]
            score = scores_flat[train_idx]
            ax = axs[row_idx, i + 1]
            ax.imshow(TF.to_pil_image(train_img_tensor))
            ax.set_title(f'{influence_type_str} #{i+1}\nIdx: {train_idx} (L: {train_label})\nScore: {score:.2e}')
            ax.axis('off')

    display_train_images(most_helpful_indices, "Helpful", 0)
    display_train_images(most_harmful_indices, "Harmful", 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        print(f"Saved influence plot to {save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory 