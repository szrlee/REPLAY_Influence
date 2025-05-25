"""
Visualization Module for REPLAY Influence Analysis
=================================================

This module provides functions for visualizing influence analysis results,
including plotting the most influential training images for target validation images.

Python >=3.8 Compatible
"""

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import torchvision.transforms.functional as TF
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import warnings
# from .data_handling import CustomCIFAR10Dataset # Removed unused and incorrect import

# Assuming config might be used for default plot save locations or styles
# from .config import MAGIC_PLOTS_DIR # Example if plots dir is needed here as a default

logger = logging.getLogger('influence_analysis.visualization')

def plot_influence_images(
    scores_flat: np.ndarray, 
    target_image_info: Dict[str, Any], 
    train_dataset_info: Dict[str, Any], 
    num_to_show: int = 5, 
    plot_title_prefix: str = "Influence Analysis", 
    save_path: Optional[Union[Path, str]] = None
) -> None:
    """
    Plots the most positively and negatively influential training images for a target image.

    Args:
        scores_flat (np.ndarray): 1D array of influence scores for all training samples.
        target_image_info (Dict[str, Any]): Contains {
            'image': torch.Tensor (C,H,W), a single target image tensor (typically unnormalized),
            'label': int, target image label,
            'id_str': str, identifier string for the target image (e.g., 'Val Idx 21')
        }
        train_dataset_info (Dict[str, Any]): Contains {
            'dataset': torch.utils.data.Dataset (should return img, label, idx),
            'name': str, name of the training dataset (e.g., 'CIFAR10 Train')
        }
        num_to_show (int): Number of top positive and top negative images to display.
        plot_title_prefix (str): Prefix for the plot title.
        save_path (Optional[Union[Path, str]]): If provided, saves the plot to this path.
        
    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If plotting fails due to data issues.
    """
    # Comprehensive input validation
    if not isinstance(scores_flat, np.ndarray):
        raise ValueError(f"scores_flat must be a numpy array, got {type(scores_flat)}")
    
    if scores_flat.ndim != 1:
        raise ValueError(f"scores_flat must be a 1D array, got shape {scores_flat.shape}")
    
    if num_to_show <= 0:
        raise ValueError(f"num_to_show must be positive, got {num_to_show}")
    
    # Validate target_image_info structure
    required_target_keys = {'image', 'label', 'id_str'}
    if not isinstance(target_image_info, dict):
        raise ValueError("target_image_info must be a dictionary")
    
    missing_keys = required_target_keys - set(target_image_info.keys())
    if missing_keys:
        raise ValueError(f"target_image_info missing required keys: {missing_keys}")
    
    # Validate train_dataset_info structure
    required_train_keys = {'dataset', 'name'}
    if not isinstance(train_dataset_info, dict):
        raise ValueError("train_dataset_info must be a dictionary")
    
    missing_keys = required_train_keys - set(train_dataset_info.keys())
    if missing_keys:
        raise ValueError(f"train_dataset_info missing required keys: {missing_keys}")
    
    train_dataset = train_dataset_info['dataset']
    
    try:
        dataset_size = len(train_dataset)
    except Exception as e:
        raise ValueError(f"Cannot determine dataset size: {e}") from e
    
    if len(scores_flat) != dataset_size:
        raise ValueError(f"scores_flat length ({len(scores_flat)}) must match dataset size ({dataset_size})")
    
    if num_to_show > dataset_size:
        logger.warning(f"num_to_show ({num_to_show}) > dataset size ({dataset_size}). Using dataset_size.")
        num_to_show = dataset_size
    
    # Extract target image information
    target_img_tensor = target_image_info['image']
    target_label = target_image_info['label']
    target_id_str = target_image_info['id_str']
    
    # Validate target image tensor
    if not isinstance(target_img_tensor, torch.Tensor):
        raise ValueError(f"Target image must be a torch.Tensor, got {type(target_img_tensor)}")
    
    if target_img_tensor.ndim != 3:
        raise ValueError(f"Target image must have 3 dimensions (C,H,W), got shape {target_img_tensor.shape}")

    # Get top N positive and negative influential training images
    try:
        sorted_indices = np.argsort(scores_flat)
        most_harmful_indices = sorted_indices[-num_to_show:][::-1]  # Top N positive (most harmful)
        most_helpful_indices = sorted_indices[:num_to_show]        # Top N negative (most helpful)
    except Exception as e:
        raise RuntimeError(f"Failed to sort influence scores: {e}") from e

    # Create the plot with error handling
    try:
        fig, axs = plt.subplots(2, num_to_show + 1, figsize=(3 * (num_to_show + 1), 6))
        
        # Handle case where num_to_show == 1 (axs might not be 2D)
        if num_to_show == 1:
            axs = axs.reshape(2, -1)
        
        fig.suptitle(f'{plot_title_prefix}: Target {target_id_str} (Label: {target_label})', fontsize=16)

        # Display target image with error handling
        try:
            axs[0, 0].imshow(TF.to_pil_image(target_img_tensor))
            axs[0, 0].set_title(f"Target Image\n{target_id_str} (L: {target_label})")
            axs[0, 0].axis('off')
            axs[1, 0].axis('off')  # Empty space below target
        except Exception as e:
            logger.error(f"Failed to display target image: {e}")
            axs[0, 0].text(0.5, 0.5, f'Target Image\nError: {str(e)[:20]}...', 
                          ha='center', va='center', transform=axs[0, 0].transAxes)
            axs[0, 0].axis('off')
            axs[1, 0].axis('off')

        def display_train_images(indices_list: np.ndarray, influence_type_str: str, row_idx: int) -> None:
            """Helper function to display training images with comprehensive error handling."""
            for i, train_idx in enumerate(indices_list):
                # Validate index is within bounds
                if train_idx < 0 or train_idx >= dataset_size:
                    logger.error(f"Train index {train_idx} out of bounds for dataset of size {dataset_size}")
                    continue
                    
                try:
                    train_img_tensor, train_label, _ = train_dataset[train_idx]
                    score = scores_flat[train_idx]
                    ax = axs[row_idx, i + 1]
                    
                    # Convert tensor to PIL image with error handling
                    try:
                        pil_image = TF.to_pil_image(train_img_tensor)
                        ax.imshow(pil_image)
                    except Exception as img_e:
                        logger.warning(f"Failed to convert image {train_idx} to PIL: {img_e}")
                        # Create a placeholder image
                        ax.text(0.5, 0.5, f'Image {train_idx}\nConversion Error', 
                               ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'{influence_type_str} #{i+1}\nIdx: {train_idx} (L: {train_label})\nScore: {score:.2e}')
                    ax.axis('off')
                    
                except Exception as e:
                    logger.error(f"Error displaying train image {train_idx}: {e}")
                    # Create empty plot for failed image
                    ax = axs[row_idx, i + 1]
                    ax.text(0.5, 0.5, f'Error\nIdx: {train_idx}\n{str(e)[:20]}...', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')

        # Display helpful and harmful images
        display_train_images(most_helpful_indices, "Helpful", 0)
        display_train_images(most_harmful_indices, "Harmful", 1)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
        
        # Save or show the plot
        if save_path:
            save_path = Path(save_path)
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved influence plot to {save_path}")
            except Exception as save_e:
                logger.error(f"Failed to save plot to {save_path}: {save_e}")
                # Show the plot instead
                plt.show()
        else:
            plt.show()
            
    except Exception as plot_e:
        logger.error(f"Failed to create influence plot: {plot_e}")
        raise RuntimeError(f"Plotting failed: {plot_e}") from plot_e
    finally:
        # Always close the figure to free memory
        if 'fig' in locals():
            plt.close(fig)

def create_correlation_plot(
    predicted_losses: np.ndarray,
    actual_losses: np.ndarray,
    correlation_coefficient: float,
    title: str = "LDS Correlation Analysis",
    save_path: Optional[Union[Path, str]] = None
) -> None:
    """
    Creates a correlation plot for LDS validation results.
    
    Args:
        predicted_losses (np.ndarray): Predicted losses from influence scores.
        actual_losses (np.ndarray): Actual losses from subset-trained models.
        correlation_coefficient (float): Correlation coefficient between predicted and actual.
        title (str): Plot title.
        save_path (Optional[Union[Path, str]]): Path to save the plot.
        
    Raises:
        ValueError: If input arrays are incompatible.
        RuntimeError: If plotting fails.
    """
    # Input validation
    if not isinstance(predicted_losses, np.ndarray) or not isinstance(actual_losses, np.ndarray):
        raise ValueError("Both predicted_losses and actual_losses must be numpy arrays")
    
    if predicted_losses.shape != actual_losses.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_losses.shape} vs actual {actual_losses.shape}")
    
    if len(predicted_losses) == 0:
        raise ValueError("Cannot create correlation plot with empty arrays")
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create scatter plot
        ax.scatter(predicted_losses, actual_losses, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(predicted_losses, actual_losses, 1)
        p = np.poly1d(z)
        ax.plot(predicted_losses, p(predicted_losses), "r--", alpha=0.8, linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Predicted Loss (from Influence Scores)')
        ax.set_ylabel('Actual Loss (from Subset Models)')
        ax.set_title(f'{title}\nCorrelation: {correlation_coefficient:.4f}')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Add correlation text box
        textstr = f'Correlation: {correlation_coefficient:.4f}\nSamples: {len(predicted_losses)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            save_path = Path(save_path)
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved correlation plot to {save_path}")
            except Exception as save_e:
                logger.error(f"Failed to save correlation plot to {save_path}: {save_e}")
                plt.show()
        else:
            plt.show()
            
    except Exception as plot_e:
        logger.error(f"Failed to create correlation plot: {plot_e}")
        raise RuntimeError(f"Correlation plotting failed: {plot_e}") from plot_e
    finally:
        if 'fig' in locals():
            plt.close(fig) 