import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import NUM_CLASSES # Import NUM_CLASSES from config

# --- Model Definition (ResNet9) ---
class Mul(torch.nn.Module):
    """
    A layer that multiplies the input by a fixed scalar weight.
    
    This is commonly used as a scaling layer in ResNet architectures.
    
    Args:
        weight (float): The scalar weight to multiply inputs by.
    """
    
    def __init__(self, weight: float) -> None:
        super(Mul, self).__init__()
        self.weight = weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multiplication layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *).
            
        Returns:
            torch.Tensor: Input tensor multiplied by the fixed weight.
        """
        return x * self.weight


class Flatten(torch.nn.Module):
    """
    A layer that flattens the input tensor, preserving the batch dimension.
    
    Converts input of shape (batch_size, *) to (batch_size, flattened_features).
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the flatten layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *).
            
        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, flattened_features).
        """
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    """
    A standard residual block: output = input + module(input).
    
    Implements the skip connection architecture fundamental to ResNet models.
    
    Args:
        module (torch.nn.Module): The module to apply to the input before adding the skip connection.
    """
    
    def __init__(self, module: torch.nn.Module) -> None:
        super(Residual, self).__init__()
        self.module = module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: x + module(x)
        """
        return x + self.module(x)


def _conv_bn(channels_in: int, channels_out: int, **kwargs) -> torch.nn.Sequential:
    """
    Helper function to create a sequence of Conv2d -> BatchNorm2d -> ReLU.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
        **kwargs: Additional arguments passed to Conv2d.
        
    Returns:
        torch.nn.Sequential: The constructed conv-bn-relu block.
    """
    # Bias is typically False in conv layers followed by BatchNorm
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, bias=False, **kwargs),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True)
    )


def construct_rn9(num_classes: Optional[int] = None) -> torch.nn.Sequential:
    """
    Constructs a ResNet9 model optimized for CIFAR-10 image classification.
    
    This architecture follows the popular ResNet9 design with:
    - 9 layers total (6 conv + 3 special layers)
    - Residual connections for improved gradient flow
    - Adaptive pooling for flexible input sizes
    - BatchNorm for training stability
    
    Architecture:
        - Conv-BN-ReLU (3->64)
        - Conv-BN-ReLU (64->128) with stride 2
        - Residual block (128->128->128)
        - Conv-BN-ReLU (128->256)
        - MaxPool2d
        - Residual block (256->256->256)
        - Conv-BN-ReLU (256->128) without padding
        - AdaptiveMaxPool2d(1,1)
        - Flatten + Linear + Mul(0.2)
    
    Args:
        num_classes (Optional[int]): Number of output classes. 
                                   Defaults to NUM_CLASSES from config.
    
    Returns:
        torch.nn.Sequential: The complete ResNet9 model ready for training.
        
    Example:
        >>> model = construct_rn9(num_classes=10)
        >>> model = model.to(device)
        >>> output = model(torch.randn(32, 3, 32, 32))  # CIFAR-10 batch
        >>> print(output.shape)  # torch.Size([32, 10])
    """
    if num_classes is None:
        num_classes = NUM_CLASSES
    
    def conv_bn(channels_in: int, channels_out: int, kernel_size: int = 3, 
                stride: int = 1, padding: int = 1, groups: int = 1) -> torch.nn.Sequential:
        """Inner helper for consistent conv-bn-relu blocks."""
        return torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
                torch.nn.BatchNorm2d(channels_out),
                torch.nn.ReLU(inplace=True)
        )
    
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model 