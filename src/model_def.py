import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

from . import config # Import the config module to access defaults

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


# === UNIFIED LOGSUMEXP POOLING IMPLEMENTATION ===
# Consolidating multiple LogSumExp implementations for better maintainability

class LogSumExpPool2d(torch.nn.Module):
    """
    Unified LogSumExp pooling implementation supporting both global and sliding window pooling.
    
    This consolidates the previous separate implementations (LogSumExpPool2dTableArch, 
    LogSumExpPool2dAirbench) into a single, robust implementation.
    
    Args:
        epsilon: Temperature parameter for LogSumExp (0 = average pooling)
        kernel_size: Pooling kernel size (-1 for global pooling)
        stride: Stride for sliding window (defaults to kernel_size)
        padding: Padding for sliding window (default: 0)
    """
    def __init__(self, epsilon: float, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super(LogSumExpPool2d, self).__init__()
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.global_pool = (kernel_size == -1)
        
        # Validation
        if not self.global_pool and (self.kernel_size <= 0 or self.stride <= 0):
            raise ValueError(f"Invalid kernel_size ({self.kernel_size}) or stride ({self.stride}) for sliding window pooling")
        
        if self.epsilon < 0:
            warnings.warn(f"Negative epsilon ({self.epsilon}) may cause numerical instability")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool:
            return self._global_logsumexp_pool(x)
        else:
            return self._sliding_logsumexp_pool(x)
    
    def _global_logsumexp_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Global LogSumExp pooling over spatial dimensions."""
        if self.epsilon == 0:
            return torch.mean(x, dim=(-1, -2), keepdim=True)
        
        # Numerically stable LogSumExp: log(mean(exp(epsilon * x)))
        # For numerical stability, we could implement log-sum-exp trick here
        max_val = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)[0].unsqueeze(-1)
        stable_exp = torch.exp(self.epsilon * (x - max_val))
        mean_exp = torch.mean(stable_exp, dim=(-1, -2), keepdim=True)
        
        result = max_val + (1.0 / self.epsilon) * torch.log(mean_exp + 1e-9)
        return result
    
    def _sliding_logsumexp_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window LogSumExp pooling using unfold."""
        N, C, H, W = x.shape
        
        # Apply padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Calculate output dimensions
        H_padded, W_padded = x.shape[-2:]
        Ho = (H_padded - self.kernel_size) // self.stride + 1
        Wo = (W_padded - self.kernel_size) // self.stride + 1
        
        # Extract sliding windows using unfold
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(N, C, Ho, Wo, -1)
        
        if self.epsilon == 0:
            return torch.mean(patches, dim=-1)
        
        # Apply LogSumExp with numerical stability
        max_patches = torch.max(patches, dim=-1, keepdim=True)[0]
        stable_exp = torch.exp(self.epsilon * (patches - max_patches))
        mean_exp = torch.mean(stable_exp, dim=-1, keepdim=True)
        
        result = max_patches.squeeze(-1) + (1.0 / self.epsilon) * torch.log(mean_exp.squeeze(-1) + 1e-9)
        return result


def _conv_bn(channels_in: int, channels_out: int, bias_scale: float = 1.0, **kwargs) -> torch.nn.Sequential:
    """
    Helper function to create a sequence of Conv2d -> BatchNorm2d -> ReLU.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
        bias_scale (float): Scale factor for bias initialization.
        **kwargs: Additional arguments passed to Conv2d.
        
    Returns:
        torch.nn.Sequential: The constructed conv-bn-relu block.
    """
    conv = torch.nn.Conv2d(channels_in, channels_out, bias=True, **kwargs)
    
    # Apply bias scaling as specified in the paper
    if conv.bias is not None:
        torch.nn.init.constant_(conv.bias, 0.0)
        conv.bias.data *= bias_scale
    
    return torch.nn.Sequential(
        conv,
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True)
    )


def construct_rn9(num_classes: int) -> torch.nn.Sequential:
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



def construct_resnet9_paper(num_classes: int) -> torch.nn.Sequential:
    """
    Constructs a ResNet-9 model according to the exact specifications from [Jor24a].
    
    This function implements the ResNet-9 architecture with all the specific
    hyperparameters from Table 1 of the paper:
    - Width multiplier: 2.5
    - Bias scale: 8.0
    - Final layer scale: 0.04
    - Log-sum-exp pooling with ε = 0.1
    
    This is the recommended function to use for reproducing paper results.
    
    Args:
        num_classes (Optional[int]): Number of output classes. 
                                   Defaults to NUM_CLASSES from config (10 for CIFAR-10).
    
    Returns:
        torch.nn.Sequential: The complete ResNet9 model ready for training.
        
    Example:
        >>> model = construct_resnet9_paper(num_classes=10)
        >>> model = model.to(device)
        >>> output = model(torch.randn(32, 3, 32, 32))  # CIFAR-10 batch
        >>> print(output.shape)  # torch.Size([32, 10])
    """
    # Paper specifications
    width_multiplier = 2.5
    bias_scale = 8.0
    final_layer_scale = 0.04
    pooling_epsilon = 0.1
    
    # Calculate channel sizes with width multiplier
    def channels(base: int) -> int:
        return int(base * width_multiplier)
    
    def conv_bn(channels_in: int, channels_out: int, kernel_size: int = 3, 
                stride: int = 1, padding: int = 1, groups: int = 1) -> torch.nn.Sequential:
        """Inner helper for consistent conv-bn-relu blocks."""
        return _conv_bn(channels_in, channels_out, bias_scale=bias_scale,
                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
    
    model = torch.nn.Sequential(
        # Initial conv layer
        conv_bn(3, channels(64), kernel_size=3, stride=1, padding=1),
        
        # First block with downsampling
        conv_bn(channels(64), channels(128), kernel_size=5, stride=2, padding=2),
        
        # First residual block
        Residual(torch.nn.Sequential(
            conv_bn(channels(128), channels(128)), 
            conv_bn(channels(128), channels(128))
        )),
        
        # Second block
        conv_bn(channels(128), channels(256), kernel_size=3, stride=1, padding=1),
        
        # Log-sum-exp pooling instead of max pooling
        LogSumExpPool2d(kernel_size=2, stride=2, epsilon=pooling_epsilon),
        
        # Second residual block
        Residual(torch.nn.Sequential(
            conv_bn(channels(256), channels(256)), 
            conv_bn(channels(256), channels(256))
        )),
        
        # Final conv layer
        conv_bn(channels(256), channels(128), kernel_size=3, stride=1, padding=0),
        
        # Global average pooling
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        
        # Flatten and final linear layer
        Flatten(),
        torch.nn.Linear(channels(128), num_classes, bias=False),
        
        # Final layer scaling
        Mul(final_layer_scale)
    )
    
    return model 


# --- New Network Definition from User Snippet (Table Architecture) ---
# UPDATED: Using unified LogSumExpPool2d implementation

# Basic Residual Block (from snippet)
class BasicBlockTableArch(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlockTableArch, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Final Layer Scale Wrapper (from snippet)
class ScaledLinearTableArch(torch.nn.Module):
    def __init__(self, linear_layer: torch.nn.Module, scale_factor: float):
        super(ScaledLinearTableArch, self).__init__()
        self.linear = linear_layer
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale_factor

# ResNet9 from snippet - UPDATED to use unified LogSumExpPool2d
class ResNet9TableArch(torch.nn.Module):
    def __init__(self, num_classes: Optional[int] = None, 
                 width_multiplier: Optional[float] = None, 
                 pooling_epsilon: Optional[float] = None, 
                 final_layer_scale: Optional[float] = None):
        super(ResNet9TableArch, self).__init__()
        
        # Resolve defaults from config if not provided
        if num_classes is None:
            num_classes = config.NUM_CLASSES
        if width_multiplier is None:
            width_multiplier = config.RESNET9_WIDTH_MULTIPLIER
        if pooling_epsilon is None:
            pooling_epsilon = config.RESNET9_POOLING_EPSILON
        if final_layer_scale is None:
            final_layer_scale = config.RESNET9_FINAL_LAYER_SCALE

        # Base channels for each stage, scaled by width_multiplier
        c = [int(ch * width_multiplier) for ch in [64, 64, 128, 256, 512]]

        self.conv1 = torch.nn.Conv2d(3, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(c[0])
        
        self.layer1 = self._make_layer(c[0], c[1], 1, stride=1) # 1 block
        self.layer2 = self._make_layer(c[1], c[2], 1, stride=2) # 1 block
        self.layer3 = self._make_layer(c[2], c[3], 1, stride=2) # 1 block
        self.layer4 = self._make_layer(c[3], c[4], 1, stride=2) # 1 block
        
        # UPDATED: Using unified LogSumExp pooling (global pooling with kernel_size=-1)
        self.pool = LogSumExpPool2d(epsilon=pooling_epsilon, kernel_size=-1) 
        
        self.linear_original = torch.nn.Linear(c[4], num_classes)
        
        if final_layer_scale != 1.0:
            self.linear = ScaledLinearTableArch(self.linear_original, final_layer_scale)
        else:
            self.linear = self.linear_original

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> torch.nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        current_in_channels = in_channels
        for s_val in strides: # Renamed s to s_val to avoid conflict if nn.Sequential is imported as s
            layers.append(BasicBlockTableArch(current_in_channels, out_channels, s_val))
            current_in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.linear(out)
        return out


# --- Network Definition for airbench94_adapted ---
# UPDATED: Using unified LogSumExpPool2d implementation

def make_airbench94_adapted(
    num_classes: Optional[int] = None,
    width_multiplier: Optional[float] = None,
    pooling_epsilon: Optional[float] = None,
    final_layer_scale: Optional[float] = None,
):
    # Resolve defaults from config if not provided
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if width_multiplier is None:
        width_multiplier = config.RESNET9_WIDTH_MULTIPLIER
    if pooling_epsilon is None:
        pooling_epsilon = config.RESNET9_POOLING_EPSILON
    if final_layer_scale is None:
        final_layer_scale = config.RESNET9_FINAL_LAYER_SCALE

    base_ch_initial = 24
    base_ch_block1 = 64
    base_ch_block2_3_linear = 256

    wm_ch_initial = int(base_ch_initial * width_multiplier)
    wm_ch_block1 = int(base_ch_block1 * width_multiplier)
    wm_ch_block2_3_linear = int(base_ch_block2_3_linear * width_multiplier)

    wm_ch_initial = max(1, wm_ch_initial)
    wm_ch_block1 = max(1, wm_ch_block1)
    wm_ch_block2_3_linear = max(1, wm_ch_block2_3_linear)
    
    act = lambda: torch.nn.GELU()
    bn = lambda ch: torch.nn.BatchNorm2d(ch) 
    
    def conv(ch_in, ch_out):
        return torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, padding='same', bias=False)

    # UPDATED: Pooling layer uses unified LogSumExpPool2d
    def create_pool_layer(kernel_size, stride):
        return LogSumExpPool2d(epsilon=pooling_epsilon, kernel_size=kernel_size, stride=stride)

    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, wm_ch_initial, kernel_size=2, padding=0, bias=True),
        act(),
        torch.nn.Sequential( # Block 1
            conv(wm_ch_initial, wm_ch_block1),
            create_pool_layer(kernel_size=2, stride=2),
            bn(wm_ch_block1), act(),
            conv(wm_ch_block1, wm_ch_block1),
            bn(wm_ch_block1), act(),
        ),
        torch.nn.Sequential( # Block 2
            conv(wm_ch_block1, wm_ch_block2_3_linear),
            create_pool_layer(kernel_size=2, stride=2),
            bn(wm_ch_block2_3_linear), act(),
            conv(wm_ch_block2_3_linear, wm_ch_block2_3_linear),
            bn(wm_ch_block2_3_linear), act(),
        ),
        torch.nn.Sequential( # Block 3
            conv(wm_ch_block2_3_linear, wm_ch_block2_3_linear),
            create_pool_layer(kernel_size=2, stride=2),
            bn(wm_ch_block2_3_linear), act(),
            conv(wm_ch_block2_3_linear, wm_ch_block2_3_linear),
            bn(wm_ch_block2_3_linear), act(),
        ),
        create_pool_layer(kernel_size=3, stride=3), # Final Pooling
        Flatten(), # Assumes Flatten class is defined above or imported
        torch.nn.Linear(wm_ch_block2_3_linear, num_classes, bias=False), 
        Mul(final_layer_scale), # Assumes Mul class is defined above or imported
    )
    return net 


 