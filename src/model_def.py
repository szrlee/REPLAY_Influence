import torch
import torch.nn as nn
from .config import NUM_CLASSES # Import NUM_CLASSES from config

# --- Model Definition (ResNet9) ---
class Mul(torch.nn.Module):
    """A layer that multiplies the input by a fixed scalar weight."""
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x):
        return x * self.weight

class Flatten(torch.nn.Module):
    """A layer that flattens the input tensor, preserving the batch dimension."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    """A standard residual block: output = input + module(input)."""
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

def _conv_bn(channels_in, channels_out, **kwargs):
    """Helper function to create a sequence of Conv2d -> BatchNorm2d -> ReLU."""
    # Bias is typically False in conv layers followed by BatchNorm
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, bias=False, **kwargs),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True)
    )

def construct_rn9(num_classes=NUM_CLASSES):
    """
    Constructs a ResNet9 model.
    This architecture is commonly used for CIFAR-10 image classification.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        torch.nn.Module: The ResNet9 model.
    """
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
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