import torch

# --- Model Definition (ResNet9) ---
class Mul(torch.nn.Module):
    """A layer that multiplies the input by a fixed scalar weight."""
    def __init__(self, weight):
        super().__init__()
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
        super().__init__()
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

def construct_resnet9(num_classes=10):
    """
    Constructs a ResNet9 model.
    This architecture is commonly used for CIFAR-10 image classification.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        torch.nn.Module: The ResNet9 model.
    """
    return torch.nn.Sequential(
        _conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        _conv_bn(64, 128, kernel_size=5, stride=2, padding=2), # Downsamples: 32x32 -> 16x16
        Residual(torch.nn.Sequential(
            _conv_bn(128, 128, kernel_size=3, stride=1, padding=1),
            _conv_bn(128, 128, kernel_size=3, stride=1, padding=1)
        )),
        _conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2), # Downsamples: 16x16 -> 8x8
        Residual(torch.nn.Sequential(
            _conv_bn(256, 256, kernel_size=3, stride=1, padding=1),
            _conv_bn(256, 256, kernel_size=3, stride=1, padding=1)
        )),
        _conv_bn(256, 128, kernel_size=3, stride=1, padding=0), # Output: 8x8 -> 6x6 (due to padding=0)
        torch.nn.AdaptiveMaxPool2d((1, 1)), # Global Average Pooling: 6x6 -> 1x1
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2) # Output scaling factor
    ) 