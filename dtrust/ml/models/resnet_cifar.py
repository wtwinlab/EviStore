"""
ResNet for CIFAR-10/100

ResNet-20 architecture designed for CIFAR datasets (32x32 images).
This matches the model used in FLTrust for fair comparison.

Reference: Deep Residual Learning for Image Recognition (He et al., 2016)
"""

from typing import Dict, Any, List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMLModel


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _norm2d(planes: int, norm: str = "bn") -> nn.Module:
    """Normalization layer. GroupNorm has NO running buffers (scale/shift are params),
    so it is compatible with federated parameters()-only delta aggregation, unlike
    BatchNorm whose running_mean/var are buffers that this harness never aggregates."""
    if norm == "gn":
        return nn.GroupNorm(num_groups=min(8, planes), num_channels=planes)
    return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, norm: str = "bn"):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = _norm2d(planes, norm)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2 = _norm2d(planes, norm)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _norm2d(planes, norm),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(BaseMLModel):
    """
    ResNet architecture for CIFAR-10/100.

    This is a smaller ResNet variant designed for 32x32 images,
    following the architecture from "Deep Residual Learning for Image Recognition".

    Default configuration is ResNet-20 with 3 stages × 3 blocks each.
    """

    def __init__(self,
                 num_blocks: List[int] = None,
                 num_classes: int = 10,
                 **kwargs):
        """
        Initialize ResNetCIFAR.

        Args:
            num_blocks: Number of blocks in each of the 3 stages. Default [3,3,3] for ResNet-20.
            num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name="ResNetCIFAR",
            num_blocks=num_blocks,
            num_classes=num_classes,
            **kwargs
        )

        if num_blocks is None:
            num_blocks = [3, 3, 3]  # ResNet-20

        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.in_planes = 16
        self.norm = kwargs.get("norm", "bn")

        # Initial convolution
        self.conv1 = _conv3x3(3, 16, 1)
        self.bn1 = _norm2d(16, self.norm)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks[2], stride=2)

        # Classifier
        self.linear = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()
        self._is_initialized = True

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, norm=getattr(self, "norm", "bn")))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        total_blocks = sum(self.num_blocks)
        # ResNet naming: 2*total_blocks + 2 (initial conv + final linear)
        resnet_depth = 2 * total_blocks + 2

        return {
            'model_name': self.model_name,
            'model_type': 'resnet',
            'architecture': f'resnet{resnet_depth}_cifar',
            'num_blocks': self.num_blocks,
            'num_classes': self.num_classes,
            'parameter_count': self.get_parameter_count(),
            'model_size_mb': self.get_model_size_mb(),
            'input_size': (3, 32, 32),
            'layers': [
                {'type': 'Conv2d', 'in': 3, 'out': 16, 'kernel': 3},
                {'type': 'BatchNorm2d', 'features': 16},
                {'type': 'ReLU'},
                {'type': 'ResidualLayer1', 'blocks': self.num_blocks[0], 'channels': 16},
                {'type': 'ResidualLayer2', 'blocks': self.num_blocks[1], 'channels': 32},
                {'type': 'ResidualLayer3', 'blocks': self.num_blocks[2], 'channels': 64},
                {'type': 'AdaptiveAvgPool2d', 'output_size': 1},
                {'type': 'Linear', 'in': 64, 'out': self.num_classes}
            ]
        }

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet20_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Create ResNet-20 for CIFAR.

    ResNet-20: 3 stages × 3 blocks = 9 blocks total
    Total layers: 2*9 + 2 = 20

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-20 model
    """
    return ResNetCIFAR(num_blocks=[3, 3, 3], num_classes=num_classes)


def resnet32_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Create ResNet-32 for CIFAR.

    ResNet-32: 3 stages × 5 blocks = 15 blocks total
    Total layers: 2*15 + 2 = 32

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-32 model
    """
    return ResNetCIFAR(num_blocks=[5, 5, 5], num_classes=num_classes)


def resnet44_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Create ResNet-44 for CIFAR.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-44 model
    """
    return ResNetCIFAR(num_blocks=[7, 7, 7], num_classes=num_classes)


def resnet56_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Create ResNet-56 for CIFAR.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-56 model
    """
    return ResNetCIFAR(num_blocks=[9, 9, 9], num_classes=num_classes)
