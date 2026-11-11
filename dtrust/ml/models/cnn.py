"""
Convolutional Neural Network for DTRUST

A simple CNN architecture suitable for image classification tasks.
This model provides better performance than SimpleNN for image data.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMLModel


class SimpleCNN(BaseMLModel):
    """
    Simple Convolutional Neural Network.
    
    A basic CNN with 2 convolutional layers followed by fully connected layers,
    suitable for image classification tasks like CIFAR-10 or MNIST.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 input_size: Tuple[int, int] = (32, 32),
                 num_classes: int = 10,
                 conv1_channels: int = 32,
                 conv2_channels: int = 64,
                 fc_hidden_size: int = 128,
                 dropout_rate: float = 0.2,
                 **kwargs):
        """
        Initialize SimpleCNN.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Input image size as (height, width)
            num_classes: Number of output classes
            conv1_channels: Number of channels in first conv layer
            conv2_channels: Number of channels in second conv layer
            fc_hidden_size: Size of fully connected hidden layer
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name="SimpleCNN",
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            fc_hidden_size=fc_hidden_size,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc_hidden_size = fc_hidden_size
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        self.feature_size = self._calculate_feature_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        self._is_initialized = True
    
    def _calculate_feature_size(self) -> int:
        """Calculate the size of features after conv and pooling layers."""
        # After conv1 + pool: size / 2
        # After conv2 + pool: size / 4
        h, w = self.input_size
        h_out = h // 4  # Two pooling operations with stride 2
        w_out = w // 4
        return self.conv2_channels * h_out * w_out
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_name': self.model_name,
            'model_type': 'convolutional',
            'architecture': 'simple_cnn',
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'conv1_channels': self.conv1_channels,
            'conv2_channels': self.conv2_channels,
            'fc_hidden_size': self.fc_hidden_size,
            'dropout_rate': self.dropout_rate,
            'feature_size': self.feature_size,
            'parameter_count': self.get_parameter_count(),
            'model_size_mb': self.get_model_size_mb(),
            'layers': [
                {'type': 'Conv2d', 'in_channels': self.input_channels, 'out_channels': self.conv1_channels, 'kernel_size': 3},
                {'type': 'ReLU'},
                {'type': 'MaxPool2d', 'kernel_size': 2},
                {'type': 'Conv2d', 'in_channels': self.conv1_channels, 'out_channels': self.conv2_channels, 'kernel_size': 3},
                {'type': 'ReLU'},
                {'type': 'MaxPool2d', 'kernel_size': 2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'in_features': self.feature_size, 'out_features': self.fc_hidden_size},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': self.dropout_rate},
                {'type': 'Linear', 'in_features': self.fc_hidden_size, 'out_features': self.num_classes}
            ]
        }
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the network (conv layers + first FC).
        
        Returns:
            Feature extractor module
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.conv1 = parent_model.conv1
                self.conv2 = parent_model.conv2
                self.pool = parent_model.pool
                self.fc1 = parent_model.fc1
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                return x
        
        return FeatureExtractor(self)
    
    def get_classifier(self) -> nn.Module:
        """
        Get the classification layer.
        
        Returns:
            Classification layer
        """
        return self.fc2
    
    def freeze_conv_layers(self):
        """Freeze convolutional layers for transfer learning."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
    
    def unfreeze_conv_layers(self):
        """Unfreeze convolutional layers."""
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True
    
    def freeze_feature_extractor(self):
        """Freeze all feature extraction layers."""
        self.freeze_conv_layers()
        for param in self.fc1.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """Unfreeze all feature extraction layers."""
        self.unfreeze_conv_layers()
        for param in self.fc1.parameters():
            param.requires_grad = True
    
    @classmethod
    def create_mnist_model(cls, dropout_rate: float = 0.2) -> 'SimpleCNN':
        """
        Create a SimpleCNN configured for MNIST dataset.
        
        Args:
            dropout_rate: Dropout rate
            
        Returns:
            SimpleCNN configured for MNIST
        """
        return cls(
            input_channels=1,  # Grayscale
            input_size=(28, 28),
            num_classes=10,
            conv1_channels=16,
            conv2_channels=32,
            fc_hidden_size=64,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def create_cifar10_model(cls, dropout_rate: float = 0.3) -> 'SimpleCNN':
        """
        Create a SimpleCNN configured for CIFAR-10 dataset.
        
        Args:
            dropout_rate: Dropout rate
            
        Returns:
            SimpleCNN configured for CIFAR-10
        """
        return cls(
            input_channels=3,  # RGB
            input_size=(32, 32),
            num_classes=10,
            conv1_channels=32,
            conv2_channels=64,
            fc_hidden_size=128,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def create_custom_model(cls,
                           input_channels: int,
                           input_size: Tuple[int, int],
                           num_classes: int,
                           conv1_channels: int = 32,
                           conv2_channels: int = 64,
                           fc_hidden_size: int = 128,
                           dropout_rate: float = 0.2) -> 'SimpleCNN':
        """
        Create a SimpleCNN with custom configuration.
        
        Args:
            input_channels: Number of input channels
            input_size: Input image size as (height, width)
            num_classes: Number of output classes
            conv1_channels: Number of channels in first conv layer
            conv2_channels: Number of channels in second conv layer
            fc_hidden_size: Size of fully connected hidden layer
            dropout_rate: Dropout rate
            
        Returns:
            SimpleCNN with custom configuration
        """
        return cls(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            fc_hidden_size=fc_hidden_size,
            dropout_rate=dropout_rate
        ) 