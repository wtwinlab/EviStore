"""
Simple Neural Network for DTRUST

A basic fully-connected neural network suitable for simple classification tasks
like MNIST or CIFAR-10. This model is commonly used for federated learning demos.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .base import BaseMLModel


class SimpleNN(BaseMLModel):
    """
    Simple fully-connected neural network.
    
    This is a basic 2-layer neural network with ReLU activation,
    suitable for simple classification tasks.
    """
    
    def __init__(self, 
                 input_size: int = 784,
                 hidden_size: int = 64, 
                 num_classes: int = 10,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        Initialize SimpleNN.
        
        Args:
            input_size: Size of input features (default: 784 for MNIST)
            hidden_size: Size of hidden layer (default: 64)
            num_classes: Number of output classes (default: 10)
            dropout_rate: Dropout rate for regularization (default: 0.0)
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name="SimpleNN",
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        self._is_initialized = True
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, *) where * can be any dimensions
               that flatten to input_size
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Ensure input size matches expected
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(1)}")
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
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
            'model_type': 'fully_connected',
            'architecture': 'simple_nn',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'parameter_count': self.get_parameter_count(),
            'model_size_mb': self.get_model_size_mb(),
            'layers': [
                {'type': 'Linear', 'in_features': self.input_size, 'out_features': self.hidden_size},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': self.dropout_rate} if self.dropout_rate > 0 else None,
                {'type': 'Linear', 'in_features': self.hidden_size, 'out_features': self.num_classes}
            ]
        }
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the network (everything before final layer).
        
        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout
        )
    
    def get_classifier(self) -> nn.Module:
        """
        Get the classification layer.
        
        Returns:
            Classification layer
        """
        return self.fc2
    
    def freeze_feature_extractor(self):
        """Freeze the feature extraction layers for transfer learning."""
        for param in self.fc1.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extraction layers."""
        for param in self.fc1.parameters():
            param.requires_grad = True
    
    @classmethod
    def create_mnist_model(cls, hidden_size: int = 64, dropout_rate: float = 0.0) -> 'SimpleNN':
        """
        Create a SimpleNN configured for MNIST dataset.
        
        Args:
            hidden_size: Size of hidden layer
            dropout_rate: Dropout rate
            
        Returns:
            SimpleNN configured for MNIST
        """
        return cls(
            input_size=784,  # 28x28 flattened
            hidden_size=hidden_size,
            num_classes=10,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def create_cifar10_model(cls, hidden_size: int = 128, dropout_rate: float = 0.2) -> 'SimpleNN':
        """
        Create a SimpleNN configured for CIFAR-10 dataset.
        
        Args:
            hidden_size: Size of hidden layer
            dropout_rate: Dropout rate
            
        Returns:
            SimpleNN configured for CIFAR-10
        """
        return cls(
            input_size=3072,  # 32x32x3 flattened
            hidden_size=hidden_size,
            num_classes=10,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def create_custom_model(cls, 
                           input_size: int,
                           num_classes: int,
                           hidden_size: int = 64,
                           dropout_rate: float = 0.0) -> 'SimpleNN':
        """
        Create a SimpleNN with custom configuration.
        
        Args:
            input_size: Size of input features
            num_classes: Number of output classes
            hidden_size: Size of hidden layer
            dropout_rate: Dropout rate
            
        Returns:
            SimpleNN with custom configuration
        """
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ) 