"""
Base ML Model for DTRUST

Provides a common interface for all machine learning models used in federated learning.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseMLModel(nn.Module, ABC):
    """
    Abstract base class for all ML models in DTRUST.
    
    This class provides a consistent interface for models used in federated learning,
    ensuring they can be easily integrated with different FL frameworks.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.model_name = model_name
        self.model_config = kwargs
        self._is_initialized = False
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
    
    def get_parameters(self) -> List[torch.Tensor]:
        """
        Get model parameters as a list of tensors.
        
        Returns:
            List of parameter tensors
        """
        return [param.data for param in self.parameters()]
    
    def set_parameters(self, parameters: List[torch.Tensor]) -> None:
        """
        Set model parameters from a list of tensors.
        
        Args:
            parameters: List of parameter tensors
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v 
                     for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
    
    def get_parameter_count(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """
        Get approximate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def train_step(self, 
                   data_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device = None) -> Dict[str, float]:
        """
        Perform one training step (epoch).
        
        Args:
            data_loader: Training data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            
        Returns:
            Dictionary with training metrics
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total
        }
    
    def evaluate_step(self, 
                     data_loader: DataLoader,
                     criterion: nn.Module,
                     device: torch.device = None) -> Dict[str, float]:
        """
        Perform evaluation on a dataset.
        
        Args:
            data_loader: Evaluation data loader
            criterion: Loss function
            device: Device to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save model state to file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_name': self.model_name,
            'model_config': self.model_config,
            'state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseMLModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(checkpoint['model_name'], **checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name}(params={self.get_parameter_count():,}, size={self.get_model_size_mb():.2f}MB)"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', config={self.model_config})" 