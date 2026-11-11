"""
Model Registry for DTRUST ML Models

Provides a centralized registry for managing and creating different ML models.
This allows for dynamic model creation and easy extensibility.
"""

from typing import Dict, Type, Any, Optional, List
import logging
from .base import BaseMLModel
from .simple_nn import SimpleNN
from .cnn import SimpleCNN
from .resnet_cifar import ResNetCIFAR

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing ML models in DTRUST.
    
    This class provides a centralized way to register, create, and manage
    different types of ML models used in federated learning.
    """
    
    _models: Dict[str, Type[BaseMLModel]] = {}
    _model_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_model(cls, 
                      name: str, 
                      model_class: Type[BaseMLModel],
                      default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name identifier for the model
            model_class: Model class that inherits from BaseMLModel
            default_config: Default configuration for the model
        """
        if not issubclass(model_class, BaseMLModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseMLModel")
        
        cls._models[name] = model_class
        cls._model_configs[name] = default_config or {}
        
        logger.info(f"Registered model: {name} -> {model_class.__name__}")
    
    @classmethod
    def unregister_model(cls, name: str) -> None:
        """
        Unregister a model type.
        
        Args:
            name: Name of the model to unregister
        """
        if name in cls._models:
            del cls._models[name]
            if name in cls._model_configs:
                del cls._model_configs[name]
            logger.info(f"Unregistered model: {name}")
        else:
            logger.warning(f"Model {name} not found in registry")
    
    @classmethod
    def create_model(cls, 
                    name: str, 
                    config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> BaseMLModel:
        """
        Create a model instance by name.
        
        Args:
            name: Name of the model to create
            config: Configuration dictionary for the model
            **kwargs: Additional keyword arguments
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        
        model_class = cls._models[name]
        
        # Merge default config with provided config
        final_config = cls._model_configs[name].copy()
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        logger.info(f"Creating model: {name} with config: {final_config}")
        
        try:
            return model_class(**final_config)
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            raise
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseMLModel]:
        """
        Get the model class by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        
        return cls._models[name]
    
    @classmethod
    def get_model_config(cls, name: str) -> Dict[str, Any]:
        """
        Get the default configuration for a model.
        
        Args:
            name: Name of the model
            
        Returns:
            Default configuration dictionary
        """
        return cls._model_configs.get(name, {}).copy()
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        Get list of all registered model names.
        
        Returns:
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found")
        
        model_class = cls._models[name]
        default_config = cls._model_configs[name]
        
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'default_config': default_config,
            'docstring': model_class.__doc__
        }
    
    @classmethod
    def get_all_model_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models.
        
        Returns:
            Dictionary mapping model names to their information
        """
        return {name: cls.get_model_info(name) for name in cls._models.keys()}
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered models (mainly for testing)."""
        cls._models.clear()
        cls._model_configs.clear()
        logger.info("Cleared model registry")


# Register built-in models
def _register_builtin_models():
    """Register the built-in models."""
    
    # SimpleNN variants
    ModelRegistry.register_model(
        "simple_nn",
        SimpleNN,
        {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    )
    
    ModelRegistry.register_model(
        "simple_nn_mnist",
        SimpleNN,
        {"input_size": 784, "hidden_size": 64, "num_classes": 10, "dropout_rate": 0.0}
    )
    
    ModelRegistry.register_model(
        "simple_nn_cifar10",
        SimpleNN,
        {"input_size": 3072, "hidden_size": 128, "num_classes": 10, "dropout_rate": 0.2}
    )
    
    # SimpleCNN variants
    ModelRegistry.register_model(
        "simple_cnn",
        SimpleCNN,
        {"input_channels": 3, "input_size": (32, 32), "num_classes": 10}
    )
    
    ModelRegistry.register_model(
        "simple_cnn_mnist",
        SimpleCNN,
        {
            "input_channels": 1,
            "input_size": (28, 28),
            "num_classes": 10,
            "conv1_channels": 16,
            "conv2_channels": 32,
            "fc_hidden_size": 64,
            "dropout_rate": 0.2
        }
    )
    
    ModelRegistry.register_model(
        "simple_cnn_cifar10",
        SimpleCNN,
        {
            "input_channels": 3,
            "input_size": (32, 32),
            "num_classes": 10,
            "conv1_channels": 32,
            "conv2_channels": 64,
            "fc_hidden_size": 128,
            "dropout_rate": 0.3
        }
    )

    # FEMNIST (LEAF): 28x28 grayscale, 62 classes (10 digits + 26 lower + 26 upper)
    ModelRegistry.register_model(
        "simple_cnn_femnist",
        SimpleCNN,
        {
            "input_channels": 1,
            "input_size": (28, 28),
            "num_classes": 62,
            "conv1_channels": 32,
            "conv2_channels": 64,
            "fc_hidden_size": 128,
            "dropout_rate": 0.2
        }
    )

    # ResNetCIFAR variants (matching FLTrust for fair comparison)
    ModelRegistry.register_model(
        "resnet20_cifar10",
        ResNetCIFAR,
        {
            "num_blocks": [3, 3, 3],
            "num_classes": 10
        }
    )

    ModelRegistry.register_model(
        "resnet32_cifar10",
        ResNetCIFAR,
        {
            "num_blocks": [5, 5, 5],
            "num_classes": 10
        }
    )

    # GroupNorm ResNet-20 — federated-friendly (no BN running buffers, so it trains
    # under this harness's parameters()-only median aggregation, unlike BN ResNet).
    ModelRegistry.register_model(
        "resnet20gn_cifar10",
        ResNetCIFAR,
        {
            "num_blocks": [3, 3, 3],
            "num_classes": 10,
            "norm": "gn"
        }
    )

    ModelRegistry.register_model(
        "resnet20_cifar100",
        ResNetCIFAR,
        {
            "num_blocks": [3, 3, 3],
            "num_classes": 100
        }
    )


# Register built-in models when module is imported
_register_builtin_models()


# Convenience functions
def create_model(name: str, **kwargs) -> BaseMLModel:
    """
    Convenience function to create a model.
    
    Args:
        name: Name of the model
        **kwargs: Model configuration
        
    Returns:
        Model instance
    """
    return ModelRegistry.create_model(name, **kwargs)


def list_available_models() -> List[str]:
    """
    Get list of available models.
    
    Returns:
        List of model names
    """
    return ModelRegistry.list_models()


def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        name: Model name
        
    Returns:
        Model information
    """
    return ModelRegistry.get_model_info(name) 