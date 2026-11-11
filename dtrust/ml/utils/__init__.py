"""
ML Utilities for DTRUST

This module provides utilities for machine learning operations in federated learning scenarios.
The utilities are designed to complement Flower's built-in capabilities rather than duplicate them.

Key components:
- Model utilities: Core model parameter handling and management
- Data utilities: Federated data loading and splitting
- Training utilities: Training and evaluation helpers
"""

from .model_utils import (
    get_model_parameters,
    set_model_parameters,
    count_parameters,
    get_model_size,
    compare_models,
    save_model_state,
    load_model_state,
    get_parameter_stats,
    initialize_model_weights,
    freeze_layers,
    unfreeze_layers,
    get_layer_names,
    clone_model,
    # Alias for backward compatibility
    compare_model_parameters
)

from .data_utils import (
    # Dataset classes
    DTRUSTDataset,
    
    # Data loading functions
    load_csv_data,
    load_json_data,
    load_numpy_data,
    load_data_auto,
    auto_detect_data_format,
    
    # Federated data splitting
    create_federated_split,
    
    # Common dataset loaders
    load_mnist_data,
    load_cifar10_data,
    
    # Data loader utilities
    create_data_loaders
)

__all__ = [
    # Model utilities
    "get_model_parameters",
    "set_model_parameters", 
    "count_parameters",
    "get_model_size",
    "compare_models",
    "compare_model_parameters",  # Alias
    "save_model_state",
    "load_model_state",
    "get_parameter_stats",
    "initialize_model_weights",
    "freeze_layers",
    "unfreeze_layers",
    "get_layer_names",
    "clone_model",
    
    # Data utilities
    "DTRUSTDataset",
    "load_csv_data",
    "load_json_data", 
    "load_numpy_data",
    "load_data_auto",
    "auto_detect_data_format",
    "create_federated_split",
    "load_mnist_data",
    "load_cifar10_data",
    "create_data_loaders"
]