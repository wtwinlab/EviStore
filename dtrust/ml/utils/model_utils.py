"""
Model Utilities for DTRUST

Core utilities for working with ML models in federated learning scenarios.
These utilities focus on model parameter handling and complement Flower's 
built-in aggregation capabilities rather than duplicating them.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


def get_model_parameters(
    model: nn.Module, 
    as_numpy: bool = True
) -> Union[List[np.ndarray], List[torch.Tensor]]:
    """
    Extract model parameters as numpy arrays or tensors.
    
    This function is optimized for use with Flower's parameter serialization.
    
    Args:
        model: PyTorch model
        as_numpy: If True, return numpy arrays; if False, return tensors
        
    Returns:
        List of parameter arrays/tensors
    """
    if as_numpy:
        return [param.detach().cpu().numpy() for param in model.parameters()]
    else:
        return [param.detach().clone() for param in model.parameters()]


def set_model_parameters(
    model: nn.Module, 
    parameters: Union[List[np.ndarray], List[torch.Tensor]]
) -> None:
    """
    Set model parameters from numpy arrays or tensors.
    
    Compatible with Flower's parameter deserialization.
    
    Args:
        model: PyTorch model to update
        parameters: List of parameter arrays/tensors
    """
    params_dict = zip(model.parameters(), parameters)
    for model_param, new_param in params_dict:
        if isinstance(new_param, np.ndarray):
            new_param = torch.from_numpy(new_param)
        model_param.data.copy_(new_param)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "parameters_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
        "total_mb": total_size / 1024 / 1024
    }


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    Compare two models for structural and parameter differences.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Dictionary with comparison results
    """
    # Compare architecture
    arch_same = str(model1) == str(model2)
    
    # Compare parameter counts
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    params_same = params1 == params2
    
    # Compare parameter values (if architectures match)
    values_same = False
    max_diff = float('inf')
    
    if arch_same and params_same:
        try:
            params1_list = get_model_parameters(model1, as_numpy=True)
            params2_list = get_model_parameters(model2, as_numpy=True)
            
            diffs = []
            for p1, p2 in zip(params1_list, params2_list):
                diff = np.abs(p1 - p2).max()
                diffs.append(diff)
            
            max_diff = max(diffs) if diffs else 0.0
            values_same = max_diff < 1e-6  # Tolerance for floating point comparison
            
        except Exception as e:
            logger.warning(f"Could not compare parameter values: {e}")
    
    return {
        "architecture_same": arch_same,
        "parameters_same": params_same,
        "values_same": values_same,
        "max_parameter_diff": max_diff,
        "model1_params": params1,
        "model2_params": params2
    }


def save_model_state(
    model: nn.Module, 
    filepath: Union[str, Path],
    include_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model state to file.
    
    Args:
        model: PyTorch model
        filepath: Path to save file
        include_optimizer: Whether to include optimizer state
        optimizer: Optimizer to save (if include_optimizer=True)
        metadata: Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_architecture": str(model),
        "parameter_count": count_parameters(model),
        "model_size": get_model_size(model)
    }
    
    if include_optimizer and optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
        save_dict["optimizer_type"] = type(optimizer).__name__
    
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model_state(
    model: nn.Module,
    filepath: Union[str, Path],
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model state from file.
    
    Args:
        model: PyTorch model to load state into
        filepath: Path to saved model file
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary with loaded metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Model loaded from {filepath}")
    
    # Return metadata
    return {
        "parameter_count": checkpoint.get("parameter_count", {}),
        "model_size": checkpoint.get("model_size", {}),
        "metadata": checkpoint.get("metadata", {}),
        "optimizer_type": checkpoint.get("optimizer_type", None)
    }


def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get statistics about model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    params = get_model_parameters(model, as_numpy=True)
    
    if not params:
        return {"error": "No parameters found"}
    
    # Flatten all parameters
    all_params = np.concatenate([p.flatten() for p in params])
    
    stats = {
        "count": len(all_params),
        "mean": float(np.mean(all_params)),
        "std": float(np.std(all_params)),
        "min": float(np.min(all_params)),
        "max": float(np.max(all_params)),
        "median": float(np.median(all_params)),
        "num_zero": int(np.sum(all_params == 0)),
        "num_positive": int(np.sum(all_params > 0)),
        "num_negative": int(np.sum(all_params < 0))
    }
    
    # Add layer-wise statistics
    layer_stats = []
    for i, param in enumerate(params):
        layer_stat = {
            "layer": i,
            "shape": param.shape,
            "size": param.size,
            "mean": float(np.mean(param)),
            "std": float(np.std(param)),
            "min": float(np.min(param)),
            "max": float(np.max(param))
        }
        layer_stats.append(layer_stat)
    
    stats["layer_stats"] = layer_stats
    return stats


def initialize_model_weights(
    model: nn.Module, 
    method: str = "xavier_uniform",
    **kwargs
) -> None:
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model
        method: Initialization method
        **kwargs: Additional arguments for initialization
    """
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, **kwargs)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(m.weight, **kwargs)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, **kwargs)
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, **kwargs)
            elif method == "normal":
                std = kwargs.get("std", 0.01)
                nn.init.normal_(m.weight, mean=0, std=std)
            elif method == "uniform":
                a = kwargs.get("a", -0.1)
                b = kwargs.get("b", 0.1)
                nn.init.uniform_(m.weight, a=a, b=b)
            else:
                logger.warning(f"Unknown initialization method: {method}")
                return
            
            # Initialize bias if present
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    logger.info(f"Model weights initialized using {method}")


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specified layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specified layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            unfrozen_count += 1
    
    logger.info(f"Unfrozen {unfrozen_count} parameters in layers: {layer_names}")


def get_layer_names(model: nn.Module) -> List[str]:
    """
    Get all layer names in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer names
    """
    return [name for name, _ in model.named_parameters()]


def clone_model(model: nn.Module) -> nn.Module:
    """
    Create a deep copy of the model.
    
    Args:
        model: PyTorch model to clone
        
    Returns:
        Cloned model
    """
    # Create a new instance of the same model class
    cloned_model = type(model)()
    
    # Copy the state dict
    cloned_model.load_state_dict(model.state_dict())
    
    return cloned_model


# Alias for backward compatibility
compare_model_parameters = compare_models 