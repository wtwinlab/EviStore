"""
Data utilities for DTRUST ML models.

This module provides utilities for loading and preparing data for federated learning,
including file-based data loading, dataset partitioning, and common dataset utilities.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DTRUSTDataset(Dataset):
    """Custom dataset class for DTRUST with flexible data loading."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform: Optional[Callable] = None):
        """Initialize dataset.
        
        Args:
            data: Input data array
            targets: Target labels array
            transform: Optional transform function
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


def load_csv_data(
    file_path: Union[str, Path],
    target_column: str = 'target',
    feature_columns: Optional[List[str]] = None,
    header: bool = True,
    delimiter: str = ','
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        target_column: Name of target column
        feature_columns: List of feature column names (if None, use all except target)
        header: Whether CSV has header row
        delimiter: CSV delimiter
        
    Returns:
        Tuple of (features, targets) as numpy arrays
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, header=0 if header else None)
        
        if not header:
            # If no header, assume last column is target
            target_column = df.columns[-1]
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")
        
        # Extract targets
        targets = df[target_column].values
        
        # Extract features
        if feature_columns:
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            features = df[feature_columns].values
        else:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != target_column]
            features = df[feature_cols].values
        
        logger.info(f"Loaded CSV data: {features.shape[0]} samples, {features.shape[1]} features")
        return features.astype(np.float32), targets.astype(np.int64)
        
    except Exception as e:
        logger.error(f"Error loading CSV data from {file_path}: {e}")
        raise


def load_json_data(
    file_path: Union[str, Path],
    data_key: str = 'data',
    target_key: str = 'targets'
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        data_key: Key for feature data in JSON
        target_key: Key for target data in JSON
        
    Returns:
        Tuple of (features, targets) as numpy arrays
    """
    try:
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        
        if data_key not in data_dict:
            raise ValueError(f"Data key '{data_key}' not found in JSON")
        if target_key not in data_dict:
            raise ValueError(f"Target key '{target_key}' not found in JSON")
        
        features = np.array(data_dict[data_key], dtype=np.float32)
        targets = np.array(data_dict[target_key], dtype=np.int64)
        
        logger.info(f"Loaded JSON data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, targets
        
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise


def load_numpy_data(
    data_path: Union[str, Path],
    target_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from numpy files.
    
    Args:
        data_path: Path to numpy file containing features (.npy or .npz)
        target_path: Path to numpy file containing targets (if None, assume targets in same file)
        
    Returns:
        Tuple of (features, targets) as numpy arrays
    """
    try:
        data_path = Path(data_path)
        
        if data_path.suffix == '.npz':
            # Load from npz file
            data_dict = np.load(data_path)
            
            if target_path is None:
                # Assume both data and targets are in the same npz file
                if 'data' in data_dict and 'targets' in data_dict:
                    features = data_dict['data']
                    targets = data_dict['targets']
                elif 'X' in data_dict and 'y' in data_dict:
                    features = data_dict['X']
                    targets = data_dict['y']
                else:
                    # Use first two arrays
                    keys = list(data_dict.keys())
                    if len(keys) < 2:
                        raise ValueError("NPZ file must contain at least 2 arrays")
                    features = data_dict[keys[0]]
                    targets = data_dict[keys[1]]
            else:
                # Load targets from separate file
                features = data_dict['data'] if 'data' in data_dict else data_dict[list(data_dict.keys())[0]]
                targets = np.load(target_path)
        else:
            # Load from npy file
            features = np.load(data_path)
            if target_path is None:
                raise ValueError("target_path must be provided when loading from .npy file")
            targets = np.load(target_path)
        
        features = features.astype(np.float32)
        targets = targets.astype(np.int64)
        
        logger.info(f"Loaded numpy data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, targets
        
    except Exception as e:
        logger.error(f"Error loading numpy data from {data_path}: {e}")
        raise


def create_federated_split(
    features: np.ndarray,
    targets: np.ndarray,
    num_clients: int,
    split_method: str = 'iid',
    alpha: float = 0.5,
    min_samples: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data for federated learning clients.
    
    Args:
        features: Feature data
        targets: Target labels
        num_clients: Number of clients to split data for
        split_method: 'iid' for independent and identically distributed,
                     'non_iid' for non-IID split based on labels,
                     'dirichlet' for Dirichlet distribution split
        alpha: Alpha parameter for Dirichlet distribution (lower = more non-IID)
        min_samples: Minimum samples per client
        
    Returns:
        List of (features, targets) tuples for each client
    """
    n_samples = len(features)
    n_classes = len(np.unique(targets))
    
    if split_method == 'iid':
        # Simple random split
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, num_clients)
        
        client_data = []
        for split in splits:
            if len(split) < min_samples:
                logger.warning(f"Client has only {len(split)} samples (min: {min_samples})")
            client_data.append((features[split], targets[split]))
        
        return client_data
    
    elif split_method == 'non_iid':
        # Sort by labels and split
        sorted_indices = np.argsort(targets)
        splits = np.array_split(sorted_indices, num_clients)
        
        client_data = []
        for split in splits:
            if len(split) < min_samples:
                logger.warning(f"Client has only {len(split)} samples (min: {min_samples})")
            client_data.append((features[split], targets[split]))
        
        return client_data
    
    elif split_method == 'dirichlet':
        # Dirichlet distribution split
        client_data = [[] for _ in range(num_clients)]
        
        for class_id in range(n_classes):
            class_indices = np.where(targets == class_id)[0]
            n_class_samples = len(class_indices)
            
            if n_class_samples == 0:
                continue
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * n_class_samples).astype(int)
            
            # Ensure at least one sample per client if possible
            remainder = n_class_samples - proportions.sum()
            for i in range(remainder):
                proportions[i % num_clients] += 1
            
            # Split class samples among clients
            start_idx = 0
            for client_id, n_samples_client in enumerate(proportions):
                if n_samples_client > 0:
                    end_idx = start_idx + n_samples_client
                    client_indices = class_indices[start_idx:end_idx]
                    client_data[client_id].extend(client_indices)
                    start_idx = end_idx
        
        # Convert to numpy arrays
        result = []
        for client_indices in client_data:
            if len(client_indices) < min_samples:
                logger.warning(f"Client has only {len(client_indices)} samples (min: {min_samples})")
            
            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)  # Shuffle within client
            result.append((features[client_indices], targets[client_indices]))
        
        return result
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def load_mnist_data(data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load MNIST data
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    try:
        from torchvision import datasets, transforms
        
        if data_dir is None:
            data_dir = './data/mnist'
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        
        # Convert to numpy
        X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_train = train_dataset.targets.numpy().astype(np.int64)
        
        X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = test_dataset.targets.numpy().astype(np.int64)
        
        logger.info(f"Loaded MNIST: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise


def load_cifar10_data(data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    try:
        from torchvision import datasets, transforms
        
        if data_dir is None:
            data_dir = './data/cifar10'
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        
        # Apply ToTensor + Normalize to the whole uint8 HWC array in one batched step
        # (same values as the per-sample transform):
        #     ToTensor  : HWC uint8 [0,255] -> CHW float [0,1]
        #     Normalize : (x - mean) / std, per channel
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)

        def _to_arrays(ds):
            x = np.asarray(ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            x = (x - mean) / std
            y = np.asarray(ds.targets, dtype=np.int64)
            # Flatten (N, C, H, W) -> (N, C*H*W) for the simple NN
            return x.reshape(x.shape[0], -1), y

        X_train, y_train = _to_arrays(train_dataset)
        X_test, y_test = _to_arrays(test_dataset)
        
        logger.info(f"Loaded CIFAR-10: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading CIFAR-10 data: {e}")
        raise


def load_fashion_mnist_data(data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision import datasets, transforms
    if data_dir is None:
        data_dir = './data/fashion_mnist'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy().astype(np.int64)
    X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy().astype(np.int64)
    logger.info(f"Loaded Fashion-MNIST: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_femnist_data(
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FEMNIST (LEAF) from the self-contained .pt produced by
    scripts/prepare_femnist.py.

    Returns:
        (X_train [N,1,28,28] float32 normalized, y_train [N] int64,
         X_test, y_test, writer_train [N] int64) — writer_train gives each
        training sample's LEAF writer id for natural non-IID partitioning.
    """
    if data_dir is None:
        data_dir = "./data"
    path = os.path.join(data_dir, "femnist", "femnist.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FEMNIST archive not found at {path}. Run scripts/prepare_femnist.py "
            f"once (needs `pip install datasets`), or scp the .pt from a machine "
            f"that has."
        )
    blob = torch.load(path, map_location="cpu")
    mean, std = float(blob["mean"]), float(blob["std"])

    def _norm(x_uint8):
        x = x_uint8.numpy().astype(np.float32) / 255.0
        return (x - mean) / std

    X_train = _norm(blob["x_train"]).reshape(-1, 1, 28, 28)
    X_test = _norm(blob["x_test"]).reshape(-1, 1, 28, 28)
    y_train = blob["y_train"].numpy().astype(np.int64)
    y_test = blob["y_test"].numpy().astype(np.int64)
    writer_train = blob["writer_train"].numpy().astype(np.int64)
    logger.info(
        f"Loaded FEMNIST: train={X_train.shape}, test={X_test.shape}, "
        f"writers={len(np.unique(writer_train))}"
    )
    return X_train, y_train, X_test, y_test, writer_train


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """Dirichlet(alpha) label-skew partition (Hsu et al.); the standard
    non-IID benchmark split. Shared by both FL harnesses.

    Small alpha can leave a client with zero samples; those clients each take
    one sample from the currently largest client so every partition is
    non-empty (deterministic, negligible distribution impact).
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    idx_by_c = [np.where(labels == c)[0] for c in range(n_classes)]
    for a in idx_by_c:
        rng.shuffle(a)
    parts: List[list] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        prop = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(prop) * len(idx_by_c[c])).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx_by_c[c], cuts)):
            parts[i].extend(chunk.tolist())
    for i in range(num_clients):
        while len(parts[i]) == 0:
            donor = max(range(num_clients), key=lambda j: len(parts[j]))
            parts[i].append(parts[donor].pop())
    return [np.array(sorted(p), dtype=np.int64) for p in parts]


def natural_writer_partition(
    writer_ids: np.ndarray, num_clients: int, min_samples: int, seed: int
) -> List[np.ndarray]:
    """FEMNIST natural non-IID partition: each FL client is one LEAF writer.

    Samples ``num_clients`` distinct writers with at least ``min_samples``
    training samples (seeded, reproducible) and returns their index sets.
    """
    writers, counts = np.unique(writer_ids, return_counts=True)
    eligible = writers[counts >= min_samples]
    if len(eligible) < num_clients:
        raise ValueError(
            f"FEMNIST: only {len(eligible)} writers have >= {min_samples} "
            f"samples; cannot form {num_clients} natural clients "
            f"(lower --num_clients or the min-samples requirement)"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=num_clients, replace=False)
    return [np.where(writer_ids == w)[0].astype(np.int64) for w in chosen]


def create_data_loaders(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True,
    transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders from features and targets.
    
    Args:
        features: Feature data
        targets: Target labels
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle training data
        transform: Optional transform function
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset = DTRUSTDataset(features, targets, transform)
    
    # Split into train and test
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train
    
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Avoid multiprocessing issues in FL
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, test={len(test_dataset)}")
    return train_loader, test_loader


def auto_detect_data_format(file_path: Union[str, Path]) -> str:
    """Auto-detect data file format.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Detected format: 'csv', 'json', 'npy', 'npz', or 'unknown'
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    format_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.npy': 'npy',
        '.npz': 'npz',
        '.txt': 'csv',  # Assume text files are CSV-like
    }
    
    return format_map.get(suffix, 'unknown')


def load_data_auto(
    file_path: Union[str, Path],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Automatically load data based on file format.
    
    Args:
        file_path: Path to data file
        **kwargs: Additional arguments passed to specific loader functions
        
    Returns:
        Tuple of (features, targets) as numpy arrays
    """
    format_type = auto_detect_data_format(file_path)
    
    if format_type == 'csv':
        return load_csv_data(file_path, **kwargs)
    elif format_type == 'json':
        return load_json_data(file_path, **kwargs)
    elif format_type in ['npy', 'npz']:
        return load_numpy_data(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {format_type}")


# Export commonly used functions
__all__ = [
    'DTRUSTDataset',
    'load_csv_data',
    'load_json_data', 
    'load_numpy_data',
    'load_data_auto',
    'create_federated_split',
    'load_mnist_data',
    'load_cifar10_data',
    'load_fashion_mnist_data',
    'load_femnist_data',
    'dirichlet_partition',
    'natural_writer_partition',
    'create_data_loaders',
    'auto_detect_data_format'
] 