"""
DTRUST Federated Learning Client.
Integrates with Flower FL framework and TEE trust management.
"""

import asyncio
import threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np

# Flower imports
import flwr as fl
from flwr.client import NumPyClient, start_client
from flwr.common import (
    GetParametersRes, Status, Code, Parameters, 
    ndarrays_to_parameters, parameters_to_ndarrays, 
    FitIns, FitRes, EvaluateIns, EvaluateRes
)

# DTRUST imports
from ..api.client import DTRUSTClient as APIClient
from ..core.logging import get_logger
from ..core.config import Config
from ..core.models import NodeIdentity, AttestationResult
from ..ml.models.registry import ModelRegistry, create_model
from ..ml.utils.model_utils import get_model_parameters, set_model_parameters
from ..ml.utils.data_utils import (
    load_data_auto, 
    create_data_loaders,
    load_mnist_data,
    load_cifar10_data,
    create_federated_split
)

logger = get_logger(__name__)


class DTRUSTFlowerClient(NumPyClient):
    """Flower client integrated with DTRUST trust management."""
    
    def __init__(
        self,
        node_id: str,
        enclave_id: str,
        api_server_url: str,
        model_fn,
        data_loader_fn,
        config: Config
    ):
        """Initialize DTRUST Flower client.
        
        Args:
            node_id: Unique node identifier
            enclave_id: SGX enclave identifier  
            api_server_url: DTRUST API server URL
            model_fn: Function that returns the model
            data_loader_fn: Function that returns (train_loader, test_loader)
            config: DTRUST configuration
        """
        self.node_id = node_id
        self.enclave_id = enclave_id
        self.api_server_url = api_server_url
        self.model_fn = model_fn
        self.data_loader_fn = data_loader_fn
        self.config = config
        
        # Initialize model and data
        self.model = model_fn()
        self.train_loader, self.test_loader = data_loader_fn()
        
        # DTRUST integration
        self.api_client = None
        self.node_identity = None
        self.attestation_result = None
        
        logger.info(f"Initialized DTRUST Flower client {node_id}")
        logger.info(f"Model: {self.model}")
    
    async def register_with_dtrust(self):
        """Register with DTRUST API server and perform attestation."""
        try:
            self.api_client = APIClient(base_url=self.api_server_url)
            
            # Register node
            self.node_identity = await self.api_client.register_node(self.enclave_id)
            logger.info(f"Registered {self.node_id} with DTRUST server")
            
            # Perform attestation
            self.attestation_result = await self.api_client.request_attestation(
                self.node_identity.node_id
            )
            logger.info(f"Attestation completed: {self.attestation_result.status} "
                       f"(confidence: {self.attestation_result.confidence:.3f})")
            
            return self.attestation_result
            
        except Exception as e:
            logger.error(f"Failed to register with DTRUST: {e}")
            raise
    
    def get_parameters(self, config: Dict[str, Any]) -> GetParametersRes:
        """Return model parameters using Flower's built-in utilities."""
        try:
            # Use DTRUST's ML utils function to get parameters
            parameters_list = get_model_parameters(self.model, as_numpy=True)
            parameters = ndarrays_to_parameters(parameters_list)
            return GetParametersRes(
                status=Status(code=Code.OK, message="Parameters retrieved successfully"),
                parameters=parameters
            )
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            # Return empty parameters on error
            empty_parameters = ndarrays_to_parameters([])
            return GetParametersRes(
                status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message=str(e)),
                parameters=empty_parameters
            )
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters using DTRUST's ML utils."""
        try:
            set_model_parameters(self.model, parameters)
        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            raise
    
    def fit(self, fit_ins: FitIns) -> FitRes:
        """Train the model using the model's built-in training capabilities."""
        # Extract parameters and config from FitIns
        parameters = parameters_to_ndarrays(fit_ins.parameters)
        config = fit_ins.config
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 0.01)
        
        # Use the model's built-in training method if available
        if hasattr(self.model, 'train_step'):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                
                # Setup optimizer and criterion
                import torch.optim as optim
                import torch.nn as nn
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
                
                total_metrics = {'loss': 0.0, 'accuracy': 0.0, 'samples': 0}
                
                for epoch in range(epochs):
                    metrics = self.model.train_step(
                        self.train_loader, 
                        optimizer, 
                        criterion, 
                        device
                    )
                    
                    # Accumulate metrics
                    for key in ['loss', 'accuracy', 'samples']:
                        total_metrics[key] += metrics.get(key, 0)
                
                # Average metrics over epochs
                avg_loss = total_metrics['loss'] / epochs if epochs > 0 else 0.0
                avg_accuracy = total_metrics['accuracy'] / epochs if epochs > 0 else 0.0
                num_examples = total_metrics['samples'] // epochs if epochs > 0 else 0
                
                logger.info(f"Training completed using model.train_step(): "
                           f"loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}, examples={num_examples}")
                
                # Return training metrics
                training_metrics = {
                    "loss": avg_loss,
                    "accuracy": avg_accuracy
                }
                
            except Exception as e:
                logger.warning(f"Failed to use model.train_step(), falling back to basic training: {e}")
                # Fall back to basic training
                avg_loss, num_examples, training_metrics = self._basic_training(epochs, learning_rate)
        else:
            # Basic training for models without train_step method
            avg_loss, num_examples, training_metrics = self._basic_training(epochs, learning_rate)
        
        # Record training participation with DTRUST
        if self.api_client and self.node_identity:
            try:
                # Use asyncio to run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.api_client.record_training_participation(
                        self.node_identity.node_id,
                        num_examples=num_examples,
                        metrics={"loss": avg_loss}
                    )
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to record training participation: {e}")
        
        # Get updated parameters
        updated_parameters = get_model_parameters(self.model, as_numpy=True)
        
        return FitRes(
            status=Status(code=Code.OK, message="Training completed successfully"),
            parameters=ndarrays_to_parameters(updated_parameters),
            num_examples=num_examples,
            metrics=training_metrics
        )
    
    def _basic_training(self, epochs: int, learning_rate: float) -> Tuple[float, int, Dict[str, float]]:
        """Basic training implementation as fallback."""
        import torch.optim as optim
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += len(data)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Basic training completed: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}, examples={total_samples}")
        
        return avg_loss, total_samples, {"loss": avg_loss, "accuracy": avg_accuracy}
    
    def evaluate(self, evaluate_ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model using the model's built-in evaluation capabilities."""
        # Extract parameters and config
        parameters = parameters_to_ndarrays(evaluate_ins.parameters)
        config = evaluate_ins.config
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Use the model's built-in evaluation method if available
        if hasattr(self.model, 'evaluate_step'):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                
                import torch.nn as nn
                criterion = nn.CrossEntropyLoss()
                
                metrics = self.model.evaluate_step(self.test_loader, criterion, device)
                
                loss = metrics.get('loss', 0.0)
                accuracy = metrics.get('accuracy', 0.0)
                num_examples = metrics.get('samples', len(self.test_loader.dataset))
                
                logger.info(f"Evaluation completed using model.evaluate_step(): "
                           f"loss={loss:.4f}, accuracy={accuracy:.4f}, examples={num_examples}")
                
                evaluation_metrics = {
                    "accuracy": accuracy
                }
                
            except Exception as e:
                logger.warning(f"Failed to use model.evaluate_step(), falling back to basic evaluation: {e}")
                # Fall back to basic evaluation
                loss, num_examples, evaluation_metrics = self._basic_evaluation()
        else:
            # Basic evaluation for models without evaluate_step method
            loss, num_examples, evaluation_metrics = self._basic_evaluation()
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Evaluation completed successfully"),
            loss=loss,
            num_examples=num_examples,
            metrics=evaluation_metrics
        )
    
    def _basic_evaluation(self) -> Tuple[float, int, Dict[str, float]]:
        """Basic evaluation implementation as fallback."""
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Basic evaluation completed: loss={avg_loss:.4f}, accuracy={accuracy:.4f}, examples={total_samples}")
        
        return avg_loss, total_samples, {"accuracy": accuracy}


class DTRUSTClient:
    """Main DTRUST client that integrates with Flower framework."""
    
    def __init__(
        self,
        node_id: str,
        enclave_id: str,
        server_address: str,
        config: Config,
        data_path: Optional[str] = None,
        model_name: str = "simple_nn_mnist"
    ):
        """Initialize DTRUST client.
        
        Args:
            node_id: Unique node identifier
            enclave_id: SGX enclave identifier
            server_address: FL server address (host:port)
            config: DTRUST configuration
            data_path: Path to client data (optional)
            model_name: Name of the model to use
        """
        self.node_id = node_id
        self.enclave_id = enclave_id
        self.server_address = server_address
        self.config = config
        self.data_path = data_path
        self.model_name = model_name
        
        # Parse server address
        if ":" in server_address:
            self.server_host, self.server_port = server_address.split(":", 1)
            self.server_port = int(self.server_port)
        else:
            self.server_host = server_address
            self.server_port = 8001  # Default FL port
        
        # API server URL (for DTRUST integration)
        self.api_server_url = f"http://{self.server_host}:8000"
        
        logger.info(f"Initialized DTRUST client {node_id}")
        logger.info(f"FL server: {self.server_host}:{self.server_port}")
        logger.info(f"API server: {self.api_server_url}")
    
    async def start(self):
        """Start the DTRUST client and connect to Flower server."""
        try:
            logger.info(f"Starting DTRUST client {self.node_id}")
            
            # Create model and data loader functions
            model_fn = self._create_model_fn()
            data_loader_fn = self._create_data_loader_fn()
            
            # Create Flower client
            flower_client = DTRUSTFlowerClient(
                node_id=self.node_id,
                enclave_id=self.enclave_id,
                api_server_url=self.api_server_url,
                model_fn=model_fn,
                data_loader_fn=data_loader_fn,
                config=self.config
            )
            
            # Register with DTRUST (optional, for trust management)
            try:
                await flower_client.register_with_dtrust()
            except Exception as e:
                logger.warning(f"Failed to register with DTRUST API: {e}")
                logger.info("Continuing without DTRUST integration")
            
            # Start Flower client in a separate thread to avoid blocking
            def run_flower_client():
                try:
                    logger.info(f"Connecting to Flower server at {self.server_host}:{self.server_port}")
                    start_client(
                        server_address=f"{self.server_host}:{self.server_port}",
                        client=flower_client
                    )
                except Exception as e:
                    logger.error(f"Flower client failed: {e}")
                    raise
            
            # Run in thread to allow for graceful shutdown
            client_thread = threading.Thread(target=run_flower_client, daemon=True)
            client_thread.start()
            
            logger.info(f"DTRUST client {self.node_id} started successfully")
            
            # Keep the main thread alive
            client_thread.join()
            
        except Exception as e:
            logger.error(f"Failed to start DTRUST client: {e}")
            raise
    
    def _create_data_loader_fn(self):
        """Create data loader function with improved data loading."""
        if self.data_path:
            # Load data from path using DTRUST's data utilities
            try:
                return self._load_data_from_path()
            except Exception as e:
                logger.warning(f"Failed to load data from {self.data_path}: {e}")
                logger.info("Falling back to synthetic data")
        
        # Fall back to synthetic data
        return self._create_default_data_loader_fn()
    
    def _load_data_from_path(self):
        """Load data from specified path using DTRUST's data utilities."""
        data_path = Path(self.data_path)
        
        if data_path.is_dir():
            # Check for federated data split directory with data.npz files
            data_npz_file = data_path / "data.npz"
            if data_npz_file.exists():
                logger.info(f"Loading federated MNIST data from {data_npz_file}")
                def data_loader_fn():
                    # Load the .npz file created by prepare-data command
                    data = np.load(str(data_npz_file))
                    
                    # The prepare-data command creates files with 'data' and 'targets' keys
                    if 'data' in data and 'targets' in data:
                        features = data['data']
                        targets = data['targets']
                        logger.info(f"Loaded MNIST data: {features.shape} features, {targets.shape} targets")
                    elif 'features' in data and 'targets' in data:
                        features = data['features']
                        targets = data['targets']
                        logger.info(f"Loaded MNIST data: {features.shape} features, {targets.shape} targets")
                    else:
                        # Fallback: try common key names
                        keys = list(data.keys())
                        logger.info(f"Available keys in {data_npz_file}: {keys}")
                        
                        # Try to find features and targets with different key names
                        feature_keys = ['data', 'features', 'X', 'x_train', 'train_data']
                        target_keys = ['targets', 'y', 'labels', 'y_train', 'train_labels']
                        
                        features = None
                        targets = None
                        
                        for key in feature_keys:
                            if key in data:
                                features = data[key]
                                break
                        
                        for key in target_keys:
                            if key in data:
                                targets = data[key]
                                break
                        
                        if features is None or targets is None:
                            raise ValueError(f"Could not find features and targets in {data_npz_file}. Available keys: {keys}")
                        
                        logger.info(f"Loaded data with keys: features={features.shape}, targets={targets.shape}")
                    
                    # Create data loaders
                    train_loader, test_loader = create_data_loaders(
                        features, targets, 
                        train_ratio=0.8,  # Split into train/test
                        batch_size=32
                    )
                    return train_loader, test_loader
                return data_loader_fn
            
            # Check for separate features.npy and targets.npy files
            features_file = data_path / "features.npy"
            targets_file = data_path / "targets.npy"
            if features_file.exists() and targets_file.exists():
                logger.info(f"Loading federated data from separate .npy files")
                def data_loader_fn():
                    features = np.load(str(features_file))
                    targets = np.load(str(targets_file))
                    logger.info(f"Loaded data: {features.shape} features, {targets.shape} targets")
                    
                    # Create data loaders
                    train_loader, test_loader = create_data_loaders(
                        features, targets,
                        train_ratio=0.8,
                        batch_size=32
                    )
                    return train_loader, test_loader
                return data_loader_fn
            
            # Check for common dataset directories
            elif (data_path / "MNIST").exists() or "mnist" in str(data_path).lower():
                logger.info("Loading MNIST dataset from directory")
                def data_loader_fn():
                    train_data, test_data = load_mnist_data(str(data_path))
                    train_loader, test_loader = create_data_loaders(
                        train_data, test_data, batch_size=32
                    )
                    return train_loader, test_loader
                return data_loader_fn
            
            elif (data_path / "CIFAR10").exists() or "cifar" in str(data_path).lower():
                logger.info("Loading CIFAR-10 dataset from directory")
                def data_loader_fn():
                    train_data, test_data = load_cifar10_data(str(data_path))
                    train_loader, test_loader = create_data_loaders(
                        train_data, test_data, batch_size=32
                    )
                    return train_loader, test_loader
                return data_loader_fn
            
            else:
                # Try to load as federated split directory (fallback)
                logger.info(f"Loading federated data split from {data_path}")
                def data_loader_fn():
                    # Use client-specific data based on node_id
                    client_hash = hash(self.node_id) % 1000  # Simple hash for client assignment
                    
                    # Look for client-specific subdirectories
                    client_dirs = list(data_path.glob("client_*"))
                    if client_dirs:
                        # Use hash to select a client directory
                        selected_dir = client_dirs[client_hash % len(client_dirs)]
                        logger.info(f"Using client data from {selected_dir}")
                        
                        train_data, test_data = load_data_auto(str(selected_dir))
                        train_loader, test_loader = create_data_loaders(
                            train_data, test_data, batch_size=32
                        )
                        return train_loader, test_loader
                    else:
                        # Load all data and create client-specific split
                        train_data, test_data = load_data_auto(str(data_path))
                        
                        # Create federated split for this client
                        federated_data = create_federated_split(
                            train_data, 
                            num_clients=4,  # Default number of clients
                            split_method='iid',
                            seed=client_hash
                        )
                        
                        # Use the first split for this client
                        client_train_data = federated_data[0]
                        train_loader, test_loader = create_data_loaders(
                            client_train_data, test_data, batch_size=32
                        )
                        return train_loader, test_loader
                
                return data_loader_fn
        
        elif data_path.is_file():
            # Load individual data file
            logger.info(f"Loading data from file {data_path}")
            def data_loader_fn():
                train_data, test_data = load_data_auto(str(data_path))
                train_loader, test_loader = create_data_loaders(
                    train_data, test_data, batch_size=32
                )
                return train_loader, test_loader
            return data_loader_fn
        
        else:
            raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    def _create_model_fn(self):
        """Create model function using DTRUST's model registry."""
        def model_fn():
            try:
                model = create_model(self.model_name)
                logger.info(f"Created model: {self.model_name}")
                return model
            except Exception as e:
                logger.error(f"Failed to create model {self.model_name}: {e}")
                # Fall back to a simple model
                logger.info("Falling back to simple neural network")
                return create_model("simple_nn_mnist")
        
        return model_fn
    
    def _create_default_data_loader_fn(self):
        """Create default synthetic data loader function."""
        def data_loader_fn():
            logger.info("Creating synthetic data for client")
            
            # Create synthetic data based on model type
            if "mnist" in self.model_name.lower():
                # MNIST-like synthetic data
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                
                # Generate synthetic MNIST-like data
                train_data = torch.randn(1000, 1, 28, 28)
                train_labels = torch.randint(0, 10, (1000,))
                test_data = torch.randn(200, 1, 28, 28)
                test_labels = torch.randint(0, 10, (200,))
                
                train_dataset = TensorDataset(train_data, train_labels)
                test_dataset = TensorDataset(test_data, test_labels)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
            elif "cifar" in self.model_name.lower():
                # CIFAR-like synthetic data
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                
                # Generate synthetic CIFAR-like data
                train_data = torch.randn(1000, 3, 32, 32)
                train_labels = torch.randint(0, 10, (1000,))
                test_data = torch.randn(200, 3, 32, 32)
                test_labels = torch.randint(0, 10, (200,))
                
                train_dataset = TensorDataset(train_data, train_labels)
                test_dataset = TensorDataset(test_data, test_labels)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
            else:
                # Generic synthetic data
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                
                # Generate generic synthetic data
                train_data = torch.randn(1000, 784)  # Flattened 28x28
                train_labels = torch.randint(0, 10, (1000,))
                test_data = torch.randn(200, 784)
                test_labels = torch.randint(0, 10, (200,))
                
                train_dataset = TensorDataset(train_data, train_labels)
                test_dataset = TensorDataset(test_data, test_labels)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            logger.info(f"Created synthetic data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
            return train_loader, test_loader
        
        return data_loader_fn 