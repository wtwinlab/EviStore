"""
Client Adapter - Wraps DTRUST Flower client to inject attacks.

This adapter uses the decorator pattern to wrap the original client
without modifying any of its code. All attack logic is contained
within the adapter.

Key Design Principles:
1. Zero modification to original DTRUST code
2. All attack state is contained in the adapter
3. Original client behavior is preserved when attacks are disabled
4. Clean separation of concerns
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Callable
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from flwr.client import NumPyClient
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays
)

from dtrust.attacks.adapters.attack_config import AttackConfig, AttackType
from dtrust.attacks.label_flipping import maybe_label_flip_batch
from dtrust.attacks.scaling_backdoor import (
    TriggerSpec, make_backdoor_batch, scale_delta
)


logger = logging.getLogger(__name__)


class AttackingDataLoader:
    """
    Wrapper around DataLoader that applies data poisoning attacks.

    This wrapper intercepts batches from the original DataLoader and
    applies the configured attack before returning the data.
    """

    def __init__(
        self,
        original_loader: DataLoader,
        attack_config: AttackConfig,
    ):
        """
        Initialize attacking data loader.

        Args:
            original_loader: Original PyTorch DataLoader
            attack_config: Attack configuration
        """
        self.original_loader = original_loader
        self.config = attack_config
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.original_loader)
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = next(self._iterator)

        if not self.config.enabled:
            return batch

        # Apply data poisoning based on attack type
        if self.config.attack_type == AttackType.LABEL_FLIPPING:
            return maybe_label_flip_batch(
                batch,
                enabled=True,
                num_classes=self.config.num_classes
            )

        elif self.config.attack_type == AttackType.BACKDOOR:
            trigger = TriggerSpec(
                kind="corner_square",
                size=self.config.trigger_size,
                value=self.config.trigger_value
            )
            return make_backdoor_batch(
                batch,
                enabled=True,
                target_label=self.config.target_label,
                trigger=trigger
            )

        # No data poisoning for model poisoning attacks
        return batch

    def __len__(self):
        return len(self.original_loader)

    @property
    def dataset(self):
        return self.original_loader.dataset


class AttackingFlowerClient(NumPyClient):
    """
    Wrapper around DTRUST Flower client that injects attacks.

    This adapter wraps the original client and intercepts calls to
    fit() and evaluate() to apply attacks. The original client's
    behavior is preserved when attacks are disabled.

    Usage:
        # Wrap existing client
        original_client = DTRUSTFlowerClient(...)
        attacking_client = AttackingFlowerClient(original_client, attack_config)

        # Use attacking_client in place of original_client
        start_client(server_address, client=attacking_client)
    """

    def __init__(
        self,
        original_client: NumPyClient,
        attack_config: AttackConfig,
    ):
        """
        Initialize attacking client wrapper.

        Args:
            original_client: Original Flower client to wrap
            attack_config: Attack configuration
        """
        self.original_client = original_client
        self.config = attack_config

        # Wrap data loaders if data poisoning attack
        if attack_config.is_data_poisoning and attack_config.enabled:
            self._wrap_data_loaders()

        # Store original parameters for model poisoning
        self._original_parameters: Optional[List[np.ndarray]] = None
        self._server_parameters: Optional[List[np.ndarray]] = None

        logger.info(
            f"AttackingFlowerClient initialized with attack: "
            f"{attack_config.attack_type.value}, enabled={attack_config.enabled}"
        )

    def _wrap_data_loaders(self):
        """Wrap the original client's data loaders with attacking versions."""
        if hasattr(self.original_client, 'train_loader'):
            self.original_client.train_loader = AttackingDataLoader(
                self.original_client.train_loader,
                self.config
            )
            logger.info("Wrapped train_loader with AttackingDataLoader")

    def get_parameters(self, config: Dict[str, Any]) -> GetParametersRes:
        """Delegate to original client."""
        return self.original_client.get_parameters(config)

    def fit(self, fit_ins: FitIns) -> FitRes:
        """
        Perform training with optional attack injection.

        For data poisoning attacks, the data loader wrapper handles
        the poisoning. For model poisoning attacks, we modify the
        model updates after training.
        """
        # Store server parameters for model poisoning attacks
        self._server_parameters = parameters_to_ndarrays(fit_ins.parameters)

        # Perform original training (data poisoning is handled by loader wrapper)
        fit_res = self.original_client.fit(fit_ins)

        # Apply model poisoning if configured
        if self.config.enabled and self.config.is_model_poisoning:
            fit_res = self._apply_model_poisoning(fit_res)

        return fit_res

    def _apply_model_poisoning(self, fit_res: FitRes) -> FitRes:
        """Apply model poisoning to the training result."""
        if self.config.attack_type == AttackType.SCALING:
            return self._apply_scaling_attack(fit_res)

        # For Krum/Trim/Adaptive attacks, we need benign updates from other clients
        # These are typically applied at the strategy level, not client level
        # Here we just apply a simple scaling as a placeholder
        logger.warning(
            f"Model poisoning attack {self.config.attack_type.value} "
            "requires coordination with other clients. "
            "Consider using ModelPoisoningAdapter at strategy level."
        )
        return fit_res

    def _apply_scaling_attack(self, fit_res: FitRes) -> FitRes:
        """Apply scaling attack to model updates."""
        # Get the model update (delta = new_params - old_params)
        new_params = parameters_to_ndarrays(fit_res.parameters)

        if self._server_parameters is not None:
            # Compute delta and scale it
            scaled_params = []
            for new_p, old_p in zip(new_params, self._server_parameters):
                delta = new_p - old_p
                scaled_delta = delta * self.config.scale_factor
                scaled_params.append(old_p + scaled_delta)

            logger.info(
                f"Applied scaling attack with factor {self.config.scale_factor}"
            )

            return FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(scaled_params),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics
            )

        return fit_res

    def evaluate(self, evaluate_ins: EvaluateIns) -> EvaluateRes:
        """Delegate evaluation to original client."""
        return self.original_client.evaluate(evaluate_ins)

    # Proxy attributes to original client
    def __getattr__(self, name):
        """Proxy attribute access to original client."""
        return getattr(self.original_client, name)


def create_attacking_client(
    node_id: str,
    enclave_id: str,
    api_server_url: str,
    model_fn: Callable,
    data_loader_fn: Callable,
    config: Any,
    attack_config: AttackConfig,
) -> AttackingFlowerClient:
    """
    Factory function to create an attacking client.

    This function creates a standard DTRUST Flower client and wraps it
    with the attacking adapter.

    Args:
        node_id: Unique node identifier
        enclave_id: SGX enclave identifier
        api_server_url: DTRUST API server URL
        model_fn: Function that returns the model
        data_loader_fn: Function that returns (train_loader, test_loader)
        config: DTRUST configuration
        attack_config: Attack configuration

    Returns:
        AttackingFlowerClient wrapping a DTRUSTFlowerClient
    """
    # Import here to avoid circular imports
    from dtrust.fl.client import DTRUSTFlowerClient

    # Create original client
    original_client = DTRUSTFlowerClient(
        node_id=node_id,
        enclave_id=enclave_id,
        api_server_url=api_server_url,
        model_fn=model_fn,
        data_loader_fn=data_loader_fn,
        config=config
    )

    # Wrap with attacking adapter
    return AttackingFlowerClient(original_client, attack_config)
