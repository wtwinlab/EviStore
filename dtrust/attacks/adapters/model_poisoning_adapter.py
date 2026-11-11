"""
Model Poisoning Adapter - Coordinates model poisoning attacks.

This adapter provides functions to apply coordinated model poisoning
attacks (Krum, Trim, Adaptive) that require knowledge of other clients'
updates. It can be used standalone or integrated at the strategy level.

Key Design Principles:
1. Zero modification to original DTRUST code
2. Works with model updates as numpy arrays or torch tensors
3. Can be used for evaluation/testing without affecting production code
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import logging

import torch
import numpy as np

from dtrust.attacks.adapters.attack_config import AttackConfig, AttackType
from dtrust.attacks.krum_attack import krum_attack, krum_attack_full_knowledge
from dtrust.attacks.trim_attack import trim_attack_optimized, trim_attack_full_knowledge
from dtrust.attacks.adaptive_attack import adaptive_attack, adaptive_attack_full_knowledge
from dtrust.attacks.scaling_backdoor import scale_delta


logger = logging.getLogger(__name__)


class ModelPoisoningAdapter:
    """
    Adapter for coordinated model poisoning attacks.

    This adapter is designed to work at the aggregation level where
    all client updates are available. It computes poisoned updates
    for malicious clients based on the benign clients' updates.

    Usage:
        adapter = ModelPoisoningAdapter(attack_config)

        # Get poisoned updates for malicious clients
        poisoned_updates = adapter.compute_poisoned_updates(
            benign_updates=[client1_update, client2_update, ...],
            server_update=server_model_update,  # For adaptive attack
            num_malicious=2,
        )
    """

    def __init__(self, attack_config: AttackConfig):
        """
        Initialize model poisoning adapter.

        Args:
            attack_config: Attack configuration
        """
        self.config = attack_config
        self.device = torch.device(attack_config.device)

        logger.info(
            f"ModelPoisoningAdapter initialized with attack: "
            f"{attack_config.attack_type.value}"
        )

    def compute_poisoned_updates(
        self,
        benign_updates: List[List[np.ndarray]],
        server_update: Optional[List[np.ndarray]] = None,
        num_malicious: int = 1,
    ) -> List[List[np.ndarray]]:
        """
        Compute poisoned model updates for malicious clients.

        Args:
            benign_updates: List of benign client updates (each is a list of layer arrays)
            server_update: Server model update (for adaptive attack)
            num_malicious: Number of malicious updates to generate

        Returns:
            List of poisoned updates (each is a list of layer arrays)
        """
        if not self.config.enabled:
            logger.warning("Attack is disabled, returning empty list")
            return []

        if self.config.attack_type == AttackType.SCALING:
            return self._apply_scaling_attack(benign_updates, num_malicious)

        elif self.config.attack_type == AttackType.KRUM:
            return self._apply_krum_attack(benign_updates, num_malicious)

        elif self.config.attack_type == AttackType.TRIM:
            return self._apply_trim_attack(benign_updates, num_malicious)

        elif self.config.attack_type == AttackType.ADAPTIVE:
            if server_update is None:
                raise ValueError("Adaptive attack requires server_update")
            return self._apply_adaptive_attack(
                benign_updates, server_update, num_malicious
            )

        else:
            logger.warning(
                f"Attack type {self.config.attack_type.value} "
                "is not a model poisoning attack"
            )
            return []

    def _flatten_updates(
        self, updates: List[List[np.ndarray]]
    ) -> List[torch.Tensor]:
        """Flatten layer-wise updates to single vectors."""
        flattened = []
        for update in updates:
            flat = np.concatenate([arr.flatten() for arr in update])
            flattened.append(torch.from_numpy(flat).float().to(self.device))
        return flattened

    def _unflatten_update(
        self,
        flat_update: torch.Tensor,
        shapes: List[Tuple[int, ...]],
    ) -> List[np.ndarray]:
        """Unflatten a vector back to layer-wise arrays."""
        flat_np = flat_update.cpu().numpy()
        unflattened = []
        offset = 0
        for shape in shapes:
            size = np.prod(shape)
            arr = flat_np[offset:offset + size].reshape(shape)
            unflattened.append(arr)
            offset += size
        return unflattened

    def _get_shapes(self, update: List[np.ndarray]) -> List[Tuple[int, ...]]:
        """Get shapes of each layer in an update."""
        return [arr.shape for arr in update]

    def _apply_scaling_attack(
        self,
        benign_updates: List[List[np.ndarray]],
        num_malicious: int,
    ) -> List[List[np.ndarray]]:
        """Apply scaling attack - scale up benign updates."""
        if not benign_updates:
            return []

        # Use mean of benign updates as base, then scale
        shapes = self._get_shapes(benign_updates[0])
        mean_update = []

        for layer_idx in range(len(benign_updates[0])):
            layer_mean = np.mean(
                [update[layer_idx] for update in benign_updates],
                axis=0
            )
            # Scale the update (negative to push in opposite direction)
            scaled = -layer_mean * self.config.scale_factor
            mean_update.append(scaled)

        logger.info(
            f"Applied scaling attack with factor {self.config.scale_factor}"
        )

        return [mean_update.copy() for _ in range(num_malicious)]

    def _apply_krum_attack(
        self,
        benign_updates: List[List[np.ndarray]],
        num_malicious: int,
    ) -> List[List[np.ndarray]]:
        """Apply Krum attack."""
        if not benign_updates:
            return []

        shapes = self._get_shapes(benign_updates[0])
        flat_benign = self._flatten_updates(benign_updates)

        result = krum_attack(
            benign_deltas=flat_benign,
            num_malicious=num_malicious,
            f=self.config.krum_f,
            device=self.device,
        )

        logger.info(
            f"Krum attack: lambda_opt={result.lambda_opt:.4f}, "
            f"successful={result.attack_successful}"
        )

        # Unflatten poisoned updates
        poisoned_updates = []
        for flat_poisoned in result.poisoned_deltas:
            poisoned_updates.append(self._unflatten_update(flat_poisoned, shapes))

        return poisoned_updates

    def _apply_trim_attack(
        self,
        benign_updates: List[List[np.ndarray]],
        num_malicious: int,
    ) -> List[List[np.ndarray]]:
        """Apply Trim attack."""
        if not benign_updates:
            return []

        shapes = self._get_shapes(benign_updates[0])
        flat_benign = self._flatten_updates(benign_updates)

        result = trim_attack_optimized(
            benign_deltas=flat_benign,
            num_malicious=num_malicious,
            k=self.config.trim_k,
            device=self.device,
        )

        logger.info(f"Trim attack: type={result.attack_type}")

        # Unflatten poisoned updates
        poisoned_updates = []
        for flat_poisoned in result.poisoned_deltas:
            poisoned_updates.append(self._unflatten_update(flat_poisoned, shapes))

        return poisoned_updates

    def _apply_adaptive_attack(
        self,
        benign_updates: List[List[np.ndarray]],
        server_update: List[np.ndarray],
        num_malicious: int,
    ) -> List[List[np.ndarray]]:
        """Apply Adaptive attack against FLTrust."""
        if not benign_updates:
            return []

        shapes = self._get_shapes(benign_updates[0])
        flat_benign = self._flatten_updates(benign_updates)

        # Flatten server update
        flat_server = np.concatenate([arr.flatten() for arr in server_update])
        flat_server_tensor = torch.from_numpy(flat_server).float().to(self.device)

        params = self.config.adaptive_params
        result = adaptive_attack(
            benign_deltas=flat_benign,
            server_delta=flat_server_tensor,
            num_malicious=num_malicious,
            device=self.device,
            sigma=params.get("sigma", 0.5),
            gamma=params.get("gamma", 0.005),
            eta=params.get("eta", 0.01),
            V=int(params.get("V", 10)),
            Q=int(params.get("Q", 10)),
        )

        logger.info(
            f"Adaptive attack: final_objective={result.final_objective:.4f}, "
            f"iterations={result.num_iterations}"
        )

        # Unflatten poisoned updates
        poisoned_updates = []
        for flat_poisoned in result.poisoned_deltas:
            poisoned_updates.append(self._unflatten_update(flat_poisoned, shapes))

        return poisoned_updates


def apply_model_poisoning(
    all_updates: List[List[np.ndarray]],
    malicious_indices: List[int],
    attack_config: AttackConfig,
    server_update: Optional[List[np.ndarray]] = None,
) -> List[List[np.ndarray]]:
    """
    Convenience function to apply model poisoning to a set of updates.

    This function takes all client updates and replaces the updates at
    malicious_indices with poisoned versions.

    Args:
        all_updates: All client updates (will be modified in place)
        malicious_indices: Indices of malicious clients
        attack_config: Attack configuration
        server_update: Server model update (for adaptive attack)

    Returns:
        Modified all_updates with poisoned updates at malicious indices
    """
    if not attack_config.enabled or not attack_config.is_model_poisoning:
        return all_updates

    # Get benign updates
    benign_indices = [i for i in range(len(all_updates)) if i not in malicious_indices]
    benign_updates = [all_updates[i] for i in benign_indices]

    # Create adapter and compute poisoned updates
    adapter = ModelPoisoningAdapter(attack_config)
    poisoned_updates = adapter.compute_poisoned_updates(
        benign_updates=benign_updates,
        server_update=server_update,
        num_malicious=len(malicious_indices),
    )

    # Replace malicious updates
    for idx, mal_idx in enumerate(malicious_indices):
        if idx < len(poisoned_updates):
            all_updates[mal_idx] = poisoned_updates[idx]

    return all_updates
