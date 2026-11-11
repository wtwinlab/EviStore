"""
Krum Attack - Local Model Poisoning Attack optimized for Krum aggregation.

Reference: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust
Federated Learning" (USENIX Security 2020)

The attack framework aims to maximize s^T(g - g'), where:
- g is the global model update before attack
- g' is the global model update after attack
- s is the sign of g (direction we want to deviate from)

For Krum, malicious clients craft updates that:
1. Are close to each other (low pairwise distances)
2. Deviate the model in the opposite direction of the correct update
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import numpy as np


@dataclass(frozen=True)
class KrumAttackResult:
    """Result of Krum attack optimization."""
    poisoned_deltas: List[torch.Tensor]
    lambda_opt: float
    attack_successful: bool


def _compute_krum_scores(deltas: List[torch.Tensor], f: int) -> List[float]:
    """
    Compute Krum scores for each delta.
    Score_i = sum of squared distances to the closest (n-f-2) deltas.
    """
    X = torch.stack(deltas, dim=0)
    n = X.size(0)
    m = n - f - 2

    with torch.no_grad():
        # Pairwise squared distances; sort/sum on device, one sync at the end
        # instead of n GPU->CPU copies.
        dist = torch.cdist(X, X, p=2) ** 2
        sorted_d, _ = torch.sort(dist, dim=1)  # column 0 is 0 (self-distance)
        scores = sorted_d[:, 1 : 1 + m].sum(dim=1)
    return scores.tolist()


def krum_attack(
    benign_deltas: List[torch.Tensor],
    num_malicious: int,
    f: int,
    device: torch.device,
    lambda_init: float = 0.1,
    lambda_max: float = 100.0,
    step_size: float = 0.1,
    max_iters: int = 1000,
) -> KrumAttackResult:
    """
    Craft poisoned local model updates optimized for Krum attack.

    The attack crafts m malicious updates as: g'_i = -lambda * s
    where s = sign(mean of benign updates).

    Lambda is searched to find the largest value such that a malicious
    update is still selected by Krum.

    Args:
        benign_deltas: List of benign client model updates
        num_malicious: Number of malicious clients (m)
        f: Krum parameter (assumed number of Byzantine clients)
        device: Torch device
        lambda_init: Initial lambda value for line search
        lambda_max: Maximum lambda to try
        step_size: Step size for lambda search
        max_iters: Maximum iterations for search

    Returns:
        KrumAttackResult with poisoned deltas and metadata
    """
    if len(benign_deltas) == 0:
        raise ValueError("Need at least one benign delta")

    # Compute the average of benign updates
    benign_stack = torch.stack([d.to(device) for d in benign_deltas], dim=0)
    benign_mean = benign_stack.mean(dim=0)

    # Attack direction: negative of the sign of mean benign update
    # This pushes the model in the opposite direction
    s = torch.sign(benign_mean)

    # Binary search for optimal lambda
    # We want the largest lambda such that a malicious update is selected
    lambda_low = 0.0
    lambda_high = lambda_max
    lambda_opt = lambda_init
    best_lambda = lambda_init
    attack_successful = False

    for _ in range(max_iters):
        lambda_mid = (lambda_low + lambda_high) / 2.0

        # Craft poisoned updates: g'_i = -lambda * s
        # All malicious clients send the same poisoned update
        poisoned = -lambda_mid * s

        # Combine benign and malicious deltas
        all_deltas = [poisoned.clone() for _ in range(num_malicious)] + \
                     [d.to(device) for d in benign_deltas]

        # Check if attack succeeds (malicious update is selected by Krum)
        scores = _compute_krum_scores(all_deltas, f)
        selected_idx = int(np.argmin(scores))

        if selected_idx < num_malicious:
            # Attack successful at this lambda, try larger
            attack_successful = True
            best_lambda = lambda_mid
            lambda_low = lambda_mid
        else:
            # Attack failed, try smaller lambda
            lambda_high = lambda_mid

        # Convergence check
        if lambda_high - lambda_low < 1e-6:
            break

    # Generate final poisoned deltas
    poisoned_deltas = [-best_lambda * s for _ in range(num_malicious)]

    return KrumAttackResult(
        poisoned_deltas=poisoned_deltas,
        lambda_opt=best_lambda,
        attack_successful=attack_successful,
    )


def krum_attack_full_knowledge(
    all_deltas: List[torch.Tensor],
    malicious_indices: List[int],
    f: int,
    device: torch.device,
    epsilon: float = 1e-3,
    max_iters: int = 100,
) -> List[torch.Tensor]:
    """
    Full-knowledge Krum attack where attacker knows all client updates.

    This implements the optimization from Fang et al.:
    max_{g'_1,...,g'_m} s^T(g - g')

    where g is computed by Krum on original deltas and g' is computed
    by Krum after replacing benign with malicious updates.

    Args:
        all_deltas: All client deltas (including placeholders for malicious)
        malicious_indices: Indices of malicious clients
        f: Krum parameter
        device: Torch device
        epsilon: Perturbation magnitude for deviation
        max_iters: Maximum optimization iterations

    Returns:
        List of poisoned deltas for malicious clients
    """
    n = len(all_deltas)
    m = len(malicious_indices)
    benign_indices = [i for i in range(n) if i not in malicious_indices]

    # Get benign deltas
    benign_deltas = [all_deltas[i].to(device) for i in benign_indices]

    if len(benign_deltas) == 0:
        # All clients are malicious - just return negative mean
        mean_delta = torch.stack([d.to(device) for d in all_deltas], dim=0).mean(dim=0)
        return [-mean_delta for _ in range(m)]

    # Compute mean of benign updates as reference
    benign_mean = torch.stack(benign_deltas, dim=0).mean(dim=0)

    # Attack direction
    s = torch.sign(benign_mean)

    # Use the basic attack strategy
    result = krum_attack(
        benign_deltas=benign_deltas,
        num_malicious=m,
        f=f,
        device=device,
    )

    return result.poisoned_deltas
