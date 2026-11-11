"""
Trim Attack - Local Model Poisoning Attack optimized for Trimmed Mean and Median.

Reference: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust
Federated Learning" (USENIX Security 2020)

The attack framework aims to maximize s^T(g - g'), where:
- g is the global model update before attack
- g' is the global model update after attack
- s is the sign of g (direction we want to deviate from)

For Trimmed Mean/Median (coordinate-wise aggregation), the attack crafts
updates that:
1. Survive coordinate-wise trimming (not removed as outliers)
2. Push each coordinate as far as possible in the attack direction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import numpy as np


@dataclass(frozen=True)
class TrimAttackResult:
    """Result of Trim attack optimization."""
    poisoned_deltas: List[torch.Tensor]
    attack_type: str


def _compute_coordinate_bounds(
    benign_values: torch.Tensor,
    k: int,
    direction: Literal["lower", "upper"],
) -> torch.Tensor:
    """
    Compute the boundary values that would survive trimming.

    For each coordinate, find the value at the trim boundary.

    Args:
        benign_values: Shape [n_benign, d] - benign update values
        k: Trim parameter (remove k smallest and k largest)
        direction: "lower" for minimum surviving value, "upper" for maximum

    Returns:
        Tensor of shape [d] with boundary values
    """
    sorted_vals, _ = torch.sort(benign_values, dim=0)
    n = benign_values.size(0)

    if direction == "lower":
        # The k-th smallest value is the first one that survives
        # We want to be just above the (k-1)-th value to not be trimmed
        return sorted_vals[k, :]
    else:
        # The (n-k-1)-th value is the last one that survives
        return sorted_vals[n - k - 1, :]


def trim_attack(
    benign_deltas: List[torch.Tensor],
    num_malicious: int,
    k: int,
    device: torch.device,
    attack_mode: Literal["trimmed_mean", "median"] = "trimmed_mean",
) -> TrimAttackResult:
    """
    Craft poisoned local model updates optimized for Trim-mean/Median attack.

    The attack crafts malicious updates such that for each coordinate:
    - The value survives trimming (is not removed)
    - The value pushes the aggregated result in the attack direction

    For coordinate j with benign mean g_j:
    - If g_j > 0: push toward lower bound (make result smaller/negative)
    - If g_j < 0: push toward upper bound (make result larger/positive)

    Args:
        benign_deltas: List of benign client model updates
        num_malicious: Number of malicious clients (m)
        k: Trim parameter for trimmed mean
        device: Torch device
        attack_mode: "trimmed_mean" or "median"

    Returns:
        TrimAttackResult with poisoned deltas
    """
    if len(benign_deltas) == 0:
        raise ValueError("Need at least one benign delta")

    # Stack benign deltas: [n_benign, d]
    benign_stack = torch.stack([d.to(device) for d in benign_deltas], dim=0)
    n_benign = benign_stack.size(0)
    d = benign_stack.size(1)

    # Compute mean of benign updates
    benign_mean = benign_stack.mean(dim=0)

    # Determine attack direction for each coordinate
    # If benign_mean[j] > 0, attack should push down (use lower bound)
    # If benign_mean[j] < 0, attack should push up (use upper bound)
    attack_direction = -torch.sign(benign_mean)

    # Total number of clients after attack
    n_total = n_benign + num_malicious

    # Effective k for the combined dataset
    # Note: The trim parameter should ideally be set knowing the attack
    # Here we use the provided k
    effective_k = min(k, (n_total - 1) // 2)

    # For each coordinate, compute the extreme value that survives trimming
    # We want malicious values to be at the boundary of trimming

    # Sort benign values for each coordinate
    sorted_benign, _ = torch.sort(benign_stack, dim=0)            # [n_benign, d]
    # Per-coordinate population std (unbiased=False matches the original np.std).
    std_benign = benign_stack.std(dim=0, unbiased=False)         # [d]

    # Boundary indices that survive trimming (identical across coordinates).
    if n_benign > effective_k:
        upper_idx = n_benign - effective_k - 1
        lower_idx = effective_k
    else:
        upper_idx = n_benign - 1
        lower_idx = 0

    # Vectorized, device-safe coordinate-wise poison (was a per-coord numpy loop that
    # both broke on CUDA — numpy scalar -> cuda tensor — and was pathologically slow):
    #   push up   -> boundary value + 3 std   (where benign_mean < 0)
    #   push down -> boundary value - 3 std   (where benign_mean > 0)
    up_val = sorted_benign[upper_idx] + std_benign * 3.0          # [d]
    down_val = sorted_benign[lower_idx] - std_benign * 3.0        # [d]
    poisoned_values = torch.where(attack_direction > 0, up_val, down_val)

    # All malicious clients send the same poisoned update
    poisoned_deltas = [poisoned_values.clone() for _ in range(num_malicious)]

    return TrimAttackResult(
        poisoned_deltas=poisoned_deltas,
        attack_type=attack_mode,
    )


def trim_attack_optimized(
    benign_deltas: List[torch.Tensor],
    num_malicious: int,
    k: int,
    device: torch.device,
    epsilon: float = 0.01,
) -> TrimAttackResult:
    """
    Optimized Trim attack using the Fang et al. formulation.

    The attack solves:
        max s^T(g - g')

    For trimmed mean, this becomes a coordinate-wise optimization:
    For each coordinate j, craft malicious value z such that:
    - z survives trimming
    - z maximizes deviation in the -sign(g_j) direction

    Args:
        benign_deltas: List of benign client model updates
        num_malicious: Number of malicious clients (m)
        k: Trim parameter
        device: Torch device
        epsilon: Small constant for numerical stability

    Returns:
        TrimAttackResult with optimized poisoned deltas
    """
    if len(benign_deltas) == 0:
        raise ValueError("Need at least one benign delta")

    # Stack benign deltas
    benign_stack = torch.stack([d.to(device) for d in benign_deltas], dim=0)
    n_benign = benign_stack.size(0)
    d = benign_stack.size(1)

    # Compute statistics of benign updates
    benign_mean = benign_stack.mean(dim=0)
    benign_std = benign_stack.std(dim=0) + epsilon

    # Attack direction: opposite of benign mean's sign
    s = torch.sign(benign_mean)

    # For the Fang et al. attack, the optimal poisoned update is:
    # g'_m = g_bar - lambda * s
    # where lambda is chosen to maximize deviation while surviving trim

    # Compute the range of benign values for each coordinate
    benign_min = benign_stack.min(dim=0).values
    benign_max = benign_stack.max(dim=0).values

    # Compute poisoned value for each coordinate
    poisoned_values = torch.zeros(d, device=device)

    # For coordinates where s[j] > 0 (benign_mean > 0):
    #   Attack wants to push result negative, so use low values
    # For coordinates where s[j] < 0 (benign_mean < 0):
    #   Attack wants to push result positive, so use high values
    # For coordinates where s[j] = 0:
    #   Use benign mean (no clear direction)

    # Scaling factor: how far to deviate
    # Use a multiple of std to be aggressive but not get trimmed
    scale_factor = 3.0

    for j in range(d):
        if s[j] > 0:
            # Push down: be at lower end of distribution
            poisoned_values[j] = benign_mean[j] - scale_factor * benign_std[j]
        elif s[j] < 0:
            # Push up: be at upper end of distribution
            poisoned_values[j] = benign_mean[j] + scale_factor * benign_std[j]
        else:
            # No clear direction
            poisoned_values[j] = benign_mean[j]

    # All malicious clients send the same poisoned update
    poisoned_deltas = [poisoned_values.clone() for _ in range(num_malicious)]

    return TrimAttackResult(
        poisoned_deltas=poisoned_deltas,
        attack_type="optimized",
    )


def trim_attack_full_knowledge(
    all_deltas: List[torch.Tensor],
    malicious_indices: List[int],
    k: int,
    device: torch.device,
    attack_mode: Literal["trimmed_mean", "median"] = "trimmed_mean",
) -> List[torch.Tensor]:
    """
    Full-knowledge Trim attack where attacker knows all client updates.

    Args:
        all_deltas: All client deltas (including placeholders for malicious)
        malicious_indices: Indices of malicious clients
        k: Trim parameter
        device: Torch device
        attack_mode: Type of aggregation being attacked

    Returns:
        List of poisoned deltas for malicious clients
    """
    n = len(all_deltas)
    m = len(malicious_indices)
    benign_indices = [i for i in range(n) if i not in malicious_indices]

    # Get benign deltas
    benign_deltas = [all_deltas[i].to(device) for i in benign_indices]

    if len(benign_deltas) == 0:
        # All clients are malicious
        mean_delta = torch.stack([d.to(device) for d in all_deltas], dim=0).mean(dim=0)
        return [-mean_delta for _ in range(m)]

    result = trim_attack_optimized(
        benign_deltas=benign_deltas,
        num_malicious=m,
        k=k,
        device=device,
    )

    return result.poisoned_deltas
