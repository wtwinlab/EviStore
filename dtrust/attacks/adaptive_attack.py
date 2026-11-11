"""
Adaptive Attack against FLTrust - Local Model Poisoning Attack specifically
designed to attack the FLTrust aggregation rule.

Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning via
Trust Bootstrapping" (NDSS 2021), Section V

The attack uses zeroth-order optimization to solve:
    max_{e'_1,...,e'_m} h(e'_1, ..., e'_m)

where h is the objective function from Equation (10) that measures how much
the attack deviates the global model update from its correct direction.

Algorithm 3 from the paper:
1. Initialize e'_i using Trim attack
2. For V iterations over all malicious clients:
   For each malicious client i, for Q iterations:
   - Sample u ~ N(0, σ²I)
   - Compute gradient: ∇h ≈ (h(e'_i + γu) - h(e'_i)) / γ * u
   - Update: e'_i = e'_i + η * ∇h
   - Normalize: e'_i = e'_i / ||e'_i||

Parameters from paper: σ²=0.5, γ=0.005, η=0.01, V=Q=10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np

from dtrust.attacks.trim_attack import trim_attack_optimized


@dataclass(frozen=True)
class AdaptiveAttackResult:
    """Result of adaptive attack optimization."""
    poisoned_deltas: List[torch.Tensor]
    final_objective: float
    num_iterations: int


def _compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors."""
    dot = torch.dot(a, b)
    norm_a = torch.sqrt(torch.sum(a * a) + eps)
    norm_b = torch.sqrt(torch.sum(b * b) + eps)
    return float((dot / (norm_a * norm_b)).item())


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vector to unit length."""
    norm = torch.sqrt(torch.sum(x * x) + eps)
    return x / norm


def _compute_objective(
    e_primes: List[torch.Tensor],  # Unit vectors for malicious clients
    e_benign: List[torch.Tensor],  # Unit vectors for benign clients
    e_0: torch.Tensor,             # Unit vector for server update
    c_benign: List[float],         # Cosine similarities for benign clients
    s: torch.Tensor,               # Sign of original global update
    g0_norm: float,                # Norm of server update
    device: torch.device,
) -> float:
    """
    Compute objective function h(e'_1, ..., e'_m) from Equation (10).

    h = ||g_0|| * s^T * [
        Σ_i (ReLU(c_i) / Σ_j ReLU(c_j)) * e_i                    (term 1: original)
        - Σ_i (ReLU(<e'_i, e_0>) / denom) * e'_i                   (term 2: malicious)
        - Σ_i (ReLU(c_i) / denom) * e_i                            (term 3: benign after attack)
    ]

    where denom = Σ_j ReLU(<e'_j, e_0>) + Σ_j ReLU(c_j)
    """
    m = len(e_primes)
    n_benign = len(e_benign)

    # Compute cosine similarities for malicious (with server update)
    c_primes = [_compute_cosine_similarity(e_p, e_0) for e_p in e_primes]

    # ReLU clipping
    ts_primes = [max(c, 0.0) for c in c_primes]  # Trust scores for malicious
    ts_benign = [max(c, 0.0) for c in c_benign]  # Trust scores for benign

    # Denominators
    sum_ts_benign = sum(ts_benign)
    sum_ts_primes = sum(ts_primes)
    sum_ts_all = sum_ts_benign + sum_ts_primes

    # Term 1: Original global update direction (before attack)
    if sum_ts_benign > 0:
        term1 = torch.zeros_like(s)
        for i, (e_b, ts_b) in enumerate(zip(e_benign, ts_benign)):
            if ts_b > 0:
                term1 = term1 + (ts_b / sum_ts_benign) * e_b.to(device)
    else:
        term1 = torch.zeros_like(s)

    # Term 2: Malicious contribution after attack
    if sum_ts_all > 0:
        term2 = torch.zeros_like(s)
        for e_p, ts_p in zip(e_primes, ts_primes):
            if ts_p > 0:
                term2 = term2 + (ts_p / sum_ts_all) * e_p.to(device)
    else:
        term2 = torch.zeros_like(s)

    # Term 3: Benign contribution after attack (with new denominator)
    if sum_ts_all > 0:
        term3 = torch.zeros_like(s)
        for e_b, ts_b in zip(e_benign, ts_benign):
            if ts_b > 0:
                term3 = term3 + (ts_b / sum_ts_all) * e_b.to(device)
    else:
        term3 = torch.zeros_like(s)

    # Objective: ||g_0|| * s^T * (term1 - term2 - term3)
    diff = term1 - term2 - term3
    h = g0_norm * torch.dot(s, diff)

    return float(h.item())


def _objective_vec(
    E_primes: torch.Tensor,      # [m, d] malicious unit vectors (rows)
    e_0: torch.Tensor,           # [d] unit server direction
    s: torch.Tensor,             # [d] sign vector
    g0_norm: float,
    sum_ts_benign: torch.Tensor, # scalar: Σ relu(c_benign)
    ts_benign_se_sum: torch.Tensor,  # scalar: Σ relu(c_benign)·(s·e_benign)
    s_term1: torch.Tensor,       # scalar: s^T term1  (fixed)
    eps: float = 1e-12,
) -> torch.Tensor:
    """Vectorized form of ``_compute_objective`` returning a 0-dim tensor (no GPU
    sync). Math is identical term-for-term — the benign quantities (term1, c_benign,
    s·e_benign) are fixed across the optimization loop and precomputed by the caller;
    only the malicious E_primes change. Reduces the per-call cost from an O(n_benign)
    Python loop of ``.item()`` syncs to two matmuls.

        h = ||g0|| * sᵀ(term1 - term2 - term3),
        term2 = Σ_i (relu(c'_i)/denom)·e'_i,   term3 = Σ_b (relu(c_b)/denom)·e_b,
        denom = Σ_b relu(c_b) + Σ_i relu(c'_i).
    """
    row_norm = E_primes.norm(dim=1).clamp(min=eps)          # [m]
    c_primes = (E_primes @ e_0) / row_norm                  # [m] cosine to e_0
    ts_primes = c_primes.clamp(min=0.0)                     # relu
    sum_ts_all = sum_ts_benign + ts_primes.sum()
    se_primes = E_primes @ s                                # [m] raw s·e'_i
    if float(sum_ts_all) > 0:
        s_term2 = (ts_primes * se_primes).sum() / sum_ts_all
        s_term3 = ts_benign_se_sum / sum_ts_all
    else:
        s_term2 = E_primes.new_zeros(())
        s_term3 = E_primes.new_zeros(())
    return g0_norm * (s_term1 - s_term2 - s_term3)


def adaptive_attack(
    benign_deltas: List[torch.Tensor],
    server_delta: torch.Tensor,
    num_malicious: int,
    device: torch.device,
    sigma: float = 0.5,
    gamma: float = 0.005,
    eta: float = 0.01,
    V: int = 10,
    Q: int = 10,
    k_trim: Optional[int] = None,
) -> AdaptiveAttackResult:
    """
    Adaptive attack against FLTrust using zeroth-order optimization.

    Implements Algorithm 3 from the paper.

    Args:
        benign_deltas: List of benign client model updates
        server_delta: Server model update (g_0)
        num_malicious: Number of malicious clients (m)
        device: Torch device
        sigma: Noise std for zeroth-order gradient estimation (paper: sqrt(0.5))
        gamma: Smoothing parameter for gradient estimation (paper: 0.005)
        eta: Learning rate / step size (paper: 0.01)
        V: Outer loop iterations (paper: 10)
        Q: Inner loop iterations per malicious client (paper: 10)
        k_trim: Trim parameter for initialization (defaults to num_malicious)

    Returns:
        AdaptiveAttackResult with poisoned deltas and metadata
    """
    if len(benign_deltas) == 0:
        raise ValueError("Need at least one benign delta")

    n_benign = len(benign_deltas)
    m = num_malicious

    # Move to device
    g_0 = server_delta.to(device)
    g_benign = [d.to(device) for d in benign_deltas]

    # Compute norms and unit vectors
    g0_norm = float(torch.sqrt(torch.sum(g_0 * g_0)).item())
    if g0_norm < 1e-12:
        # Server update is zero - return zeros
        return AdaptiveAttackResult(
            poisoned_deltas=[torch.zeros_like(g_0) for _ in range(m)],
            final_objective=0.0,
            num_iterations=0,
        )

    e_0 = g_0 / g0_norm

    # Unit vectors and cosine similarities for benign clients
    e_benign = []
    c_benign = []
    for g_b in g_benign:
        norm_b = float(torch.sqrt(torch.sum(g_b * g_b)).item())
        if norm_b < 1e-12:
            e_benign.append(torch.zeros_like(g_0))
            c_benign.append(0.0)
        else:
            e_b = g_b / norm_b
            e_benign.append(e_b)
            c_benign.append(_compute_cosine_similarity(e_b, e_0))

    # Compute sign of original global update (for benign only)
    # This is s = sign(g) where g is FLTrust aggregation of benign updates
    ts_benign = [max(c, 0.0) for c in c_benign]
    sum_ts = sum(ts_benign)
    if sum_ts > 0:
        g_original = torch.zeros_like(g_0)
        for e_b, ts_b in zip(e_benign, ts_benign):
            g_original = g_original + ts_b * g0_norm / sum_ts * e_b
        s = torch.sign(g_original)
    else:
        s = torch.zeros_like(g_0)

    # Step 1: Initialize e'_i using Trim attack
    if k_trim is None:
        k_trim = m

    try:
        trim_result = trim_attack_optimized(
            benign_deltas=benign_deltas,
            num_malicious=m,
            k=k_trim,
            device=device,
        )
        e_primes = []
        for delta in trim_result.poisoned_deltas:
            norm_p = float(torch.sqrt(torch.sum(delta * delta)).item())
            if norm_p < 1e-12:
                # Use negative of e_0 as default attack direction
                e_primes.append(-e_0.clone())
            else:
                e_primes.append(delta / norm_p)
    except Exception:
        # Fallback: initialize with negative of e_0
        e_primes = [-e_0.clone() for _ in range(m)]

    # Step 2: Zeroth-order optimization (Algorithm 3)
    d = e_0.numel()  # Dimension of the update vector

    # Precompute fixed benign quantities (constant across the optimization loop;
    # only the malicious vectors change). This is what turns the per-call objective
    # into two matmuls instead of an O(n_benign) Python loop of .item() GPU syncs.
    if e_benign:
        E_benign = torch.stack([e.to(device) for e in e_benign])      # [n_benign, d]
        ts_benign = torch.tensor(c_benign, device=device, dtype=e_0.dtype).clamp(min=0.0)
        se_benign = E_benign @ s                                       # [n_benign]
        ts_benign_se_sum = (ts_benign * se_benign).sum()
        sum_ts_benign = ts_benign.sum()
    else:
        ts_benign_se_sum = e_0.new_zeros(())
        sum_ts_benign = e_0.new_zeros(())
    s_term1 = ts_benign_se_sum / sum_ts_benign if float(sum_ts_benign) > 0 else e_0.new_zeros(())

    E_primes = torch.stack(e_primes)                                  # [m, d], mutated in place
    obj_args = (e_0, s, g0_norm, sum_ts_benign, ts_benign_se_sum, s_term1)

    total_iters = 0
    for v in range(V):
        for i in range(m):
            for t in range(Q):
                u = torch.randn(d, device=device) * sigma            # N(0, σ²I)
                old_row = E_primes[i].clone()
                # h(e'_i + γu): perturb row i in place, evaluate, restore
                E_primes[i] = _normalize(old_row + gamma * u)
                h_plus = _objective_vec(E_primes, *obj_args)
                E_primes[i] = old_row
                h_current = _objective_vec(E_primes, *obj_args)
                # Gradient estimate ∇h ≈ (h_plus - h_current)/γ · u, projected update
                grad = ((h_plus - h_current) / gamma) * u
                E_primes[i] = _normalize(old_row + eta * grad)
                total_iters += 1

    e_primes = [E_primes[i] for i in range(m)]

    # Convert unit vectors back to poisoned deltas (||g_0|| · e'_i)
    poisoned_deltas = [g0_norm * e_p for e_p in e_primes]

    # Compute final objective
    final_obj = float(_objective_vec(E_primes, *obj_args).item())

    return AdaptiveAttackResult(
        poisoned_deltas=poisoned_deltas,
        final_objective=final_obj,
        num_iterations=total_iters,
    )


def adaptive_attack_full_knowledge(
    all_deltas: List[torch.Tensor],
    server_delta: torch.Tensor,
    malicious_indices: List[int],
    device: torch.device,
    sigma: float = 0.5,
    gamma: float = 0.005,
    eta: float = 0.01,
    V: int = 10,
    Q: int = 10,
) -> List[torch.Tensor]:
    """
    Full-knowledge adaptive attack where attacker knows all client updates.

    Args:
        all_deltas: All client deltas (including placeholders for malicious)
        server_delta: Server model update
        malicious_indices: Indices of malicious clients
        device: Torch device
        sigma, gamma, eta, V, Q: Attack parameters

    Returns:
        List of poisoned deltas for malicious clients
    """
    n = len(all_deltas)
    m = len(malicious_indices)
    benign_indices = [i for i in range(n) if i not in malicious_indices]

    # Get benign deltas
    benign_deltas = [all_deltas[i].to(device) for i in benign_indices]

    if len(benign_deltas) == 0:
        # All clients are malicious - use negative of server delta
        return [-server_delta.to(device) for _ in range(m)]

    result = adaptive_attack(
        benign_deltas=benign_deltas,
        server_delta=server_delta,
        num_malicious=m,
        device=device,
        sigma=sigma,
        gamma=gamma,
        eta=eta,
        V=V,
        Q=Q,
    )

    return result.poisoned_deltas
