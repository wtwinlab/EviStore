"""
Min-Max (AGR-agnostic) model-poisoning attack.

Reference: Shejwalkar & Houmansadr, "Manipulating the Byzantine: Optimizing Model
Poisoning Attacks and Defenses for Federated Learning" (NDSS 2021).

The min-max attack crafts a SINGLE malicious update m = mu + gamma * pert, where mu is
the benign mean and pert is a perturbation direction (here the inverse unit standard
deviation, which carries components ORTHOGONAL to mu — important against cosine-based
defenses like FLTrust: a perturbation parallel to mu would just be re-scaled away by
FLTrust's clip-to-||g0||, doing no damage). gamma is maximized by binary search subject
to the min-max constraint:

    max_i || m - benign_i ||  <=  max_{i,j} || benign_i - benign_j ||

i.e. the malicious update stays no farther from any benign update than two benign
updates are from each other — so it is not a distance/norm outlier (evades Krum,
trimmed-mean, norm-bounding) while perturbing as much as the benign spread allows.

This is the standard "strong adaptive / AGR-agnostic" attack; it is white-box on the
update vectors but blind to the COMPUTATION — so a CFI-attested defense (AGRA) excludes
it regardless of how it is crafted, while update-space defenses must let it through.
"""

from __future__ import annotations

from typing import List

import torch


def min_max_attack_full_knowledge(
    all_deltas: List[torch.Tensor],
    malicious_indices: List[int],
    num_malicious: int,
    device: torch.device,
    pert_mode: str = "std",          # "std" (orthogonal-bearing) | "sign" | "unit"
) -> List[torch.Tensor]:
    """Craft the min-max malicious update; all malicious clients send the same vector."""
    mal_set = set(malicious_indices)
    benign = [all_deltas[i].to(device) for i in range(len(all_deltas)) if i not in mal_set]
    if len(benign) == 0:
        mean = torch.stack([d.to(device) for d in all_deltas], dim=0).mean(dim=0)
        return [(-mean).clone() for _ in range(num_malicious)]

    B = torch.stack(benign, dim=0)                       # [n_b, d]
    mu = B.mean(dim=0)                                   # [d]

    if pert_mode == "sign":
        pert = -torch.sign(mu)
    elif pert_mode == "unit":
        pert = -mu / (mu.norm() + 1e-12)
    else:  # "std": inverse standard-deviation direction (carries components ⊥ mu)
        std = B.std(dim=0, unbiased=False)
        pert = -std / (std.norm() + 1e-12)

    # min-max threshold: largest pairwise distance among benign updates
    max_dist = torch.cdist(B, B).max()

    # binary search the largest gamma keeping m within the benign spread
    lo, hi = 0.0, float((max_dist / (pert.norm() + 1e-12)).item()) * 2.0 + 1.0
    for _ in range(30):
        gamma = 0.5 * (lo + hi)
        m = mu + gamma * pert
        if (m.unsqueeze(0) - B).norm(dim=1).max() <= max_dist:
            lo = gamma
        else:
            hi = gamma

    m = mu + lo * pert
    return [m.clone() for _ in range(num_malicious)]
