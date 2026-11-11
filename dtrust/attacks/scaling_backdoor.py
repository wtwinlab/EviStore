"""
Scaling/Backdoor Attack - Targeted attack with trigger-embedded samples.

Reference: FLTrust paper backdoor attack implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class TriggerSpec:
    """
    A simple pattern trigger implementation.
    For MNIST/Fashion (1x28x28): a 3x3 white square at bottom-right.
    For CIFAR10 (3x32x32): a 4x4 white square at bottom-right.
    """

    kind: str = "corner_square"
    size: int = 3
    value: float = 1.0


def apply_corner_square_trigger(x: torch.Tensor, spec: TriggerSpec) -> torch.Tensor:
    x2 = x.clone()
    if x2.dim() != 4:
        raise ValueError("Expected batch tensor [B,C,H,W]")
    _, _, h, w = x2.shape
    s = spec.size
    x2[:, :, h - s : h, w - s : w] = spec.value
    return x2


def make_backdoor_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    enabled: bool,
    target_label: int,
    trigger: TriggerSpec,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if not enabled:
        return x, y
    x2 = apply_corner_square_trigger(x, trigger)
    y2 = torch.full_like(y, int(target_label))
    return x2, y2


def scale_delta(delta: torch.Tensor, enabled: bool, scale: float) -> torch.Tensor:
    if not enabled:
        return delta
    return delta * float(scale)
