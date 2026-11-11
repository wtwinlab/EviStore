"""
Label Flipping Attack - Data poisoning attack that flips labels.

Reference: FLTrust paper (Section VI-A.2)
"""

from __future__ import annotations

from typing import Tuple

import torch


def flip_labels_l_to_m_minus_l_minus_1(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Paper (Section VI-A.2): flip label l to M-l-1.
    """
    return (num_classes - 1) - y


def maybe_label_flip_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    enabled: bool,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if not enabled:
        return x, y
    return x, flip_labels_l_to_m_minus_l_minus_1(y, num_classes=num_classes)
