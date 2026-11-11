"""DP-derived evidence channel for the DTRUST trust model.

Motivation (paper Section 5, RQ "CIFAR-10 adaptive gap"): Olive beats DTRUST's
base policy on CIFAR-10 adaptive attacks because its mandatory per-client
clip-and-Gaussian-noise step also dampens adaptive perturbations. That step is
an *evidence channel* DTRUST's base policy does not yet exploit. This module
adds it as a sixth channel so the SPRT trust model can consume the same signal
without changing any audit operator.

The DP channel observes two per-client, per-round scalars that any
clip-and-noise defence already computes:

    clip_ratio_i(t) = ||Delta_i(t)|| / C            (pre-clip norm / clip bound)
    noise_align_i(t) = <Delta_i(t), g_noise> / ...  (optional; default unused)

Under H_0 (honest) the clip ratio concentrates below 1 (most honest updates do
not hit the clip bound); under H_1 (adaptive poisoning) the attacker repeatedly
saturates or exceeds the clip bound to maximise damage, so clip_ratio is
right-shifted. We model clip_ratio with a Gamma/Gaussian pair and emit a
closed-form LLR exactly like the other channels in ``_llr.py``.

This module is intentionally additive: it plugs into ``SPRTScorer`` as one more
``_accumulate`` term and into the experiment ``TrustEngine`` as one more
``enabled_channels`` entry. No audit operator (Q1-Q11) changes.
"""

from __future__ import annotations

import math
from typing import Dict


def llr_dp(clip_ratio: float, mu0: float, sigma0: float, mu1: float) -> float:
    """Gaussian LLR log P(c|H_0)/P(c|H_1) on the clip-ratio scalar.

    H_0 ~ N(mu0, sigma0^2) (honest, clip rarely binds; mu0 < 1)
    H_1 ~ N(mu1, sigma0^2) (adaptive, clip saturates; mu1 > 1)
    Positive value = evidence for H_0, matching the sign convention in _llr.py.
    """
    return ((mu0 - mu1) * (2.0 * clip_ratio - mu0 - mu1)) / (2.0 * sigma0 * sigma0)


def default_dp_config() -> Dict[str, float]:
    """Calibration defaults; override from a no-attack run."""
    return {
        "mu0":    0.55,   # honest clip ratio mean (most updates well under bound)
        "sigma0": 0.20,
        "mu1":    1.15,   # adaptive attacker saturates / exceeds the clip bound
    }


def clip_ratio_from_update(update_norm: float, clip_bound: float) -> float:
    """Derive the clip-ratio evidence scalar from a raw update norm.

    This is the only quantity the channel needs; both Olive-style and
    DTRUST-style deployments already compute ``update_norm`` and ``clip_bound``
    during aggregation, so the channel adds no new gradient access.
    """
    if clip_bound <= 0:
        return 0.0
    return update_norm / clip_bound
