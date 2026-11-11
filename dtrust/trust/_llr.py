"""Closed-form per-channel LLR helpers for the DTRUST SPRT trust model.

Pure math, no DTRUST-internal imports. Used by both the production scorer
(:mod:`dtrust.trust.sprt_scorer`) and the experiment harness
(:mod:`dtrust.attack_experiments.trust_model.engine`), so the math is
defined in exactly one place.

Sign convention (matches Section IV-A of the paper):
    LLR returns  log P(e | H_0) - log P(e | H_1)
    positive value  ==>  evidence for H_0 (honest)
    so  lambda(t) := sum_t LLR  drifts UP under H_0 and DOWN under H_1.
"""

from __future__ import annotations

import math
from typing import Dict


def llr_att(r: float, mu0: float, sigma0: float, mu1: float) -> float:
    """Gaussian LLR for the attestation residual channel."""
    return ((mu0 - mu1) * (2.0 * r - mu0 - mu1)) / (2.0 * sigma0 * sigma0)


def llr_grad(g: float, a0: float, b0: float, a1: float, b1: float) -> float:
    """Beta LLR for the gradient-cosine channel. Clamps g away from 0/1."""
    g = min(max(g, 1e-6), 1.0 - 1e-6)
    log_const = (math.lgamma(a1) + math.lgamma(b1) - math.lgamma(a1 + b1)) \
              - (math.lgamma(a0) + math.lgamma(b0) - math.lgamma(a0 + b0))
    return log_const + (a0 - a1) * math.log(g) + (b0 - b1) * math.log(1.0 - g)


def llr_perf(success: int, p0: float, p1: float) -> float:
    """Bernoulli LLR for the per-round success channel."""
    if success:
        return math.log(p0 / p1)
    return math.log((1.0 - p0) / (1.0 - p1))


def llr_beh(event: str, table: Dict[str, float]) -> float:
    return table.get(event, table.get("none", 0.0))


def llr_seal(ok: int, cfg: Dict[str, float]) -> float:
    return cfg["ok"] if ok else cfg["violation"]


def sigmoid(x: float) -> float:
    """Numerically-stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
