"""SPRT / Bayesian Trust Scorer for DTRUST.

This is the source-code instantiation of the trust model described in
Section IV-A of the DTRUST paper:

    lambda_i(t) = clamp( lambda_i(t-1)
                        + sum_k  L_k(e_k^(i)(t))         # per-channel LLR
                        - eta * delta_i(t) )              # AdvDev penalty
    T_i(t)      = sigmoid(lambda_i(t))                    # = Pr[H_0 | e(1:t)]

It is a drop-in alternative to :class:`dtrust.trust.scorer.TrustScorer` and
exposes the same four update methods plus :meth:`calculate_overall_score`.
Internally it accumulates a per-node log-odds score (``_lambda``) and turns
that into the ``overall_score`` field of :class:`TrustMetrics` through a
sigmoid mapping, so downstream DTRUST components that read
``trust_metrics.overall_score`` keep working unchanged.

The five channels exactly mirror the paper:

    * ``att``   --- attestation residual                       (Gaussian LLR)
    * ``grad``  --- gradient cosine                            (Beta LLR)
    * ``perf``  --- per-round success rate                     (Bernoulli LLR)
    * ``beh``   --- behavioural events                         (per-event constants)
    * ``seal``  --- sealed-state continuity                    (deterministic)

For the standalone experiment harness see
``dtrust.attack_experiments.trust_model``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import TrustConfig
from ..core.models import (
    AttestationResult,
    AttestationStatus,
    TrustMetrics,
)
from ._llr import llr_att, llr_beh, llr_grad, llr_perf, llr_seal, sigmoid


# ---------------------------------------------------------------------------#
# Default calibration -- can be overridden via SPRTScorer.from_config(path)
# ---------------------------------------------------------------------------#

DEFAULT_SPRT_CONFIG: Dict = {
    "channels": {
        "att":  {"mu0": 0.0, "sigma0": 1.0, "mu1": 0.45},
        "grad": {"alpha0": 5.0, "beta0": 3.0,
                 "alpha1": 3.0, "beta1": 5.0},
        "perf": {"p0": 0.92, "p1": 0.78},
        "beh":  {"none":  0.02, "timeout": -0.30,
                 "comm": -0.15, "susp":    -0.80},
        "seal": {"ok":    0.01, "violation": -30.0},
    },
    "thresholds": {
        "lambda_max":    0.847,
        "lambda_min":   -0.847,
        "clamp_lambda": 10.0,
        "clamp_per_llr": 2.0,
    },
    "advdev":     {"eta": 0.0, "window": 20},
    "quarantine": {"tau_min": 0.30, "tau_max": 0.70, "susp_max": 3},
    # Forgetting factor of the leaky / discounted SPRT register (paper Section IV-A, deployed model).
    # decay=1.0 is the classical additive SPRT (exact posterior log-odds; Wald-calibration limit);
    # decay<1.0 is the deployed bounded-memory model. Single value, single recursion across the
    # production scorer, trust_model.TrustEngine, and the policy-replay engine.
    "decay":      0.95,
}


# ---------------------------------------------------------------------------#
# Closed-form LLR helpers live in :mod:`dtrust.trust._llr` so the experiment
# harness can import them without pulling in TrustManager / core / loguru.
# ---------------------------------------------------------------------------#

@dataclass
class _PerNodeState:
    """Mutable per-node state held by the SPRT scorer."""
    lam:           float = 0.0
    susp_streak:   int   = 0
    history:       List[float] = field(default_factory=list)
    banned:        bool  = False
    # tracking the most recent contribution of each channel for explainability
    last_channels: Dict[str, float] = field(default_factory=dict)


class SPRTScorer:
    """SPRT / Bayesian implementation of the DTRUST trust scorer interface.

    Methods mirror :class:`dtrust.trust.scorer.TrustScorer` so that
    :class:`dtrust.trust.manager.TrustManager` can use either implementation
    interchangeably.
    """

    # The channels of TrustMetrics that we map back into for downstream code.
    # We keep the legacy attestation/performance/behavior fields as
    # *per-channel posteriors* so that legacy callers continue to see a
    # number in [0, 1] but the numerical authority is ``overall_score``.

    # ------------------------------------------------------------------ ctor
    def __init__(self, config: TrustConfig,
                 sprt_config: Optional[Dict] = None):
        self.config = config
        self.sprt   = sprt_config or DEFAULT_SPRT_CONFIG
        self._state: Dict[str, _PerNodeState] = {}

    @classmethod
    def from_config(cls, config: TrustConfig,
                    sprt_config_path: str | Path) -> "SPRTScorer":
        with open(sprt_config_path, "r") as f:
            cfg = json.load(f)
        return cls(config=config, sprt_config=cfg)

    # ---------------------------------------------------------------- state
    def _get_state(self, node_id: str) -> _PerNodeState:
        if node_id not in self._state:
            self._state[node_id] = _PerNodeState()
        return self._state[node_id]

    def lam(self, node_id: str) -> float:
        return self._get_state(node_id).lam

    def is_banned(self, node_id: str) -> bool:
        return self._get_state(node_id).banned

    # --------------------------------------------------------------- update
    def _accumulate(self, st: _PerNodeState,
                    contributions: Dict[str, float]) -> float:
        cfg = self.sprt
        clamp_per_llr = cfg["thresholds"]["clamp_per_llr"]
        clamp_lambda  = cfg["thresholds"]["clamp_lambda"]
        eta           = cfg["advdev"]["eta"]
        window        = cfg["advdev"]["window"]

        # per-channel clamp
        clipped: Dict[str, float] = {}
        for k, v in contributions.items():
            if k == "seal":
                clipped[k] = max(-clamp_per_llr * 10.0, min(clamp_per_llr, v))
            else:
                clipped[k] = max(-clamp_per_llr, min(clamp_per_llr, v))
        increment = sum(clipped.values())

        # AdvDev penalty = eta * stdev of recent increments
        st.history.append(increment)
        if len(st.history) > window:
            del st.history[0]
        if len(st.history) >= 4:
            mean = sum(st.history) / len(st.history)
            var  = sum((x - mean) ** 2 for x in st.history) / len(st.history)
            delta = math.sqrt(var)
        else:
            delta = 0.0
        penalty = eta * delta

        st.lam = max(-clamp_lambda, min(clamp_lambda, st.lam + increment - penalty))
        st.last_channels = clipped
        st.last_channels["advdev"] = penalty
        return st.lam

    # ----------------------------------------------- TrustScorer-compatible
    def update_attestation_score(self, trust_metrics: TrustMetrics,
                                 attestation: AttestationResult) -> None:
        """Convert one AttestationResult into an attestation-channel LLR."""
        st = self._get_state(trust_metrics.node_id)
        cfg_att = self.sprt["channels"]["att"]
        if attestation.status == AttestationStatus.SUCCESS:
            residual = 1.0 - attestation.confidence
            l_a = llr_att(residual, cfg_att["mu0"], cfg_att["sigma0"], cfg_att["mu1"])
        else:
            # failed attestation -> strong negative evidence (cap by clamp)
            l_a = -self.sprt["thresholds"]["clamp_per_llr"]
        self._accumulate(st, {"att": l_a})
        trust_metrics.attestation_score = sigmoid(l_a)
        trust_metrics.last_attestation  = attestation.verification_time

    def update_performance_score(self, trust_metrics: TrustMetrics,
                                 num_examples: int,
                                 metrics: Dict[str, float]) -> None:
        st = self._get_state(trust_metrics.node_id)
        cfg_perf = self.sprt["channels"]["perf"]
        cfg_grad = self.sprt["channels"]["grad"]

        # Bernoulli signal: use success_rate threshold p0 as the boundary
        success_rate = trust_metrics.success_rate
        succeeded    = 1 if success_rate >= cfg_perf["p0"] else 0
        l_p = llr_perf(succeeded, cfg_perf["p0"], cfg_perf["p1"])

        # Optional gradient-cosine evidence from the FL round
        l_g = 0.0
        if "gradient_cosine" in metrics:
            g = float(metrics["gradient_cosine"])
            l_g = llr_grad(
                g, cfg_grad["alpha0"], cfg_grad["beta0"],
                cfg_grad["alpha1"], cfg_grad["beta1"],
            )

        self._accumulate(st, {"perf": l_p, "grad": l_g})
        trust_metrics.performance_score = sigmoid(l_p + l_g)

    def update_behavior_score(self, trust_metrics: TrustMetrics,
                              suspicious: bool = False,
                              timeout: bool = False,
                              communication_error: bool = False) -> None:
        st = self._get_state(trust_metrics.node_id)
        if suspicious:
            event = "susp"
        elif timeout:
            event = "timeout"
        elif communication_error:
            event = "comm"
        else:
            event = "none"
        l_b = llr_beh(event, self.sprt["channels"]["beh"])
        self._accumulate(st, {"beh": l_b})
        trust_metrics.behavior_score = sigmoid(l_b)

    def update_seal_score(self, trust_metrics: TrustMetrics,
                          counter_ok: bool) -> None:
        """Optional seal channel (no equivalent on legacy TrustScorer)."""
        st = self._get_state(trust_metrics.node_id)
        l_s = llr_seal(1 if counter_ok else 0, self.sprt["channels"]["seal"])
        self._accumulate(st, {"seal": l_s})

    def calculate_overall_score(self, trust_metrics: TrustMetrics) -> None:
        """Map the accumulated log-odds to a [0,1] trust score."""
        st = self._get_state(trust_metrics.node_id)
        trust_metrics.overall_score = sigmoid(st.lam)

    # --------------------------------------------------- canonical per-round step
    def step(self, node_id: str, evidence: Dict) -> float:
        """Canonical per-round leaky-SPRT update -- the SINGLE SOURCE OF TRUTH recursion,
        bit-identical to ``trust_model.TrustEngine.update`` and ``c1_audit.sprt_replay`` (verified
        by ``tests/test_sprt_engine_consistency.py``). Applies the round's five channel LLRs at once
        with one decay step:  lambda <- clamp(decay*lambda + sum_k L_k - eta*delta).  Returns
        T = sigmoid(lambda).  The per-channel ``update_*`` methods remain for the incremental
        TrustManager path; ``step`` is the batch/governance path used by the FL-in-loop run and replay.
        """
        st = self._get_state(node_id)
        cfg = self.sprt
        cpl = cfg["thresholds"]["clamp_per_llr"]; cl = cfg["thresholds"]["clamp_lambda"]
        eta = cfg["advdev"]["eta"]; window = cfg["advdev"]["window"]
        decay = cfg.get("decay", 1.0); ca = cfg["channels"]
        l_a = max(-cpl, min(cpl, llr_att(evidence["att_resid"], ca["att"]["mu0"], ca["att"]["sigma0"], ca["att"]["mu1"])))
        l_g = max(-cpl, min(cpl, llr_grad(evidence["grad_cos"], ca["grad"]["alpha0"], ca["grad"]["beta0"], ca["grad"]["alpha1"], ca["grad"]["beta1"])))
        l_p = max(-cpl, min(cpl, llr_perf(int(evidence["perf_success"]), ca["perf"]["p0"], ca["perf"]["p1"])))
        l_b = max(-cpl, min(cpl, llr_beh(evidence["beh_event"], ca["beh"])))
        l_s = max(-cpl * 10.0, min(cpl, llr_seal(int(evidence["seal_ok"]), ca["seal"])))
        inc = l_a + l_g + l_p + l_b + l_s
        st.history.append(inc)
        if len(st.history) > window:
            del st.history[0]
        delta = 0.0
        if len(st.history) >= 4:
            mean = sum(st.history) / len(st.history)
            delta = math.sqrt(sum((x - mean) ** 2 for x in st.history) / len(st.history))
        st.lam = max(-cl, min(cl, decay * st.lam + inc - eta * delta))
        st.last_channels = {"att": l_a, "grad": l_g, "perf": l_p, "beh": l_b, "seal": l_s,
                            "advdev": eta * delta}
        return sigmoid(st.lam)

    # -------------------------------------------------------------- decision
    def decide(self, trust_metrics: TrustMetrics) -> Optional[str]:
        """Return 'quarantine' / 'readmit' / 'ban' / None for this node.

        Mirrors the SPRT decision rule from the standalone experiment engine.
        """
        st = self._get_state(trust_metrics.node_id)
        cfg = self.sprt
        lam_min  = cfg["thresholds"]["lambda_min"]
        lam_max  = cfg["thresholds"]["lambda_max"]
        susp_max = cfg["quarantine"]["susp_max"]
        clamp    = cfg["thresholds"]["clamp_lambda"]

        if st.lam <= -clamp * 0.99:
            st.banned = True
            return "ban"
        if st.lam <= lam_min:
            st.susp_streak += 1
            if st.susp_streak >= susp_max:
                st.banned = True
                return "ban"
            return "quarantine"
        if st.lam >= lam_max:
            st.susp_streak = 0
            return "readmit"
        return None

    # -------------------------------------------------------- explainability
    def get_trust_explanation(self, trust_metrics: TrustMetrics) -> Dict:
        st = self._get_state(trust_metrics.node_id)
        return {
            "overall_score": trust_metrics.overall_score,
            "lambda":        st.lam,
            "posterior":     sigmoid(st.lam),
            "channels":      dict(st.last_channels),
            "susp_streak":   st.susp_streak,
            "banned":        st.banned,
            "method":        "sprt_bayesian",
        }
