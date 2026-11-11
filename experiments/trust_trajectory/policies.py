"""Policy pair (rho_v1, rho_v2) for the differential-audit case study.

The base SPRT config is the production trust model from
``dtrust/attack_experiments/trust_model/configs/trust_model.json``. The two
policies differ ONLY in scalar theta parameters: the SPRT decision threshold tau_min and the
CUSUM drift threshold h. No new statistic is introduced, so both policies
stay inside the replayable boundary of the sealed evidence (Section IV-B).

Mapping (exact, by construction):
    tau_min  <->  lambda_min = logit(tau_min)      [SPRT lower threshold]
    tau_max  <->  lambda_max = logit(tau_max)      [SPRT upper threshold]
Because t_score = sigmoid(lambda) and lambda does NOT depend on tau_min,
re-deciding under a new tau_min is a pure threshold query on the sealed
t_score -- i.e. exactly TrustStore.differential_audit, with no re-execution.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Tuple

_BASE_CFG_PATH = (
    Path(__file__).resolve().parents[3]
    / "dtrust" / "attack_experiments" / "trust_model" / "configs" / "trust_model.json"
)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def load_base_config() -> dict:
    with open(_BASE_CFG_PATH, "r") as f:
        return json.load(f)


def _with_thresholds(base: dict, tau_min: float, cusum_h: float,
                     tau_max: float | None = None, decay: float | None = None) -> dict:
    """Return a deep copy of ``base`` with tau_min / tau_max / h overridden.

    ``cusum_h`` is stored under a dedicated ``cusum`` block consumed by the
    Page-CUSUM drift detector in :mod:`sprt_replay`; it does not touch the
    SPRT lambda recursion (the two detectors share evidence, not state).
    """
    cfg = copy.deepcopy(base)
    cfg["quarantine"]["tau_min"] = tau_min
    cfg["thresholds"]["lambda_min"] = logit(tau_min)
    if tau_max is not None:
        cfg["quarantine"]["tau_max"] = tau_max
        cfg["thresholds"]["lambda_max"] = logit(tau_max)
    cfg.setdefault("cusum", {})["h"] = cusum_h
    if decay is not None:
        cfg["decay"] = decay
    return cfg


# Baseline CUSUM threshold. The spec calls for h ~ 4*sigma0 (sigma0 = 1.0 for
# the attestation channel). The per-round honest LLR increment is the sum of
# five channels, so the effective drift unit is larger than a single sigma0;
# H_V1 is calibrated in run_c1_local so that v1 raises the CUSUM alarm for a
# delayed-compromise client around R_inc rather than instantly or never.
H_V1_DEFAULT = 4.0

# Trust forgetting factor (leaky-integrator / half-life decay). With a leaky
# lambda recursion lambda_t = gamma*lambda_{t-1} + increment, a persistent weak
# attack drives trust to a STEADY state sigmoid(increment_mean/(1-gamma)) rather
# than marching past every threshold -- so a stealthy client can sit in the
# [tau_v1, tau_v2] gap for the whole incident (admitted by v1, rejected by v2).
# decay is an explicit theta parameter.
DECAY_DEFAULT = 0.9


def policy_pair(base: dict | None = None,
                h_v1: float = H_V1_DEFAULT,
                decay: float = DECAY_DEFAULT,
                admission_includes_cusum: bool = False) -> Tuple[dict, dict]:
    """Return (rho_v1, rho_v2).

    rho_v1 (online):     tau_min = 0.30, h = h_v1          (the lenient policy)
    rho_v2 (tightened):  tau_min = 0.45, h = h_v1 / 2       (the audit policy)
    Both share the forgetting factor ``decay`` (the same trust dynamics; only the
    decision thresholds tighten).
    """
    base = base or load_base_config()
    v1 = _with_thresholds(base, tau_min=0.30, cusum_h=h_v1, decay=decay)
    v2 = _with_thresholds(base, tau_min=0.45, cusum_h=h_v1 / 2.0, decay=decay)
    v1["admission_includes_cusum"] = admission_includes_cusum
    v2["admission_includes_cusum"] = admission_includes_cusum
    v1["_label"] = "rho_v1"
    v2["_label"] = "rho_v2"
    return v1, v2
