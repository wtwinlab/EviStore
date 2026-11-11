"""Faithful synthetic delayed-compromise evidence (local dev harness).

Generates the Eq.8 evidence stream for a delayed-compromise incident WITHOUT
running FL training, so the whole seal -> audit pipeline can be exercised on a
laptop in seconds and the SPRT channel parameters calibrated before the real
GPU run. The adversary/honest draws come from
``dtrust.attack_experiments.trust_model.simulate`` -- the same H0/H1 priors the
production trust model was tuned against -- so the synthetic grad_cos /
att_resid trajectories are distributionally consistent with the live runner.

Delayed compromise: a fixed 20% malicious subset behaves benignly (H0 draws)
in [0, R_atk), then turns adversarial (H1 "delayed" family) from R_atk on.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtrust.attack_experiments.trust_model.simulate import (  # noqa: E402
    honest_evidence, _beta_sample,
)

from .sprt_replay import Evidence, BEH_NAMES  # noqa: E402


def _ramped_adversary_evidence(rng: random.Random, cfg: dict, intensity: float) -> dict:
    """Draw H1 evidence at a continuous attack ``intensity`` in [0, 1].

    Mirrors ``simulate.adversary_evidence`` but exposes intensity directly so a
    *stealthy ramp* (intensity rising R_atk -> R_inc) can be modelled. At
    intensity 0 the draw is H0 (honest); at 1 it is the full adversary. The
    ramp is what makes detection timing threshold-dependent: a stricter policy
    crosses its decision boundary earlier on the same rising evidence.
    """
    ch = cfg["channels"]
    mu0, mu1 = ch["att"]["mu0"], ch["att"]["mu1"]
    a0, b0 = ch["grad"]["alpha0"], ch["grad"]["beta0"]
    a1, b1 = ch["grad"]["alpha1"], ch["grad"]["beta1"]
    p0, p1 = ch["perf"]["p0"], ch["perf"]["p1"]

    mu_eff = mu0 + intensity * (mu1 - mu0)
    a_eff = a0 + intensity * (a1 - a0)
    b_eff = b0 + intensity * (b1 - b0)
    p_eff = p0 + intensity * (p1 - p0)

    att = rng.gauss(mu_eff, ch["att"]["sigma0"])
    grad = _beta_sample(rng, a_eff, b_eff)
    perf = 1 if rng.random() < p_eff else 0
    u = rng.random()
    if u < 0.05 * intensity:
        beh = "susp"
    elif u < 0.10 * intensity:
        beh = "timeout"
    else:
        beh = "none"
    return {"att_resid": att, "grad_cos": grad, "perf_success": perf,
            "beh_event": beh, "seal_ok": 1}


def _intensity(r: int, r_atk: int, r_inc: int, profile: str) -> float:
    if r < r_atk:
        return 0.0
    if profile == "step":
        return 1.0
    # 'ramp': linear from 0 at r_atk to 1 at r_inc, saturating after.
    if r >= r_inc:
        return 1.0
    return (r - r_atk) / max(1, (r_inc - r_atk))


def make_delayed_compromise_event(
    theta: dict,
    n_clients: int = 100,
    malicious_frac: float = 0.2,
    r_total: int = 500,
    r_atk: int = 50,
    r_inc: int = 150,
    seed: int = 0,
    profile: str = "ramp",
) -> Tuple[Evidence, List[int]]:
    """Return (Evidence, malicious_ids) for one seed.

    Malicious clients draw H0 (honest, indistinguishable) in [0, R_atk), then
    ramp from honest to full adversary over [R_atk, R_inc] (``profile='ramp'``)
    or switch immediately (``profile='step'``). The ramp is the realistic
    sleeper-that-escalates and the regime where tightening the policy buys you
    earlier detection, which the case study examines.
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    n_mal = int(round(malicious_frac * n_clients))
    malicious_ids = sorted(int(x) for x in nprng.choice(n_clients, n_mal, replace=False))
    mal_set = set(malicious_ids)

    R, N = r_total, n_clients
    att = np.zeros((R, N)); grad = np.zeros((R, N))
    perf = np.zeros((R, N), dtype=np.int64); beh = np.zeros((R, N), dtype=np.int64)
    seal = np.ones((R, N), dtype=np.int64); is_adv = np.zeros((R, N), dtype=np.int64)

    for r in range(R):
        inten = _intensity(r, r_atk, r_inc, profile)
        for pid in range(N):
            if pid in mal_set and inten > 0.0:
                ev = _ramped_adversary_evidence(rng, theta, inten)
                is_adv[r, pid] = 1
            else:
                ev = honest_evidence(rng, theta)
            att[r, pid] = ev["att_resid"]
            grad[r, pid] = ev["grad_cos"]
            perf[r, pid] = ev["perf_success"]
            beh[r, pid] = BEH_NAMES[ev["beh_event"]]
            seal[r, pid] = ev["seal_ok"]

    return Evidence(att_resid=att, grad_cos=grad, perf_success=perf,
                    beh_event=beh, seal_ok=seal, is_adversary=is_adv), malicious_ids
