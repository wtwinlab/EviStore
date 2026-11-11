"""Build SPRT evidence E from a real FL delayed-compromise dump.

The recovery runner (``scripts/run_recovery_experiment.py``) trains the
real Fashion-MNIST delayed-compromise event and, per round, computes each
client's cosine-to-median ``cos`` and update ``norm``. With ``--dump_ledger_dir``
it now also writes those per-(round, client) arrays to an npz. This module maps
that dump into the Eq.8 evidence E that the SPRT engine consumes.

Channel construction under the MEASURED-CLIENT threat model
-----------------------------------------------------------
The attacker is a correctly-attested client; the enclave measurement is clean.
So the attestation/seal channels are deliberately BLIND -- they carry no signal
about the data/participation attack -- and the discriminative evidence is the
real gradient-cosine drift:

  grad_cos     <- real cos-to-median (clipped to (0,1)); the load-bearing signal
  att_resid    <- H0 draw N(mu0, sigma0): attestation passes for everyone
  seal_ok      <- 1 for everyone: the enclave is genuinely sealed
  perf_success <- 1 iff the client agrees with the round's reference direction
  beh_event    <- 'susp' on a norm spike (>mean+3std), else 'none'

This keeps the att channel non-informative (the anti-oracle property:
attestation cannot see a data/participation attack) and makes the trust verdict
a consequence of observable runtime behaviour, exactly as the motivation argues.
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

from .sprt_replay import Evidence, BEH_NAMES  # noqa: E402


def evidence_from_dump(dump_path: str | Path, theta: dict, seed: int = 0,
                       att_blind: bool = True) -> Tuple[Evidence, List[int], dict]:
    """Return (Evidence, malicious_ids, meta) from a recovery-runner dump npz."""
    z = np.load(dump_path, allow_pickle=True)
    cos = np.asarray(z["cos"], dtype=np.float64)        # [R, N]
    norm = np.asarray(z["norm"], dtype=np.float64)      # [R, N]
    malicious_ids = sorted(int(x) for x in z["malicious_ids"])
    r_atk = int(z["r_atk"]); r_inc = int(z["r_inc"])
    R, N = cos.shape
    mal_set = set(malicious_ids)

    rng = random.Random(10_000 + seed)
    ch = theta["channels"]
    mu0, sigma0 = ch["att"]["mu0"], ch["att"]["sigma0"]

    # Map cosine-to-median from [-1, 1] to the (0, 1) support of the Beta grad
    # channel via the standard affine remap. Empirically (real Fashion-MNIST
    # trim event) honest cos ~ +0.5 -> g ~ 0.75 (above the Beta(5,3) honest
    # mean) and post-attack malicious cos ~ -0.55 -> g ~ 0.22 (Beta(3,5)
    # region), so the validated default priors separate them without retuning.
    # This is sensor calibration of the G channel, applied identically under
    # BOTH policies -- it is not a policy knob, so the v1 replay stays exact.
    grad = np.clip((cos + 1.0) / 2.0, 1e-3, 1.0 - 1e-3)
    att = np.zeros((R, N)); perf = np.zeros((R, N), dtype=np.int64)
    beh = np.zeros((R, N), dtype=np.int64); seal = np.ones((R, N), dtype=np.int64)
    is_adv = np.zeros((R, N), dtype=np.int64)

    # update-norm channel threshold (policy-normalized; default 3x round median).
    k_norm = float(ch.get("norm", {}).get("k_median", 3.0))
    for r in range(R):
        ref = np.median(cos[r])
        nmean, nstd = float(np.mean(norm[r])), max(float(np.std(norm[r])), 1e-6)
        nmed = max(float(np.median(norm[r])), 1e-9)
        for i in range(N):
            att[r, i] = rng.gauss(mu0, sigma0) if att_blind else (mu0 if cos[r, i] >= ref else ch["att"]["mu1"])
            perf[r, i] = 1 if cos[r, i] >= ref else 0
            beh[r, i] = BEH_NAMES["susp"] if norm[r, i] > nmean + 3.0 * nstd else BEH_NAMES["none"]
            # update-norm LLR: an update whose norm is k_norm x above the round median
            # has its grad-quality reading suppressed (norm anomaly => suspect), so a
            # magnitude attack (scaling) that leaves cosine intact is still caught. A
            # normal-norm update (sign-flip/trim) has nratio ~ 1 and is left unchanged.
            nratio = norm[r, i] / nmed
            if nratio > k_norm:
                grad[r, i] = float(np.clip(grad[r, i] * (k_norm / nratio), 1e-3, 1.0 - 1e-3))
            if i in mal_set and r >= r_atk:
                is_adv[r, i] = 1

    evidence = Evidence(att_resid=att, grad_cos=grad, perf_success=perf,
                        beh_event=beh, seal_ok=seal, is_adversary=is_adv)
    meta = {"r_atk": r_atk, "r_inc": r_inc, "n_rounds": R, "n_participants": N}
    return evidence, malicious_ids, meta
