"""f_rho = replay(E, theta): the pure DTRUST decision function (paper Section IV-A).

This is the single source of truth for the online decision AND the offline
counterfactual. Sealing the ledger (``seal.py``) runs it once under theta_v1;
the audit (``audit.py``) re-runs it under theta_v2. Because it is a pure
function of the sealed evidence E and the scalar policy theta, replaying it
under theta_v1 reproduces the online decision bit-for-bit (self-check #1), and
replaying under theta_v2 yields the counterfactual D' with NO enclave
re-execution -- the Theorem-3.5 sufficiency claim, made operational.

Two coupled detectors share the evidence but not their state:

  SPRT (tau_min knob).  Per-channel LLR -> clamp -> AdvDev penalty -> Wald
      recursion lambda_i(t). Trust t_score = sigmoid(lambda). decide() emits
      quarantine / ban / readmit from lambda vs lambda_min/max. This is exactly
      ``dtrust.attack_experiments.trust_model.engine.TrustEngine`` -- reused,
      not reimplemented, so the math is identical to the production scorer.

  CUSUM (h knob).  A one-sided Page statistic S_i(t) = max(0, S_i(t-1) - g_i(t))
      on the honest-evidence increment g (positive under H0). S rises only when
      evidence turns against the honest hypothesis; the drift alarm fires at
      S >= h. S is sealed so that DRI (TrustStore.dri_replay) can recompute the
      earliest crossing under a fresh h without re-running anything.

Committed per-round decision: by default a participant is excluded from
aggregation iff the persistent SPRT ban fires (the tau_min knob); its
aggregation weight w_agg is its t_score when admitted, else 0. The CUSUM drift
alarm (the h knob) is, by default, a SEPARATE early-warning/monitoring signal
that does NOT itself gate admission -- so an earlier CUSUM crossing under a
tighter h is an earlier *detector signal*, not earlier *exclusion* (empirically
the SPRT ban dominates exclusion timing). Set
theta["admission_includes_cusum"]=True to make the CUSUM alarm also exclude.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Reuse the production per-channel LLR closed-forms (single source of truth for
# the SPRT math); the leaky-integrator recursion + decision are inlined here so
# the forgetting factor (decay) -- an explicit theta parameter -- can be applied.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dtrust.trust._llr import (  # noqa: E402
    llr_att, llr_grad, llr_perf, llr_beh, llr_seal,
)

# Behaviour-event integer codes (compact, npz-friendly). Map to the string keys
# the LLR behaviour table in trust_model.json expects.
BEH_CODES: Dict[int, str] = {0: "none", 1: "timeout", 2: "comm", 3: "susp"}
BEH_NAMES: Dict[str, int] = {v: k for k, v in BEH_CODES.items()}

# Event integer codes for the sealed decision log.
EVENT_CODES: Dict[Optional[str], int] = {None: 0, "quarantine": 1, "ban": 2, "readmit": 3}
EVENT_NAMES: Dict[int, str] = {0: "none", 1: "quarantine", 2: "ban", 3: "readmit"}

MODE_NORMAL, MODE_DEGRADED, MODE_PAUSED = 0, 1, 2


@dataclass
class Evidence:
    """Sealed per-(round, participant) evidence E (Eq. 8 fields).

    Arrays are shape [R, N]. ``is_adversary`` is ground truth, kept alongside
    E for validation only -- the decision function never reads it.
    """
    att_resid: np.ndarray       # float  [R, N]
    grad_cos: np.ndarray        # float  [R, N], in (0, 1)
    perf_success: np.ndarray    # int    [R, N], {0,1}
    beh_event: np.ndarray       # int    [R, N], BEH_CODES
    seal_ok: np.ndarray         # int    [R, N], {0,1}
    is_adversary: np.ndarray    # int    [R, N], {0,1} (ground truth)

    @property
    def n_rounds(self) -> int:
        return self.att_resid.shape[0]

    @property
    def n_participants(self) -> int:
        return self.att_resid.shape[1]

    def round_dict(self, r: int, pid: int) -> dict:
        return {
            "att_resid": float(self.att_resid[r, pid]),
            "grad_cos": float(self.grad_cos[r, pid]),
            "perf_success": int(self.perf_success[r, pid]),
            "beh_event": BEH_CODES[int(self.beh_event[r, pid])],
            "seal_ok": int(self.seal_ok[r, pid]),
        }

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            att_resid=self.att_resid, grad_cos=self.grad_cos,
            perf_success=self.perf_success, beh_event=self.beh_event,
            seal_ok=self.seal_ok, is_adversary=self.is_adversary,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "Evidence":
        z = np.load(path)
        return cls(
            att_resid=z["att_resid"], grad_cos=z["grad_cos"],
            perf_success=z["perf_success"], beh_event=z["beh_event"],
            seal_ok=z["seal_ok"], is_adversary=z["is_adversary"],
        )


@dataclass
class ReplayResult:
    """Per-(round, participant) committed decision D, arrays shape [R, N]."""
    lam: np.ndarray             # SPRT lambda
    t_score: np.ndarray         # sigmoid(lambda) in [0, 1]
    s_cusum: np.ndarray         # Page CUSUM statistic >= 0
    w_agg: np.ndarray           # aggregation weight in [0, 1]
    admitted: np.ndarray        # bool: included in aggregation this round
    event: np.ndarray           # int  EVENT_CODES (SPRT decision)
    cusum_alarm: np.ndarray     # bool: S >= h this round
    banned: np.ndarray          # bool: SPRT ban committed (sticky)
    suspect: np.ndarray         # bool: transient t_score < tau_min this round
    a_conf: np.ndarray          # attestation confidence in [0, 1]
    b_score: np.ndarray         # behaviour posterior in [0, 1]
    increment: np.ndarray       # raw LLR sum g (pre-penalty)
    mode: np.ndarray            # int  [R] per-round aggregator mode
    theta_label: str = ""


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    z = np.exp(x)
    return z / (1.0 + z)


def _llr_vector(ev: dict, ch: dict, clamp: float):
    """Per-channel clamped LLRs for one (round, participant) evidence tuple."""
    l_a = max(-clamp, min(clamp, llr_att(ev["att_resid"], ch["att"]["mu0"],
              ch["att"]["sigma0"], ch["att"]["mu1"])))
    l_g = max(-clamp, min(clamp, llr_grad(ev["grad_cos"], ch["grad"]["alpha0"],
              ch["grad"]["beta0"], ch["grad"]["alpha1"], ch["grad"]["beta1"])))
    l_p = max(-clamp, min(clamp, llr_perf(int(ev["perf_success"]),
              ch["perf"]["p0"], ch["perf"]["p1"])))
    l_b = max(-clamp, min(clamp, llr_beh(ev["beh_event"], ch["beh"])))
    l_s = max(-clamp * 10.0, min(clamp, llr_seal(int(ev["seal_ok"]), ch["seal"])))
    return l_a, l_g, l_p, l_b, l_s


def replay(evidence: Evidence, theta: dict) -> ReplayResult:
    """Run the committed decision function f_rho over the whole sealed event.

    Leaky-integrator SPRT: lambda_t = decay*lambda_{t-1} + increment - penalty,
    so trust reaches a steady state set by the persistent evidence (decay < 1).
    Deterministic and stateful: participants are processed in id order within
    each round, rounds in time order, mirroring the online coordinator loop.
    """
    R, N = evidence.n_rounds, evidence.n_participants
    ch = theta["channels"]
    clamp_llr = theta["thresholds"]["clamp_per_llr"]
    clamp_lam = theta["thresholds"]["clamp_lambda"]
    lam_min = theta["thresholds"]["lambda_min"]
    lam_max = theta["thresholds"]["lambda_max"]
    susp_max = theta["quarantine"]["susp_max"]
    readmit_after = int(theta["quarantine"].get("readmit_after", 10**9))
    tau_min = float(theta["quarantine"]["tau_min"])
    decay = float(theta.get("decay", 1.0))
    eta = float(theta.get("advdev", {}).get("eta", 0.0))
    window = int(theta.get("advdev", {}).get("window", 20))
    h = float(theta.get("cusum", {}).get("h", float("inf")))
    # If True, a CUSUM drift alarm (the h knob) also excludes from aggregation,
    # not just the persistent SPRT ban (the tau_min knob). When True the h knob
    # is operationally load-bearing (it gates admission), so the earlier crossing
    # under a tighter h yields genuinely earlier EXCLUSION, not just an earlier
    # detector signal.
    cusum_gates = bool(theta.get("admission_includes_cusum", False))

    prev_lam = np.zeros(N); S = np.zeros(N)
    susp_streak = np.zeros(N, dtype=int); is_banned = np.zeros(N, dtype=bool)
    good_streak = np.zeros(N, dtype=int)
    hist = [[] for _ in range(N)]

    lam = np.zeros((R, N)); t_score = np.zeros((R, N)); s_cusum = np.zeros((R, N))
    w_agg = np.zeros((R, N)); admitted = np.zeros((R, N), dtype=bool)
    event = np.zeros((R, N), dtype=np.int8); cusum_alarm = np.zeros((R, N), dtype=bool)
    banned = np.zeros((R, N), dtype=bool); suspect = np.zeros((R, N), dtype=bool)
    a_conf = np.zeros((R, N)); b_score = np.zeros((R, N)); increment = np.zeros((R, N))
    mode = np.full(R, MODE_NORMAL, dtype=np.int8)

    for r in range(R):
        for pid in range(N):
            ev = evidence.round_dict(r, pid)
            l_a, l_g, l_p, l_b, l_s = _llr_vector(ev, ch, clamp_llr)
            inc = l_a + l_g + l_p + l_b + l_s

            # AdvDev penalty (std of recent increments), as in the engine.
            hb = hist[pid]; hb.append(inc)
            if len(hb) > window:
                del hb[0]
            penalty = 0.0
            if eta > 0.0 and len(hb) >= 4:
                mean = sum(hb) / len(hb)
                penalty = eta * math.sqrt(sum((x - mean) ** 2 for x in hb) / len(hb))

            lam_v = decay * prev_lam[pid] + inc - penalty
            lam_v = max(-clamp_lam, min(clamp_lam, lam_v))
            prev_lam[pid] = lam_v
            ts = _sigmoid(lam_v)

            # Decision: ban after susp_max consecutive sub-threshold rounds.
            # Asymmetric readmit (B): un-ban after readmit_after consecutive
            # non-sub-threshold rounds (slow retrust; default inf = sticky).
            evt = None
            if lam_v <= lam_min:
                susp_streak[pid] += 1
                good_streak[pid] = 0
                if susp_streak[pid] >= susp_max:
                    is_banned[pid] = True
                    evt = "ban"
                else:
                    evt = "quarantine"
            elif lam_v >= lam_max:
                susp_streak[pid] = 0
                good_streak[pid] += 1
                if is_banned[pid] and good_streak[pid] >= readmit_after:
                    is_banned[pid] = False
                    susp_streak[pid] = 0
                evt = "readmit"
            else:
                good_streak[pid] += 1

            S[pid] = max(0.0, S[pid] - inc)            # Page CUSUM on -increment
            alarm = S[pid] >= h
            # Committed exclusion = persistent SPRT ban (tau_min knob), and -- when
            # cusum_gates is on -- also a CUSUM drift alarm (h knob).
            adm = (not is_banned[pid]) and not (cusum_gates and alarm)

            lam[r, pid] = lam_v
            t_score[r, pid] = ts
            s_cusum[r, pid] = S[pid]
            event[r, pid] = EVENT_CODES[evt]
            cusum_alarm[r, pid] = alarm
            banned[r, pid] = is_banned[pid]
            suspect[r, pid] = ts < tau_min
            increment[r, pid] = inc
            a_conf[r, pid] = _sigmoid(l_a)
            b_score[r, pid] = _sigmoid(l_b)
            admitted[r, pid] = adm
            w_agg[r, pid] = ts if adm else 0.0

        # Degraded mode: the runtime reference set R(t) -- the
        # clients with FRESH attestation this round -- fell below the honest-
        # majority quorum N/2 the aggregator needs. This is a per-round,
        # recoverable evidence-sufficiency condition (it clears when attestation
        # is renewed), distinct from the sticky trust ban.
        if int(evidence.seal_ok[r].sum()) < (N // 2):
            mode[r] = MODE_DEGRADED

    return ReplayResult(
        lam=lam, t_score=t_score, s_cusum=s_cusum, w_agg=w_agg,
        admitted=admitted, event=event, cusum_alarm=cusum_alarm,
        banned=banned, suspect=suspect,
        a_conf=a_conf, b_score=b_score, increment=increment, mode=mode,
        theta_label=theta.get("_label", ""),
    )


def decisions_equal(a: ReplayResult, b: ReplayResult, *, quantized: bool = True) -> dict:
    """Compare two replay results cell-by-cell (self-check #1).

    With ``quantized=True`` the t_score / w_agg are compared at the uint16
    ledger resolution (what is actually sealed); admit/event/mode are compared
    exactly. Returns a dict of mismatch counts (all zero == bit-exact).
    """
    from dtrust.store.schema import quantize_unit

    def q(x):
        vq = np.vectorize(quantize_unit)
        return vq(x) if quantized else x

    return {
        "t_score_mismatch": int(np.sum(q(a.t_score) != q(b.t_score))),
        "w_agg_mismatch": int(np.sum(q(a.w_agg) != q(b.w_agg))),
        "admitted_mismatch": int(np.sum(a.admitted != b.admitted)),
        "event_mismatch": int(np.sum(a.event != b.event)),
        "s_cusum_mismatch": int(np.sum(~np.isclose(a.s_cusum, b.s_cusum))),
        "mode_mismatch": int(np.sum(a.mode != b.mode)),
    }
