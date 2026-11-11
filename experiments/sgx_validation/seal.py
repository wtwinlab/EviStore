"""Seal a delayed-compromise event into the binary TrustStore ledger.

Drives the committed decision function f_rho (``sprt_replay.replay``) over the
sealed evidence under the ONLINE policy rho_v1, and writes one TrustTuple per
(participant, round) into ``dtrust.store.TrustStore`` -- the Merkle-anchored
sealed ledger whose Q5/Q6/Q9/Q11 + DRI operators the differential audit consumes.

Alongside the binary ledger it persists:
  * ``evidence_seed<S>.npz`` -- the raw Eq.8 evidence E (for the self-check
    replay and the from-E v2 counterfactual);
  * ``meta_seed<S>.json``    -- ground truth (malicious set, R_atk, R_inc),
    the pid<->participant-hash map, and theta_v1.

The split matters: D goes in the integrity-protected binary ledger;
E is the input the auditor replays. The evidence contract says E + theta suffices to
reproduce D, which the audit verifies bit-for-bit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtrust.store import TrustStore, TrustTuple, Mode  # noqa: E402
from dtrust.store.schema import quantize_unit  # noqa: E402

from .sprt_replay import Evidence, ReplayResult, replay  # noqa: E402


def participant_hash(pid: int) -> bytes:
    """Stable 16-byte enclave-identity hash for a client id."""
    return hashlib.sha256(f"client-{pid}".encode()).digest()[:16]


def _h_grad(evidence: Evidence, r: int, pid: int) -> bytes:
    """32-byte commitment to the submitted update. For the synthetic harness we
    commit to the evidence row; the real runner passes the actual delta hash."""
    row = np.array([
        evidence.att_resid[r, pid], evidence.grad_cos[r, pid],
        evidence.perf_success[r, pid], evidence.beh_event[r, pid],
        evidence.seal_ok[r, pid], r, pid,
    ], dtype=np.float64)
    return hashlib.sha256(row.tobytes()).digest()


@dataclass
class SealedEvent:
    ledger_path: Path
    evidence_path: Path
    meta_path: Path


def seal_event(
    evidence: Evidence,
    decision: ReplayResult,
    theta_v1: dict,
    out_dir: str | Path,
    seed: int,
    r_atk: int,
    r_inc: int,
    malicious_ids: List[int],
    anchor_interval: int = 50,
    h_grad_from=None,
) -> SealedEvent:
    """Write the binary ledger + evidence sidecar + meta for one seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / f"ledger_seed{seed}.db"
    evidence_path = out_dir / f"evidence_seed{seed}.npz"
    meta_path = out_dir / f"meta_seed{seed}.json"
    if ledger_path.exists():
        ledger_path.unlink()

    R, N = evidence.n_rounds, evidence.n_participants
    pid_to_hash = {pid: participant_hash(pid) for pid in range(N)}

    store = TrustStore(ledger_path, anchor_interval=anchor_interval)
    for r in range(R):
        tuples = []
        for pid in range(N):
            hg = (h_grad_from(r, pid) if h_grad_from is not None
                  else _h_grad(evidence, r, pid))
            tuples.append(TrustTuple(
                round=r,
                participant=pid_to_hash[pid],
                a_conf=quantize_unit(float(decision.a_conf[r, pid])),
                fresh=bool(evidence.seal_ok[r, pid]),
                s_cusum=float(decision.s_cusum[r, pid]),
                t_score=quantize_unit(float(decision.t_score[r, pid])),
                b_score=quantize_unit(float(decision.b_score[r, pid])),
                w_agg=quantize_unit(float(decision.w_agg[r, pid])),
                mode=Mode(int(decision.mode[r])),
                h_grad=hg,
            ))
        store.append_round(r, tuples)
    store.close()

    evidence.save_npz(evidence_path)
    meta = {
        "seed": seed,
        "n_rounds": R,
        "n_participants": N,
        "r_atk": r_atk,
        "r_inc": r_inc,
        "malicious_ids": sorted(int(x) for x in malicious_ids),
        "theta_v1": theta_v1,
        "pid_to_hash_hex": {str(pid): pid_to_hash[pid].hex() for pid in range(N)},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return SealedEvent(ledger_path=ledger_path, evidence_path=evidence_path,
                       meta_path=meta_path)


def seal_from_evidence(evidence: Evidence, theta_v1: dict, out_dir, seed,
                       r_atk, r_inc, malicious_ids, **kw) -> SealedEvent:
    """Convenience: replay under theta_v1 then seal."""
    decision = replay(evidence, theta_v1)
    return seal_event(evidence, decision, theta_v1, out_dir, seed,
                      r_atk, r_inc, malicious_ids, **kw)
