"""
Certified Federated Repair (CFR) machinery used by the recovery experiments.

Implements certified federated repair as specified in the paper (Section IV-C):

- commitments h_delta(i,r) = SHA-256 over a canonical, version-pinned encoding
  of each admitted update (Def "Realized trace and anchor");
- a collector-enforced retention cache of update bytes with a sliding window
  of H_sim rounds (the declared retention horizon);
- requirement-set derivation for the recurrence execution mode under
  sigma = renorm-exclude (Def 3: R_rec = all non-excluded admitted cells after
  the anchor; artifact list is empty here, so B = R_rec);
- commitment-verified re-supply and exact committed-trace correction
  (Def 2 recurrence; the Lemma-1 linear shortcut is NOT used because the
  harness never seals full-precision realized weights);
- complete / incomplete CFR receipts (Def 4). An incomplete receipt enumerates
  the missing/unverified cells and declares its fallback PLAN (resume training
  from the anchor); which execution was actually taken is recorded once, at
  the run level, by the caller (several variants probe the same sealed trace
  but only one execution runs). It never silently degrades.

Receipt fields the host simulation cannot bind are recorded as explicit
"not-simulated" stand-ins (anchor inclusion proof, replay receipts for the
omega' derivation) rather than omitted or faked — the SGX pipeline, not this
harness, owns those objects.

Exactness is reported in two layers (plan discipline): (i) commitment/trace
exactness — encode/decode roundtrips are bitwise and C_N is computed under a
pinned execution order (float32, rounds ascending, clients ascending,
element-wise sequential accumulation, which is deterministic on GPU); and
(ii) a bitwise comparison of C_N against an independent numpy-CPU oracle that
replays the same pinned order (element-wise IEEE float32 ops are correctly
rounded, so CPU and GPU should agree bit-for-bit; any residual difference is
reported as max_abs_diff rather than asserted away).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

ENC_VERSION = 1
# Simulation stand-in for the authenticated ledger root: a flat hash over the
# sorted per-cell commitments (no Merkle tree in the simulator; the REPORT
# labels it as such).
LEDGER_ROOT_KIND = "flat_hash_of_commitments"


def _as_f4_bytes(x) -> bytes:
    """Canonical byte serialization: little-endian float32, C order."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(x), dtype="<f4")
    return arr.tobytes()


def canonical_encode(vec, *, job_id: str, rnd: int, client: int,
                     base_model_hash: str, agg: str = "fedavg") -> bytes:
    """Version-pinned canonical encoding enc(u_i(r)).

    The commitment binds job identity, round, participant, base-model hash,
    encoding version, and aggregation metadata — not just (i, r) — per the
    spec's re-supply verification requirements.
    """
    header = json.dumps(
        {"v": ENC_VERSION, "job": job_id, "r": int(rnd), "i": int(client),
         "base": base_model_hash, "agg": agg, "dtype": "<f4"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(4, "little") + header + _as_f4_bytes(vec)


def canonical_decode(enc: bytes) -> Tuple[dict, np.ndarray]:
    """Inverse of canonical_encode; the roundtrip is bitwise."""
    hlen = int.from_bytes(enc[:4], "little")
    header = json.loads(enc[4:4 + hlen].decode("utf-8"))
    vec = np.frombuffer(enc[4 + hlen:], dtype="<f4")
    return header, vec


def commitment_of(enc: bytes) -> str:
    return hashlib.sha256(enc).hexdigest()


def model_hash(flat) -> str:
    """Digest of a flat parameter vector under the canonical serialization."""
    return hashlib.sha256(_as_f4_bytes(flat)).hexdigest()


class CommitmentLedger:
    """Sealed commitments h_delta(i,r) for every ADMITTED update.

    Committed != certified honest: malicious clients' updates are committed
    too (the commitment proves identity with what the coordinator consumed,
    never benignness).
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.base_hash: Dict[int, str] = {}          # rnd -> H(M_{r-1})
        self.cells: Dict[Tuple[int, int], str] = {}  # (rnd, client) -> hash

    def record_round(self, rnd: int, mat: np.ndarray, base_model_hash: str) -> None:
        """Seal one round's [N, D] update matrix (clients in id order)."""
        self.base_hash[rnd] = base_model_hash
        for cid in range(mat.shape[0]):
            enc = canonical_encode(mat[cid], job_id=self.job_id, rnd=rnd,
                                   client=cid, base_model_hash=base_model_hash)
            self.cells[(rnd, cid)] = commitment_of(enc)

    def verify(self, rnd: int, cid: int, vec: np.ndarray) -> bool:
        """Accept a re-supplied vector iff H(enc(u~)) equals the sealed h_delta."""
        expect = self.cells.get((rnd, cid))
        if expect is None:
            return False
        enc = canonical_encode(vec, job_id=self.job_id, rnd=rnd, client=cid,
                               base_model_hash=self.base_hash[rnd])
        return commitment_of(enc) == expect

    def root(self) -> str:
        h = hashlib.sha256()
        for key in sorted(self.cells):
            h.update(self.cells[key].encode("ascii"))
        return h.hexdigest()


class UpdateCache:
    """Disk-backed per-round update cache with a sliding retention window.

    Simulates the collector-enforced retention policy: after storing round r,
    rounds <= r - window are evicted. This is a policy the collector enforces,
    NOT a statement about client willingness/availability (plan discipline);
    window-miss and client refusal are exercised as two independent factors at
    repair time. The cache is scratch — call cleanup() after repair; only the
    accounting numbers survive into the result JSON.
    """

    def __init__(self, cache_dir: str, window: int):
        self.dir = cache_dir
        self.window = int(window)
        self.bytes_per_round: Optional[int] = None
        self.n_clients: Optional[int] = None
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, rnd: int) -> str:
        return os.path.join(self.dir, f"round_{rnd:04d}.npy")

    def store_round(self, rnd: int, mat: np.ndarray) -> None:
        mat = np.ascontiguousarray(mat, dtype="<f4")
        np.save(self._path(rnd), mat)
        self.bytes_per_round = mat.nbytes
        self.n_clients = mat.shape[0]
        evict = rnd - self.window
        while evict >= 0 and os.path.exists(self._path(evict)):
            os.remove(self._path(evict))
            evict -= 1

    def has_round(self, rnd: int) -> bool:
        return os.path.exists(self._path(rnd))

    def load_round(self, rnd: int) -> np.ndarray:
        return np.load(self._path(rnd))

    def retained_bytes(self, H: int, r_terminal: int) -> int:
        """Storage the declared horizon H would retain at repair time:
        rounds (r_terminal - H, r_terminal], each [N, D] float32."""
        if self.bytes_per_round is None:
            return 0
        return self.bytes_per_round * min(int(H), r_terminal)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _missing_summary(missing: List[Tuple[int, int]]) -> List[dict]:
    """Compress the missing-cell enumeration into per-client round ranges
    (the sets that arise here are Cartesian products, so ranges are exact)."""
    by_client: Dict[int, List[int]] = {}
    for rnd, cid in missing:
        by_client.setdefault(cid, []).append(rnd)
    out = []
    for cid in sorted(by_client):
        rounds = sorted(by_client[cid])
        ranges, lo, prev = [], rounds[0], rounds[0]
        for r in rounds[1:]:
            if r == prev + 1:
                prev = r
                continue
            ranges.append([lo, prev])
            lo = prev = r
        ranges.append([lo, prev])
        out.append({"client": cid, "round_ranges": ranges})
    return out


def cfr_repair(
    ledger: CommitmentLedger,
    cache: UpdateCache,
    anchor_state: torch.Tensor,
    a: int,
    r_terminal: int,
    flagged: Set[int],
    n_clients: int,
    H: int,
    refuse_set: Set[int],
    device: torch.device,
    policy_params: Optional[dict] = None,
    weights: Optional[Dict[int, np.ndarray]] = None,
    sigma: str = "renorm-exclude",
) -> Tuple[dict, Optional[torch.Tensor]]:
    """One CFR execution attempt: plan -> re-supply -> verify -> correct -> receipt.

    Returns (info, C_N). C_N is None iff the receipt is incomplete; the receipt
    then carries fallback_planned (resume training from the anchor k_a — never
    a partial correction). The caller decides which attempt's outcome is the
    execution actually taken and records that separately.

    ``policy_params`` documents the omega'-derivation rule (e.g. tau_safe /
    k_persist of the trust-trajectory flag) so a verifier can re-derive the
    excluded set from the sealed evidence.
    """
    rounds = list(range(a + 1, r_terminal + 1))
    if (weights is not None) != (sigma == "reweight"):
        raise ValueError("sigma/weights mismatch: reweight iff weights given")
    if weights is not None:
        for rnd in rounds:
            w = weights.get(rnd)
            if w is None or len(w) != n_clients:
                raise ValueError(f"weights missing/misshaped for round {rnd}")
            if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"weights invalid (non-finite/negative/zero-mass) at round {rnd}")
    # Def 3 (recurrence mode): the dependency set is every admitted cell with
    # a nonzero corrected weight. renorm-exclude: kept clients x all rounds.
    # reweight (sigma="reweight", `weights` = per-round PRE-NORMALIZED f32
    # trust-gated weight vectors): exactly the cells with omega'_i(r) > 0 —
    # including not-yet-condemned adversary cells, so completeness depends on
    # their re-supply too (disclosed tradeoff vs renorm-exclude).
    # No aggregate artifacts -> B = R_rec.
    if weights is None:
        kept = sorted(set(range(n_clients)) - set(flagged))
        cells_by_round = {rnd: kept for rnd in rounds}
    else:
        cells_by_round = {
            rnd: [cid for cid in range(n_clients) if weights[rnd][cid] > 0]
            for rnd in rounds}
        kept = sorted({cid for cids in cells_by_round.values() for cid in cids})
    r_rec = [(rnd, cid) for rnd in rounds for cid in cells_by_round[rnd]]

    def available(rnd: int, cid: int) -> bool:
        if cid in refuse_set:                    # refusal factor
            return False
        if rnd <= r_terminal - H:                # window-miss factor
            return False
        return cache.has_round(rnd)

    missing = [(rnd, cid) for rnd, cid in r_rec if not available(rnd, cid)]

    receipt = {
        "sigma": sigma,
        "execution_mode": "recurrence",          # Lemma-1 shortcut not certified
        "anchor_round": a,
        "anchor_digest": model_hash(anchor_state),
        # The spec binds the anchor by a ledger inclusion proof; the host
        # simulation has no Merkle store, so this is an explicit stand-in.
        "anchor_inclusion_proof": "not-simulated (host sim; ledger_root is a "
                                  "flat commitment hash, no Merkle path)",
        "terminal_round": r_terminal,
        "retention_horizon_H": int(H),
        # omega' derivation record: rule + parameters that produced the
        # excluded set from the SEALED trust trajectory. The replay receipts
        # certifying that derivation exist in the SGX pipeline, not here.
        "omega_prime_derivation": dict(
            {"rule": ("flag_from_trajectory over sealed trust + " + sigma
                      + (" (weight-only counterfactual rule, no clipping — "
                         "NOT an M6 online-policy replay)" if weights is not None else "")),
             "replay_receipts": "not-simulated (trust-replay exactness is an "
                                "existing paper result; the sim consumes the "
                                "sealed trajectory directly)"},
            **(policy_params or {})),
        # omega'/S/R derivation record: S(r) admits ALL n_clients every round
        # (FedAvg phase); omega' = renorm-exclude over excluded_clients, so
        # R_rec = kept_clients x (anchor, terminal]. A verifier re-derives
        # R from these fields alone.
        "admitted_per_round": "all",
        "n_clients": int(n_clients),
        # under reweight nobody is zeroed by fiat: omega' comes from the
        # gated weights; the flag set is disclosed via omega_prime_derivation
        "excluded_clients": sorted(flagged) if weights is None else [],
        "kept_clients": kept,   # reweight: union of positive-weight clients
        "refused_clients": sorted(refuse_set),
        "n_cells_required": len(r_rec),
        "trivial_empty_dependency_set": len(r_rec) == 0,
        # Pinned execution specification (Prop 1 requires the receipt to fix
        # operation order, precision, and exceptional-value policy).
        "execution_spec": {
            "dtype": "float32",
            "rounding": "IEEE-754 roundTiesToEven (fp32 default arithmetic)",
            "order": "rounds ascending; clients ascending; element-wise "
                     "sequential accumulation; "
                     + ("one f32 multiply by 1/|kept|" if weights is None else
                        "per-cell f32 multiply by pre-normalized omega'"),
            "agg_version": "fedavg-mean-delta-v1",
            "exceptional_values": "none permitted (enforced: any non-finite "
                                  "supplied cell or C_N voids completeness)",
            "subnormals": "not rejected; CPU/GPU agreement is CHECKED per run "
                          "by the oracle cross-check (exact_bitwise/max_abs_diff), "
                          "never assumed",
        },
        "enc_version": ENC_VERSION,
        "model_enc": "flat-params <f4 C-order (model_hash)",
        "artifact_list": [],                     # no sealed aggregate artifacts
        "ledger_root": ledger.root(),
        "ledger_root_kind": LEDGER_ROOT_KIND,
    }
    info = {
        "H": int(H),
        "refused": sorted(refuse_set),
        "complete": False,
        "trivial_empty_dependency_set": len(r_rec) == 0,
        "missing_cells": len(missing),
        "resupplied_bytes": 0,
        "verify_vectors": 0,
        "verify_time_s": 0.0,
        "verify_vectors_per_s": None,
        "verify_mb_per_s": None,
        "correction_time_s": None,
        "exact_bitwise": None,
        "max_abs_diff": None,
        "c_n_digest": None,
        "retained_bytes": cache.retained_bytes(H, r_terminal),
        "receipt": receipt,
    }

    if missing:
        # Incomplete receipt (Def 4): enumerate missing cells and DECLARE the
        # fallback plan. Whether that fallback is the execution actually taken
        # is decided (and recorded as fallback_taken) by the caller — several
        # variants probe the same sealed trace, and only one execution runs.
        receipt["complete"] = False
        receipt["missing_cells"] = len(missing)
        receipt["missing_summary"] = _missing_summary(missing)
        receipt["fallback_planned"] = "resume_training_from_anchor"
        return info, None

    # ---- Re-supply + per-vector commitment verification (measured) ----
    # A cell whose bytes fail the commitment check is UNVERIFIED: per Def 4
    # it makes the receipt incomplete (enumerated separately from unavailable
    # cells) — it never aborts the run and never enters the correction.
    t0 = time.perf_counter()
    supplied: Dict[int, np.ndarray] = {}
    failed: List[Tuple[int, int]] = []
    nonfinite: List[Tuple[int, int]] = []
    total_bytes = 0
    for rnd in rounds:
        mat = cache.load_round(rnd)
        for cid in cells_by_round[rnd]:
            if not ledger.verify(rnd, cid, mat[cid]):
                failed.append((rnd, cid))
                continue
            if not np.isfinite(mat[cid]).all():   # enforce exceptional-value policy
                nonfinite.append((rnd, cid))
                continue
            total_bytes += mat[cid].nbytes
        supplied[rnd] = mat
    verify_time = time.perf_counter() - t0
    n_vec = len(r_rec) - len(failed)
    info.update(
        resupplied_bytes=total_bytes,
        verify_vectors=n_vec,
        verify_time_s=verify_time,
        verify_vectors_per_s=(n_vec / verify_time) if verify_time > 0 else None,
        verify_mb_per_s=(total_bytes / 1e6 / verify_time) if verify_time > 0 else None,
    )
    if failed or nonfinite:
        bad = failed + nonfinite
        receipt["complete"] = False
        receipt["missing_cells"] = len(bad)
        receipt["missing_summary"] = _missing_summary(bad)
        receipt["failed_verification_cells"] = len(failed)
        receipt["exceptional_value_cells"] = len(nonfinite)
        receipt["fallback_planned"] = "resume_training_from_anchor"
        info["missing_cells"] = len(bad)
        return info, None

    # ---- Committed-trace correction (Def 2 recurrence), pinned order ----
    # float32; rounds ascending; clients ascending; element-wise sequential
    # accumulation, then one division by |kept| (renorm-exclude weights).
    t0 = time.perf_counter()
    c = anchor_state.detach().to(device=device, dtype=torch.float32).clone()
    inv_k = np.float32(1.0) / np.float32(len(kept))  # pinned f32 constant
    for rnd in rounds:
        mat_t = torch.from_numpy(supplied[rnd]).to(device)
        acc = torch.zeros_like(c)
        for cid in cells_by_round[rnd]:
            if weights is None:
                acc += mat_t[cid]
            else:  # pre-normalized f32 scalar weight, correctly-rounded fmul+fadd
                acc += mat_t[cid] * float(np.float32(weights[rnd][cid]))
        c += acc * float(inv_k) if weights is None else acc
    correction_time = time.perf_counter() - t0

    # ---- Independent numpy-CPU oracle, same pinned order ----
    oracle = np.ascontiguousarray(
        anchor_state.detach().cpu().numpy(), dtype=np.float32).copy()
    for rnd in rounds:
        mat = supplied[rnd]
        acc = np.zeros_like(oracle)
        for cid in cells_by_round[rnd]:
            if weights is None:
                acc += mat[cid]
            else:
                acc += mat[cid] * np.float32(weights[rnd][cid])
        oracle += acc * inv_k if weights is None else acc
    c_np = c.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(c_np).all():
        # Exceptional-value policy enforced on the corrected trace itself:
        # a non-finite C_N voids completeness (never silently substituted).
        receipt["complete"] = False
        receipt["exceptional_values_in_c_n"] = True
        receipt["fallback_planned"] = "resume_training_from_anchor"
        return info, None
    exact_bitwise = bool(c_np.tobytes() == oracle.tobytes())
    max_abs_diff = float(np.max(np.abs(c_np.astype(np.float64) -
                                       oracle.astype(np.float64))))

    receipt["complete"] = True
    receipt["c_n_digest"] = model_hash(c_np)
    info.update(
        complete=True,
        correction_time_s=correction_time,
        exact_bitwise=exact_bitwise,
        max_abs_diff=max_abs_diff,
        c_n_digest=receipt["c_n_digest"],
    )
    return info, c
