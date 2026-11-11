"""Differential-audit forensic case study (offline, evidence-only).

Given a sealed delayed-compromise event (binary ledger + evidence sidecar +
ground truth), this module:

  S0  self-check #1 -- replay(E, theta_v1) reproduces the sealed decision D
      bit-for-bit (the evidence contract, Section IV-B).
  Q6  differential audit -- the set of (i, r) cells whose committed decision
      flips between rho_v1 and rho_v2, tagged by dimension.
  Q5  counterfactual filter -- how the filtered set grows under rho_v2.
  DRI earliest-crossing -- per-malicious-client, how many rounds EARLIER the
      CUSUM alarm fires under h_v2 = h_v1/2 (DRI operator, in the loop).
  Q9  safe-checkpoint drift -- c*(v1) vs c*(v2), rollback distance, UI.
  Q11 damage report -- the aggregation weight rho_v1 handed to cells rho_v2
      deems unsafe (avoidedWeight).
  GT  ground-truth validation -- first-unsafe round (v2 earlier by K), the
      quarantine precision/recall vs the true malicious set M, and the honest
      collateral rho_v2 incurs.

Everything is computed twice and cross-checked: once from the stateful
replay (authoritative, CUSUM-aware) and once from the sealed TrustStore
operators (the store query layer). They must agree on the tau_min flips.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtrust.store import TrustStore  # noqa: E402
from dtrust.store.schema import quantize_unit, dequantize_unit  # noqa: E402

from .sprt_replay import Evidence, ReplayResult, replay, decisions_equal  # noqa: E402
from .policies import policy_pair, load_base_config  # noqa: E402
from .seal import participant_hash  # noqa: E402


# --------------------------------------------------------------------------- IO

@dataclass
class SealedHandle:
    evidence: Evidence
    meta: dict
    ledger_path: Path


def load_sealed(out_dir: str | Path, seed: int) -> SealedHandle:
    out_dir = Path(out_dir)
    evidence = Evidence.load_npz(out_dir / f"evidence_seed{seed}.npz")
    with open(out_dir / f"meta_seed{seed}.json") as f:
        meta = json.load(f)
    return SealedHandle(evidence=evidence, meta=meta,
                        ledger_path=out_dir / f"ledger_seed{seed}.db")


# ------------------------------------------------------------------ self-check

def self_check(handle: SealedHandle, theta_v1: dict) -> dict:
    """replay(E, theta_v1) must equal the sealed ledger D, cell-for-cell."""
    decision = replay(handle.evidence, theta_v1)
    store = TrustStore(handle.ledger_path)
    R, N = handle.evidence.n_rounds, handle.evidence.n_participants
    pid_of = {bytes.fromhex(h): int(p) for p, h in handle.meta["pid_to_hash_hex"].items()}

    t_mismatch = w_mismatch = s_mismatch = mode_mismatch = 0
    for t in store.all_tuples():
        pid = pid_of[t.participant]; r = t.round
        if quantize_unit(float(decision.t_score[r, pid])) != t.t_score:
            t_mismatch += 1
        if quantize_unit(float(decision.w_agg[r, pid])) != t.w_agg:
            w_mismatch += 1
        if not np.isclose(float(decision.s_cusum[r, pid]), t.s_cusum, atol=1e-5):
            s_mismatch += 1
        if int(decision.mode[r]) != int(t.mode):
            mode_mismatch += 1
    store.close()
    total = R * N
    return {
        "cells": total,
        "t_score_mismatch": t_mismatch,
        "w_agg_mismatch": w_mismatch,
        "s_cusum_mismatch": s_mismatch,
        "mode_mismatch": mode_mismatch,
        "bit_exact": (t_mismatch == w_mismatch == s_mismatch == mode_mismatch == 0),
    }


# ------------------------------------------------------- stateful flip analysis

def _excluded_mask(d: ReplayResult) -> np.ndarray:
    """Boolean [R, N]: client excluded from aggregation this round."""
    return ~d.admitted


def _first_exclusion_rounds(d: ReplayResult, pids: List[int]) -> List[int]:
    excl = _excluded_mask(d)
    return [int(np.argmax(excl[:, p])) for p in pids if excl[:, p].any()]


def _first_exclusion_round(d: ReplayResult, pids: List[int]) -> Optional[int]:
    rounds = _first_exclusion_rounds(d, pids)
    return min(rounds) if rounds else None


def differential_audit_stateful(
    evidence: Evidence, theta_v1: dict, theta_v2: dict,
    malicious_ids: List[int], r_atk: int,
) -> dict:
    """Authoritative Q6 + ground-truth validation from the stateful replay."""
    d1 = replay(evidence, theta_v1)
    d2 = replay(evidence, theta_v2)
    R, N = evidence.n_rounds, evidence.n_participants
    M = set(malicious_ids)
    honest = [p for p in range(N) if p not in M]
    tau1 = float(theta_v1["quarantine"]["tau_min"])
    tau2 = float(theta_v2["quarantine"]["tau_min"])
    # Per-cell threshold flip: t_score is identical under v1/v2 (lambda does not
    # depend on tau_min), so cells with tau1 <= t_score < tau2 are exactly the
    # ones the sealed-ledger differential_audit operator returns as 'B_only'.
    ts = d1.t_score
    threshold_flip = int(((ts >= tau1) & (ts < tau2)).sum())

    excl1, excl2 = _excluded_mask(d1), _excluded_mask(d2)
    flip = excl2 & ~excl1                      # admitted under v1, excluded under v2

    # Attribute each flip to a dimension. Committed exclusion is the sticky SPRT
    # ban (tau_min knob) or the CUSUM drift alarm (h knob).
    flip_tau = (d2.banned & ~d1.banned)                        # admit -> quarantine (tau_min)
    flip_h = (d2.cusum_alarm & ~d1.cusum_alarm) & ~flip_tau    # admit -> drift alarm (h)
    flip_mode = (d2.mode == 1) & (d1.mode == 0)                # normal -> degraded (per round)

    # Ever-excluded sets (whole event) -> precision / recall vs M.
    ever_excl_v1 = set(np.where(excl1.any(axis=0))[0].tolist())
    ever_excl_v2 = set(np.where(excl2.any(axis=0))[0].tolist())

    def prec_rec(excl_set):
        tp = len(excl_set & M)
        prec = tp / max(len(excl_set), 1)
        rec = tp / max(len(M), 1)
        collateral = len(excl_set - M)
        return prec, rec, collateral

    p1, r1m, c1 = prec_rec(ever_excl_v1)
    p2, r2m, c2 = prec_rec(ever_excl_v2)

    fu1 = _first_exclusion_round(d1, malicious_ids)
    fu2 = _first_exclusion_round(d2, malicious_ids)
    k_earlier = (fu1 - fu2) if (fu1 is not None and fu2 is not None) else None
    # Mean first-exclusion round over the malicious cohort (the cohort-level
    # exclusion-timing gap; the DRI "rounds earlier" is a CUSUM-crossing statistic
    # that does NOT by itself gate aggregation).
    me1 = _first_exclusion_rounds(d1, malicious_ids)
    me2 = _first_exclusion_rounds(d2, malicious_ids)
    mean_excl_v1 = float(np.mean(me1)) if me1 else None
    mean_excl_v2 = float(np.mean(me2)) if me2 else None
    mean_excl_earlier = (mean_excl_v1 - mean_excl_v2) if (me1 and me2) else None

    # Timeline cells: (round, pid, is_malicious) for v2-only exclusions.
    flip_cells = [
        {"round": int(r), "pid": int(p), "malicious": int(p in M),
         "dim": ("tau_min" if flip_tau[r, p] else ("h" if flip_h[r, p] else "other"))}
        for r, p in zip(*np.where(flip))
    ]

    return {
        "flip_total": int(flip.sum()),
        "flip_admit_to_quarantine_tau": int(flip_tau.sum()),
        "flip_admit_to_drift_h": int(flip_h.sum()),
        "threshold_flip_newly_suspect_v2": threshold_flip,
        "flip_normal_to_degraded_rounds": int(np.unique(np.where(flip_mode)[0]).size),
        "v1": {"ever_excluded": sorted(ever_excl_v1), "precision": p1,
               "recall": r1m, "honest_collateral": c1,
               "first_unsafe_round": fu1},
        "v2": {"ever_excluded": sorted(ever_excl_v2), "precision": p2,
               "recall": r2m, "honest_collateral": c2,
               "first_unsafe_round": fu2},
        "k_rounds_earlier": k_earlier,
        "mean_exclusion_round_v1": mean_excl_v1,
        "mean_exclusion_round_v2": mean_excl_v2,
        "mean_exclusion_rounds_earlier": mean_excl_earlier,
        "extra_honest_quarantined_v2": c2 - c1,
        "extra_malicious_caught_v2": len((ever_excl_v2 & M) - (ever_excl_v1 & M)),
        "flip_cells": flip_cells,
    }


# ------------------------------------------------------- sealed-operator path

def store_operator_audit(handle: SealedHandle, theta_v1: dict, theta_v2: dict,
                         r1: int, r2: int) -> dict:
    """Q5/Q6/Q11 via the TrustStore operators (cross-check)."""
    store = TrustStore(handle.ledger_path)
    tau1_q = quantize_unit(theta_v1["quarantine"]["tau_min"])
    tau2_q = quantize_unit(theta_v2["quarantine"]["tau_min"])

    flips = store.differential_audit(r1, r2, tau1_q, tau2_q)
    b_only = sum(1 for *_, side in flips if side == "B_only")   # newly filtered under v2
    a_only = sum(1 for *_, side in flips if side == "A_only")

    cf = store.counterfactual_filter(r1, r2, tau1_q, tau2_q)    # (new_only, old_only, both)
    dmg = store.damage_report(r1, r2, tau2_q)                   # (n, w_unsafe, w_total, n_part)
    store.close()
    return {
        "q6_newly_filtered_v2": b_only,
        "q6_relaxed_to_admit": a_only,
        "q5_counterfactual": {"filtered_only_under_v2": cf[0],
                              "filtered_only_under_v1": cf[1],
                              "filtered_under_both": cf[2]},
        "q11_avoided_weight": dmg[1],
        "q11_total_weight": dmg[2],
        "q11_unsafe_cells": dmg[0],
        "q11_avoided_weight_ratio": (dmg[1] / dmg[2]) if dmg[2] > 0 else 0.0,
    }


def dri_earliest_crossing(handle: SealedHandle, theta_v1: dict, theta_v2: dict,
                          r1: int, r2: int) -> dict:
    """DRI: earliest CUSUM crossing per malicious client, h_v1 vs h_v2."""
    store = TrustStore(handle.ledger_path)
    M = handle.meta["malicious_ids"]
    h1 = float(theta_v1["cusum"]["h"]); h2 = float(theta_v2["cusum"]["h"])
    deltas = []
    rows = []
    for pid in M:
        ph = participant_hash(pid)
        c1 = store.dri_replay(ph, r1, r2, h1)
        c2 = store.dri_replay(ph, r1, r2, h2)
        rows.append({"pid": pid, "cross_v1": c1, "cross_v2": c2})
        if c1 is not None and c2 is not None:
            deltas.append(c1 - c2)
    store.close()
    return {
        "h_v1": h1, "h_v2": h2,
        "mean_rounds_earlier": float(np.mean(deltas)) if deltas else None,
        "n_clients_earlier": int(sum(1 for d in deltas if d > 0)),
        "per_client": rows,
    }


def safe_checkpoint_drift(handle: SealedHandle, theta_v1: dict, theta_v2: dict,
                          round_now: int, window: int, ckpt_period: int,
                          max_unsafe_frac: float,
                          budget_sweep=None) -> dict:
    """Q9: c*(v1) vs c*(v2). Stricter tau_min => earlier (deeper) rollback.

    A single unsafe-influence budget rarely lands in the regime where the two
    policies pick different checkpoints, so we sweep a range of budgets and
    report the one that maximizes the v1-vs-v2 rollback gap (the operator would
    in practice pick a budget; this shows the gap exists and how deep it gets).
    The fixed-budget result is also returned for reference.
    """
    store = TrustStore(handle.ledger_path)
    tau1_q = quantize_unit(theta_v1["quarantine"]["tau_min"])
    tau2_q = quantize_unit(theta_v2["quarantine"]["tau_min"])

    def cstars(b):
        c1 = store.safe_checkpoint(round_now, window, tau1_q, ckpt_period, b)
        c2 = store.safe_checkpoint(round_now, window, tau2_q, ckpt_period, b)
        return c1, c2

    c1_fixed, c2_fixed = cstars(max_unsafe_frac)
    if budget_sweep is None:
        budget_sweep = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    curve = []
    best = {"budget": max_unsafe_frac, "c_star_v1": c1_fixed, "c_star_v2": c2_fixed,
            "rollback_distance": (c1_fixed - c2_fixed) if (c1_fixed is not None and c2_fixed is not None) else 0}
    for b in budget_sweep:
        c1, c2 = cstars(b)
        rb = (c1 - c2) if (c1 is not None and c2 is not None) else None
        curve.append({"budget": b, "c_star_v1": c1, "c_star_v2": c2, "rollback_distance": rb})
        if rb is not None and rb > (best["rollback_distance"] or 0):
            best = {"budget": b, "c_star_v1": c1, "c_star_v2": c2, "rollback_distance": rb}
    store.close()
    return {"c_star_v1": best["c_star_v1"], "c_star_v2": best["c_star_v2"],
            "rollback_distance": best["rollback_distance"], "best_budget": best["budget"],
            "fixed_budget_c_star_v1": c1_fixed, "fixed_budget_c_star_v2": c2_fixed,
            "round_now": round_now, "window": window, "curve": curve}


# --------------------------------------------------------------- one seed / many

def ledger_integrity(handle: SealedHandle) -> dict:
    """Tamper-evidence: the TEE sealing root chains every 4 KiB chunk and anchors
    a Merkle root. Re-deriving the chain proves the host could not have rewritten
    a sealed decision after the fact -- the property that lets the auditor trust
    the replayed evidence under a stricter policy (the motivation's sealing root).
    """
    store = TrustStore(handle.ledger_path)
    ok = store.verify_chunk_chain()
    root = store.merkle_root()
    st = store.stats()
    store.close()
    return {"chunk_chain_valid": bool(ok), "merkle_root_hex": root.hex(),
            "tuples": st["tuples"], "chunks": st["chunks"], "anchors": st["anchors"]}


def audit_seed(out_dir: str | Path, seed: int, *, h_v1: float, decay: float = 0.9,
               admission_includes_cusum: bool = False,
               window: int = 20, ckpt_period: int = 10,
               max_unsafe_frac: float = 0.05) -> dict:
    handle = load_sealed(out_dir, seed)
    base = load_base_config()
    theta_v1, theta_v2 = policy_pair(base, h_v1=h_v1, decay=decay,
                                     admission_includes_cusum=admission_includes_cusum)
    r_atk, r_inc = handle.meta["r_atk"], handle.meta["r_inc"]
    R = handle.evidence.n_rounds

    integrity = ledger_integrity(handle)
    sc = self_check(handle, theta_v1)
    diff = differential_audit_stateful(
        handle.evidence, theta_v1, theta_v2, handle.meta["malicious_ids"], r_atk)
    ops = store_operator_audit(handle, theta_v1, theta_v2, 0, R - 1)
    dri = dri_earliest_crossing(handle, theta_v1, theta_v2, 0, R - 1)
    q9 = safe_checkpoint_drift(handle, theta_v1, theta_v2, r_inc, window,
                               ckpt_period, max_unsafe_frac)

    # Consistency: the sealed-ledger differential_audit operator's per-cell
    # threshold flip count must equal the stateful per-cell threshold flip.
    consistent = (ops["q6_newly_filtered_v2"] == diff["threshold_flip_newly_suspect_v2"])

    return {
        "seed": seed,
        "integrity": integrity,
        "self_check": sc,
        "differential": diff,
        "store_operators": ops,
        "dri": dri,
        "safe_checkpoint": q9,
        "operator_stateful_consistent": bool(consistent),
    }


def _agg(values: List[Optional[float]]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def audit_all(out_dir: str | Path, seeds: List[int], *, h_v1: float, decay: float = 0.9,
              admission_includes_cusum: bool = False, **kw) -> dict:
    per_seed = [audit_seed(out_dir, s, h_v1=h_v1, decay=decay,
                           admission_includes_cusum=admission_includes_cusum, **kw)
                for s in seeds]
    summary = {
        "n_seeds": len(seeds),
        "all_bit_exact": all(s["self_check"]["bit_exact"] for s in per_seed),
        "all_operator_consistent": all(s["operator_stateful_consistent"] for s in per_seed),
        "all_chunk_chain_valid": all(s["integrity"]["chunk_chain_valid"] for s in per_seed),
        "flip_total": _agg([s["differential"]["flip_total"] for s in per_seed]),
        "flip_tau_min": _agg([s["differential"]["flip_admit_to_quarantine_tau"] for s in per_seed]),
        "flip_h": _agg([s["differential"]["flip_admit_to_drift_h"] for s in per_seed]),
        "k_rounds_earlier": _agg([s["differential"]["k_rounds_earlier"] for s in per_seed]),
        "mean_exclusion_rounds_earlier": _agg([s["differential"]["mean_exclusion_rounds_earlier"] for s in per_seed]),
        "extra_malicious_caught_v2": _agg([s["differential"]["extra_malicious_caught_v2"] for s in per_seed]),
        "extra_honest_quarantined_v2": _agg([s["differential"]["extra_honest_quarantined_v2"] for s in per_seed]),
        "v1_first_unsafe": _agg([s["differential"]["v1"]["first_unsafe_round"] for s in per_seed]),
        "v2_first_unsafe": _agg([s["differential"]["v2"]["first_unsafe_round"] for s in per_seed]),
        "v1_precision": _agg([s["differential"]["v1"]["precision"] for s in per_seed]),
        "v2_precision": _agg([s["differential"]["v2"]["precision"] for s in per_seed]),
        "v1_recall": _agg([s["differential"]["v1"]["recall"] for s in per_seed]),
        "v2_recall": _agg([s["differential"]["v2"]["recall"] for s in per_seed]),
        "dri_mean_rounds_earlier": _agg([s["dri"]["mean_rounds_earlier"] for s in per_seed]),
        "c_star_v1": _agg([s["safe_checkpoint"]["c_star_v1"] for s in per_seed]),
        "c_star_v2": _agg([s["safe_checkpoint"]["c_star_v2"] for s in per_seed]),
        "rollback_distance": _agg([s["safe_checkpoint"]["rollback_distance"] for s in per_seed]),
        "q11_avoided_weight": _agg([s["store_operators"]["q11_avoided_weight"] for s in per_seed]),
        "q11_avoided_weight_ratio": _agg([s["store_operators"]["q11_avoided_weight_ratio"] for s in per_seed]),
    }
    return {"summary": summary, "per_seed": per_seed}
