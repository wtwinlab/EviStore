#!/usr/bin/env python3
"""Tier-tight exact-replayability round-trip on REAL Fashion-MNIST evidence.

This checks the determinacy result (T1) and the advisor on real data:
on a REAL FL delayed-compromise run, we seal ONLY the advisor's minimal sufficient
statistic S for each policy family, discard everything else, replay a counterfactual
policy θ_v2 in that family, and check the decision matches the full-evidence ground
truth `replay(E, θ_v2)` AT THE uint16 LEDGER RESOLUTION (quantized-exact).

Unlike `e2_second_policy.py` (which replays different policy *families* from the FULL
sealed evidence), this isolates the *minimal* S the advisor prescribes per family, and
exhibits the determinacy BOUNDARY: the current production tuple ({t_score,...}) is
*insufficient* for the weight/decay family (λ saturates at the clamp under the real
attack ⇒ the per-step increment is unrecoverable from a sealed t_score), while sealing
the per-channel LLRs is *exactly* sufficient. Real input is the recovery-run dump
mapped to Eq.8 evidence (real gradient-cosine is the load-bearing channel).

Families (all replays compared vs `replay(E, θ_v2)` = single source of truth):
  F1 τ_min      : S = sealed per-round t_score (current tuple)         → replayable now
  F2 weight/decay: S = sealed 5 clamped per-channel LLRs               → needs augmentation
                   NEGATIVE control: reconstruct from t_score-only      → fails at λ-clamp
  F3 drift any-h': S = sealed s_cusum trajectory (and residuals/DRI)   → any-h replayable

Pure CPU, reuses real evidence npz; no GPU, no retraining.
"""
import glob
import importlib.util as u
import json
import math
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load(name, rel):
    s = u.spec_from_file_location(name, ROOT / rel)
    m = u.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


# Stub parent packages so absolute `from dtrust...` imports resolve to file-loaded
# modules WITHOUT running dtrust/__init__ (which pulls torch).
for pkg in ("dtrust", "dtrust.trust", "dtrust.store"):
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        sys.modules[pkg] = mod

_llr = _load("dtrust.trust._llr", "dtrust/trust/_llr.py")
schema = _load("dtrust.store.schema", "dtrust/store/schema.py")
dri = _load("dtrust.store.dri_dyadic", "dtrust/store/dri_dyadic.py")
SR = _load("sprt_replay", "experiment_scripts/dtrust_experiments/c1_audit/sprt_replay.py")
pol = _load("policies", "experiment_scripts/dtrust_experiments/c1_audit/policies.py")

qz = np.vectorize(schema.quantize_unit)


def _theta(base, tau_min, h, decay):
    return pol._with_thresholds(base, tau_min=tau_min, cusum_h=h, decay=decay)


# ---- per-family minimal-S reconstructions ---------------------------------------------
def recon_f1_full(sealed_tscore_q, base, tau_min):
    """F1: re-decide the FULL decision machine (suspect/event/banned/admitted/w_agg) under a
    new tau_min from the sealed per-round t_score alone. tau_min ↦ lambda_min=logit(tau_min)
    drives the quarantine-streak / ban machine; λ itself does NOT depend on tau_min, and the
    ban-relevant λ region is far from the sigmoid tail, so logit(quantized t_score) recovers
    the clamped λ that replay compares — the whole machine is replayable from the current tuple."""
    R, N = sealed_tscore_q.shape
    ts = sealed_tscore_q.astype(np.float64) / 65535.0
    ts_c = np.clip(ts, 1e-12, 1 - 1e-12)
    lam = np.log(ts_c / (1 - ts_c))                    # recovered clamped λ
    lam_min = math.log(tau_min / (1 - tau_min))
    lam_max = base["thresholds"]["lambda_max"]
    susp_max = base["quarantine"]["susp_max"]
    suspect = ts < tau_min
    event = np.zeros((R, N), dtype=np.int8)
    banned = np.zeros((R, N), dtype=bool)
    admitted = np.zeros((R, N), dtype=bool)
    w_agg = np.zeros((R, N))
    streak = np.zeros(N, dtype=int); is_banned = np.zeros(N, dtype=bool)
    EV = {"quarantine": 1, "ban": 2, "readmit": 3}
    for r in range(R):
        for p in range(N):
            evt = 0
            if lam[r, p] <= lam_min:
                streak[p] += 1
                evt = EV["ban"] if streak[p] >= susp_max else EV["quarantine"]
                if streak[p] >= susp_max:
                    is_banned[p] = True
            elif lam[r, p] >= lam_max:
                streak[p] = 0; evt = EV["readmit"]
            event[r, p] = evt
            banned[r, p] = is_banned[p]
            admitted[r, p] = not is_banned[p]
            w_agg[r, p] = ts[r, p] if admitted[r, p] else 0.0
    return suspect, event, banned, admitted, w_agg


def seal_llrs(E, base):
    """Compute the 5 clamped per-channel LLRs per (r,p) under the (family-invariant)
    channel priors — the advisor's weight/decay augmentation S."""
    ch = base["channels"]
    clamp = base["thresholds"]["clamp_per_llr"]
    R, N = E.n_rounds, E.n_participants
    L = np.zeros((R, N, 5))
    for r in range(R):
        for p in range(N):
            ev = E.round_dict(r, p)
            L[r, p] = SR._llr_vector(ev, ch, clamp)
    return L


def recon_from_llrs(L, base, decay):
    """F2 (positive): re-run the leaky SPRT recursion under a new decay from sealed LLRs
    only (no raw evidence). eta=0 ⇒ no AdvDev penalty; matches the deployed config."""
    R, N, _ = L.shape
    clamp_lam = base["thresholds"]["clamp_lambda"]
    lam_min = base["thresholds"]["lambda_min"]
    susp_max = base["quarantine"]["susp_max"]
    inc = L.sum(axis=2)
    lam = np.zeros((R, N)); ts = np.zeros((R, N)); w = np.zeros((R, N))
    prev = np.zeros(N); streak = np.zeros(N, dtype=int); banned = np.zeros(N, dtype=bool)
    for r in range(R):
        v = np.clip(decay * prev + inc[r], -clamp_lam, clamp_lam)
        prev = v
        for p in range(N):
            if v[p] <= lam_min:
                streak[p] += 1
                if streak[p] >= susp_max:
                    banned[p] = True
            elif v[p] >= base["thresholds"]["lambda_max"]:
                streak[p] = 0
        s = 1.0 / (1.0 + np.exp(-np.clip(v, -700, 700)))
        lam[r] = v; ts[r] = s
        w[r] = np.where(banned, 0.0, s)
    return ts, w


def recon_from_tscore_only(sealed_ts, base, decay_v1, decay_v2, true_lam):
    """F2 (NEGATIVE control): try to reconstruct the decay-v2 t_score from the sealed
    per-round t_score ALONE (current tuple). Invert sigmoid→λ_v1, recover the increment
    inc=λ_v1(t)-decay_v1·λ_v1(t-1), re-run under decay_v2.

    This is the determinacy boundary in action. BOTH write-time projections are lossy:
      (1) λ_t = clamp(decay·λ_{t-1}+inc_t)  — DOMINANT: a clamped λ does not reveal the
          pre-clamp increment, so wherever λ saturates the increment is unrecoverable;
      (2) t_score = quantize(sigmoid(λ))    — secondary: ill-conditioned in the high-confidence
          sigmoid tail where logit blows up.
    Either way the increment is NOT a function of the sealed t_score (π_S-measurability fails)
    and the decay reconstruction diverges. We report the clamp-saturation fraction (on the TRUE
    λ), the sigmoid-tail fraction, and the λ-recovery MAE."""
    R, N = sealed_ts.shape
    clamp_lam = base["thresholds"]["clamp_lambda"]
    eps = 1e-12
    ts_q = sealed_ts.astype(np.float64) / 65535.0
    ts_c = np.clip(ts_q, eps, 1 - eps)
    lam_v1 = np.log(ts_c / (1 - ts_c))                # logit of the (quantized) sealed t_score
    inc = np.zeros((R, N))
    prev = np.zeros(N)
    for r in range(R):
        inc[r] = lam_v1[r] - decay_v1 * prev
        prev = lam_v1[r]
    # re-run under decay_v2 with the (lossily) recovered increments
    lam2 = np.zeros((R, N)); prev2 = np.zeros(N)
    for r in range(R):
        v = np.clip(decay_v2 * prev2 + inc[r], -clamp_lam, clamp_lam)
        prev2 = v; lam2[r] = v
    ts2 = 1.0 / (1.0 + np.exp(-np.clip(lam2, -700, 700)))
    clamp_sat = float(np.mean(np.abs(true_lam) >= clamp_lam - 1e-6))  # DOMINANT loss (on TRUE λ)
    tail_frac = float(np.mean((ts_q < 1e-3) | (ts_q > 1 - 1e-3)))     # secondary loss
    lam_recovery_mae = float(np.mean(np.abs(lam_v1 - true_lam)))
    return ts2, clamp_sat, tail_frac, lam_recovery_mae


def channels_recoverable_from_full_tuple(E, base, gt):
    """How much of the increment does the FULL current tuple {t_score,w_agg,s_cusum,b_score,
    a_conf} reveal? a_conf=sigmoid(l_a) and b_score=sigmoid(l_b) recover exactly 2 of the 5
    per-channel clamped LLRs; grad/perf/seal are absent. So even the full tuple cannot form the
    increment inc=Σ5 LLRs — confirming the negative control is not an artifact of using t_score
    only. Returns (#channels exactly recovered out of 5)."""
    ch = base["channels"]; cl = base["thresholds"]["clamp_per_llr"]
    R, N = E.n_rounds, E.n_participants
    la = np.zeros((R, N)); lb = np.zeros((R, N))
    for r in range(R):
        for p in range(N):
            ev = E.round_dict(r, p); v = SR._llr_vector(ev, ch, cl)
            la[r, p] = v[0]; lb[r, p] = v[3]
    # use the SEALED (uint16-quantized) a_conf/b_score, not in-memory floats: recovery is exact at
    # LEDGER precision (max LLR error ~5e-5 from the uint16 round-trip), not bit-exact.
    ac = np.clip(qz(gt.a_conf) / 65535.0, 1e-12, 1 - 1e-12)
    bc = np.clip(qz(gt.b_score) / 65535.0, 1e-12, 1 - 1e-12)
    LEDGER_TOL = 1e-3                                   # ledger-resolution tolerance (>> 5e-5)
    rec_a = np.mean(np.abs(np.log(ac / (1 - ac)) - la) < LEDGER_TOL)
    rec_b = np.mean(np.abs(np.log(bc / (1 - bc)) - lb) < LEDGER_TOL)
    return int(round(float(rec_a))) + int(round(float(rec_b)))   # grad/perf/seal not in tuple


def earliest_crossing_from_scusum(sealed_S, h):
    """F3: earliest CUSUM crossing per client by thresholding the sealed S trajectory."""
    R, N = sealed_S.shape
    out = np.full(N, R, dtype=int)
    for p in range(N):
        w = np.where(sealed_S[:, p] >= h)[0]
        if len(w):
            out[p] = int(w[0])
    return out


def earliest_crossing_from_dri(increments, h):
    """F3 cross-check: earliest crossing via the dyadic reflected-walk DRI on residuals
    r_t = -inc (replay uses S=max(0,S-inc) ⇔ W=max(0,W+r))."""
    R, N = increments.shape
    out = np.full(N, R, dtype=int)
    for p in range(N):
        resid = [-float(increments[r, p]) for r in range(R)]
        d = dri.DyadicDRI(resid)                        # O(log R) dyadic reflected-walk DRI
        t = d.earliest_crossing(h, 0, R)
        out[p] = t if (t is not None and t >= 0) else R
    return out


def run_one(ev_path, base):
    E = SR.Evidence.load_npz(ev_path)
    R, N = E.n_rounds, E.n_participants
    decay = pol.DECAY_DEFAULT
    H = pol.H_V1_DEFAULT
    v1 = _theta(base, 0.30, H, decay)
    gt1 = SR.replay(E, v1)                              # online ground truth (θ_v1)

    # F1 τ_min: counterfactual changes ONLY tau_min; reconstruct the FULL decision machine
    # (suspect+event+banned+admitted+w_agg) over a SWEEP of tau_min — empirical ledger-resolution
    # round-trip for the whole sampled range (not just one tau).
    f1_mismatch = 0
    for tau2 in (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95):
        gt_tau = SR.replay(E, _theta(base, tau2, H, decay))
        r_susp, r_evt, r_ban, r_adm, r_w = recon_f1_full(qz(gt1.t_score), base, tau2)
        f1_mismatch += int(np.sum(r_susp != gt_tau.suspect)
                           + np.sum(r_evt != gt_tau.event)
                           + np.sum(r_ban != gt_tau.banned)
                           + np.sum(r_adm != gt_tau.admitted)
                           + np.sum(qz(r_w) != qz(gt_tau.w_agg)))

    # F2 weight/decay: counterfactual changes ONLY decay (0.9 → 0.8)
    decay_v2 = 0.8
    v2_dec = _theta(base, 0.30, H, decay_v2)
    gt_dec = SR.replay(E, v2_dec)
    L = seal_llrs(E, base)
    ts_llr, w_llr = recon_from_llrs(L, base, decay_v2)
    f2_pos_ts = int(np.sum(qz(ts_llr) != qz(gt_dec.t_score)))
    f2_pos_w = int(np.sum(qz(w_llr) != qz(gt_dec.w_agg)))
    # NEGATIVE control: reconstruct from sealed t_score ONLY
    ts_neg, clamp_sat, tail_frac, lam_mae = recon_from_tscore_only(qz(gt1.t_score), base, decay, decay_v2, gt1.lam)
    f2_neg_ts = int(np.sum(qz(ts_neg) != qz(gt_dec.t_score)))
    # full-tuple baseline: how many of the 5 channel LLRs the WHOLE current tuple reveals
    full_tuple_channels = channels_recoverable_from_full_tuple(E, base, gt1)

    # F3 drift any-h': counterfactual changes ONLY h (4.0 → 2.0)
    h2 = H / 2.0
    v2_h = _theta(base, 0.30, h2, decay)
    gt_h = SR.replay(E, v2_h)
    truth_cross = np.full(N, R, dtype=int)
    for p in range(N):
        w = np.where(gt_h.cusum_alarm[:, p])[0]
        if len(w):
            truth_cross[p] = int(w[0])
    cross_scusum = earliest_crossing_from_scusum(gt1.s_cusum, h2)
    cross_dri = earliest_crossing_from_dri(gt1.increment, h2)
    f3_scusum_mismatch = int(np.sum(cross_scusum != truth_cross))
    f3_dri_mismatch = int(np.sum(cross_dri != truth_cross))

    return {
        "rounds": R, "participants": N,
        "F1_tau_mismatch": f1_mismatch,
        "F2_decay_from_LLRs_tscore_mismatch": f2_pos_ts,
        "F2_decay_from_LLRs_wagg_mismatch": f2_pos_w,
        "F2_decay_from_tscore_only_mismatch": f2_neg_ts,
        "F2_lambda_clamp_saturation_frac": round(clamp_sat, 4),
        "F2_sigmoid_tail_frac": round(tail_frac, 4),
        "F2_tscore_only_lambda_recovery_mae": round(lam_mae, 4),
        "F2_full_tuple_channels_recovered_of_5": full_tuple_channels,
        "F3_anyh_from_scusum_mismatch": f3_scusum_mismatch,
        "F3_anyh_from_DRI_mismatch": f3_dri_mismatch,
    }


def main():
    base = pol.load_base_config()
    globs = [
        "results/recovery/c1_5seed_audit_int1.0/evidence_seed*.npz",
        "results/recovery/c1_5seed_audit_int0.7/evidence_seed*.npz",
        "results/recovery/c1_mnist_audit/evidence_seed*.npz",
    ]
    files = []
    for g in globs:
        files = sorted(glob.glob(str(ROOT / g)))
        if files:
            chosen = g
            break
    if not files:
        sys.exit("no real evidence npz found")

    per = []
    for f in files:
        r = run_one(f, base)
        per.append(r)
        print(f"{Path(f).parent.name}/{Path(f).name}: "
              f"F1_tau={r['F1_tau_mismatch']} "
              f"F2_LLR(ts/w)={r['F2_decay_from_LLRs_tscore_mismatch']}/{r['F2_decay_from_LLRs_wagg_mismatch']} "
              f"F2_tscore_only={r['F2_decay_from_tscore_only_mismatch']} (clampsat={r['F2_lambda_clamp_saturation_frac']},tail={r['F2_sigmoid_tail_frac']},chan={r['F2_full_tuple_channels_recovered_of_5']}/5) "
              f"F3_scusum/dri={r['F3_anyh_from_scusum_mismatch']}/{r['F3_anyh_from_DRI_mismatch']}")

    n_cells = sum(p["rounds"] * p["participants"] for p in per)
    agg = {
        "n_seeds": len(per), "total_cells": n_cells, "source_glob": chosen,
        "F1_tau_total_mismatch": sum(p["F1_tau_mismatch"] for p in per),
        "F2_LLR_tscore_total_mismatch": sum(p["F2_decay_from_LLRs_tscore_mismatch"] for p in per),
        "F2_LLR_wagg_total_mismatch": sum(p["F2_decay_from_LLRs_wagg_mismatch"] for p in per),
        "F2_tscore_only_total_mismatch": sum(p["F2_decay_from_tscore_only_mismatch"] for p in per),
        "F2_mean_clamp_saturation_frac": round(float(np.mean([p["F2_lambda_clamp_saturation_frac"] for p in per])), 4),
        "F2_mean_sigmoid_tail_frac": round(float(np.mean([p["F2_sigmoid_tail_frac"] for p in per])), 4),
        "F2_mean_lambda_recovery_mae": round(float(np.mean([p["F2_tscore_only_lambda_recovery_mae"] for p in per])), 4),
        "F2_full_tuple_channels_recovered_of_5": per[0]["F2_full_tuple_channels_recovered_of_5"],
        "F3_scusum_total_mismatch": sum(p["F3_anyh_from_scusum_mismatch"] for p in per),
        "F3_DRI_total_mismatch": sum(p["F3_anyh_from_DRI_mismatch"] for p in per),
    }
    agg["VERDICT"] = {
        "F1_tau_replayable_from_current_tuple": agg["F1_tau_total_mismatch"] == 0,
        "F2_decay_replayable_from_sealed_LLRs": agg["F2_LLR_tscore_total_mismatch"] == 0
                                                and agg["F2_LLR_wagg_total_mismatch"] == 0,
        "F2_decay_NOT_replayable_from_tscore_only": agg["F2_tscore_only_total_mismatch"] > 0,
        "F3_anyh_replayable_from_scusum": agg["F3_scusum_total_mismatch"] == 0,
        "F3_anyh_replayable_from_DRI_residuals": agg["F3_DRI_total_mismatch"] == 0,
    }
    out = {"_note": "Tier-tight exact-replayability round-trip on REAL Fashion-MNIST "
                    "delayed-compromise evidence. Each family's minimal advisor-S is sealed, "
                    "a counterfactual policy in that family is replayed, and the decision is "
                    "compared vs replay(E,θ_v2) at uint16 ledger resolution. F2 also shows the "
                    "determinacy BOUNDARY: t_score-only (current tuple) cannot replay weight/decay "
                    "(λ-clamp ~59% dominant + sigmoid quantization), but sealed LLRs can.",
           "_provenance": {
               "data_kind": "REAL federated run (not synthetic)",
               "training": "scripts/run_recovery_experiment.py (recovery runner), Fashion-MNIST, "
                           "100 clients, 151 rounds, trim attack, 20 malicious, r_atk=50, r_inc=150, "
                           "5 seeds; --dump_ledger_dir writes per-(round,client) real cos-to-median + norm",
               "raw_dump": "results/recovery/c1_5seed_int1.0/dump_seed{0..4}.npz",
               "evidence_mapping": "c1_audit/real_event.py:evidence_from_dump (measured-client threat "
                                   "model: real grad-cosine is the load-bearing channel; att/seal blind)",
               "sealed_evidence": chosen,
               "decision_fn": "c1_audit/sprt_replay.replay(E,theta) — single source of truth"},
           "per_seed": per, "aggregate": agg}
    print("\nVERDICT:", json.dumps(agg["VERDICT"], indent=2))
    od = ROOT / "results/artifact"
    (od / "L7_exactness_roundtrip.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "L7_exactness_roundtrip.json")


if __name__ == "__main__":
    main()
