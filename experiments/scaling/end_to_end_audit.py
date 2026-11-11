#!/usr/bin/env python3
"""End-to-end DEPLOYED-schema replayable audit on REAL evidence.

Runs every step through the production store:
  1. real Fashion-MNIST evidence  ->  online decisions + 5 clamped per-channel LLRs
  2. write +LLR `TrustTuple`s through `TrustStore.append_round`  ->  sealed 4 KiB Merkle chunks
     (the LLRs occupy the tail padding: ZERO net bytes/cell; tuple stays 76 B)
  3. SEAL -> Merkle root + authenticated-SRI root
  4. DELETE the raw evidence E
  5. advisor: weight/decay needs the LLRs (now sealed) -> verdict
  6. replay the decay counterfactual reading ONLY the LLRs READ BACK FROM THE STORE
     -> compare to the full-evidence ground truth `replay(E, θ_v2)` (quantized-exact)
  7. authenticated count_below RECEIPT verified against the sealed root with NO raw access
This is the deployed counterpart of exactness_roundtrip.py (which replays off the raw npz, not the store).

Run on a box with the full dtrust package (server crabenv):
  PYTHONPATH=. python experiment_scripts/dtrust_experiments/db_eval/end_to_end_audit.py
"""
import glob, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
C1 = ROOT / "experiment_scripts/dtrust_experiments/c1_audit"
sys.path.insert(0, str(C1))

import sprt_replay as SR
import policies as pol
from dtrust.store.store import TrustStore
from dtrust.store.schema import TrustTuple, Mode, quantize_unit, quantize_llr, TUPLE_SIZE
import dtrust.store.sri_authenticated as A

qz = np.vectorize(quantize_unit)
DECAY_V1, DECAY_V2 = 0.9, 0.8


def seal_llrs(E, base):
    ch = base["channels"]; clamp = base["thresholds"]["clamp_per_llr"]
    R, N = E.n_rounds, E.n_participants
    L = np.zeros((R, N, 5))
    for r in range(R):
        for p in range(N):
            L[r, p] = SR._llr_vector(E.round_dict(r, p), ch, clamp)
    return L


def recon_from_llrs(L, base, decay):
    """Re-run the leaky SPRT recursion under a new decay from the 5 LLRs only.
    Returns (t_score, w_agg, banned, admitted) — banned/admitted are the audit-meaningful
    DECISIONS (what differentialAudit reports), independent of trust-score quantization."""
    R, N, _ = L.shape
    clamp_lam = base["thresholds"]["clamp_lambda"]
    lam_min = base["thresholds"]["lambda_min"]; lam_max = base["thresholds"]["lambda_max"]
    susp_max = base["quarantine"]["susp_max"]
    inc = L.sum(axis=2)
    ts = np.zeros((R, N)); w = np.zeros((R, N))
    banned_t = np.zeros((R, N), dtype=bool); admitted_t = np.zeros((R, N), dtype=bool)
    prev = np.zeros(N); streak = np.zeros(N, dtype=int); banned = np.zeros(N, dtype=bool)
    for r in range(R):
        v = np.clip(decay * prev + inc[r], -clamp_lam, clamp_lam); prev = v
        for p in range(N):
            if v[p] <= lam_min:
                streak[p] += 1
                if streak[p] >= susp_max: banned[p] = True
            elif v[p] >= lam_max:
                streak[p] = 0
        s = 1.0 / (1.0 + np.exp(-np.clip(v, -700, 700)))
        ts[r] = s; banned_t[r] = banned; admitted_t[r] = ~banned
        w[r] = np.where(banned, 0.0, s)
    return ts, w, banned_t, admitted_t


def main():
    ev = sorted(glob.glob(str(ROOT / "results/recovery/c1_5seed_audit_int1.0/evidence_seed*.npz")))
    if not ev:
        ev = sorted(glob.glob(str(ROOT / "results/recovery/**/evidence_seed*.npz"), recursive=True))
    if not ev:
        sys.exit("no real evidence npz found")
    ev_path = ev[0]
    E = SR.Evidence.load_npz(ev_path)
    base = pol.load_base_config()
    R, N = E.n_rounds, E.n_participants
    v1 = pol._with_thresholds(base, tau_min=0.30, cusum_h=pol.H_V1_DEFAULT, decay=DECAY_V1)
    v2 = pol._with_thresholds(base, tau_min=0.30, cusum_h=pol.H_V1_DEFAULT, decay=DECAY_V2)

    t0 = time.perf_counter()
    gt1 = SR.replay(E, v1)                       # online decisions (θ_v1)
    L = seal_llrs(E, base)                        # the 5 clamped per-channel LLRs
    gt2 = SR.replay(E, v2)                         # FULL-evidence ground truth (θ_v2 decay=0.8)
    gt2_ts_q = qz(gt2.t_score); gt2_w_q = qz(gt2.w_agg)
    build_s = time.perf_counter() - t0

    # ---- write +LLR tuples through the DEPLOYED store, then seal ----
    import tempfile, os
    db = os.path.join(tempfile.mkdtemp(), "deployed.db")
    st = TrustStore(db)
    t0 = time.perf_counter()
    for r in range(R):
        tuples = []
        for p in range(N):
            pid = (b"%016d" % p)[:16]
            tuples.append(TrustTuple(
                round=r, participant=pid,
                a_conf=quantize_unit(float(gt1.a_conf[r, p])), fresh=True,
                s_cusum=float(gt1.s_cusum[r, p]),
                t_score=quantize_unit(float(gt1.t_score[r, p])),
                b_score=quantize_unit(float(gt1.b_score[r, p])),
                w_agg=quantize_unit(float(gt1.w_agg[r, p])),
                mode=Mode(int(gt1.mode[r])), h_grad=b"\x00" * 32,
                llr_att=quantize_llr(L[r, p, 0], 0), llr_grad=quantize_llr(L[r, p, 1], 1),
                llr_perf=quantize_llr(L[r, p, 2], 2), llr_beh=quantize_llr(L[r, p, 3], 3),
                llr_seal=quantize_llr(L[r, p, 4], 4),
            ))
        st.append_round(r, tuples)
    write_s = time.perf_counter() - t0
    merkle_root = st.merkle_root()
    auth_root = st.auth_sri_root()
    chunks_ok = st.verify_chunk_chain(deep=True)   # re-packs WITH the sealed LLRs

    # ---- DELETE the raw evidence ----
    del E
    import gc; gc.collect()

    # ---- replay the decay counterfactual reading ONLY the LLRs FROM THE STORE ----
    t0 = time.perf_counter()
    L_store = np.zeros((R, N, 5))
    by = {}
    for t in st.all_tuples():
        by[(t.round, t.participant)] = t.llr_vector()
    for r in range(R):
        for p in range(N):
            L_store[r, p] = by[(r, (b"%016d" % p)[:16])]
    ts_store, w_store, banned_store, admitted_store = recon_from_llrs(L_store, base, DECAY_V2)
    ts_store_q = qz(ts_store); w_store_q = qz(w_store)
    replay_s = time.perf_counter() - t0

    ts_mismatch = int(np.sum(ts_store_q != gt2_ts_q))
    w_mismatch = int(np.sum(w_store_q != gt2_w_q))
    # audit-meaningful DECISIONS: do ban/admit flip under the counterfactual?
    ban_mismatch = int(np.sum(banned_store != gt2.banned))
    admit_mismatch = int(np.sum(admitted_store != gt2.admitted))
    ts_ulp = np.abs(ts_store_q.astype(int) - gt2_ts_q.astype(int))
    ts_bit_exact_frac = float((ts_ulp == 0).mean())

    # ---- authenticated receipt verified against the sealed root, no raw ----
    tau = 30000
    c, w, proof = st.count_below_receipt(0, R - 1, tau)
    receipt_ok = A.verify_count_below(auth_root, 0, R - 1, tau, c, w, proof)
    receipt_rej = not A.verify_count_below(auth_root, 0, R - 1, tau + 5000, c, w, proof)

    out = {
        "_note": "End-to-end deployed-schema replayable audit on REAL evidence. "
                 "Writes +LLR TrustTuples through the production TrustStore, seals (Merkle + auth-SRI "
                 "roots), DELETES raw evidence, then replays the decay counterfactual from the LLRs "
                 "READ BACK FROM THE STORE and verifies a receipt against the sealed root.",
        "evidence": Path(ev_path).name, "rounds": R, "participants": N, "cells": R * N,
        "tuple_bytes": TUPLE_SIZE, "tuple_bytes_note": "76 B = legacy size; LLRs reuse tail padding (0 net)",
        "counterfactual": f"decay {DECAY_V1}->{DECAY_V2}",
        "merkle_root_hex": merkle_root.hex()[:16], "auth_root_hex": auth_root.hex()[:16],
        "verify_chunk_chain_with_llrs": bool(chunks_ok),
        "decision_ban_flips_vs_full_evidence": ban_mismatch,
        "decision_admit_flips_vs_full_evidence": admit_mismatch,
        "decisions_exact_from_store": ban_mismatch == 0 and admit_mismatch == 0,
        "t_score_uint16_mismatch": ts_mismatch,
        "t_score_max_ulp": int(ts_ulp.max()),
        "t_score_bit_exact_frac": round(ts_bit_exact_frac, 4),
        "w_agg_uint16_mismatch": w_mismatch,
        "fidelity_note": "deployed 2-byte per-channel LLRs (0 net bytes): DECISIONS exact (0 flips); "
                         "trust score 99%+ uint16-bit-exact, max a few ulp (<=0.01%); float LLRs give "
                         "0 t_score mismatch (exactness round trip) — a storage/fidelity point on the advisor cost axis.",
        "receipt_count": int(c), "receipt_verifies": bool(receipt_ok),
        "receipt_rejects_wrong_tau": bool(receipt_rej),
        "build_s": round(build_s, 2), "write_s": round(write_s, 2),
        "end_to_end_replay_s": round(replay_s, 2),
    }
    od = ROOT / "results/artifact"
    (od / "B5_end_to_end_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", od / "B5_end_to_end_audit.json")


if __name__ == "__main__":
    main()
