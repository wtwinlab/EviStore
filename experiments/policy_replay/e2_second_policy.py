#!/usr/bin/env python3
"""Alternate scoring rule: replay the SAME sealed audit evidence under a SECOND,
materially different policy family, to show the store + audit operators are
policy-agnostic (not bound to the SPRT trust scorer).

Policy family A (in the paper): SPRT/CUSUM multi-channel sequential LLR.
Policy family B (here):        cosine-gate, FLTrust-style -- a client is excluded
                               at round r if its trailing-window mean gradient
                               cosine < tau_cos. Single-channel, threshold-based,
                               NOT sequential -- materially different from SPRT.

Both decide from the SAME sealed evidence (grad_cos is a sealed channel), so by the
evidence contract (Section IV-B) family B is replayable from the schema. We run
counterfactual (decisions under tau), differentialAudit (flips between tau_v1->tau_v2),
and precision/recall vs the sealed ground-truth adversary set -- the same operators
the DRI audit runs for SPRT, now for the cosine family.

Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/e2_second_policy.py
"""
import json, glob, sys, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EV_GLOB = str(ROOT / "results/recovery/c1_5seed_audit_int1.0/evidence_seed*.npz")
W = 10            # trailing window for the cosine gate
TAU_V1, TAU_V2 = 0.10, 0.30   # lenient vs stricter cosine gate (family B's tau knob)


def windowed_mean_cos(grad_cos, W):
    R, N = grad_cos.shape
    out = np.empty_like(grad_cos)
    for r in range(R):
        lo = max(0, r - W + 1)
        out[r] = grad_cos[lo:r + 1].mean(axis=0)
    return out


def cosine_gate_decisions(grad_cos, tau, W=W):
    """Family B: excluded[r,p] = trailing-window mean cosine < tau (FLTrust-style)."""
    mc = windowed_mean_cos(grad_cos, W)
    return mc < tau          # True = excluded


def multifactor_score(ev, W=W):
    """Family C: a fixed-weight multi-factor quality score (non-sequential, non-SPRT),
    combining gradient cosine, performance, and sealing into one scalar per cell."""
    gc = ev["grad_cos"]; perf = ev["perf_success"].astype(float); seal = ev["seal_ok"].astype(float)
    raw = 0.5 * np.clip(gc, 0, 1) + 0.3 * perf + 0.2 * seal
    R, N = raw.shape
    return np.array([raw[max(0, r - W + 1):r + 1].mean(axis=0) for r in range(R)])


def multifactor_gate_decisions(ev, tau, W=W):
    return multifactor_score(ev, W) < tau     # True = excluded


def audit_with(ev, decide):
    """Run the SAME counterfactual/differentialAudit metrics under a given decision fn."""
    adv = ev["is_adversary"]; R, N = adv.shape
    is_adv = adv.max(axis=0).astype(bool)
    e1, e2 = decide(TAU_V1), decide(TAU_V2)
    flips = int((e2 & ~e1).sum())
    ev2 = e2.any(axis=0)
    tp = int((ev2 & is_adv).sum()); fp = int((ev2 & ~is_adv).sum()); fn = int((~ev2 & is_adv).sum())
    prec = round(tp / (tp + fp), 3) if tp + fp else 0.0
    rec = round(tp / (tp + fn), 3) if tp + fn else 0.0
    return {"flips_v1_to_v2": flips, "precision_v2": prec, "recall_v2": rec}


def audit_one(ev):
    grad_cos = ev["grad_cos"]; adv = ev["is_adversary"]
    R, N = grad_cos.shape
    is_adv = adv.max(axis=0).astype(bool)          # client is adversary if ever flagged adv
    excl1 = cosine_gate_decisions(grad_cos, TAU_V1)  # counterfactual under rho_v1
    excl2 = cosine_gate_decisions(grad_cos, TAU_V2)  # counterfactual under rho_v2
    # differentialAudit(v1,v2): cells admitted under v1 but excluded under v2
    flips = excl2 & ~excl1
    # per-client "excluded by end of window" (ever excluded)
    ev_excl1 = excl1.any(axis=0); ev_excl2 = excl2.any(axis=0)

    def prec_rec(ev_excl):
        tp = int((ev_excl & is_adv).sum()); fp = int((ev_excl & ~is_adv).sum())
        fn = int((~ev_excl & is_adv).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return round(prec, 3), round(rec, 3)

    p1, r1 = prec_rec(ev_excl1); p2, r2 = prec_rec(ev_excl2)

    def first_excl(excl, mask):
        rounds = []
        for p in np.where(mask)[0]:
            w = np.where(excl[:, p])[0]
            rounds.append(int(w[0]) if len(w) else R)
        return float(np.mean(rounds)) if rounds else float(R)

    fe1 = first_excl(excl1, is_adv); fe2 = first_excl(excl2, is_adv)
    return {
        "n_adv": int(is_adv.sum()),
        "flips_v1_to_v2": int(flips.sum()),
        "precision_v1": p1, "recall_v1": r1,
        "precision_v2": p2, "recall_v2": r2,
        "first_excl_round_v1": round(fe1, 1), "first_excl_round_v2": round(fe2, 1),
        "first_excl_lead": round(fe1 - fe2, 1),
    }


def main():
    files = sorted(glob.glob(EV_GLOB))
    if not files:
        sys.exit(f"no sealed evidence at {EV_GLOB}")
    per = [audit_one(np.load(f)) for f in files]
    keys = [k for k in per[0] if k != "n_adv"]
    agg = {"n_seeds": len(per), "n_adv": per[0]["n_adv"], "W": W,
           "tau_v1": TAU_V1, "tau_v2": TAU_V2}
    for k in keys:
        vals = [d[k] for d in per]
        agg[k] = {"mean": round(float(np.mean(vals)), 3),
                  "std": round(float(np.std(vals)), 3)}
    # Family C: a third, fixed-weight multi-factor gate (an alternate: >1 alternative family)
    perC = [audit_with(np.load(f), lambda t, e=np.load(f): multifactor_gate_decisions(e, t)) for f in files]
    aggC = {}
    for k in perC[0]:
        vals = [d[k] for d in perC]
        aggC[k] = {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals)), 3)}
    out = {"_note": "SAME sealed C1 evidence replayed under TWO non-SPRT policy families "
                    "(cosine-gate and a fixed-weight multi-factor gate). Proves the store + "
                    "counterfactual/differentialAudit operators are policy-agnostic; both "
                    "families are replayable because their read-sets are sealed sufficient "
                    "statistics (Sufficient Statistic Schema Theorem).",
           "policy_family_B": "cosine_gate (exclude if trailing mean grad-cosine < tau)",
           "policy_family_C": "fixed-weight multi-factor gate (0.5*cos+0.3*perf+0.2*seal < tau)",
           "per_seed": per, "aggregate": agg, "family_C_aggregate": aggC}
    print(json.dumps(out, indent=2))
    od = ROOT / "results/recovery/c1_sri"; od.mkdir(parents=True, exist_ok=True)
    (od / "e2_second_policy.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "e2_second_policy.json")


if __name__ == "__main__":
    main()
