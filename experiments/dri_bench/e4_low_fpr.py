#!/usr/bin/env python3
"""Report the audit at a LOW-FALSE-POSITIVE operating point.

The main policy-revision case study tightens the decision threshold from tau_min=0.30 to
tau_min=0.45 (an aggressive audit policy). An auditor at a regulated institution
may instead want a conservative, low-collateral policy: catch the cohort while
excluding as few honest participants as possible. Because the sealed t_score
(hence lambda) is independent of tau_min, re-deciding at any tau_min is a pure
threshold query on the SAME sealed evidence (no enclave re-execution) -- so we
sweep tau_min and report precision / recall / honest-FPR at each point, using
the SAME committed decision function f_rho and the SAME ever-excluded-vs-M
convention as the differential audit (audit.py).

Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/e4_low_fpr.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_scripts.dtrust_experiments.c1_audit.policies import (
    load_base_config, _with_thresholds)
from experiment_scripts.dtrust_experiments.c1_audit.sprt_replay import Evidence, replay

# Two measured-client traces: full-intensity (int1.0, the main case study) and a
# stealthier int0.7 trace where the cohort is harder to separate, so the FPR knob
# matters. Both are reported.
EV_GLOBS = {
    "int1.0": str(ROOT / "results/recovery/c1_5seed_audit_int1.0/evidence_seed*.npz"),
    "int0.7": str(ROOT / "results/recovery/c1_5seed_audit_int0.7/evidence_seed*.npz"),
}
TAU_GRID = [0.10, 0.15, 0.20, 0.30, 0.45]   # 0.10 = low-FPR; 0.45 = aggressive audit
H_V1, DECAY = 15.0, 0.9


def operating_point(ev: Evidence, tau_min: float, base: dict) -> dict:
    theta = _with_thresholds(base, tau_min=tau_min, cusum_h=H_V1, decay=DECAY)
    theta["admission_includes_cusum"] = False
    theta["_label"] = f"tau{tau_min}"
    d = replay(ev, theta)
    excl = ~d.admitted                              # [R, N], audit.py convention
    ever_excl = excl.any(axis=0)                    # client ever excluded
    is_adv = ev.is_adversary.max(axis=0).astype(bool)
    tp = int((ever_excl & is_adv).sum())
    fp = int((ever_excl & ~is_adv).sum())
    fn = int((~ever_excl & is_adv).sum())
    n_honest = int((~is_adv).sum())
    return {
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        "fpr": round(fp / n_honest, 3) if n_honest else 0.0,   # honest collateral rate
        "honest_excluded": fp,
    }


def sweep(files: list, base: dict) -> dict:
    evs = [Evidence.load_npz(f) for f in files]
    per_tau = {}
    for tau in TAU_GRID:
        rows = [operating_point(ev, tau, base) for ev in evs]
        agg = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            agg[k] = {"mean": round(float(np.mean(vals)), 3),
                      "std": round(float(np.std(vals)), 3)}
        per_tau[f"{tau:.2f}"] = agg
    return {"n_seeds": len(evs), "per_tau_min": per_tau}


def main():
    base = load_base_config()
    traces = {}
    for tag, gl in EV_GLOBS.items():
        files = sorted(glob.glob(gl))
        if files:
            traces[tag] = sweep(files, base)
    if not traces:
        sys.exit("no sealed evidence found")
    out = {
        "_note": "Operating-point sweep of the SPRT decision threshold on the SAME "
                 "sealed measured-client evidence (no re-execution; t_score is "
                 "tau_min-independent, so re-deciding is a pure threshold query). "
                 "tau_min=0.10 is the low-FPR (conservative) point; 0.45 is the "
                 "aggressive audit point used in the C1 case study. On the "
                 "full-intensity trace the cohort separates cleanly (FPR 0 at every "
                 "tau_min); the stealthier int0.7 trace is where the FPR knob bites.",
        "h_v1": H_V1, "decay": DECAY, "traces": traces,
    }
    print(json.dumps(out, indent=2))
    od = ROOT / "results/recovery/c1_sri"
    od.mkdir(parents=True, exist_ok=True)
    (od / "e4_low_fpr.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "e4_low_fpr.json")


if __name__ == "__main__":
    main()
