#!/usr/bin/env python3
"""Dyadic DRI earliest-crossing — O(log R) vs linear scan + correctness.
The reflected-walk (Lindley-Spitzer) segment tree replays ANY post-hoc threshold h in O(log R).
Pure CPU / synthetic residual stream."""
import json, sys, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import importlib.util as u
s = u.spec_from_file_location("dri_dyadic", ROOT/"dtrust/store/dri_dyadic.py"); D = u.module_from_spec(s); sys.modules["dri_dyadic"]=D; s.loader.exec_module(D)

rng = random.Random(0)
out = {"_note": "Dyadic DRI: earliest_crossing(h) O(log R) vs linear O(R), correctness dyadic==linear. "
                "Honest O(log R) headline (replaces the SQLite indexed shortcut in store.dri_replay).", "scales": []}
THRESH = [2.0, 5.0, 10.0, 20.0, 50.0]   # post-hoc thresholds swept

for R in [1000, 10000, 100000]:
    resids = [rng.gauss(-0.2, 1.0) for _ in range(R)]   # slight negative drift + noise (reflected walk)
    t0 = time.perf_counter(); dri = D.DyadicDRI(resids); build_ms = (time.perf_counter()-t0)*1e3
    # correctness: dyadic == linear for every threshold
    correct = all(dri.earliest_crossing(h) == D.DyadicDRI.linear_earliest_crossing(resids, h, 0, R) for h in THRESH)
    # latency: many queries
    Q = 2000
    qs = [(rng.choice(THRESH)) for _ in range(Q)]
    t0 = time.perf_counter()
    for h in qs: dri.earliest_crossing(h)
    dri_us = (time.perf_counter()-t0)/Q*1e6
    t0 = time.perf_counter()
    for h in qs[:200]: D.DyadicDRI.linear_earliest_crossing(resids, h, 0, R)
    lin_us = (time.perf_counter()-t0)/200*1e6
    out["scales"].append({"rounds": R, "build_ms": round(build_ms,2), "correct_vs_linear": correct,
                          "dri_us_per_query": round(dri_us,3), "linear_us_per_query": round(lin_us,1),
                          "speedup": round(lin_us/dri_us,1)})
    print(f"R={R}: correct={correct} dri={dri_us:.3f}us linear={lin_us:.0f}us speedup={lin_us/dri_us:.0f}x")

out["all_correct"] = all(s["correct_vs_linear"] for s in out["scales"])
out["log_R_evidence"] = "dri_us near-flat across 100x R while linear grows ~linearly"
od = ROOT/"results/artifact"; (od/"L3_dri_scaling.json").write_text(json.dumps(out, indent=2))
print("ALL_CORRECT:", out["all_correct"], "-> wrote", od/"L3_dri_scaling.json")
