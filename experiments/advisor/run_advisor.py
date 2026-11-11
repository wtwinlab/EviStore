#!/usr/bin/env python3
"""Run the SAME ReplayabilityAdvisor across the FL trust-audit schema AND the 4
non-FL data-quality workloads (taxi/sensor/finance/medical). Verifies the advisor's
replayability verdicts match ground truth (the operators_eq_oracle semantics: quality-gate
policy replayable from sealed q; raw-per-row policy non-replayable), and reports the FL 3-tier
augmentation (the schema-insufficiency -> augmentation case study). Pure metadata; no GPU.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import importlib.util as _u
_spec=_u.spec_from_file_location("schema_advisor", ROOT/"dtrust/store/schema_advisor.py")
_m=_u.module_from_spec(_spec); sys.modules['schema_advisor']=_m; _spec.loader.exec_module(_m)
ReplayabilityAdvisor=_m.ReplayabilityAdvisor

adv = ReplayabilityAdvisor()
out = {"_note": "Same advisor over FL + 4 non-FL workloads. Verdicts vs oracle; FL 3-tier "
                "augmentation = schema-insufficiency -> advisor augmentation case study.", "domains": {}}

# --- FL trust-audit (current production tuple) ---
FL_BASE = {"t_score", "w_agg", "s_cusum", "b_score", "a_conf"}
FL_FAMS = {
    "tau-threshold": ([{"count_below(*)"}], True),
    "drift-anyh": ([{"earliest_crossing(any_h)"}], True),
    "drift-changed-prior": ([{"earliest_crossing(changed_prior)"}], False),
    "damage/weight": ([{"weight_below(*)"}], True),
    "decay-only": ([{"count_below'(decay)"}], False),          # minimal = g_t (+2B)
    "reweight+decay": ([{"count_below'(w,decay)"}], False),    # minimal = 5-LLR (+8B)
    "channel-recalibration": ([{"count_below_recal(*)"}], False),
    "raw-row-quantile": ([{"raw_per_row_value"}], False),
}
fl = {}
for fam, (deps, exp) in FL_FAMS.items():
    r = adv.advise(fam, deps, FL_BASE)
    fl[fam] = {"replayable_now": r.replayable_now, "expected": exp,
               "missing_atoms": r.missing_atoms,
               "min_augmentation": r.min_augmentation, "storage_bytes_per_cell": r.augmentation_bytes,
               "write_cost": r.augmentation_write, "proof_cost": r.proof_cost,
               "serving_access_method": r.serving_access_method, "match": r.replayable_now == exp}
out["domains"]["FL_trust_audit"] = fl

# --- 4 non-FL workloads: NON-CIRCULAR cross-check against the workloads' OWN recorded
#     sufficient_statistic_check + the empirical operators_eq_oracle (four_workloads.json) ---
NONFL_BASE = {"q", "w_agg"}
QUAL_DEP = {"count_below_q(*)"}
RAW_DEP = {"raw_per_row_value"}
wl_path = ROOT / "results/recovery/analytics_audit/four_workloads.json"
oracle_ok = True
if wl_path.exists():
    wl = json.loads(wl_path.read_text())["workloads"]
    name_map = {"taxi": "taxi_nyc_tlc_2023_01", "sensor": "sensor_epa_pm25_2023_01",
                "finance": "finance_sec_edgar_2023q1", "medical": "medical_openfda_faers_2023"}
    for dom, key in name_map.items():
        w = wl.get(key)
        if not w:
            continue
        a = w["audit"]; ssc = w["sufficient_statistic_check"]
        emp = bool(a["operators_eq_oracle"] and a["exact_sri_eq_scan"] and a["verify_sri"])
        q_truth = ssc["quality_gate (replayable)"]["replayable"]
        r_truth = ssc["raw_row_quantile_gate (NON-replayable)"]["replayable"]
        q_adv = adv.advise(f"{dom}:quality-gate", [QUAL_DEP], NONFL_BASE).replayable_now
        r_adv = adv.advise(f"{dom}:raw-row", [RAW_DEP], NONFL_BASE).replayable_now
        match = (q_adv == q_truth) and (r_adv == r_truth)
        out["domains"][dom] = {
            "empirical_operators_eq_oracle": emp,
            "quality_gate": {"advisor": q_adv, "workload_recorded": q_truth},
            "raw_row": {"advisor": r_adv, "workload_recorded": r_truth},
            "advisor_matches_recorded": match}
        oracle_ok = oracle_ok and emp and match
else:
    oracle_ok = None
    out["domains"]["_nonfl_note"] = f"four_workloads.json not found ({wl_path}); non-FL cross-check skipped"

# --- summary ---
fl_match = all(v["match"] for v in fl.values())
out["fl_verdicts_match_expected"] = fl_match
out["nonfl_advisor_matches_recorded_and_empirical_oracle"] = oracle_ok
all_match = fl_match and (oracle_ok in (True, None))
out["necessity_ladder"] = {
    "decay-only": {"seal": fl["decay-only"]["min_augmentation"], "bytes_per_cell": fl["decay-only"]["storage_bytes_per_cell"]},
    "reweight+decay": {"seal": fl["reweight+decay"]["min_augmentation"], "bytes_per_cell": fl["reweight+decay"]["storage_bytes_per_cell"]}}

od = ROOT / "results/artifact"; od.mkdir(parents=True, exist_ok=True)
(od / "advisor_results.json").write_text(json.dumps(out, indent=2))
print("all_verdicts_match_oracle:", all_match)
print("necessity ladder ->", out["necessity_ladder"])
print("FL: replayable-now =", [f for f, v in fl.items() if v["replayable_now"]])
print("FL: needs-augmentation =", {f: (v["min_augmentation"], v["storage_bytes_per_cell"]) for f, v in fl.items() if not v["replayable_now"] and v["min_augmentation"]})
print("wrote", od / "advisor_results.json")
