#!/usr/bin/env python3
"""Advisor access-method DECISION table.

Shows the ReplayabilityAdvisor choosing a DIFFERENT physical access method / proof per policy family
(>=4 distinct outcomes), each tied to what the sealed schema makes provable: the advisor selects the
minimal PROVABLE sealed view for deletion-safe replay and the access method that serves it.
"""
import json, sys
from pathlib import Path
import importlib.util as u

ROOT = Path(__file__).resolve().parents[3]
s = u.spec_from_file_location("schema_advisor", ROOT / "dtrust/store/schema_advisor.py")
adv = u.module_from_spec(s); sys.modules["schema_advisor"] = adv; s.loader.exec_module(adv)

_ADV = adv.ReplayabilityAdvisor()
CUR = {"t_score", "w_agg", "s_cusum", "b_score", "a_conf"}   # current production tuple
AUG = set(CUR) | {"llr_att", "llr_grad", "llr_perf", "llr_beh", "llr_seal"}

FAMILIES = {
    "tau-threshold (count_below)":   [{"count_below(*)"}],
    "drift any-h (earliest_crossing)": [{"earliest_crossing(any_h)"}],
    "drift changed-prior":           [{"earliest_crossing(changed_prior)"}],
    "damage/weight (weight_below)":  [{"weight_below(*)"}],
    "weight/decay re-decision":      [{"count_below'(w,decay)"}],
    "channel recalibration":         [{"count_below_recal(*)"}],
    "raw-row quantile (neg control)":[{"raw_per_row_value"}],
}


def row(fam, deps, sealed):
    r = _ADV.advise(fam, deps, sealed)
    return {
        "family": fam,
        "replayable_now": r.replayable_now,
        "serving_access_method": r.serving_access_method,
        "proof_cost_lemma_v": r.proof_cost,
        "min_augmentation": r.min_augmentation,
        "augmentation_bytes": r.augmentation_bytes,
    }


def main():
    cur_rows = [row(f, d, set(CUR)) for f, d in FAMILIES.items()]
    aug_rows = [row(f, d, AUG) for f, d in FAMILIES.items()]
    # distinct serving methods chosen (the >=4-case claim)
    methods = set()
    for rr in cur_rows + aug_rows:
        for m in rr["serving_access_method"].values():
            methods.add(m)
    out = {
        "_note": "B4 advisor access-method decision table. The advisor picks a different physical "
                 "access method + Lemma-V proof per family; >=4 distinct outcomes proves the selection "
                 "is non-trivial. SRI/DRI are off-the-shelf ADS; the contribution is selecting the "
                 "minimal PROVABLE sealed view for deletion-safe replay.",
        "current_tuple": sorted(CUR),
        "augmented_tuple_+5LLR": sorted(AUG),
        "distinct_serving_methods": sorted(methods),
        "n_distinct_serving_methods": len(methods),
        "current_schema": cur_rows,
        "augmented_schema": aug_rows,
    }
    od = ROOT / "results/artifact"
    (od / "B4_decision_table.json").write_text(json.dumps(out, indent=2))
    print(f"distinct serving methods ({len(methods)}):", sorted(methods))
    print("\n=== CURRENT tuple ===")
    for rr in cur_rows:
        print(f"  {rr['family']:<32} replayable={str(rr['replayable_now']):<5} "
              f"serve={list(rr['serving_access_method'].values())} aug={rr['min_augmentation']}")
    print("\n=== +5 LLR augmented tuple ===")
    for rr in aug_rows:
        print(f"  {rr['family']:<32} replayable={rr['replayable_now']}")
    print("wrote", od / "B4_decision_table.json")


if __name__ == "__main__":
    main()
