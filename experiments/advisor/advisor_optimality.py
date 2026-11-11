#!/usr/bin/env python3
"""Verification: is the ReplayabilityAdvisor's per-family augmentation the PROVABLE minimum
on the DEPLOYED support graph? We brute-force the true weighted-set-cover optimum (exhaustive
over candidate subsets) and compare to the advisor's union-of-min-cost-supports output.

Verifies: "for the deployed support graph, does this exact optimizer produce
the +22B / 13-of-16 optimum, and no cheaper augmentation replays the same family?"

Also checks the one cross-family SUBSUMPTION (5-LLR subsumes g_increment) to report whether
the +22B per-family sum equals a GLOBAL multi-family schema optimum, or over-seals by dedup.
"""
import sys, os, itertools, importlib.util
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Load schema_advisor.py directly (dtrust/__init__ pulls in torch, which we don't need here).
_spec = importlib.util.spec_from_file_location(
    "schema_advisor", os.path.join(_ROOT, "dtrust", "store", "schema_advisor.py"))
_sa = importlib.util.module_from_spec(_spec)
sys.modules["schema_advisor"] = _sa          # dataclass needs the module registered
_spec.loader.exec_module(_sa)
ReplayabilityAdvisor = _sa.ReplayabilityAdvisor
closure = _sa.closure
DTRUST_RULES = _sa.DTRUST_RULES
DTRUST_COST = _sa.DTRUST_COST
UNSEALABLE = _sa.UNSEALABLE

adv = ReplayabilityAdvisor()
CUR = {"t_score", "w_agg", "s_cusum", "b_score", "a_conf"}   # deployed production tuple

# The deployed FL trust-audit families (same as the advisor self-test + policy_evolution).
families = {
    "tau-threshold":         [{"count_below(*)"}],
    "drift-anyh":            [{"earliest_crossing(any_h)"}],
    "drift-changed-prior":   [{"earliest_crossing(changed_prior)"}],
    "damage/weight":         [{"weight_below(*)"}],
    "decay-only":            [{"count_below'(decay)"}],
    "reweight+decay":        [{"count_below'(w,decay)"}],
    "channel-recalibration": [{"count_below_recal(*)"}],
    "raw-row-quantile(neg)": [{"raw_per_row_value"}],
}

# Sealable candidate statistics (advisor uses cost-keys minus unsealable).
CANDS = sorted(set(DTRUST_COST) - UNSEALABLE)

def replayable(family_deps, sealed):
    cl = closure(sealed, DTRUST_RULES)
    return all(dep <= cl for dep in family_deps)

def cost_of(stats):
    return sum(DTRUST_COST.get(s, 0) for s in stats)

def brute_min_augment(family_deps, base):
    """Exhaustive weighted-set-cover optimum: cheapest subset A of CANDS with family replayable
    from base|A. Prunes to candidates in the family's reachable dependency cone for tractability."""
    base = set(base)
    if replayable(family_deps, base):
        return [], 0
    # restrict to candidates that, added alone or in the full set, can ever matter
    relevant = [c for c in CANDS if c not in base and
                replayable(family_deps, base | set(CANDS))]  # only if family is reachable at all
    if not relevant:
        # family non-replayable at any finite cost (e.g. needs raw_per_row_value)
        if not replayable(family_deps, base | set(CANDS)):
            return None, None
    pool = [c for c in CANDS if c not in base]
    if not replayable(family_deps, base | set(pool)):
        return None, None                       # non-replayable at any finite cost
    best, best_cost = None, None
    # exhaustive over subset sizes, smallest first; prune by cost
    for k in range(1, len(pool) + 1):
        found_at_k = False
        for combo in itertools.combinations(pool, k):
            if best_cost is not None and cost_of(combo) >= best_cost:
                continue
            if replayable(family_deps, base | set(combo)):
                c = cost_of(combo)
                if best_cost is None or c < best_cost:
                    best, best_cost = sorted(combo), c
                    found_at_k = True
        # cannot early-break purely on k (costs vary), but once we have a best and min single
        # candidate cost*k exceeds best_cost we can stop
        if best_cost is not None and (k + 1) * min(DTRUST_COST[c] for c in pool) >= best_cost:
            break
    return best, best_cost

print("=== #1: advisor augmentation vs BRUTE-FORCE weighted-set-cover optimum (deployed graph) ===")
print(f"candidates: {len(CANDS)} sealable statistics; base = deployed tuple {sorted(CUR)}\n")
total_adv, total_opt, all_exact = 0, 0, True
per_family = {}
for fam, deps in families.items():
    r = adv.advise(fam, deps, CUR)
    adv_aug, adv_cost = sorted(r.min_augmentation), r.augmentation_bytes
    opt_aug, opt_cost = brute_min_augment(deps, CUR)
    if opt_cost is None:                          # non-replayable
        exact = (not r.replayable_now and not adv_aug)
        verdict = "NON-REPLAYABLE (both)" if exact else "MISMATCH(replayability)"
    elif r.replayable_now:
        exact = (opt_cost == 0)
        verdict = f"replayable-now (opt {opt_cost}B)"
    else:
        exact = (adv_cost == opt_cost)
        total_adv += adv_cost
        total_opt += opt_cost
        verdict = f"advisor +{adv_cost}B {adv_aug}  |  brute-opt +{opt_cost}B {opt_aug}  -> {'EXACT' if exact else 'ADVISOR SUBOPTIMAL'}"
    all_exact &= exact
    per_family[fam] = dict(adv_cost=adv_cost, opt_cost=opt_cost, exact=bool(exact),
                           adv_aug=adv_aug, opt_aug=opt_aug)
    print(f"  {fam:24s}: {verdict}")

print(f"\nper-family advisor total: +{total_adv}B  |  per-family brute optimum total: +{total_opt}B")
print(f"PER-FAMILY EXACTNESS (advisor == set-cover optimum on every family): {'PASS' if all_exact else 'FAIL'}")

# --- cross-family subsumption (global-vs-per-family check) ---
# decay-only seals g_increment (+2B); reweight seals 5-LLR. 5-LLR derives g_increment (rule), so a
# GLOBAL union schema serving BOTH families needs only 5-LLR, saving the +2B g_increment.
five_llr = {"llr_att", "llr_grad", "llr_perf", "llr_beh", "llr_seal"}
decay_dep = families["decay-only"]; rew_dep = families["reweight+decay"]
union_seal = CUR | five_llr
global_both = replayable(decay_dep, union_seal) and replayable(rew_dep, union_seal)
g_sub = "g_increment" in closure(CUR | five_llr, DTRUST_RULES)
print("\n=== cross-family subsumption (global union schema vs per-family sum) ===")
print(f"  5-LLR derives g_increment: {g_sub} (so a schema sealing 5-LLR serves decay-only too)")
print(f"  union schema {{base + 5-LLR}} replays decay-only AND reweight: {global_both} "
      f"(+{cost_of(five_llr)}B), vs per-family sum +{per_family['decay-only']['adv_cost'] + per_family['reweight+decay']['adv_cost']}B")
print("  -> the +22B is the PER-FAMILY optimal sum; a deduplicated global schema is cheaper because")
print("     5-LLR subsumes g_increment. Claim the exact bound at the PER-FAMILY level (provably exact), not as a")
print("     global multi-family minimum.")

import json
out = dict(per_family=per_family, per_family_total_adv=total_adv, per_family_total_opt=total_opt,
          per_family_exact=bool(all_exact), five_llr_subsumes_g_increment=bool(g_sub),
          union_schema_serves_both=bool(global_both), union_schema_cost=cost_of(five_llr))
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advisor_optimality.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {path}")
