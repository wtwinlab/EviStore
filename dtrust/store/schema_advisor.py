"""ReplayabilityAdvisor — policy-family-complete sealed sufficient-statistic design.

Operationalizes the replayability theory of the paper (Section IV-B):
- T1 (exact-replayability boundary): a policy family P is replayable from a sealed schema S iff
  dep(rho) ⊆ cl(S) for every rho in P, where cl(S) is the least fixpoint of positive Horn
  derivation rules D (the trust-policy algebra's deterministic derivations).
- T2 (augmentation = weighted Set Cover): the minimum-cost schema augmentation making P
  replayable is weighted set cover (NP-hard in general; greedy gives H(d)). The DEPLOYED advisor
  here computes union-of-min-cost-supports (AND-OR backward derivation) — **exact when each atom's
  support is a unique chain (the DTRUST rule graph); NOT generally min-cost** when atoms share
  cheaper alternative supports (that general case is the weighted-Set-Cover greedy of T2).

Pure metadata reasoning — NO store/torch dependency. Atoms are symbolic strings; rules are
positive Horn (premise-set ⊢ conclusion). cl() is monotone/idempotent and poly-time by
saturation. Crypto commitments are NOT derivation rules (a digest does not derive its preimage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Derivation rules D (premise-set ⊢ conclusion) — the trust-policy algebra closure.
# Parametric thresholds are symbolic (count_below(*), earliest_crossing(*)) — never enumerate τ.
# ---------------------------------------------------------------------------
Rule = Tuple[FrozenSet[str], str]

DTRUST_RULES: List[Rule] = [
    # τ-threshold family: trust-score threshold aggregates from the sealed scalar.
    (frozenset({"t_score"}), "count_below(*)"),
    (frozenset({"t_score", "w_agg"}), "weight_below(*)"),
    # drift family (checked against a real-data round-trip). h is ONLY the alarm
    # THRESHOLD; it does NOT enter the CUSUM recursion S_t=max(0,S_{t-1}-inc_t). So the sealed
    # s_cusum trajectory (frozen under the deployed increments) determines the earliest crossing
    # for ANY h' — a pure threshold/scan over the sealed walk (0 mismatch, 5 seeds). The
    # sealed per-round residuals give the SAME any-h crossing via the dyadic DRI in O(log R), and
    # additionally support a changed reflection/clamp on the SAME increment stream. A counterfactual
    # that changes the channel PRIORS changes the increment itself -> needs the (recomputable) LLRs
    # from raw evidence, not s_cusum or residuals.
    # s_cusum is the LOGICAL sufficiency source for any-h (already in the current tuple → replayable
    # now by an O(R) scan / O(log R) max-tree over the sealed walk). residuals are NOT extra evidence
    # the tuple lacks — they are an ALTERNATIVE PHYSICAL ACCESS PATH (the dyadic DRI) for the SAME
    # any-h query in O(log R), and additionally support a changed reflection/clamp on the same
    # increment stream. A counterfactual that changes the channel PRIORS changes the increment itself
    # → needs the (recomputable) LLRs from raw evidence, not s_cusum or residuals.
    (frozenset({"s_cusum"}), "earliest_crossing(any_h)"),                  # replayable now (scan)
    (frozenset({"residuals"}), "earliest_crossing(any_h)"),               # SAME query, DRI O(log R) path
    (frozenset({"att_resid", "grad_cos", "success_rate", "beh_event", "seal_ok"}),
     "earliest_crossing(changed_prior)"),                                  # changed increment -> raw
    # weight/decay family: re-run the clamped linear λ-recursion under any (w, decay) from the
    # sealed RAW (pre-clamp) per-channel LLRs (so weight, decay AND a changed per-channel clamp
    # are all covered). Exact up to ledger quantization; replay is a step-by-step re-run
    # (per-step λ-clamp), not a closed form. (NOTE: dropped the contrived weight_below'(w,decay)
    # rule — old sealed w_agg is the realized weight, not the weight a new policy would assign.)
    (frozenset({"llr_att", "llr_grad", "llr_perf", "llr_beh", "llr_seal"}), "lambda(w,decay)"),
    (frozenset({"lambda(w,decay)"}), "t_score'(w,decay)"),
    (frozenset({"t_score'(w,decay)"}), "count_below'(w,decay)"),
    # decay-ONLY family (fixed channel weights, change decay): the recursion λ_t=clamp(decay·λ_{t-1}+g_t)
    # depends on the channels ONLY through the scalar g_t = Σ_c clip(ℓ_c). So the minimal sealed
    # statistic is a single per-cell scalar g_increment (+2B), NOT the 5-LLR vector. g_t is the SUM of
    # the 5 LLRs (so the ladder nests: 5-LLR ⊢ g_increment), but a sealed g_increment alone suffices for
    # any decay. It is NOT sufficient for a per-channel REWEIGHT (Σ w_c·ℓ_c is not a function of Σ ℓ_c)
    # — that requires the 5-LLR vector above. (necessity ladder; witnessed on real data.)
    (frozenset({"llr_att", "llr_grad", "llr_perf", "llr_beh", "llr_seal"}), "g_increment"),  # nests
    (frozenset({"g_increment"}), "lambda(decay)"),
    (frozenset({"lambda(decay)"}), "t_score'(decay)"),
    (frozenset({"t_score'(decay)"}), "count_below'(decay)"),
    # channel-recalibration family: recompute LLRs from raw evidence under new calibration.
    # perf recal needs the CONTINUOUS success_rate (production binarizes via success_rate>=p0,
    # sprt_scorer.py:186) — the binary perf_success cannot move p0.
    (frozenset({"att_resid"}), "llr_att(calib)"),
    (frozenset({"grad_cos"}), "llr_grad(calib)"),
    (frozenset({"success_rate"}), "llr_perf(calib)"),
    (frozenset({"beh_event"}), "llr_beh(calib)"),
    (frozenset({"seal_ok"}), "llr_seal(calib)"),
    # raw evidence also yields the PLAIN LLR (at default calibration), so the raw tier strictly
    # subsumes the LLR tier (raw ⊃ LLR ⊃ t_score): raw -> weight/decay AND recalibration.
    (frozenset({"att_resid"}), "llr_att"),
    (frozenset({"grad_cos"}), "llr_grad"),
    (frozenset({"success_rate"}), "llr_perf"),
    (frozenset({"beh_event"}), "llr_beh"),
    (frozenset({"seal_ok"}), "llr_seal"),
    (frozenset({"llr_att(calib)", "llr_grad(calib)", "llr_perf(calib)",
                "llr_beh(calib)", "llr_seal(calib)"}), "lambda(calib,w,decay)"),
    (frozenset({"lambda(calib,w,decay)"}), "count_below_recal(*)"),
    # analytics (non-FL) quality-gate family: threshold over a sealed per-cell quality score.
    (frozenset({"q"}), "count_below_q(*)"),
    (frozenset({"q", "w_agg"}), "weight_below_q(*)"),
]

# Per-cell storage cost (bytes) of sealing each candidate statistic.
DTRUST_COST: Dict[str, int] = {
    "t_score": 2, "w_agg": 2, "s_cusum": 4, "b_score": 2, "a_conf": 2,        # current tuple scalars
    "llr_att": 2, "llr_grad": 2, "llr_perf": 2, "llr_beh": 2, "llr_seal": 2,   # +LLR tier (~+10B)
    "g_increment": 2,     # scalar g_t = Σ_c clip(ℓ_c) -> decay-only family minimal statistic (+2B)
    "att_resid": 4, "grad_cos": 4, "success_rate": 4, "beh_event": 1, "seal_ok": 1,  # +raw tier (perf recal needs continuous success_rate)
    "residuals": 4,        # per-round reflected-walk residual -> any-h' drift counterfactual (DRI)
    "q": 2,
}


def closure(sealed: Set[str], rules: List[Rule] = DTRUST_RULES) -> Set[str]:
    """Least fixpoint cl(S): everything derivable from `sealed` under positive Horn rules D.
    Monotone, idempotent, terminates (finite symbolic atoms; positive rules)."""
    cl: Set[str] = set(sealed)
    changed = True
    while changed:
        changed = False
        for premises, concl in rules:
            if concl not in cl and premises <= cl:
                cl.add(concl)
                changed = True
    return cl


def is_replayable(dep: Set[str], sealed: Set[str], rules: List[Rule] = DTRUST_RULES) -> bool:
    """T1: dep(rho) ⊆ cl(S)."""
    return dep <= closure(sealed, rules)


@dataclass
class AdviceReport:
    """Full replayability-aware schema-design output (the 7-column advisor product):
    replayable? · missing atoms · min-cost augmentation · storage cost (B/cell) ·
    write cost (per-round maintenance) · proof cost (witness for external verification) ·
    serving method (SRI / DRI / scan / Q1)."""
    family: str
    base_schema: List[str]
    replayable_now: bool                       # (1) does current schema support the family?
    missing_atoms: List[str]                   # (2) which statistics are missing
    min_augmentation: List[str]                # (3) which new statistics to seal
    augmentation_bytes: int                    # (4) storage cost: +bytes/cell
    augmentation_write: str                    # (5) write cost: per-round maintenance
    proof_cost: Dict[str, str]                 # (6) external-verification witness per dep atom
    serving_access_method: Dict[str, str]      # (7) physical access method per dep atom


# Which physical access method serves a derived query atom (advisor's serving-method column).
# Serving methods: the EXISTING SRI indexes the sealed t_score / q, so it directly serves
# count_below(*)/weight_below(*). DERIVED scores (t_score'(w,decay), recal) are NOT in the live
# SRI — they need a scan or a rebuilt/materialized SRI over the derived score. Drift earliest-
# crossing is served by the dyadic DRI ONLY once wired (store.py today = indexed SQL, not O(log R)).
_ACCESS_METHOD = {
    "count_below(*)": "SRI", "weight_below(*)": "SRI",
    "count_below_q(*)": "SRI", "weight_below_q(*)": "SRI",
    "count_below'(decay)": "scan / rebuilt-SRI over derived score",
    "count_below'(w,decay)": "scan / rebuilt-SRI over derived score",
    "count_below_recal(*)": "scan / rebuilt-SRI over derived score",
    "earliest_crossing(any_h)": "scan over sealed s_cusum (any h), or DRI O(log R) from residuals",
    "earliest_crossing(changed_prior)": "recompute LLRs from raw evidence, then DRI / scan",
}
# Atoms that can never be sealed (unsealed raw per-row attributes / confidential) -> any policy
# depending on them is non-replayable at any finite cost.
UNSEALABLE = {"raw_per_row_value"}

# Statistics maintained inside the SRI (incur O(log U) index upkeep per cell, not just an append).
INDEXED = {"t_score", "q"}

# (6) Proof cost = external-verification witness, by serving method (authenticated SRI).
_PROOF_COST = {
    "SRI": "O(log U) sibling-digest witness (authenticated SRI) vs sealed root",
    "count_below'(decay)": "re-run recursion over sealed g_t + O(log U) witness on a rebuilt SRI over t_score'",
    "count_below'(w,decay)": "re-run recursion over sealed LLRs + O(log U) witness on a rebuilt SRI over t_score'",
    "count_below_recal(*)": "recompute LLRs from raw + O(log U) witness on a rebuilt SRI",
    "earliest_crossing(any_h)": "O(log R) dyadic-DRI path-summary proof (from residuals); s_cusum scan otherwise",
    "earliest_crossing(changed_prior)": "recompute LLRs from raw + O(log R) DRI path summary",
    "scan / rebuilt-SRI over derived score": "O(range) re-derivation, or O(log U) after rebuilding an SRI over the derived score",
    "DRI (when wired) / indexed-scan": "O(log R) path summary (when wired); else none",
    "scan/Q1": "O(log U) point-inclusion witness",
}


def _proof_for(atom: str, serving: str) -> str:
    return _PROOF_COST.get(atom) or _PROOF_COST.get(serving, "re-derivation (no succinct witness)")


class ReplayabilityAdvisor:
    """Given a policy family + candidate statistics, decide replayability and recommend the
    sealed-schema augmentation (union-of-min-cost-supports; exact for unique-chain rule graphs,
    else the weighted-Set-Cover T2 general case)."""

    def __init__(self, rules: List[Rule] = DTRUST_RULES, cost: Dict[str, int] = DTRUST_COST):
        self.rules = rules
        self.cost = cost

    def _cost_of(self, stats) -> int:
        return sum(self.cost.get(s, 0) for s in stats)

    def _support(self, q: str, sealed: Set[str], candidates: Set[str],
                 memo: Dict[str, Optional[FrozenSet[str]]],
                 visiting: Set[str]) -> Optional[FrozenSet[str]]:
        """Min-cost set of sealable statistics A (beyond `sealed`) s.t. q ∈ cl(sealed∪A).
        AND-OR backward derivation: a rule needs ALL premises (AND, union of supports);
        a quantity may be derivable several ways or sealed directly (OR, take min cost).
        Returns None if q is not derivable from any sealable statistics (-> non-replayable).
        Correctly handles synergy bundles ({a,b}⊢q): the rule's support is the UNION of
        premise supports (so partial bundles are never scored as progress)."""
        if q in closure(sealed, self.rules):
            return frozenset()                      # already available, nothing to add
        if q in memo:
            return memo[q]
        if q in visiting:
            return None                             # cycle guard (rules are positive)
        visiting.add(q)
        best: Optional[FrozenSet[str]] = None
        if q in candidates and q not in UNSEALABLE:  # OR-option: seal q directly
            best = frozenset({q})
        for premises, concl in self.rules:           # OR over rules deriving q
            if concl != q:
                continue
            acc: Set[str] = set()
            ok = True
            for p in premises:                       # AND over premises (union supports)
                s = self._support(p, sealed, candidates, memo, visiting)
                if s is None:
                    ok = False
                    break
                acc |= s
            if ok:
                cand = frozenset(acc)
                if best is None or self._cost_of(cand) < self._cost_of(best):
                    best = cand
        visiting.discard(q)
        memo[q] = best
        return best

    def min_augment(self, policy_deps: List[Set[str]], base: Set[str],
                    candidates: Set[str]) -> Tuple[List[str], int]:
        """Union of min-cost supports over all required atoms. (When supports overlap with
        cheaper alternatives shared across policies this reduces to weighted Set Cover — the
        NP-hard general case of T2; for the deployed DTRUST instance the supports are unique
        chains, so the union is exact.) Returns ([], 0) if some atom is non-replayable."""
        sealed = set(base)
        memo: Dict[str, Optional[FrozenSet[str]]] = {}
        aug: Set[str] = set()
        for dep in policy_deps:
            for q in dep:
                s = self._support(q, sealed, candidates, memo, set())
                if s is None:
                    return [], 0                      # non-replayable at any finite cost
                aug |= s
        aug -= sealed
        return sorted(aug), self._cost_of(aug)

    def advise(self, family: str, policy_deps: List[Set[str]], base: Set[str],
               candidates: Optional[Set[str]] = None) -> AdviceReport:
        if candidates is None:
            candidates = set(self.cost) - UNSEALABLE  # sealable stats only
        cl_base = closure(base, self.rules)
        all_dep: Set[str] = set().union(*policy_deps) if policy_deps else set()
        missing = sorted(all_dep - cl_base)
        replayable_now = not missing
        aug, aug_bytes = ([], 0) if replayable_now else self.min_augment(policy_deps, base, candidates)
        cl_after = closure(base | set(aug), self.rules)
        serving = {a: _ACCESS_METHOD.get(a, "scan/Q1") for a in sorted(all_dep) if a in cl_after}
        # (5) write cost: per-round maintenance of the augmentation (append + any index upkeep)
        if not aug:
            write = "none" if replayable_now else "n/a (non-replayable)"
        else:
            idx = sorted(set(aug) & INDEXED)
            write = f"append-only +{aug_bytes} B/cell/round" + (
                f"; + O(log U) SRI upkeep for {idx}" if idx else "; no new index")
        # (6) proof cost per dep atom (witness for external verification), by serving method
        proof = {a: _proof_for(a, serving[a]) for a in serving}
        return AdviceReport(
            family=family, base_schema=sorted(base), replayable_now=replayable_now,
            missing_atoms=missing, min_augmentation=aug, augmentation_bytes=aug_bytes,
            augmentation_write=write, proof_cost=proof, serving_access_method=serving,
        )


if __name__ == "__main__":
    adv = ReplayabilityAdvisor()
    CUR = {"t_score", "w_agg", "s_cusum", "b_score", "a_conf"}   # current production tuple

    families = {
        "tau-threshold":            [{"count_below(*)"}],
        "drift-anyh":               [{"earliest_crossing(any_h)"}],        # any h' from sealed s_cusum
        "drift-changed-prior":      [{"earliest_crossing(changed_prior)"}],# changed increment -> raw
        "damage/weight":            [{"weight_below(*)"}],
        "decay-only":               [{"count_below'(decay)"}],               # minimal = g_t (+2B)
        "reweight+decay":           [{"count_below'(w,decay)"}],             # minimal = 5-LLR (+8B)
        "channel-recalibration":    [{"count_below_recal(*)"}],
        "raw-row-quantile (neg)":   [{"raw_per_row_value"}],
    }
    print("=== FL trust-audit policy families vs CURRENT production tuple ===")
    for fam, deps in families.items():
        r = adv.advise(fam, deps, CUR)
        tag = "REPLAYABLE" if r.replayable_now else f"needs {r.min_augmentation} (+{r.augmentation_bytes}B/cell)"
        print(f"  {fam:24s}: {'OK' if r.replayable_now else 'NO ':3s} -> {tag}")

    # expected ground truth (matches the inline operators_eq_oracle semantics + theory)
    exp = {"tau-threshold": True, "drift-anyh": True, "drift-changed-prior": False,
           "damage/weight": True, "decay-only": False, "reweight+decay": False,
           "channel-recalibration": False, "raw-row-quantile (neg)": False}
    ok = all(adv.advise(f, d, CUR).replayable_now == exp[f] for f, d in families.items())
    # the negative family must stay non-replayable even after augmentation:
    rq = adv.advise("raw-row-quantile (neg)", families["raw-row-quantile (neg)"], CUR)
    neg_ok = (not rq.replayable_now) and ("raw_per_row_value" in rq.missing_atoms) and not rq.min_augmentation
    print(f"\nverdicts == expected: {ok};  non-replayable-rejected: {neg_ok}")
    # necessity ladder: decay-only -> scalar g_t (+2B); reweight+decay -> 5-LLR vector (+8B)
    do = adv.advise("decay-only", families["decay-only"], CUR)
    rw = adv.advise("reweight+decay", families["reweight+decay"], CUR)
    print(f"ladder: decay-only -> {do.min_augmentation} (+{do.augmentation_bytes}B/cell); "
          f"reweight+decay -> {sorted(rw.min_augmentation)} (+{rw.augmentation_bytes}B/cell)")
