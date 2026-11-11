"""Authenticated SRI — succinct, *sound* threshold-aggregate receipts.

Merkle persistent segment tree: every node commits (tag, cnt, wsum, d_lc, d_rc). A
count_below(r1,r2,τ) answer ships an O(log U) witness verified vs the sealed root WITHOUT raw data.
The tree is an authenticated-aggregation tree over the sealed schema, so the schema choice
determines which aggregates are provable.

SOUNDNESS: the verifier is bound to the requested (τ, r1, r2):
- it re-derives the *canonical* segment decomposition for (qh, [0,U)) itself (the prover cannot
  choose the covered/opaque frontier — it is forced by qh and the interval), recomputing each
  node digest bottom-up; tag is decided by the interval (leaf ⟺ lo==hi), not by child nullness;
- version leaves bind (round_label, version_root_digest); the range is bound by Merkle-including
  the boundary leaves so hi/lo are exactly the prefix versions implied by r1,r2.
Weights are committed at milliunit fixed point ⇒ replay/verification is quantized-exact.
"""
from __future__ import annotations

import bisect
import hashlib
import struct
from typing import List, Optional, Tuple

T_SCORE_DOMAIN = 65536
NULL = b"\x00" * 32


class _VerifyError(Exception):
    pass


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _enc(cnt: int, wsum: float) -> bytes:
    return struct.pack("<qq", int(cnt), int(round(wsum * 1000)))  # milliunit fixed point


class AuthPersistentRankSum:
    __slots__ = ("U", "lc", "rc", "cnt", "wsum", "dig")

    def __init__(self, U: int = T_SCORE_DOMAIN):
        self.U = U
        self.lc: List[int] = [0]; self.rc: List[int] = [0]
        self.cnt: List[int] = [0]; self.wsum: List[float] = [0.0]
        self.dig: List[bytes] = [NULL]

    def _new(self, lc, rc, cnt, wsum, leaf):
        tag = b"L" if leaf else b"I"
        d = _h(tag, _enc(cnt, wsum), self.dig[lc], self.dig[rc])
        self.lc.append(lc); self.rc.append(rc)
        self.cnt.append(cnt); self.wsum.append(wsum); self.dig.append(d)
        return len(self.cnt) - 1

    def _update(self, node, lo, hi, pos, w):
        if lo == hi:
            return self._new(0, 0, self.cnt[node] + 1, self.wsum[node] + w, leaf=True)
        mid = (lo + hi) >> 1
        if pos <= mid:
            nlc = self._update(self.lc[node], lo, mid, pos, w); nrc = self.rc[node]
        else:
            nrc = self._update(self.rc[node], mid + 1, hi, pos, w); nlc = self.lc[node]
        return self._new(nlc, nrc, self.cnt[node] + 1, self.wsum[node] + w, leaf=False)

    def insert(self, root, pos, w=1.0):
        if not 0 <= pos < self.U:
            raise ValueError(f"key {pos} outside [0,{self.U})")
        return self._update(root, 0, self.U - 1, pos, w)

    def prove_below(self, node, qh, lo=0, hi=-1) -> dict:
        """Proof for #keys <= qh in subtree. Frontier is canonical for (qh,[lo,hi])."""
        if hi == -1:
            hi = self.U - 1
        if node == 0:
            return {"t": "null"}
        if qh < lo:                                  # entirely above qh -> contributes 0
            return {"t": "opaque", "d": self.dig[node]}
        if hi <= qh:                                 # entirely <= qh -> fully counted
            return {"t": "covered", "cnt": self.cnt[node], "wsum": self.wsum[node],
                    "dl": self.dig[self.lc[node]], "dr": self.dig[self.rc[node]]}
        mid = (lo + hi) >> 1
        return {"t": "node", "cnt": self.cnt[node], "wsum": self.wsum[node],
                "L": self.prove_below(self.lc[node], qh, lo, mid),
                "R": self.prove_below(self.rc[node], qh, mid + 1, hi)}


def _wmu(w) -> int:
    return int(round(float(w) * 1000))               # canonical milliunit fixed point (matches _enc)


def _verify_below(p: dict, qh: int, lo: int, hi: int) -> Tuple[bytes, int, int]:
    """Recompute (node_digest, counted_cnt, counted_weight_milliunits), FORCING the case by
    (qh,[lo,hi]). Weight is returned in the COMMITTED milliunit fixed point — not the prover's raw
    float — so a sub-milliunit weight shift (which leaves the digest unchanged) cannot move the
    verified weight. Raises _VerifyError if the prover's node type does not match the canonical case."""
    if p.get("t") == "null":
        return NULL, 0, 0                            # empty subtree (digest bound by parent)
    if qh < lo:                                       # must be opaque (contributes 0)
        if p.get("t") != "opaque":
            raise _VerifyError("expected opaque")
        return p["d"], 0, 0
    if hi <= qh:                                      # must be covered (fully counted)
        if p.get("t") != "covered":
            raise _VerifyError("expected covered")
        tag = b"L" if lo == hi else b"I"             # tag by INTERVAL, not child-nullness
        d = _h(tag, _enc(p["cnt"], p["wsum"]), p["dl"], p["dr"])
        return d, p["cnt"], _wmu(p["wsum"])          # canonicalize weight to committed milliunits
    if p.get("t") != "node":                          # must recurse
        raise _VerifyError("expected node")
    mid = (lo + hi) >> 1
    dl, cl, wl = _verify_below(p["L"], qh, lo, mid)
    dr, cr, wr = _verify_below(p["R"], qh, mid + 1, hi)
    d = _h(b"I", _enc(p["cnt"], p["wsum"]), dl, dr)
    return d, cl + cr, wl + wr


# ---- Merkle over version leaves (each leaf binds round_label + version_root_digest) ----
def _vleaf(round_label: int, vdig: bytes) -> bytes:
    return _h(b"V", struct.pack("<q", int(round_label)), vdig)


def _merkle_root(leaves):
    if not leaves:
        return NULL
    lvl = list(leaves)
    while len(lvl) > 1:
        lvl = [_h(b"M", lvl[i], lvl[i + 1] if i + 1 < len(lvl) else lvl[i]) for i in range(0, len(lvl), 2)]
    return lvl[0]


def _merkle_path(leaves, idx):
    path = []; lvl = list(leaves); i = idx
    while len(lvl) > 1:
        sib = i ^ 1
        path.append(("R" if i % 2 == 0 else "L", lvl[sib] if sib < len(lvl) else lvl[i]))
        lvl = [_h(b"M", lvl[j], lvl[j + 1] if j + 1 < len(lvl) else lvl[j]) for j in range(0, len(lvl), 2)]
        i //= 2
    return path


def _merkle_compute(leaf, idx, path):
    """Recompute the Merkle root for `leaf` at position `idx`. The combine side at each level is
    derived from `idx` parity (NOT trusted from the prover-supplied path), so the witness is bound
    to the claimed leaf index — a path for a different leaf cannot pass as index `idx`."""
    h = leaf
    i = idx
    for _side, sib in path:
        h = _h(b"M", h, sib) if i % 2 == 0 else _h(b"M", sib, h)
        i >>= 1
    return h


class AuthSRIIndex:
    """Round-versioned authenticated SRI. Version i = prefix after i rounds; round_label[i] is the
    i-th appended round (round_label[0] = sentinel -1 for the empty prefix). auth_root = Merkle over
    version leaves H(round_label, version_root_digest)."""
    __slots__ = ("tree", "rounds", "roots", "vdig", "vlabel")

    def __init__(self, U: int = T_SCORE_DOMAIN):
        self.tree = AuthPersistentRankSum(U)
        self.rounds: List[int] = []
        self.roots: List[int] = [0]
        self.vdig: List[bytes] = [NULL]              # version-root node digest after i rounds
        self.vlabel: List[int] = [-1]                # round label of version i (i>=1); -1 = empty

    def append_round(self, round_num, cells):
        if self.rounds and round_num <= self.rounds[-1]:
            raise ValueError("rounds must be strictly increasing")
        root = self.roots[-1]
        for tq, w in cells:
            root = self.tree.insert(root, int(tq), float(w))
        self.rounds.append(round_num); self.roots.append(root)
        self.vdig.append(self.tree.dig[root]); self.vlabel.append(round_num)

    def _leaves(self):
        return [_vleaf(self.vlabel[i], self.vdig[i]) for i in range(len(self.vdig))]

    def auth_root(self):
        leaves = self._leaves()
        # bind the version COUNT into the root, so a verifier can require "v is genuinely the last
        # leaf" (closes the omit-the-next-leaf range forgery).
        return _h(b"ROOT", struct.pack("<q", len(leaves)), _merkle_root(leaves))

    def _ver(self, round_num):
        return bisect.bisect_right(self.rounds, round_num)        # #rounds <= round_num

    def count_below(self, r1, r2, tau_q):
        qh = min(tau_q, self.tree.U) - 1
        hi, lo = self._ver(r2), self._ver(r1 - 1)
        _, c2, w2 = _verify_below(self.tree.prove_below(self.roots[hi], qh), qh, 0, self.tree.U - 1)
        _, c1, w1 = _verify_below(self.tree.prove_below(self.roots[lo], qh), qh, 0, self.tree.U - 1)
        return c2 - c1, (w2 - w1) / 1000.0           # weight returned in float units (milliunit-exact)

    def prove_count_below(self, r1, r2, tau_q) -> dict:
        qh = min(tau_q, self.tree.U) - 1
        hi, lo = self._ver(r2), self._ver(r1 - 1)
        leaves = self._leaves(); V = len(leaves)

        def boundary(v):
            # leaf v (round_label<=bound) + adjacent leaf v+1 (round_label>bound) if it exists,
            # so the verifier confirms v is the correct prefix version for the bound.
            out = {"v": v, "label": self.vlabel[v], "vdig": self.vdig[v],
                   "path": _merkle_path(leaves, v)}
            if v + 1 < V:
                out["next_label"] = self.vlabel[v + 1]
                out["next_vdig"] = self.vdig[v + 1]
                out["next_path"] = _merkle_path(leaves, v + 1)
            return out

        return {"r1": r1, "r2": r2, "tau_q": tau_q, "U": self.tree.U, "n_versions": V,
                "proof_hi": self.tree.prove_below(self.roots[hi], qh), "hi": boundary(hi),
                "proof_lo": self.tree.prove_below(self.roots[lo], qh), "lo": boundary(lo)}


def verify_count_below(auth_root, r1, r2, tau_q, claimed_count, claimed_weight, proof,
                       U=T_SCORE_DOMAIN) -> bool:
    """Sound external verification: bind to (r1,r2,tau_q); no raw data."""
    try:
        if (proof.get("r1"), proof.get("r2"), proof.get("tau_q"), proof.get("U")) != (r1, r2, tau_q, U):
            return False                              # proof must answer the REQUESTED query
        qh = min(tau_q, U) - 1
        V = proof.get("n_versions")
        if not isinstance(V, int) or V <= 0:
            return False

        def _root_ok(leaf, idx, path):
            # bind to idx (idx-derived sides) AND to the committed version count V.
            if not 0 <= idx < V:
                return False
            return _h(b"ROOT", struct.pack("<q", V), _merkle_compute(leaf, idx, path)) == auth_root

        def check_boundary(b, bound):
            # leaf binds (label, vdig); label must be <= bound, and the *adjacent* (v+1) version's
            # label must exceed bound — so v is exactly the prefix version for `bound`.
            if not _root_ok(_vleaf(b["label"], b["vdig"]), b["v"], b["path"]):
                return None
            if b["v"] == 0:
                if b["label"] != -1:
                    return None                       # empty prefix sentinel
            elif b["label"] > bound:
                return None
            if b["v"] == V - 1:
                # v is genuinely the last version (count bound by the root): no later round exists.
                if "next_vdig" in b:
                    return None
            else:
                # a later version exists: the prover MUST exhibit the adjacent leaf at v+1 with
                # label > bound (cannot omit it to present an earlier prefix).
                if "next_vdig" not in b:
                    return None
                if not _root_ok(_vleaf(b["next_label"], b["next_vdig"]), b["v"] + 1, b["next_path"]):
                    return None
                if not (b["next_label"] > bound):
                    return None
            return b["vdig"]

        vdig_hi = check_boundary(proof["hi"], r2)
        vdig_lo = check_boundary(proof["lo"], r1 - 1)
        if vdig_hi is None or vdig_lo is None:
            return False
        dhi, chi, whi = _verify_below(proof["proof_hi"], qh, 0, U - 1)
        dlo, clo, wlo = _verify_below(proof["proof_lo"], qh, 0, U - 1)
        if dhi != vdig_hi or dlo != vdig_lo:
            return False                              # below-proof root != bound version digest
        # compare at the COMMITTED milliunit granularity (whi/wlo are already milliunit ints): a
        # sub-milliunit forged weight quantizes to the same value the digest commits, so it cannot
        # shift the verified weight.
        return (chi - clo) == claimed_count and (whi - wlo) == _wmu(claimed_weight)
    except (_VerifyError, KeyError, TypeError):
        return False
