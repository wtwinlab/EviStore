"""Sealed Replay Index (SRI) — sublinear, sealed threshold-aggregate operators.

The four base query operators in ``store.py`` map to SQL over SQLite indexes, so
``range_below`` is an O(z) row *listing* (output-bound) and a threshold *count*
degrades to a scan/index-range. The SRI adds the missing **sublinear aggregate**:

    count_below(r1, r2, tau_q)  = #{cells in rounds [r1,r2] with t_score < tau_q}
    weight_below(r1, r2, tau_q) = Σ w_agg over those cells

both in **O(log U)** (U = 65536, the uint16 t_score domain), exact, and
incrementally maintained as rounds are appended (one persistent version/round,
O(log U) new nodes per cell). It is a *deterministic function of the sealed
tuples*, so it is tamper-evident with the ledger: ``TrustStore.verify_sri``
rebuilds it from the sealed rows and compares, and ``root_digest`` folds the
per-round (count, weight-sum) totals into the Merkle bookkeeping.

This is the framework-embedded counterpart of the standalone perf benchmark in
``experiment_scripts/dtrust_experiments/c1_audit/sri/`` (same algorithm; the
benchmark's numba kernels characterize the constants at M=10⁷, this pure-Python
version is what the coordinator runs at recovery-ledger scale, M≈10⁴–10⁵).
"""

from __future__ import annotations

import bisect
import hashlib
import struct
import sys
from typing import Iterable, List, Tuple

T_SCORE_DOMAIN = 65536  # uint16 t_score


class _PersistentRankSum:
    """Persistent segment tree over key domain [0, U); a new version per insert.
    Each node carries (count, weight-sum) over its key sub-range. O(log U)/op.

    Nodes live in parallel arrays; node 0 is the shared empty/null node. An
    insert copies only the O(log U) nodes on the root→leaf path (path copying),
    so every historical root stays valid and queryable — that is what makes the
    per-round snapshots free."""

    __slots__ = ("U", "lc", "rc", "cnt", "wsum")

    def __init__(self, U: int = T_SCORE_DOMAIN):
        self.U = U
        self.lc: List[int] = [0]
        self.rc: List[int] = [0]
        self.cnt: List[int] = [0]
        self.wsum: List[float] = [0.0]
        # path copying recurses to depth log2(U)=16; default limit is ample, but
        # guard against environments that lowered it.
        if sys.getrecursionlimit() < 1000:
            sys.setrecursionlimit(1000)

    def _new(self, lc: int, rc: int, cnt: int, wsum: float) -> int:
        self.lc.append(lc); self.rc.append(rc)
        self.cnt.append(cnt); self.wsum.append(wsum)
        return len(self.cnt) - 1

    def _update(self, node: int, lo: int, hi: int, pos: int, w: float) -> int:
        if lo == hi:
            return self._new(0, 0, self.cnt[node] + 1, self.wsum[node] + w)
        mid = (lo + hi) >> 1
        if pos <= mid:
            nlc = self._update(self.lc[node], lo, mid, pos, w); nrc = self.rc[node]
        else:
            nrc = self._update(self.rc[node], mid + 1, hi, pos, w); nlc = self.lc[node]
        return self._new(nlc, nrc, self.cnt[node] + 1, self.wsum[node] + w)

    def insert(self, root: int, pos: int, w: float = 1.0) -> int:
        if not 0 <= pos < self.U:
            raise ValueError(f"key {pos} outside [0,{self.U})")
        return self._update(root, 0, self.U - 1, pos, w)

    def _below(self, node: int, lo: int, hi: int, qh: int) -> Tuple[int, float]:
        if node == 0 or qh < lo:
            return 0, 0.0
        if hi <= qh:
            return self.cnt[node], self.wsum[node]
        mid = (lo + hi) >> 1
        c0, w0 = self._below(self.lc[node], lo, mid, qh)
        c1, w1 = self._below(self.rc[node], mid + 1, hi, qh)
        return c0 + c1, w0 + w1

    def below(self, root: int, tau_q: int) -> Tuple[int, float]:
        """(count, weight-sum) of keys strictly below tau_q at version `root`."""
        if tau_q <= 0:
            return 0, 0.0
        if tau_q > self.U:
            tau_q = self.U
        return self._below(root, 0, self.U - 1, tau_q - 1)

    def n_nodes(self) -> int:
        return len(self.cnt)


class SRIIndex:
    """Round-versioned sealed threshold-aggregate index over a trust ledger.

    Fed one round at a time (``append_round``) in increasing round order, mirroring
    ``TrustStore.append_round``. Query ranges are by **round number** (inclusive);
    ``bisect`` over the appended rounds maps a round range to the two persistent
    versions whose difference answers the query — so non-contiguous round numbers
    are handled correctly."""

    __slots__ = ("tree", "rounds", "roots", "_round_totals")

    def __init__(self, U: int = T_SCORE_DOMAIN):
        self.tree = _PersistentRankSum(U)
        self.rounds: List[int] = []      # appended round numbers (sorted, increasing)
        self.roots: List[int] = [0]      # roots[i] = version after first i rounds
        self._round_totals: List[Tuple[int, float]] = []  # (cnt, wsum) added per round

    def append_round(self, round_num: int, cells: Iterable[Tuple[int, float]]) -> None:
        """Add one round's cells. `cells` yields (t_score_q, w_agg)."""
        if self.rounds and round_num <= self.rounds[-1]:
            raise ValueError(
                f"SRI rounds must be strictly increasing; got {round_num} after {self.rounds[-1]}")
        root = self.roots[-1]
        n = 0
        wsum = 0.0
        for tq, w in cells:
            root = self.tree.insert(root, int(tq), float(w))
            n += 1
            wsum += float(w)
        self.rounds.append(round_num)
        self.roots.append(root)
        self._round_totals.append((n, wsum))

    def _version_upto(self, round_num: int) -> int:
        """Number of appended rounds whose round number is <= round_num."""
        return bisect.bisect_right(self.rounds, round_num)

    def count_below(self, r1: int, r2: int, tau_q: int) -> int:
        """#cells with t_score < tau_q over rounds [r1, r2] inclusive. O(log U)."""
        hi = self._version_upto(r2)
        lo = self._version_upto(r1 - 1)
        c2, _ = self.tree.below(self.roots[hi], tau_q)
        c1, _ = self.tree.below(self.roots[lo], tau_q)
        return c2 - c1

    def weight_below(self, r1: int, r2: int, tau_q: int) -> float:
        """Σ w_agg over cells with t_score < tau_q in rounds [r1, r2]. O(log U)."""
        hi = self._version_upto(r2)
        lo = self._version_upto(r1 - 1)
        _, w2 = self.tree.below(self.roots[hi], tau_q)
        _, w1 = self.tree.below(self.roots[lo], tau_q)
        return w2 - w1

    def count_between(self, r1: int, r2: int, tau_lo: int, tau_hi: int) -> int:
        """#cells with tau_lo <= t_score < tau_hi over [r1, r2]. Two O(log U) queries."""
        return self.count_below(r1, r2, tau_hi) - self.count_below(r1, r2, tau_lo)

    def root_digest(self) -> bytes:
        """SHA-256 over the per-round (count, weight-sum) totals — a compact,
        order-sensitive commitment to the index content. Because the totals are a
        deterministic function of the sealed tuples, equality of this digest after
        a rebuild proves the index matches the ledger (see TrustStore.verify_sri)."""
        h = hashlib.sha256()
        for rnd, (n, wsum) in zip(self.rounds, self._round_totals):
            # quantize wsum to milliunits to keep the commitment exact & portable
            h.update(struct.pack("<iiq", int(rnd), int(n), int(round(wsum * 1000))))
        return h.digest()

    def n_nodes(self) -> int:
        return self.tree.n_nodes()

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)
