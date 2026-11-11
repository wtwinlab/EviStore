"""Dyadic mergeable summary for path-dependent CUSUM replay.

The CUSUM trajectory is the reflected random walk
    W_t = max(0, W_{t-1} + r_t),  W_0 = c >= 0.

It is path-dependent because of the max(0, .) reset, so standard
prefix-sum indexing does not support threshold-replay queries.
This module implements a segment-tree-style dyadic summary that
*does* support them in O(log R) time.

Each tree node summarizes an interval of length n with four scalars
derived from the Lindley-Spitzer identity for reflected walks. Let
S_i = r_1 + ... + r_i with S_0 = 0; then over [1..n]:

    sumr     = S_n
    max_pre  = max_{1 <= i <= n} S_i              (non-empty prefix)
    max_suf  = max_{0 <= j < n}  (S_n - S_j)      (non-empty suffix)
    max_sub  = max_{0 <= j < i <= n} (S_i - S_j)  (non-empty sub-array)

These four quantities compose under concatenation L = [1..m), R = [m+1..n):

    sumr_new    = L.sumr + R.sumr
    max_pre_new = max(L.max_pre,  L.sumr + R.max_pre)
    max_suf_new = max(R.max_suf,  L.max_suf + R.sumr)
    max_sub_new = max(L.max_sub,  R.max_sub,  L.max_suf + R.max_pre)

For the reflected walk with start state c >= 0:

    max_{1 <= t <= n} W_t = max( 0,
                                  c + max_pre,
                                  max_sub )
    W_n                   = max( 0,
                                  c + sumr,
                                  max_suf )

(The outer max(0, .) accounts for the reflection.)

The earliest-crossing query "first t in [a, b) where W_t >= h, given
W_{a-1} = c" is answered by walking the dyadic cover of [a, b) left
to right, threading the hot state through each node, and descending
into the first node whose max trajectory upper bound exceeds h. The
descent uses the same recurrence recursively, so the total cost is
O(log R) merge operations plus one leaf verification.

Theorem (DRI Sufficiency, prose form):
    For any threshold h and any sub-interval [a, b) of length R, the
    earliest-crossing round of the reflected walk W with start state
    c is computable from a dyadic decomposition of [a, b) into
    O(log R) summary nodes (sumr, max_pre, max_suf, max_sub). The
    summary is a sufficient statistic: no information needed for any
    threshold-replay query is lost relative to retaining the raw
    residual sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


_NEG_INF = float("-inf")


@dataclass(slots=True)
class Node:
    """Lindley-Spitzer summary of a dyadic interval (strict: empty
    prefix / suffix / sub-array NOT included so the formula correctly
    excludes the W_0 = c degenerate case)."""
    n: int            # interval length (#rounds)
    sumr: float       # sum of residuals
    max_pre: float    # max non-empty prefix sum (-inf if n == 0)
    max_suf: float    # max non-empty suffix sum (-inf if n == 0)
    max_sub: float    # max non-empty sub-array sum (-inf if n == 0)


# Identity element for the merge monoid: an empty interval.
NEUTRAL = Node(n=0, sumr=0.0, max_pre=_NEG_INF, max_suf=_NEG_INF, max_sub=_NEG_INF)


def leaf(r: float) -> Node:
    """Singleton summary from one residual r. The only non-empty prefix /
    suffix / sub-array of a length-1 interval is the singleton itself,
    so all three maxima equal r (which may be negative)."""
    return Node(n=1, sumr=r, max_pre=r, max_suf=r, max_sub=r)


def merge(L: Node, R: Node) -> Node:
    """O(1) concatenation of two adjacent dyadic interval summaries.

    Correctly handles either side being the NEUTRAL identity by virtue
    of -inf propagation and sumr addition.
    """
    if L.n == 0:
        return R
    if R.n == 0:
        return L
    return Node(
        n       = L.n + R.n,
        sumr    = L.sumr + R.sumr,
        max_pre = max(L.max_pre,  L.sumr + R.max_pre),
        max_suf = max(R.max_suf,  L.max_suf + R.sumr),
        max_sub = max(L.max_sub,  R.max_sub,  L.max_suf + R.max_pre),
    )


def max_traj(node: Node, c: float) -> float:
    """Maximum value of the reflected walk W_t for t in [1..n], given
    W_0 = c >= 0. Returns 0 if the bound suggests no positive excursion
    (i.e., W_t reflects to 0 immediately)."""
    if node.n == 0:
        return c
    return max(0.0, c + node.max_pre, node.max_sub)


def end_traj(node: Node, c: float) -> float:
    """End value W_n given W_0 = c >= 0."""
    if node.n == 0:
        return c
    return max(0.0, c + node.sumr, node.max_suf)


class DyadicDRI:
    """Segment tree of Lindley-Spitzer summaries over a residual
    sequence. Tree is stored implicitly in an array with leaves at
    [p, p + n) where p is the next power of 2 >= n. Indices outside
    [p, p + n) hold NEUTRAL so merges are well-defined."""

    __slots__ = ("n", "p", "nodes", "leaf_resids")

    def __init__(self, residuals: Sequence[float]):
        self.n = len(residuals)
        self.p = 1
        while self.p < max(1, self.n):
            self.p *= 2
        self.nodes: List[Node] = [NEUTRAL] * (2 * self.p)
        self.leaf_resids: List[float] = list(residuals)
        for i, r in enumerate(residuals):
            self.nodes[self.p + i] = leaf(r)
        for i in range(self.p - 1, 0, -1):
            self.nodes[i] = merge(self.nodes[2 * i], self.nodes[2 * i + 1])

    # ----- linear-scan oracle ---------------------------------------

    @staticmethod
    def linear_earliest_crossing(
        residuals: Sequence[float],
        h: float,
        a: int = 0,
        b: Optional[int] = None,
        s_start: float = 0.0,
    ) -> Optional[int]:
        """Replay W linearly, return first t in [a, b) with W_t >= h."""
        if b is None:
            b = len(residuals)
        s = s_start
        for t in range(a, b):
            s = max(0.0, s + residuals[t])
            if s >= h:
                return t
        return None

    # ----- dyadic descent -------------------------------------------

    def earliest_crossing(
        self,
        h: float,
        a: int = 0,
        b: Optional[int] = None,
        s_start: float = 0.0,
    ) -> Optional[int]:
        if b is None:
            b = self.n
        if a >= b:
            return None

        hot = s_start
        for ni in self._cover_indices(a, b):
            node = self.nodes[ni]
            if node.n == 0:
                continue
            if max_traj(node, hot) < h:
                hot = end_traj(node, hot)
                continue
            return self._descend(ni, h, hot)
        return None

    def _cover_indices(self, a: int, b: int) -> List[int]:
        """Standard segment-tree cover of [a, b) into O(log R) dyadic
        intervals, in left-to-right order."""
        left_part: List[int] = []
        right_part: List[int] = []
        l = a + self.p
        r = b + self.p
        while l < r:
            if l & 1:
                left_part.append(l)
                l += 1
            if r & 1:
                r -= 1
                right_part.append(r)
            l //= 2
            r //= 2
        return left_part + list(reversed(right_part))

    def _descend(self, ni: int, h: float, hot: float) -> Optional[int]:
        """Descend from an internal node ni whose max_traj under hot >= h
        to the leftmost leaf where the crossing occurs. The recurrence
        on children mirrors the top-level walk: if the left child can
        produce a crossing under the current hot state, recurse left;
        otherwise advance hot through the left child and recurse right."""
        while ni < self.p:
            left = self.nodes[2 * ni]
            right = self.nodes[2 * ni + 1]
            if max_traj(left, hot) >= h:
                ni = 2 * ni
            else:
                hot = end_traj(left, hot)
                ni = 2 * ni + 1
        # Leaf: verify directly.
        leaf_idx = ni - self.p
        if leaf_idx >= self.n:
            return None
        s_after = max(0.0, hot + self.leaf_resids[leaf_idx])
        if s_after >= h:
            return leaf_idx
        return None

    # ----- batched API ----------------------------------------------

    def earliest_crossings_batch(
        self,
        thresholds: Sequence[float],
        a: int = 0,
        b: Optional[int] = None,
        s_start: float = 0.0,
    ) -> List[Optional[int]]:
        """K-threshold sweep. Independent O(K log R) descents; an
        interleaved variant that amortizes the tree walk is left to
        the production engine."""
        return [self.earliest_crossing(h, a, b, s_start) for h in thresholds]
