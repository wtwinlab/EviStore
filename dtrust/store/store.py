"""TrustStore: schema-typed sealed provenance ledger with 4 query operators.

This implements the TrustLog of Section V of the paper: an append-only
relation of 4 KiB sealed chunks chained into a Merkle tree. The relation is
backed by SQLite for its B-tree, with chunk-hash bookkeeping and Merkle-root
computation on top, so integrity holds end-to-end.

The four query operators map to single SQL statements over the indexed
columns; their cost follows SQLite's B-tree, measured by the scaling
benchmarks in experiments/.

This is a single-writer, single-process design intended to be embedded
inside the coordinator's TEE. Concurrent readers are served by SQLite's
WAL mode; the writer holds a serialized transaction per round.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .schema import (
    CHUNK_BODY_SIZE,
    CHUNK_HEADER_SIZE,
    CHUNK_SIZE,
    Mode,
    TUPLE_SIZE,
    TUPLES_PER_CHUNK,
    TrustTuple,
)
from .sri_index import SRIIndex
from . import sri_authenticated as _asri_mod

DEFAULT_ANCHOR_INTERVAL = 256  # M: rounds between Merkle anchors


# ----- helpers ------------------------------------------------------------

def _h(*chunks: bytes) -> bytes:
    """SHA-256 of concatenated bytes; the prototype uses SHA-256 instead
    of Blake3 to keep the dependency surface to stdlib only."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.digest()


def _build_merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Binary Merkle tree over leaf hashes; pairwise SHA-256.

    Empty -> 32 zero bytes. Odd levels duplicate the last leaf, which
    matches RFC 6962 except we don't bother distinguishing leaf vs
    internal nodes since the prototype is internally consistent.
    """
    if not leaves:
        return b"\x00" * 32
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [_h(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


# ----- main store ---------------------------------------------------------

class TrustStore:
    """Append-only, indexed, Merkle-anchored trust state relation.

    SQL indexes:
      * primary B-tree on (participant, round) for pointInTime;
      * secondary index on (t_score, round) for rangeBelow;
      * separate event table for degradedEvents to avoid a full scan.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        anchor_interval: int = DEFAULT_ANCHOR_INTERVAL,
        wal: bool = True,
    ):
        self.db_path = Path(db_path)
        self.anchor_interval = anchor_interval
        self.conn = sqlite3.connect(str(self.db_path))
        if wal:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._init_schema()
        # Sealed Replay Index: in-memory sublinear (O(log U)) threshold-aggregate
        # operators (count_below / weight_below), replayed from the sealed tuples
        # when an existing ledger is reopened so it always matches the relation.
        self.sri = SRIIndex()
        self._rebuild_sri()
        # Authenticated SRI: per-query count_below/weight_below RECEIPTS verified
        # against the sealed auth root with no raw access. Built lazily so it
        # only costs anything when receipts are actually requested.
        self.asri = None

    # ----- schema -------------------------------------------------------

    def _init_schema(self) -> None:
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tuples (
            round       INTEGER NOT NULL,
            participant BLOB    NOT NULL,
            a_conf      INTEGER NOT NULL,
            fresh       INTEGER NOT NULL,
            s_cusum     REAL    NOT NULL,
            t_score     INTEGER NOT NULL,
            b_score     INTEGER NOT NULL,
            w_agg       INTEGER NOT NULL,
            mode        INTEGER NOT NULL,
            h_grad      BLOB    NOT NULL,
            llr_att     INTEGER NOT NULL DEFAULT 0,
            llr_grad    INTEGER NOT NULL DEFAULT 0,
            llr_perf    INTEGER NOT NULL DEFAULT 0,
            llr_beh     INTEGER NOT NULL DEFAULT 0,
            llr_seal    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (participant, round)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_t_round  ON tuples (t_score, round);
        CREATE INDEX IF NOT EXISTS idx_round    ON tuples (round);
        CREATE INDEX IF NOT EXISTS idx_mode     ON tuples (mode, round) WHERE mode != 0;
        -- Drift Replay Index: per-participant ordered
        -- (s_cusum, round) so that "earliest round in [r1,r2] where
        -- s_cusum >= h'" is a single-row LIMIT 1 lookup rather than a
        -- linear scan. Index storage cost is one B-tree leaf entry per
        -- tuple (~16 bytes), comparable to the primary key.
        CREATE INDEX IF NOT EXISTS idx_dri      ON tuples (participant, s_cusum, round);

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    INTEGER PRIMARY KEY,
            first_round INTEGER NOT NULL,
            last_round  INTEGER NOT NULL,
            n_tuples    INTEGER NOT NULL,
            chunk_hash  BLOB    NOT NULL,
            prev_hash   BLOB    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS merkle_anchors (
            anchor_round INTEGER PRIMARY KEY,
            root         BLOB    NOT NULL,
            chunk_count  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        c.commit()

    # ----- writer -------------------------------------------------------

    def append_round(self, round_num: int, tuples: Sequence[TrustTuple]) -> None:
        """Atomically append all tuples for a single round.

        Packs them into 4 KiB chunks for hash bookkeeping, inserts them
        into the indexed `tuples` table, and emits a Merkle anchor every
        `anchor_interval` rounds.
        """
        # Validate round consistency.
        for t in tuples:
            if t.round != round_num:
                raise ValueError(f"tuple round={t.round} != append round={round_num}")

        with self.conn:  # transaction
            self._insert_tuples(tuples)
            self._record_chunks(round_num, tuples)
            if (round_num + 1) % self.anchor_interval == 0:
                self._anchor(round_num)
        # Maintain the SRI incrementally (O(log U)/cell). Outside the SQL
        # transaction: it is a derived, rebuildable view of the sealed rows.
        self.sri.append_round(round_num, ((t.t_score, t.w_agg) for t in tuples))

    def _insert_tuples(self, tuples: Sequence[TrustTuple]) -> None:
        rows = [
            (
                t.round, t.participant, t.a_conf, int(t.fresh), t.s_cusum,
                t.t_score, t.b_score, t.w_agg, int(t.mode), t.h_grad,
                t.llr_att, t.llr_grad, t.llr_perf, t.llr_beh, t.llr_seal,
            )
            for t in tuples
        ]
        self.conn.executemany(
            "INSERT INTO tuples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def _record_chunks(self, round_num: int, tuples: Sequence[TrustTuple]) -> None:
        """Pack into 4 KiB chunks and record per-chunk hashes.

        Tuples within a single round are packed in (participant)-sorted
        order so chunk content is reproducible. Chunks are sized
        TUPLES_PER_CHUNK (=53) tuples; partial last chunk is padded.
        """
        sorted_tuples = sorted(tuples, key=lambda t: t.participant)
        prev_hash_row = self.conn.execute(
            "SELECT chunk_hash FROM chunks ORDER BY chunk_id DESC LIMIT 1"
        ).fetchone()
        prev_hash = prev_hash_row[0] if prev_hash_row else b"\x00" * 32
        next_chunk_id = self.conn.execute(
            "SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM chunks"
        ).fetchone()[0]

        chunks_to_insert = []
        for offset in range(0, len(sorted_tuples), TUPLES_PER_CHUNK):
            window = sorted_tuples[offset : offset + TUPLES_PER_CHUNK]
            body = b"".join(t.pack() for t in window)
            body = body.ljust(CHUNK_BODY_SIZE, b"\x00")
            header = (
                next_chunk_id.to_bytes(4, "little")
                + prev_hash[:28]
            )
            assert len(header) == CHUNK_HEADER_SIZE
            chunk_bytes = header + body
            assert len(chunk_bytes) == CHUNK_SIZE
            chunk_hash = _h(chunk_bytes)
            chunks_to_insert.append((
                next_chunk_id,
                round_num,
                round_num,
                len(window),
                chunk_hash,
                prev_hash,
            ))
            prev_hash = chunk_hash
            next_chunk_id += 1

        if chunks_to_insert:
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                chunks_to_insert,
            )

    def _anchor(self, anchor_round: int) -> None:
        leaves = [
            row[0]
            for row in self.conn.execute(
                "SELECT chunk_hash FROM chunks ORDER BY chunk_id"
            )
        ]
        root = _build_merkle_root(leaves)
        self.conn.execute(
            "INSERT OR REPLACE INTO merkle_anchors VALUES (?, ?, ?)",
            (anchor_round, root, len(leaves)),
        )

    # ----- query operators ---------------------------------------------

    def point_in_time(
        self, participant: bytes, round_num: int
    ) -> Optional[TrustTuple]:
        row = self.conn.execute(
            "SELECT round, participant, a_conf, fresh, s_cusum, "
            "       t_score, b_score, w_agg, mode, h_grad, "
            "       llr_att, llr_grad, llr_perf, llr_beh, llr_seal "
            "FROM tuples WHERE participant=? AND round=?",
            (participant, round_num),
        ).fetchone()
        if row is None:
            return None
        return TrustTuple(
            round=row[0], participant=row[1], a_conf=row[2],
            fresh=bool(row[3]), s_cusum=row[4],
            t_score=row[5], b_score=row[6], w_agg=row[7],
            mode=Mode(row[8]), h_grad=row[9],
            llr_att=row[10], llr_grad=row[11], llr_perf=row[12],
            llr_beh=row[13], llr_seal=row[14],
        )

    def range_below(
        self, r1: int, r2: int, tau_q: int
    ) -> List[Tuple[int, bytes, int]]:
        """Return (round, participant, t_score) for every cell in [r1, r2]
        with t_score < tau_q. Trust scores are quantized uint16."""
        cur = self.conn.execute(
            "SELECT round, participant, t_score "
            "FROM tuples "
            "WHERE round BETWEEN ? AND ? AND t_score < ? "
            "ORDER BY round, participant",
            (r1, r2, tau_q),
        )
        return [(row[0], row[1], row[2]) for row in cur]

    # ----- SRI: sublinear sealed threshold-aggregate operators ----------

    def _rebuild_sri(self) -> None:
        """Replay the sealed tuples into the in-memory SRI in round order.
        Called on open so a reopened ledger has a consistent index."""
        cur = self.conn.execute(
            "SELECT round, t_score, w_agg FROM tuples ORDER BY round"
        )
        cur_round: Optional[int] = None
        cells: List[Tuple[int, int]] = []
        for rnd, t_score, w_agg in cur:
            if cur_round is None:
                cur_round = rnd
            elif rnd != cur_round:
                self.sri.append_round(cur_round, cells)
                cur_round, cells = rnd, []
            cells.append((t_score, w_agg))
        if cur_round is not None:
            self.sri.append_round(cur_round, cells)

    def count_below(self, r1: int, r2: int, tau_q: int) -> int:
        """#cells in rounds [r1, r2] with t_score < tau_q. O(log U) via the SRI
        — the sublinear aggregate counterpart of ``range_below`` (which lists
        rows in O(z)). Exact; verified against the SQL scan in the test suite."""
        return self.sri.count_below(r1, r2, tau_q)

    def weight_below(self, r1: int, r2: int, tau_q: int) -> int:
        """Σ w_agg over cells in [r1, r2] with t_score < tau_q. O(log U) via the
        SRI. This is the "avoided aggregation weight" audit aggregate."""
        return int(round(self.sri.weight_below(r1, r2, tau_q)))

    def count_between(self, r1: int, r2: int, tau_lo: int, tau_hi: int) -> int:
        """#cells in [r1, r2] with tau_lo <= t_score < tau_hi. Two O(log U) queries."""
        return self.sri.count_between(r1, r2, tau_lo, tau_hi)

    def sri_root_digest(self) -> bytes:
        """Compact commitment to the SRI content (per-round count/weight totals)."""
        return self.sri.root_digest()

    # ----- authenticated SRI: per-query count_below RECEIPTS ----------------

    def _ensure_asri(self):
        """Build the authenticated SRI from the sealed tuples on first use. The
        authenticated root commits the full per-cell t_score distribution (not just
        per-round totals), so a count_below answer ships an O(log U) witness verified
        with no raw access — unlike ``verify_sri`` which is totals-only tamper-evidence."""
        if self.asri is not None:
            return
        asri = _asri_mod.AuthSRIIndex()
        cur = self.conn.execute(
            "SELECT round, t_score, w_agg FROM tuples ORDER BY round"
        )
        cur_round = None
        cells = []
        for rnd, t_score, w_agg in cur:
            if cur_round is None:
                cur_round = rnd
            elif rnd != cur_round:
                asri.append_round(cur_round, cells)
                cur_round, cells = rnd, []
            cells.append((int(t_score), float(w_agg)))
        if cur_round is not None:
            asri.append_round(cur_round, cells)
        self.asri = asri

    def auth_sri_root(self) -> bytes:
        """Sealed authenticated-SRI root (binds the per-cell t_score distribution)."""
        self._ensure_asri()
        return self.asri.auth_root()

    def count_below_receipt(self, r1: int, r2: int, tau_q: int):
        """Return (count, weight, proof) where proof is verifiable against
        ``auth_sri_root()`` with no raw access via
        ``sri_authenticated.verify_count_below``."""
        self._ensure_asri()
        count, weight = self.asri.count_below(r1, r2, tau_q)
        proof = self.asri.prove_count_below(r1, r2, tau_q)
        return count, weight, proof

    def verify_sri(self) -> bool:
        """Tamper-evidence: rebuild the SRI from the sealed tuples and check the
        live index commits to the same content. Because the SRI is a deterministic
        function of the sealed rows, digest equality proves it matches the ledger."""
        shadow = TrustStore.__new__(TrustStore)
        shadow.conn = self.conn
        shadow.sri = SRIIndex()
        shadow._rebuild_sri()
        return shadow.sri.root_digest() == self.sri.root_digest()

    def cusum_replay(
        self, participant: bytes, r1: int, r2: int, h_prime: float
    ) -> Optional[int]:
        """Replay CUSUM trajectory against a fresh threshold h_prime.
        Returns the earliest round in [r1, r2] where s_cusum >= h_prime,
        or None if the threshold is never crossed in that interval."""
        cur = self.conn.execute(
            "SELECT round, s_cusum "
            "FROM tuples WHERE participant=? AND round BETWEEN ? AND ? "
            "ORDER BY round",
            (participant, r1, r2),
        )
        for round_, s_cusum in cur:
            if s_cusum >= h_prime:
                return round_
        return None

    def degraded_events(self, r1: int, r2: int) -> List[int]:
        """Return distinct rounds in [r1, r2] where mode != NORMAL."""
        cur = self.conn.execute(
            "SELECT DISTINCT round FROM tuples "
            "WHERE mode != 0 AND round BETWEEN ? AND ? "
            "ORDER BY round",
            (r1, r2),
        )
        return [row[0] for row in cur]

    # ----- advanced operators (DRI / Q5 / Q9) ------------------------

    def dri_replay(
        self, participant: bytes, r1: int, r2: int, h_prime: float
    ) -> Optional[int]:
        """Indexed variant of cusum_replay.

        Uses the (participant, s_cusum, round) secondary index to satisfy the
        ``s_cusum >= h_prime`` predicate without a full per-round scan; SQLite
        still applies an ORDER BY round / LIMIT 1 over the matching rows, so this
        is faster than ``cusum_replay`` but is NOT the asymptotically O(log R)
        earliest-crossing. The true O(log R) path-dependent earliest-crossing is
        ``store.dri_dyadic.DyadicDRI`` (a segment-tree over the reflected walk);
        the prototype store exposes this indexed-SQL path and leaves the dyadic
        structure as a separate, not-yet-wired component. Returns the earliest
        round in [r1, r2] where s_cusum >= h_prime, or None.
        """
        row = self.conn.execute(
            "SELECT round FROM tuples "
            "WHERE participant=? AND round BETWEEN ? AND ? "
            "  AND s_cusum >= ? "
            "ORDER BY round LIMIT 1",
            (participant, r1, r2, h_prime),
        ).fetchone()
        return row[0] if row else None

    def dri_replay_dyadic(
        self, dri: "object", r1: int, r2: int, h_prime: float, s_start: float = 0.0
    ) -> Optional[int]:
        """True O(log R) path-dependent earliest-crossing via a prebuilt dyadic DRI.

        Takes a *prebuilt* ``store.dri_dyadic.DyadicDRI`` over the participant's sealed reflected-walk
        ``residuals`` (built once via ``build_dri(participant)``; the index is the amortized
        structure — building it per call would be O(R)). Residuals are required because s_cusum is
        the walk at the deployed h and cannot replay an arbitrary h' (the advisor prescribes sealing
        residuals for the post-hoc-any-threshold drift family). ``s_start`` is the reflected-walk
        value just before r1 (= 0 for a window that starts the walk; pass W_{r1-1}(h') for a global
        crossing inside [r1,r2]). Returns the earliest round in [r1,r2] with W_t >= h_prime in
        O(log R). Correctness + O(log R) validated in db_eval/dri_scaling.py (dyadic == linear)."""
        return dri.earliest_crossing(h_prime, r1, r2 + 1, s_start=s_start) \
            if "s_start" in dri.earliest_crossing.__code__.co_varnames \
            else dri.earliest_crossing(h_prime, r1, r2 + 1)

    def build_dri(self, residuals: "Sequence[float]") -> "object":
        """Build the (amortized) dyadic DRI once over a participant's sealed residual stream."""
        from .dri_dyadic import DyadicDRI
        return DyadicDRI(list(residuals))

    def safe_checkpoint(
        self,
        round_now: int,
        window: int,
        tau_min_q: int,
        ckpt_period: int = 50,
        max_unsafe_frac: float = 0.0,
    ) -> Optional[int]:
        """Q9: latest checkpoint round at or before round_now whose
        preceding window of size `window` is evidence-safe.

        A checkpoint round c is safe if, in the window [c-window, c],
        the fraction of (round, participant) cells with t_score < tau_min_q
        does not exceed max_unsafe_frac. Returns the largest safe c <= round_now,
        or None if no checkpoint in [0, round_now] is safe.

        Implements Safe-Checkpoint Maximality: the latest
        checkpoint whose preceding window stays under an explicit
        unsafe-influence budget.
        """
        # Walk candidate checkpoints from the latest back, return the first
        # that satisfies the predicate. Each predicate check is one indexed
        # range query.
        c = (round_now // ckpt_period) * ckpt_period
        while c >= 0:
            w_lo = max(0, c - window)
            unsafe_count = self.conn.execute(
                "SELECT COUNT(*) FROM tuples "
                "WHERE round BETWEEN ? AND ? AND t_score < ?",
                (w_lo, c, tau_min_q),
            ).fetchone()[0]
            total = self.conn.execute(
                "SELECT COUNT(*) FROM tuples WHERE round BETWEEN ? AND ?",
                (w_lo, c),
            ).fetchone()[0]
            if total == 0:
                return None
            if unsafe_count / total <= max_unsafe_frac:
                return c
            c -= ckpt_period
        return None

    def counterfactual_filter(
        self, r1: int, r2: int, tau_min_q_old: int, tau_min_q_new: int
    ) -> Tuple[int, int, int]:
        """Q5 (restricted to threshold counterfactual): count how the
        filtered set changes when tau_min is moved from old to new.

        Returns (n_now_filtered_only_under_new, n_now_filtered_only_under_old,
        n_filtered_under_both). The first count is the cells that would have
        been quarantined under the new policy but were not under the old;
        the second count is the cells that would have been admitted under
        the new policy but were not under the old (rare for tightening).
        """
        cur = self.conn.execute(
            "SELECT "
            "  SUM(CASE WHEN t_score >= ? AND t_score < ? THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN t_score < ?  AND t_score >= ? THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN t_score < ?  AND t_score < ?  THEN 1 ELSE 0 END) "
            "FROM tuples WHERE round BETWEEN ? AND ?",
            (
                tau_min_q_old, tau_min_q_new,  # admitted_old, filtered_new
                tau_min_q_old, tau_min_q_new,  # filtered_old, admitted_new
                tau_min_q_old, tau_min_q_new,  # filtered_under_both
                r1, r2,
            ),
        ).fetchone()
        return tuple(int(x or 0) for x in cur)

    def differential_audit(
        self,
        r1: int,
        r2: int,
        tau_min_q_a: int,
        tau_min_q_b: int,
    ) -> List[Tuple[int, bytes, int, str]]:
        """Q6: return per-cell decision flips between two trust policies.

        Each returned row is (round, participant, t_score, side) where side
        is 'A_only' if the cell was admitted under policy B but filtered
        under A, or 'B_only' if filtered under B but admitted under A.
        Cells decided the same way by both policies are not returned.

        This is the structural form a regression-test would consume when
        an operator considers tightening or loosening tau_min: only the
        cells that flip are interesting.
        """
        # SQL UNION: cells filtered under A but not under B + vice versa.
        rows = self.conn.execute(
            "SELECT round, participant, t_score, 'A_only' AS side "
            "FROM tuples "
            "WHERE round BETWEEN ? AND ? AND t_score < ? AND t_score >= ? "
            "UNION ALL "
            "SELECT round, participant, t_score, 'B_only' AS side "
            "FROM tuples "
            "WHERE round BETWEEN ? AND ? AND t_score >= ? AND t_score < ? "
            "ORDER BY round, participant",
            (
                r1, r2, tau_min_q_a, tau_min_q_b,  # A only
                r1, r2, tau_min_q_a, tau_min_q_b,  # B only
            ),
        ).fetchall()
        return [(int(r), p, int(t), s) for (r, p, t, s) in rows]

    def damage_report(
        self,
        r1: int,
        r2: int,
        unsafe_t_q: int,
    ) -> Tuple[int, float, float, int]:
        r"""Q11: aggregate the unsafe-influence accumulated over [r1, r2].

        For every cell whose composite trust fell below `unsafe_t_q` (the
        quantized unsafe threshold), the cell still received non-zero
        aggregation weight $w_{\mathrm{agg}}$ in some deployments; the
        total weight assigned to such cells over the window quantifies
        the cumulative damage that an unsafe regime could have inflicted.

        Returns (n_unsafe_cells, sum_w_agg_unsafe, sum_w_agg_total,
        n_distinct_unsafe_participants). The unsafe-influence ratio is
        sum_w_agg_unsafe / sum_w_agg_total.
        """
        # Three of the four components are threshold count/weight-sum aggregates,
        # answered by the SRI in O(log U) instead of an O(M) SQL scan. Only the
        # distinct-participant count is colored-range-distinct (not a monoid sum),
        # so it stays in SQL — the theory excludes it from the sublinear claim.
        n_unsafe = self.sri.count_below(r1, r2, unsafe_t_q)
        sum_w_unsafe = self.sri.weight_below(r1, r2, unsafe_t_q) / 65535.0  # dequantize
        sum_w_total = self.sri.weight_below(r1, r2, self.sri.tree.U) / 65535.0
        n_participants = int(self.conn.execute(
            "SELECT COUNT(DISTINCT participant) FROM tuples "
            "WHERE round BETWEEN ? AND ? AND t_score < ?",
            (r1, r2, unsafe_t_q),
        ).fetchone()[0] or 0)
        return n_unsafe, sum_w_unsafe, sum_w_total, n_participants

    def dri_batch_replay(
        self,
        participant: bytes,
        r1: int,
        r2: int,
        thresholds: Sequence[float],
    ) -> List[Optional[int]]:
        """H: batched threshold sweep over the DRI.

        Issues one SQL statement that returns, for each h_k in thresholds,
        the earliest round in [r1, r2] where s_cusum >= h_k (or NULL if
        the threshold is never crossed). The K subqueries share the same
        index probe path on the (participant, s_cusum, round) index, so
        the per-threshold cost approaches the index-traversal cost rather
        than re-walking the full window K times.

        Returns a list of length len(thresholds), in input order.
        """
        if not thresholds:
            return []
        # Build a CTE that exposes thresholds as a virtual relation, then
        # join via correlated subquery that scans the DRI only for the
        # exact predicate.  SQLite executes this as one query plan.
        values_clause = ", ".join(f"({i}, ?)" for i in range(len(thresholds)))
        sql = (
            f"WITH thresholds(idx, h) AS (VALUES {values_clause}) "
            "SELECT idx, ("
            "  SELECT round FROM tuples "
            "  WHERE participant = ? AND round BETWEEN ? AND ? "
            "    AND s_cusum >= thresholds.h "
            "  ORDER BY round LIMIT 1"
            ") FROM thresholds ORDER BY idx"
        )
        params = list(thresholds) + [participant, r1, r2]
        rows = self.conn.execute(sql, params).fetchall()
        # rows is in idx order
        return [row[1] for row in rows]

    # ----- integrity ----------------------------------------------------

    def merkle_root(self, anchor_round: Optional[int] = None) -> bytes:
        """Return the Merkle root at the most recent anchor <= round, or
        compute the live root over all chunks if anchor_round is None."""
        if anchor_round is None:
            leaves = [
                row[0]
                for row in self.conn.execute(
                    "SELECT chunk_hash FROM chunks ORDER BY chunk_id"
                )
            ]
            return _build_merkle_root(leaves)
        row = self.conn.execute(
            "SELECT root FROM merkle_anchors "
            "WHERE anchor_round <= ? ORDER BY anchor_round DESC LIMIT 1",
            (anchor_round,),
        ).fetchone()
        return row[0] if row else b"\x00" * 32

    def verify_chunk_chain(self, *, deep: bool = True) -> bool:
        """Verify ledger integrity. Returns False if the host tampered.

        With ``deep=True`` (default) this RECONSTRUCTS each chunk's bytes from
        the current contents of the ``tuples`` table (sorted by participant,
        packed in TUPLES_PER_CHUNK windows, exactly as ``_record_chunks`` did),
        recomputes the chunk hash, and compares it to the stored ``chunk_hash``
        in addition to checking the prev_hash chain. This BINDS the query-
        visible tuples to the hash chain: editing any ``tuples`` row (e.g. a
        single ``t_score``) changes the recomputed chunk hash and is detected.

        With ``deep=False`` only the prev_hash linkage between stored chunks is
        checked (the original, weaker behaviour) -- kept for benchmarking.
        """
        prev = b"\x00" * 32
        if not deep:
            for cid, ch_hash, p_hash in self.conn.execute(
                "SELECT chunk_id, chunk_hash, prev_hash FROM chunks ORDER BY chunk_id"
            ):
                if p_hash != prev:
                    return False
                prev = ch_hash
            return True

        cur_round = None
        sorted_tuples: List[TrustTuple] = []
        offset = 0
        for cid, first_round, n_tuples, ch_hash, p_hash in self.conn.execute(
            "SELECT chunk_id, first_round, n_tuples, chunk_hash, prev_hash "
            "FROM chunks ORDER BY chunk_id"
        ):
            if p_hash != prev:
                return False
            if first_round != cur_round:
                cur_round = first_round
                offset = 0
                rows = self.conn.execute(
                    "SELECT round, participant, a_conf, fresh, s_cusum, "
                    "       t_score, b_score, w_agg, mode, h_grad, "
                    "       llr_att, llr_grad, llr_perf, llr_beh, llr_seal "
                    "FROM tuples WHERE round=?",
                    (first_round,),
                ).fetchall()
                tuples = [
                    TrustTuple(
                        round=r[0], participant=r[1], a_conf=r[2], fresh=bool(r[3]),
                        s_cusum=r[4], t_score=r[5], b_score=r[6], w_agg=r[7],
                        mode=Mode(r[8]), h_grad=r[9],
                        llr_att=r[10], llr_grad=r[11], llr_perf=r[12],
                        llr_beh=r[13], llr_seal=r[14],
                    )
                    for r in rows
                ]
                sorted_tuples = sorted(tuples, key=lambda t: t.participant)
            window = sorted_tuples[offset : offset + TUPLES_PER_CHUNK]
            offset += TUPLES_PER_CHUNK
            if len(window) != n_tuples:
                return False
            body = b"".join(t.pack() for t in window).ljust(CHUNK_BODY_SIZE, b"\x00")
            header = cid.to_bytes(4, "little") + p_hash[:28]
            if _h(header + body) != ch_hash:
                return False
            prev = ch_hash
        return True

    # ----- statistics ---------------------------------------------------

    def stats(self) -> dict:
        n_tuples = self.conn.execute("SELECT COUNT(*) FROM tuples").fetchone()[0]
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_anchors = self.conn.execute("SELECT COUNT(*) FROM merkle_anchors").fetchone()[0]
        n_rounds = self.conn.execute(
            "SELECT COUNT(DISTINCT round) FROM tuples"
        ).fetchone()[0]
        n_participants = self.conn.execute(
            "SELECT COUNT(DISTINCT participant) FROM tuples"
        ).fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "tuples": n_tuples,
            "chunks": n_chunks,
            "anchors": n_anchors,
            "rounds": n_rounds,
            "participants": n_participants,
            "db_bytes": size_bytes,
        }

    # ----- iteration ----------------------------------------------------

    def all_tuples(self) -> Iterator[TrustTuple]:
        cur = self.conn.execute(
            "SELECT round, participant, a_conf, fresh, s_cusum, "
            "       t_score, b_score, w_agg, mode, h_grad, "
            "       llr_att, llr_grad, llr_perf, llr_beh, llr_seal FROM tuples"
        )
        for row in cur:
            yield TrustTuple(
                round=row[0], participant=row[1], a_conf=row[2],
                fresh=bool(row[3]), s_cusum=row[4],
                t_score=row[5], b_score=row[6], w_agg=row[7],
                mode=Mode(row[8]), h_grad=row[9],
                llr_att=row[10], llr_grad=row[11], llr_perf=row[12],
                llr_beh=row[13], llr_seal=row[14],
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "TrustStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
