"""Sealed Replay Index (SRI) — the threshold-monoid core + audit-workload benchmark.

Implements the rank/sum monoid (the centerpiece that makes Q5/Q6-count, Q11
count+sum, and Q9 window queries sublinear and incrementally maintainable) as an
**append-only persistent segment tree** over quantized t_score, versioned by
round:

  count_below(tau, [r1,r2]) = below(root[r2], tau) - below(root[r1-1], tau)      O(log U)
  avoidedWeight(tau,[r1,r2]) = wsum_below(root[r2], tau) - wsum_below(root[r1-1])  O(log U)

Each cell insert creates O(log U) new nodes (persistent), so the index is built
online in O(log U) amortized per cell and is a frozen, queryable snapshot for the
audit — no per-audit rebuild. Answers are EXACT (verified against a scan oracle).

Run:
    python -m experiment_scripts.dtrust_experiments.c1_audit.sri.sri \
        --verify --bench --R 2000 --N 100 --out results/recovery/c1_sri
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PersistentRankSum:
    """Persistent segment tree over key domain [0, U); versions chained per insert.
    Each node carries (count, weight-sum) of keys in its sub-range. O(log U)/op."""

    def __init__(self, U: int = 65536):
        self.U = U
        # node 0 is the shared empty/null node
        self.lc = [0]; self.rc = [0]; self.cnt = [0]; self.wsum = [0.0]
        sys.setrecursionlimit(100000)

    def _new(self, lc, rc, cnt, wsum):
        self.lc.append(lc); self.rc.append(rc); self.cnt.append(cnt); self.wsum.append(wsum)
        return len(self.cnt) - 1

    def _update(self, node, lo, hi, pos, w):
        if lo == hi:
            return self._new(0, 0, self.cnt[node] + 1, self.wsum[node] + w)
        mid = (lo + hi) >> 1
        if pos <= mid:
            nlc = self._update(self.lc[node], lo, mid, pos, w); nrc = self.rc[node]
        else:
            nrc = self._update(self.rc[node], mid + 1, hi, pos, w); nlc = self.lc[node]
        return self._new(nlc, nrc, self.cnt[node] + 1, self.wsum[node] + w)

    def insert(self, root, pos, w=1.0):
        return self._update(root, 0, self.U - 1, pos, w)

    def _below(self, node, lo, hi, qh):
        # sum of (cnt, wsum) for keys in [0, qh]
        if node == 0 or qh < lo:
            return 0, 0.0
        if hi <= qh:
            return self.cnt[node], self.wsum[node]
        mid = (lo + hi) >> 1
        c0, w0 = self._below(self.lc[node], lo, mid, qh)
        c1, w1 = self._below(self.rc[node], mid + 1, hi, qh)
        return c0 + c1, w0 + w1

    def below(self, root, tau_q):
        """(count, wsum) of keys strictly below tau_q."""
        if tau_q <= 0:
            return 0, 0.0
        return self._below(root, 0, self.U - 1, tau_q - 1)

    def n_nodes(self):
        return len(self.cnt)


class SRI:
    """Sealed Replay Index threshold-monoid over a (round-major) ledger."""

    def __init__(self, tscore_q, round_ptr, w_agg, U=65536):
        """tscore_q: int[M] quantized t_score, cells in round order.
        round_ptr: int[R+1], cells of round r are [round_ptr[r], round_ptr[r+1]).
        w_agg: float[M]."""
        self.R = len(round_ptr) - 1
        self.tree = PersistentRankSum(U)
        self.roots = [0] * (self.R + 1)  # roots[r] = version after rounds 0..r-1
        t0 = time.perf_counter()
        root = 0
        for r in range(self.R):
            for j in range(round_ptr[r], round_ptr[r + 1]):
                root = self.tree.insert(root, int(tscore_q[j]), float(w_agg[j]))
            self.roots[r + 1] = root
        self.build_s = time.perf_counter() - t0

    def count_below(self, tau_q, r1, r2):
        c2, _ = self.tree.below(self.roots[r2 + 1], tau_q)
        c1, _ = self.tree.below(self.roots[r1], tau_q)
        return c2 - c1

    def weight_below(self, tau_q, r1, r2):
        _, w2 = self.tree.below(self.roots[r2 + 1], tau_q)
        _, w1 = self.tree.below(self.roots[r1], tau_q)
        return w2 - w1

    def diff_count(self, tau_a, tau_b, r1, r2):
        """Q6/Q5 count: cells with tau_a <= t_score < tau_b in [r1,r2]."""
        return self.count_below(tau_b, r1, r2) - self.count_below(tau_a, r1, r2)


# ----------------------------------------------------------- synthetic ledger

def make_ledger(R, N, seed=0, U=65536):
    rng = np.random.default_rng(seed)
    tq = (rng.random((R, N)) * (U - 1)).astype(np.int64).ravel()        # round-major
    w = (rng.random((R, N))).astype(np.float64).ravel()
    round_ptr = np.arange(0, R * N + 1, N)
    return tq, w, round_ptr, R, N


# ----------------------------------------------------------- scan oracle

def scan_count_below(tq2d, tau_q, r1, r2):
    return int((tq2d[r1:r2 + 1] < tau_q).sum())


def scan_weight_below(tq2d, w2d, tau_q, r1, r2):
    sl = tq2d[r1:r2 + 1] < tau_q
    return float(w2d[r1:r2 + 1][sl].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--U", type=int, default=65536)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 5, 10, 50, 100, 1000])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--scale", action="store_true",
                    help="per-query latency vs M (build excluded): SRI O(log U) flat vs scan O(M)")
    ap.add_argument("--scale_M", type=int, nargs="+",
                    default=[10000, 50000, 100000, 500000, 1000000, 2000000])
    ap.add_argument("--db", action="store_true",
                    help="compare SRI vs real DBs (DuckDB indexed + columnar scan, SQLite indexed) + accounting")
    ap.add_argument("--db_M", type=int, default=2000000)
    ap.add_argument("--out", default="results/recovery/c1_sri")
    args = ap.parse_args()

    if args.db:
        import duckdb, sqlite3, tempfile, os
        N = 100; R = max(2, args.db_M // N)
        tq, w, round_ptr, R, N = make_ledger(R, N, U=args.U)
        M = R * N
        rnd = np.repeat(np.arange(R), N)
        print(f"=== real-DB baselines @ M={M} (R={R} N={N}) ===")
        # SRI build + accounting
        s = SRI(tq, round_ptr, w, U=args.U)
        bytes_per_cell = s.tree.n_nodes() * 4 * 8 / M   # 4 int/float lists * 8B
        print(f"SRI: build {s.build_s:.2f}s ({M/s.build_s/1e6:.2f}M cells/s online) | "
              f"~{s.tree.n_nodes()/M:.1f} nodes/cell ≈ {bytes_per_cell:.0f} B/cell")
        # DuckDB: table sorted by round (zonemap prune) + ART index on t_score
        con = duckdb.connect()
        con.execute("CREATE TABLE cells(round INTEGER, t_score INTEGER, w DOUBLE)")
        con.register("df", {"round": rnd, "t_score": tq, "w": w})
        con.execute("INSERT INTO cells SELECT * FROM df ORDER BY round")
        # DuckDB's fast path is vectorized columnar scan with round min-max
        # (zonemap) pruning; a secondary ART index on the low-selectivity t_score
        # predicate is pathological, so we use the columnar-scan configuration.
        con.execute("PRAGMA threads=4")
        con.execute("SELECT count(*) FROM cells WHERE t_score < 1").fetchone()  # warm
        # SQLite indexed
        sq = sqlite3.connect(":memory:")
        sq.execute("CREATE TABLE cells(round INT, t_score INT, w REAL)")
        sq.executemany("INSERT INTO cells VALUES (?,?,?)",
                       zip(rnd.tolist(), tq.tolist(), w.tolist()))
        sq.execute("CREATE INDEX idx ON cells(round, t_score)"); sq.commit()
        rng = np.random.default_rng(7)
        qs = [(*sorted(int(x) for x in rng.integers(0, R, 2)), int(rng.integers(1, args.U))) for _ in range(100)]
        def timeit(fn):
            t0 = time.perf_counter()
            for a, b, tau in qs: fn(a, b, tau)
            return (time.perf_counter() - t0) / len(qs) * 1e6
        res = {
            "SRI O(log U)": timeit(lambda a, b, tau: s.count_below(tau, a, b)),
            "numpy scan O(M)": timeit(lambda a, b, tau: int((tq[a*N:(b+1)*N] < tau).sum())),
            "DuckDB (columnar scan)": timeit(lambda a, b, tau: con.execute(
                "SELECT count(*) FROM cells WHERE round BETWEEN ? AND ? AND t_score < ?", [a, b, tau]).fetchone()[0]),
            "SQLite (idx round,tscore)": timeit(lambda a, b, tau: sq.execute(
                "SELECT count(*) FROM cells WHERE round BETWEEN ? AND ? AND t_score < ?", (a, b, tau)).fetchone()[0]),
        }
        # correctness vs SRI
        ok = all(s.count_below(tau, a, b) == con.execute("SELECT count(*) FROM cells WHERE round BETWEEN ? AND ? AND t_score < ?", [a, b, tau]).fetchone()[0] for a, b, tau in qs[:20])
        print(f"correctness SRI==DuckDB (20 q): {'PASS' if ok else 'FAIL'}")
        print(f"\n{'engine':>28} {'us/query':>10} {'vs SRI':>8}")
        base = res["SRI O(log U)"]
        for k in ("SRI O(log U)", "numpy scan O(M)", "DuckDB (columnar scan)", "SQLite (idx round,tscore)"):
            print(f"{k:>28} {res[k]:>10.2f} {res[k]/base:>7.2f}x")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump({"M": M, "build_s": s.build_s, "bytes_per_cell": bytes_per_cell,
                   "nodes_per_cell": s.tree.n_nodes()/M, "us_per_query": res,
                   "correct_vs_duckdb": ok}, open(out / "sri_db_bench.json", "w"), indent=2)
        print(f"wrote {out/'sri_db_bench.json'}")
        return

    if args.scale:
        print(f"=== per-query latency vs M (K=200 queries, build EXCLUDED: it is "
              f"one-time online) ===\n{'M':>9} {'build_s':>8} {'SRI_us/q':>9} "
              f"{'scan_us/q':>10} {'speedup':>8}")
        srows = []
        for M in args.scale_M:
            N = 100; R = max(2, M // N)
            tq, w, round_ptr, R, N = make_ledger(R, N, U=args.U)
            tq2d = tq.reshape(R, N)
            s = SRI(tq, round_ptr, w, U=args.U)
            rng = np.random.default_rng(7)
            qs = [(*sorted(rng.integers(0, R, 2)), int(rng.integers(1, args.U))) for _ in range(200)]
            t0 = time.perf_counter()
            for a, b, tau in qs: s.count_below(tau, a, b)
            sri_us = (time.perf_counter() - t0) / len(qs) * 1e6
            t0 = time.perf_counter()
            for a, b, tau in qs: scan_count_below(tq2d, tau, a, b)
            scan_us = (time.perf_counter() - t0) / len(qs) * 1e6
            srows.append({"M": R * N, "build_s": round(s.build_s, 2),
                          "sri_us": round(sri_us, 2), "scan_us": round(scan_us, 2),
                          "speedup": round(scan_us / max(sri_us, 1e-9), 2)})
            print(f"{R*N:>9} {s.build_s:>8.2f} {sri_us:>9.2f} {scan_us:>10.2f} "
                  f"{scan_us/max(sri_us,1e-9):>8.2f}")
            del s
        xover = next((r["M"] for r in srows if r["speedup"] >= 1.0), None)
        print(f"\nSRI per-query is ~flat (O(log U)); scan grows ~linearly (O(M)). "
              f"per-query crossover: M≈{xover}")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump({"scale": srows, "crossover_M": xover}, open(out / "sri_scale.json", "w"), indent=2)
        print(f"wrote {out/'sri_scale.json'}")
        return

    tq, w, round_ptr, R, N = make_ledger(args.R, args.N, U=args.U)
    tq2d = tq.reshape(R, N); w2d = w.reshape(R, N)
    M = R * N
    print(f"ledger M={M} (R={R} N={N}) U={args.U}")

    sri = SRI(tq, round_ptr, w, U=args.U)
    print(f"SRI build {sri.build_s:.2f}s | nodes={sri.tree.n_nodes()} "
          f"(~{sri.tree.n_nodes()/M:.1f}/cell, expect ~log2(U)={np.log2(args.U):.0f})")

    rng = np.random.default_rng(123)
    def rand_query():
        a, b = sorted(rng.integers(0, R, 2)); tau = int(rng.integers(1, args.U))
        return a, b, tau

    if args.verify:
        ok = True
        for _ in range(300):
            a, b, tau = rand_query()
            if sri.count_below(tau, a, b) != scan_count_below(tq2d, tau, a, b):
                ok = False; break
            if abs(sri.weight_below(tau, a, b) - scan_weight_below(tq2d, w2d, tau, a, b)) > 1e-6:
                ok = False; break
        print(f"VERIFY exact vs scan oracle (count+weight, 300 queries): {'PASS' if ok else 'FAIL'}")

    rows = []
    if args.bench:
        # pre-draw a fixed query batch
        maxK = max(args.Ks)
        Q = [rand_query() for _ in range(maxK)]
        for K in args.Ks:
            qs = Q[:K]
            t0 = time.perf_counter()
            for a, b, tau in qs:
                sri.count_below(tau, a, b)
            sri_q = time.perf_counter() - t0
            t0 = time.perf_counter()
            for a, b, tau in qs:
                scan_count_below(tq2d, tau, a, b)
            scan_q = time.perf_counter() - t0
            # total audit cost: SRI = one-time build + K queries; scan = K scans (no build)
            sri_total = sri.build_s + sri_q
            rows.append({"K": K, "sri_query_ms": round(sri_q * 1e3, 3),
                         "scan_query_ms": round(scan_q * 1e3, 3),
                         "sri_total_ms (build+Kq)": round(sri_total * 1e3, 1),
                         "scan_total_ms": round(scan_q * 1e3, 1),
                         "speedup_total": round(scan_q / max(sri_total, 1e-9), 2)})
        print(f"\n{'K':>6} {'SRI_q_ms':>10} {'scan_q_ms':>10} {'SRI_tot(build+Kq)':>18} "
              f"{'scan_tot':>10} {'tot_speedup':>11}")
        for r in rows:
            print(f"{r['K']:>6} {r['sri_query_ms']:>10} {r['scan_query_ms']:>10} "
                  f"{r['sri_total_ms (build+Kq)']:>18} {r['scan_total_ms']:>10} "
                  f"{r['speedup_total']:>11}")
        crossover = next((r["K"] for r in rows if r["speedup_total"] >= 1.0), None)
        print(f"\nper-query: SRI is O(log U) (flat in M); scan is O(M). "
              f"audit-workload crossover (SRI total incl. build beats scan): K≈{crossover}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"R": R, "N": N, "M": M, "U": args.U, "build_s": sri.build_s,
               "n_nodes": sri.tree.n_nodes(), "bench": rows},
              open(out / "sri_bench.json", "w"), indent=2)
    print(f"wrote {out/'sri_bench.json'}")


if __name__ == "__main__":
    main()
