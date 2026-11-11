"""SRI native-kernel benchmark.

Compares SRI and the scan when both are native code with compact memory layouts:
both the SRI persistent-segtree build/query AND the scan baseline are numba-JIT
kernels over PACKED contiguous typed arrays (int32 node pool), so neither side
pays Python-interpreter overhead.

Reported measurements:
  (1) warm-kernel latency, JIT compile time reported SEPARATELY;
  (2) memory: used pool / allocated pool / peak RSS;
  (3) range-length workload-shape sweep (narrow windows can favour scan);
  (4) weighted-sum (wsum monoid) query latency, not just count;
  (5) REPS with median / p95 (and mean/std).

    python -m experiment_scripts.dtrust_experiments.c1_audit.sri.sri_native \
        --Ms 2000000 10000000 --reps 5 --out results/recovery/c1_sri
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEPTH_NODES = 18  # log2(65536)=16 internal + leaf + slack, per persistent insert
BYTES_NODE = 20   # int32*3 (lc,rc,cnt) + float64 (wsum, for exact weighted sums)


def _peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # macOS bytes, Linux KB


@njit
def _insert(node, lo, hi, pos, w, lc, rc, cnt, wsum, nxt):
    cur = nxt[0]; nxt[0] += 1
    cnt[cur] = cnt[node] + 1
    wsum[cur] = wsum[node] + w
    if lo == hi:
        lc[cur] = 0; rc[cur] = 0
        return cur
    mid = (lo + hi) >> 1
    if pos <= mid:
        lc[cur] = _insert(lc[node], lo, mid, pos, w, lc, rc, cnt, wsum, nxt)
        rc[cur] = rc[node]
    else:
        rc[cur] = _insert(rc[node], mid + 1, hi, pos, w, lc, rc, cnt, wsum, nxt)
        lc[cur] = lc[node]
    return cur


@njit
def _build(tq, w, round_ptr, lc, rc, cnt, wsum, roots, U):
    nxt = np.zeros(1, dtype=np.int64); nxt[0] = 1   # node 0 = null
    root = 0
    for r in range(len(round_ptr) - 1):
        for j in range(round_ptr[r], round_ptr[r + 1]):
            root = _insert(root, 0, U - 1, tq[j], w[j], lc, rc, cnt, wsum, nxt)
        roots[r + 1] = root
    return nxt[0]


@njit
def _cnt_below(node, lo, hi, qh, lc, rc, cnt):
    if node == 0 or qh < lo:
        return 0
    if hi <= qh:
        return cnt[node]
    mid = (lo + hi) >> 1
    return (_cnt_below(lc[node], lo, mid, qh, lc, rc, cnt)
            + _cnt_below(rc[node], mid + 1, hi, qh, lc, rc, cnt))


@njit
def _wsum_below(node, lo, hi, qh, lc, rc, wsum):
    if node == 0 or qh < lo:
        return 0.0
    if hi <= qh:
        return wsum[node]
    mid = (lo + hi) >> 1
    return (_wsum_below(lc[node], lo, mid, qh, lc, rc, wsum)
            + _wsum_below(rc[node], mid + 1, hi, qh, lc, rc, wsum))


@njit
def _count_below(roots, lc, rc, cnt, r1, r2, tau, U):
    if tau <= 0:
        return 0
    return (_cnt_below(roots[r2 + 1], 0, U - 1, tau - 1, lc, rc, cnt)
            - _cnt_below(roots[r1], 0, U - 1, tau - 1, lc, rc, cnt))


@njit
def _weight_below(roots, lc, rc, wsum, r1, r2, tau, U):
    if tau <= 0:
        return 0.0
    return (_wsum_below(roots[r2 + 1], 0, U - 1, tau - 1, lc, rc, wsum)
            - _wsum_below(roots[r1], 0, U - 1, tau - 1, lc, rc, wsum))


@njit
def _scan_count(tq, start, end, tau):     # native O(M) scan over packed cells
    c = 0
    for i in range(start, end):
        if tq[i] < tau:
            c += 1
    return c


@njit
def _scan_weight(tq, w, start, end, tau):
    s = 0.0
    for i in range(start, end):
        if tq[i] < tau:
            s += w[i]
    return s


@njit
def _bench_sri_count(roots, lc, rc, cnt, qa, qb, qtau, U):
    s = 0
    for k in range(len(qa)):
        s += _count_below(roots, lc, rc, cnt, qa[k], qb[k], qtau[k], U)
    return s


@njit
def _bench_sri_weight(roots, lc, rc, wsum, qa, qb, qtau, U):
    s = 0.0
    for k in range(len(qa)):
        s += _weight_below(roots, lc, rc, wsum, qa[k], qb[k], qtau[k], U)
    return s


@njit
def _bench_scan_count(tq, qa, qb, qtau, N):
    s = 0
    for k in range(len(qa)):
        s += _scan_count(tq, qa[k] * N, (qb[k] + 1) * N, qtau[k])
    return s


@njit
def _bench_scan_weight(tq, w, qa, qb, qtau, N):
    s = 0.0
    for k in range(len(qa)):
        s += _scan_weight(tq, w, qa[k] * N, (qb[k] + 1) * N, qtau[k])
    return s


def _time_reps(fn, args, K, reps):
    """Return per-query us: mean, std, median, p95 over `reps` batch runs."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) / K * 1e6)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    return {"mean": round(st.mean(samples), 3),
            "std": round(st.pstdev(samples), 3),
            "median": round(st.median(samples), 3),
            "p95": round(p95, 3)}


def run(M, N=100, U=65536, reps=5, K=1000):
    R = max(2, M // N); M = R * N
    rng = np.random.default_rng(0)
    tq = (rng.random(M) * (U - 1)).astype(np.int32)
    w = rng.random(M).astype(np.float64)
    round_ptr = np.arange(0, M + 1, N).astype(np.int64)

    cap = M * DEPTH_NODES + 8
    lc = np.zeros(cap, np.int32); rc = np.zeros(cap, np.int32)
    cnt = np.zeros(cap, np.int32); wsum = np.zeros(cap, np.float64)
    roots = np.zeros(R + 1, np.int64)

    t0 = time.perf_counter()
    n_nodes = _build(tq, w, round_ptr, lc, rc, cnt, wsum, roots, U)
    build_s = time.perf_counter() - t0   # (2) includes the build-JIT compile on first M

    rng2 = np.random.default_rng(7)
    ab = np.sort(rng2.integers(0, R, size=(K, 2)), axis=1)
    qa = ab[:, 0].astype(np.int64); qb = ab[:, 1].astype(np.int64)
    qtau = rng2.integers(1, U, size=K).astype(np.int64)

    # (1) JIT compile time = first (cold) call; warm = subsequent
    jit = {}
    for name, fn, a in (("count", _bench_sri_count, (roots, lc, rc, cnt, qa[:2], qb[:2], qtau[:2], U)),
                        ("weight", _bench_sri_weight, (roots, lc, rc, wsum, qa[:2], qb[:2], qtau[:2], U)),
                        ("scan_c", _bench_scan_count, (tq, qa[:2], qb[:2], qtau[:2], N)),
                        ("scan_w", _bench_scan_weight, (tq, w, qa[:2], qb[:2], qtau[:2], N))):
        tc = time.perf_counter(); fn(*a); jit[name] = round(time.perf_counter() - tc, 3)

    # correctness: native SRI == native scan (count AND weight)
    ok_c = all(int(_count_below(roots, lc, rc, cnt, int(qa[k]), int(qb[k]), int(qtau[k]), U))
               == int(_scan_count(tq, int(qa[k]) * N, (int(qb[k]) + 1) * N, int(qtau[k])))
               for k in range(50))
    def _wclose(k):
        sv = _weight_below(roots, lc, rc, wsum, int(qa[k]), int(qb[k]), int(qtau[k]), U)
        cv = _scan_weight(tq, w, int(qa[k]) * N, (int(qb[k]) + 1) * N, int(qtau[k]))
        return abs(sv - cv) <= 1e-6 * max(1.0, abs(cv))   # float64 reassociation tol
    ok_w = all(_wclose(k) for k in range(50))

    # (5) reps -> mean/std/median/p95, for (4) count AND weight
    sri_c = _time_reps(_bench_sri_count, (roots, lc, rc, cnt, qa, qb, qtau, U), K, reps)
    scan_c = _time_reps(_bench_scan_count, (tq, qa, qb, qtau, N), K, reps)
    sri_w = _time_reps(_bench_sri_weight, (roots, lc, rc, wsum, qa, qb, qtau, U), K, reps)
    scan_w = _time_reps(_bench_scan_weight, (tq, w, qa, qb, qtau, N), K, reps)

    # (3) range-length sweep: fixed window length L (rounds), SRI vs scan
    sweep = []
    for L in (1, 4, 16, 64, 256, 1024, R):
        L = min(L, R)
        a0 = rng2.integers(0, R - L + 1, size=K).astype(np.int64)
        b0 = (a0 + (L - 1)).astype(np.int64)
        s_sri = _time_reps(_bench_sri_count, (roots, lc, rc, cnt, a0, b0, qtau, U), K, reps)["median"]
        s_scn = _time_reps(_bench_scan_count, (tq, a0, b0, qtau, N), K, reps)["median"]
        sweep.append({"window_rounds": int(L), "cells": int(L * N),
                      "sri_us": s_sri, "scan_us": s_scn,
                      "speedup": round(s_scn / max(s_sri, 1e-9), 2),
                      "winner": "SRI" if s_sri < s_scn else "scan"})

    return {
        "M": M, "R": R, "N": N, "n_nodes": int(n_nodes),
        "build_s": round(build_s, 2),
        "build_Mcells_s": round(M / build_s / 1e6, 2),
        "mem_used_GB": round(n_nodes * BYTES_NODE / 1e9, 2),
        "mem_alloc_GB": round(cap * BYTES_NODE / 1e9, 2),
        "peak_rss_GB": round(_peak_rss_gb(), 2),
        "bytes_per_cell_used": round(n_nodes * BYTES_NODE / M),
        "bytes_per_cell_alloc": round(cap * BYTES_NODE / M),
        "jit_compile_s": jit,
        "exact_count": bool(ok_c), "exact_weight": bool(ok_w),
        "count_query": {"sri": sri_c, "scan": scan_c,
                        "speedup_median": round(scan_c["median"] / max(sri_c["median"], 1e-9), 2)},
        "weight_query": {"sri": sri_w, "scan": scan_w,
                         "speedup_median": round(scan_w["median"] / max(sri_w["median"], 1e-9), 2)},
        "range_length_sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[2000000, 10000000])
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/recovery/c1_sri")
    args = ap.parse_args()
    rows = []
    for M in args.Ms:
        r = run(M, args.N, reps=args.reps)
        rows.append(r)
        c, wq = r["count_query"], r["weight_query"]
        print(f"\nM={r['M']:,} R={r['R']} | build {r['build_s']}s ({r['build_Mcells_s']} Mcell/s) "
              f"| mem used {r['mem_used_GB']}GB alloc {r['mem_alloc_GB']}GB RSS {r['peak_rss_GB']}GB "
              f"| {r['bytes_per_cell_used']}/{r['bytes_per_cell_alloc']} B/cell")
        print(f"  count : SRI {c['sri']['median']}us (p95 {c['sri']['p95']}) vs scan "
              f"{c['scan']['median']}us -> {c['speedup_median']}x  exact={r['exact_count']}")
        print(f"  weight: SRI {wq['sri']['median']}us (p95 {wq['sri']['p95']}) vs scan "
              f"{wq['scan']['median']}us -> {wq['speedup_median']}x  exact={r['exact_weight']}")
        print(f"  JIT compile (one-time): {r['jit_compile_s']}")
        print(f"  range-length sweep (window_rounds: speedup, winner):")
        for s in r["range_length_sweep"]:
            print(f"    {s['window_rounds']:>7} rounds ({s['cells']:>9,} cells): "
                  f"SRI {s['sri_us']}us vs scan {s['scan_us']}us -> {s['speedup']}x [{s['winner']}]")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "sri_native_bench.json", "w"), indent=2)
    print(f"\nwrote {out/'sri_native_bench.json'}")


if __name__ == "__main__":
    main()
