#!/usr/bin/env python3
"""DB-scale access-method benchmark.

Compares the sealed
access methods (SRI count_below O(log U), DRI earliest_crossing O(log R)) vs production query engines
(DuckDB columnar scan, SQLite B-tree-indexed) and a native numpy scan, at 1e6 -> 1e8 cells.

The query is the measured-client audit aggregate count_below(r1,r2,tau) = #cells in rounds [r1,r2]
with t_score < tau, plus the path-dependent earliest_crossing(h) for DRI. We report median per-query
latency; the SRI/DRI stay ~flat (O(log)) while the engines scan, so the speedup grows with scale.
The benchmark measures single-query latency; concurrent workloads are not part of it.

Run on a many-core box:  python benchmark_db_scale.py --scales 1000000 10000000 100000000
"""
import argparse, json, sqlite3, sys, time
from pathlib import Path
import importlib.util as u
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load(name, rel):
    s = u.spec_from_file_location(name, ROOT / rel); m = u.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


SRI = _load("sri_index", "dtrust/store/sri_index.py")
DRI = _load("dri_dyadic", "dtrust/store/dri_dyadic.py")
U = SRI.T_SCORE_DOMAIN

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False


def _median_us(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort(); return ts[len(ts) // 2]


def gen(n_cells, N, seed=0):
    rng = np.random.default_rng(seed)
    R = n_cells // N
    rounds = np.repeat(np.arange(R, dtype=np.int64), N)
    tscore = rng.integers(0, U, size=R * N, dtype=np.int64)
    weight = rng.integers(1, 101, size=R * N).astype(np.float64)
    return R, rounds, tscore, weight


def bench_scale(n_cells, N, queries, seed=0):
    R, rounds, tscore, weight = gen(n_cells, N, seed)
    n = rounds.size
    res = {"n_cells": int(n), "rounds": int(R), "participants": int(N), "queries": queries}

    # query set: random windows over the full range, fixed tau = U/2
    rng = np.random.default_rng(123)
    tau = U // 2
    qs = []
    for _ in range(queries):
        a = int(rng.integers(0, R)); b = int(rng.integers(a, R)); qs.append((a, b))

    # ---- SRI (sealed access method, O(log U)) ----
    t0 = time.perf_counter()
    sri = SRI.SRIIndex(U)
    for r in range(R):
        lo = r * N; sri.append_round(r, list(zip(tscore[lo:lo + N].tolist(), weight[lo:lo + N].tolist())))
    res["sri_build_s"] = round(time.perf_counter() - t0, 2)
    res["sri_us"] = round(_median_us(lambda: [sri.count_below(a, b, tau) for a, b in qs[:50]], 5) / 50, 3)

    # correctness oracle on a few queries (native scan)
    ok = True
    for a, b in qs[:5]:
        mask = (rounds >= a) & (rounds <= b) & (tscore < tau)
        if int(mask.sum()) != sri.count_below(a, b, tau):
            ok = False; break
    res["sri_correct_vs_scan"] = ok

    # ---- native numpy scan ----
    def np_scan():
        a, b = qs[0]
        mask = (rounds >= a) & (rounds <= b) & (tscore < tau); int(mask.sum())
    res["numpy_scan_us"] = round(_median_us(np_scan, 20), 1)

    # ---- DuckDB columnar ----
    if HAVE_DUCKDB:
        con = duckdb.connect()
        con.execute("CREATE TABLE led(round BIGINT, t_score BIGINT, weight DOUBLE)")
        con.register("df", {"round": rounds, "t_score": tscore, "weight": weight})
        t0 = time.perf_counter()
        con.execute("INSERT INTO led SELECT * FROM df"); con.unregister("df")
        res["duckdb_load_s"] = round(time.perf_counter() - t0, 2)
        def duck():
            a, b = qs[0]
            con.execute("SELECT count(*) FROM led WHERE round BETWEEN ? AND ? AND t_score < ?",
                        [a, b, tau]).fetchone()
        res["duckdb_us"] = round(_median_us(duck, 20), 1)
        con.close()

    # ---- SQLite B-tree indexed ----
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE led(round INT, t_score INT, weight REAL)")
    con.executemany("INSERT INTO led VALUES(?,?,?)",
                    zip(rounds.tolist(), tscore.tolist(), weight.tolist()))
    t0 = time.perf_counter()
    con.execute("CREATE INDEX ix ON led(round, t_score)"); con.commit()
    res["sqlite_index_s"] = round(time.perf_counter() - t0, 2)
    def sl():
        a, b = qs[0]
        con.execute("SELECT count(*) FROM led WHERE round BETWEEN ? AND ? AND t_score < ?",
                    (a, b, tau)).fetchone()
    res["sqlite_btree_us"] = round(_median_us(sl, 20), 1)
    con.close()

    # speedups vs SRI
    for eng in ("numpy_scan", "duckdb", "sqlite_btree"):
        k = f"{eng}_us"
        if k in res and res["sri_us"] > 0:
            res[f"sri_speedup_vs_{eng}"] = round(res[k] / res["sri_us"], 1)

    # ---- DRI earliest_crossing vs linear (path-dependent, O(log R)) ----
    # one participant's CUSUM residual stream of length R
    resid = (rng.standard_normal(R) * 0.5 + 0.05)
    t0 = time.perf_counter(); dri = DRI.DyadicDRI(resid.tolist()); res["dri_build_s"] = round(time.perf_counter() - t0, 2)
    hq = [0.5, 1.0, 2.0, 4.0, 8.0]
    res["dri_us"] = round(_median_us(lambda: [dri.earliest_crossing(h) for h in hq], 20) / len(hq), 3)
    res["dri_linear_us"] = round(_median_us(
        lambda: [DRI.DyadicDRI.linear_earliest_crossing(resid.tolist(), h, 0, R) for h in hq], 5) / len(hq), 1)
    if res["dri_us"] > 0:
        res["dri_speedup_vs_linear"] = round(res["dri_linear_us"] / res["dri_us"], 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[1_000_000, 10_000_000])
    ap.add_argument("--participants", type=int, default=100)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results/artifact/B5_db_scale.json"))
    args = ap.parse_args()
    out = {"_note": "B5 DB-scale benchmark: sealed access methods (SRI O(log U), DRI O(log R)) vs "
                    "DuckDB columnar + SQLite B-tree + native scan. Median per-query us, "
                    "single-query latency. count_below(r1,r2,tau) audit aggregate; earliest_crossing(h) drift.",
           "have_duckdb": HAVE_DUCKDB, "U": U, "scales": []}
    for nc in args.scales:
        print(f"=== scale {nc:,} cells (N={args.participants}) ===", flush=True)
        r = bench_scale(nc, args.participants, args.queries)
        out["scales"].append(r)
        print(json.dumps(r, indent=2), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
