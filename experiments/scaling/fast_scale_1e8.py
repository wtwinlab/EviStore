#!/usr/bin/env python3
"""Fast REAL 1e8 point for the SCALABLE comparators only: numpy native scan, DuckDB, and the
Fenwick (bincount) index class -- all O(n)-build-fast or O(U)-fast, so they reach 1e8 real cells in
minutes, unlike the pure-Python persistent-segment-tree build and the per-row SQLite insert (which the
authenticated prototype runs up to about 1e6 cells).

Each value is REAL (quantized to U=2^16); we report the index-class flat latency vs the scan engines'
growth at 1e8 real cells across the diverse datasets. Reuses real_serving_bench.load_real.
"""
import sys, time, json
from pathlib import Path
import importlib.util as ilu
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
_s = ilu.spec_from_file_location("rsb", str(ROOT / "experiment_scripts/dtrust_experiments/db_eval/real_serving_bench.py"))
rsb = ilu.module_from_spec(_s); sys.modules["rsb"] = rsb; _s.loader.exec_module(rsb)
U, Fenwick, load_real = rsb.U, rsb.Fenwick, rsb.load_real
try:
    import duckdb; HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False


def med_us(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort(); return ts[len(ts) // 2]


def bench_fast(workload, n_cells):
    q, src = load_real(workload, n_cells)
    n = q.size; tau = U // 2
    res = {"workload": workload, "source": src, "n_cells": int(n), "scale_target": n_cells, "tau_q": tau}
    truth = int((q < tau).sum())
    # Fenwick (the flat index class, O(U) build via bincount, O(log U) query)
    freq = np.bincount(q, minlength=U)[:U]
    t0 = time.perf_counter(); fen = Fenwick(freq); res["fenwick_build_s"] = round(time.perf_counter() - t0, 3)
    res["fenwick_us"] = round(med_us(lambda: fen.count_below(tau), 100), 4)
    res["fenwick_correct"] = (fen.count_below(tau) == truth)
    # native numpy scan
    res["numpy_scan_us"] = round(med_us(lambda: int((q < tau).sum()), 15), 1)
    # DuckDB
    if HAVE_DUCKDB:
        con = duckdb.connect(); con.execute("CREATE TABLE t(v BIGINT)")
        con.register("df", {"v": q}); con.execute("INSERT INTO t SELECT * FROM df"); con.unregister("df")
        res["duckdb_us"] = round(med_us(lambda: con.execute("SELECT count(*) FROM t WHERE v<?", [tau]).fetchone(), 15), 1)
        con.close()
    base = res["fenwick_us"]
    for eng in ("numpy_scan", "duckdb"):
        k = f"{eng}_us"
        if k in res and base > 0: res[f"index_speedup_vs_{eng}"] = round(res[k] / base, 1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["taxi_big", "ghcn", "epa_hourly", "hvfhv"])
    ap.add_argument("--scale", type=int, default=100_000_000)
    ap.add_argument("--out", default=str(ROOT / "results/artifact/E2_fast_1e8.json"))
    a = ap.parse_args()
    out = {"_note": "REAL 1e8 point, scalable comparators only (Fenwick index class vs numpy-scan/DuckDB); "
                    "pure-Python PST + per-row SQLite + authenticated structures cap near 1e6 (honest, stated).",
           "U": U, "have_duckdb": HAVE_DUCKDB, "results": []}
    for w in a.workloads:
        print(f"=== {w} @ {a.scale:,} real cells ===", flush=True)
        try:
            r = bench_fast(w, a.scale); out["results"].append(r)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  SKIP {w}: {type(e).__name__}: {e}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
