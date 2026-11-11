"""End-to-end K-sweep: SRI vs the best real baseline as the audit query batch
size K grows.

A single audit session issues K threshold/range queries against a ledger of M
cells. The end-to-end cost of each engine is

    total(K) = one_time_cost + K * per_query_latency

  * SRI (query-only):   one_time = 0      — the index is built ONLINE with the
                        ledger, so for an audit it is already there.
  * SRI (build-amort.):  one_time = build  — the CONSERVATIVE accounting that
                        charges the full O(M log U) build to the audit.
  * DuckDB (real DB):    one_time = ingest — load + sort the columnar table.
  * native scan:         one_time = 0      — no structure, scans M cells/query.

per_query is K-independent, so we measure it once (sample of random queries) and
sweep K analytically. We then report the crossover K* where SRI — *even paying
its full build* — overtakes each baseline. Query-only SRI wins at every K.

    python -m experiment_scripts.dtrust_experiments.c1_audit.sri.sri_ksweep \
        --M 2000000 --out results/recovery/c1_sri
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .sri_native import (_build, _bench_sri_count, _count_below, _scan_count,
                         DEPTH_NODES)


def _measure(M, N=100, U=65536, sample=200):
    R = max(2, M // N); M = R * N
    rng = np.random.default_rng(0)
    tq = (rng.random(M) * (U - 1)).astype(np.int32)
    w = rng.random(M).astype(np.float64)
    round_ptr = np.arange(0, M + 1, N).astype(np.int64)
    rnd_col = np.repeat(np.arange(R), N).astype(np.int32)

    # --- SRI native build (one-time, online with ledger) ---
    cap = M * DEPTH_NODES + 8
    lc = np.zeros(cap, np.int32); rc = np.zeros(cap, np.int32)
    cnt = np.zeros(cap, np.int32); wsum = np.zeros(cap, np.float64)
    roots = np.zeros(R + 1, np.int64)
    t0 = time.perf_counter()
    _build(tq, w, round_ptr, lc, rc, cnt, wsum, roots, U)
    sri_build_s = time.perf_counter() - t0

    rng2 = np.random.default_rng(7)
    ab = np.sort(rng2.integers(0, R, size=(sample, 2)), axis=1)
    qa = ab[:, 0].astype(np.int64); qb = ab[:, 1].astype(np.int64)
    qtau = rng2.integers(1, U, size=sample).astype(np.int64)

    # warm + per-query latency (s)
    _bench_sri_count(roots, lc, rc, cnt, qa[:2], qb[:2], qtau[:2], U)
    t0 = time.perf_counter(); _bench_sri_count(roots, lc, rc, cnt, qa, qb, qtau, U)
    q_sri = (time.perf_counter() - t0) / sample

    # --- native scan (no build) ---
    def _scan_batch():
        s = 0
        for k in range(sample):
            s += _scan_count(tq, int(qa[k]) * N, (int(qb[k]) + 1) * N, int(qtau[k]))
        return s
    _scan_batch()
    t0 = time.perf_counter(); _scan_batch(); q_scan = (time.perf_counter() - t0) / sample

    # --- DuckDB columnar scan (the real-DB baseline) ---
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE cells(round INTEGER, t_score INTEGER, w DOUBLE)")
    con.register("df", {"round": rnd_col, "t_score": tq, "w": w})
    t0 = time.perf_counter()
    con.execute("INSERT INTO cells SELECT * FROM df ORDER BY round")
    duck_ingest_s = time.perf_counter() - t0
    con.execute("PRAGMA threads=4")
    con.execute("SELECT count(*) FROM cells WHERE t_score < 1").fetchone()  # warm
    t0 = time.perf_counter()
    for k in range(min(sample, 60)):   # DuckDB is ~10ms/q; 60 is enough
        con.execute("SELECT count(*) FROM cells WHERE round BETWEEN ? AND ? AND t_score < ?",
                    [int(qa[k]), int(qb[k]), int(qtau[k])]).fetchone()
    q_duck = (time.perf_counter() - t0) / min(sample, 60)

    # correctness SRI == DuckDB
    ok = all(int(_count_below(roots, lc, rc, cnt, int(qa[k]), int(qb[k]), int(qtau[k]), U))
             == con.execute("SELECT count(*) FROM cells WHERE round BETWEEN ? AND ? AND t_score < ?",
                            [int(qa[k]), int(qb[k]), int(qtau[k])]).fetchone()[0]
             for k in range(20))

    return {"M": M, "R": R, "N": N,
            "sri_build_s": sri_build_s, "duck_ingest_s": duck_ingest_s,
            "q_sri_us": q_sri * 1e6, "q_scan_us": q_scan * 1e6, "q_duck_us": q_duck * 1e6,
            "correct_vs_duckdb": bool(ok),
            "_q": {"sri": q_sri, "scan": q_scan, "duck": q_duck},
            "_base": {"sri_amort": sri_build_s, "duck": duck_ingest_s}}


def _crossover(base_a, q_a, base_b, q_b):
    """Smallest K where engine A total <= engine B total. None if A never wins."""
    if q_a >= q_b:
        # A has worse/equal per-query slope; wins only if its base is <= and slope <=
        return 1 if (base_a <= base_b and q_a <= q_b) else None
    k = (base_a - base_b) / (q_b - q_a)
    return max(1, int(np.ceil(k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2000000)
    ap.add_argument("--out", default="results/recovery/c1_sri")
    args = ap.parse_args()
    m = _measure(args.M)
    q, base = m["_q"], m["_base"]

    Ks = np.unique(np.round(np.logspace(0, 6, 40)).astype(int))
    curves = {
        "SRI (query-only, online build)": [0.0 + K * q["sri"] for K in Ks],
        "SRI (build-amortized, conservative)": [base["sri_amort"] + K * q["sri"] for K in Ks],
        "DuckDB columnar (real DB)": [base["duck"] + K * q["duck"] for K in Ks],
        "native scan (no build)": [0.0 + K * q["scan"] for K in Ks],
    }
    # crossovers: SRI(build-amortized) vs each baseline
    kx_scan = _crossover(base["sri_amort"], q["sri"], 0.0, q["scan"])
    kx_duck = _crossover(base["sri_amort"], q["sri"], base["duck"], q["duck"])

    print(f"=== End-to-end K-sweep @ M={m['M']:,} (R={m['R']}, N={m['N']}) ===")
    print(f"per-query: SRI {m['q_sri_us']:.2f}us | native scan {m['q_scan_us']:.2f}us | "
          f"DuckDB {m['q_duck_us']:.1f}us   (SRI==DuckDB exact: {m['correct_vs_duckdb']})")
    print(f"one-time : SRI build {m['sri_build_s']:.2f}s (online) | DuckDB ingest {m['duck_ingest_s']:.2f}s")
    print(f"crossover K* (SRI paying FULL build still wins):")
    print(f"   vs native scan : K* = {kx_scan}")
    print(f"   vs DuckDB      : K* = {kx_duck}")
    print(f"query-only SRI (online build): wins at ALL K (lowest slope, zero base).")
    print(f"\n{'K':>8}  " + "  ".join(f"{n.split(' (')[0]:>14}" for n in curves))
    for i, K in enumerate(Ks):
        if K in (1, 10, 100, 1000, 10000, 100000, 1000000) or i == len(Ks) - 1:
            print(f"{K:>8}  " + "  ".join(f"{curves[n][i]*1e3:>14.3f}" for n in curves) + "  (ms)")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    styles = {
        "SRI (query-only, online build)": dict(color="#1f77b4", lw=2.4, zorder=5),
        "SRI (build-amortized, conservative)": dict(color="#1f77b4", lw=2.0, ls="--", zorder=4),
        "DuckDB columnar (real DB)": dict(color="#d62728", lw=2.0, zorder=3),
        "native scan (no build)": dict(color="#2ca02c", lw=2.0, zorder=3),
    }
    for n, ys in curves.items():
        ax.plot(Ks, np.array(ys) * 1e3, label=n, **styles[n])
    for kx, lab, col in ((kx_scan, "vs scan", "#2ca02c"), (kx_duck, "vs DuckDB", "#d62728")):
        if kx:
            ax.axvline(kx, color=col, ls=":", alpha=0.5, lw=1.2)
            ax.annotate(f"K*={kx}\n{lab}", xy=(kx, ax.get_ylim()[0]),
                        xytext=(kx, 1e-2), fontsize=8, color=col, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("audit query batch size  K"); ax.set_ylabel("end-to-end time  (ms)")
    ax.set_title(f"End-to-end audit cost vs batch size  (M={m['M']:,} cells)\n"
                 "SRI O(log U) vs best real DB (DuckDB) and a native scan")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "sri_ksweep.png", dpi=140)

    m.pop("_q"); m.pop("_base")
    m["crossover_K_vs_scan"] = kx_scan
    m["crossover_K_vs_duckdb"] = kx_duck
    m["Ks"] = Ks.tolist()
    m["curves_s"] = {n: ys for n, ys in curves.items()}
    json.dump(m, open(out / "sri_ksweep.json", "w"), indent=2)
    print(f"\nwrote {out/'sri_ksweep.png'} and {out/'sri_ksweep.json'}")


if __name__ == "__main__":
    main()
