"""The Drift-Replay-Index O(log R) benchmark.

The store's `dri_replay` can use an indexed SQL scan, which is O(R) per query.
This benchmark measures the dyadic DyadicDRI against it: the dyadic Lindley-Spitzer summary answers the path-dependent
earliest-CUSUM-crossing query in O(log R) per threshold (after an O(R) build),
versus O(R) per query for the linear scan — so for K replayed thresholds the
dyadic index is O(R + K log R) vs O(K R).

Also verifies CORRECTNESS: dyadic earliest_crossing == linear oracle on every
probe (the summary is a sufficient statistic, per the DRI Sufficiency theorem).

Pure-Python, no torch; runs locally in seconds.

    python -m experiment_scripts.dtrust_experiments.c1_audit.dri_microbench \
        --rounds 1000 10000 100000 1000000 --K 200 --out results/recovery/c1_dri_bench
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from dtrust.store.dri_dyadic import DyadicDRI


def _time(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--K", type=int, default=200, help="thresholds replayed (batch)")
    ap.add_argument("--out", default="results/recovery/c1_dri_bench")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for R in args.rounds:
        # Residuals of a reflected CUSUM walk: slightly negative mean + noise
        # (honest H0-ish drift) so crossings are non-trivial.
        resids = (rng.standard_normal(R) - 0.05).tolist()
        build = _time(lambda: DyadicDRI(resids), reps=2)
        dri = DyadicDRI(resids)
        thresholds = (rng.random(args.K) * 6.0 + 1.0).tolist()

        # Correctness vs linear oracle on a sample of thresholds.
        mism = 0
        for h in thresholds[:40]:
            if dri.earliest_crossing(h) != DyadicDRI.linear_earliest_crossing(resids, h):
                mism += 1

        # Single-query latency.
        h0 = float(thresholds[0])
        t_dri = _time(lambda: dri.earliest_crossing(h0), reps=20)
        t_lin = _time(lambda: DyadicDRI.linear_earliest_crossing(resids, h0), reps=5)
        # Batch of K thresholds.
        bt_dri = _time(lambda: dri.earliest_crossings_batch(thresholds), reps=3)
        bt_lin = _time(lambda: [DyadicDRI.linear_earliest_crossing(resids, h) for h in thresholds], reps=2)

        rows.append({
            "R": R, "build_ms": round(build * 1e3, 2),
            "dyadic_query_us": round(t_dri * 1e6, 2),
            "linear_query_us": round(t_lin * 1e6, 2),
            "speedup_single": round(t_lin / max(t_dri, 1e-12), 1),
            f"batch{args.K}_dyadic_ms": round((build + bt_dri) * 1e3, 2),
            f"batch{args.K}_linear_ms": round(bt_lin * 1e3, 2),
            "batch_speedup": round(bt_lin / max(build + bt_dri, 1e-12), 1),
            "correct_vs_oracle": (mism == 0),
            "dyadic_us_over_log2R": round(t_dri * 1e6 / max(math.log2(R), 1), 2),
        })

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "dri_bench.json", "w"), indent=2)
    print(f"\n{'R':>9} {'build_ms':>9} {'dyadic_us':>10} {'linear_us':>10} {'1x_spd':>7} "
          f"{'batchK_dy_ms':>12} {'batchK_lin_ms':>13} {'batch_spd':>9} {'us/log2R':>9} {'ok':>4}")
    for r in rows:
        print(f"{r['R']:>9} {r['build_ms']:>9} {r['dyadic_query_us']:>10} {r['linear_query_us']:>10} "
              f"{r['speedup_single']:>7} {r[f'batch{args.K}_dyadic_ms']:>12} "
              f"{r[f'batch{args.K}_linear_ms']:>13} {r['batch_speedup']:>9} "
              f"{r['dyadic_us_over_log2R']:>9} {str(r['correct_vs_oracle']):>4}")
    print(f"\nwrote {out/'dri_bench.json'}  (dyadic_us/log2R ~constant ⇒ O(log R) per query)")


if __name__ == "__main__":
    main()
