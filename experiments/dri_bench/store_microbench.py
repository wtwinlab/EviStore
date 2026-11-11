"""Scaling microbenchmark for the sealed TrustStore query operators.

Measures what the indexed sealed store gives over a plain audit table by measuring, as the ledger grows (rounds R, fixed N
participants):
  - the store query operators (range_below, differential_audit Q6, safe_checkpoint
    Q9, damage_report Q11, dri_replay) on the INDEXED TrustStore, vs
  - the same queries on a PLAIN audit table with NO secondary indexes
    (full-scan baseline),
  - and the deep integrity verification cost (verify_chunk_chain, which rebuilds
    every chunk from the tuples).

Pure stdlib SQLite + the sealed store; runs locally on CPU in seconds.

    python -m experiment_scripts.dtrust_experiments.c1_audit.store_microbench \
        --rounds 200 1000 5000 20000 --n 100 --out results/recovery/c1_microbench
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from .seal import participant_hash
from .policies import policy_pair, load_base_config

import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dtrust.store import TrustStore, TrustTuple, Mode  # noqa: E402
from dtrust.store.schema import quantize_unit  # noqa: E402


def _rng_tuples(R, N, seed=0):
    rng = np.random.default_rng(seed)
    phash = {i: participant_hash(i) for i in range(N)}
    for r in range(R):
        t = rng.random(N)
        for i in range(N):
            yield TrustTuple(
                round=r, participant=phash[i],
                a_conf=quantize_unit(float(t[i])), fresh=True,
                s_cusum=float(rng.random() * 8.0),
                t_score=quantize_unit(float(t[i])), b_score=quantize_unit(float(t[i])),
                w_agg=quantize_unit(float(t[i])), mode=Mode.NORMAL,
                h_grad=bytes(32))


def _build_indexed(path, R, N):
    if path.exists():
        path.unlink()
    store = TrustStore(path, anchor_interval=256)
    by_round = {}
    for t in _rng_tuples(R, N):
        by_round.setdefault(t.round, []).append(t)
    for r in range(R):
        store.append_round(r, by_round[r])
    return store


def _build_plain(path, R, N):
    """Same rows, single table, NO secondary indexes (audit-table baseline)."""
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE tuples (round INT, participant BLOB, t_score INT, "
                "s_cusum REAL, w_agg INT, mode INT)")
    rows = [(t.round, t.participant, t.t_score, t.s_cusum, t.w_agg, int(t.mode))
            for t in _rng_tuples(R, N)]
    con.executemany("INSERT INTO tuples VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    return con


def _time(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/recovery/c1_microbench")
    args = ap.parse_args()
    base = load_base_config(); v1, v2 = policy_pair(base)
    tau1 = quantize_unit(v1["quarantine"]["tau_min"]); tau2 = quantize_unit(v2["quarantine"]["tau_min"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ph0 = participant_hash(0)

    rows = []
    for R in args.rounds:
        idx_path = out / f"idx_R{R}.db"
        store = _build_indexed(idx_path, R, args.n)
        plain = _build_plain(out / f"plain_R{R}.db", R, args.n)

        # Indexed store operators.
        q6 = _time(lambda: store.differential_audit(0, R - 1, tau1, tau2))
        q9 = _time(lambda: store.safe_checkpoint(R - 1, 20, tau2, 50, 0.05))
        q11 = _time(lambda: store.damage_report(0, R - 1, tau2))
        dri = _time(lambda: store.dri_replay(ph0, 0, R - 1, 4.0))
        verify = _time(lambda: store.verify_chunk_chain(deep=True), reps=2)
        # Plain audit-table equivalents (full scan, no index).
        p_q6 = _time(lambda: plain.execute(
            "SELECT round,participant FROM tuples WHERE round BETWEEN 0 AND ? "
            "AND ((t_score>=? AND t_score<?) OR (t_score>=? AND t_score<?))",
            (R - 1, tau1, tau2, tau2, tau1)).fetchall())
        p_dri = _time(lambda: plain.execute(
            "SELECT round FROM tuples WHERE participant=? AND round BETWEEN 0 AND ? "
            "AND s_cusum>=? ORDER BY round LIMIT 1", (ph0, R - 1, 4.0)).fetchone())
        plain.close(); store.close()
        size_mb = idx_path.stat().st_size / 1e6
        rows.append({"R": R, "tuples": R * args.n, "db_MB": round(size_mb, 2),
                     "Q6_idx_ms": round(q6, 2), "Q6_plain_ms": round(p_q6, 2),
                     "Q9_idx_ms": round(q9, 2), "Q11_idx_ms": round(q11, 2),
                     "DRI_idx_ms": round(dri, 3), "DRI_plain_ms": round(p_dri, 3),
                     "deep_verify_ms": round(verify, 1)})
        for f in (idx_path, out / f"plain_R{R}.db"):
            f.unlink(missing_ok=True)

    json.dump(rows, open(out / "microbench.json", "w"), indent=2)
    print(f"\n{'R':>7} {'tuples':>8} {'db_MB':>6} {'Q6idx':>7} {'Q6plain':>8} "
          f"{'Q9idx':>7} {'Q11idx':>7} {'DRIidx':>7} {'DRIplain':>8} {'deepVerify':>10}")
    for r in rows:
        print(f"{r['R']:>7} {r['tuples']:>8} {r['db_MB']:>6} {r['Q6_idx_ms']:>7} "
              f"{r['Q6_plain_ms']:>8} {r['Q9_idx_ms']:>7} {r['Q11_idx_ms']:>7} "
              f"{r['DRI_idx_ms']:>7} {r['DRI_plain_ms']:>8} {r['deep_verify_ms']:>10}")
    print(f"\nwrote {out/'microbench.json'}")


if __name__ == "__main__":
    main()
