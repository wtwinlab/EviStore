#!/usr/bin/env python3
"""SRI measured ON-STORE.

For Q5/Q6/Q11 (threshold count/weight over a round window) we measure, on the
SAME TrustStore over the SAME sealed rows:
  - pre-SRI  : the SQLite path on store.conn (B-tree on (t_score,round)) -- the
               24-25ms-class scan the paper reports as the pre-index baseline;
  - with-SRI : store.count_below / store.weight_below (the O(log U) SRI path).
Plus append cost (ms/round incl. SRI maintenance), reopen-rebuild time (SRI
replayed from the sealed tuples), and SRI memory. Exactness is asserted: SRI ==
SQL on every query.

Run:  PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/e1_sri_onstore.py
"""
import json, os, sys, time, types, tempfile, random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
# torch-free import of the store (dtrust/__init__ pulls torch otherwise)
if "dtrust" not in sys.modules:
    pkg = types.ModuleType("dtrust"); pkg.__path__ = [str(_ROOT / "dtrust")]
    sys.modules["dtrust"] = pkg
from dtrust.store.store import TrustStore
from dtrust.store.schema import Mode, TrustTuple

rng = random.Random(0)
H32 = b"\x00" * 32


def mk_round(rnd, N):
    out = []
    for pid in range(N):
        ts = rng.randint(0, 65535)
        out.append(TrustTuple(round=rnd, participant=pid.to_bytes(16, "little"),
                              a_conf=40000, fresh=True, s_cusum=0.0,
                              t_score=ts, b_score=ts, w_agg=rng.randint(0, 65535),
                              mode=Mode.NORMAL, h_grad=H32))
    return out


def sql_threshold(conn, r1, r2, tau):
    """pre-SRI baseline: count + weight of cells with t_score < tau in [r1,r2]."""
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(w_agg),0) FROM tuples "
        "WHERE round BETWEEN ? AND ? AND t_score < ?", (r1, r2, tau)).fetchone()
    return int(row[0]), int(row[1])


def bench_scale(R, N, n_queries=200):
    db = tempfile.mktemp(suffix=".db")
    st = TrustStore(db, anchor_interval=256)
    # ---- append (write path incl. SRI maintenance) ----
    t0 = time.perf_counter()
    for r in range(R):
        st.append_round(r, mk_round(r, N))
    append_s = time.perf_counter() - t0
    M = R * N

    # ---- audit-window queries: pre-SRI scan vs SRI, exactness ----
    scan_us, sri_us, mism = [], [], 0
    for _ in range(n_queries):
        w = rng.randint(20, 200)          # audit window: tens-hundreds of rounds
        r1 = rng.randint(0, max(0, R - w - 1)); r2 = r1 + w
        tau = rng.randint(1, 65535)
        t = time.perf_counter(); c_sql, w_sql = sql_threshold(st.conn, r1, r2, tau)
        scan_us.append((time.perf_counter() - t) * 1e6)
        t = time.perf_counter(); c_sri = st.count_below(r1, r2, tau); w_sri = st.weight_below(r1, r2, tau)
        sri_us.append((time.perf_counter() - t) * 1e6)
        if c_sql != c_sri or w_sql != w_sri:
            mism += 1
    st.conn.close()

    # ---- reopen-rebuild: SRI replayed from sealed tuples ----
    t = time.perf_counter(); st2 = TrustStore(db, anchor_interval=256)
    reopen_s = time.perf_counter() - t
    ok = st2.verify_sri()
    # ---- SRI memory proxy ----
    try:
        nodes = st2.sri.tree.n_nodes()
    except Exception:
        nodes = None
    st2.conn.close(); os.unlink(db)

    def med(x): return sorted(x)[len(x) // 2]
    return {
        "R": R, "N": N, "M": M,
        "append_ms_per_round": round(append_s / R * 1e3, 4),
        "scan_p50_us": round(med(scan_us), 2), "sri_p50_us": round(med(sri_us), 2),
        "speedup_p50": round(med(scan_us) / med(sri_us), 1),
        "exact": mism == 0, "mismatches": mism,
        "reopen_rebuild_s": round(reopen_s, 3), "verify_sri": ok,
        "sri_nodes": nodes,
    }


def main():
    scales = [(1000, 100), (10000, 100)]
    if "--big" in sys.argv:
        scales.append((100000, 100))   # 1e7 cells (slow append)
    res = [bench_scale(R, N) for R, N in scales]
    out = {"_note": "E1: on-store SRI vs pre-SRI SQL scan on the SAME TrustStore; "
                    "exact, same rows. pre-SRI = SQLite B-tree path; with-SRI = O(log U).",
           "scales": res}
    print(json.dumps(out, indent=2))
    od = _ROOT / "results/recovery/c1_sri"; od.mkdir(parents=True, exist_ok=True)
    (od / "e1_onstore.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "e1_onstore.json")


if __name__ == "__main__":
    main()
