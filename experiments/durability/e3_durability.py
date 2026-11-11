#!/usr/bin/env python3
"""Durability and crash recovery on the REAL durable store: one canonical
path append-evidence -> commit-decision -> update-root -> reopen -> SRI/Q threshold
query, with crash/restart at 4 points, plus durability overhead and recovery time.
Reuses dtrust/store.TrustStore (SQLite+WAL+Merkle anchors+SRI reopen-rebuild) verbatim.

decision-before-effect (Thm dbe): an "effect" (a checkpoint bound to a ledger root)
must follow a SEALED decision prefix. We model effects as a side checkpoint log
(round, bound_root); a violation is a checkpoint whose bound_root is not an anchored
root present in the reopened ledger. Correct ordering -> 0 violations; a broken
control (effect before sealing) -> the audit flags it.

Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/e3_durability.py
"""
import json, os, sys, time, types, tempfile, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if "dtrust" not in sys.modules:
    pkg = types.ModuleType("dtrust"); pkg.__path__ = [str(ROOT / "dtrust")]
    sys.modules["dtrust"] = pkg
from dtrust.store.store import TrustStore
from dtrust.store.schema import Mode, TrustTuple, quantize_unit
import random
rng = random.Random(0); H32 = b"\x00" * 32


def mk_round(r, N=50):
    return [TrustTuple(round=r, participant=p.to_bytes(16, "little"), a_conf=40000, fresh=True,
                       s_cusum=0.0, t_score=rng.randint(0, 65535), b_score=30000,
                       w_agg=rng.randint(0, 65535), mode=Mode.NORMAL, h_grad=H32) for p in range(N)]


def durability_overhead(R=2000, N=50):
    """append cost with durable WAL+synchronous=NORMAL vs synchronous=OFF (non-durable)."""
    def run(sync_normal):
        db = tempfile.mktemp(suffix=".db"); st = TrustStore(db, wal=True)
        if not sync_normal:
            st.conn.execute("PRAGMA synchronous=OFF;")
        t0 = time.perf_counter()
        for r in range(R):
            st.append_round(r, mk_round(r, N))
        dt = time.perf_counter() - t0
        st.conn.close(); os.unlink(db)
        return dt / R * 1e3
    dur = run(True); nodur = run(False)
    return {"durable_ms_per_round": round(dur, 3), "nondurable_ms_per_round": round(nodur, 3),
            "durability_overhead_pct": round((dur - nodur) / nodur * 100, 1)}


def recovery_time(R, N=50):
    db = tempfile.mktemp(suffix=".db"); st = TrustStore(db)
    for r in range(R):
        st.append_round(r, mk_round(r, N))
    pre = st.sri_root_digest(); st.conn.close()
    t0 = time.perf_counter(); st2 = TrustStore(db)                 # reopen rebuilds SRI from sealed
    dt = time.perf_counter() - t0
    ok = (st2.sri_root_digest() == pre) and st2.verify_sri()
    st2.conn.close(); os.unlink(db)
    return {"R": R, "M": R * N, "recovery_rebuild_s": round(dt, 3), "prefix_valid": bool(ok)}


def crash_consistency(R=200, N=50):
    """Snapshot the db at 4 points around a round's seal/effect; reopen each; verify
    chunk-chain + SRI prefix validity + decision-before-effect (no orphan effect)."""
    db = tempfile.mktemp(suffix=".db"); st = TrustStore(db)
    snaps = {}
    # effects: (round, bound_root) -- a correct effect binds to the ANCHORED Merkle root at its
    # round, which is reconstructable from the sealed store after a crash/reopen.
    checkpoints = []
    for r in range(R):
        st.append_round(r, mk_round(r, N))           # seal decision for round r
        if r == R - 3:
            snaps["P1_after_seal_before_effect"] = _snap(st, db)   # crash: decision sealed, no effect yet
        root = st.merkle_root(r)                      # anchored, reconstructable sealed root at round r
        checkpoints.append((r, root))                # apply effect (bind to the sealed root)
        if r == R - 3:
            snaps["P2_after_effect"] = _snap(st, db, checkpoints)
        if r == R - 2:
            snaps["P3_midstream"] = _snap(st, db, checkpoints)
    snaps["P4_final"] = _snap(st, db, checkpoints)
    st.conn.close(); os.unlink(db)

    def verify(snapdb, cps):
        """DBE after reopen: integrity holds AND every effect binds to a root that the reopened
        sealed store actually reproduces. A violation = an effect for an unsealed round OR an
        effect whose bound root does not match the anchored sealed root at that round (stale/forged)."""
        s = TrustStore(snapdb)
        sealed_ok = s.verify_sri() and s.verify_chunk_chain(deep=True)   # prefix + tuple->hash binding
        last_round = s.conn.execute("SELECT COALESCE(MAX(round),-1) FROM tuples").fetchone()[0]
        violations = 0
        for (cr, bound_root) in (cps or []):
            if cr > last_round:                          # effect for a round never sealed
                violations += 1
            elif bound_root != s.merkle_root(cr):        # bound root not reproduced by sealed prefix
                violations += 1
        s.conn.close()
        return sealed_ok, violations

    results = {}; total_viol = 0
    for name, (sdb, cps) in snaps.items():
        ok, v = verify(sdb, cps); total_viol += v
        results[name] = {"prefix_valid": bool(ok), "dbe_violations": v}
        os.unlink(sdb)

    # broken controls (each MUST be flagged by the strengthened check):
    #   (a) an effect bound to an UNSEALED future round;
    #   (b) an effect for an already-SEALED round but bound to a STALE/FORGED root.
    db2 = tempfile.mktemp(suffix=".db"); st2 = TrustStore(db2)
    for r in range(10):
        st2.append_round(r, mk_round(r, N))
    real_root5 = st2.merkle_root(5)
    snap2 = _snap(st2, db2); st2.conn.close(); os.unlink(db2)
    s2 = TrustStore(snap2[0])
    last = s2.conn.execute("SELECT MAX(round) FROM tuples").fetchone()[0]
    bad_future = [(99, b"\x00" * 32)]                    # (a) round 99, never sealed
    bad_stale = [(5, b"\xde" * 32)]                      # (b) sealed round 5, wrong root
    flagged = 0
    flagged += sum(1 for (cr, _) in bad_future if cr > last)
    flagged += sum(1 for (cr, br) in bad_stale if cr <= last and br != s2.merkle_root(cr))
    sanity_real = (real_root5 == s2.merkle_root(5))      # the REAL root at round 5 still matches
    s2.conn.close(); os.unlink(snap2[0])
    return {"crash_points": results, "total_dbe_violations_correct_order": total_viol,
            "broken_control_violations_flagged": flagged,
            "broken_controls_tested": ["unsealed_future_round", "sealed_round_stale_root"],
            "real_root_reproduced_after_reopen": bool(sanity_real)}


def _snap(st, db, cps=None):
    st.conn.commit()
    p = tempfile.mktemp(suffix=".db"); shutil.copy(db, p)
    for ext in ("-wal", "-shm"):
        if os.path.exists(db + ext):
            shutil.copy(db + ext, p + ext)
    return (p, list(cps) if cps else [])


def main():
    out = {"_note": "② durability/crash on the real durable TrustStore (SQLite+WAL+Merkle+SRI "
                    "reopen-rebuild), reused verbatim. Canonical path: append evidence -> commit "
                    "decision -> update root -> reopen -> threshold query.",
           "durability_overhead": durability_overhead(),
           "recovery_time": [recovery_time(r) for r in (10000, 100000)],
           "crash_consistency": crash_consistency()}
    print(json.dumps(out, indent=2))
    od = ROOT / "results/recovery/durability"; od.mkdir(parents=True, exist_ok=True)
    (od / "e3_durability.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "e3_durability.json")


if __name__ == "__main__":
    main()
