"""Tamper test: prove the sealed ledger detects post-hoc tuple edits.

``verify_chunk_chain(deep=False)`` checks only the prev_hash links in the
``chunks`` table; ``deep=True`` also re-hashes the ``tuples`` rows against the
chain. This test demonstrates:

  1. an untampered ledger verifies (deep=True).
  2. after editing ONE ``tuples.t_score`` row, the link-only check (deep=False)
     still passes.
  3. the deep check (deep=True) DETECTS the edit, because the tuples are
     bound to the hash chain.

    python -m experiment_scripts.dtrust_experiments.c1_audit.tamper_test
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtrust.store import TrustStore  # noqa: E402

from .policies import policy_pair, load_base_config
from .synth_event import make_delayed_compromise_event
from .sprt_replay import replay
from .seal import seal_event


def main():
    theta_v1, _v2 = policy_pair(load_base_config(), h_v1=15.0, decay=0.95)
    ev, mal = make_delayed_compromise_event(
        theta_v1, n_clients=40, malicious_frac=0.2, r_total=80, r_atk=20,
        r_inc=60, seed=0, profile="ramp")
    d = replay(ev, theta_v1)
    with tempfile.TemporaryDirectory() as td:
        sealed = seal_event(ev, d, theta_v1, td, 0, 20, 60, mal, anchor_interval=50)
        db = str(sealed.ledger_path)

        s = TrustStore(db)
        ok_clean_deep = s.verify_chunk_chain(deep=True)
        ok_clean_shallow = s.verify_chunk_chain(deep=False)
        n_anchors = s.stats()["anchors"]
        s.close()

        # Tamper: flip one tuple's sealed trust score directly in SQLite.
        con = sqlite3.connect(db)
        row = con.execute("SELECT participant, round, t_score FROM tuples "
                          "ORDER BY round LIMIT 1").fetchone()
        before = (row[1], row[2])
        con.execute("UPDATE tuples SET t_score = (t_score + 1) % 65536 "
                    "WHERE participant=? AND round=?", (row[0], row[1]))
        con.commit(); con.close()

        s = TrustStore(db)
        ok_tampered_shallow = s.verify_chunk_chain(deep=False)
        ok_tampered_deep = s.verify_chunk_chain(deep=True)
        s.close()

    print("\n==== LEDGER TAMPER TEST ====")
    print(f"  clean  deep-verify           : {ok_clean_deep}      (expect True)")
    print(f"  clean  shallow-verify        : {ok_clean_shallow}      (expect True)")
    print(f"  Merkle anchors in ledger     : {n_anchors}          (expect >=1)")
    print(f"  tampered one t_score at round {before[0]}")
    print(f"  tampered SHALLOW (old check)  : {ok_tampered_shallow}      <-- vulnerability: edit slips through")
    print(f"  tampered DEEP   (fixed check) : {ok_tampered_deep}      <-- FIX: tamper detected")
    passed = (ok_clean_deep and ok_clean_shallow and n_anchors >= 1
              and ok_tampered_shallow and not ok_tampered_deep)
    print(f"  ==> TEST {'PASS' if passed else 'FAIL'}: deep verify binds tuples to the hash chain")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
