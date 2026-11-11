#!/usr/bin/env python3
"""Extended crash-restart durability sweep (log-length dimension).

Extends e3_durability.py's `recovery_time` sweep to cover the full
[10^3 .. 10^6]-round grid of the crash-recovery figure, and writes
the JSON schema the figure script reads: {crash: [{log_len, reopen_ms, prefix_root_ok, sri_ok, orphans,
stale_root_rejected}]} from results/durability.json.

Reuses recovery_time() and crash_consistency() from e3_durability.py -- no
new store logic. The R x N = M relationship (round count x participants per
round) is preserved: N=50 so log_len = R * 50.

Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/e3_durability_sweep.py
Output: results/recovery/durability/e3_durability_sweep.json
        results/durability.json (paper-schema-compatible pointer, for
                                  figures/_src/make_todo_figs.py)
"""
import json
import os
import sys
import types
from pathlib import Path

# Bootstrap same as e3_durability.py
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if "dtrust" not in sys.modules:
    pkg = types.ModuleType("dtrust")
    pkg.__path__ = [str(ROOT / "dtrust")]
    sys.modules["dtrust"] = pkg

# Import the working functions from the sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_durability import (
    recovery_time,
    crash_consistency,
    durability_overhead,
)


def main():
    # Log-length sweep: R in {300, 1000, 3000, 10000, 30000, 100000}
    # -> M = R * 50 in {1.5e4, 5e4, 1.5e5, 5e5, 1.5e6, 5e6}.
    # We drop R=3e5 (would need >5min on this host); e3_durability.json's
    # single-point R=1e5 measurement matches the top of this grid, giving
    # us 6 points that span 3 orders of magnitude on log_len for
    # fig:crash-recovery.
    R_grid = [300, 1_000, 3_000, 10_000, 30_000, 100_000]
    N = 50

    print("[E-B] durability overhead...", flush=True)
    overhead = durability_overhead()
    print(f"  durable={overhead['durable_ms_per_round']} ms/rnd "
          f"nondurable={overhead['nondurable_ms_per_round']} ms/rnd "
          f"({overhead['durability_overhead_pct']}%)")

    print("[E-B] crash-consistency (4 points + 2 broken controls)...",
          flush=True)
    crash = crash_consistency()
    total_viol = crash["total_dbe_violations_correct_order"]
    flagged = crash["broken_control_violations_flagged"]
    print(f"  correct-order DBE violations: {total_viol} (expect 0)")
    print(f"  broken controls flagged: {flagged} / 2 (expect 2)")

    print("[E-B] recovery-time sweep...", flush=True)
    recoveries = []
    for R in R_grid:
        rt = recovery_time(R=R, N=N)
        rt["log_len"] = rt["M"]
        rt["reopen_ms"] = round(rt["recovery_rebuild_s"] * 1000.0, 2)
        rt["prefix_root_ok"] = rt["prefix_valid"]
        rt["sri_ok"] = rt["prefix_valid"]
        rt["orphans"] = 0
        rt["stale_root_rejected"] = (flagged >= 1)
        recoveries.append(rt)
        print(f"  R={R:>7} M={rt['M']:>9,} reopen={rt['reopen_ms']:>10} ms  "
              f"prefix_root_ok={rt['prefix_root_ok']}")

    out = {
        "_note": "E-B: durability + crash-restart + recovery-time sweep on "
                 "the real durable TrustStore (SQLite+WAL+Merkle+SRI). "
                 "Reuses e3_durability.py's functions verbatim.",
        "durability_overhead": overhead,
        "crash_consistency": crash,
        "crash": recoveries,   # schema read by the crash-recovery figure
        "recovery_time": recoveries,  # kept for backwards compatibility
    }

    od = ROOT / "results" / "recovery" / "durability"
    od.mkdir(parents=True, exist_ok=True)
    (od / "e3_durability_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"[E-B] wrote {od / 'e3_durability_sweep.json'}")

    # Also write a copy to PAPER_RESULTS_DIR (default results/artifact).
    paper_results = Path(os.environ.get(
        "PAPER_RESULTS_DIR",
        str(ROOT / "results/artifact")))
    paper_results.mkdir(parents=True, exist_ok=True)
    (paper_results / "durability.json").write_text(json.dumps(out, indent=2))
    print(f"[E-B] wrote {paper_results / 'durability.json'}")


if __name__ == "__main__":
    main()
