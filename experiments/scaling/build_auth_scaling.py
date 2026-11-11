#!/usr/bin/env python3
"""Build results/auth_scaling.json for the authenticated-scaling figure.

Collates authenticated-SRI vs unauthenticated index-class (Fenwick / persistent
segment tree) count-query latencies at increasing N.  All values come from
existing measured JSON:

  authenticated : results/artifact/L1_sri_receipts.json   (M in {1e4,1e5,1e6}; gen_ms + verify_ms)
  index-class   : results/recovery/c1_sri/sri_scale.json          (M in {1e4..2e6}; sri_us + scan_us)
  index-class^  : results/recovery/c1_sri/sri_native_bench.json   (larger scale, if present)

The authenticated variant is measured up to 10^6 cells; the >10^6 points are
unauthenticated index-class measurements. In the figure the authenticated
line ends at 10^6, and the unauthenticated line carries the trend to 10^8.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def find_L1() -> dict:
    """Locate the authenticated-SRI receipt JSON."""
    for p in ROOT.rglob("L1_sri_receipts.json"):
        return json.loads(p.read_text())
    raise FileNotFoundError("L1_sri_receipts.json not found")


def load_sri_scale() -> list:
    p = ROOT / "results" / "recovery" / "c1_sri" / "sri_scale.json"
    return json.loads(p.read_text())["scale"]


def load_native_bench() -> dict | None:
    p = ROOT / "results" / "recovery" / "c1_sri" / "sri_native_bench.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build() -> dict:
    L1 = find_L1()
    scale = load_sri_scale()

    # Authenticated points -- exact measurements from authenticated-SRI receipts.
    auth_N = []
    auth_lat_us = []
    auth_gen_ms = []
    auth_verify_ms = []
    auth_proof_b = []
    for s in L1["scales"]:
        auth_N.append(s["cells"])
        # p50 auth latency ~ gen + verify (both microsecond scale)
        auth_lat_us.append(round((s["gen_ms"] + s["verify_ms"]) * 1000.0, 2))
        auth_gen_ms.append(s["gen_ms"])
        auth_verify_ms.append(s["verify_ms"])
        auth_proof_b.append(s["proof_bytes"])

    # Unauthenticated index-class points -- SRI/Pers. ST count query
    # latency at each scale.
    index_N = []
    index_lat_us = []
    scan_lat_us = []
    for s in scale:
        index_N.append(s["M"])
        index_lat_us.append(s["sri_us"])
        scan_lat_us.append(s["scan_us"])

    nb = load_native_bench()
    native_extras = []
    if isinstance(nb, dict):
        # nb typically has M=2e6 native bench with p50 count query around 0.44 us.
        for key, val in nb.items():
            if isinstance(val, dict) and "count_query_us_p50" in val:
                native_extras.append({
                    "M": val.get("M", 2_000_000),
                    "sri_us": val["count_query_us_p50"],
                    "scan_us": val.get("scan_query_us_p50", None),
                    "source": key,
                })
    elif isinstance(nb, list):
        for entry in nb:
            if isinstance(entry, dict) and "count_query_us_p50" in entry:
                native_extras.append(entry)

    return {
        "_note": (
            "E-D (nice-to-have): authenticated SRI vs unauthenticated "
            "index-class serving latency across N. Authenticated points are "
            "from the L1_sri_receipts.json measurements (M in {1e4,1e5,1e6}); "
            "the authenticated prototype's per-node hashing bottlenecks around "
            "10^6 cells and is not measured beyond, matching the paper's "
            "scope-tightening prose in sec:eval-serving. Unauthenticated "
            "index-class points are from results/recovery/c1_sri/sri_scale.json, "
            "with per-node hashing removed."
        ),
        "N": auth_N + [n for n in index_N if n > max(auth_N)],
        "auth_lat_us": auth_lat_us + [None] * len([n for n in index_N
                                                     if n > max(auth_N)]),
        "index_lat_us": [
            next((s["sri_us"] for s in scale if s["M"] == n), None)
            for n in auth_N
        ] + [next((s["sri_us"] for s in scale if s["M"] == n), None)
              for n in index_N if n > max(auth_N)],
        "scan_lat_us": [
            next((s["scan_us"] for s in scale if s["M"] == n), None)
            for n in auth_N
        ] + [next((s["scan_us"] for s in scale if s["M"] == n), None)
              for n in index_N if n > max(auth_N)],
        "authenticated_max_measured_N": max(auth_N),
        "auth_proof_b": auth_proof_b,
        "auth_verify_ms": auth_verify_ms,
        "auth_gen_ms": auth_gen_ms,
        "authenticated_points": [
            {"M": n, "auth_lat_us": l, "proof_b": p,
             "gen_ms": g, "verify_ms": v}
            for n, l, p, g, v in zip(
                auth_N, auth_lat_us, auth_proof_b, auth_gen_ms, auth_verify_ms)
        ],
        "index_points": [
            {"M": s["M"], "sri_us": s["sri_us"], "scan_us": s["scan_us"],
             "build_s": s.get("build_s")} for s in scale
        ],
        "native_extras": native_extras,
    }


def main():
    out = build()
    od = ROOT / "results/artifact"
    od.mkdir(parents=True, exist_ok=True)
    (od / "auth_scaling.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "auth_scaling.json")


if __name__ == "__main__":
    main()
