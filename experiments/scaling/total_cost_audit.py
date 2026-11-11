#!/usr/bin/env python3
"""Total-cost-of-audit crossover curves.

Reads the three measured portfolio JSONs (same host, same generator, same
scales — see baselines/ccf_q1/bench_{ext,immudb,dtrust}.py) and derives, per
query class and horizon R, the total cost per committed cell as a function of
audit intensity:

    y_s(a) = w_s + a * q_s(R)        [ms per cell]

where w_s = measured per-cell write cost of system s (seal+index for DTRUST,
putIdx for CCF-Idx, plain put for CCF-Ext, batched INSERT for immudb) and
q_s(R) = measured per-query latency normalized per cell is NOT used — q_s is
the whole-query latency and a = audit queries per committed cell, so a*q_s is
the per-cell share of query cost:

    y_s(a) = w_s + a * Q_s(R)        with Q_s the full-query p50 at horizon R.

The crossover a* between DTRUST and baseline s solves
    w_D + a Q_D = w_s + a Q_s  =>  a* = (w_D - w_s) / (Q_s - Q_D).
DTRUST pays more per write (sealing + authenticated index maintenance) and
less per query; a* is the audit intensity above which DTRUST is cheaper in
total. Storage bytes/cell is reported alongside (not summed into y — time and
bytes are different units; the paper reports both).

Usage:
    python total_cost_audit.py \
        --ccf     results/baseline_portfolio/ccf_ext_idx.json \
        --immudb  results/baseline_portfolio/immudb.json \
        --dtrust  results/baseline_portfolio/dtrust_native.json \
        --out-json results/total_cost/total_cost_curves.json \
        --out-fig  results/total_cost/fig_cost_crossover.pdf
"""

from __future__ import annotations

import argparse
import json
import os

QUERY_CLASSES = ["rangeBelow", "counterfactual", "differentialAudit",
                 "safeCheckpoint"]

# fixed categorical order (dataviz palette slots 1-4); color follows the
# system identity in every panel
SYSTEMS = ["DTRUST", "CCF-Ext", "CCF-Idx", "immudb"]
COLORS = {"DTRUST": "#2a78d6", "CCF-Ext": "#1baf7a",
          "CCF-Idx": "#eda100", "immudb": "#008300"}


def p50(d):
    return d["p50"] if isinstance(d, dict) else None


# ---------------------------------------------------------------- adapters
# One adapter per source JSON: -> {R: {"write_per_cell_ms": w,
#   "queries": {class: Q_ms}, "storage_bytes_per_cell": b}}

def adapt_ccf(doc, variant):
    """Both CCF variants share one deployment loaded via putRound (per-round
    batch, index maintained) — the same write unit as DTRUST append_round and
    immudb per-round INSERT. This bundles roundcells maintenance into
    CCF-Ext's write cost too; the per-cell put/putIdx probe numbers in the
    JSON quantify that overhead for the figure's disclosure note."""
    out = {}
    qmap = {
        "rangeBelow": f"{variant}_rangeBelow",
        "counterfactual": "ext_counterfactual",  # Idx has no counterfactual
        "differentialAudit": f"{variant}_differentialAudit",
        "safeCheckpoint": f"{variant}_safeCheckpoint",
        "pointInTime": "pointInTime",
    }
    for key, m in doc["scales"].items():
        if not key.isdigit():
            continue
        R = int(key)
        w = None
        if m.get("putRound_write_ms"):
            w = p50(m["putRound_write_ms"]["per_cell_ms"])
        out[R] = {
            "write_per_cell_ms": w,
            "queries": {c: p50(m["queries"][qmap[c]])
                        for c in QUERY_CLASSES + ["pointInTime"]
                        if qmap[c] in m["queries"]},
            "storage_bytes_per_cell": (m["ledger_bytes"] / m["cells_total"]
                                       if m.get("ledger_bytes") else None),
        }
    return out


def adapt_immudb(doc):
    """immudb latencies are PLAIN SQL timings (comparable to CCF variants,
    which likewise have no per-result verification); the verified-range
    number (SQL + per-row VerifiedGet) stays in immudb.json as secondary and
    the figure/docs label the immudb row as plain-SQL timing."""
    out = {}
    qmap = {"rangeBelow": "rangeBelow", "counterfactual": "counterfactual",
            "differentialAudit": "differentialAudit",
            "safeCheckpoint": "safeCheckpoint", "pointInTime": "pointInTime_sql"}
    for key, m in doc["scales"].items():
        if not key.isdigit():
            continue
        R = int(key)
        w = None
        if m.get("write_ms"):
            w = p50(m["write_ms"]["sql_insert_per_cell_ms"])
        out[R] = {
            "write_per_cell_ms": w,
            "queries": {c: p50(m["queries"][qmap[c]])
                        for c in QUERY_CLASSES + ["pointInTime"]
                        if qmap[c] in m["queries"]},
            "storage_bytes_per_cell": (m["data_dir_bytes"] / m["cells_total"]
                                       if m.get("data_dir_bytes") else None),
        }
    return out


def adapt_dtrust(doc):
    """dtrust_native.json (bench_dtrust.py schema). rangeBelow uses the
    ENUMERATION path (rangeBelow_enum) — every system in the figure returns
    the qualifying cells, not just a count; the SRI count-only latency stays
    in the JSON as a secondary number."""
    out = {}
    qmap = {"rangeBelow": "rangeBelow_enum", "counterfactual": "counterfactual",
            "differentialAudit": "differentialAudit",
            "safeCheckpoint": "safeCheckpoint", "pointInTime": "pointInTime"}
    for key, m in doc["scales"].items():
        if not key.isdigit():
            continue
        R = int(key)
        w = None
        if m.get("append_write_ms"):
            w = p50(m["append_write_ms"]["per_cell_ms"])
        out[R] = {
            "write_per_cell_ms": w,
            "queries": {c: p50(m["queries"][qmap[c]])
                        for c in QUERY_CLASSES + ["pointInTime"]
                        if qmap[c] in m["queries"]},
            "storage_bytes_per_cell": (m.get("store_bytes") / m["cells_total"]
                                       if m.get("store_bytes") else None),
        }
    return out


# ---------------------------------------------------------------- curves
def crossover(wd, qd, ws, qs):
    """Audit intensity a* where DTRUST total cost overtakes system s."""
    if None in (wd, qd, ws, qs) or qs <= qd:
        return None if None in (wd, qd, ws, qs) else 0.0 if wd <= ws else None
    a = (wd - ws) / (qs - qd)
    return max(a, 0.0)


def build(systems_data):
    horizons = sorted(set.intersection(
        *[set(d.keys()) for d in systems_data.values()]))
    doc = {"horizons": horizons, "systems": {}, "crossovers": {}}
    for s, d in systems_data.items():
        doc["systems"][s] = {str(R): d[R] for R in horizons}
    for R in horizons:
        dd = systems_data["DTRUST"][R]
        for c in QUERY_CLASSES:
            for s in SYSTEMS[1:]:
                sd = systems_data[s][R]
                a = crossover(dd["write_per_cell_ms"], dd["queries"].get(c),
                              sd["write_per_cell_ms"], sd["queries"].get(c))
                doc["crossovers"][f"R={R}|{c}|DTRUST-vs-{s}"] = a
    return doc


# ---------------------------------------------------------------- figure
def make_fig(systems_data, horizons, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a = np.logspace(-5, 1, 200)  # audit queries per committed cell
    R = horizons[-1]             # headline horizon; JSON has the rest
    # paper form: no in-figure titles/disclosures (the caption carries them),
    # legend instead of direct labels (curve pairs overlap in log space)
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.2), sharex=True)
    for ax, c in zip(axes.flat, QUERY_CLASSES):
        for s in SYSTEMS:
            d = systems_data[s].get(R)
            if not d or d["queries"].get(c) is None or \
               d["write_per_cell_ms"] is None:
                continue
            y = d["write_per_cell_ms"] + a * d["queries"][c]
            ax.plot(a, y, color=COLORS[s], linewidth=1.8, label=s)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(c, fontsize=9, color="#333333")
        ax.grid(True, which="major", color="#dddddd", linewidth=0.5)
        ax.tick_params(labelsize=7, colors="#555555")
        for sp in ax.spines.values():
            sp.set_color("#bbbbbb")
    for ax in axes[1]:
        ax.set_xlabel("audit queries per sealed cell", fontsize=8.5,
                      color="#333333")
    for ax in axes[:, 0]:
        ax.set_ylabel("total ms per cell", fontsize=8.5, color="#333333")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"[fig] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccf", required=True)
    ap.add_argument("--immudb", required=True)
    ap.add_argument("--dtrust", required=True)
    ap.add_argument("--dtrust-counts", default=None,
                    help="followup_gate2 dtrust_counts.json: overlays the SRI "
                         "count-path latencies onto DTRUST's differentialAudit"
                         "/counterfactual so those panels compare count-vs-"
                         "count (baselines return counts; the enumeration "
                         "path materializes the flip set and is reported in "
                         "text, not here)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", default=None)
    args = ap.parse_args()

    with open(args.ccf) as f:
        ccf = json.load(f)
    with open(args.immudb) as f:
        imm = json.load(f)
    with open(args.dtrust) as f:
        dt = json.load(f)

    systems_data = {
        "DTRUST": adapt_dtrust(dt),
        "CCF-Ext": adapt_ccf(ccf, "ext"),
        "CCF-Idx": adapt_ccf(ccf, "idx"),
        "immudb": adapt_immudb(imm),
    }
    if args.dtrust_counts:
        with open(args.dtrust_counts) as f:
            dc = json.load(f)
        # count-path numbers were measured on the R=1e5 store only
        top = max(systems_data["DTRUST"].keys())
        systems_data["DTRUST"][top]["queries"]["differentialAudit"] = \
            dc["diff_count_sri"]["p50"]
        systems_data["DTRUST"][top]["queries"]["counterfactual"] = \
            dc["cf_count_sri"]["p50"]
        systems_data["DTRUST"][top]["count_path_overlay"] = {
            "source": args.dtrust_counts,
            "note": "differentialAudit/counterfactual are SRI count-path "
                    "p50s (count-vs-count with the baselines); enumeration "
                    "path reported separately in text"}
    doc = build(systems_data)
    doc["provenance"] = {
        "inputs": {"ccf": args.ccf, "immudb": args.immudb,
                   "dtrust": args.dtrust},
        "model": "y_s(a) = w_s + a*Q_s(R); a* = (w_D - w_s)/(Q_s - Q_D)",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[json] wrote {args.out_json}")
    for k, v in doc["crossovers"].items():
        if v is not None:
            print(f"  {k}: a* = {v:.4g}")
    if args.out_fig:
        make_fig(systems_data, doc["horizons"], args.out_fig)


if __name__ == "__main__":
    main()
