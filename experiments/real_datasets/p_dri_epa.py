#!/usr/bin/env python3
"""P-DRI on REAL EPA PM2.5 daily data: changed-prior drift replay.

Real workload: data_ext/daily_88101_2023/daily_88101_2023.csv (271 MB, 847K rows). Per monitoring station
we form the daily PM2.5 series (Arithmetic Mean = a_t, b_t = 1, increment x_t(mu) = a_t - mu). A drift
policy is a reflected CUSUM crossing threshold h with baseline mu. Policy EVOLUTION changes h AND/OR the
baseline mu. DRI (fixed mu0) can only vary h; P-DRI serves any (mu, h) from one sealed index.

We measure, against a brute-force oracle: exactness, and P-DRI query latency vs raw-scan vs rebuild-DRI-
per-mu; plus the hull-size distribution (PD2) and a proof-size proxy (PD3).
Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/db_eval/p_dri_epa.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import importlib.util as ilu
spec = ilu.spec_from_file_location("p_dri", Path(__file__).parent / "p_dri.py")
PD = ilu.module_from_spec(spec); spec.loader.exec_module(PD)

CSV = ROOT / "data_ext/daily_88101_2023/daily_88101_2023.csv"
MIN_DAYS = 200          # stations with at least this many daily records
MAX_STATIONS = 400      # headline station count


def load_stations():
    cols = ["State Code", "County Code", "Site Num", "POC", "Date Local", "Arithmetic Mean"]
    df = pd.read_csv(CSV, usecols=cols)
    df["station"] = (df["State Code"].astype(str) + "-" + df["County Code"].astype(str) + "-" +
                     df["Site Num"].astype(str) + "-" + df["POC"].astype(str))
    df = df.sort_values(["station", "Date Local"])
    series = {}
    for st, g in df.groupby("station"):
        v = g["Arithmetic Mean"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) >= MIN_DAYS:
            # quantize to integer tenths of ug/m3 -- DTRUST operates over a fixed quantized domain (like
            # SRI's U=2^16), so reflected-CUSUM sums are EXACT integers and crossing has no float ties.
            series[st] = np.rint(v * 10).astype(np.int64)
        if len(series) >= MAX_STATIONS:
            break
    return series


def brute_crossing(a, mu, h):
    carry = 0           # integer arithmetic over the quantized domain
    for t in range(len(a)):
        carry = max(0, carry + (int(a[t]) - mu))
        if carry >= h:
            return t
    return None


def main():
    print("loading real EPA PM2.5 ...", flush=True)
    series = load_stations()
    n_days = [len(v) for v in series.values()]
    print(f"  {len(series)} stations, days median={int(np.median(n_days))} "
          f"min={min(n_days)} max={max(n_days)} total={sum(n_days)}", flush=True)

    # baseline mu grid (changed prior) and threshold grid (changed h), in integer tenths of ug/m3
    mus = [50, 80, 100, 120, 150]           # PM2.5 baselines x10; EPA annual NAAQS ~ 9-12 ug/m3
    hs = [100, 250, 500]                     # CUSUM thresholds x10

    # build P-DRI per station once (a_t = quantized daily PM2.5, b_t = 1)
    t0 = time.perf_counter()
    idx = {st: PD.PDRI([(int(x), 1.0) for x in v]) for st, v in series.items()}
    build_s = time.perf_counter() - t0

    # exactness + latency across all (station, mu, h)
    xfail = 0; nq = 0
    pdri_us, scan_us = [], []
    hull_sub, hull_pre, hull_suf = [], [], []
    crossings = 0
    for st, v in series.items():
        a = v; R = len(a)
        # collect node hull sizes for PD2
        I = idx[st]
        for ni in range(1, 2 * I.size):
            node = I.tree[ni]
            if node is not None and node.sub[0][0] > -1e17:    # skip padding leaves
                hull_sub.append(len(node.sub)); hull_pre.append(len(node.pre)); hull_suf.append(len(node.suf))
        for mu in mus:
            for h in hs:
                t1 = time.perf_counter(); got = I.earliest_crossing(0, R - 1, mu, h)
                pdri_us.append((time.perf_counter() - t1) * 1e6)
                t2 = time.perf_counter(); exp = brute_crossing(a, mu, h)
                scan_us.append((time.perf_counter() - t2) * 1e6)
                nq += 1
                if got != exp: xfail += 1
                if got is not None: crossings += 1

    # rebuild-DRI-per-mu cost: rebuilding the index for a NEW baseline = full O(R) rebuild per mu
    # (a fixed-mu DRI must be rebuilt when mu changes). Measure one representative station rebuild.
    rep = max(series, key=lambda s: len(series[s])); a_rep = series[rep]
    t3 = time.perf_counter()
    for mu in mus:
        _ = PD.PDRI([(float(x), 0.0) for x in a_rep])   # a numeric DRI would rebuild from x_t=a_t-mu each time
    rebuild_per_mu_ms = (time.perf_counter() - t3) / len(mus) * 1e3

    # PD update cost: per-round append = re-merge the O(log R) root-to-leaf path. Measure on every station.
    update_us = []
    for st, v in series.items():
        I = idx[st]; R = len(v)
        for i in range(0, R, max(1, R // 40)):          # sample ~40 leaf positions per station
            t = time.perf_counter(); I.update_leaf(i, int(v[i]), 1.0); update_us.append((time.perf_counter()-t)*1e6)

    def stats(xs):
        xs = sorted(xs); n = len(xs)
        return {"median": round(xs[n // 2], 3), "p95": round(xs[int(0.95 * n)], 3), "max": round(xs[-1], 3)}

    out = {"_note": "P-DRI on REAL EPA PM2.5 2023 daily. driftReplay(mu,h) over per-station reflected CUSUM; "
                    "DRI(fixed mu0) cannot answer changed-mu; P-DRI serves any (mu,h) from one sealed index, "
                    "exact vs brute-force oracle.",
           "workload": {"stations": len(series), "total_station_days": int(sum(n_days)),
                        "days_median": int(np.median(n_days)), "mu_grid": mus, "h_grid": hs},
           "exactness": {"queries": nq, "mismatches_vs_oracle": xfail, "crossings_found": crossings,
                         "exact": xfail == 0},
           "PD1_latency": {"pdri_query_us": stats(pdri_us), "raw_scan_us": stats(scan_us),
                           "pdri_update_us": stats(update_us), "pdri_update_samples": len(update_us),
                           "rebuild_dri_per_mu_ms_rep_station": round(rebuild_per_mu_ms, 2),
                           "pdri_build_all_stations_s": round(build_s, 2),
                           "speedup_pdri_vs_scan_median": round(
                               sorted(scan_us)[len(scan_us)//2] / max(sorted(pdri_us)[len(pdri_us)//2], 1e-6), 1)},
           "PD2_hull_sizes": {"sub": stats(hull_sub), "pre": stats(hull_pre), "suf": stats(hull_suf),
                              "nodes_measured": len(hull_sub)},
           "PD3_proof_proxy": {"active_line_bytes": 16, "proof_lines_per_query_<=": stats(hull_sub)["p95"],
                               "note": "receipt = O(log R) node Merkle paths + 1 active line + breakpoints "
                                       "per cover node; hull size bounds the active-line set"}}
    print(json.dumps(out, indent=2))
    od = ROOT / "results/artifact"; od.mkdir(parents=True, exist_ok=True)
    (od / "p_dri_epa.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "p_dri_epa.json")


if __name__ == "__main__":
    main()
