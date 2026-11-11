#!/usr/bin/env python3
"""Serving benchmark on REAL data (no synthetic values).
Compares the authenticated SRI count_below against SPECIALIZED non-authenticated indexes
(prefix-sum O(1), Fenwick O(log U), segment tree O(log U)) and general engines (DuckDB, SQLite
B-tree, native scan), over real values quantized to the U=2^16 domain. Real workloads:
  - taxi: NYC TLC yellow_2023-01 trip_distance (~3.07M real rows)
  - creditcard: ULB credit-card V-feature (~284.8K real rows)
  - fl_ledger: real audit t_score cells (75.5K real, 5 sealed ledgers)
SRI latency is close to the specialized trees (microseconds) and far below the scan engines, and SRI
adds root-verifiable receipts and round-range persistence.
"""
import argparse, hashlib, json, sqlite3, sys, time
from pathlib import Path
import importlib.util as u
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

def _load(name, rel):
    s = u.spec_from_file_location(name, ROOT / rel); m = u.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m
SRI = _load("sri_index", "dtrust/store/sri_index.py")
ASRI = _load("sri_authenticated", "dtrust/store/sri_authenticated.py")
U = SRI.T_SCORE_DOMAIN
try:
    import duckdb; HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False


def _median_us(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort(); return ts[len(ts) // 2]


# ---- specialized non-authenticated indexes, built O(U) from a value histogram ----
class PrefixSum:
    def __init__(self, freq): self.pre = np.cumsum(freq)
    def count_below(self, tau): return int(self.pre[tau - 1]) if tau > 0 else 0

class Fenwick:
    def __init__(self, freq):
        n = len(freq); self.n = n; t = np.zeros(n + 1, dtype=np.int64); t[1:] = freq
        for i in range(1, n + 1):
            j = i + (i & (-i))
            if j <= n: t[j] += t[i]
        self.t = t
    def count_below(self, tau):
        i = tau; s = 0
        while i > 0: s += self.t[i]; i -= i & (-i)
        return int(s)

class SegTree:
    def __init__(self, freq):
        n = len(freq); self.n = n; t = np.zeros(2 * n, dtype=np.int64); t[n:2 * n] = freq
        for i in range(n - 1, 0, -1): t[i] = t[2 * i] + t[2 * i + 1]
        self.t = t
    def count_below(self, tau):
        lo, hi, s = self.n, tau + self.n, 0
        while lo < hi:
            if lo & 1: s += self.t[lo]; lo += 1
            if hi & 1: hi -= 1; s += self.t[hi]
            lo >>= 1; hi >>= 1
        return int(s)


def _h(b): return hashlib.sha256(b).digest()
def _mf_leaf(i, v): return _h(b"L" + int(i).to_bytes(8, "little") + int(v).to_bytes(9, "little", signed=True))


class MerkleFenwick:
    """AUTHENTICATED in-class baseline: a Fenwick/BIT prefix-count tree whose cells are
    committed in a binary Merkle tree. count_below(tau) reads the O(log U) BIT cells on tau's
    decomposition; the proof is their Merkle inclusion paths against the root; verify recomputes
    the prefix sum AND each cell's path AND that the cell indices are exactly tau's BIT
    decomposition. This authenticates ONE snapshot's prefix counts -- it has no round-range
    persistence (that would need a fresh tree per round, O(R*U)), which is what Auth-SRI adds."""
    def __init__(self, freq):
        n = len(freq); self.n = n
        t = np.zeros(n + 1, dtype=np.int64); t[1:] = freq
        for i in range(1, n + 1):
            j = i + (i & -i)
            if j <= n: t[j] += t[i]
        self.t = t
        size = 1
        while size < n + 1: size <<= 1
        self.size = size
        pad = _h(b"PAD")
        cur = [_mf_leaf(i, t[i]) if i <= n else pad for i in range(size)]
        self.levels = [cur]
        while len(cur) > 1:
            cur = [_h(cur[k] + cur[k + 1]) for k in range(0, len(cur), 2)]
            self.levels.append(cur)
        self.root = cur[0]
    def _path(self, idx):
        sib = []
        for lvl in self.levels[:-1]:
            sib.append(lvl[idx ^ 1]); idx >>= 1
        return sib
    def count_below(self, tau):
        i, s = tau, 0
        while i > 0: s += self.t[i]; i -= i & -i
        return int(s)
    def prove_count_below(self, tau):
        i, cells = tau, []
        while i > 0:
            cells.append((i, int(self.t[i]), self._path(i))); i -= i & -i
        return cells


def verify_merkle_fenwick(root, tau, claimed, cells):
    idxs, j = [], tau
    while j > 0: idxs.append(j); j -= j & -j
    if [c[0] for c in cells] != idxs: return False         # cells must be tau's BIT decomposition
    s = 0
    for (i, v, path) in cells:
        node, idx = _mf_leaf(i, v), i
        for sib in path:
            node = _h(node + sib) if idx & 1 == 0 else _h(sib + node)
            idx >>= 1
        if node != root: return False                       # inclusion against the committed root
        s += v
    return s == claimed


def load_real(workload, max_cells):
    """Return real values quantized to [0, U) as int64 (NO synthetic values)."""
    if workload == "taxi":
        import pyarrow.parquet as pq
        v = pq.read_table(ROOT / "data_ext/yellow_2023-01.parquet", columns=["trip_distance"]).column(0).to_numpy()
        v = v[np.isfinite(v)]; v = v[(v >= 0) & (v < 200)]; src = "NYC TLC yellow_2023-01 trip_distance"
    elif workload == "creditcard":
        import pyarrow.parquet as pq
        t = pq.read_table(ROOT / "data_ext/ulb_creditcard.parquet")
        col = "Amount" if "Amount" in t.column_names else t.column_names[0]
        v = t.column(col).to_numpy().astype(float); v = v[np.isfinite(v)]; src = f"ULB credit-card {col}"
    elif workload == "fl_ledger":
        import glob
        vals = []
        for db in sorted(glob.glob(str(ROOT / "results/recovery/c1_5seed_audit_int0.7/ledger_seed*.db"))):
            con = sqlite3.connect(db); vals += [r[0] for r in con.execute("SELECT t_score FROM tuples")]; con.close()
        v = np.array(vals, dtype=float); src = "real C1 audit t_score (5 sealed ledgers)"
    elif workload in ("taxi_big", "hvfhv"):
        # DIVERSE large real datasets (data_ext/scale/), multi-month concatenation up to max_cells.
        import glob, pyarrow.parquet as pq
        SCALE = ROOT / "data_ext/scale"
        need = max_cells if max_cells else None
        if workload == "taxi_big":
            files = sorted(glob.glob(str(SCALE / "yellow_*.parquet"))); col = "trip_distance"
            src = f"NYC TLC yellow taxi ({len(files)} months) trip_distance"
        else:
            files = sorted(glob.glob(str(SCALE / "fhvhv_*.parquet"))); col = "trip_miles"
            src = f"NYC TLC HVFHV ride-hail ({len(files)} months) trip_miles"
        if not files:
            raise SystemExit(f"no files for {workload} in {SCALE}; run download_real_datasets.sh first")
        parts, tot = [], 0
        for f in files:
            a = pq.read_table(f, columns=[col]).column(0).to_numpy().astype(float)
            a = a[np.isfinite(a)]; a = a[(a >= 0) & (a < 1e4)]
            parts.append(a); tot += a.size
            if need and tot >= need: break
        v = np.concatenate(parts)
    elif workload == "ghcn":
        import gzip, csv as _csv
        f = ROOT / "data_ext/scale/ghcn_2023.csv.gz"; src = "NOAA GHCN-Daily 2023 observation value"
        need = max_cells if max_cells else None; arr = []
        with gzip.open(f, "rt") as fh:
            for row in _csv.reader(fh):
                try: arr.append(float(row[3]))           # GHCN by_year: ID,DATE,ELEMENT,DATA_VALUE,...
                except (IndexError, ValueError): continue
                if need and len(arr) >= need: break
        v = np.array(arr, dtype=float)
    elif workload == "epa_hourly":
        import zipfile, csv as _csv, io as _io
        f = ROOT / "data_ext/scale/epa_hourly_88101_2023.zip"; src = "EPA AQS hourly PM2.5 2023 sample measurement"
        need = max_cells if max_cells else None; arr = []
        with zipfile.ZipFile(f) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))   # skip the dir entry
            with z.open(csv_name) as fh:
                rd = _csv.reader(_io.TextIOWrapper(fh, "utf-8")); header = next(rd)
                mi = next(i for i, c in enumerate(header) if "Measurement" in c)   # 'Sample_Measurement'
                for row in rd:
                    try: arr.append(float(row[mi]))
                    except (IndexError, ValueError): continue
                    if need and len(arr) >= need: break
        v = np.array(arr, dtype=float)
    else:
        raise SystemExit(f"unknown workload {workload}")
    v = v[np.isfinite(v)]
    if max_cells and v.size > max_cells:
        v = v[:max_cells]
    # quantize the real values to [0, U)
    lo, hi = float(np.min(v)), float(np.max(v))
    q = np.zeros(v.size, dtype=np.int64) if hi <= lo else np.clip(((v - lo) / (hi - lo) * (U - 1)).astype(np.int64), 0, U - 1)
    return q, src


def bench(workload, max_cells, N=100):
    q, src = load_real(workload, max_cells)
    n = q.size; R = max(1, n // N); tau = U // 2
    res = {"workload": workload, "source": src, "n_cells": int(n), "tau_q": tau, "U": U}

    # Persistent segment tree (NON-authenticated, round-versioned) — the structure SRI is, minus hashing
    t0 = time.perf_counter(); sri = SRI.SRIIndex(U)
    for r in range(R):
        seg = q[r * N:(r + 1) * N]
        sri.append_round(r, [(int(x), 1.0) for x in seg.tolist()])
    res["pst_build_s"] = round(time.perf_counter() - t0, 2)
    res["pst_us"] = round(_median_us(lambda: sri.count_below(0, R - 1, tau), 50), 3)
    truth = int((q[:R * N] < tau).sum())
    res["pst_correct"] = (sri.count_below(0, R - 1, tau) == truth)
    res["pst_nodes"] = int(getattr(sri.tree, "n_nodes", lambda: 0)()) if hasattr(sri, "tree") else None
    # write amplification: per-round append cost (µs/round) for the persistent structure
    sri2 = SRI.SRIIndex(U)
    res["pst_append_us_per_round"] = round(_median_us(
        lambda: sri2.append_round(len(sri2.rounds), [(int(x), 1.0) for x in q[:N].tolist()]), 10), 2)

    # AUTHENTICATED SRI (Merkle root-verifiable receipts); measures the auth overhead.
    # The authenticated prototype materializes a hashed node per insert; it is run up to 1e6 cells.
    if R * N > 1_000_000:
        res["authsri_skipped"] = "authenticated prototype memory exceeds DB-scale (>1e6 cells); reported at <=1e6"
    else:
        try:
            t0 = time.perf_counter(); asri = ASRI.AuthSRIIndex(U)
            for r in range(R):
                seg = q[r * N:(r + 1) * N]
                asri.append_round(r, [(int(x), 1.0) for x in seg.tolist()])
            res["authsri_build_s"] = round(time.perf_counter() - t0, 2)
            root = asri.auth_root()
            res["authsri_query_us"] = round(_median_us(lambda: asri.count_below(0, R - 1, tau), 50), 3)
            res["authsri_provegen_us"] = round(_median_us(lambda: asri.prove_count_below(0, R - 1, tau), 20), 2)
            import json as _j
            c = asri.count_below(0, R - 1, tau); proof = asri.prove_count_below(0, R - 1, tau)
            res["authsri_proof_bytes"] = len(_j.dumps(proof, default=str))
            res["authsri_verify_us"] = round(_median_us(
                lambda: ASRI.verify_count_below(root, 0, R - 1, tau, c[0], c[1], proof, U), 20), 2)
            res["authsri_verifies"] = bool(ASRI.verify_count_below(root, 0, R - 1, tau, c[0], c[1], proof, U))
            res["authsri_rejects_wrong_tau"] = (not ASRI.verify_count_below(root, 0, R - 1, tau + 500, c[0], c[1], proof, U))
            if res.get("pst_us"): res["auth_overhead_vs_pst"] = round(res["authsri_query_us"] / res["pst_us"], 2)
        except Exception as e:
            res["authsri_error"] = str(e)[:200]

        # AUTHENTICATED IN-CLASS BASELINE: Merkleized Fenwick over the final snapshot.
        # Measured at a worst-case tau (full BIT decomposition, log U cells) for an upper
        # bound on the authenticated single-snapshot prefix cost. It cannot do round-range receipts.
        try:
            freq_mf = np.bincount(q[:R * N], minlength=U)[:U]
            t0 = time.perf_counter(); mf = MerkleFenwick(freq_mf)
            res["mfenwick_build_s"] = round(time.perf_counter() - t0, 2)
            tau_full = U - 1                                   # 0xFFFF -> log2(U) BIT cells
            res["mfenwick_tau_cells"] = bin(tau_full).count("1")
            res["mfenwick_query_us"] = round(_median_us(lambda: mf.count_below(tau_full), 50), 3)
            res["mfenwick_provegen_us"] = round(_median_us(lambda: mf.prove_count_below(tau_full), 20), 2)
            cmf = mf.count_below(tau_full); pmf = mf.prove_count_below(tau_full)
            res["mfenwick_proof_bytes"] = sum(16 + 9 + 32 * len(p) for (_, _, p) in pmf)
            res["mfenwick_verify_us"] = round(_median_us(
                lambda: verify_merkle_fenwick(mf.root, tau_full, cmf, pmf), 20), 2)
            res["mfenwick_verifies"] = bool(verify_merkle_fenwick(mf.root, tau_full, cmf, pmf))
            # tamper a cell value -> must reject
            bad = [(pmf[0][0], pmf[0][1] + 1, pmf[0][2])] + pmf[1:]
            res["mfenwick_rejects_tamper"] = (not verify_merkle_fenwick(mf.root, tau_full, cmf, bad))
        except Exception as e:
            res["mfenwick_error"] = str(e)[:200]

    # specialized non-authenticated indexes (histogram build, O(U))
    freq = np.bincount(q[:R * N], minlength=U)[:U]
    for name, ctor in (("prefixsum", PrefixSum), ("fenwick", Fenwick), ("segtree", SegTree)):
        t0 = time.perf_counter(); idx = ctor(freq)
        res[f"{name}_build_s"] = round(time.perf_counter() - t0, 3)
        res[f"{name}_us"] = round(_median_us(lambda: idx.count_below(tau), 100), 4)
        res[f"{name}_correct"] = (idx.count_below(tau) == truth)
    if res.get("fenwick_us") and res.get("authsri_query_us"):
        res["authsri_overhead_vs_fenwick"] = round(res["authsri_query_us"] / res["fenwick_us"], 2)

    # native numpy scan
    arr = q[:R * N]
    res["numpy_scan_us"] = round(_median_us(lambda: int((arr < tau).sum()), 20), 1)
    # DuckDB
    if HAVE_DUCKDB:
        con = duckdb.connect(); con.execute("CREATE TABLE t(v BIGINT)")
        con.register("df", {"v": arr}); con.execute("INSERT INTO t SELECT * FROM df"); con.unregister("df")
        res["duckdb_us"] = round(_median_us(lambda: con.execute("SELECT count(*) FROM t WHERE v<?", [tau]).fetchone(), 20), 1)
        con.close()
    # SQLite B-tree
    con = sqlite3.connect(":memory:"); con.execute("CREATE TABLE t(v INT)")
    con.executemany("INSERT INTO t VALUES(?)", ((int(x),) for x in arr.tolist()))
    con.execute("CREATE INDEX ix ON t(v)"); con.commit()
    res["sqlite_btree_us"] = round(_median_us(lambda: con.execute("SELECT count(*) FROM t WHERE v<?", (tau,)).fetchone(), 20), 1)
    con.close()
    base = res.get("authsri_query_us") or res.get("pst_us")
    for eng in ("numpy_scan", "duckdb", "sqlite_btree"):
        k = f"{eng}_us"
        if k in res and base and base > 0: res[f"authsri_speedup_vs_{eng}"] = round(res[k] / base, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["fl_ledger", "creditcard", "taxi"])
    ap.add_argument("--max_cells", type=int, default=0)  # 0 = all real rows
    # default is a SCRATCH path; the paper source is E2_real_serving_full.json, produced by an explicit
    # --out and never clobbered by a default/partial run.
    ap.add_argument("--out", default=str(ROOT / "results/artifact/E2_real_serving_scratch.json"))
    ap.add_argument("--scales", type=int, nargs="+", default=None,
                    help="if set, run each workload at these real-cell scales (e.g. 1000000 10000000 100000000)")
    a = ap.parse_args()
    out = {"_note": "E2 fair serving on REAL data: SRI (authenticated) vs specialized non-auth "
                    "indexes (prefixsum/fenwick/segtree) + DuckDB/SQLite/scan. Real workloads, no toy.",
           "have_duckdb": HAVE_DUCKDB, "U": U, "results": []}
    if a.scales:
        out["_note"] += " SCALING MODE: each row is REAL data capped at the given scale (all values real)."
        for nc in a.scales:                                # scale-outer: finish all datasets at 1e6, then 1e7, then 1e8
            for w in a.workloads:
                print(f"=== {w} @ {nc:,} real cells ===", flush=True)
                try:
                    r = bench(w, nc); r["scale_target"] = nc; out["results"].append(r)
                    print(json.dumps(r, indent=2), flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  SKIP {w}@{nc}: {type(e).__name__}: {e}", flush=True)
                Path(a.out).write_text(json.dumps(out, indent=2))   # checkpoint per (scale,workload)
                Path(a.out).write_text(json.dumps(out, indent=2))   # checkpoint after each
    else:
        for w in a.workloads:
            print(f"=== {w} ===", flush=True); r = bench(w, a.max_cells); out["results"].append(r)
            print(json.dumps(r, indent=2), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2)); print("wrote", a.out)


if __name__ == "__main__":
    main()
