#!/usr/bin/env python3
"""A ROUND-RANGE-capable authenticated baseline in SRI's own class, so the in-class
serving comparison is on equal footing (the Merkle-Fenwick baseline is snapshot-only).

PersistentMerkleFenwick: a versioned Fenwick/BIT whose every cell keeps a per-round value history,
authenticated by a per-cell Merkle tree over its versions plus a global Merkle tree over the cells.
It answers root-verifiable count_below over a round range (r1,r2] exactly like Auth-SRI -- but built on
a BIT instead of a persistent segment tree -- so we can ask whether SRI is competitive in its class.

We build both Auth-SRI (persistent rank-sum segment tree) and PersistentMerkleFenwick on the real
trust ledger and compare the SAME round-range count_below query: correctness vs a linear oracle, query /
proof-gen / proof-size / verify, storage, build, and tamper rejection.
Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/db_eval/persistent_serving_bench.py
"""
import bisect, copy, hashlib, json, sqlite3, struct, sys, time
from pathlib import Path
import importlib.util as u
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

def proof_size(obj):
    """Wire size: 32 B per hash, ~6 B per integer, recursing into dicts/lists."""
    if isinstance(obj, (bytes, bytearray)): return len(obj)
    if isinstance(obj, bool): return 1
    if isinstance(obj, int): return 6
    if isinstance(obj, str): return len(obj)
    if isinstance(obj, dict): return sum(proof_size(v) for v in obj.values())
    if isinstance(obj, (list, tuple)): return sum(proof_size(v) for v in obj)
    return 8

def _load(name, rel):
    s = u.spec_from_file_location(name, ROOT / rel); m = u.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m
SRI = _load("sri_index", "dtrust/store/sri_index.py")
ASRI = _load("sri_authenticated", "dtrust/store/sri_authenticated.py")
U = SRI.T_SCORE_DOMAIN

def _h(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(p)
    return h.digest()
NULL = b"\x00" * 32
def _vleaf(rnd, val): return _h(b"V", struct.pack("<qq", int(rnd), int(val)))


def _merkle_root(leaves):
    if not leaves: return NULL
    cur = list(leaves)
    while len(cur) > 1:
        cur = [_h(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i]) for i in range(0, len(cur), 2)]
    return cur[0]

def _merkle_levels(leaves):
    """Return all levels (level 0 = leaves, last = [root]) so paths are O(log n) lookups."""
    if not leaves: return [[NULL]]
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_h(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i]) for i in range(0, len(cur), 2)])
    return levels

def _path_from_levels(levels, idx):
    path, i = [], idx
    for lvl in levels[:-1]:
        sib = i ^ 1
        path.append(lvl[sib] if sib < len(lvl) else lvl[i])
        i //= 2
    return path

def _merkle_path(leaves, idx):
    return _path_from_levels(_merkle_levels(leaves), idx)

def _root_from_path(leaf, idx, path):
    node, i = leaf, idx
    for sib in path:
        node = _h(node, sib) if i % 2 == 0 else _h(sib, node)
        i //= 2
    return node


class PersistentMerkleFenwick:
    """Versioned BIT; each cell keeps (round, value) history. Authenticated by per-cell Merkle roots
    committed in a global Merkle tree over the U+1 cells. Round-range count_below via two prefix proofs."""
    def __init__(self, U=U):
        self.U = U
        self.cur = np.zeros(U + 1, dtype=np.int64)
        self.hist_r = [[] for _ in range(U + 1)]     # per-cell list of rounds touched
        self.hist_v = [[] for _ in range(U + 1)]     # per-cell list of values after that round
        self._built = False

    def append_round(self, r, values):
        touched = set()
        for v in values:
            i = int(v) + 1
            while i <= self.U:
                self.cur[i] += 1; touched.add(i); i += i & (-i)
        for i in touched:
            self.hist_r[i].append(r); self.hist_v[i].append(int(self.cur[i]))

    def storage_versions(self):
        return int(sum(len(h) for h in self.hist_r))

    def finalize(self):
        # per-cell version leaves + Merkle levels; global Merkle levels over the U+1 cell roots.
        self.cell_leaves = [None] * (self.U + 1)
        self.cell_levels = [None] * (self.U + 1)
        self.cell_root = [NULL] * (self.U + 1)
        for i in range(self.U + 1):
            leaves = [_vleaf(self.hist_r[i][k], self.hist_v[i][k]) for k in range(len(self.hist_r[i]))]
            self.cell_leaves[i] = leaves
            if leaves:
                self.cell_levels[i] = _merkle_levels(leaves); self.cell_root[i] = self.cell_levels[i][-1][0]
        self.global_levels = _merkle_levels(self.cell_root)
        self.global_root = self.global_levels[-1][0]
        self._built = True

    def auth_root(self):
        return _h(b"PMF", struct.pack("<q", self.U + 1), self.global_root)

    def _ver_at(self, i, r):
        """index k of the largest hist_r[i] <= r, or -1."""
        return bisect.bisect_right(self.hist_r[i], r) - 1

    def _prefix_path(self, tau, r):
        """cells on tau's BIT path, each with its chosen version + boundary proofs (sound)."""
        cells, total, i = [], 0, tau
        while i > 0:
            k = self._ver_at(i, r); n = len(self.hist_r[i])
            entry = {"i": i, "n": n, "cell_path": _path_from_levels(self.global_levels, i)}
            if k >= 0:                                 # cell has a version <= r
                entry.update({"k": k, "rk": self.hist_r[i][k], "vk": self.hist_v[i][k],
                              "vpath": _path_from_levels(self.cell_levels[i], k)})
                if k + 1 < n:                          # boundary: next version is strictly later than r
                    entry.update({"rk1": self.hist_r[i][k + 1], "vk1": self.hist_v[i][k + 1],
                                  "vpath1": _path_from_levels(self.cell_levels[i], k + 1)})
                total += self.hist_v[i][k]
            elif n > 0:                                # value 0 at round r: prove first touch is after r
                entry.update({"k": -1, "r0": self.hist_r[i][0], "v0": self.hist_v[i][0],
                              "vpath0": _path_from_levels(self.cell_levels[i], 0)})
            else:
                entry["k"] = -1                        # genuinely empty cell (cell_root = NULL)
            cells.append(entry); i -= i & (-i)
        return total, cells

    def count_below_range(self, r1, r2, tau):
        return self._prefix_path(tau, r2)[0] - self._prefix_path(tau, r1 - 1)[0]

    def prove_range(self, r1, r2, tau):
        thi, hi = self._prefix_path(tau, r2)
        tlo, lo = self._prefix_path(tau, r1 - 1)
        return {"r1": r1, "r2": r2, "tau": tau, "U": self.U, "count": thi - tlo,
                "hi": hi, "lo": lo, "hi_total": thi, "lo_total": tlo}


def verify_pmf(auth_root, U, r1, r2, tau, claimed, proof):
    """Sound external verify: recompute per-cell + global roots, check version boundary, sum, bind query."""
    if (proof.get("r1"), proof.get("r2"), proof.get("tau"), proof.get("U")) != (r1, r2, tau, U):
        return False
    global_root_expected = None

    def check_prefix(cells, r):
        nonlocal global_root_expected
        # cells must be exactly tau's BIT decomposition
        idxs, j = [], tau
        while j > 0: idxs.append(j); j -= j & (-j)
        if [c["i"] for c in cells] != idxs: return None
        total = 0
        for c in cells:
            n, i = c["n"], c["i"]
            if c["k"] >= 0:                            # chosen version <= r contributes vk
                if not (c["rk"] <= r): return None
                if "rk1" in c and not (c["rk1"] > r): return None  # boundary: next version strictly > r
                if "rk1" not in c and c["k"] != n - 1: return None # no boundary => must be the last version
                leaf = _vleaf(c["rk"], c["vk"]); cell_root = _root_from_path(leaf, c["k"], c["vpath"])
                if "rk1" in c:
                    leaf1 = _vleaf(c["rk1"], c["vk1"])
                    if _root_from_path(leaf1, c["k"] + 1, c["vpath1"]) != cell_root: return None
                total += c["vk"]
            elif n > 0:                                # value 0 at round r: first touch must be after r
                if not (c["r0"] > r): return None
                leaf0 = _vleaf(c["r0"], c["v0"]); cell_root = _root_from_path(leaf0, 0, c["vpath0"])
            else:
                cell_root = NULL                       # genuinely empty cell
            gr = _root_from_path(cell_root, i, c["cell_path"])
            if global_root_expected is None: global_root_expected = gr
            elif gr != global_root_expected: return None
        return total

    thi = check_prefix(proof["hi"], r2)
    tlo = check_prefix(proof["lo"], r1 - 1)
    if thi is None or tlo is None: return False
    if (thi - tlo) != claimed: return False
    return _h(b"PMF", struct.pack("<q", U + 1), global_root_expected) == auth_root


def load_ledger(max_cells=0):
    vals = []
    for db in sorted((ROOT / "results/recovery/c1_5seed_audit_int0.7").glob("ledger_seed*.db")):
        con = sqlite3.connect(db); vals += [r[0] for r in con.execute("SELECT t_score FROM tuples")]; con.close()
    v = np.array(vals, dtype=float)
    if max_cells and v.size > max_cells: v = v[:max_cells]
    lo, hi = float(v.min()), float(v.max())
    q = np.zeros(v.size, dtype=np.int64) if hi <= lo else np.clip(((v - lo) / (hi - lo) * (U - 1)).astype(np.int64), 0, U - 1)
    return q


def med_us(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort(); return ts[len(ts) // 2]


def bench_one(q, label, N=50):
    n = q.size; R = max(2, n // N); tau = U // 2 - 1   # non-power-of-2 -> full BIT path
    r1, r2 = R // 4, 3 * R // 4
    res = {"workload": label, "n_cells": int(n), "rounds": R, "N": N,
           "tau_q": int(tau), "range": [r1, r2], "tau_bit_cells": bin(tau).count("1")}

    # linear oracle over the range
    seg = q[r1 * N:(r2 + 1) * N]; oracle = int((seg < tau).sum())

    # Auth-SRI
    t0 = time.perf_counter(); asri = ASRI.AuthSRIIndex(U)
    for r in range(R): asri.append_round(r, [(int(x), 1.0) for x in q[r * N:(r + 1) * N].tolist()])
    res["sri_build_s"] = round(time.perf_counter() - t0, 2)
    res["sri_nodes"] = int(getattr(asri.tree, "n_nodes", lambda: len(getattr(asri.tree, "dig", [])))())
    aroot = asri.auth_root()
    c_sri = asri.count_below(r1, r2, tau)[0]
    res["sri_correct"] = (c_sri == oracle)
    res["sri_query_us"] = round(med_us(lambda: asri.count_below(r1, r2, tau), 30), 2)
    res["sri_provegen_us"] = round(med_us(lambda: asri.prove_count_below(r1, r2, tau), 20), 2)
    pr = asri.prove_count_below(r1, r2, tau)
    res["sri_proof_bytes"] = proof_size(pr)
    res["sri_verify_us"] = round(med_us(lambda: ASRI.verify_count_below(aroot, r1, r2, tau, c_sri, asri.count_below(r1, r2, tau)[1], pr, U), 20), 2)
    res["sri_verifies"] = bool(ASRI.verify_count_below(aroot, r1, r2, tau, c_sri, asri.count_below(r1, r2, tau)[1], pr, U))
    res["sri_rejects_wrong_tau"] = (not ASRI.verify_count_below(aroot, r1, r2, tau + 7, c_sri, 0.0, pr, U))

    # PersistentMerkleFenwick
    t0 = time.perf_counter(); pmf = PersistentMerkleFenwick(U)
    for r in range(R): pmf.append_round(r, q[r * N:(r + 1) * N].tolist())
    res["pmf_versions"] = pmf.storage_versions()
    pmf.finalize(); res["pmf_build_s"] = round(time.perf_counter() - t0, 2)
    proot = pmf.auth_root()
    c_pmf = pmf.count_below_range(r1, r2, tau)
    res["pmf_correct"] = (c_pmf == oracle)
    res["pmf_query_us"] = round(med_us(lambda: pmf.count_below_range(r1, r2, tau), 30), 2)
    res["pmf_provegen_us"] = round(med_us(lambda: pmf.prove_range(r1, r2, tau), 20), 2)
    pp = pmf.prove_range(r1, r2, tau)
    res["pmf_proof_bytes"] = proof_size(pp)
    res["pmf_verify_us"] = round(med_us(lambda: verify_pmf(proot, U, r1, r2, tau, c_pmf, pp), 20), 2)
    res["pmf_verifies"] = bool(verify_pmf(proot, U, r1, r2, tau, c_pmf, pp))
    res["pmf_rejects_wrong_tau"] = (not verify_pmf(proot, U, r1, r2, tau + 7, c_pmf, pp))
    # tamper: flip one proven cell value -> must reject (deepcopy preserves the byte-string hashes)
    pp_bad = copy.deepcopy(pp)
    for c in pp_bad["hi"]:
        if c.get("k", -1) >= 0: c["vk"] = c["vk"] + 1; break
    res["pmf_rejects_tamper"] = (not verify_pmf(proot, U, r1, r2, tau, c_pmf, pp_bad))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["fl_ledger"],
                    help="fl_ledger / creditcard / taxi_big / ghcn / epa_hourly / hvfhv")
    ap.add_argument("--max_cells", type=int, default=200000,
                    help="cap (authenticated structures fit in memory; equal-footing across datasets)")
    ap.add_argument("--out", default=str(ROOT / "results/artifact/persistent_serving_diverse.json"))
    a = ap.parse_args()
    # reuse real_serving_bench's diverse loaders (same U = SRI.T_SCORE_DOMAIN)
    rsb = _load("real_serving_bench", "experiment_scripts/dtrust_experiments/db_eval/real_serving_bench.py")
    out = {"_note": "Round-range authenticated in-class comparison (Auth-SRI persistent segment tree vs "
                    "PersistentMerkleFenwick versioned BIT) on DIVERSE real datasets, capped where the "
                    "authenticated structures fit; equal-footing, not snapshot-only.",
           "max_cells": a.max_cells, "results": []}
    for w in a.workloads:
        try:
            q, label = rsb.load_real(w, a.max_cells)
            print(f"=== {w} (n={q.size}) ===", flush=True)
            res = bench_one(q, label); out["results"].append(res)
            print(json.dumps(res, indent=2), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  SKIP {w}: {type(e).__name__}: {e}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2))   # checkpoint per workload
    print("wrote", a.out)


if __name__ == "__main__":
    main()
