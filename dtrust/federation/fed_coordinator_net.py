"""EF3 networked coordinator — broadcasts hash-chained heads to silo witnesses over
HTTP, then runs the rollback/truncation/fork audit against the live witnesses.

This is the networked transport for the EF3 protocol validated locally in
fed_witness.py: instead of in-process Witness objects, the heads are POSTed to real
silo witness servers across the federation, and the audit GETs each silo's latest
head over the network. With >=1 honest silo witness, a host that rolls back,
truncates, or forks the coordinator ledger is detected — no external transparency log.

    python3 fed_coordinator_net.py --silos S1=http://ip1:8137,S2=http://ip2:8137,...
stdlib only.
"""
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

EPOCH = 1


def H(*bs):
    h = hashlib.sha256()
    for b in bs:
        h.update(b)
    return h.digest()


def post(url, obj, timeout=10):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silos", required=True,
                    help="comma list name=url, e.g. S1=http://ip:8137,S2=...")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", default="results/recovery/federation/ef3_networked.json")
    a = ap.parse_args()
    silos = dict(kv.split("=", 1) for kv in a.silos.split(","))
    R = a.rounds

    # reset witnesses, then anchor each round and broadcast the head over the network
    for url in silos.values():
        try: post(url + "/reset", {})
        except Exception as e: print(f"reset {url} FAIL {e}")
    log = []
    prev = b"\x00" * 32
    broadcast_ok = {n: 0 for n in silos}
    for r in range(R):
        root = H(f"round-{r}-payload".encode())
        entry = struct.pack("<I I 32s 32s", r, EPOCH, root, prev)
        log_hash = H(entry)
        msg = {"round": r, "root": root.hex(), "prev_log": prev.hex(), "log_hash": log_hash.hex()}
        log.append(msg)
        prev = log_hash
        for n, url in silos.items():
            try:
                post(url + "/head", msg); broadcast_ok[n] += 1
            except Exception as e:
                print(f"broadcast r{r} -> {n} FAIL {e}")
    true_head, true_round = log[-1]["log_hash"], R - 1

    def witness_heads():
        out = {}
        for n, url in silos.items():
            try: out[n] = get(url + "/head")
            except Exception as e: out[n] = {"error": str(e)}
        return out

    def audit(presented_hash, presented_round):
        # detected iff ANY reachable witness's head/round disagrees with the presented one
        wh = witness_heads()
        for n, w in wh.items():
            if w.get("log_hash") is None or "error" in w:
                continue
            if w["log_hash"] != presented_hash or w["round"] != presented_round:
                return True, n
        return False, None

    results = {}
    det, _ = audit(true_head, true_round); results["honest_presentation"] = {"detected": det}
    k = R // 3; old = log[R - 1 - k]
    det, by = audit(old["log_hash"], old["round"]); results["rollback"] = {"detected": det, "by_witness": by}
    t = max(1, R // 10); tr = log[R - 1 - t]
    det, by = audit(tr["log_hash"], tr["round"]); results["truncation"] = {"detected": det, "by_witness": by}
    forged = H(b"forged-head").hex()
    det, by = audit(forged, true_round); results["fork_forged_head"] = {"detected": det, "by_witness": by}

    wh = witness_heads()
    out = {"transport": "http-network", "rounds": R, "silos": silos,
           "broadcast_ok": broadcast_ok,
           "witness_state": {n: {"honest": w.get("honest"), "round": w.get("round"),
                                 "len": w.get("len"), "head": (w.get("log_hash") or "")[:16]}
                             for n, w in wh.items()},
           "results": results,
           "honest_witnesses_needed": 1}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"=== EF3 networked root-continuity ({R} rounds over HTTP) ===")
    print("witnesses:", {n: ("honest" if w.get("honest") else "dishonest") for n, w in wh.items()})
    print(f"{'scenario':>22} | detected")
    print("-" * 40)
    for k2, v in results.items():
        b = f" (by {v['by_witness']})" if v.get("by_witness") else ""
        print(f"{k2:>22} | {v['detected']}{b}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
