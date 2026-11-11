"""EF3 - distributed root-continuity across federation witnesses (locally testable).

The coordinator appends a hash-chained root-log entry per anchor and BROADCASTS each
new head to the silo witnesses. Each witness stores heads it has seen, accepting only
a monotonic, prev-matching extension. At audit time the auditor collects every
witness's latest head and compares it against the ledger/head the (possibly malicious)
host of the coordinator's storage presents.

Property: with >=1 honest witness, a host that rolls back, truncates, or forks the
coordinator's ledger is detected -- WITHOUT any external transparency log, because the
federation's own silos are the witnesses. This is the distributed upgrade of the local
M2 root-log.

This module simulates the protocol end-to-end (coordinator + N witnesses + a malicious
host) so the protocol is validated locally; on the real federation the same logic runs
with heads shipped over the network. stdlib only.

    python fed_witness.py
"""
import hashlib
import json
import struct
from pathlib import Path

EPOCH = 1


def H(*bs):
    h = hashlib.sha256()
    for b in bs:
        h.update(b)
    return h.digest()


class Coordinator:
    """Builds the hash-chained root sequence and broadcasts heads."""
    def __init__(self):
        self.roots = []      # root_r per anchor (stand-in: a per-round digest)
        self.log = []        # entries (round, epoch, root, prev_log_hash)
        self.prev_log = b"\x00" * 32

    def anchor(self, round_no, round_payload):
        root = H(round_payload)                       # ledger Merkle root at this round
        entry = struct.pack("<I I 32s 32s", round_no, EPOCH, root, self.prev_log)
        log_hash = H(entry)
        self.roots.append(root)
        self.log.append({"round": round_no, "root": root,
                         "prev_log": self.prev_log, "log_hash": log_hash})
        self.prev_log = log_hash
        return {"round": round_no, "root": root, "log_hash": log_hash,
                "prev_log": self.log[-1]["prev_log"]}

    def head(self):
        return self.log[-1]["log_hash"] if self.log else b"\x00" * 32


class Witness:
    """A silo that stores broadcast heads, accepting only monotonic, chained extensions."""
    def __init__(self, name, honest=True):
        self.name = name
        self.honest = honest
        self.chain = []     # accepted heads in order

    def receive(self, head_msg):
        # honest witness verifies the chain link + monotonic round; a dishonest one
        # accepts anything (or is simply unavailable).
        if not self.honest:
            return
        r = head_msg["round"]
        if self.chain and r <= self.chain[-1]["round"]:
            return  # reject non-monotonic
        if self.chain and head_msg["prev_log"] != self.chain[-1]["log_hash"]:
            return  # reject broken chain
        self.chain.append(head_msg)

    def latest_head(self):
        return self.chain[-1]["log_hash"] if self.chain else None

    def latest_round(self):
        return self.chain[-1]["round"] if self.chain else -1


def audit(presented_head, presented_round, witnesses):
    """Detected iff any honest witness's head/round disagrees with what the host presents."""
    for w in witnesses:
        h = w.latest_head()
        if h is None:
            continue
        if h != presented_head or w.latest_round() != presented_round:
            return True, w.name      # mismatch with a witness -> tamper/rollback detected
    return False, None


def run():
    R = 20
    coord = Coordinator()
    # federation witnesses: 2 honest silos, 1 dishonest (colludes/unavailable)
    W = [Witness("S1", honest=True), Witness("S2", honest=True), Witness("S3", honest=False)]
    # normal operation: coordinator anchors each round and broadcasts head to witnesses
    for r in range(R):
        msg = coord.anchor(r, f"round-{r}-payload".encode())
        for w in W:
            w.receive(msg)
    true_head, true_round = coord.head(), R - 1

    results = {}

    # honest presentation: host shows the real latest head -> no detection
    det, _ = audit(true_head, true_round, W)
    results["honest_presentation"] = {"detected": det}  # expect False

    # attack 1: ROLLBACK -- host presents an older complete head (round R-1-k)
    k = R // 3
    old = coord.log[R - 1 - k]
    det, by = audit(old["log_hash"], old["round"], W)
    results["rollback"] = {"detected": det, "by_witness": by}

    # attack 2: TRUNCATION -- host drops the suffix, presents head at round R-1-t
    t = max(1, R // 10)
    tr = coord.log[R - 1 - t]
    det, by = audit(tr["log_hash"], tr["round"], W)
    results["truncation"] = {"detected": det, "by_witness": by}

    # attack 3: FORK -- host forges a different head at the latest round
    forged = H(b"forged-head")
    det, by = audit(forged, true_round, W)
    results["fork_forged_head"] = {"detected": det, "by_witness": by}

    # honest-witness threshold: how many honest witnesses needed? (>=1)
    only_dishonest = [Witness("Sx", honest=False)]
    det_none, _ = audit(old["log_hash"], old["round"], only_dishonest)

    out = {"rounds": R, "witnesses": {w.name: w.honest for w in W},
           "results": results,
           "with_zero_honest_witnesses_rollback_detected": det_none,
           "honest_witnesses_needed": 1}
    Path("results/recovery/federation").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/recovery/federation/ef3_witness.json", "w"),
              indent=2, default=str)

    print(f"=== EF3 distributed root-continuity ({R} rounds, witnesses "
          f"{[w.name+('(honest)' if w.honest else '(dishonest)') for w in W]}) ===")
    print(f"{'scenario':>22} | detected")
    print("-" * 36)
    for k2, v in results.items():
        b = f" (by {v['by_witness']})" if v.get("by_witness") else ""
        print(f"{k2:>22} | {v['detected']}{b}")
    print(f"\nwith ZERO honest witnesses, rollback detected: {det_none} "
          f"(=> need >=1 honest witness)")
    print("Net: with >=1 honest silo witness, rollback/truncation/fork of the "
          "coordinator ledger is detected -- no external transparency log required.")
    print("wrote results/recovery/federation/ef3_witness.json")


if __name__ == "__main__":
    run()
