"""EF3 networked witness — runs on a silo, receives broadcast heads over HTTP.

An honest witness accepts only monotonic, prev-matching chain extensions (the same
rule as fed_witness.Witness); a dishonest witness (--dishonest) accepts anything
(models a colluding/compromised silo). At audit time the coordinator's auditor GETs
each witness's latest head and compares it to what the (possibly malicious) host
presents — with >=1 honest witness, rollback/truncation/fork is detected.

    python3 fed_witness_server.py --port 8137 [--dishonest]
stdlib only. Heads travel as hex strings over the wire.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8137)
ap.add_argument("--dishonest", action="store_true")
ap.add_argument("--name", default="S?")
A = ap.parse_args()

CHAIN = []  # accepted heads in order; each {round, root, prev_log, log_hash} (hex strs)


def receive(m):
    if A.dishonest:                       # colluding/compromised: accept anything
        CHAIN.append(m); return "accept(dishonest)"
    r = m["round"]
    if CHAIN and r <= CHAIN[-1]["round"]:
        return "reject(non-monotonic)"
    if CHAIN and m["prev_log"] != CHAIN[-1]["log_hash"]:
        return "reject(broken-chain)"
    CHAIN.append(m)
    return "accept"


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        m = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/head":
            self._json({"status": receive(m), "len": len(CHAIN)})
        elif self.path == "/reset":
            CHAIN.clear(); self._json({"status": "reset"})
        else:
            self._json({"error": "unknown"}, 404)

    def do_GET(self):
        if self.path == "/head":
            last = CHAIN[-1] if CHAIN else None
            self._json({"name": A.name, "honest": not A.dishonest,
                        "log_hash": last["log_hash"] if last else None,
                        "round": last["round"] if last else -1, "len": len(CHAIN)})
        else:
            self._json({"error": "unknown"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"witness {A.name} honest={not A.dishonest} listening :{A.port}", flush=True)
    HTTPServer(("0.0.0.0", A.port), Handler).serve_forever()
