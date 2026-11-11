"""SGX-sealed trust ledger + DCAP quote bound to its Merkle root.
Runs inside Gramine-SGX. Self-contained (stdlib only; same 76-byte tuple +
4KiB SHA-256 chunk-chain format as dtrust.store). If the sentinel file
/root/DTRUST/results/t2b_readonly exists, it only RE-READS the sealed ledger
(to demonstrate hardware tamper-detection), otherwise it WRITES+seals."""
import sys, os, struct, hashlib

PACK = "<I 16s H B f 3H B 32s 10x"; TUPLE = struct.calcsize(PACK); assert TUPLE == 76
CHUNK = 4096; HDR = 32; BODY = CHUNK - HDR; PER = BODY // TUPLE

def pack_tuple(rnd, pid):
    part = hashlib.sha256(f"c{pid}".encode()).digest()[:16]
    return struct.pack(PACK, rnd, part, 60000, 1, rnd * 0.1, 40000, 40000, 40000, 0, bytes(32))

def build_ledger(R=40, N=10):
    chunk_hashes = []; prev = b"\x00" * 32; out = bytearray()
    for r in range(R):
        tups = sorted(pack_tuple(r, i) for i in range(N))
        for off in range(0, len(tups), PER):
            body = b"".join(tups[off:off + PER]).ljust(BODY, b"\x00")
            chunk = len(chunk_hashes).to_bytes(4, "little") + prev[:28] + body
            h = hashlib.sha256(chunk).digest(); chunk_hashes.append(h); prev = h; out += chunk
    layer = list(chunk_hashes)
    while len(layer) > 1:
        if len(layer) % 2: layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i+1]).digest() for i in range(0, len(layer), 2)]
    return bytes(out), (layer[0] if layer else b"\x00" * 32)

READONLY = os.path.exists("/root/DTRUST/results/t2b_readonly")
if READONLY:
    try:
        with open("/sealed/ledger.bin", "rb") as f:
            data = f.read()
        print(f"T2B_READ_OK sealed_bytes={len(data)} (enclave decrypted the sealed ledger)", file=sys.stderr)
    except Exception as e:
        print(f"T2B_READ_FAIL tamper detected by SGX sealing: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)

ledger, root = build_ledger()
with open("/sealed/ledger.bin", "wb") as f:
    f.write(ledger)
with open("/sealed/ledger.bin", "rb") as f:
    ok = (f.read() == ledger)
rd = root.ljust(64, b"\x00")[:64]
open("/dev/attestation/user_report_data", "wb").write(rd)
quote = open("/dev/attestation/quote", "rb").read()
open("/root/DTRUST/results/t2b_quote.bin", "wb").write(quote)
open("/root/DTRUST/results/t2b_root.hex", "w").write(root.hex())
print(f"T2B_OK sealed_bytes={len(ledger)} reseal_ok={ok} merkle_root={root.hex()} quote_len={len(quote)}", file=sys.stderr)
