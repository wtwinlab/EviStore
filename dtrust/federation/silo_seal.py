"""EF2 - cross-enclave sealing isolation (runs INSIDE a Gramine-SGX enclave).

Each silo runs an enclave with a DISTINCT MRENCLAVE (a per-silo id file is in the
manifest's trusted_files, so the measured manifest -- hence the MRENCLAVE-derived
sealing key -- differs per silo). This script does two things:

  seal:  seal this silo's evidence to the encrypted-files mount (/sealed), keyed to
         THIS enclave's MRENCLAVE, and read it back (reseal_ok).
  read:  try to read a file presented at /sealed that was sealed by ANOTHER silo.
         If the foreign file was sealed under a different MRENCLAVE, the encrypted-
         files layer cannot derive its key and the read FAILS -- that failure is the
         isolation result.

The host-side driver copies each silo's sealed file to the others and runs `read` on
each, producing a 4x4 "who-can-decrypt-whose-seal" matrix (diagonal ok, off-diagonal
fail). stdlib only.

    gramine-sgx silo_seal seal    # seals /sealed/evidence.bin
    gramine-sgx silo_seal read    # tries to read /sealed/foreign.bin
"""
import hashlib
import os
import sys

SILO_ID = open("/root/DTRUST/federation/silo_id.txt").read().strip() \
    if os.path.exists("/root/DTRUST/federation/silo_id.txt") else "unknown"


def evidence_blob():
    # deterministic per-silo evidence (stands in for the silo's sealed Eq.8 evidence)
    h = hashlib.sha256(f"evidence::{SILO_ID}".encode()).digest()
    return (f"SILO={SILO_ID}\n".encode() + h * 64)  # ~2 KB


def do_seal():
    blob = evidence_blob()
    with open("/sealed/evidence.bin", "wb") as f:
        f.write(blob)
    with open("/sealed/evidence.bin", "rb") as f:
        ok = (f.read() == blob)
    digest = hashlib.sha256(blob).hexdigest()[:16]
    print(f"EF2_SEAL silo={SILO_ID} sealed_bytes={len(blob)} reseal_ok={ok} "
          f"evidence_digest={digest}", file=sys.stderr)
    # EF1: also emit a DCAP quote binding report_data to this silo's evidence, so the
    # coordinator can verify the silo's enclave (mutual attestation).
    try:
        import base64
        rd = hashlib.sha256(b"EF1:" + SILO_ID.encode() + blob).digest().ljust(64, b"\0")[:64]
        open("/dev/attestation/user_report_data", "wb").write(rd)
        q = open("/dev/attestation/quote", "rb").read()
        print(f"EF1_QUOTE silo={SILO_ID} quote_len={len(q)} rd={rd[:8].hex()} "
              f"b64={base64.b64encode(q).decode()}", file=sys.stderr)
    except Exception as e:
        print(f"EF1_QUOTE silo={SILO_ID} FAIL {type(e).__name__}:{e}", file=sys.stderr)


def do_read():
    # Read the sealed file at /sealed/evidence.bin. Gramine protected files bind the
    # ciphertext to BOTH the MRENCLAVE-derived key AND the in-enclave path, so a foreign
    # silo's file must be staged at the SAME path (/sealed/evidence.bin) it was sealed at;
    # decryption then succeeds iff this enclave's MRENCLAVE matches the one that sealed it.
    try:
        with open("/sealed/evidence.bin", "rb") as f:
            data = f.read()
        head = data.split(b"\n", 1)[0].decode(errors="replace")
        print(f"EF2_READ reader={SILO_ID} result=DECRYPTED bytes={len(data)} "
              f"plaintext_head={head}", file=sys.stderr)
    except Exception as e:
        print(f"EF2_READ reader={SILO_ID} result=FAIL_ISOLATED "
              f"err={type(e).__name__}:{str(e)[:70]}", file=sys.stderr)


if __name__ == "__main__":
    # ONE manifest/argv for both seal and read (loader.argv is measured -> changing it
    # changes the MRENCLAVE and the sealing key). The mode is chosen from a host env var
    # (NOT measured; manifest sets loader.insecure__use_host_env=true), so MRENCLAVE is
    # stable across seal and read. read mode does NOT reseal (that would overwrite the
    # foreign file the host staged at /sealed/evidence.bin).
    if os.environ.get("EF2_READ"):
        do_read()
    else:
        do_seal()
