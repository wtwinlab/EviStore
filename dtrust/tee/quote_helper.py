"""
Minimal SGX quote generation helper — runs INSIDE a Gramine SGX enclave.

Usage (called by SubprocessGramineSGXQuoteGenerator):
  gramine-sgx dtrust_attest

Protocol (via stdin/stdout):
  stdin:  64 bytes of challenge/report_data (binary)
  stdout: raw DCAP quote bytes (binary)
  stderr: diagnostics only

The manifest dtrust_attest.manifest must have:
  sgx.remote_attestation = "dcap"
  libos.entrypoint = "/usr/bin/python3.10"
  loader.argv = ["python3.10", "/path/to/quote_helper.py"]
"""

import sys
import os


def main() -> None:
    # 从 stdin 读 64 字节 challenge（不足则补零）
    raw = sys.stdin.buffer.read(64)
    report_data = raw[:64].ljust(64, b"\x00")

    attest_dir = "/dev/attestation"
    type_file = attest_dir + "/attestation_type"
    report_data_file = attest_dir + "/user_report_data"
    quote_file = attest_dir + "/quote"

    # 检查是否在 Gramine DCAP enclave 里
    try:
        with open(type_file) as f:
            mode = f.read().strip()
        if mode != "dcap":
            print(f"ERROR: attestation_type={mode}, expected dcap", file=sys.stderr)
            sys.exit(1)
    except OSError as e:
        print(f"ERROR: cannot read {type_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # 写入 report_data（绑定 challenge 到 quote）
    try:
        with open(report_data_file, "wb") as f:
            f.write(report_data)
    except OSError as e:
        print(f"ERROR: cannot write {report_data_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # 读取硬件签名的 DCAP quote
    try:
        with open(quote_file, "rb") as f:
            quote = f.read()
    except OSError as e:
        print(f"ERROR: cannot read {quote_file}: {e}", file=sys.stderr)
        sys.exit(1)

    if len(quote) < 432:  # sgx_quote_header(48) + sgx_report_body(384)
        print(f"ERROR: quote too short ({len(quote)} bytes)", file=sys.stderr)
        sys.exit(1)

    # 输出 raw quote 到 stdout
    sys.stdout.buffer.write(quote)
    sys.stdout.buffer.flush()
    print(f"quote_len={len(quote)}", file=sys.stderr)


if __name__ == "__main__":
    main()
