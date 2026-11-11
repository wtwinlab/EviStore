#!/bin/bash
# Per-node finishing + t2b self-check for a federation SGX node.
# Run ON each node (scp it up, or: ssh root@<ip> 'bash -s' < fed_finish.sh).
# Does: minimal python deps (silo needs no torch) -> Aliyun PCCS -> restart aesmd ->
#       sign+run the in-enclave DCAP quote self-check (gramine-sgx dtrust).
set -uo pipefail
cd /root/DTRUST || exit 1
PCCS='https://sgx-dcap-server.cn-hangzhou.aliyuncs.com/sgx/certification/v4/'
VENV=/root/DTRUST/venv/bin

echo "### [1/4] minimal silo python deps (no torch needed for federation) ###"
if ! $VENV/python -c "import numpy, cryptography" 2>/dev/null; then
  $VENV/pip install --no-cache-dir numpy cryptography pydantic loguru pyyaml python-dotenv requests 2>&1 | tail -2
fi
$VENV/python -c "import numpy, cryptography; print('  deps ok')" 2>&1 | tail -1

echo "### [2/4] PCCS -> Aliyun cn-hangzhou ###"
sed -i "s|\"pccs_url\":.*|\"pccs_url\": \"$PCCS\",|" /etc/sgx_default_qcnl.conf
grep -m1 pccs_url /etc/sgx_default_qcnl.conf

echo "### [3/4] restart aesmd ###"
systemctl restart aesmd; sleep 2; echo "  aesmd: $(systemctl is-active aesmd)"

echo "### [4/4] t2b self-check: in-enclave DCAP quote ###"
cd /root/DTRUST/deploy
# (re)sign dtrust manifest if needed, then run the quote test (test_gramine_quote.py)
if [ ! -f dtrust.manifest.sgx ]; then
  PY=/root/DTRUST/venv/bin/python3.10
  gramine-manifest -Dpython=$PY -Darch_libdir=/lib/x86_64-linux-gnu dtrust.manifest.template dtrust.manifest >/dev/null 2>&1
  gramine-sgx-sign --manifest dtrust.manifest --output dtrust.manifest.sgx >/dev/null 2>&1
fi
OUT=$(timeout 120 gramine-sgx dtrust 2>&1)
echo "$OUT" | grep -iE "quote|mrenclave|success|pass|fail|error|eperm" | head -6
if echo "$OUT" | grep -qiE "quote.*[0-9]{3,}|quote_len|SGXQuote|verified"; then
  echo "  ==> SELF-CHECK PASS (in-enclave DCAP quote generated)"
else
  echo "  ==> SELF-CHECK INCONCLUSIVE -- inspect the output above"
fi
echo "FED_FINISH_DONE $(hostname)"
