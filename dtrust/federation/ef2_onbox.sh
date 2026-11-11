#!/bin/bash
set -e
ID=$1
cd /root/DTRUST/deploy
mkdir -p /root/DTRUST/federation /root/DTRUST/ef2_sealed
echo "$ID" > /root/DTRUST/federation/silo_id.txt
python3 - << 'PY'
t=open("t2b.manifest.template").read()
t=t.replace('loader.argv = ["python3.10", "/root/DTRUST/t2b_seal_ledger.py"]',
            'loader.argv = ["python3.10", "/root/DTRUST/dtrust/federation/silo_seal.py"]\nloader.insecure__use_host_env = true')
t=t.replace('t2b_sealed','ef2_sealed')
t=t.replace('{ path = "/root/DTRUST/t2b_seal_ledger.py", uri = "file:/root/DTRUST/t2b_seal_ledger.py" }',
            '{ path = "/root/DTRUST/federation/silo_id.txt", uri = "file:/root/DTRUST/federation/silo_id.txt" }')
t=t.replace('"file:/root/DTRUST/t2b_seal_ledger.py"','"file:/root/DTRUST/federation/silo_id.txt"')
open("ef2.manifest.template","w").write(t)
PY
PY=/root/DTRUST/venv/bin/python3.10
gramine-manifest -Dpython=$PY -Darch_libdir=/lib/x86_64-linux-gnu ef2.manifest.template ef2.manifest >/dev/null 2>&1
gramine-sgx-sign -k /root/DTRUST/deploy/enclave_signing.pem --manifest ef2.manifest --output ef2.manifest.sgx 2>&1 | grep -iA1 "^Measurement" | tail -1
timeout 90 gramine-sgx ef2 2>&1 | grep -E "EF2_SEAL" | head -1
