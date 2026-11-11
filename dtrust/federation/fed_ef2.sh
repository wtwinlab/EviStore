#!/bin/bash
# EF2 driver (run from the orchestrator/Mac): cross-enclave sealing isolation matrix.
# For each silo: bake a distinct silo_id.txt (a TRUSTED file -> distinct MRENCLAVE ->
# distinct sealing key), build a silo_seal manifest, seal evidence. Then cross-copy
# each silo's sealed file to every other node and run `read` -> who-can-decrypt matrix.
#
#   bash fed_ef2.sh "C=121.199.66.60 S1=8.136.35.117 S2=8.136.33.103 S3=8.136.47.218"
set -uo pipefail
declare -A NODE
for kv in $1; do NODE[${kv%%=*}]=${kv##*=}; done
NAMES=(C S1 S2 S3)
REMOTE=/root/DTRUST
PY=/root/DTRUST/venv/bin/python3.10
STAGE=/tmp/ef2_stage; mkdir -p "$STAGE"

build_and_seal() {   # $1 = node name
  local n=$1 ip=${NODE[$1]}
  ssh -o ConnectTimeout=20 root@"$ip" "cd $REMOTE/deploy
    mkdir -p $REMOTE/federation $REMOTE/ef2_sealed
    echo '$n' > $REMOTE/federation/silo_id.txt
    # clone coord template -> silo_seal: swap entrypoint, sealed dir, and add the
    # silo_id file to trusted_files so the MRENCLAVE differs per silo.
    sed -e 's|sgx_coordinator.py|dtrust/federation/silo_seal.py|g' \
        -e 's|coord_sealed|ef2_sealed|g' \
        -e 's|loader.argv = \[\"python3.10\", \"/root/DTRUST/dtrust/federation/silo_seal.py\"\]|loader.argv = [\"python3.10\", \"/root/DTRUST/dtrust/federation/silo_seal.py\", \"seal\"]|' \
        coord.manifest.template > ef2.manifest.template
    # add silo_id.txt to trusted_files (measured -> distinct MRENCLAVE)
    sed -i 's|sgx.trusted_files = \[|sgx.trusted_files = [\n  \"file:/root/DTRUST/federation/silo_id.txt\",|' ef2.manifest.template
    gramine-manifest -Dpython=$PY -Darch_libdir=/lib/x86_64-linux-gnu ef2.manifest.template ef2.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest ef2.manifest --output ef2.manifest.sgx 2>&1 | grep -i measurement
    rm -f $REMOTE/ef2_sealed/*
    gramine-sgx ef2 2>&1 | grep EF2_SEAL"
  # pull this silo's sealed evidence (GRAFS_PF ciphertext under MRENCLAVE_$n)
  scp -o ConnectTimeout=20 root@"$ip":$REMOTE/ef2_sealed/evidence.bin "$STAGE/evidence_$n.bin" 2>/dev/null
}

read_foreign() {   # reader=$1 owner=$2
  local r=$1 o=$2 ip=${NODE[$1]}
  scp -o ConnectTimeout=20 "$STAGE/evidence_$o.bin" root@"$ip":$REMOTE/ef2_sealed/foreign.bin 2>/dev/null
  ssh -o ConnectTimeout=20 root@"$ip" "cd $REMOTE/deploy
    # rebuild manifest in 'read' mode for this reader (its own silo_id is already set)
    sed 's|silo_seal.py\", \"seal\"|silo_seal.py\", \"read\"|' ef2.manifest.template > ef2r.manifest.template
    gramine-manifest -Dpython=$PY -Darch_libdir=/lib/x86_64-linux-gnu ef2r.manifest.template ef2r.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest ef2r.manifest --output ef2r.manifest.sgx >/dev/null 2>&1
    gramine-sgx ef2r 2>&1 | grep EF2_READ"
}

echo '### EF2: seal on each node (distinct MRENCLAVE) ###'
for n in "${NAMES[@]}"; do [ -n "${NODE[$n]:-}" ] && { echo "-- $n (${NODE[$n]}) --"; build_and_seal "$n"; }; done

echo; echo '### EF2: cross-read matrix (reader x owner) ###'
printf '%6s' "rd\\own"; for o in "${NAMES[@]}"; do [ -n "${NODE[$o]:-}" ] && printf '%8s' "$o"; done; echo
for r in "${NAMES[@]}"; do
  [ -z "${NODE[$r]:-}" ] && continue
  printf '%6s' "$r"
  for o in "${NAMES[@]}"; do
    [ -z "${NODE[$o]:-}" ] && continue
    res=$(read_foreign "$r" "$o")
    if echo "$res" | grep -q DECRYPTED; then printf '%8s' "OK"; else printf '%8s' "iso"; fi
  done; echo
done
echo '(diagonal=OK self-read; off-diagonal=iso isolated. EF2 isolation holds iff off-diagonal all iso.)'
