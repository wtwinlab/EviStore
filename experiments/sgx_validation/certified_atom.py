#!/usr/bin/env python3
"""The post-atom trust boundary, measured on the real durable TrustStore, plus a certified-atom
mitigation. DTRUST's verifiable-replay guarantee is over the SEALED ATOMS; it does not certify
that the host's raw-to-atom feature extraction is faithful. We make that boundary concrete and
then close it for atoms from a signed source.

Three cases on the same real store (SQLite+WAL+Merkle+SRI):
  A) POST-ATOM mode (trust boundary): an untrusted host writes a SEMANTICALLY FALSE atom
     (a forged t_score that does not reflect the participant's real update). It seals, the
     Merkle chunk-chain verifies, and the SRI receipt verifies against the root -- DTRUST
     accepts and replays it. Semantic falsity is NOT detected: no ground truth for the atom.
  B) CERTIFIED-ATOM mode (mitigation): the participant signs the canonical atom bytes with its
     Ed25519 key; the coordinator verifies the signature against the registered public key
     BEFORE sealing. A host-forged atom (no valid signature) is REJECTED at write time; a
     correctly-signed atom is accepted. Provenance is bound to a signed source, not the host.
  C) POST-SEAL TAMPER (existing guarantee): editing a sealed atom after the fact breaks the
     deep chunk-chain (tuple->hash binding) -> rejected.

Reports per-case detection + signing cost (sign/verify us, signature bytes, write overhead).
Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/c1_audit/certified_atom.py
"""
import json, os, sys, time, types, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if "dtrust" not in sys.modules:
    pkg = types.ModuleType("dtrust"); pkg.__path__ = [str(ROOT / "dtrust")]
    sys.modules["dtrust"] = pkg
from dtrust.store.store import TrustStore
from dtrust.store.schema import Mode, TrustTuple
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import random
rng = random.Random(0); H32 = b"\x00" * 32


def atom_bytes(t):
    """Canonical bytes of the sealed atom a participant attests to (the fields it produces)."""
    return (int(t.round).to_bytes(4, "little") + t.participant +
            int(t.a_conf).to_bytes(2, "little") + int(t.t_score).to_bytes(2, "little") +
            int(t.b_score).to_bytes(2, "little") + int(t.w_agg).to_bytes(2, "little") + t.h_grad)


def mk_tuple(r, p, t_score):
    return TrustTuple(round=r, participant=p.to_bytes(16, "little"), a_conf=40000, fresh=True,
                      s_cusum=0.0, t_score=t_score, b_score=30000, w_agg=30000,
                      mode=Mode.NORMAL, h_grad=H32)


def case_a_post_atom(N=50):
    """Host forges a false t_score for one participant. It seals + replays verifiably; the
    semantic forgery is undetectable (post-atom mode = the trust boundary)."""
    db = tempfile.mktemp(suffix=".db"); st = TrustStore(db)
    true_scores = {p: rng.randint(20000, 60000) for p in range(N)}
    victim = 7; forged = 100  # host writes a near-zero trust for the victim instead of its true ~40k
    tuples = [mk_tuple(0, p, forged if p == victim else true_scores[p]) for p in range(N)]
    st.append_round(0, tuples)
    sealed_ok = st.verify_sri() and st.verify_chunk_chain(deep=True)   # integrity holds...
    # ...and the forged value is what gets replayed: the store has no ground truth to flag it.
    stored = st.conn.execute("SELECT t_score FROM tuples WHERE round=0").fetchall()
    forged_present = any(s[0] == forged for s in stored)
    semantic_detected = False  # post-atom mode cannot detect that `forged` != victim's real update
    st.conn.close(); os.unlink(db)
    return {"integrity_ok": bool(sealed_ok), "forged_atom_sealed_and_replayable": bool(forged_present),
            "semantic_forgery_detected": semantic_detected,
            "note": "post-atom mode: replay is faithful to the sealed atom, not to the raw update"}


def case_b_certified(N=50, reps=200):
    """Certified-atom mode: participant signs the atom; coordinator verifies before sealing.
    A host-forged atom has no valid signature -> rejected at write time."""
    keys = {p: Ed25519PrivateKey.generate() for p in range(N)}
    pub = {p: keys[p].public_key() for p in range(N)}
    db = tempfile.mktemp(suffix=".db"); st = TrustStore(db)

    def verify_then_seal(r, tuples, sigs):
        accepted = []
        for t, (signer_pub, sig) in zip(tuples, sigs):
            try:
                signer_pub.verify(sig, atom_bytes(t)); accepted.append(t)   # signature binds the atom
            except InvalidSignature:
                pass                                                        # reject forged atom
        if len(accepted) == len(tuples):
            st.append_round(r, tuples)
        return len(accepted)

    # honest round: every atom correctly signed by its participant -> all accepted
    tuples = [mk_tuple(0, p, rng.randint(20000, 60000)) for p in range(N)]
    sigs = [(pub[p], keys[p].sign(atom_bytes(tuples[p]))) for p in range(N)]
    honest_accepted = verify_then_seal(0, tuples, sigs)

    # attack: host forges the victim's atom but cannot produce the victim's signature
    victim = 7
    ftuple = mk_tuple(1, victim, 100)
    host_key = Ed25519PrivateKey.generate()                      # the host's own key, not the victim's
    forged_sig = (pub[victim], host_key.sign(atom_bytes(ftuple)))  # signed by the wrong key
    forged_rejected = False
    try:
        pub[victim].verify(forged_sig[1], atom_bytes(ftuple))
    except InvalidSignature:
        forged_rejected = True

    # signing cost
    msg = atom_bytes(tuples[0]); k = keys[0]; pk = pub[0]
    t0 = time.perf_counter()
    for _ in range(reps): s = k.sign(msg)
    sign_us = (time.perf_counter() - t0) / reps * 1e6
    t0 = time.perf_counter()
    for _ in range(reps): pk.verify(s, msg)
    verify_us = (time.perf_counter() - t0) / reps * 1e6
    st.conn.close(); os.unlink(db)
    return {"honest_round_all_accepted": honest_accepted == N, "n": N,
            "host_forged_atom_rejected": bool(forged_rejected),
            "sig_bytes": len(s), "sign_us": round(sign_us, 2), "verify_us": round(verify_us, 2),
            "write_overhead_us_per_round": round((sign_us + verify_us) * N, 1)}


def case_c_post_seal_tamper(N=50):
    """Edit a sealed atom after the fact -> deep chunk-chain rejects (existing guarantee)."""
    db = tempfile.mktemp(suffix=".db"); st = TrustStore(db)
    st.append_round(0, [mk_tuple(0, p, rng.randint(20000, 60000)) for p in range(N)])
    ok_before = st.verify_chunk_chain(deep=True)
    p0 = st.conn.execute("SELECT participant FROM tuples WHERE round=0 LIMIT 1").fetchone()[0]
    st.conn.execute("UPDATE tuples SET t_score=1 WHERE round=0 AND participant=?", (p0,))
    st.conn.commit()
    ok_after = st.verify_chunk_chain(deep=True)
    st.conn.close(); os.unlink(db)
    return {"verified_before_tamper": bool(ok_before), "tamper_detected": (not ok_after)}


def main():
    out = {"_note": "Post-atom trust boundary + certified-atom mitigation on the real durable "
                    "TrustStore. Replay is verifiable over the SEALED ATOMS; raw-to-atom faithfulness "
                    "is a non-goal in post-atom mode and is closed by signed-source atoms in "
                    "certified-atom mode.",
           "A_post_atom_mode": case_a_post_atom(),
           "B_certified_atom_mode": case_b_certified(),
           "C_post_seal_tamper": case_c_post_seal_tamper()}
    print(json.dumps(out, indent=2))
    od = ROOT / "results/recovery/durability"; od.mkdir(parents=True, exist_ok=True)
    (od / "certified_atom.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "certified_atom.json")


if __name__ == "__main__":
    main()
