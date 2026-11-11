#!/usr/bin/env python3
"""The attested-extraction layer. Replay-critical atoms are
produced by a DETERMINISTIC extractor Ext_v (the paper's real per-channel LLR math) that binds raw_hash,
atom_hash, rule/quantization version, and extractor identity into an Ed25519 certificate; the coordinator
verifies the certificate before sealing. This closes host-side atom forgery and extraction tampering for
replay-critical atoms -- it does NOT certify that the raw source is truthful.

Torch-free (stdlib + cryptography + the pure-math _llr/quantize_llr loaded by path), so the same Ext_v can
run inside a Gramine-SGX extractor enclave (the no-`import dtrust` / no-torch rule). Ext_v reuses the
production extraction math verbatim (dtrust/trust/_llr.py + dtrust/store/schema.quantize_llr) -- the paper's
algorithm, not a reimplementation.
"""
import hashlib, importlib.util as ilu, struct, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

ROOT = Path(__file__).resolve().parents[3]

def _load(name, rel):
    s = ilu.spec_from_file_location(name, ROOT / rel); m = ilu.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m
_LLR = _load("_dtrust_llr", "dtrust/trust/_llr.py")          # llr_att/grad/perf/seal -- pure math
_SCHEMA = _load("_dtrust_schema", "dtrust/store/schema.py")  # quantize_llr, LLR_SCALES, LLR_CHANNELS

def H(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(p if isinstance(p, (bytes, bytearray)) else str(p).encode())
    return h.digest()

# canonical atom-kind ids and channel index for quantization
KIND = {"llr_grad": 1, "llr_perf": 2, "seal_ok": 3, "llr_att": 4}
CH = {"llr_grad": 1, "llr_perf": 2, "seal_ok": 4, "llr_att": 0}   # index into LLR_SCALES
# default declared extraction rule parameters (rule_version v1) -- the deterministic Ext_v semantics
RULE_V1 = {
    "llr_grad": dict(a0=8.0, b0=2.0, a1=2.0, b1=8.0, clip=2.0),    # Beta(honest)/Beta(malicious)
    "llr_perf": dict(p0=0.95, p1=0.60, clip=2.0),                  # Bernoulli success
    "seal_ok":  dict(ok=0.1, violation=-20.0, clip_lo=-20.0, clip_hi=2.0),
    "llr_att":  dict(mu0=0.0, sigma0=1.0, mu1=2.0, clip=2.0),
}
def rule_hash(kind, params): return H("rule_v1", kind, *sorted(f"{k}={v}" for k, v in params.items()))
QUANT_VERSION = H("quant", *[f"{c}={s}" for c, s in zip(_SCHEMA.LLR_CHANNELS, _SCHEMA.LLR_SCALES)])


def _clip(x, lo, hi): return max(lo, min(hi, x))

def ext_v(kind, raw_value, params):
    """Deterministic extraction: canonical raw value -> int16 LLR atom, using the REAL _llr math."""
    if kind == "llr_grad":
        x = _clip(_LLR.llr_grad(float(raw_value), params["a0"], params["b0"], params["a1"], params["b1"]),
                  -params["clip"], params["clip"])
    elif kind == "llr_perf":
        x = _clip(_LLR.llr_perf(int(raw_value), params["p0"], params["p1"]), -params["clip"], params["clip"])
    elif kind == "seal_ok":
        x = _clip(_LLR.llr_seal(int(raw_value), {"ok": params["ok"], "violation": params["violation"]}),
                  params["clip_lo"], params["clip_hi"])
    elif kind == "llr_att":
        x = _clip(_LLR.llr_att(float(raw_value), params["mu0"], params["sigma0"], params["mu1"]),
                  -params["clip"], params["clip"])
    else:
        raise ValueError(f"unknown atom_kind {kind}")
    return int(_SCHEMA.quantize_llr(x, CH[kind]))               # int16 quantized atom


def canonical_raw(kind, round_, participant, raw_value):
    """Deterministic bytes of the canonical raw payload that the cert's raw_hash commits to."""
    return H("raw", kind, struct.pack("<I", int(round_)), participant, repr(float(raw_value)))

def atom_bytes_of(kind, llr_int):
    return struct.pack("<hB", int(llr_int), KIND[kind])        # int16 LLR + kind tag


@dataclass
class ExtractionCertificate:
    cell_id: bytes; extractor_id: bytes; extractor_mrenclave: bytes; extractor_mrsigner: bytes
    extraction_rule_version: bytes; quantization_version: bytes; atom_kind: str
    raw_hash: bytes; atom_hash: bytes; attestation_record_hash: Optional[bytes]
    coordinator_challenge: bytes; nonce: int; issued_at_ns: int
    policy_family_id: bytes = b""; round: int = 0; participant_id: bytes = b""; root_epoch: bytes = b""
    cert_sig: bytes = b""
    def body(self):
        # the signed body binds routing metadata (cell/round/participant/family/kind), the raw+atom
        # hashes, the rule+quantization versions, the extractor identity, and the root epoch + freshness.
        return H("cert_v2", self.cell_id, self.extractor_id, self.extractor_mrenclave,
                 self.extractor_mrsigner, self.extraction_rule_version, self.quantization_version,
                 self.atom_kind, self.raw_hash, self.atom_hash, self.attestation_record_hash or b"",
                 self.policy_family_id, self.participant_id, self.root_epoch,
                 self.coordinator_challenge, struct.pack("<QQQ", self.round, self.nonce, self.issued_at_ns))
    def cert_hash(self): return H("certhash", self.body(), self.cert_sig)


@dataclass
class CertifiedAtom:
    cell_id: bytes; round: int; participant_id: bytes; atom_kind: str; atom_bytes: bytes
    raw_hash: bytes; extraction_certificate_hash: bytes; policy_family_id: bytes; rule_version: bytes


class Extractor:
    """An attested extractor. mrenclave/mrsigner are placeholders locally; the real MRENCLAVE is bound by
    the DCAP quote on SGX-DCAP."""
    def __init__(self, mrenclave=b"\x9c\xa1" * 16, mrsigner=b"\x01\x1d" * 16, seed=b"extractor-v1"):
        self.key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed).digest())
        self.pub = self.key.public_key()
        self.extractor_id = H("ext_id", self._pub_bytes())
        self.mrenclave = mrenclave; self.mrsigner = mrsigner

    def _pub_bytes(self):
        from cryptography.hazmat.primitives import serialization
        return self.pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def extract(self, round_, participant, kind, raw_value, family_id, challenge, nonce,
                params=None, att_record_hash=None, now_ns=None, root_epoch=b""):
        params = params or RULE_V1[kind]
        cell_id = H("cell", family_id, struct.pack("<I", round_), participant, kind)
        raw = canonical_raw(kind, round_, participant, raw_value)
        llr = ext_v(kind, raw_value, params)
        ab = atom_bytes_of(kind, llr)
        cert = ExtractionCertificate(
            cell_id=cell_id, extractor_id=self.extractor_id, extractor_mrenclave=self.mrenclave,
            extractor_mrsigner=self.mrsigner, extraction_rule_version=rule_hash(kind, params),
            quantization_version=QUANT_VERSION, atom_kind=kind, raw_hash=H(raw), atom_hash=H(ab),
            attestation_record_hash=att_record_hash, coordinator_challenge=challenge, nonce=nonce,
            issued_at_ns=now_ns if now_ns is not None else 1,
            policy_family_id=family_id, round=round_, participant_id=participant, root_epoch=root_epoch)
        cert.cert_sig = self.key.sign(cert.body())
        atom = CertifiedAtom(cell_id=cell_id, round=round_, participant_id=participant, atom_kind=kind,
                             atom_bytes=ab, raw_hash=H(raw), extraction_certificate_hash=cert.cert_hash(),
                             policy_family_id=family_id, rule_version=rule_hash(kind, params))
        return atom, cert, raw, llr


def coordinator_verify(atom, cert, canonical_raw_payload, *, expected, registered_pubkeys,
                       accepted_mrenclaves, accepted_rule_versions, fresh_challenge, fresh_root_epoch,
                       seen_nonces):
    """Pre-seal verification. `expected` = the cell the coordinator is sealing:
    {family_id, round, participant, atom_kind}. Binds the atom's routing metadata to the SIGNED cert via a
    recomputed cell_id, so a valid cert cannot be transplanted to another cell/round/participant/kind.
    Returns (accepted, reason)."""
    # 0: recompute the expected cell_id from the coordinator's own context and bind atom+cert to it
    exp_cell = H("cell", expected["family_id"], struct.pack("<I", expected["round"]),
                 expected["participant"], expected["atom_kind"])
    if cert.cell_id != exp_cell or atom.cell_id != exp_cell: return False, "cell_id_mismatch"
    if cert.atom_kind != expected["atom_kind"] or atom.atom_kind != expected["atom_kind"]:
        return False, "atom_kind_mismatch"
    if cert.policy_family_id != expected["family_id"] or atom.policy_family_id != expected["family_id"]:
        return False, "family_mismatch"
    if cert.round != expected["round"]: return False, "round_mismatch"
    if cert.participant_id != expected["participant"] or atom.participant_id != expected["participant"]:
        return False, "participant_mismatch"
    if atom.rule_version != cert.extraction_rule_version: return False, "atom_rule_link_mismatch"
    # 1-2: recompute raw_hash and atom_hash from the supplied payloads
    if cert.raw_hash != H(canonical_raw_payload): return False, "raw_hash_mismatch"
    if cert.atom_hash != H(atom.atom_bytes): return False, "atom_hash_mismatch"
    if atom.raw_hash != cert.raw_hash: return False, "atom_raw_link_mismatch"
    if atom.extraction_certificate_hash != cert.cert_hash(): return False, "cert_link_mismatch"
    # 3: extractor signature over the canonical cert body (which now binds all the metadata above)
    pub = registered_pubkeys.get(cert.extractor_id)
    if pub is None: return False, "extractor_not_registered"
    try:
        pub.verify(cert.cert_sig, cert.body())
    except InvalidSignature:
        return False, "bad_extractor_signature"
    # 5: MRENCLAVE on the accepted measurement list
    if cert.extractor_mrenclave not in accepted_mrenclaves: return False, "mrenclave_not_accepted"
    # 6: rule + quantization version match the declared replay policy
    if cert.extraction_rule_version not in accepted_rule_versions: return False, "rule_version_mismatch"
    if cert.quantization_version != QUANT_VERSION: return False, "quant_version_mismatch"
    # 7: freshness -- challenge, root epoch, and nonce bound to THIS sealing round
    if cert.coordinator_challenge != fresh_challenge: return False, "stale_challenge"
    if cert.root_epoch != fresh_root_epoch: return False, "stale_root_epoch"
    if cert.nonce in seen_nonces: return False, "replayed_nonce"
    return True, "accepted"


# ----------------------------- self-test -----------------------------
def _selftest():
    ext = Extractor()
    reg = {ext.extractor_id: ext.pub}; mre = {ext.mrenclave}
    fam = H("family", "fl_governance"); chal = H("challenge", 1); epoch = H("root_epoch", 42)
    p = b"p" * 16; rv = {rule_hash("llr_grad", RULE_V1["llr_grad"])}
    seen = set()
    atom, cert, raw, llr = ext.extract(5, p, "llr_grad", 0.92, fam, chal, nonce=1, root_epoch=epoch)
    expected = {"family_id": fam, "round": 5, "participant": p, "atom_kind": "llr_grad"}
    base_ok = dict(expected=expected, registered_pubkeys=reg, accepted_mrenclaves=mre,
                   accepted_rule_versions=rv, fresh_challenge=chal, fresh_root_epoch=epoch, seen_nonces=seen)
    results = {}
    results["honest_accept"] = coordinator_verify(atom, cert, raw, **base_ok)
    # 1: host forges atom bytes
    bad_atom = CertifiedAtom(**{**atom.__dict__, "atom_bytes": atom_bytes_of("llr_grad", llr + 100)})
    results["forge_atom"] = coordinator_verify(bad_atom, cert, raw, **base_ok)
    # 2: host modifies raw after extraction
    results["modify_raw"] = coordinator_verify(atom, cert, raw + b"x", **base_ok)
    # 3: wrong extraction rule (different params -> different rule_version, not on accepted list)
    a2, c2, r2, _ = ext.extract(5, p, "llr_grad", 0.92, fam, chal, nonce=2, root_epoch=epoch,
                                params={**RULE_V1["llr_grad"], "a0": 5.0})
    results["wrong_rule"] = coordinator_verify(a2, c2, r2, **base_ok)
    # 4: replay an old cert (nonce already seen)
    results["replay_nonce"] = coordinator_verify(atom, cert, raw, **{**base_ok, "seen_nonces": {1}})
    # 5: stale root epoch
    results["stale_epoch"] = coordinator_verify(atom, cert, raw, **{**base_ok, "fresh_root_epoch": H("e", 99)})
    # 6: wrong extractor identity (different key)
    ext2 = Extractor(seed=b"rogue-extractor")
    a3, c3, r3, _ = ext2.extract(5, p, "llr_grad", 0.92, fam, chal, nonce=3, root_epoch=epoch)
    results["wrong_extractor"] = coordinator_verify(a3, c3, r3, **base_ok)
    # 7: CERT TRANSPLANT -- valid (atom,cert) for cell A presented as cell B (different round/participant).
    #    Coordinator expects round=9, participant=q: the recomputed cell_id won't match the signed cert.
    exp_b = {"family_id": fam, "round": 9, "participant": b"q" * 16, "atom_kind": "llr_grad"}
    results["cert_transplant"] = coordinator_verify(atom, cert, raw, **{**base_ok, "expected": exp_b})
    for name, (ok, reason) in results.items():
        print(f"  {name:18s} -> {'ACCEPT' if ok else f'REJECT({reason})'}")
    assert results["honest_accept"][0] and not any(results[k][0] for k in results if k != "honest_accept")
    print(f"D1 selftest PASS: honest accepted, all {len(results)-1} attacks rejected "
          "(incl. cert-transplant + stale-epoch).")


if __name__ == "__main__":
    _selftest()
