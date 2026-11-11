"""Schema and binary format for the trust-indexed provenance store.

Each round of training emits one TrustTuple per registered participant.
Tuples are packed into 4 KiB sealed chunks; chunks are chained into a
binary Merkle tree whose root is anchored to an external transparency
log every M rounds (default 256); see Section V of the paper.

Layout:
    round         uint32  monotonic
    participant   16 B    enclave identity hash
    a_conf        uint16  attestation confidence (quantized 2^-16)
    fresh         uint8   1 if attestation not expired
    s_cusum       float32 CUSUM statistic
    t_score       uint16  composite trust (quantized)
    b_score       uint16  behavior score (quantized)
    w_agg         uint16  aggregation weight (quantized)
    mode          uint8   {NORMAL=0, DEGRADED=1, PAUSED=2}
    h_grad        32 B    Blake3/SHA-256 hash of submitted update
    llr_att..seal 5x int16 per-channel clamped LLRs (milli-LLR fixed point); the
                          advisor's weight/decay augmentation.

Total: 76 bytes. The 5 LLR fields (10 B) occupy the former forward-compat tail
padding, so the augmentation that makes the weight/decay policy family replayable
(Sufficient Statistic Schema / Minimal Sealed-Schema theorems) costs ZERO net
bytes/cell over the legacy tuple. Legacy tuples (LLRs = 0) round-trip unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# Wire format. Little-endian; explicit padding to keep alignment.
# I (round) + 16s (participant) + H (a_conf) + B (fresh) + f (s_cusum)
# + 3H (t,b,w) + B (mode) + 32s (h_grad) = 4 + 16 + 2 + 1 + 4 + 6 + 1 + 32 = 66 B.
# The 5 per-channel clamped LLRs (5h = 10 B, signed milli-LLR fixed point) occupy
# what was the 10-byte forward-compat tail padding, keeping the tuple at 76 B:
# the weight/decay augmentation adds zero net bytes/cell to the 66 B layout.
_PACK_FMT = "<I 16s H B f 3H B 32s 5h"
TUPLE_SIZE = struct.calcsize(_PACK_FMT)
assert TUPLE_SIZE == 76, f"expected 76, got {TUPLE_SIZE}"

# Per-channel LLRs are clamped to a few units in the SPRT scorer; we seal each as a
# signed int16 fixed point. The four evidence channels clamp to +/-clamp_per_llr (=2),
# so scale 16000 covers +/-2.048 at ~6e-5 resolution; the seal channel can reach
# -clamp*10 (=-20) on a violation, so it uses a wider/coarser scale. Per-channel scales
# make the deployed (2-byte) LLR sealing reproduce the decay/weight counterfactual with
# ZERO decision flips and the trust score to within a few uint16 ulps (float LLRs are
# exact); the scales are part of the public replay statement (bound in the policy hash).
_LLR_MIN, _LLR_MAX = -32768, 32767
LLR_CHANNELS = ("llr_att", "llr_grad", "llr_perf", "llr_beh", "llr_seal")
LLR_SCALES = (16000, 16000, 16000, 16000, 1600)
LLR_SCALE = LLR_SCALES[0]  # legacy alias


def quantize_llr(x: float, ch: int = 0) -> int:
    q = int(round(float(x) * LLR_SCALES[ch]))
    return _LLR_MIN if q < _LLR_MIN else _LLR_MAX if q > _LLR_MAX else q


def dequantize_llr(q: int, ch: int = 0) -> float:
    return q / LLR_SCALES[ch]

CHUNK_SIZE = 4096
CHUNK_HEADER_SIZE = 32
CHUNK_BODY_SIZE = CHUNK_SIZE - CHUNK_HEADER_SIZE  # 4064
TUPLES_PER_CHUNK = CHUNK_BODY_SIZE // TUPLE_SIZE  # 53


class Mode(IntEnum):
    NORMAL = 0
    DEGRADED = 1   # |R(t)| < N/2; aggregator fell back to full set
    PAUSED = 2     # operator-initiated halt


@dataclass(slots=True)
class TrustTuple:
    round: int
    participant: bytes  # 16 bytes
    a_conf: int         # uint16
    fresh: bool
    s_cusum: float
    t_score: int        # uint16
    b_score: int        # uint16
    w_agg: int          # uint16
    mode: Mode
    h_grad: bytes       # 32 bytes
    # weight/decay augmentation: 5 clamped per-channel LLRs (milli-LLR int16).
    # Default 0 so legacy call sites (no LLRs) construct and round-trip unchanged.
    llr_att: int = 0
    llr_grad: int = 0
    llr_perf: int = 0
    llr_beh: int = 0
    llr_seal: int = 0

    def pack(self) -> bytes:
        if len(self.participant) != 16:
            raise ValueError(f"participant must be 16 bytes, got {len(self.participant)}")
        if len(self.h_grad) != 32:
            raise ValueError(f"h_grad must be 32 bytes, got {len(self.h_grad)}")
        return struct.pack(
            _PACK_FMT,
            self.round,
            self.participant,
            self.a_conf,
            int(self.fresh),
            self.s_cusum,
            self.t_score,
            self.b_score,
            self.w_agg,
            int(self.mode),
            self.h_grad,
            self.llr_att,
            self.llr_grad,
            self.llr_perf,
            self.llr_beh,
            self.llr_seal,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TrustTuple":
        if len(data) != TUPLE_SIZE:
            raise ValueError(f"expected {TUPLE_SIZE} bytes, got {len(data)}")
        (
            round_, participant, a_conf, fresh, s_cusum,
            t_score, b_score, w_agg, mode, h_grad,
            llr_att, llr_grad, llr_perf, llr_beh, llr_seal,
        ) = struct.unpack(_PACK_FMT, data)
        return cls(
            round=round_,
            participant=participant,
            a_conf=a_conf,
            fresh=bool(fresh),
            s_cusum=s_cusum,
            t_score=t_score,
            b_score=b_score,
            w_agg=w_agg,
            mode=Mode(mode),
            h_grad=h_grad,
            llr_att=llr_att,
            llr_grad=llr_grad,
            llr_perf=llr_perf,
            llr_beh=llr_beh,
            llr_seal=llr_seal,
        )

    def llr_vector(self) -> tuple:
        """The 5 clamped per-channel LLRs as floats (per-channel dequantized)."""
        qs = (self.llr_att, self.llr_grad, self.llr_perf, self.llr_beh, self.llr_seal)
        return tuple(dequantize_llr(q, ch) for ch, q in enumerate(qs))


def quantize_unit(x: float) -> int:
    """Quantize a [0, 1] real to uint16, mirroring the ledger column types."""
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    return int(round(x * 65535))


def dequantize_unit(q: int) -> float:
    return q / 65535.0
