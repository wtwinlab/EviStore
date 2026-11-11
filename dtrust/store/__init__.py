"""Trust-indexed provenance store (TrustLog, SRI, DRI, schema advisor).

Per-round, per-participant trust state is appended as a sealed,
schema-typed relation served through declarative query operators:
pointInTime, rangeBelow, cusumReplay, degradedEvents (base, SQL-backed),
plus the Sealed Replay Index (SRI) aggregate operators count_below /
weight_below / count_between — O(log U) sublinear, exact, and rebuildable
from the sealed tuples (see sri_index.SRIIndex).
"""

from .schema import Mode, TrustTuple, CHUNK_SIZE, TUPLE_SIZE, TUPLES_PER_CHUNK
from .sri_index import SRIIndex
from .store import TrustStore

__all__ = [
    "Mode",
    "TrustTuple",
    "TrustStore",
    "SRIIndex",
    "CHUNK_SIZE",
    "TUPLE_SIZE",
    "TUPLES_PER_CHUNK",
]
