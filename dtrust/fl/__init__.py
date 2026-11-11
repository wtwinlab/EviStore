"""FL client, trust-aware aggregation strategies, and certified federated repair."""

from .strategy import TrustAwareStrategy
from .client import DTRUSTClient

__all__ = [
    "TrustAwareStrategy",
    "DTRUSTClient",
] 