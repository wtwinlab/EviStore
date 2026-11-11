"""DTRUST: dynamic trust management for federated learning with sealed evidence."""

__version__ = "1.0.0"

from .core.config import DTRUSTConfig, SGXConfig, TrustConfig, FLConfig
from .core.models import (
    NodeIdentity, NodeState, AttestationResult, TrustMetrics,
    FederationInfo, TrainingTask, SGXQuote,
)
from .core.exceptions import DTRUSTError, AttestationError, TrustError, NodeError

from .tee import (
    SGXQuoteGenerator, MockSGXQuoteGenerator, create_quote_generator,
    AttestationService, MockAttestationService, create_attestation_service,
    AttestationProtocol, NodeIdentityManager,
)
from .trust import TrustManager, AttestationBinder, TrustScorer
from .fl import TrustAwareStrategy
from .api import DTRUSTServer, DTRUSTClient, create_app

__all__ = [
    "__version__",
    "DTRUSTConfig", "SGXConfig", "TrustConfig", "FLConfig",
    "NodeIdentity", "NodeState", "AttestationResult", "TrustMetrics",
    "FederationInfo", "TrainingTask", "SGXQuote",
    "DTRUSTError", "AttestationError", "TrustError", "NodeError",
    "SGXQuoteGenerator", "MockSGXQuoteGenerator", "create_quote_generator",
    "AttestationService", "MockAttestationService", "create_attestation_service",
    "AttestationProtocol", "NodeIdentityManager",
    "TrustManager", "AttestationBinder", "TrustScorer",
    "TrustAwareStrategy",
    "DTRUSTServer", "DTRUSTClient", "create_app",
]
