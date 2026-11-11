"""REST server and client for attestation, trust management, and federation control."""

from .server import create_app, DTRUSTServer
from .client import DTRUSTClient

__all__ = [
    "create_app",
    "DTRUSTServer", 
    "DTRUSTClient"
] 