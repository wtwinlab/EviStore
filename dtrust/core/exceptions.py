"""
Exception classes for DTRUST framework.
"""


class DTRUSTError(Exception):
    """Base exception for DTRUST framework."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "DTRUST_ERROR"


class AttestationError(DTRUSTError):
    """Exception raised during SGX attestation process."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "ATTESTATION_ERROR")


class TrustError(DTRUSTError):
    """Exception raised during trust management operations."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "TRUST_ERROR")


class FederatedLearningError(DTRUSTError):
    """Exception raised during federated learning operations."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "FL_ERROR")


class ConfigurationError(DTRUSTError):
    """Exception raised for configuration-related issues."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "CONFIG_ERROR")


class DatabaseError(DTRUSTError):
    """Exception raised for database-related issues."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "DATABASE_ERROR")


class NodeError(DTRUSTError):
    """Exception raised for node-related issues."""
    
    def __init__(self, message: str, node_id: str = None, error_code: str = None):
        super().__init__(message, error_code or "NODE_ERROR")
        self.node_id = node_id


class SecurityError(DTRUSTError):
    """Exception raised for security-related issues."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code or "SECURITY_ERROR") 