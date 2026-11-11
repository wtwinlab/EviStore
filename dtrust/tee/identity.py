"""
Node Identity Manager - Manages cryptographic identities for nodes.
"""

from typing import Dict, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import hashlib

from ..core.models import NodeIdentity
from ..core.exceptions import NodeError
from ..core.config import SGXConfig
from ..core.logging import get_logger, log_exception


logger = get_logger(__name__)


class NodeIdentityManager:
    """
    Manages cryptographic identities for nodes in the DTRUST system.
    
    Handles:
    - RSA key pair generation
    - Node identity creation and storage
    - Digital signature operations
    - Identity verification
    """
    
    def __init__(self, config: SGXConfig):
        self.config = config
        
        # Storage for node identities (in production, this would be a database)
        self._identities: Dict[str, NodeIdentity] = {}
    
    def create_node_identity(self, enclave_id: str) -> NodeIdentity:
        """
        Create a new cryptographic identity for a node.
        
        Args:
            enclave_id: Unique identifier for the enclave
            
        Returns:
            NodeIdentity: Created identity with key pair
        """
        try:
            logger.info(f"Creating node identity for enclave {enclave_id}")
            
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Create node identity
            node_identity = NodeIdentity(
                node_id=f"node_{enclave_id}",
                enclave_id=enclave_id,
                public_key=public_pem.decode(),
                private_key_hash=hashlib.sha256(private_pem).hexdigest()
            )
            
            # Store the private key separately for signing operations
            self._private_keys = getattr(self, '_private_keys', {})
            self._private_keys[node_identity.node_id] = private_pem.decode()
            
            # Store identity
            self._identities[node_identity.node_id] = node_identity
            
            logger.info(f"Created identity for node {node_identity.node_id}")
            return node_identity
            
        except Exception as e:
            raise NodeError(f"Failed to create node identity for {enclave_id}: {e}")
    
    def get_node_identity(self, node_id: str) -> Optional[NodeIdentity]:
        """Get node identity by node ID."""
        return self._identities.get(node_id)
    
    def get_identity_by_enclave(self, enclave_id: str) -> Optional[NodeIdentity]:
        """Get node identity by enclave ID."""
        for identity in self._identities.values():
            if identity.enclave_id == enclave_id:
                return identity
        return None
    
    def sign_message(self, node_id: str, message: bytes) -> bytes:
        """
        Sign a message with the node's private key.
        
        Args:
            node_id: Node identifier
            message: Message to sign
            
        Returns:
            bytes: Digital signature
        """
        identity = self._identities.get(node_id)
        if not identity:
            raise NodeError(f"Node identity not found: {node_id}")
        
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                self._private_keys[node_id].encode(),
                password=None
            )
            
            # Sign message
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature
            
        except Exception as e:
            raise NodeError(f"Failed to sign message for {node_id}: {e}")
    
    def verify_signature(
        self, 
        node_id: str, 
        message: bytes, 
        signature: bytes
    ) -> bool:
        """
        Verify a signature using the node's public key.
        
        Args:
            node_id: Node identifier
            message: Original message
            signature: Signature to verify
            
        Returns:
            bool: True if signature is valid
        """
        identity = self._identities.get(node_id)
        if not identity:
            logger.warning(f"Cannot verify signature: node identity not found: {node_id}")
            return False
        
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                identity.public_key.encode()
            )
            
            # Verify signature
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.debug(f"Signature verification failed for {node_id}: {e}")
            return False
    
    def verify_signature_with_public_key(
        self, 
        public_key_pem: str, 
        message: bytes, 
        signature: bytes
    ) -> bool:
        """
        Verify a signature using a provided public key.
        
        Args:
            public_key_pem: Public key in PEM format
            message: Original message
            signature: Signature to verify
            
        Returns:
            bool: True if signature is valid
        """
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode()
            )
            
            # Verify signature
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.debug(f"Signature verification failed with provided key: {e}")
            return False
    
    def remove_node_identity(self, node_id: str) -> bool:
        """
        Remove a node identity.
        
        Args:
            node_id: Node identifier
            
        Returns:
            bool: True if identity was removed
        """
        if node_id in self._identities:
            del self._identities[node_id]
            logger.info(f"Removed identity for node {node_id}")
            return True
        return False
    
    def list_node_identities(self) -> Dict[str, str]:
        """
        List all node identities.
        
        Returns:
            Dict mapping node_id to enclave_id
        """
        return {
            node_id: identity.enclave_id 
            for node_id, identity in self._identities.items()
        }
    
    def get_identity_stats(self) -> Dict[str, int]:
        """Get statistics about managed identities."""
        return {
            "total_identities": len(self._identities),
            "unique_enclaves": len(set(
                identity.enclave_id for identity in self._identities.values()
            ))
        } 