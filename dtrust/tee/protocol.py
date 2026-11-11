"""
Attestation Protocol - Orchestrates the complete SGX attestation process.
"""

import asyncio
import hashlib
import secrets
from typing import Optional, Dict, Any

from ..core.models import AttestationResult, NodeIdentity, SGXQuote
from ..core.exceptions import AttestationError
from ..core.config import SGXConfig
from ..core.logging import get_logger, log_exception
from .sgx_quote import SGXQuoteGeneratorBase, create_quote_generator
from .attestation import AttestationServiceBase, create_attestation_service
from .identity import NodeIdentityManager


logger = get_logger(__name__)


class AttestationProtocol:
    """
    Orchestrates the complete SGX attestation process.
    
    Handles:
    - Challenge generation and validation
    - Quote generation coordination
    - Attestation verification
    - Response signing and validation
    """
    
    def __init__(
        self, 
        config: SGXConfig,
        identity_manager: NodeIdentityManager,
        quote_generator: Optional[SGXQuoteGeneratorBase] = None,
        attestation_service: Optional[AttestationServiceBase] = None
    ):
        self.config = config
        self.identity_manager = identity_manager
        self.quote_generator = quote_generator or create_quote_generator(config)
        self.attestation_service = attestation_service or create_attestation_service(config)
        
        # Active challenges (in production, this would be in a database)
        self._active_challenges: Dict[str, Dict[str, Any]] = {}
    
    async def process_attestation_request(
        self, 
        node_id: str, 
        challenge: Optional[str] = None
    ) -> AttestationResult:
        """
        Process a complete attestation request.
        
        Args:
            node_id: Node requesting attestation
            challenge: Optional challenge string (will generate if not provided)
            
        Returns:
            AttestationResult: Complete attestation result
        """
        try:
            logger.info(f"Processing attestation request for node {node_id}")

            node_identity = self._require_identity(node_id)

            # Generate or validate challenge
            if not challenge:
                challenge = self._generate_challenge()

            # Track challenge (relying-party state) so verify_attestation_response()
            # can later confirm freshness/nonce.
            self._active_challenges[node_id] = {
                "challenge": challenge,
                "challenge_hash": self._hash_challenge(challenge, node_id),
                "timestamp": asyncio.get_event_loop().time(),
                "enclave_id": node_identity.enclave_id,
            }

            # Client side (runs inside the enclave for the real DCAP backend) then
            # relying-party side. Co-located here; split across machines for real
            # deployments via generate_quote_for() + verify_quote().
            quote = await self.generate_quote_for(node_id, challenge, node_identity)
            attestation_result = await self.verify_quote(node_id, challenge, quote)

            signed_result = self._sign_attestation_result(attestation_result, node_identity)

            self._active_challenges.pop(node_id, None)

            logger.info(
                f"Attestation completed for {node_id}: "
                f"status={attestation_result.status}, "
                f"confidence={attestation_result.confidence:.3f}"
            )

            return signed_result

        except Exception as e:
            # Cleanup on error
            self._active_challenges.pop(node_id, None)
            logger.error(f"Attestation failed for {node_id}: {e}")
            raise AttestationError(f"Attestation process failed: {e}")

    async def generate_quote_for(
        self,
        node_id: str,
        challenge: str,
        node_identity: Optional[NodeIdentity] = None,
    ) -> SGXQuote:
        """Client/enclave side: produce a quote binding ``challenge`` for ``node_id``.

        For the real Gramine/DCAP backend this MUST run inside the SGX enclave (it
        reads /dev/attestation); for the mock backend it runs anywhere. Split this
        from verify_quote() to run generation on the (weak) SGX host and
        verification + ML elsewhere.
        """
        node_identity = node_identity or self._require_identity(node_id)
        challenge_hash = self._hash_challenge(challenge, node_id)
        logger.debug(f"Generating SGX quote for {node_id}")
        return await self.quote_generator.generate_quote(
            node_identity.enclave_id, challenge_hash
        )

    async def verify_quote(
        self,
        node_id: str,
        challenge: str,
        quote: SGXQuote,
    ) -> AttestationResult:
        """Relying-party side: verify ``quote`` answers ``challenge`` and score trust.

        With the DCAP backend this performs real signature/TCB verification and
        checks that the quote's report_data binds sha256(challenge:node_id); with
        the mock backend it returns the configured scenario result.
        """
        logger.debug(f"Verifying attestation for {node_id}")
        return await self.attestation_service.verify_attestation(quote, challenge, node_id)

    def _require_identity(self, node_id: str) -> NodeIdentity:
        node_identity = self.identity_manager.get_node_identity(node_id)
        if not node_identity:
            raise AttestationError(f"Node identity not found for {node_id}")
        return node_identity

    async def verify_attestation_response(
        self, 
        node_id: str, 
        attestation_result: AttestationResult
    ) -> bool:
        """
        Verify an attestation response from a client.
        
        Args:
            node_id: Node that provided the attestation
            attestation_result: Attestation result to verify
            
        Returns:
            bool: True if verification succeeds
        """
        try:
            # Get node identity
            node_identity = self.identity_manager.get_node_identity(node_id)
            if not node_identity:
                logger.warning(f"Cannot verify attestation: no identity for {node_id}")
                return False
            
            # Verify signature (if present)
            if not self._verify_attestation_signature(attestation_result, node_identity):
                logger.warning(f"Attestation signature verification failed for {node_id}")
                return False
            
            # Verify challenge response
            challenge_info = self._active_challenges.get(node_id)
            if challenge_info:
                if not self._verify_challenge_response(attestation_result, challenge_info):
                    logger.warning(f"Challenge verification failed for {node_id}")
                    return False
            
            # Verify attestation is recent
            if not attestation_result.is_valid:
                logger.warning(f"Attestation is expired or invalid for {node_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying attestation response for {node_id}: {e}")
            return False
    
    def _generate_challenge(self) -> str:
        """Generate a cryptographic challenge."""
        return secrets.token_hex(32)
    
    def _hash_challenge(self, challenge: str, node_id: str) -> bytes:
        """Create a hash of the challenge for use in SGX quote."""
        combined = f"{challenge}:{node_id}".encode()
        return hashlib.sha256(combined).digest()
    
    def _sign_attestation_result(
        self, 
        attestation_result: AttestationResult, 
        node_identity: NodeIdentity
    ) -> AttestationResult:
        """Sign the attestation result with node's private key."""
        try:
            # In a real implementation, this would:
            # 1. Serialize the attestation result
            # 2. Sign with the node's private key
            # 3. Attach signature to the result
            
            # For now, we'll just add a signature placeholder
            # The actual signing would be done by the node itself
            
            return attestation_result
            
        except Exception as e:
            logger.error(f"Error signing attestation result: {e}")
            return attestation_result
    
    def _verify_attestation_signature(
        self, 
        attestation_result: AttestationResult, 
        node_identity: NodeIdentity
    ) -> bool:
        """Verify the signature on an attestation result."""
        try:
            # In a real implementation, this would:
            # 1. Extract signature from attestation result
            # 2. Verify signature using node's public key
            # 3. Validate signature covers all critical fields
            
            # For now, assume signature is valid
            return True
            
        except Exception as e:
            logger.error(f"Error verifying attestation signature: {e}")
            return False
    
    def _verify_challenge_response(
        self, 
        attestation_result: AttestationResult, 
        challenge_info: Dict[str, Any]
    ) -> bool:
        """Verify that the attestation properly responds to the challenge."""
        try:
            # Check challenge freshness (5 minute window)
            current_time = asyncio.get_event_loop().time()
            challenge_age = current_time - challenge_info["timestamp"]
            
            if challenge_age > 300:  # 5 minutes
                logger.warning("Challenge is too old")
                return False
            
            # Verify nonce matches challenge
            if attestation_result.nonce != challenge_info["challenge"]:
                logger.warning("Challenge nonce mismatch")
                return False
            
            # In a real implementation, this would also verify:
            # - The SGX quote contains the challenge hash in report_data
            # - The enclave measurement matches expected values
            # - The quote signature is valid
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying challenge response: {e}")
            return False
    
    def cleanup_expired_challenges(self):
        """Clean up expired challenges."""
        current_time = asyncio.get_event_loop().time()
        expired_nodes = []
        
        for node_id, challenge_info in self._active_challenges.items():
            if current_time - challenge_info["timestamp"] > 600:  # 10 minutes
                expired_nodes.append(node_id)
        
        for node_id in expired_nodes:
            self._active_challenges.pop(node_id, None)
            logger.debug(f"Cleaned up expired challenge for {node_id}")
    
    def get_active_challenges(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active challenges (for debugging)."""
        return {
            node_id: {
                "timestamp": info["timestamp"],
                "enclave_id": info["enclave_id"],
                "age_seconds": asyncio.get_event_loop().time() - info["timestamp"]
            }
            for node_id, info in self._active_challenges.items()
        } 