"""
Attestation Binder - Manages binding between attestations and node identities.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

from ..core.models import AttestationResult, NodeIdentity
from ..core.exceptions import TrustError
from ..core.config import TrustConfig
from ..core.logging import get_logger, log_exception


logger = get_logger(__name__)


@dataclass
class TrustBinding:
    """Represents a trust binding between a node and its attestation."""
    node_id: str
    attestation_id: str
    attestation_result: AttestationResult
    binding_time: datetime
    expiry_time: datetime
    is_active: bool = True


class AttestationBinder:
    """
    Manages the binding between node identities and their attestation results.
    
    Handles:
    - Creating trust bindings
    - Validating binding integrity
    - Managing binding lifecycle
    - Automatic expiry and renewal
    """
    
    def __init__(self, config: TrustConfig):
        self.config = config
        
        # Storage for trust bindings (in production, this would be a database)
        self._bindings: Dict[str, TrustBinding] = {}
        
        # Index by attestation ID for quick lookup
        self._attestation_index: Dict[str, str] = {}  # attestation_id -> node_id
    
    def bind_attestation(
        self, 
        node_id: str, 
        attestation: AttestationResult
    ) -> TrustBinding:
        """
        Create a trust binding between a node and its attestation.
        
        Args:
            node_id: Node identifier
            attestation: Attestation result to bind
            
        Returns:
            TrustBinding: Created binding
            
        Raises:
            TrustError: If binding creation fails
        """
        try:
            # Validate attestation belongs to this node
            if attestation.node_id != node_id:
                raise TrustError(
                    f"Attestation node_id {attestation.node_id} does not match {node_id}"
                )
            
            # Calculate expiry time
            expiry_time = datetime.utcnow() + timedelta(
                hours=self.config.attestation_validity_hours
            )
            
            # Create binding
            binding = TrustBinding(
                node_id=node_id,
                attestation_id=attestation.attestation_id,
                attestation_result=attestation,
                binding_time=datetime.utcnow(),
                expiry_time=expiry_time,
                is_active=True
            )
            
            # Deactivate any existing binding for this node
            if node_id in self._bindings:
                old_binding = self._bindings[node_id]
                old_binding.is_active = False
                # Remove from attestation index
                self._attestation_index.pop(old_binding.attestation_id, None)
            
            # Store new binding
            self._bindings[node_id] = binding
            self._attestation_index[attestation.attestation_id] = node_id
            
            logger.info(
                f"Created trust binding for {node_id} "
                f"(attestation: {attestation.attestation_id[:8]}..., "
                f"expires: {expiry_time.isoformat()})"
            )
            
            return binding
            
        except Exception as e:
            raise TrustError(f"Failed to bind attestation for {node_id}: {e}")
    
    def get_binding(self, node_id: str) -> Optional[TrustBinding]:
        """Get current trust binding for a node."""
        return self._bindings.get(node_id)
    
    def get_binding_by_attestation(self, attestation_id: str) -> Optional[TrustBinding]:
        """Get trust binding by attestation ID."""
        node_id = self._attestation_index.get(attestation_id)
        if node_id:
            return self._bindings.get(node_id)
        return None
    
    def is_binding_valid(self, node_id: str) -> bool:
        """Check if a node has a valid trust binding."""
        binding = self._bindings.get(node_id)
        if not binding:
            return False
        
        return (
            binding.is_active and
            datetime.utcnow() < binding.expiry_time and
            binding.attestation_result.is_valid
        )
    
    def validate_binding_integrity(self, node_id: str) -> bool:
        """
        Validate the cryptographic integrity of a trust binding.
        
        This would typically involve:
        - Verifying digital signatures
        - Checking attestation quote integrity
        - Validating binding timestamps
        """
        binding = self._bindings.get(node_id)
        if not binding:
            return False
        
        try:
            attestation = binding.attestation_result
            
            # Check basic validity
            if not attestation.is_valid:
                return False
            
            # Check binding hasn't been tampered with
            if binding.attestation_id != attestation.attestation_id:
                logger.warning(f"Binding integrity check failed for {node_id}: ID mismatch")
                return False
            
            # Check timing constraints
            if binding.binding_time > attestation.verification_time:
                logger.warning(f"Binding integrity check failed for {node_id}: time inconsistency")
                return False
            
            # In a real implementation, this would also:
            # - Verify SGX quote signatures
            # - Check attestation service signatures
            # - Validate nonce freshness
            # - Verify enclave measurements
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating binding integrity for {node_id}: {e}")
            return False
    
    def renew_binding(self, node_id: str, new_attestation: AttestationResult) -> TrustBinding:
        """Renew an existing trust binding with a new attestation."""
        old_binding = self._bindings.get(node_id)
        if old_binding:
            logger.info(f"Renewing trust binding for {node_id}")
        
        return self.bind_attestation(node_id, new_attestation)
    
    def revoke_binding(self, node_id: str, reason: str = "manual_revocation"):
        """Revoke a trust binding."""
        binding = self._bindings.get(node_id)
        if binding:
            binding.is_active = False
            # Remove from attestation index
            self._attestation_index.pop(binding.attestation_id, None)
            
            logger.warning(f"Revoked trust binding for {node_id}: {reason}")
    
    def cleanup_expired_bindings(self) -> List[str]:
        """Clean up expired bindings and return list of affected node IDs."""
        expired_nodes = []
        current_time = datetime.utcnow()
        
        for node_id, binding in list(self._bindings.items()):
            if (binding.is_active and 
                (current_time >= binding.expiry_time or 
                 not binding.attestation_result.is_valid)):
                
                binding.is_active = False
                expired_nodes.append(node_id)
                
                # Remove from attestation index
                self._attestation_index.pop(binding.attestation_id, None)
                
                logger.info(f"Trust binding expired for {node_id}")
        
        return expired_nodes
    
    def get_binding_stats(self) -> Dict[str, int]:
        """Get statistics about trust bindings."""
        total_bindings = len(self._bindings)
        active_bindings = sum(1 for b in self._bindings.values() if b.is_active)
        expired_bindings = total_bindings - active_bindings
        
        # Count bindings expiring soon (within 2 hours)
        expiring_soon = 0
        threshold = datetime.utcnow() + timedelta(hours=2)
        
        for binding in self._bindings.values():
            if binding.is_active and binding.expiry_time <= threshold:
                expiring_soon += 1
        
        return {
            "total_bindings": total_bindings,
            "active_bindings": active_bindings,
            "expired_bindings": expired_bindings,
            "expiring_soon": expiring_soon
        }
    
    def get_bindings_for_renewal(self, hours_ahead: int = 2) -> List[TrustBinding]:
        """Get bindings that need renewal within specified hours."""
        threshold = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        renewal_candidates = []
        for binding in self._bindings.values():
            if (binding.is_active and 
                binding.expiry_time <= threshold and
                binding.attestation_result.is_valid):
                renewal_candidates.append(binding)
        
        return renewal_candidates 