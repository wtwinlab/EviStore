"""
DTRUST API Client - Client library for interacting with DTRUST server.
"""

import asyncio
from typing import Dict, List, Optional, Any

import httpx
from pydantic import BaseModel

from ..core.models import AttestationResult, NodeState, NodeIdentity, FederationInfo
from ..core.exceptions import DTRUSTError
from ..core.logging import get_logger, log_exception


logger = get_logger(__name__)


class DTRUSTClient:
    """
    Client for interacting with DTRUST API server.
    
    Provides high-level methods for:
    - Node registration and management
    - Attestation requests
    - Trust management
    - Federation participation
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key
        
        # HTTP client configuration
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, str]:
        """Check server health."""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise DTRUSTError(f"Health check failed: {e}")
    
    # Node management methods
    async def register_node(
        self, 
        enclave_id: str, 
        node_metadata: Optional[Dict[str, Any]] = None
    ) -> NodeIdentity:
        """Register a new node with the DTRUST system."""
        try:
            payload = {
                "enclave_id": enclave_id,
                "node_metadata": node_metadata or {}
            }
            
            response = await self.client.post("/api/v1/nodes/register", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return NodeIdentity(**data)
            
        except httpx.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.json().get("detail", str(e))
                raise DTRUSTError(f"Node registration failed: {error_detail}")
            raise DTRUSTError(f"HTTP error during node registration: {e}")
    
    async def get_node_state(self, node_id: str) -> NodeState:
        """Get current state of a node."""
        try:
            response = await self.client.get(f"/api/v1/nodes/{node_id}")
            response.raise_for_status()
            
            data = response.json()
            return NodeState(**data)
            
        except httpx.HTTPError as e:
            if e.response.status_code == 404:
                raise DTRUSTError(f"Node not found: {node_id}")
            raise DTRUSTError(f"HTTP error getting node state: {e}")
    
    async def list_nodes(self) -> List[str]:
        """List all registered nodes."""
        try:
            response = await self.client.get("/api/v1/nodes")
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise DTRUSTError(f"HTTP error listing nodes: {e}")
    
    # Attestation methods
    async def request_attestation(
        self, 
        node_id: str, 
        challenge: Optional[str] = None
    ) -> AttestationResult:
        """Request SGX attestation for a node."""
        try:
            payload = {
                "node_id": node_id,
                "challenge": challenge
            }
            
            response = await self.client.post("/api/v1/attest", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return AttestationResult(**data)
            
        except httpx.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.json().get("detail", str(e))
                raise DTRUSTError(f"Attestation failed: {error_detail}")
            raise DTRUSTError(f"HTTP error during attestation: {e}")
    
    # Trust management methods
    async def get_trusted_nodes(
        self, 
        min_trust_score: Optional[float] = None
    ) -> List[NodeState]:
        """Get list of trusted nodes eligible for federated learning."""
        try:
            params = {}
            if min_trust_score is not None:
                params["min_trust_score"] = min_trust_score
            
            response = await self.client.get("/api/v1/trust/nodes", params=params)
            response.raise_for_status()
            
            data = response.json()
            return [NodeState(**item) for item in data]
            
        except httpx.HTTPError as e:
            raise DTRUSTError(f"HTTP error getting trusted nodes: {e}")
    
    async def get_suspicious_nodes(self) -> List[NodeState]:
        """Get list of suspicious nodes."""
        try:
            response = await self.client.get("/api/v1/trust/suspicious")
            response.raise_for_status()
            
            data = response.json()
            return [NodeState(**item) for item in data]
            
        except httpx.HTTPError as e:
            raise DTRUSTError(f"HTTP error getting suspicious nodes: {e}")
    
    async def record_training_participation(
        self, 
        node_id: str, 
        num_examples: int,
        metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Record a node's participation in federated learning training."""
        try:
            payload = {
                "node_id": node_id,
                "num_examples": num_examples,
                "metrics": metrics
            }
            
            response = await self.client.post("/api/v1/trust/training", json=payload)
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            if e.response.status_code == 404:
                raise DTRUSTError(f"Node not found: {node_id}")
            raise DTRUSTError(f"HTTP error recording training participation: {e}")
    
    async def record_suspicious_activity(
        self, 
        node_id: str, 
        reason: str
    ) -> Dict[str, str]:
        """Record suspicious activity for a node."""
        try:
            payload = {
                "node_id": node_id,
                "reason": reason
            }
            
            response = await self.client.post("/api/v1/trust/suspicious", json=payload)
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            if e.response.status_code == 404:
                raise DTRUSTError(f"Node not found: {node_id}")
            raise DTRUSTError(f"HTTP error recording suspicious activity: {e}")
    
    # Federation methods
    async def get_federation_stats(self) -> Dict[str, Any]:
        """Get federation statistics."""
        try:
            response = await self.client.get("/api/v1/federation/stats")
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise DTRUSTError(f"HTTP error getting federation stats: {e}")
    
    async def get_federation_info(self) -> FederationInfo:
        """Get federation information."""
        try:
            response = await self.client.get("/api/v1/federation/info")
            response.raise_for_status()
            
            data = response.json()
            return FederationInfo(**data)
            
        except httpx.HTTPError as e:
            raise DTRUSTError(f"HTTP error getting federation info: {e}")
    
    # Convenience methods for common workflows
    async def register_and_attest_node(self, enclave_id: str) -> tuple[NodeIdentity, AttestationResult]:
        """Register a node and immediately request attestation."""
        # Register node
        identity = await self.register_node(enclave_id)
        
        # Request attestation
        attestation = await self.request_attestation(identity.node_id)
        
        return identity, attestation
    
    async def wait_for_trusted_status(
        self, 
        node_id: str, 
        timeout: float = 60.0,
        check_interval: float = 2.0
    ) -> NodeState:
        """Wait for a node to achieve trusted status."""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            node_state = await self.get_node_state(node_id)
            
            if node_state.can_participate_in_fl:
                return node_state
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise DTRUSTError(
                    f"Timeout waiting for node {node_id} to achieve trusted status"
                )
            
            await asyncio.sleep(check_interval)
    
    async def monitor_federation(
        self, 
        callback: Optional[callable] = None,
        interval: float = 10.0
    ):
        """Monitor federation status and call callback with updates."""
        while True:
            try:
                stats = await self.get_federation_stats()
                
                if callback:
                    callback(stats)
                else:
                    logger.info(f"Federation stats: {stats}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring federation: {e}")
                await asyncio.sleep(interval)


# Synchronous wrapper for convenience
class DTRUSTClientSync:
    """Synchronous wrapper for DTRUSTClient."""
    
    def __init__(self, *args, **kwargs):
        self._client = DTRUSTClient(*args, **kwargs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._client.close())
    
    def _run_async(self, coro):
        """Run async method synchronously."""
        return asyncio.run(coro)
    
    def health_check(self) -> Dict[str, str]:
        return self._run_async(self._client.health_check())
    
    def register_node(self, enclave_id: str, node_metadata: Optional[Dict[str, Any]] = None) -> NodeIdentity:
        return self._run_async(self._client.register_node(enclave_id, node_metadata))
    
    def get_node_state(self, node_id: str) -> NodeState:
        return self._run_async(self._client.get_node_state(node_id))
    
    def list_nodes(self) -> List[str]:
        return self._run_async(self._client.list_nodes())
    
    def request_attestation(self, node_id: str, challenge: Optional[str] = None) -> AttestationResult:
        return self._run_async(self._client.request_attestation(node_id, challenge))
    
    def get_trusted_nodes(self, min_trust_score: Optional[float] = None) -> List[NodeState]:
        return self._run_async(self._client.get_trusted_nodes(min_trust_score))
    
    def get_suspicious_nodes(self) -> List[NodeState]:
        return self._run_async(self._client.get_suspicious_nodes())
    
    def record_training_participation(self, node_id: str, num_examples: int, metrics: Dict[str, float]) -> Dict[str, str]:
        return self._run_async(self._client.record_training_participation(node_id, num_examples, metrics))
    
    def record_suspicious_activity(self, node_id: str, reason: str) -> Dict[str, str]:
        return self._run_async(self._client.record_suspicious_activity(node_id, reason))
    
    def get_federation_stats(self) -> Dict[str, Any]:
        return self._run_async(self._client.get_federation_stats())
    
    def get_federation_info(self) -> FederationInfo:
        return self._run_async(self._client.get_federation_info())
    
    def register_and_attest_node(self, enclave_id: str) -> tuple[NodeIdentity, AttestationResult]:
        return self._run_async(self._client.register_and_attest_node(enclave_id)) 