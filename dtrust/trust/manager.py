"""Per-node trust state, attestation binding, and decay callbacks."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from ..core.models import (
    NodeState, NodeStatus, TrustMetrics, AttestationResult, 
    NodeIdentity, TrustLevel
)
from ..core.exceptions import TrustError, NodeError
from ..core.config import TrustConfig
from ..core.logging import get_logger, log_exception, async_catch_exceptions
from .scorer import TrustScorer
from .binder import AttestationBinder


logger = get_logger(__name__)


@dataclass
class TrustEvent:
    """Trust-related event for callbacks."""
    event_type: str
    node_id: str
    old_state: Optional[NodeState]
    new_state: NodeState
    timestamp: datetime
    metadata: Dict[str, Any]


class TrustManager:
    """
    Central trust management system for DTRUST.
    
    Manages node trust scores, attestation bindings, and dynamic trust updates.
    Provides callbacks for trust state changes and automatic trust decay.
    """
    
    def __init__(
        self, 
        config: TrustConfig,
        scorer: Optional[TrustScorer] = None,
        binder: Optional[AttestationBinder] = None
    ):
        self.config = config
        self.scorer = scorer or TrustScorer(config)
        self.binder = binder or AttestationBinder(config)
        
        # Node state storage (in production, this would be a database)
        self._node_states: Dict[str, NodeState] = {}
        
        # Event callbacks
        self._callbacks: List[Callable[[TrustEvent], None]] = []
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
    
    @async_catch_exceptions(reraise=True)
    async def start(self):
        """Start the trust manager and background tasks."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting trust manager")
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._trust_decay_task()),
            asyncio.create_task(self._attestation_renewal_task()),
            asyncio.create_task(self._suspicious_detection_task())
        ]
    
    @async_catch_exceptions(reraise=True)
    async def stop(self):
        """Stop the trust manager and cleanup."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping trust manager")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
    
    def register_callback(self, callback: Callable[[TrustEvent], None]):
        """Register a callback for trust events."""
        self._callbacks.append(callback)
    
    def create_node_state(self, identity: NodeIdentity) -> NodeState:
        """Create initial node state."""
        trust_metrics = TrustMetrics(node_id=identity.node_id)
        
        node_state = NodeState(
            identity=identity,
            status=NodeStatus.REGISTERED,
            trust_metrics=trust_metrics
        )
        
        self._node_states[identity.node_id] = node_state
        
        self._emit_event("node_registered", identity.node_id, None, node_state)
        
        logger.info(f"Created node state for {identity.node_id}")
        return node_state
    
    def get_node_state(self, node_id: str) -> Optional[NodeState]:
        """Get current node state."""
        return self._node_states.get(node_id)
    
    def update_attestation_result(
        self, 
        node_id: str, 
        attestation: AttestationResult
    ) -> NodeState:
        """Update node state with new attestation result."""
        node_state = self._node_states.get(node_id)
        if not node_state:
            raise NodeError(f"Node {node_id} not found", node_id)
        
        old_state = NodeState(**node_state.dict())
        
        # Update attestation
        node_state.current_attestation = attestation
        node_state.attestation_history.append(attestation)
        
        # Update trust metrics based on attestation
        self.scorer.update_attestation_score(node_state.trust_metrics, attestation)
        
        # Update status based on attestation result
        if attestation.status == "success":
            if node_state.status == NodeStatus.REGISTERED:
                node_state.status = NodeStatus.TRUSTED
            elif node_state.status == NodeStatus.ATTESTING:
                node_state.status = NodeStatus.TRUSTED
        else:
            node_state.status = NodeStatus.UNTRUSTED
        
        # Recalculate overall trust score
        self.scorer.calculate_overall_score(node_state.trust_metrics)
        
        # Update last activity
        node_state.last_activity = datetime.utcnow()
        
        self._emit_event("attestation_updated", node_id, old_state, node_state)
        
        logger.info(
            f"Updated attestation for {node_id}: "
            f"status={attestation.status}, "
            f"trust_score={node_state.trust_metrics.overall_score:.3f}"
        )
        
        return node_state
    
    def record_training_participation(
        self, 
        node_id: str, 
        num_examples: int,
        metrics: Dict[str, float]
    ) -> NodeState:
        """Record federated learning training participation."""
        node_state = self._node_states.get(node_id)
        if not node_state:
            raise NodeError(f"Node {node_id} not found", node_id)
        
        old_state = NodeState(**node_state.dict())
        
        # Update training metrics
        trust_metrics = node_state.trust_metrics
        trust_metrics.training_rounds_participated += 1
        trust_metrics.total_interactions += 1
        trust_metrics.successful_interactions += 1
        trust_metrics.last_seen = datetime.utcnow()
        
        # Update performance score based on participation
        self.scorer.update_performance_score(trust_metrics, num_examples, metrics)
        
        # Update status to active if trusted
        if node_state.status == NodeStatus.TRUSTED:
            node_state.status = NodeStatus.ACTIVE
        
        # Recalculate overall score
        self.scorer.calculate_overall_score(trust_metrics)
        
        node_state.last_activity = datetime.utcnow()
        
        self._emit_event("training_participation", node_id, old_state, node_state)
        
        logger.debug(f"Recorded training participation for {node_id}")
        return node_state
    
    def record_suspicious_activity(self, node_id: str, reason: str) -> NodeState:
        """Record suspicious activity for a node."""
        node_state = self._node_states.get(node_id)
        if not node_state:
            raise NodeError(f"Node {node_id} not found", node_id)
        
        old_state = NodeState(**node_state.dict())
        
        # Update suspicious count
        trust_metrics = node_state.trust_metrics
        trust_metrics.suspicious_count += 1
        trust_metrics.failed_interactions += 1
        trust_metrics.total_interactions += 1
        
        # Update behavior score
        self.scorer.update_behavior_score(trust_metrics, suspicious=True)
        
        # Check if node should be marked as suspicious
        if trust_metrics.is_suspicious:
            node_state.status = NodeStatus.SUSPICIOUS
        
        # Recalculate overall score
        self.scorer.calculate_overall_score(trust_metrics)
        
        # Auto-isolate if configured
        if (self.config.auto_isolation and 
            trust_metrics.suspicious_count >= self.config.max_suspicious_count):
            node_state.status = NodeStatus.BANNED
        
        node_state.last_activity = datetime.utcnow()
        
        self._emit_event("suspicious_activity", node_id, old_state, node_state, {
            "reason": reason,
            "suspicious_count": trust_metrics.suspicious_count
        })
        
        logger.warning(
            f"Recorded suspicious activity for {node_id}: {reason} "
            f"(count: {trust_metrics.suspicious_count})"
        )
        
        return node_state
    
    def get_trusted_nodes(self, min_trust_score: Optional[float] = None) -> List[NodeState]:
        """Get list of trusted nodes that can participate in FL."""
        min_score = min_trust_score or self.config.min_trust_score
        
        trusted_nodes = []
        for node_state in self._node_states.values():
            if (node_state.can_participate_in_fl and 
                node_state.trust_metrics.overall_score >= min_score):
                trusted_nodes.append(node_state)
        
        return trusted_nodes
    
    def get_suspicious_nodes(self) -> List[NodeState]:
        """Get list of suspicious nodes."""
        return [
            node_state for node_state in self._node_states.values()
            if node_state.status == NodeStatus.SUSPICIOUS
        ]
    
    def get_all_nodes(self) -> List[NodeState]:
        """Get list of all nodes."""
        return list(self._node_states.values())
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get federation trust statistics."""
        total_nodes = len(self._node_states)
        active_nodes = sum(1 for n in self._node_states.values() 
                          if n.status == NodeStatus.ACTIVE)
        trusted_nodes = sum(1 for n in self._node_states.values() 
                           if n.status in [NodeStatus.TRUSTED, NodeStatus.ACTIVE])
        suspicious_nodes = sum(1 for n in self._node_states.values() 
                              if n.status == NodeStatus.SUSPICIOUS)
        
        avg_trust_score = 0.0
        if total_nodes > 0:
            avg_trust_score = sum(n.trust_metrics.overall_score 
                                 for n in self._node_states.values()) / total_nodes
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "trusted_nodes": trusted_nodes,
            "suspicious_nodes": suspicious_nodes,
            "average_trust_score": avg_trust_score,
            "min_trust_threshold": self.config.min_trust_score
        }
    
    @async_catch_exceptions()
    async def _trust_decay_task(self):
        """Background task to apply trust decay over time."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                current_time = datetime.utcnow()
                
                for node_state in self._node_states.values():
                    if node_state.status in [NodeStatus.ACTIVE, NodeStatus.TRUSTED]:
                        # Apply time-based trust decay
                        hours_since_activity = (
                            current_time - node_state.last_activity
                        ).total_seconds() / 3600
                        
                        if hours_since_activity > 24:  # Decay after 24 hours of inactivity
                            old_state = NodeState(**node_state.dict())
                            
                            decay_factor = min(hours_since_activity / 168, 0.5)  # Max 50% decay over 1 week
                            decay_amount = self.config.trust_decay_rate * decay_factor
                            
                            # Apply decay to all trust components
                            trust_metrics = node_state.trust_metrics
                            trust_metrics.performance_score = max(
                                0.0, trust_metrics.performance_score - decay_amount
                            )
                            trust_metrics.behavior_score = max(
                                0.0, trust_metrics.behavior_score - decay_amount
                            )
                            
                            # Recalculate overall score
                            self.scorer.calculate_overall_score(trust_metrics)
                            
                            # Update status if trust dropped too low
                            if trust_metrics.overall_score < self.config.min_trust_score:
                                if node_state.status == NodeStatus.ACTIVE:
                                    node_state.status = NodeStatus.INACTIVE
                            
                            self._emit_event("trust_decay", node_state.identity.node_id, 
                                           old_state, node_state)
                
            except Exception as e:
                log_exception(e, "Error in trust decay task")
    
    @async_catch_exceptions()
    async def _attestation_renewal_task(self):
        """Background task to check for attestation renewal needs."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes
                
                for node_state in self._node_states.values():
                    if node_state.needs_attestation:
                        old_state = NodeState(**node_state.dict())
                        node_state.status = NodeStatus.EXPIRED
                        
                        self._emit_event("attestation_expired", 
                                       node_state.identity.node_id, old_state, node_state)
                        
                        logger.info(f"Attestation expired for {node_state.identity.node_id}")
                
            except Exception as e:
                log_exception(e, "Error in attestation renewal task")
    
    @async_catch_exceptions()
    async def _suspicious_detection_task(self):
        """Background task for advanced suspicious behavior detection."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Implement advanced anomaly detection here
                # For now, just check basic patterns
                
                for node_state in self._node_states.values():
                    trust_metrics = node_state.trust_metrics
                    
                    # Check for unusual patterns
                    if (trust_metrics.total_interactions > 10 and
                        trust_metrics.success_rate < 0.3):
                        
                        if node_state.status not in [NodeStatus.SUSPICIOUS, NodeStatus.BANNED]:
                            self.record_suspicious_activity(
                                node_state.identity.node_id, 
                                "low_success_rate_pattern"
                            )
                
            except Exception as e:
                log_exception(e, "Error in suspicious detection task")
    
    def _emit_event(
        self, 
        event_type: str, 
        node_id: str, 
        old_state: Optional[NodeState], 
        new_state: NodeState,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Emit trust event to registered callbacks."""
        event = TrustEvent(
            event_type=event_type,
            node_id=node_id,
            old_state=old_state,
            new_state=new_state,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                log_exception(e, f"Error in trust event callback for {event_type}") 