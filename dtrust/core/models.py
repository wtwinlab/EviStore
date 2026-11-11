"""
Core data models for DTRUST framework.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel as PydanticBaseModel, Field
import uuid


class TrustLevel(Enum):
    """Trust levels for nodes."""
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class NodeStatus(Enum):
    """Node status in the federation."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    ATTESTING = "attesting"
    TRUSTED = "trusted"
    ACTIVE = "active"
    SUSPICIOUS = "suspicious"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    BANNED = "banned"


class AttestationStatus(Enum):
    """Attestation result status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    EXPIRED = "expired"
    ERROR = "error"


class BaseModel(PydanticBaseModel):
    """Base model with common fields."""
    
    class Config:
        use_enum_values = True
        protected_namespaces = ()  # Allow fields with 'model_' prefix
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


@dataclass
class SGXQuote:
    """SGX Quote data structure."""
    quote: bytes
    signature: bytes
    report_data: bytes
    enclave_measurement: str
    signer_measurement: str
    product_id: int
    security_version: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NodeIdentity(BaseModel):
    """Node identity information."""
    node_id: str = Field(..., description="Unique node identifier")
    enclave_id: str = Field(..., description="SGX enclave identifier")
    public_key: str = Field(..., description="Node's public key (PEM format)")
    private_key_hash: str = Field(..., description="Hash of private key for verification")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def age_hours(self) -> float:
        """Get age of identity in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600


class AttestationResult(BaseModel):
    """Attestation result information."""
    node_id: str = Field(..., description="Node identifier")
    attestation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AttestationStatus = Field(..., description="Attestation status")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    trust_level: TrustLevel = Field(..., description="Determined trust level")
    
    # SGX-specific data
    quote_data: Optional[Dict[str, Any]] = Field(None, description="SGX quote information")
    enclave_measurement: Optional[str] = Field(None, description="Enclave measurement")
    signer_measurement: Optional[str] = Field(None, description="Signer measurement")
    
    # Verification details
    verification_time: datetime = Field(default_factory=datetime.utcnow)
    expiry_time: datetime = Field(..., description="When attestation expires")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    
    # Metadata
    attestation_service: str = Field("intel_ias", description="Attestation service used")
    nonce: Optional[str] = Field(None, description="Challenge nonce")
    
    @property
    def is_valid(self) -> bool:
        """Check if attestation is still valid."""
        return datetime.utcnow() < self.expiry_time and self.status == AttestationStatus.SUCCESS
    
    @property
    def expires_in_hours(self) -> float:
        """Get hours until expiration."""
        delta = self.expiry_time - datetime.utcnow()
        return max(0, delta.total_seconds() / 3600)


class TrustMetrics(BaseModel):
    """Trust metrics for a node."""
    node_id: str = Field(..., description="Node identifier")
    
    # Core trust scores
    attestation_score: float = Field(0.0, ge=0.0, le=1.0, description="Attestation-based trust")
    performance_score: float = Field(0.0, ge=0.0, le=1.0, description="Performance-based trust")
    behavior_score: float = Field(0.0, ge=0.0, le=1.0, description="Behavior-based trust")
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall trust score")
    
    # Interaction statistics
    total_interactions: int = Field(0, description="Total interactions")
    successful_interactions: int = Field(0, description="Successful interactions")
    failed_interactions: int = Field(0, description="Failed interactions")
    suspicious_count: int = Field(0, description="Number of suspicious activities")
    
    # Time-based metrics
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    last_attestation: Optional[datetime] = Field(None, description="Last successful attestation")
    
    # Federated learning metrics
    training_rounds_participated: int = Field(0, description="FL rounds participated")
    training_rounds_completed: int = Field(0, description="FL rounds completed")
    average_training_time: float = Field(0.0, description="Average training time in seconds")
    model_quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Model contribution quality")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions
    
    @property
    def trust_score(self) -> float:
        """Alias for overall_score for backward compatibility."""
        return self.overall_score
    
    @property
    def trust_level(self) -> TrustLevel:
        """Determine trust level based on overall score."""
        if self.overall_score >= 0.9:
            return TrustLevel.VERIFIED
        elif self.overall_score >= 0.7:
            return TrustLevel.HIGH
        elif self.overall_score >= 0.5:
            return TrustLevel.MEDIUM
        elif self.overall_score >= 0.3:
            return TrustLevel.LOW
        elif self.overall_score > 0.0:
            return TrustLevel.UNTRUSTED
        else:
            return TrustLevel.UNKNOWN
    
    @property
    def is_suspicious(self) -> bool:
        """Check if node should be considered suspicious."""
        return (
            self.suspicious_count >= 3 or
            self.success_rate < 0.5 or
            self.overall_score < 0.3
        )


class NodeState(BaseModel):
    """Complete node state information."""
    identity: NodeIdentity = Field(..., description="Node identity")
    status: NodeStatus = Field(NodeStatus.UNREGISTERED, description="Current status")
    trust_metrics: TrustMetrics = Field(..., description="Trust metrics")
    current_attestation: Optional[AttestationResult] = Field(None, description="Current attestation")
    attestation_history: List[AttestationResult] = Field(default_factory=list, description="Attestation history")
    
    # Federation participation
    federation_joined_at: Optional[datetime] = Field(None, description="When joined federation")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @property
    def needs_attestation(self) -> bool:
        """Check if node needs new attestation."""
        if not self.current_attestation:
            return True
        return not self.current_attestation.is_valid
    
    @property
    def can_participate_in_fl(self) -> bool:
        """Check if node can participate in federated learning."""
        return (
            self.status in [NodeStatus.TRUSTED, NodeStatus.ACTIVE] and
            self.current_attestation and
            self.current_attestation.is_valid and
            self.trust_metrics.overall_score >= 0.5 and
            not self.trust_metrics.is_suspicious
        )


class FederationInfo(BaseModel):
    """Information about the federation."""
    federation_id: str = Field(..., description="Federation identifier")
    coordinator_node_id: str = Field(..., description="Coordinator node ID")
    status: str = Field("active", description="Federation status")
    
    # Participation stats
    total_nodes: int = Field(0, description="Total registered nodes")
    active_nodes: int = Field(0, description="Currently active nodes")
    trusted_nodes: int = Field(0, description="Trusted nodes")
    suspicious_nodes: int = Field(0, description="Suspicious nodes")
    
    # Training info
    current_round: int = Field(0, description="Current training round")
    total_rounds: int = Field(10, description="Total planned rounds")
    round_start_time: Optional[datetime] = Field(None, description="Current round start time")
    
    # Trust requirements
    min_trust_score: float = Field(0.7, description="Minimum trust score for participation")
    min_participants: int = Field(2, description="Minimum participants per round")
    max_participants: int = Field(100, description="Maximum participants per round")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingTask(BaseModel):
    """Federated learning training task."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int = Field(..., description="Training round number")
    node_id: str = Field(..., description="Assigned node ID")
    
    # Task parameters
    model_parameters: Dict[str, Any] = Field(..., description="Model parameters")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    dataset_info: Dict[str, Any] = Field(..., description="Dataset information")
    
    # Timing
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="When training started")
    completed_at: Optional[datetime] = Field(None, description="When training completed")
    deadline: datetime = Field(..., description="Task deadline")
    
    # Results
    result_parameters: Optional[Dict[str, Any]] = Field(None, description="Training results")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    status: str = Field("assigned", description="Task status")
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        return datetime.utcnow() > self.deadline and self.status != "completed"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None 