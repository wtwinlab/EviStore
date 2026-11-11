"""Static-weighted trust scorer (baseline for comparison with the SPRT scorer)."""

import math
from datetime import datetime, timedelta
from typing import Dict, Any

from ..core.models import TrustMetrics, AttestationResult, AttestationStatus
from ..core.config import TrustConfig


class TrustScorer:
    """
    Trust scoring system that calculates trust scores based on:
    - Attestation results and confidence
    - Performance in federated learning
    - Behavioral patterns and reliability
    """
    
    def __init__(self, config: TrustConfig):
        self.config = config
        
        # Scoring weights
        self.attestation_weight = 0.4
        self.performance_weight = 0.3
        self.behavior_weight = 0.3
    
    def update_attestation_score(
        self, 
        trust_metrics: TrustMetrics, 
        attestation: AttestationResult
    ):
        """Update attestation-based trust score."""
        # AttestationResult stores status as a plain string ("success") because the
        # model sets use_enum_values=True; comparing to the enum member itself is
        # always False -> every attestation would wrongly take the failure branch.
        # Compare against the enum's .value (matches manager.py's "success" check).
        if attestation.status == AttestationStatus.SUCCESS.value:
            # Base score from attestation confidence
            base_score = attestation.confidence
            
            # Bonus for high-confidence attestations
            if attestation.confidence >= 0.9:
                base_score = min(1.0, base_score + 0.1)
            
            # Consider trust level
            trust_level_bonus = {
                "verified": 0.1,
                "high": 0.05,
                "medium": 0.0,
                "low": -0.1,
                "untrusted": -0.2,
                "unknown": -0.1
            }.get(attestation.trust_level, 0.0)
            
            base_score = max(0.0, min(1.0, base_score + trust_level_bonus))
            
            # Update with exponential moving average
            alpha = 0.3  # Learning rate
            trust_metrics.attestation_score = (
                alpha * base_score + 
                (1 - alpha) * trust_metrics.attestation_score
            )
            
        else:
            # Failed attestation significantly reduces score
            trust_metrics.attestation_score *= 0.5
        
        trust_metrics.last_attestation = attestation.verification_time
    
    def update_performance_score(
        self, 
        trust_metrics: TrustMetrics, 
        num_examples: int,
        metrics: Dict[str, float]
    ):
        """Update performance-based trust score."""
        # Base performance score from participation
        participation_score = min(1.0, trust_metrics.training_rounds_participated / 10.0)
        
        # Success rate component
        success_rate = trust_metrics.success_rate
        
        # Training quality component (if available)
        quality_score = 0.5  # Default neutral score
        if "accuracy" in metrics:
            quality_score = min(1.0, metrics["accuracy"])
        elif "loss" in metrics:
            # Convert loss to quality score (assuming loss < 2.0 is good)
            quality_score = max(0.0, 1.0 - min(metrics["loss"] / 2.0, 1.0))
        
        # Data contribution component
        contribution_score = min(1.0, num_examples / 1000.0)  # Normalize to 1000 examples
        
        # Combine components
        new_performance = (
            0.3 * participation_score +
            0.3 * success_rate +
            0.2 * quality_score +
            0.2 * contribution_score
        )
        
        # Update with exponential moving average
        alpha = 0.2
        trust_metrics.performance_score = (
            alpha * new_performance + 
            (1 - alpha) * trust_metrics.performance_score
        )
        
        # Update training metrics
        if trust_metrics.training_rounds_participated > 0:
            # Update average training time (if available)
            if "training_time" in metrics:
                current_avg = trust_metrics.average_training_time
                count = trust_metrics.training_rounds_participated
                trust_metrics.average_training_time = (
                    (current_avg * (count - 1) + metrics["training_time"]) / count
                )
            
            # Update model quality score
            trust_metrics.model_quality_score = (
                alpha * quality_score + 
                (1 - alpha) * trust_metrics.model_quality_score
            )
    
    def update_behavior_score(
        self, 
        trust_metrics: TrustMetrics, 
        suspicious: bool = False,
        timeout: bool = False,
        communication_error: bool = False
    ):
        """Update behavior-based trust score."""
        # Start with current score
        behavior_score = trust_metrics.behavior_score
        
        if suspicious:
            # Significant penalty for suspicious behavior
            behavior_score *= 0.7
        
        if timeout:
            # Moderate penalty for timeouts
            behavior_score *= 0.9
        
        if communication_error:
            # Small penalty for communication issues
            behavior_score *= 0.95
        
        # Positive reinforcement for consistent good behavior
        if (trust_metrics.total_interactions > 5 and 
            trust_metrics.success_rate > 0.8 and
            trust_metrics.suspicious_count == 0):
            behavior_score = min(1.0, behavior_score + 0.01)
        
        # Time-based decay for old negative behaviors
        if trust_metrics.last_seen:
            hours_since_last_seen = (
                datetime.utcnow() - trust_metrics.last_seen
            ).total_seconds() / 3600
            
            if hours_since_last_seen > 168:  # 1 week
                # Gradual recovery from past issues
                recovery_factor = min(0.1, hours_since_last_seen / 1680)  # 10 weeks for full recovery
                behavior_score = min(1.0, behavior_score + recovery_factor)
        
        trust_metrics.behavior_score = max(0.0, behavior_score)
    
    def calculate_overall_score(self, trust_metrics: TrustMetrics):
        """Calculate overall trust score from components."""
        # Weighted combination of trust components
        overall_score = (
            self.attestation_weight * trust_metrics.attestation_score +
            self.performance_weight * trust_metrics.performance_score +
            self.behavior_weight * trust_metrics.behavior_score
        )
        
        # Apply penalties for suspicious behavior
        if trust_metrics.suspicious_count > 0:
            suspicion_penalty = min(0.5, trust_metrics.suspicious_count * 0.1)
            overall_score *= (1.0 - suspicion_penalty)
        
        # Apply time-based factors
        if trust_metrics.last_seen:
            hours_inactive = (
                datetime.utcnow() - trust_metrics.last_seen
            ).total_seconds() / 3600
            
            # Gradual decay for inactivity
            if hours_inactive > 24:
                inactivity_factor = max(0.5, 1.0 - (hours_inactive - 24) / 1680)  # 10 weeks
                overall_score *= inactivity_factor
        
        # Ensure score is within bounds
        trust_metrics.overall_score = max(0.0, min(1.0, overall_score))
    
    def get_trust_explanation(self, trust_metrics: TrustMetrics) -> Dict[str, Any]:
        """Get detailed explanation of trust score calculation."""
        return {
            "overall_score": trust_metrics.overall_score,
            "components": {
                "attestation": {
                    "score": trust_metrics.attestation_score,
                    "weight": self.attestation_weight,
                    "contribution": trust_metrics.attestation_score * self.attestation_weight
                },
                "performance": {
                    "score": trust_metrics.performance_score,
                    "weight": self.performance_weight,
                    "contribution": trust_metrics.performance_score * self.performance_weight
                },
                "behavior": {
                    "score": trust_metrics.behavior_score,
                    "weight": self.behavior_weight,
                    "contribution": trust_metrics.behavior_score * self.behavior_weight
                }
            },
            "factors": {
                "success_rate": trust_metrics.success_rate,
                "suspicious_count": trust_metrics.suspicious_count,
                "training_participation": trust_metrics.training_rounds_participated,
                "last_attestation": trust_metrics.last_attestation.isoformat() if trust_metrics.last_attestation else None,
                "last_seen": trust_metrics.last_seen.isoformat() if trust_metrics.last_seen else None
            },
            "trust_level": trust_metrics.trust_level,
            "is_suspicious": trust_metrics.is_suspicious
        } 