"""
Static Trust Scorer - Baseline trust models for comparison with dynamic trust.

This module provides several static trust models that can be used as baselines
to compare against the dynamic trust model in DTRUST.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from enum import Enum

from ..core.models import TrustMetrics, AttestationResult, AttestationStatus
from ..core.config import TrustConfig


class StaticTrustModel(Enum):
    """Types of static trust models available."""
    UNIFORM = "uniform"  # All nodes have equal trust
    RANDOM = "random"    # Random trust scores
    FIXED_HIGH = "fixed_high"  # All nodes have high trust (0.9)
    FIXED_MEDIUM = "fixed_medium"  # All nodes have medium trust (0.5)
    FIXED_LOW = "fixed_low"  # All nodes have low trust (0.3)
    FIXED_SCORE = "fixed_score"  # All nodes have a specific fixed trust score
    ATTESTATION_ONLY = "attestation_only"  # Trust based only on initial attestation
    PERFORMANCE_ONLY = "performance_only"  # Trust based only on performance
    TIME_BASED = "time_based"  # Trust decreases over time
    REPUTATION_BASED = "reputation_based"  # Trust based on historical success rate


class StaticTrustScorer:
    """
    Static trust scoring system that provides baseline trust models.
    
    Unlike the dynamic TrustScorer, this maintains fixed or simple trust patterns
    that don't adapt based on ongoing behavior and performance.
    """
    
    def __init__(self, config: TrustConfig, model_type: StaticTrustModel = StaticTrustModel.UNIFORM, fixed_score: float = 0.7):
        self.config = config
        self.model_type = model_type
        self.fixed_score = fixed_score  # For FIXED_SCORE model
        
        # Static trust parameters
        self.static_scores = {}
        self.initial_attestation_scores = {}
        self.performance_history = {}
        self.node_creation_times = {}
        
        # Model-specific parameters
        self.time_decay_rate = 0.01  # Per day decay for time-based model
        self.reputation_window = 10  # Number of interactions for reputation calculation
        
        logger.info(f"StaticTrustScorer initialized with model: {model_type.value}")
        if model_type == StaticTrustModel.FIXED_SCORE:
            logger.info(f"Fixed score set to: {fixed_score}")
    
    def get_static_trust_score(self, node_id: str) -> float:
        """Get the static trust score for a node based on the selected model."""
        if node_id not in self.static_scores:
            self._initialize_node_trust(node_id)
        
        return self.static_scores[node_id]
    
    def _initialize_node_trust(self, node_id: str):
        """Initialize trust score for a new node based on the selected model."""
        if self.model_type == StaticTrustModel.UNIFORM:
            self.static_scores[node_id] = 0.7  # Default medium trust
            
        elif self.model_type == StaticTrustModel.RANDOM:
            import random
            self.static_scores[node_id] = random.uniform(0.3, 0.9)
            
        elif self.model_type == StaticTrustModel.FIXED_HIGH:
            self.static_scores[node_id] = 0.9
            
        elif self.model_type == StaticTrustModel.FIXED_MEDIUM:
            self.static_scores[node_id] = 0.5
            
        elif self.model_type == StaticTrustModel.FIXED_LOW:
            self.static_scores[node_id] = 0.3
            
        elif self.model_type == StaticTrustModel.FIXED_SCORE:
            self.static_scores[node_id] = self.fixed_score
            
        elif self.model_type == StaticTrustModel.ATTESTATION_ONLY:
            # Will be set when first attestation occurs
            self.static_scores[node_id] = 0.5  # Default until attestation
            
        elif self.model_type == StaticTrustModel.PERFORMANCE_ONLY:
            # Will be updated based on performance
            self.static_scores[node_id] = 0.5  # Default until performance data
            
        elif self.model_type == StaticTrustModel.TIME_BASED:
            self.static_scores[node_id] = 1.0  # Start with full trust
            self.node_creation_times[node_id] = datetime.utcnow()
            
        elif self.model_type == StaticTrustModel.REPUTATION_BASED:
            self.static_scores[node_id] = 0.5  # Default until reputation builds
            self.performance_history[node_id] = []
        
        logger.debug(f"Initialized trust score for node {node_id}: {self.static_scores[node_id]}")
    
    def update_attestation_score(self, trust_metrics: TrustMetrics, attestation: AttestationResult):
        """Update attestation-based trust score (only for attestation-only model)."""
        if self.model_type == StaticTrustModel.ATTESTATION_ONLY:
            if attestation.status == AttestationStatus.SUCCESS:
                # Set static score based on attestation confidence
                self.static_scores[trust_metrics.node_id] = attestation.confidence
                self.initial_attestation_scores[trust_metrics.node_id] = attestation.confidence
            else:
                # Failed attestation gets low trust
                self.static_scores[trust_metrics.node_id] = 0.1
                self.initial_attestation_scores[trust_metrics.node_id] = 0.1
        
        # For other models, attestation doesn't affect static scores
        trust_metrics.last_attestation = attestation.verification_time
    
    def update_performance_score(self, trust_metrics: TrustMetrics, num_examples: int, metrics: Dict[str, float]):
        """Update performance-based trust score (only for performance-only and reputation-based models)."""
        node_id = trust_metrics.node_id
        
        if self.model_type == StaticTrustModel.PERFORMANCE_ONLY:
            # Simple performance-based scoring
            if 'loss' in metrics:
                # Lower loss = higher trust
                loss = metrics['loss']
                if loss < 0.1:
                    self.static_scores[node_id] = 0.9
                elif loss < 0.3:
                    self.static_scores[node_id] = 0.7
                elif loss < 0.5:
                    self.static_scores[node_id] = 0.5
                else:
                    self.static_scores[node_id] = 0.3
        
        elif self.model_type == StaticTrustModel.REPUTATION_BASED:
            # Track performance history for reputation calculation
            if node_id not in self.performance_history:
                self.performance_history[node_id] = []
            
            # Add current performance to history
            performance_score = self._calculate_performance_score(metrics)
            self.performance_history[node_id].append(performance_score)
            
            # Keep only recent history
            if len(self.performance_history[node_id]) > self.reputation_window:
                self.performance_history[node_id] = self.performance_history[node_id][-self.reputation_window:]
            
            # Calculate reputation as average of recent performance
            if self.performance_history[node_id]:
                reputation = sum(self.performance_history[node_id]) / len(self.performance_history[node_id])
                self.static_scores[node_id] = reputation
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a performance score from metrics."""
        score = 0.5  # Default score
        
        if 'loss' in metrics:
            loss = metrics['loss']
            # Convert loss to score (lower loss = higher score)
            score = max(0.1, min(1.0, 1.0 - loss))
        
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            # Use accuracy directly
            score = max(score, accuracy)
        
        return score
    
    def update_behavior_score(self, trust_metrics: TrustMetrics, behavior_data: Dict[str, Any]):
        """Update behavior-based trust score (minimal for static models)."""
        # Static models generally don't update based on behavior
        # Only reputation-based model might consider some behavior aspects
        if self.model_type == StaticTrustModel.REPUTATION_BASED:
            # Simple behavior scoring based on suspicious activities
            suspicious_count = behavior_data.get('suspicious_count', 0)
            if suspicious_count > 0:
                # Reduce trust based on suspicious activities
                reduction = min(0.3, suspicious_count * 0.1)
                self.static_scores[trust_metrics.node_id] = max(0.1, 
                    self.static_scores[trust_metrics.node_id] - reduction)
    
    def calculate_overall_score(self, trust_metrics: TrustMetrics) -> float:
        """Calculate overall trust score based on the static model."""
        node_id = trust_metrics.node_id
        
        if self.model_type == StaticTrustModel.TIME_BASED:
            # Apply time-based decay
            if node_id in self.node_creation_times:
                creation_time = self.node_creation_times[node_id]
                days_since_creation = (datetime.utcnow() - creation_time).days
                decay = self.time_decay_rate * days_since_creation
                return max(0.1, self.static_scores[node_id] - decay)
        
        # For other models, return the static score
        return self.static_scores.get(node_id, 0.5)
    
    def get_model_description(self) -> str:
        """Get a description of the current static trust model."""
        descriptions = {
            StaticTrustModel.UNIFORM: "All nodes have equal trust scores (0.7)",
            StaticTrustModel.RANDOM: "Nodes have random trust scores between 0.3-0.9",
            StaticTrustModel.FIXED_HIGH: "All nodes have high trust scores (0.9)",
            StaticTrustModel.FIXED_MEDIUM: "All nodes have medium trust scores (0.5)",
            StaticTrustModel.FIXED_LOW: "All nodes have low trust scores (0.3)",
            StaticTrustModel.FIXED_SCORE: f"All nodes have fixed trust score ({self.fixed_score})",
            StaticTrustModel.ATTESTATION_ONLY: "Trust based only on initial attestation confidence",
            StaticTrustModel.PERFORMANCE_ONLY: "Trust based only on training performance metrics",
            StaticTrustModel.TIME_BASED: "Trust decreases linearly over time",
            StaticTrustModel.REPUTATION_BASED: "Trust based on historical performance average"
        }
        return descriptions.get(self.model_type, "Unknown static trust model")
    
    def reset_node_trust(self, node_id: str):
        """Reset trust score for a node (useful for experiments)."""
        if node_id in self.static_scores:
            del self.static_scores[node_id]
        if node_id in self.initial_attestation_scores:
            del self.initial_attestation_scores[node_id]
        if node_id in self.performance_history:
            del self.performance_history[node_id]
        if node_id in self.node_creation_times:
            del self.node_creation_times[node_id]
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all current trust scores."""
        return self.static_scores.copy()


# Import logger at the end to avoid circular imports
from ..core.logging import get_logger
logger = get_logger(__name__) 