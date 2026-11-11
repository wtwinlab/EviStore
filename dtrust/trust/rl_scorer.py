"""
RL-Enhanced Trust Scorer for DTRUST.

This module extends the basic trust scoring system with reinforcement learning
capabilities to dynamically adapt trust policies based on federated learning performance.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from .scorer import TrustScorer
from .rl_agent import TrustRLAgent, RLState, RLAction, RLExperience
from ..core.models import TrustMetrics, AttestationResult, AttestationStatus
from ..core.config import TrustConfig
from ..core.logging import get_logger

logger = get_logger(__name__)


class RLTrustScorer(TrustScorer):
    """
    Reinforcement Learning enhanced trust scorer.
    
    Extends the basic TrustScorer with RL capabilities to learn optimal
    trust policies for federated learning scenarios.
    """
    
    def __init__(
        self, 
        config: TrustConfig,
        rl_agent: Optional[TrustRLAgent] = None,
        enable_rl: bool = True,
        rl_update_frequency: int = 5,
        save_rl_model: bool = True,
        rl_model_path: str = "models/rl_trust_model.pth"
    ):
        super().__init__(config)
        
        self.enable_rl = enable_rl
        self.rl_update_frequency = rl_update_frequency
        self.save_rl_model = save_rl_model
        self.rl_model_path = Path(rl_model_path)
        
        # Initialize RL agent
        if rl_agent is None and enable_rl:
            self.rl_agent = TrustRLAgent()
        else:
            self.rl_agent = rl_agent
        
        # RL state tracking
        self.current_round = 0
        self.round_history = []
        self.last_action = None
        self.last_state = None
        self.episode_rewards = []
        
        # Dynamic policy parameters (adjusted by RL)
        self.dynamic_trust_weight = self.attestation_weight
        self.dynamic_min_trust_score = 0.5
        self.dynamic_client_selection_strategy = 'hybrid'
        self.dynamic_aggregation_weight = 0.3
        
        # Performance tracking for RL
        self.performance_history = {
            'accuracy': [],
            'loss': [],
            'trust_scores': [],
            'participation_rates': [],
            'suspicious_activities': []
        }
        
        # Load pre-trained model if available
        if self.rl_model_path.exists() and enable_rl:
            try:
                self.rl_agent.load_model(str(self.rl_model_path))
                logger.info(f"Loaded pre-trained RL model from {self.rl_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load RL model: {e}")
        
        logger.info(f"RLTrustScorer initialized with RL enabled: {enable_rl}")
    
    def update_attestation_score(
        self, 
        trust_metrics: TrustMetrics, 
        attestation: AttestationResult
    ):
        """Update attestation-based trust score with RL enhancements."""
        # Use base implementation
        super().update_attestation_score(trust_metrics, attestation)
        
        # Apply RL-based adjustments if enabled
        if self.enable_rl and self.rl_agent:
            self._apply_rl_attestation_adjustment(trust_metrics, attestation)
    
    def update_performance_score(
        self, 
        trust_metrics: TrustMetrics, 
        num_examples: int,
        metrics: Dict[str, float]
    ):
        """Update performance-based trust score with RL enhancements."""
        # Use base implementation
        super().update_performance_score(trust_metrics, num_examples, metrics)
        
        # Track performance for RL
        self._track_performance_metrics(metrics)
        
        # Apply RL-based adjustments if enabled
        if self.enable_rl and self.rl_agent:
            self._apply_rl_performance_adjustment(trust_metrics, metrics)
    
    def update_behavior_score(
        self, 
        trust_metrics: TrustMetrics, 
        suspicious: bool = False,
        timeout: bool = False,
        communication_error: bool = False
    ):
        """Update behavior-based trust score with RL enhancements."""
        # Use base implementation
        super().update_behavior_score(trust_metrics, suspicious, timeout, communication_error)
        
        # Track suspicious activity for RL
        if suspicious:
            self.performance_history['suspicious_activities'].append(1.0)
        else:
            self.performance_history['suspicious_activities'].append(0.0)
    
    def calculate_overall_score(self, trust_metrics: TrustMetrics):
        """Calculate overall trust score with RL-based dynamic weighting."""
        if self.enable_rl and self.rl_agent:
            # Use RL-adjusted weights
            overall_score = (
                self.dynamic_trust_weight * trust_metrics.attestation_score +
                self.dynamic_aggregation_weight * trust_metrics.performance_score +
                (1.0 - self.dynamic_trust_weight - self.dynamic_aggregation_weight) * trust_metrics.behavior_score
            )
        else:
            # Use base implementation
            overall_score = super().calculate_overall_score(trust_metrics)
        
        # Apply penalties and bounds
        if trust_metrics.suspicious_count > 0:
            suspicion_penalty = min(0.5, trust_metrics.suspicious_count * 0.1)
            overall_score *= (1.0 - suspicion_penalty)
        
        # Time-based decay
        if trust_metrics.last_seen:
            hours_inactive = (
                datetime.utcnow() - trust_metrics.last_seen
            ).total_seconds() / 3600
            
            if hours_inactive > 24:
                inactivity_factor = max(0.5, 1.0 - (hours_inactive - 24) / 1680)
                overall_score *= inactivity_factor
        
        trust_metrics.overall_score = max(0.0, min(1.0, overall_score))
    
    def process_round_completion(
        self, 
        round_number: int, 
        round_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process federated learning round completion and update RL agent.
        
        Args:
            round_number: Current round number
            round_metrics: Metrics from the completed round
            
        Returns:
            Updated round metrics with RL insights
        """
        self.current_round = round_number
        
        # Calculate round improvements
        round_metrics = self._calculate_round_improvements(round_metrics)
        
        # Update performance history
        self._update_performance_history(round_metrics)
        
        # RL processing
        if self.enable_rl and self.rl_agent:
            rl_metrics = self._process_rl_round(round_metrics)
            round_metrics.update(rl_metrics)
        
        # Store round history
        self.round_history.append({
            'round': round_number,
            'metrics': round_metrics,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return round_metrics
    
    def _process_rl_round(self, round_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process round with RL agent and return RL-specific metrics."""
        # Get current state
        current_state = self.rl_agent.get_state(round_metrics)
        
        # Select action
        action = self.rl_agent.select_action(current_state, training=True)
        
        # Apply action to adjust policy parameters
        self._apply_rl_action(action)
        
        # Calculate reward
        reward = self.rl_agent.calculate_reward(round_metrics, action)
        
        # Store experience for training
        if self.last_state is not None and self.last_action is not None:
            experience = RLExperience(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=current_state,
                done=False,  # Not the final round
                timestamp=datetime.utcnow()
            )
            self.rl_agent.store_experience(experience)
        
        # Train RL agent periodically
        if self.current_round % self.rl_update_frequency == 0:
            loss = self.rl_agent.train()
            if loss is not None:
                logger.info(f"RL training loss at round {self.current_round}: {loss:.4f}")
        
        # Update tracking variables
        self.last_state = current_state
        self.last_action = action
        self.episode_rewards.append(reward)
        
        # Save model periodically
        if self.save_rl_model and self.current_round % 50 == 0:
            self.rl_model_path.parent.mkdir(exist_ok=True)
            self.rl_agent.save_model(str(self.rl_model_path))
        
        return {
            'rl_reward': reward,
            'rl_action': {
                'trust_weight_adjustment': action.trust_weight_adjustment,
                'min_trust_threshold_adjustment': action.min_trust_threshold_adjustment,
                'client_selection_strategy': action.client_selection_strategy,
                'aggregation_weight_adjustment': action.aggregation_weight_adjustment
            },
            'rl_policy_summary': self.rl_agent.get_policy_summary(),
            'dynamic_trust_weight': self.dynamic_trust_weight,
            'dynamic_min_trust_score': self.dynamic_min_trust_score,
            'dynamic_client_selection_strategy': self.dynamic_client_selection_strategy,
            'dynamic_aggregation_weight': self.dynamic_aggregation_weight
        }
    
    def _apply_rl_action(self, action: RLAction):
        """Apply RL action to adjust policy parameters."""
        # Adjust trust weight
        self.dynamic_trust_weight = max(0.1, min(0.8, 
            self.dynamic_trust_weight + action.trust_weight_adjustment))
        
        # Adjust minimum trust threshold
        self.dynamic_min_trust_score = max(0.1, min(0.9,
            self.dynamic_min_trust_score + action.min_trust_threshold_adjustment))
        
        # Update client selection strategy
        self.dynamic_client_selection_strategy = action.client_selection_strategy
        
        # Adjust aggregation weight
        self.dynamic_aggregation_weight = max(0.1, min(0.8,
            self.dynamic_aggregation_weight + action.aggregation_weight_adjustment))
        
        logger.debug(f"Applied RL action: trust_weight={self.dynamic_trust_weight:.3f}, "
                    f"min_threshold={self.dynamic_min_trust_score:.3f}, "
                    f"strategy={self.dynamic_client_selection_strategy}, "
                    f"agg_weight={self.dynamic_aggregation_weight:.3f}")
    
    def _calculate_round_improvements(self, round_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements from previous round."""
        if len(self.performance_history['accuracy']) < 2:
            round_metrics['accuracy_improvement'] = 0.0
            round_metrics['loss_reduction'] = 0.0
            return round_metrics
        
        # Calculate accuracy improvement
        current_accuracy = round_metrics.get('avg_accuracy', 0.0)
        previous_accuracy = self.performance_history['accuracy'][-1]
        accuracy_improvement = max(0.0, current_accuracy - previous_accuracy)
        
        # Calculate loss reduction
        current_loss = round_metrics.get('avg_loss', 1.0)
        previous_loss = self.performance_history['loss'][-1]
        loss_reduction = max(0.0, previous_loss - current_loss)
        
        # Calculate trust score variance
        trust_scores = round_metrics.get('trust_scores', [])
        trust_variance = np.var(trust_scores) if len(trust_scores) > 1 else 0.0
        
        # Calculate loss variance
        loss_variance = round_metrics.get('loss_variance', 0.0)
        
        # Calculate participation rate
        active_clients = round_metrics.get('active_clients', 0)
        total_clients = round_metrics.get('total_clients', 1)
        participation_rate = active_clients / total_clients
        
        # Calculate suspicious activity level
        suspicious_activity = round_metrics.get('suspicious_activity', 0.0)
        
        round_metrics.update({
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'trust_score_variance': trust_variance,
            'loss_variance': loss_variance,
            'participation_rate': participation_rate,
            'suspicious_activity': suspicious_activity
        })
        
        return round_metrics
    
    def _update_performance_history(self, round_metrics: Dict[str, Any]):
        """Update performance history with current round metrics."""
        self.performance_history['accuracy'].append(round_metrics.get('avg_accuracy', 0.0))
        self.performance_history['loss'].append(round_metrics.get('avg_loss', 1.0))
        self.performance_history['trust_scores'].append(round_metrics.get('avg_trust_score', 0.5))
        self.performance_history['participation_rates'].append(round_metrics.get('participation_rate', 0.8))
        
        # Keep only recent history (last 100 rounds)
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def _track_performance_metrics(self, metrics: Dict[str, float]):
        """Track performance metrics for RL analysis."""
        if 'accuracy' in metrics:
            self.performance_history['accuracy'].append(metrics['accuracy'])
        if 'loss' in metrics:
            self.performance_history['loss'].append(metrics['loss'])
    
    def _apply_rl_attestation_adjustment(self, trust_metrics: TrustMetrics, attestation: AttestationResult):
        """Apply RL-based adjustments to attestation scoring."""
        # RL can adjust attestation confidence based on historical performance
        if len(self.performance_history['accuracy']) > 5:
            recent_accuracy_trend = np.mean(self.performance_history['accuracy'][-5:])
            
            # Adjust attestation score based on performance trend
            if recent_accuracy_trend > 0.8:
                # High performance: boost attestation confidence
                trust_metrics.attestation_score = min(1.0, trust_metrics.attestation_score * 1.05)
            elif recent_accuracy_trend < 0.5:
                # Low performance: reduce attestation confidence
                trust_metrics.attestation_score = max(0.0, trust_metrics.attestation_score * 0.95)
    
    def _apply_rl_performance_adjustment(self, trust_metrics: TrustMetrics, metrics: Dict[str, float]):
        """Apply RL-based adjustments to performance scoring."""
        # RL can adjust performance scoring based on overall system performance
        if len(self.performance_history['accuracy']) > 3:
            system_accuracy_trend = np.mean(self.performance_history['accuracy'][-3:])
            
            # Adjust performance score based on system trend
            if system_accuracy_trend > 0.7:
                # System performing well: boost individual performance
                trust_metrics.performance_score = min(1.0, trust_metrics.performance_score * 1.02)
            elif system_accuracy_trend < 0.4:
                # System struggling: be more critical of individual performance
                trust_metrics.performance_score = max(0.0, trust_metrics.performance_score * 0.98)
    
    def get_rl_insights(self) -> Dict[str, Any]:
        """Get insights from RL agent for analysis and debugging."""
        if not self.enable_rl or not self.rl_agent:
            return {'rl_enabled': False}
        
        policy_summary = self.rl_agent.get_policy_summary()
        
        return {
            'rl_enabled': True,
            'policy_summary': policy_summary,
            'current_policy': {
                'trust_weight': self.dynamic_trust_weight,
                'min_trust_threshold': self.dynamic_min_trust_score,
                'client_selection_strategy': self.dynamic_client_selection_strategy,
                'aggregation_weight': self.dynamic_aggregation_weight
            },
            'performance_trends': {
                'avg_accuracy': np.mean(self.performance_history['accuracy'][-10:]) if self.performance_history['accuracy'] else 0.0,
                'avg_loss': np.mean(self.performance_history['loss'][-10:]) if self.performance_history['loss'] else 1.0,
                'avg_trust_score': np.mean(self.performance_history['trust_scores'][-10:]) if self.performance_history['trust_scores'] else 0.5,
                'avg_participation_rate': np.mean(self.performance_history['participation_rates'][-10:]) if self.performance_history['participation_rates'] else 0.8
            },
            'recent_rewards': self.episode_rewards[-10:] if self.episode_rewards else [],
            'round_count': self.current_round
        }
    
    def save_rl_state(self, filepath: str):
        """Save RL state and performance history."""
        rl_state = {
            'performance_history': self.performance_history,
            'round_history': self.round_history,
            'episode_rewards': self.episode_rewards,
            'current_round': self.current_round,
            'dynamic_policy': {
                'trust_weight': self.dynamic_trust_weight,
                'min_trust_threshold': self.dynamic_min_trust_score,
                'client_selection_strategy': self.dynamic_client_selection_strategy,
                'aggregation_weight': self.dynamic_aggregation_weight
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(rl_state, f, indent=2, default=str)
        
        logger.info(f"RL state saved to {filepath}")
    
    def load_rl_state(self, filepath: str):
        """Load RL state and performance history."""
        try:
            with open(filepath, 'r') as f:
                rl_state = json.load(f)
            
            self.performance_history = rl_state['performance_history']
            self.round_history = rl_state['round_history']
            self.episode_rewards = rl_state['episode_rewards']
            self.current_round = rl_state['current_round']
            
            dynamic_policy = rl_state['dynamic_policy']
            self.dynamic_trust_weight = dynamic_policy['trust_weight']
            self.dynamic_min_trust_score = dynamic_policy['min_trust_threshold']
            self.dynamic_client_selection_strategy = dynamic_policy['client_selection_strategy']
            self.dynamic_aggregation_weight = dynamic_policy['aggregation_weight']
            
            logger.info(f"RL state loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to load RL state: {e}") 