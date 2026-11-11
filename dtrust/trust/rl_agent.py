"""
Reinforcement Learning Agent for Dynamic Trust Management in DTRUST.

This module implements a Q-learning based agent that learns optimal trust policies
for federated learning scenarios, adapting to client behavior and performance patterns.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.models import TrustMetrics, NodeStatus
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLState:
    """State representation for the RL agent."""
    avg_trust_score: float
    trust_score_variance: float
    accuracy_improvement: float
    loss_reduction: float
    client_participation_rate: float
    suspicious_activity_level: float
    round_number: int
    total_clients: int
    active_clients: int


@dataclass
class RLAction:
    """Action representation for the RL agent."""
    trust_weight_adjustment: float  # [-0.2, 0.2]
    min_trust_threshold_adjustment: float  # [-0.1, 0.1]
    client_selection_strategy: str  # 'trust_based', 'performance_based', 'hybrid'
    aggregation_weight_adjustment: float  # [-0.15, 0.15]


@dataclass
class RLExperience:
    """Experience tuple for RL training."""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
    timestamp: datetime


class TrustQNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TrustRLAgent:
    """
    Reinforcement Learning agent for dynamic trust management.
    
    Uses Q-learning with experience replay and neural network approximation
    to learn optimal trust policies for federated learning.
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 4,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = TrustQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = TrustQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        
        # Action space definitions
        self.action_space = {
            'trust_weight_adjustment': [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
            'min_trust_threshold_adjustment': [-0.1, -0.05, 0.0, 0.05, 0.1],
            'client_selection_strategy': ['trust_based', 'performance_based', 'hybrid'],
            'aggregation_weight_adjustment': [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
        }
        
        logger.info(f"TrustRLAgent initialized with {state_dim} state dimensions and {action_dim} action dimensions")
        logger.info(f"Device: {self.device}")
    
    def get_state(self, round_metrics: Dict[str, Any]) -> RLState:
        """Convert round metrics to RL state representation."""
        return RLState(
            avg_trust_score=round_metrics.get('avg_trust_score', 0.5),
            trust_score_variance=round_metrics.get('trust_score_variance', 0.0),
            accuracy_improvement=round_metrics.get('accuracy_improvement', 0.0),
            loss_reduction=round_metrics.get('loss_reduction', 0.0),
            client_participation_rate=round_metrics.get('participation_rate', 0.8),
            suspicious_activity_level=round_metrics.get('suspicious_activity', 0.0),
            round_number=round_metrics.get('round', 0),
            total_clients=round_metrics.get('total_clients', 0),
            active_clients=round_metrics.get('active_clients', 0)
        )
    
    def state_to_tensor(self, state: RLState) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        state_vector = [
            state.avg_trust_score,
            state.trust_score_variance,
            state.accuracy_improvement,
            state.loss_reduction,
            state.client_participation_rate,
            state.suspicious_activity_level,
            state.round_number / 100.0,  # Normalize round number
            state.active_clients / max(state.total_clients, 1),
            state.active_clients / max(state.total_clients, 1)  # Participation rate
        ]
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def select_action(self, state: RLState, training: bool = True) -> RLAction:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return self._random_action()
        else:
            # Exploitation: best action according to Q-network
            return self._best_action(state)
    
    def _random_action(self) -> RLAction:
        """Generate a random action."""
        return RLAction(
            trust_weight_adjustment=random.choice(self.action_space['trust_weight_adjustment']),
            min_trust_threshold_adjustment=random.choice(self.action_space['min_trust_threshold_adjustment']),
            client_selection_strategy=random.choice(self.action_space['client_selection_strategy']),
            aggregation_weight_adjustment=random.choice(self.action_space['aggregation_weight_adjustment'])
        )
    
    def _best_action(self, state: RLState) -> RLAction:
        """Select the best action according to Q-network."""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        # Convert action index to RLAction
        return self._action_index_to_action(action_idx)
    
    def _action_index_to_action(self, action_idx: int) -> RLAction:
        """Convert action index to RLAction object."""
        # Flatten action space for indexing
        trust_weights = self.action_space['trust_weight_adjustment']
        threshold_adjustments = self.action_space['min_trust_threshold_adjustment']
        strategies = self.action_space['client_selection_strategy']
        aggregation_adjustments = self.action_space['aggregation_weight_adjustment']
        
        # Calculate indices
        total_actions = len(trust_weights) * len(threshold_adjustments) * len(strategies) * len(aggregation_adjustments)
        
        if action_idx >= total_actions:
            action_idx = 0  # Fallback to first action
        
        # Convert to individual indices
        idx1 = action_idx % len(trust_weights)
        idx2 = (action_idx // len(trust_weights)) % len(threshold_adjustments)
        idx3 = (action_idx // (len(trust_weights) * len(threshold_adjustments))) % len(strategies)
        idx4 = (action_idx // (len(trust_weights) * len(threshold_adjustments) * len(strategies))) % len(aggregation_adjustments)
        
        return RLAction(
            trust_weight_adjustment=trust_weights[idx1],
            min_trust_threshold_adjustment=threshold_adjustments[idx2],
            client_selection_strategy=strategies[idx3],
            aggregation_weight_adjustment=aggregation_adjustments[idx4]
        )
    
    def calculate_reward(self, round_metrics: Dict[str, Any], action: RLAction) -> float:
        """
        Calculate reward based on federated learning performance and trust metrics.
        
        Reward components:
        1. Model accuracy improvement (primary objective)
        2. Training stability (loss reduction)
        3. Trust score quality and consistency
        4. Client participation efficiency
        5. Action cost (penalty for large changes)
        """
        # Base reward from accuracy improvement
        accuracy_gain = round_metrics.get('accuracy_improvement', 0.0)
        accuracy_reward = accuracy_gain * 20.0  # High weight for primary objective
        
        # Stability reward (penalize high loss variance)
        loss_variance = round_metrics.get('loss_variance', 0.0)
        stability_reward = -loss_variance * 10.0  # Penalty for instability
        
        # Trust quality reward
        avg_trust_score = round_metrics.get('avg_trust_score', 0.5)
        trust_variance = round_metrics.get('trust_score_variance', 0.0)
        trust_reward = (avg_trust_score - 0.5) * 5.0 - trust_variance * 8.0
        
        # Participation efficiency reward
        participation_rate = round_metrics.get('participation_rate', 0.8)
        participation_reward = (participation_rate - 0.5) * 3.0
        
        # Suspicious activity penalty
        suspicious_activity = round_metrics.get('suspicious_activity', 0.0)
        security_penalty = -suspicious_activity * 15.0
        
        # Action cost (penalty for large policy changes)
        action_cost = -(
            abs(action.trust_weight_adjustment) * 2.0 +
            abs(action.min_trust_threshold_adjustment) * 3.0 +
            abs(action.aggregation_weight_adjustment) * 2.0
        )
        
        # Combined reward
        total_reward = (
            accuracy_reward +
            stability_reward +
            trust_reward +
            participation_reward +
            security_penalty +
            action_cost
        )
        
        return total_reward
    
    def store_experience(self, experience: RLExperience):
        """Store experience in replay buffer."""
        self.memory.append(experience)
    
    def train(self) -> Optional[float]:
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.cat([self.state_to_tensor(exp.state) for exp in batch])
        next_states = torch.cat([self.state_to_tensor(exp.next_state) for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Convert actions to indices
        action_indices = torch.LongTensor([
            self._action_to_index(exp.action) for exp in batch
        ]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        
        # Next Q-values (for target calculation)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _action_to_index(self, action: RLAction) -> int:
        """Convert RLAction to index for neural network output."""
        trust_weights = self.action_space['trust_weight_adjustment']
        threshold_adjustments = self.action_space['min_trust_threshold_adjustment']
        strategies = self.action_space['client_selection_strategy']
        aggregation_adjustments = self.action_space['aggregation_weight_adjustment']
        
        # Find closest matches
        idx1 = min(range(len(trust_weights)), key=lambda i: abs(trust_weights[i] - action.trust_weight_adjustment))
        idx2 = min(range(len(threshold_adjustments)), key=lambda i: abs(threshold_adjustments[i] - action.min_trust_threshold_adjustment))
        idx3 = strategies.index(action.client_selection_strategy)
        idx4 = min(range(len(aggregation_adjustments)), key=lambda i: abs(aggregation_adjustments[i] - action.aggregation_weight_adjustment))
        
        # Convert to single index
        return (
            idx1 +
            idx2 * len(trust_weights) +
            idx3 * len(trust_weights) * len(threshold_adjustments) +
            idx4 * len(trust_weights) * len(threshold_adjustments) * len(strategies)
        )
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of current policy and training statistics."""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'total_reward': self.total_reward,
            'episode_count': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards
        }, filepath)
        logger.info(f"RL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.total_reward = checkpoint['total_reward']
        self.episode_rewards = checkpoint['episode_rewards']
        logger.info(f"RL model loaded from {filepath}") 