"""
RL-Enhanced Federated Learning Strategy for DTRUST.

This module implements a federated learning strategy that uses reinforcement learning
for dynamic trust management and client selection optimization.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from ..core.models import NodeStatus, TrustLevel
from ..core.exceptions import FederatedLearningError
from ..trust.rl_scorer import RLTrustScorer
from ..trust.manager import TrustManager
from ..core.logging import get_logger, log_exception

import time
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

logger = get_logger(__name__)


class RLTrustAwareStrategy(FedAvg):
    """
    RL-Enhanced federated learning strategy with dynamic trust management.
    
    This strategy extends the basic TrustAwareStrategy with reinforcement learning
    capabilities to automatically optimize trust policies and client selection
    based on federated learning performance.
    """
    
    def __init__(
        self,
        trust_manager: TrustManager,
        rl_scorer: RLTrustScorer,
        min_trust_score: float = 0.5,
        trust_weight: float = 0.3,
        auto_replace_suspicious: bool = True,
        max_suspicious_rounds: int = 2,
        results_dir: str = "rl_training_results",
        num_rounds: int = 3,
        rl_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize RL-enhanced trust-aware strategy.
        
        Args:
            trust_manager: Trust management system
            rl_scorer: RL-enhanced trust scorer
            min_trust_score: Minimum trust score for participation
            trust_weight: How much trust affects aggregation weights
            auto_replace_suspicious: Whether to auto-replace suspicious clients
            max_suspicious_rounds: Max rounds before replacing suspicious clients
            results_dir: Directory to save training results JSON files
            num_rounds: Total number of training rounds
            rl_enabled: Whether RL is enabled for dynamic policy adjustment
            **kwargs: Additional arguments passed to FedAvg
        """
        # Set up evaluation metrics aggregation if not provided
        if 'evaluate_metrics_aggregation_fn' not in kwargs:
            kwargs['evaluate_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        if 'fit_metrics_aggregation_fn' not in kwargs:
            kwargs['fit_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        super().__init__(**kwargs)
        self.trust_manager = trust_manager
        self.rl_scorer = rl_scorer
        self.min_trust_score = min_trust_score
        self.trust_weight = trust_weight
        self.auto_replace_suspicious = auto_replace_suspicious
        self.max_suspicious_rounds = max_suspicious_rounds
        self.num_rounds = num_rounds
        self.rl_enabled = rl_enabled
        
        # Results saving configuration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.training_start_time = datetime.now()
        
        # Track client performance and RL insights
        self.client_performance: Dict[str, Dict] = {}
        self.suspicious_clients: Dict[str, int] = {}
        self.replaced_clients: List[str] = []
        self.rl_insights_history = []
        
        # Training progress tracking
        self.current_round = 0
        self.training_history = []
        self.client_participation = {}
        self.training_completed = False
        
        # RL-specific tracking
        self.round_metrics_history = []
        self.policy_evolution = []
        
        logger.info(f"RLTrustAwareStrategy initialized with RL enabled: {rl_enabled}")
        logger.info(f"Training will run for {num_rounds} rounds")
        logger.info(f"Training results will be saved to: {self.results_dir}")
    
    def _weighted_average_metrics(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate metrics using weighted averaging with RL enhancements."""
        if not metrics:
            return {}
        
        # Get all unique metric names
        metric_names = set()
        for _, metric_dict in metrics:
            metric_names.update(metric_dict.keys())
        
        aggregated = {}
        
        for metric_name in metric_names:
            # Get values and weights for this metric
            values_and_weights = []
            for num_examples, metric_dict in metrics:
                if metric_name in metric_dict:
                    try:
                        # Try to convert to float for numeric metrics
                        value = metric_dict[metric_name]
                        if isinstance(value, (int, float)):
                            values_and_weights.append((float(value), num_examples))
                        else:
                            # Skip non-numeric metrics
                            logger.debug(f"Skipping non-numeric metric '{metric_name}': {value}")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert metric '{metric_name}' value '{metric_dict[metric_name]}' to float: {e}")
                        continue
            
            if values_and_weights:
                # Weighted average
                total_value = sum(value * weight for value, weight in values_and_weights)
                total_weight = sum(weight for _, weight in values_and_weights)
                aggregated[metric_name] = total_value / total_weight if total_weight > 0 else 0.0
            else:
                logger.debug(f"No valid numeric values found for metric '{metric_name}'")
        
        return aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with RL-enhanced client selection."""
        self.current_round = server_round
        
        # Get available clients
        available_clients = list(client_manager.all().values())
        
        if not available_clients:
            logger.warning("No clients available for training")
            return []
        
        # Apply RL-enhanced client selection if enabled
        if self.rl_enabled and self.rl_scorer:
            selected_clients = self._rl_enhanced_client_selection(available_clients, server_round)
        else:
            # Fall back to basic trust-based selection
            selected_clients = self._basic_trust_client_selection(available_clients)
        
        if not selected_clients:
            logger.warning("No clients selected for training")
            return []
        
        # Track participation
        for client in selected_clients:
            client_id = client.cid
            if client_id not in self.client_participation:
                self.client_participation[client_id] = []
            self.client_participation[client_id].append(server_round)
        
        # Configure fit parameters
        config = self._get_fit_config(server_round)
        fit_ins = FitIns(parameters, config)
        
        logger.info(f"Round {server_round}: Selected {len(selected_clients)} clients for training")
        logger.debug(f"Selected clients: {[c.cid for c in selected_clients]}")
        
        return [(client, fit_ins) for client in selected_clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation with RL-enhanced client selection."""
        # Get available clients
        available_clients = list(client_manager.all().values())
        
        if not available_clients:
            logger.warning("No clients available for evaluation")
            return []
        
        # Apply RL-enhanced client selection if enabled
        if self.rl_enabled and self.rl_scorer:
            selected_clients = self._rl_enhanced_client_selection(available_clients, server_round, is_evaluation=True)
        else:
            # Fall back to basic trust-based selection
            selected_clients = self._basic_trust_client_selection(available_clients)
        
        if not selected_clients:
            logger.warning("No clients selected for evaluation")
            return []
        
        # Configure evaluation parameters
        config = self._get_evaluate_config(server_round)
        evaluate_ins = EvaluateIns(parameters, config)
        
        logger.info(f"Round {server_round}: Selected {len(selected_clients)} clients for evaluation")
        
        return [(client, evaluate_ins) for client in selected_clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using RL-enhanced trust-weighted aggregation."""
        if not results:
            logger.warning("No successful fit results to aggregate")
            return None, {}
        
        # Process failures and update trust scores
        self._process_failures(failures)
        
        # Update client performance tracking
        for client, fit_res in results:
            self._update_client_performance(client.cid, fit_res, "fit")
        
        # Apply RL-enhanced trust weighting
        if self.rl_enabled and self.rl_scorer:
            weighted_results = self._apply_rl_trust_weights(results, "fit")
        else:
            weighted_results = self._apply_basic_trust_weights(results, "fit")
        
        # Use Flower's built-in aggregation with weighted results
        aggregated_ndarrays = aggregate(weighted_results)
        
        # Convert aggregated ndarrays back to Parameters object
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate metrics
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results if fit_res.metrics]
            if metrics_list:
                aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
        
        # Add trust-specific metrics
        trust_metrics = self._aggregate_trust_metrics(results)
        aggregated_metrics.update(trust_metrics)
        
        # Process round completion with RL if enabled
        if self.rl_enabled and self.rl_scorer:
            round_metrics = self._prepare_round_metrics(server_round, results, aggregated_metrics)
            enhanced_metrics = self.rl_scorer.process_round_completion(server_round, round_metrics)
            aggregated_metrics.update(enhanced_metrics)
            
            # Store RL insights
            rl_insights = self.rl_scorer.get_rl_insights()
            self.rl_insights_history.append({
                'round': server_round,
                'insights': rl_insights,
                'timestamp': datetime.now().isoformat()
            })
        
        # Track training history
        self.training_history.append({
            "round": server_round,
            "num_clients": len(results),
            "metrics": aggregated_metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save intermediate results
        self._save_intermediate_results(server_round)
        
        logger.info(f"Round {server_round}: Aggregated fit results from {len(results)} clients")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using RL-enhanced trust-weighted aggregation."""
        if not results:
            logger.warning("No successful evaluation results to aggregate")
            return None, {}
        
        # Process failures and update trust scores
        self._process_failures(failures)
        
        # Update client performance tracking
        for client, eval_res in results:
            self._update_client_performance(client.cid, eval_res, "evaluate")
        
        # Apply RL-enhanced trust weighting
        if self.rl_enabled and self.rl_scorer:
            weighted_results = self._apply_rl_trust_weights(results, "evaluate")
        else:
            weighted_results = self._apply_basic_trust_weights(results, "evaluate")
        
        # Use Flower's built-in weighted loss aggregation
        aggregated_loss = weighted_loss_avg(weighted_results)
        
        # Aggregate metrics
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            metrics_list = [(eval_res.num_examples, eval_res.metrics) for _, eval_res in results if eval_res.metrics]
            if metrics_list:
                aggregated_metrics = self.evaluate_metrics_aggregation_fn(metrics_list)
        
        # Add trust-specific metrics
        trust_metrics = self._aggregate_trust_metrics(results)
        aggregated_metrics.update(trust_metrics)
        
        logger.info(f"Round {server_round}: Aggregated evaluation results from {len(results)} clients")
        logger.info(f"Round {server_round}: Loss = {aggregated_loss:.4f}")
        
        return aggregated_loss, aggregated_metrics
    
    def _rl_enhanced_client_selection(
        self, 
        available_clients: List[ClientProxy], 
        server_round: int,
        is_evaluation: bool = False
    ) -> List[ClientProxy]:
        """Select clients using RL-enhanced strategy."""
        # Get current RL policy
        if self.rl_scorer:
            strategy = self.rl_scorer.dynamic_client_selection_strategy
            min_trust_threshold = self.rl_scorer.dynamic_min_trust_score
        else:
            strategy = 'hybrid'
            min_trust_threshold = self.min_trust_score
        
        # Filter clients by minimum trust threshold
        trusted_clients = []
        for client in available_clients:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                
                if trust_score >= min_trust_threshold:
                    trusted_clients.append((client, trust_score))
                else:
                    logger.debug(f"Client {client.cid} excluded: trust_score={trust_score:.3f} < {min_trust_threshold}")
            except Exception as e:
                logger.warning(f"Error getting trust score for {client.cid}: {e}")
                # Include with default trust score
                trusted_clients.append((client, 0.5))
        
        if not trusted_clients:
            return []
        
        # Apply strategy-specific selection
        if strategy == 'trust_based':
            selected_clients = self._select_by_trust(trusted_clients, is_evaluation)
        elif strategy == 'performance_based':
            selected_clients = self._select_by_performance(trusted_clients, is_evaluation)
        else:  # hybrid
            selected_clients = self._select_hybrid(trusted_clients, is_evaluation)
        
        return selected_clients
    
    def _select_by_trust(self, trusted_clients: List[Tuple[ClientProxy, float]], is_evaluation: bool) -> List[ClientProxy]:
        """Select clients based on trust scores."""
        # Sort by trust score (descending)
        trusted_clients.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        sample_size = min(
            len(trusted_clients),
            max(1, int(len(trusted_clients) * (self.fraction_evaluate if is_evaluation else self.fraction_fit)))
        )
        
        return [client for client, _ in trusted_clients[:sample_size]]
    
    def _select_by_performance(self, trusted_clients: List[Tuple[ClientProxy, float]], is_evaluation: bool) -> List[ClientProxy]:
        """Select clients based on historical performance."""
        # Get performance scores
        client_performance_scores = []
        for client, trust_score in trusted_clients:
            performance_data = self.client_performance.get(client.cid, {})
            
            # Calculate performance score from historical data
            if 'accuracy' in performance_data:
                performance_score = performance_data['accuracy']
            elif 'loss' in performance_data:
                performance_score = 1.0 - min(performance_data['loss'], 1.0)
            else:
                performance_score = trust_score  # Fall back to trust score
            
            client_performance_scores.append((client, performance_score))
        
        # Sort by performance score (descending)
        client_performance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        sample_size = min(
            len(client_performance_scores),
            max(1, int(len(client_performance_scores) * (self.fraction_evaluate if is_evaluation else self.fraction_fit)))
        )
        
        return [client for client, _ in client_performance_scores[:sample_size]]
    
    def _select_hybrid(self, trusted_clients: List[Tuple[ClientProxy, float]], is_evaluation: bool) -> List[ClientProxy]:
        """Select clients using hybrid trust-performance approach."""
        # Calculate hybrid scores
        client_hybrid_scores = []
        for client, trust_score in trusted_clients:
            performance_data = self.client_performance.get(client.cid, {})
            
            # Get performance score
            if 'accuracy' in performance_data:
                performance_score = performance_data['accuracy']
            elif 'loss' in performance_data:
                performance_score = 1.0 - min(performance_data['loss'], 1.0)
            else:
                performance_score = 0.5  # Default performance score
            
            # Hybrid score: 60% trust + 40% performance
            hybrid_score = 0.6 * trust_score + 0.4 * performance_score
            client_hybrid_scores.append((client, hybrid_score))
        
        # Sort by hybrid score (descending)
        client_hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        sample_size = min(
            len(client_hybrid_scores),
            max(1, int(len(client_hybrid_scores) * (self.fraction_evaluate if is_evaluation else self.fraction_fit)))
        )
        
        return [client for client, _ in client_hybrid_scores[:sample_size]]
    
    def _basic_trust_client_selection(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Basic trust-based client selection (fallback)."""
        trusted_clients = []
        
        for client in available_clients:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                
                if trust_score >= self.min_trust_score:
                    trusted_clients.append(client)
                else:
                    logger.debug(f"Client {client.cid} excluded: trust_score={trust_score:.3f} < {self.min_trust_score}")
            except Exception as e:
                logger.warning(f"Error getting trust score for {client.cid}: {e}")
        
        # Sample clients
        sample_size = min(
            len(trusted_clients),
            max(1, int(len(available_clients) * self.fraction_fit))
        )
        
        return trusted_clients[:sample_size]
    
    def _apply_rl_trust_weights(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]], 
        result_type: str
    ) -> List[Tuple[Union[List, float], int]]:
        """Apply RL-enhanced trust weights to aggregation."""
        weighted_results = []
        
        for client, result in results:
            try:
                # Get trust score
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                
                # Get RL-adjusted aggregation weight
                if self.rl_scorer:
                    aggregation_weight = self.rl_scorer.dynamic_aggregation_weight
                else:
                    aggregation_weight = self.trust_weight
                
                # Calculate trust-adjusted weight
                base_weight = result.num_examples
                trust_adjustment = 1.0 + (trust_score - 0.5) * aggregation_weight
                adjusted_weight = int(base_weight * max(0.1, trust_adjustment))
                
                if result_type == "fit":
                    # For fit results, extract parameters and convert to ndarrays
                    parameters = result.parameters
                    parameters_ndarrays = parameters_to_ndarrays(parameters)
                    weighted_results.append((parameters_ndarrays, adjusted_weight))
                else:
                    # For evaluate results, extract loss
                    loss = result.loss
                    weighted_results.append((loss, adjusted_weight))
                
                logger.debug(f"RL Client {client.cid}: trust={trust_score:.3f}, "
                           f"base_weight={base_weight}, adjusted_weight={adjusted_weight}")
                
            except Exception as e:
                logger.warning(f"Error applying RL trust weight for {client.cid}: {e}")
                # Fall back to original weight
                if result_type == "fit":
                    parameters_ndarrays = parameters_to_ndarrays(result.parameters)
                    weighted_results.append((parameters_ndarrays, result.num_examples))
                else:
                    weighted_results.append((result.loss, result.num_examples))
        
        return weighted_results
    
    def _apply_basic_trust_weights(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]], 
        result_type: str
    ) -> List[Tuple[Union[List, float], int]]:
        """Apply basic trust weights to aggregation (fallback)."""
        weighted_results = []
        
        for client, result in results:
            try:
                # Get trust score
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                
                # Calculate trust-adjusted weight
                base_weight = result.num_examples
                trust_adjustment = 1.0 + (trust_score - 0.5) * self.trust_weight
                adjusted_weight = int(base_weight * max(0.1, trust_adjustment))
                
                if result_type == "fit":
                    parameters_ndarrays = parameters_to_ndarrays(result.parameters)
                    weighted_results.append((parameters_ndarrays, adjusted_weight))
                else:
                    weighted_results.append((result.loss, adjusted_weight))
                
            except Exception as e:
                logger.warning(f"Error applying basic trust weight for {client.cid}: {e}")
                # Fall back to original weight
                if result_type == "fit":
                    parameters_ndarrays = parameters_to_ndarrays(result.parameters)
                    weighted_results.append((parameters_ndarrays, result.num_examples))
                else:
                    weighted_results.append((result.loss, result.num_examples))
        
        return weighted_results
    
    def _prepare_round_metrics(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]],
        aggregated_metrics: Dict[str, Scalar]
    ) -> Dict[str, Any]:
        """Prepare round metrics for RL processing."""
        # Calculate basic metrics
        total_examples = sum(result.num_examples for _, result in results)
        trust_scores = []
        
        for client, _ in results:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                trust_scores.append(trust_score)
            except Exception:
                trust_scores.append(0.5)
        
        # Calculate trust statistics
        avg_trust_score = np.mean(trust_scores) if trust_scores else 0.5
        trust_variance = np.var(trust_scores) if len(trust_scores) > 1 else 0.0
        
        # Calculate participation rate
        total_clients = len(self.trust_manager.get_all_node_states())
        participation_rate = len(results) / max(total_clients, 1)
        
        # Calculate suspicious activity level
        suspicious_count = sum(1 for client_id in self.suspicious_clients.values() if self.suspicious_clients[client_id] > 0)
        suspicious_activity = suspicious_count / max(len(results), 1)
        
        # Prepare round metrics
        round_metrics = {
            'round': server_round,
            'avg_accuracy': aggregated_metrics.get('avg_accuracy', 0.0),
            'avg_loss': aggregated_metrics.get('avg_loss', 1.0),
            'avg_trust_score': avg_trust_score,
            'trust_scores': trust_scores,
            'trust_score_variance': trust_variance,
            'participation_rate': participation_rate,
            'suspicious_activity': suspicious_activity,
            'total_examples': total_examples,
            'active_clients': len(results),
            'total_clients': total_clients,
            'loss_variance': aggregated_metrics.get('loss_variance', 0.0)
        }
        
        return round_metrics
    
    def _aggregate_trust_metrics(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]]
    ) -> Dict[str, Scalar]:
        """Aggregate trust-specific metrics."""
        if not results:
            return {}
        
        trust_scores = []
        total_examples = 0
        
        for client, result in results:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                trust_scores.append(trust_score)
                total_examples += result.num_examples
            except Exception:
                trust_scores.append(0.5)
                total_examples += result.num_examples
        
        # Calculate trust metrics
        avg_trust_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        min_trust_score = min(trust_scores) if trust_scores else 0.0
        max_trust_score = max(trust_scores) if trust_scores else 0.0
        
        return {
            "avg_trust_score": avg_trust_score,
            "min_trust_score": min_trust_score,
            "max_trust_score": max_trust_score,
            "total_examples": total_examples,
            "num_clients": len(results)
        }
    
    def _update_client_performance(self, client_id: str, result: Union[FitRes, EvaluateRes], result_type: str):
        """Update client performance tracking."""
        if client_id not in self.client_performance:
            self.client_performance[client_id] = {}
        
        # Update performance metrics
        if hasattr(result, 'metrics') and result.metrics:
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    self.client_performance[client_id][key] = float(value)
        
        # Update trust manager
        try:
            if result_type == "fit":
                self.trust_manager.record_training_participation(
                    client_id, result.num_examples, result.metrics or {}
                )
        except Exception as e:
            logger.warning(f"Error updating trust manager for {client_id}: {e}")
    
    def _process_failures(self, failures: List[Union[Tuple[ClientProxy, Union[FitRes, EvaluateRes]], BaseException]]):
        """Process client failures and update trust scores."""
        for failure in failures:
            if isinstance(failure, tuple):
                client, _ = failure
                client_id = client.cid
                
                # Mark as suspicious
                if client_id not in self.suspicious_clients:
                    self.suspicious_clients[client_id] = 0
                self.suspicious_clients[client_id] += 1
                
                # Update trust manager
                try:
                    self.trust_manager.record_suspicious_activity(client_id, "training_failure")
                except Exception as e:
                    logger.warning(f"Error recording suspicious activity for {client_id}: {e}")
    
    def _save_intermediate_results(self, server_round: int):
        """Save intermediate training results."""
        if server_round % 10 == 0:  # Save every 10 rounds
            results_file = self.results_dir / f"intermediate_results_round_{server_round}.json"
            
            results_data = {
                "round": server_round,
                "training_history": self.training_history,
                "client_performance": self.client_performance,
                "rl_insights": self.rl_insights_history,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Intermediate results saved to {results_file}")
    
    def save_final_results(self):
        """Save final training results and RL insights."""
        final_results_file = self.results_dir / "final_results.json"
        
        # Get final RL insights
        final_rl_insights = {}
        if self.rl_enabled and self.rl_scorer:
            final_rl_insights = self.rl_scorer.get_rl_insights()
            
            # Save RL state
            rl_state_file = self.results_dir / "rl_state.json"
            self.rl_scorer.save_rl_state(str(rl_state_file))
        
        results_data = {
            "training_completed": True,
            "total_rounds": self.current_round,
            "training_duration": (datetime.now() - self.training_start_time).total_seconds(),
            "training_history": self.training_history,
            "client_performance": self.client_performance,
            "rl_insights_history": self.rl_insights_history,
            "final_rl_insights": final_rl_insights,
            "client_participation": self.client_participation,
            "suspicious_clients": self.suspicious_clients,
            "replaced_clients": self.replaced_clients,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(final_results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {final_results_file}")
        self.training_completed = True
    
    def get_rl_summary(self) -> Dict[str, Any]:
        """Get summary of RL performance and insights."""
        if not self.rl_enabled or not self.rl_scorer:
            return {'rl_enabled': False}
        
        return {
            'rl_enabled': True,
            'current_insights': self.rl_scorer.get_rl_insights(),
            'insights_history': self.rl_insights_history,
            'training_rounds': self.current_round,
            'policy_evolution': self.policy_evolution
        } 