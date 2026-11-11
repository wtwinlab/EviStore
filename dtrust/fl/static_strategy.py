"""
Static Trust-aware Flower strategy for baseline experiments.

This strategy uses static trust models instead of dynamic trust updates,
providing baseline comparisons for the dynamic trust model in DTRUST.
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
from ..trust.static_scorer import StaticTrustScorer, StaticTrustModel
from ..core.logging import get_logger, log_exception

import time
import json
import os
from datetime import datetime
from pathlib import Path


logger = get_logger(__name__)


class StaticTrustAwareStrategy(FedAvg):
    """
    Federated learning strategy that uses static trust models for baseline experiments.
    
    This strategy provides a baseline comparison against the dynamic TrustAwareStrategy
    by using static trust scores that don't adapt based on ongoing behavior.
    """
    
    def __init__(
        self,
        static_scorer: StaticTrustScorer,
        min_trust_score: float = 0.5,
        trust_weight: float = 0.3,
        results_dir: str = "training_results",
        num_rounds: int = 3,
        experiment_name: str = "static_baseline",
        **kwargs
    ):
        """
        Initialize static trust-aware strategy.
        
        Args:
            static_scorer: Static trust scoring system
            min_trust_score: Minimum trust score for participation
            trust_weight: How much trust affects aggregation weights
            results_dir: Directory to save training results JSON files
            num_rounds: Total number of training rounds
            experiment_name: Name for this experiment (used in results files)
            **kwargs: Additional arguments passed to FedAvg
        """
        # Set up evaluation metrics aggregation if not provided
        if 'evaluate_metrics_aggregation_fn' not in kwargs:
            kwargs['evaluate_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        if 'fit_metrics_aggregation_fn' not in kwargs:
            kwargs['fit_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        super().__init__(**kwargs)
        self.static_scorer = static_scorer
        self.min_trust_score = min_trust_score
        self.trust_weight = trust_weight
        self.num_rounds = num_rounds
        self.experiment_name = experiment_name
        
        # Results saving configuration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.training_start_time = datetime.now()
        
        # Track client performance and trust scores
        self.client_performance: Dict[str, Dict] = {}
        self.client_trust_scores: Dict[str, float] = {}
        self.training_history = []
        self.client_participation = {}
        self.training_completed = False
        
        logger.info(f"StaticTrustAwareStrategy initialized with model: {static_scorer.model_type.value}")
        logger.info(f"Experiment name: {experiment_name}")
        logger.info(f"Training will run for {num_rounds} rounds")
    
    def _weighted_average_metrics(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate metrics using weighted average based on trust scores."""
        if not metrics:
            return {}
        
        aggregated_metrics = {}
        total_weight = 0
        
        for num_examples, metric_dict in metrics:
            # Use number of examples as base weight
            weight = float(num_examples)
            total_weight += weight
            
            for key, value in metric_dict.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0.0
                aggregated_metrics[key] += float(value) * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in aggregated_metrics:
                aggregated_metrics[key] /= total_weight
        
        return aggregated_metrics
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with static trust-based client selection."""
        # Get available clients
        available_clients = list(client_manager.all().values())
        
        if not available_clients:
            logger.warning("No clients available for training")
            return []
        
        # Select clients based on static trust scores
        trusted_clients = self._select_trusted_clients(available_clients)
        
        if not trusted_clients:
            logger.warning("No clients meet minimum trust score requirement")
            return []
        
        # Sample clients for this round
        sample_size = min(
            len(trusted_clients),
            max(1, int(len(available_clients) * self.fraction_fit))
        )
        
        selected_clients = self._sample_clients_by_trust(trusted_clients, sample_size)
        
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
        """Configure the next round of evaluation with static trust-based client selection."""
        # Get available clients
        available_clients = list(client_manager.all().values())
        
        if not available_clients:
            logger.warning("No clients available for evaluation")
            return []
        
        # Select clients based on static trust scores
        trusted_clients = self._select_trusted_clients(available_clients)
        
        if not trusted_clients:
            logger.warning("No clients meet minimum trust score requirement for evaluation")
            return []
        
        # Sample clients for evaluation
        sample_size = min(
            len(trusted_clients),
            max(1, int(len(available_clients) * self.fraction_evaluate))
        )
        
        selected_clients = self._sample_clients_by_trust(trusted_clients, sample_size)
        
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
        """Aggregate fit results using static trust-weighted aggregation."""
        if not results:
            logger.warning("No fit results to aggregate")
            return None, {}
        
        # Update client performance tracking
        for client, fit_res in results:
            self._update_client_performance(client.cid, fit_res, "fit")
        
        # Apply trust weights to aggregation
        weighted_results = self._apply_trust_weights(results, "fit")
        
        # Aggregate parameters using weighted average
        parameters_aggregated = aggregate(weighted_results)
        
        # Convert aggregated tensors back to Parameters
        parameters_final = ndarrays_to_parameters(parameters_aggregated)
        
        # Aggregate metrics
        metrics_aggregated = self._aggregate_trust_metrics(results)
        
        # Add round information
        metrics_aggregated["round"] = server_round
        metrics_aggregated["num_clients"] = len(results)
        metrics_aggregated["trust_model"] = self.static_scorer.model_type.value
        
        # Track training history
        self.training_history.append({
            "round": server_round,
            "num_clients": len(results),
            "metrics": metrics_aggregated,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save intermediate results
        self._save_intermediate_results(server_round)
        
        logger.info(f"Round {server_round}: Aggregated fit results from {len(results)} clients")
        
        return parameters_final, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using static trust-weighted aggregation."""
        if not results:
            logger.warning("No evaluation results to aggregate")
            return None, {}
        
        # Update client performance tracking
        for client, eval_res in results:
            self._update_client_performance(client.cid, eval_res, "evaluate")
        
        # Apply trust weights to aggregation
        weighted_results = self._apply_trust_weights(results, "evaluate")
        
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(weighted_results)
        
        # Aggregate metrics
        metrics_aggregated = self._aggregate_trust_metrics(results)
        
        # Add round information
        metrics_aggregated["round"] = server_round
        metrics_aggregated["num_clients"] = len(results)
        metrics_aggregated["trust_model"] = self.static_scorer.model_type.value
        
        logger.info(f"Round {server_round}: Aggregated evaluation results from {len(results)} clients")
        logger.info(f"Round {server_round}: Loss = {loss_aggregated:.4f}")
        
        return loss_aggregated, metrics_aggregated
    
    def _select_trusted_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Select clients that meet the minimum trust score requirement."""
        trusted_clients = []
        
        for client in available_clients:
            client_id = client.cid
            trust_score = self.static_scorer.get_static_trust_score(client_id)
            self.client_trust_scores[client_id] = trust_score
            
            if trust_score >= self.min_trust_score:
                trusted_clients.append(client)
            else:
                logger.debug(f"Client {client_id} excluded: trust_score={trust_score:.3f} < {self.min_trust_score}")
        
        return trusted_clients
    
    def _sample_clients_by_trust(
        self, 
        clients: List[ClientProxy], 
        num_clients: int
    ) -> List[ClientProxy]:
        """Sample clients based on their static trust scores."""
        if len(clients) <= num_clients:
            return clients
        
        # Sort clients by trust score (descending)
        client_trust_pairs = []
        for client in clients:
            trust_score = self.client_trust_scores.get(client.cid, 0.5)
            client_trust_pairs.append((client, trust_score))
        
        # Sort by trust score (highest first)
        client_trust_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        selected_clients = [client for client, _ in client_trust_pairs[:num_clients]]
        
        logger.debug(f"Selected clients by trust: {[(c.cid, self.client_trust_scores.get(c.cid, 0.5)) for c in selected_clients]}")
        
        return selected_clients
    
    def _apply_trust_weights(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]], 
        result_type: str
    ) -> List[Tuple[Union[List, float], int]]:
        """Apply trust weights to client results for aggregation."""
        weighted_results = []
        
        for client, result in results:
            client_id = client.cid
            trust_score = self.client_trust_scores.get(client_id, 0.5)
            
            # Calculate trust-weighted number of examples
            if result_type == "fit":
                num_examples = result.num_examples
                # Apply trust weight to number of examples
                weighted_examples = int(num_examples * (1 + self.trust_weight * (trust_score - 0.5)))
                # Convert Parameters to tensors for aggregation
                parameters_tensors = parameters_to_ndarrays(result.parameters)
                weighted_results.append((parameters_tensors, weighted_examples))
                
            elif result_type == "evaluate":
                num_examples = result.num_examples
                loss = result.loss
                # Apply trust weight to loss (higher trust = lower effective loss)
                weighted_loss = loss * (1 - self.trust_weight * (trust_score - 0.5))
                weighted_results.append((weighted_loss, num_examples))
        
        return weighted_results
    
    def _aggregate_trust_metrics(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]]
    ) -> Dict[str, Scalar]:
        """Aggregate trust-related metrics from client results."""
        metrics = {}
        
        if not results:
            return metrics
        
        # Calculate average trust score
        trust_scores = []
        for client, _ in results:
            trust_score = self.client_trust_scores.get(client.cid, 0.5)
            trust_scores.append(trust_score)
        
        metrics["avg_trust_score"] = sum(trust_scores) / len(trust_scores)
        metrics["min_trust_score"] = min(trust_scores)
        metrics["max_trust_score"] = max(trust_scores)
        metrics["trust_score_std"] = self._calculate_std(trust_scores)
        
        # Aggregate client metrics
        all_metrics = {}
        for client, result in results:
            if hasattr(result, 'metrics') and result.metrics:
                for key, value in result.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(float(value))
        
        # Calculate averages for each metric
        for key, values in all_metrics.items():
            metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return metrics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _update_client_performance(
        self, 
        client_id: str, 
        result: Union[FitRes, EvaluateRes], 
        result_type: str
    ):
        """Update client performance tracking."""
        if client_id not in self.client_performance:
            self.client_performance[client_id] = {
                "fit_rounds": 0,
                "evaluate_rounds": 0,
                "total_examples": 0,
                "metrics_history": []
            }
        
        self.client_performance[client_id][f"{result_type}_rounds"] += 1
        self.client_performance[client_id]["total_examples"] += result.num_examples
        
        # Store metrics
        if hasattr(result, 'metrics') and result.metrics:
            metrics_entry = {
                "round": len(self.training_history),
                "type": result_type,
                "metrics": result.metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.client_performance[client_id]["metrics_history"].append(metrics_entry)
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for fit round."""
        return {
            "epoch_global": server_round,
            "epoch": 1,
            "round": server_round,
            "trust_model": self.static_scorer.model_type.value
        }
    
    def _get_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for evaluate round."""
        return {
            "epoch_global": server_round,
            "round": server_round,
            "trust_model": self.static_scorer.model_type.value
        }
    
    def _save_intermediate_results(self, server_round: int):
        """Save intermediate training results."""
        if not self.training_history:
            return
        
        # Save round-specific results
        round_results = {
            "experiment_name": self.experiment_name,
            "trust_model": self.static_scorer.model_type.value,
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "training_history": self.training_history,
            "client_trust_scores": self.client_trust_scores,
            "client_participation": self.client_participation
        }
        
        filename = f"{self.experiment_name}_round_{server_round}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(round_results, f, indent=2, default=str)
        
        logger.debug(f"Saved intermediate results to {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training experiment."""
        if not self.training_history:
            return {"error": "No training history available"}
        
        # Calculate summary statistics
        total_rounds = len(self.training_history)
        total_clients = len(self.client_trust_scores)
        avg_trust_score = sum(self.client_trust_scores.values()) / total_clients if total_clients > 0 else 0
        
        # Get final metrics from last round
        final_metrics = self.training_history[-1]["metrics"] if self.training_history else {}
        
        summary = {
            "experiment_name": self.experiment_name,
            "trust_model": self.static_scorer.model_type.value,
            "model_description": self.static_scorer.get_model_description(),
            "training_start_time": self.training_start_time.isoformat(),
            "training_end_time": datetime.now().isoformat(),
            "total_rounds": total_rounds,
            "total_clients": total_clients,
            "average_trust_score": avg_trust_score,
            "min_trust_score": min(self.client_trust_scores.values()) if self.client_trust_scores else 0,
            "max_trust_score": max(self.client_trust_scores.values()) if self.client_trust_scores else 0,
            "final_metrics": final_metrics,
            "client_participation_summary": {
                client_id: len(rounds) for client_id, rounds in self.client_participation.items()
            }
        }
        
        return summary
    
    def save_final_results(self):
        """Save final training results."""
        summary = self.get_training_summary()
        
        filename = f"{self.experiment_name}_final_results.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved final results to {filepath}")
        return filepath 