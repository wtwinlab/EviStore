"""Trust-conditioned FedAvg strategy: filter, weight, and quarantine by trust state."""

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
from ..trust.manager import TrustManager
from ..core.logging import get_logger, log_exception

import time
import json
import os
from datetime import datetime
from pathlib import Path


logger = get_logger(__name__)


class TrustAwareStrategy(FedAvg):
    """
    Federated learning strategy that incorporates TEE trust management.
    
    Extends Flower's FedAvg strategy to:
    - Select clients based on trust scores
    - Automatically replace suspicious clients
    - Weight aggregation by trust levels
    - Handle dynamic client management
    - Automatically save training results to JSON files
    
    This strategy leverages Flower's built-in aggregation utilities while adding
    trust-aware client selection and weighting.
    """
    
    def __init__(
        self,
        trust_manager: TrustManager,
        min_trust_score: float = 0.5,
        trust_weight: float = 0.3,
        auto_replace_suspicious: bool = True,
        max_suspicious_rounds: int = 2,
        results_dir: str = "training_results",
        num_rounds: int = 3,
        **kwargs
    ):
        """
        Initialize trust-aware strategy.
        
        Args:
            trust_manager: Trust management system
            min_trust_score: Minimum trust score for participation
            trust_weight: How much trust affects aggregation weights
            auto_replace_suspicious: Whether to auto-replace suspicious clients
            max_suspicious_rounds: Max rounds before replacing suspicious clients
            results_dir: Directory to save training results JSON files
            num_rounds: Total number of training rounds
            **kwargs: Additional arguments passed to FedAvg
        """
        # Set up evaluation metrics aggregation if not provided
        if 'evaluate_metrics_aggregation_fn' not in kwargs:
            kwargs['evaluate_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        if 'fit_metrics_aggregation_fn' not in kwargs:
            kwargs['fit_metrics_aggregation_fn'] = self._weighted_average_metrics
        
        super().__init__(**kwargs)
        self.trust_manager = trust_manager
        self.min_trust_score = min_trust_score
        self.trust_weight = trust_weight
        self.auto_replace_suspicious = auto_replace_suspicious
        self.max_suspicious_rounds = max_suspicious_rounds
        self.num_rounds = num_rounds
        
        # Results saving configuration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.training_start_time = datetime.now()
        
        # Track client performance
        self.client_performance: Dict[str, Dict] = {}
        self.suspicious_clients: Dict[str, int] = {}  # client_id -> suspicious_count
        self.replaced_clients: List[str] = []
        
        # Training progress tracking
        self.current_round = 0
        self.training_history = []
        self.client_participation = {}
        self.training_completed = False
        
        logger.info(f"TrustAwareStrategy initialized with min_trust_score={min_trust_score}")
        logger.info(f"Training will run for {num_rounds} rounds")
        logger.info(f"Training results will be saved to: {self.results_dir}")
    
    def _weighted_average_metrics(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate metrics using weighted averaging."""
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
                            # Skip non-numeric metrics (like client_id, data_partition)
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
        """Configure the next round of training with trust-aware client selection."""
        self.current_round = server_round
        
        logger.info(f" Starting training round {server_round}")
        logger.info(f"Available clients: {client_manager.num_available()}")
        
        # For the first round, wait a bit longer for clients to connect
        if server_round == 1:
            logger.info("First round: waiting for clients to connect...")
            import time
            time.sleep(15)  # Wait 15 seconds for clients to connect
            logger.info(f"Available clients after wait: {client_manager.num_available()}")
        
        # Get all available clients
        available_clients = list(client_manager.all().values())
        logger.info(f"Available client IDs: {[client.cid for client in available_clients]}")
        
        # If we don't have enough clients, wait a bit more (but not forever)
        if len(available_clients) < self.min_fit_clients and server_round == 1:
            logger.info(f"Waiting for more clients (have {len(available_clients)}, need {self.min_fit_clients})...")
            import time
            for wait_round in range(6):  # Wait up to 30 more seconds
                time.sleep(5)
                available_clients = list(client_manager.all().values())
                logger.info(f"Available clients after {(wait_round + 1) * 5}s: {len(available_clients)}")
                if len(available_clients) >= self.min_fit_clients:
                    break
        
        # Filter clients based on trust scores
        trusted_clients = self._select_trusted_clients(available_clients)
        
        if not trusted_clients:
            logger.warning(" No trusted clients available, using all available clients")
            trusted_clients = available_clients
        
        # Select clients for this round using Flower's built-in sampling
        num_clients = min(
            max(int(len(trusted_clients) * self.fraction_fit), self.min_fit_clients),
            len(trusted_clients)
        )
        
        # Use trust scores to influence selection (higher trust = higher probability)
        selected_clients = self._sample_clients_by_trust(trusted_clients, num_clients)
        
        logger.info(f"Selected {len(selected_clients)} clients for round {server_round}")
        
        # Track client participation
        for client in selected_clients:
            if client.cid not in self.client_participation:
                self.client_participation[client.cid] = []
            self.client_participation[client.cid].append(server_round)
        
        # Create fit instructions using Flower's pattern
        config = self._get_fit_config(server_round)
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in selected_clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation with trust-aware client selection."""
        
        # Get available clients
        available_clients = list(client_manager.all().values())
        
        # Filter by trust scores
        trusted_clients = self._select_trusted_clients(available_clients)
        
        if not trusted_clients:
            trusted_clients = available_clients
        
        # Select clients for evaluation
        num_clients = min(
            len(trusted_clients),
            max(int(len(trusted_clients) * self.fraction_evaluate), self.min_evaluate_clients)
        )
        
        selected_clients = self._sample_clients_by_trust(trusted_clients, num_clients)
        
        # Create evaluation instructions
        config = self._get_evaluate_config(server_round)
        evaluate_ins = EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in selected_clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with trust-aware weighting."""
        
        logger.info(f"Aggregating results for round {server_round}")
        logger.info(f"Received {len(results)} results, {len(failures)} failures")
        
        if failures:
            logger.warning(f" Training failures in round {server_round}:")
            for i, failure in enumerate(failures):
                if isinstance(failure, tuple):
                    client, fit_res = failure
                    logger.warning(f"   {i+1}. Client {client.cid}: {fit_res}")
                else:
                    logger.warning(f"   {i+1}. Exception: {failure}")
        
        # Enhanced logging: Log detailed client results and individual contributions
        logger.info(f"ROUND {server_round} - INDIVIDUAL CLIENT CONTRIBUTIONS:")
        client_contributions = []
        total_examples = 0
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            num_examples = fit_res.num_examples
            total_examples += num_examples
            
            # Get trust score for this client
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
            except Exception:
                trust_score = 0.5
            
            # Log individual client contribution
            logger.info(f"   Client {client.cid}:")
            logger.info(f"      Examples: {num_examples}")
            logger.info(f"       Trust Score: {trust_score:.4f}")
            
            if metrics:
                logger.info(f"      Client Metrics: {metrics}")
                # Log specific metrics if available
                if 'accuracy' in metrics:
                    accuracy_pct = metrics['accuracy'] * 100
                    logger.info(f"      Local Accuracy: {accuracy_pct:.2f}%")
                if 'loss' in metrics:
                    logger.info(f"      Local Loss: {metrics['loss']:.4f}")
            
            # Store contribution for analysis
            contribution = {
                "client_id": client.cid,
                "num_examples": num_examples,
                "trust_score": trust_score,
                "metrics": metrics or {},
                "weight_percentage": 0  # Will calculate after we know total
            }
            client_contributions.append(contribution)
            
            # Update client performance tracking
            self._update_client_performance(client.cid, fit_res, "fit")
        
        # Calculate weight percentages
        for contrib in client_contributions:
            contrib["weight_percentage"] = (contrib["num_examples"] / total_examples * 100) if total_examples > 0 else 0
        
        if not results:
            logger.warning("No successful fit results to aggregate")
            return None, {}
        
        # Apply trust-aware weighting to results
        weighted_results = self._apply_trust_weights(results, "fit")
        
        # Log trust weighting effects
        logger.info(f"ROUND {server_round} - TRUST-AWARE WEIGHTING ANALYSIS:")
        for i, ((client, fit_res), contrib) in enumerate(zip(results, client_contributions)):
            original_weight = fit_res.num_examples
            trust_score = contrib["trust_score"]
            trust_adjustment = 1.0 + (trust_score - 0.5) * self.trust_weight
            adjusted_weight = int(original_weight * max(0.1, trust_adjustment))
            
            weight_change_pct = ((adjusted_weight - original_weight) / original_weight * 100) if original_weight > 0 else 0
            
            logger.info(f"   {client.cid}:")
            logger.info(f"      Original Weight: {original_weight}")
            logger.info(f"       Adjusted Weight: {adjusted_weight}")
            logger.info(f"      Weight Change: {weight_change_pct:+.1f}%")
            logger.info(f"       Trust Impact: {trust_adjustment:.3f}x")
        
        # Use Flower's built-in aggregation with trust-weighted results
        aggregated_ndarrays = aggregate(weighted_results)
        
        # Convert aggregated ndarrays back to Parameters object
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate metrics using the configured function
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results if fit_res.metrics]
            if metrics_list:
                aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
        
        # Add trust-specific metrics
        trust_metrics = self._aggregate_trust_metrics(results)
        aggregated_metrics.update(trust_metrics)
        
        # Enhanced logging: Log comprehensive aggregated metrics analysis
        logger.info(f"ROUND {server_round} - AGGREGATED MODEL PERFORMANCE:")
        logger.info(f"   Total Training Examples: {aggregated_metrics.get('total_examples', 0)}")
        logger.info(f"   Participating Clients: {aggregated_metrics.get('num_clients', 0)}")
        logger.info(f"    Average Trust Score: {aggregated_metrics.get('avg_trust_score', 0):.4f}")
        logger.info(f"    Trust Score Range: {aggregated_metrics.get('min_trust_score', 0):.4f} - {aggregated_metrics.get('max_trust_score', 0):.4f}")
        
        if 'accuracy' in aggregated_metrics:
            accuracy_pct = aggregated_metrics['accuracy'] * 100
            logger.info(f"   Aggregated Accuracy: {accuracy_pct:.2f}%")
        if 'loss' in aggregated_metrics:
            logger.info(f"   Aggregated Loss: {aggregated_metrics['loss']:.4f}")
        
        # Log additional performance metrics if available
        for metric_name, metric_value in aggregated_metrics.items():
            if metric_name not in ['total_examples', 'num_clients', 'avg_trust_score', 'min_trust_score', 'max_trust_score', 'accuracy', 'loss']:
                logger.info(f"   {metric_name}: {metric_value}")
        
        # Record enhanced training history with detailed analysis
        round_info = {
            "round": server_round,
            "aggregated_metrics": aggregated_metrics,
            "num_clients": len(results),
            "num_failures": len(failures),
            "timestamp": datetime.now().isoformat(),
            "client_contributions": client_contributions,  # New: Individual client analysis
            "trust_analysis": {  # New: Trust-based analysis
                "trust_weight_factor": self.trust_weight,
                "min_trust_threshold": self.min_trust_score,
                "trust_score_distribution": {
                    "mean": aggregated_metrics.get('avg_trust_score', 0),
                    "min": aggregated_metrics.get('min_trust_score', 0),
                    "max": aggregated_metrics.get('max_trust_score', 0)
                }
            }
        }
        self.training_history.append(round_info)
        
        # Enhanced logging: Performance trend analysis (if we have previous rounds)
        if len(self.training_history) > 1:
            logger.info(f"ROUND {server_round} - PERFORMANCE TREND ANALYSIS:")
            prev_round = self.training_history[-2]
            
            # Compare accuracy trend
            if 'accuracy' in aggregated_metrics and 'accuracy' in prev_round.get('aggregated_metrics', {}):
                prev_accuracy = prev_round['aggregated_metrics']['accuracy'] * 100
                curr_accuracy = aggregated_metrics['accuracy'] * 100
                accuracy_change = curr_accuracy - prev_accuracy
                logger.info(f"   Accuracy Trend: {prev_accuracy:.2f}% → {curr_accuracy:.2f}% ({accuracy_change:+.2f}%)")
            
            # Compare loss trend
            if 'loss' in aggregated_metrics and 'loss' in prev_round.get('aggregated_metrics', {}):
                prev_loss = prev_round['aggregated_metrics']['loss']
                curr_loss = aggregated_metrics['loss']
                loss_change = curr_loss - prev_loss
                loss_change_pct = (loss_change / prev_loss * 100) if prev_loss != 0 else 0
                logger.info(f"   Loss Trend: {prev_loss:.4f} → {curr_loss:.4f} ({loss_change:+.4f}, {loss_change_pct:+.1f}%)")
            
            # Compare trust score trend
            prev_trust = prev_round.get('aggregated_metrics', {}).get('avg_trust_score', 0)
            curr_trust = aggregated_metrics.get('avg_trust_score', 0)
            trust_change = curr_trust - prev_trust
            logger.info(f"    Trust Trend: {prev_trust:.4f} → {curr_trust:.4f} ({trust_change:+.4f})")
            
            # Compare client participation
            prev_clients = prev_round.get('num_clients', 0)
            curr_clients = len(results)
            client_change = curr_clients - prev_clients
            logger.info(f"   Client Participation: {prev_clients} → {curr_clients} ({client_change:+d})")
        
        # Check if this is the final round and save results
        if server_round >= self.num_rounds:
            logger.info(f"Training completed after {server_round} rounds")
            self.training_completed = True
            self._save_training_results()
        
        logger.info(f"Round {server_round} aggregation completed")
        logger.info(f"Final Aggregated Metrics: {aggregated_metrics}")
        
        self._save_intermediate_results(server_round)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with trust-aware weighting."""
        
        if not results:
            return None, {}
        
        # Enhanced logging: Log detailed evaluation results
        logger.info(f"ROUND {server_round} - INDIVIDUAL CLIENT EVALUATION:")
        client_evaluations = []
        total_examples = 0
        
        for client, eval_res in results:
            total_examples += eval_res.num_examples
            
            # Get trust score for this client
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
            except Exception:
                trust_score = 0.5
            
            # Log individual client evaluation
            logger.info(f"   Client {client.cid}:")
            logger.info(f"      Evaluation Examples: {eval_res.num_examples}")
            logger.info(f"      Local Loss: {eval_res.loss:.4f}")
            logger.info(f"       Trust Score: {trust_score:.4f}")
            
            if eval_res.metrics:
                logger.info(f"      Evaluation Metrics: {eval_res.metrics}")
                # Log specific metrics if available
                if 'accuracy' in eval_res.metrics:
                    accuracy_pct = eval_res.metrics['accuracy'] * 100
                    logger.info(f"      Local Accuracy: {accuracy_pct:.2f}%")
            
            # Store evaluation for analysis
            evaluation = {
                "client_id": client.cid,
                "num_examples": eval_res.num_examples,
                "loss": eval_res.loss,
                "trust_score": trust_score,
                "metrics": eval_res.metrics or {},
                "weight_percentage": 0  # Will calculate after we know total
            }
            client_evaluations.append(evaluation)
            
            # Update client performance tracking
            self._update_client_performance(client.cid, eval_res, "evaluate")
        
        # Calculate weight percentages
        for eval_data in client_evaluations:
            eval_data["weight_percentage"] = (eval_data["num_examples"] / total_examples * 100) if total_examples > 0 else 0
        
        # Apply trust-aware weighting
        weighted_results = self._apply_trust_weights(results, "evaluate")
        
        # Log trust weighting effects for evaluation
        logger.info(f"ROUND {server_round} - EVALUATION TRUST-AWARE WEIGHTING:")
        for i, ((client, eval_res), eval_data) in enumerate(zip(results, client_evaluations)):
            original_weight = eval_res.num_examples
            trust_score = eval_data["trust_score"]
            trust_adjustment = 1.0 + (trust_score - 0.5) * self.trust_weight
            adjusted_weight = int(original_weight * max(0.1, trust_adjustment))
            
            weight_change_pct = ((adjusted_weight - original_weight) / original_weight * 100) if original_weight > 0 else 0
            
            logger.info(f"   {client.cid}:")
            logger.info(f"      Original Weight: {original_weight}")
            logger.info(f"       Adjusted Weight: {adjusted_weight}")
            logger.info(f"      Weight Change: {weight_change_pct:+.1f}%")
            logger.info(f"       Trust Impact: {trust_adjustment:.3f}x")
        
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
        
        # Enhanced logging: Log comprehensive evaluation analysis
        logger.info(f"ROUND {server_round} - AGGREGATED EVALUATION PERFORMANCE:")
        logger.info(f"   Aggregated Loss: {aggregated_loss:.4f}")
        logger.info(f"   Total Evaluation Examples: {aggregated_metrics.get('total_examples', 0)}")
        logger.info(f"   Participating Clients: {aggregated_metrics.get('num_clients', 0)}")
        logger.info(f"    Average Trust Score: {aggregated_metrics.get('avg_trust_score', 0):.4f}")
        logger.info(f"    Trust Score Range: {aggregated_metrics.get('min_trust_score', 0):.4f} - {aggregated_metrics.get('max_trust_score', 0):.4f}")
        
        if 'accuracy' in aggregated_metrics:
            accuracy_pct = aggregated_metrics['accuracy'] * 100
            logger.info(f"   Aggregated Accuracy: {accuracy_pct:.2f}%")
        
        # Log additional evaluation metrics if available
        for metric_name, metric_value in aggregated_metrics.items():
            if metric_name not in ['total_examples', 'num_clients', 'avg_trust_score', 'min_trust_score', 'max_trust_score', 'accuracy']:
                logger.info(f"   {metric_name}: {metric_value}")
        
        # Enhanced logging: Evaluation performance analysis
        logger.info(f"ROUND {server_round} - EVALUATION PERFORMANCE BREAKDOWN:")
        
        # Calculate loss distribution statistics
        losses = [eval_data["loss"] for eval_data in client_evaluations]
        if losses:
            min_loss = min(losses)
            max_loss = max(losses)
            avg_loss = sum(losses) / len(losses)
            logger.info(f"   Loss Distribution: min={min_loss:.4f}, avg={avg_loss:.4f}, max={max_loss:.4f}")
        
        # Show client ranking by performance
        sorted_evaluations = sorted(client_evaluations, key=lambda x: x["loss"])
        logger.info(f"   Client Performance Ranking (by loss, lower is better):")
        for i, eval_data in enumerate(sorted_evaluations[:5]):  # Show top 5
            rank = i + 1
            client_id = eval_data["client_id"]
            loss = eval_data["loss"]
            trust = eval_data["trust_score"]
            examples = eval_data["num_examples"]
            logger.info(f"      {rank}. {client_id}: loss={loss:.4f}, trust={trust:.3f}, examples={examples}")
        
        logger.info(f"Evaluation aggregated: loss={aggregated_loss:.4f}, metrics={aggregated_metrics}")
        
        return aggregated_loss, aggregated_metrics
    
    def _select_trusted_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Select clients based on trust scores."""
        trusted_clients = []
        
        for client in available_clients:
            try:
                # Get trust score from trust manager
                node_state = self.trust_manager.get_node_state(client.cid)
                if node_state and node_state.trust_score >= self.min_trust_score:
                    trusted_clients.append(client)
                    logger.debug(f"Client {client.cid} selected (trust: {node_state.trust_score:.3f})")
                else:
                    trust_score = node_state.trust_score if node_state else 0.0
                    logger.debug(f"Client {client.cid} rejected (trust: {trust_score:.3f})")
            except Exception as e:
                logger.warning(f"Error getting trust score for {client.cid}: {e}")
                # In case of error, include client with default trust
                trusted_clients.append(client)
        
        return trusted_clients
    
    def _sample_clients_by_trust(
        self, 
        clients: List[ClientProxy], 
        num_clients: int
    ) -> List[ClientProxy]:
        """Sample clients with probability proportional to trust scores."""
        if len(clients) <= num_clients:
            return clients
        
        # Get trust scores for weighting
        client_weights = []
        for client in clients:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                # Convert trust score to sampling weight (higher trust = higher weight)
                weight = max(0.1, trust_score)  # Minimum weight to ensure all clients have some chance
                client_weights.append(weight)
            except Exception:
                client_weights.append(0.5)  # Default weight
        
        # Sample clients based on weights
        import random
        import numpy as np
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in client_weights]
        else:
            probabilities = [1.0 / len(clients)] * len(clients)
        
        # Sample without replacement
        selected_indices = np.random.choice(
            len(clients), 
            size=min(num_clients, len(clients)), 
            replace=False, 
            p=probabilities
        )
        
        selected_clients = [clients[i] for i in selected_indices]
        
        # Log trust scores of selected clients
        for client in selected_clients:
            try:
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                logger.info(f"Selected {client.cid} (trust: {trust_score:.3f})")
            except Exception:
                logger.info(f"Selected {client.cid} (trust: unknown)")
        
        return selected_clients
    
    def _apply_trust_weights(
        self, 
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]], 
        result_type: str
    ) -> List[Tuple[Union[List, float], int]]:
        """Apply trust-aware weighting to aggregation."""
        weighted_results = []
        
        for client, result in results:
            try:
                # Get trust score
                node_state = self.trust_manager.get_node_state(client.cid)
                trust_score = node_state.trust_score if node_state else 0.5
                
                # Calculate trust-adjusted weight
                base_weight = result.num_examples
                trust_adjustment = 1.0 + (trust_score - 0.5) * self.trust_weight
                adjusted_weight = int(base_weight * max(0.1, trust_adjustment))  # Minimum 10% weight
                
                if result_type == "fit":
                    # For fit results, extract parameters and convert to ndarrays
                    parameters = result.parameters
                    # Convert Parameters object to list of ndarrays for aggregation
                    parameters_ndarrays = parameters_to_ndarrays(parameters)
                    weighted_results.append((parameters_ndarrays, adjusted_weight))
                else:
                    # For evaluate results, extract loss
                    loss = result.loss
                    weighted_results.append((loss, adjusted_weight))
                
                logger.debug(f"Client {client.cid}: trust={trust_score:.3f}, "
                           f"base_weight={base_weight}, adjusted_weight={adjusted_weight}")
                
            except Exception as e:
                logger.warning(f"Error applying trust weight for {client.cid}: {e}")
                # Fall back to original weight
                if result_type == "fit":
                    # Convert parameters to ndarrays for fallback too
                    parameters_ndarrays = parameters_to_ndarrays(result.parameters)
                    weighted_results.append((parameters_ndarrays, result.num_examples))
                else:
                    weighted_results.append((result.loss, result.num_examples))
        
        return weighted_results
    
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
    
    def _update_client_performance(
        self, 
        client_id: str, 
        result: Union[FitRes, EvaluateRes], 
        result_type: str
    ):
        """Update client performance tracking."""
        if client_id not in self.client_performance:
            self.client_performance[client_id] = {
                "fit_count": 0,
                "evaluate_count": 0,
                "total_examples": 0,
                "avg_loss": 0.0,
                "last_seen": self.current_round
            }
        
        perf = self.client_performance[client_id]
        perf["last_seen"] = self.current_round
        perf["total_examples"] += result.num_examples
        
        if result_type == "fit":
            perf["fit_count"] += 1
        else:
            perf["evaluate_count"] += 1
            # Update average loss for evaluation results
            if hasattr(result, 'loss'):
                current_avg = perf["avg_loss"]
                count = perf["evaluate_count"]
                perf["avg_loss"] = (current_avg * (count - 1) + result.loss) / count
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for fit round."""
        return {
            "round": server_round,
            "epochs": 1,  # Default epochs per round
            "learning_rate": 0.01,  # Default learning rate
        }
    
    def _get_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for evaluate round."""
        return {
            "round": server_round,
        }
    
    def _save_training_results(self):
        """Save comprehensive training results to JSON file."""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dtrust_training_results_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Get comprehensive training summary
            training_summary = self.get_training_summary()
            
            # Add additional metadata
            results_data = {
                "metadata": {
                    "training_start_time": self.training_start_time.isoformat(),
                    "training_end_time": datetime.now().isoformat(),
                    "training_duration_seconds": (datetime.now() - self.training_start_time).total_seconds(),
                    "dtrust_version": "1.0.0",  # You can make this dynamic
                    "results_file": str(filepath),
                    "generated_at": datetime.now().isoformat()
                },
                "training_config": {
                    "num_rounds": self.num_rounds,
                    "min_fit_clients": self.min_fit_clients,
                    "min_evaluate_clients": self.min_evaluate_clients,
                    "fraction_fit": self.fraction_fit,
                    "fraction_evaluate": self.fraction_evaluate,
                    "min_trust_score": self.min_trust_score,
                    "trust_weight": self.trust_weight,
                    "auto_replace_suspicious": self.auto_replace_suspicious,
                    "max_suspicious_rounds": self.max_suspicious_rounds
                },
                "results": training_summary
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Training results saved to: {filepath}")
            
            # Also save a "latest" symlink/copy for easy access
            latest_filepath = self.results_dir / "dtrust_training_results_latest.json"
            with open(latest_filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Latest results also saved to: {latest_filepath}")
            
            # Log summary to console
            summary = training_summary.get("summary", {})
            logger.info(f"Training Summary:")
            logger.info(f"   Total rounds: {summary.get('total_rounds', 0)}")
            logger.info(f"   Total clients: {summary.get('total_clients', 0)}")
            logger.info(f"   Final examples: {summary.get('total_examples_final', 0)}")
            
            last_metrics = summary.get("last_round_metrics", {})
            if "accuracy" in last_metrics:
                accuracy_pct = last_metrics["accuracy"] * 100
                logger.info(f"   Final accuracy: {accuracy_pct:.2f}%")
            if "loss" in last_metrics:
                logger.info(f"   Final loss: {last_metrics['loss']:.4f}")
            
        except Exception as e:
            log_exception(e, "Error saving training results")
            logger.error(f"Failed to save training results: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_history:
            return {"status": "no_training", "message": "No training rounds completed"}
        
        last_round = self.training_history[-1]
        
        # Calculate participation statistics
        participation_stats = {}
        for client_id, rounds in self.client_participation.items():
            participation_stats[client_id] = {
                "rounds_participated": len(rounds),
                "participation_rate": len(rounds) / self.current_round if self.current_round > 0 else 0.0,
                "rounds": rounds
            }
        
        return {
            "status": "completed" if self.training_completed else "active",
            "training_completed": self.training_completed,
            "current_round": self.current_round,
            "total_rounds": self.num_rounds,
            "summary": {
                "total_rounds": len(self.training_history),
                "total_clients": len(self.client_participation),
                "strategy_config": {
                    "min_trust_score": self.min_trust_score,
                    "trust_weight": self.trust_weight,
                    "fraction_fit": self.fraction_fit,
                    "fraction_evaluate": self.fraction_evaluate
                },
                "last_round_metrics": last_round.get("aggregated_metrics", {}),
                "total_examples_final": last_round.get("aggregated_metrics", {}).get("total_examples", 0)
            },
            "round_history": self.training_history,
            "client_participation": participation_stats
        }

    def _save_intermediate_results(self, server_round: int):
        """Save intermediate training results after each round for analysis."""
        try:
            # Create intermediate results directory
            intermediate_dir = self.results_dir / "intermediate"
            intermediate_dir.mkdir(exist_ok=True)
            
            # Generate filename for this round
            filename = f"dtrust_round_{server_round:03d}_results.json"
            filepath = intermediate_dir / filename
            
            # Get current training state
            current_summary = self.get_training_summary()
            
            # Add round-specific metadata
            round_data = {
                "metadata": {
                    "round": server_round,
                    "timestamp": datetime.now().isoformat(),
                    "training_start_time": self.training_start_time.isoformat(),
                    "elapsed_time_seconds": (datetime.now() - self.training_start_time).total_seconds(),
                    "is_final_round": server_round >= self.num_rounds,
                    "dtrust_version": "1.0.0"
                },
                "training_config": {
                    "num_rounds": self.num_rounds,
                    "current_round": server_round,
                    "min_fit_clients": self.min_fit_clients,
                    "min_evaluate_clients": self.min_evaluate_clients,
                    "fraction_fit": self.fraction_fit,
                    "fraction_evaluate": self.fraction_evaluate,
                    "min_trust_score": self.min_trust_score,
                    "trust_weight": self.trust_weight
                },
                "round_results": current_summary
            }
            
            # Save intermediate results
            with open(filepath, 'w') as f:
                json.dump(round_data, f, indent=2, default=str)
            
            logger.info(f"Round {server_round} results saved to: {filepath}")
            
            # Also update a "current" file for easy monitoring
            current_filepath = intermediate_dir / "dtrust_current_round_results.json"
            with open(current_filepath, 'w') as f:
                json.dump(round_data, f, indent=2, default=str)
            
            # Log key metrics for this round
            if self.training_history:
                last_round_info = self.training_history[-1]
                metrics = last_round_info.get("aggregated_metrics", {})
                
                logger.info(f"Round {server_round} Key Metrics:")
                if "accuracy" in metrics:
                    accuracy_pct = metrics["accuracy"] * 100
                    logger.info(f"   Accuracy: {accuracy_pct:.2f}%")
                if "loss" in metrics:
                    logger.info(f"   Loss: {metrics['loss']:.4f}")
                logger.info(f"   Clients: {metrics.get('num_clients', 0)}")
                logger.info(f"    Avg Trust: {metrics.get('avg_trust_score', 0):.3f}")
                
        except Exception as e:
            log_exception(e, f"Error saving intermediate results for round {server_round}")
            logger.error(f"Failed to save round {server_round} results: {e}") 