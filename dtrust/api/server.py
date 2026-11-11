"""
DTRUST API Server - FastAPI-based REST API for attestation and trust management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.config import DTRUSTConfig
from ..core.models import (
    AttestationResult, NodeState, TrustMetrics, 
    NodeIdentity, FederationInfo
)
from ..core.exceptions import DTRUSTError, AttestationError, TrustError, NodeError
from ..core.logging import setup_logging, get_logger, log_exception, async_catch_exceptions
from ..trust.manager import TrustManager
from ..tee.protocol import AttestationProtocol
from ..tee.identity import NodeIdentityManager
from ..tee.sgx_quote import create_quote_generator
from ..tee.attestation import create_attestation_service


logger = get_logger(__name__)


# Request/Response Models
class RegisterNodeRequest(BaseModel):
    enclave_id: str
    node_metadata: Optional[Dict[str, Any]] = None


class AttestationRequest(BaseModel):
    node_id: str
    challenge: Optional[str] = None


class TrainingParticipationRequest(BaseModel):
    node_id: str
    num_examples: int
    metrics: Dict[str, float]


class SuspiciousActivityRequest(BaseModel):
    node_id: str
    reason: str


class StartTrainingRequest(BaseModel):
    rounds: int
    min_clients: int
    fraction_fit: float = 0.8
    fraction_evaluate: float = 0.8


class DTRUSTServer:
    """Main DTRUST server managing all components."""
    
    def __init__(self, config: DTRUSTConfig):
        self.config = config
        
        # Initialize components
        self.identity_manager = NodeIdentityManager(config.sgx)
        self.trust_manager = TrustManager(config.trust)
        self.attestation_protocol = AttestationProtocol(
            config.sgx, 
            self.identity_manager
        )
        
        # FL server components
        self.fl_server_task = None
        self.fl_server_running = False
        self.fl_strategy = None  # Store reference to FL strategy
        
        # Register trust event callbacks
        self.trust_manager.register_callback(self._on_trust_event)
        
        self._running = False
    
    @async_catch_exceptions(reraise=True)
    async def start(self):
        """Start the DTRUST server."""
        if self._running:
            return
        
        logger.info("Starting DTRUST server")
        self._running = True
        
        # Start trust manager
        await self.trust_manager.start()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_task())
    
    @async_catch_exceptions(reraise=True)
    async def stop(self):
        """Stop the DTRUST server."""
        if not self._running:
            return
        
        logger.info("Stopping DTRUST server")
        self._running = False
        
        # Stop trust manager
        await self.trust_manager.stop()
    
    def _on_trust_event(self, event):
        """Handle trust events."""
        logger.info(f"Trust event: {event.event_type} for {event.node_id}")
        
        # In a real implementation, this could:
        # - Send notifications
        # - Update external systems
        # - Trigger automated responses
    
    @async_catch_exceptions()
    async def _cleanup_task(self):
        """Background cleanup task."""
        while self._running:
            try:
                # Cleanup expired challenges
                self.attestation_protocol.cleanup_expired_challenges()
                
                # Cleanup expired bindings
                self.trust_manager.binder.cleanup_expired_bindings()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                log_exception(e, "Error in cleanup task")
    
    @async_catch_exceptions(reraise=True)
    async def start_fl_training(self, rounds: int, min_clients: int, fraction_fit: float = 0.8, fraction_evaluate: float = 0.8):
        """Start federated learning training."""
        if self.fl_server_running:
            raise DTRUSTError("FL training is already running")
        
        logger.info(f"Starting FL training: {rounds} rounds, min {min_clients} clients")
        
        try:
            from ..fl.strategy import TrustAwareStrategy
            import flwr as fl
            from flwr.server import ServerConfig, start_server
            import threading
            import socket
            import time
            
            # Create trust-aware strategy
            strategy = TrustAwareStrategy(
                trust_manager=self.trust_manager,
                min_trust_score=self.config.trust.min_trust_score,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients,
                min_available_clients=min_clients,
                initial_parameters=None,  # Will get from first client
                num_rounds=rounds,  # Add num_rounds parameter
                results_dir=getattr(self.config, 'results_dir', 'training_results'),  # Add configurable results_dir
            )
            
            # Configure FL server with longer client wait times
            config = ServerConfig(
                num_rounds=rounds,
                round_timeout=120.0,  # 2 minutes per round
            )
            
            # Check if port is available
            def is_port_available(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('0.0.0.0', port))
                        return True
                    except OSError:
                        return False
            
            if not is_port_available(self.config.fl.port):
                raise DTRUSTError(f"FL server port {self.config.fl.port} is already in use")
            
            # Start FL server in background thread
            server_ready_event = threading.Event()
            server_error = None
            
            def run_fl_server():
                nonlocal server_error
                try:
                    logger.info(f"Starting Flower server on 0.0.0.0:{self.config.fl.port}")
                    
                    # Set running flag before starting (but server may not be ready yet)
                    self.fl_server_running = True
                    
                    # Signal that we're about to start (not that we're ready)
                    server_ready_event.set()
                    
                    start_server(
                        server_address=f"0.0.0.0:{self.config.fl.port}",
                        config=config,
                        strategy=strategy,
                    )
                    
                    logger.info("Flower server completed")
                    
                except Exception as e:
                    logger.error(f"FL server error: {e}")
                    server_error = e
                finally:
                    self.fl_server_running = False
                    logger.info("FL server thread finished")
            
            self.fl_server_task = threading.Thread(target=run_fl_server, name="FL-Server")
            self.fl_server_task.daemon = True
            self.fl_server_task.start()
            
            # Wait for server thread to start
            if not server_ready_event.wait(timeout=10):
                raise DTRUSTError("FL server failed to start within timeout")
            
            if server_error:
                raise DTRUSTError(f"FL server startup failed: {server_error}")
            
            # Give the server a moment to initialize
            await asyncio.sleep(2)
            
            self.fl_strategy = strategy
            
            logger.info(f"FL server started on port {self.config.fl.port}")
            return {"training_id": f"training_{rounds}_{min_clients}", "status": "started"}
            
        except Exception as e:
            logger.error(f"Failed to start FL training: {e}")
            self.fl_server_running = False  # Reset flag on error
            raise DTRUSTError(f"Failed to start FL training: {e}")
    
    @async_catch_exceptions(reraise=True)
    async def stop_fl_training(self):
        """Stop federated learning training."""
        if not self.fl_server_running:
            return {"status": "not_running"}
        
        logger.info("Stopping FL training")
        
        # Note: Flower server doesn't have a clean shutdown mechanism
        # In a production system, you'd need a more sophisticated approach
        self.fl_server_running = False
        
        return {"status": "stopped"}


# Global server instance
server_instance: Optional[DTRUSTServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global server_instance
    
    try:
        # Startup
        config = DTRUSTConfig.from_file("config/dtrust.yaml")
        setup_logging(
            log_level=config.logging.level,
            log_file="logs/dtrust.log",
            enable_rich_tracebacks=True
        )
        
        server_instance = DTRUSTServer(config)
        await server_instance.start()
        
        yield
        
    except Exception as e:
        log_exception(e, "Error during server startup")
        raise
    finally:
        # Shutdown
        if server_instance:
            try:
                await server_instance.stop()
            except Exception as e:
                log_exception(e, "Error during server shutdown")


def get_server() -> DTRUSTServer:
    """Dependency to get server instance."""
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    return server_instance


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="DTRUST API",
        description="Federated Learning with Trusted Execution Environment",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "dtrust"}
    
    # Node registration endpoints
    @app.post("/api/v1/nodes/register", response_model=NodeIdentity)
    async def register_node(
        request: RegisterNodeRequest,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Register a new node and create its identity."""
        try:
            # Create node identity
            identity = server.identity_manager.create_node_identity(request.enclave_id)
            
            # Create initial node state in trust manager
            server.trust_manager.create_node_state(identity)
            
            logger.info(f"Registered new node: {identity.node_id}")
            return identity
            
        except Exception as e:
            log_exception(e, f"Node registration failed for enclave {request.enclave_id}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/v1/nodes/{node_id}", response_model=NodeState)
    async def get_node_state(
        node_id: str,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Get current state of a node."""
        try:
            node_state = server.trust_manager.get_node_state(node_id)
            if not node_state:
                raise HTTPException(status_code=404, detail="Node not found")
            return node_state
        except HTTPException:
            raise
        except Exception as e:
            log_exception(e, f"Error getting node state for {node_id}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/api/v1/nodes", response_model=List[str])
    async def list_nodes(server: DTRUSTServer = Depends(get_server)):
        """List all registered nodes."""
        try:
            return list(server.identity_manager.list_node_identities().keys())
        except Exception as e:
            log_exception(e, "Error listing nodes")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Attestation endpoints
    @app.post("/api/v1/attest", response_model=AttestationResult)
    async def request_attestation(
        request: AttestationRequest,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Request SGX attestation for a node."""
        try:
            # Process attestation
            result = await server.attestation_protocol.process_attestation_request(
                request.node_id, request.challenge
            )
            
            # Update trust manager with attestation result
            server.trust_manager.update_attestation_result(request.node_id, result)
            
            return result
            
        except AttestationError as e:
            log_exception(e, f"Attestation failed for {request.node_id}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log_exception(e, f"Unexpected error during attestation for {request.node_id}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Trust management endpoints
    @app.get("/api/v1/trust/nodes", response_model=List[NodeState])
    async def get_trusted_nodes(
        min_trust_score: Optional[float] = None,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Get list of trusted nodes eligible for federated learning."""
        try:
            return server.trust_manager.get_trusted_nodes(min_trust_score)
        except Exception as e:
            log_exception(e, "Error getting trusted nodes")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/api/v1/trust/suspicious", response_model=List[NodeState])
    async def get_suspicious_nodes(server: DTRUSTServer = Depends(get_server)):
        """Get list of suspicious nodes."""
        try:
            return server.trust_manager.get_suspicious_nodes()
        except Exception as e:
            log_exception(e, "Error getting suspicious nodes")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/api/v1/trust/training")
    async def record_training_participation(
        request: TrainingParticipationRequest,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Record a node's participation in federated learning training."""
        try:
            server.trust_manager.record_training_participation(
                request.node_id, request.num_examples, request.metrics
            )
            return {"status": "recorded"}
            
        except NodeError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            log_exception(e, f"Error recording training participation for {request.node_id}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/api/v1/trust/suspicious")
    async def record_suspicious_activity(
        request: SuspiciousActivityRequest,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Record suspicious activity for a node."""
        try:
            server.trust_manager.record_suspicious_activity(
                request.node_id, request.reason
            )
            return {"status": "recorded"}
            
        except NodeError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            log_exception(e, f"Error recording suspicious activity for {request.node_id}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Federation management endpoints
    @app.get("/api/v1/federation/stats")
    async def get_federation_stats(server: DTRUSTServer = Depends(get_server)):
        """Get federation statistics."""
        try:
            trust_stats = server.trust_manager.get_federation_stats()
            identity_stats = server.identity_manager.get_identity_stats()
            binding_stats = server.trust_manager.binder.get_binding_stats()
            
            return {
                "trust": trust_stats,
                "identities": identity_stats,
                "bindings": binding_stats
            }
        except Exception as e:
            log_exception(e, "Error getting federation stats")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/api/v1/federation/info", response_model=FederationInfo)
    async def get_federation_info(server: DTRUSTServer = Depends(get_server)):
        """Get federation information."""
        try:
            # Get node counts
            all_nodes = server.trust_manager.get_all_nodes()
            trusted_nodes = server.trust_manager.get_trusted_nodes()
            suspicious_nodes = server.trust_manager.get_suspicious_nodes()
            
            return FederationInfo(
                federation_id="dtrust-main",
                coordinator_node_id="dtrust-coordinator",
                total_nodes=len(all_nodes),
                active_nodes=len([n for n in all_nodes if n.status == "active"]),
                trusted_nodes=len(trusted_nodes),
                suspicious_nodes=len(suspicious_nodes),
                current_round=0,  # round counter is not tracked by the API server
                total_rounds=0    # round counter is not tracked by the API server
            )
        except Exception as e:
            log_exception(e, "Error getting federation info")
            raise HTTPException(status_code=500, detail=str(e))
    
    # FL training endpoints
    @app.post("/api/v1/federation/start-training")
    async def start_training(
        request: StartTrainingRequest,
        server: DTRUSTServer = Depends(get_server)
    ):
        """Start federated learning training."""
        try:
            result = await server.start_fl_training(
                rounds=request.rounds,
                min_clients=request.min_clients,
                fraction_fit=request.fraction_fit,
                fraction_evaluate=request.fraction_evaluate
            )
            return result
        except DTRUSTError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log_exception(e, "Error starting training")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/federation/stop-training")
    async def stop_training(server: DTRUSTServer = Depends(get_server)):
        """Stop federated learning training."""
        try:
            result = await server.stop_fl_training()
            return result
        except Exception as e:
            log_exception(e, "Error stopping training")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/federation/training-status")
    async def get_training_status(server: DTRUSTServer = Depends(get_server)):
        """Get current training status."""
        return {
            "running": server.fl_server_running,
            "port": server.config.fl.port if server.fl_server_running else None
        }
    
    @app.get("/api/v1/federation/training-progress")
    async def get_training_progress(server: DTRUSTServer = Depends(get_server)):
        """Get detailed training progress and results."""
        try:
            if not hasattr(server, 'fl_strategy') or not server.fl_strategy:
                return {
                    "status": "no_training",
                    "message": "No training session active"
                }
            
            # Get training summary from strategy
            summary = server.fl_strategy.get_training_summary()
            
            return {
                "status": "active" if server.fl_server_running else "completed",
                "current_round": summary.get("current_round", 0),
                "total_rounds": summary.get("total_rounds", 0),
                "client_participation": summary.get("client_participation", {}),
                "training_history": summary.get("training_history", []),
                "strategy_config": summary.get("strategy_config", {}),
                "server_running": server.fl_server_running
            }
        except Exception as e:
            log_exception(e, "Error getting training progress")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @app.get("/api/v1/federation/training-results")
    async def get_training_results(server: DTRUSTServer = Depends(get_server)):
        """Get final training results and metrics."""
        try:
            if not hasattr(server, 'fl_strategy') or not server.fl_strategy:
                return {
                    "status": "no_training",
                    "message": "No training session found"
                }
            
            summary = server.fl_strategy.get_training_summary()
            training_history = summary.get("training_history", [])
            
            if not training_history:
                return {
                    "status": "no_results",
                    "message": "No training results available"
                }
            
            # Calculate summary statistics
            total_rounds = len(training_history)
            total_clients = len(summary.get("client_participation", {}))
            
            # Get metrics from last round
            last_round = training_history[-1] if training_history else {}
            
            # Calculate client participation statistics
            participation_stats = {}
            for client_id, rounds in summary.get("client_participation", {}).items():
                participation_stats[client_id] = {
                    "rounds_participated": len(rounds),
                    "participation_rate": len(rounds) / total_rounds if total_rounds > 0 else 0,
                    "rounds": rounds
                }
            
            return {
                "status": "completed",
                "summary": {
                    "total_rounds": total_rounds,
                    "total_clients": total_clients,
                    "strategy_config": summary.get("strategy_config", {}),
                    "last_round_metrics": last_round.get("aggregated_metrics", {}),
                    "total_examples_final": last_round.get("total_examples", 0)
                },
                "client_participation": participation_stats,
                "round_history": training_history,
                "training_completed": not server.fl_server_running
            }
        except Exception as e:
            log_exception(e, "Error getting training results")
            return {
                "status": "error",
                "message": str(e)
            }
    
    # Debug endpoints (remove in production)
    @app.get("/api/v1/debug/challenges")
    async def get_active_challenges(server: DTRUSTServer = Depends(get_server)):
        """Get active attestation challenges (debug only)."""
        try:
            return server.attestation_protocol.get_active_challenges()
        except Exception as e:
            log_exception(e, "Error getting active challenges")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 