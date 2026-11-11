"""
Command Line Interface for DTRUST framework.
"""

import asyncio
import click
from pathlib import Path
from typing import Optional
import os

from .core.config import load_config, DTRUSTConfig
from .core.logging import setup_logging, get_logger, log_exception
from .core.exceptions import DTRUSTError
from .api.server import DTRUSTServer
from .api.client import DTRUSTClient


logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', '-l', type=click.Path(), help='Log file path')
@click.option('--logs-dir', type=click.Path(), help='Directory for all log files (overrides individual log file paths)')
@click.option('--results-dir', type=click.Path(), help='Directory for training results and intermediate files')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, log_file: Optional[str], logs_dir: Optional[str], results_dir: Optional[str]):
    """DTRUST: Federated Learning with Trusted Execution Environment"""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    try:
        # Load configuration
        config_path = Path(config) if config else None
        ctx.obj['config'] = load_config(config_path)
        
        # Store CLI parameters in context for use by subcommands
        ctx.obj['logs_dir'] = logs_dir
        ctx.obj['results_dir'] = results_dir
        
        # Setup logging with custom log file if provided
        log_level = "DEBUG" if verbose else ctx.obj['config'].log_level
        
        # Use log file from CLI parameter, environment variable, or config
        # If logs_dir is specified, use it to construct the log file path
        if logs_dir and not log_file:
            # Create logs directory if it doesn't exist
            logs_path = Path(logs_dir)
            logs_path.mkdir(parents=True, exist_ok=True)
            log_file_path = logs_path / "dtrust.log"
        else:
            log_file_path = log_file or os.getenv('DTRUST_LOGGING_FILE') or ctx.obj['config'].logging.file
        
        setup_logging(log_level, log_file=log_file_path)
        
        logger.info(f"DTRUST CLI initialized with config: {config_path or 'default'}")
        if log_file_path:
            logger.info(f"Logging to file: {log_file_path}")
        if logs_dir:
            logger.info(f"Logs directory: {logs_dir}")
        if results_dir:
            logger.info(f"Results directory: {results_dir}")
        
    except Exception as e:
        click.echo(f"Error initializing DTRUST: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--host', help='Server host address')
@click.option('--port', type=int, help='Server port')
@click.option('--workers', type=int, help='Number of worker processes')
@click.pass_context
def server(ctx, host: Optional[str], port: Optional[int], workers: Optional[int]):
    """Start the DTRUST federation server."""
    
    config = ctx.obj['config']
    
    # Override config with CLI options
    if host:
        config.api.host = host
    if port:
        config.api.port = port
    if workers:
        config.api.workers = workers
    
    # Set results directory from CLI parameter
    if ctx.obj.get('results_dir'):
        config.results_dir = ctx.obj['results_dir']
        logger.info(f"Using custom results directory: {config.results_dir}")
    
    click.echo(f"Starting DTRUST server on {config.api.host}:{config.api.port}")
    
    try:
        from .api.server import create_app
        import uvicorn
        
        app = create_app()
        
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            workers=config.api.workers,
            reload=config.api.reload,
            log_level=config.api.log_level
        )
        
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--node-id', required=True, help='Unique node identifier')
@click.option('--enclave-id', required=True, help='SGX enclave identifier')
@click.option('--server-address', default='localhost:8000', help='Federation server address')
@click.option('--data-path', type=click.Path(exists=True), help='Path to training data')
@click.option('--model', default='simple_nn_mnist', help='Model name from ML registry (default: simple_nn_mnist)')
@click.pass_context
def client(ctx, node_id: str, enclave_id: str, server_address: str, data_path: Optional[str], model: str):
    """Start a DTRUST client node."""
    
    config = ctx.obj['config']
    
    click.echo(f"Starting DTRUST client: {node_id}")
    click.echo(f"Enclave ID: {enclave_id}")
    click.echo(f"Server: {server_address}")
    click.echo(f"Model: {model}")
    
    try:
        from .fl.client import DTRUSTClient
        
        client_instance = DTRUSTClient(
            node_id=node_id,
            enclave_id=enclave_id,
            server_address=server_address,
            config=config,
            data_path=data_path,
            model_name=model
        )
        
        # Run client
        asyncio.run(client_instance.start())
        
    except Exception as e:
        click.echo(f"Error starting client: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--node-id', required=True, help='Node ID to register')
@click.option('--enclave-id', required=True, help='SGX enclave ID')
@click.option('--server-address', default='localhost:8000', help='Server address')
@click.pass_context
def register(ctx, node_id: str, enclave_id: str, server_address: str):
    """Register a new node with the federation."""
    
    click.echo(f"Registering node {node_id} with enclave {enclave_id}")
    
    try:
        import httpx
        
        response = httpx.post(
            f"http://{server_address}/api/v1/nodes/register",
            json={
                "enclave_id": enclave_id
            }
        )
        
        if response.status_code == 200:
            click.echo("[OK] Node registered successfully")
            result = response.json()
            click.echo(f"Public key: {result.get('public_key', 'N/A')[:64]}...")
        else:
            click.echo(f"[ERR] Registration failed: {response.text}", err=True)
            ctx.exit(1)
            
    except Exception as e:
        click.echo(f"Error registering node: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--node-id', required=True, help='Node ID to attest')
@click.option('--server-address', default='localhost:8000', help='Server address')
@click.pass_context
def attest(ctx, node_id: str, server_address: str):
    """Perform SGX attestation for a node."""
    
    click.echo(f"Performing attestation for node {node_id}")
    
    try:
        import httpx
        import secrets
        
        # Generate challenge
        challenge = secrets.token_hex(32)
        
        response = httpx.post(
            f"http://{server_address}/api/v1/attest",
            json={"node_id": node_id, "challenge": challenge}
        )
        
        if response.status_code == 200:
            result = response.json()
            click.echo("[OK] Attestation successful")
            click.echo(f"Trust level: {result.get('trust_level', 'unknown')}")
            click.echo(f"Confidence: {result.get('confidence', 0):.3f}")
            click.echo(f"Valid until: {result.get('expiry_time', 'unknown')}")
        else:
            click.echo(f"[ERR] Attestation failed: {response.text}", err=True)
            ctx.exit(1)
            
    except Exception as e:
        click.echo(f"Error during attestation: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--server-address', default='localhost:8000', help='Server address')
@click.pass_context
def status(ctx, server_address: str):
    """Show federation status and node information."""
    
    try:
        import httpx
        
        # Get federation info
        response = httpx.get(f"http://{server_address}/api/v1/federation/info")
        
        if response.status_code == 200:
            info = response.json()
            
            click.echo("[FED] Federation Status")
            click.echo("=" * 50)
            click.echo(f"Federation ID: {info.get('federation_id', 'unknown')}")
            click.echo(f"Current Round: {info.get('current_round', 0)}/{info.get('total_rounds', 0)}")
            click.echo(f"Total Nodes: {info.get('total_nodes', 0)}")
            click.echo(f"Active Nodes: {info.get('active_nodes', 0)}")
            click.echo(f"Trusted Nodes: {info.get('trusted_nodes', 0)}")
            click.echo(f"Suspicious Nodes: {info.get('suspicious_nodes', 0)}")
            
            # Get node list (returns list of node IDs)
            nodes_response = httpx.get(f"http://{server_address}/api/v1/nodes")
            if nodes_response.status_code == 200:
                node_ids = nodes_response.json()
                
                click.echo("\nNode Details")
                click.echo("=" * 50)
                
                if not node_ids:
                    click.echo("No nodes registered")
                else:
                    for node_id in node_ids:
                        try:
                            # Get detailed node information
                            node_response = httpx.get(f"http://{server_address}/api/v1/nodes/{node_id}")
                            if node_response.status_code == 200:
                                node = node_response.json()
                                
                                status_emoji = {
                                    'active': '',
                                    'trusted': '', 
                                    'suspicious': '',
                                    'banned': '',
                                    'inactive': ''
                                }.get(node.get('status', 'unknown'), '')
                                
                                click.echo(
                                    f"{status_emoji} {node_id[:16]}... "
                                    f"Trust: {node.get('trust_score', 0):.3f} "
                                    f"Status: {node.get('status', 'unknown')}"
                                )
                            else:
                                click.echo(f"[WARN] {node_id[:16]}... (details unavailable)")
                        except Exception as e:
                            click.echo(f"[WARN] {node_id[:16]}... (error: {e})")
        else:
            click.echo(f"[ERR] Failed to get status: {response.text}", err=True)
            ctx.exit(1)
            
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--rounds', type=int, help='Number of training rounds')
@click.option('--min-clients', type=int, help='Minimum clients required')
@click.option('--server-address', default='localhost:8000', help='Server address')
@click.pass_context
def train(ctx, rounds: Optional[int], min_clients: Optional[int], server_address: str):
    """Start federated learning training."""
    
    config = ctx.obj['config']
    
    # Override config with CLI options
    if rounds:
        config.fl.rounds = rounds
    if min_clients:
        config.fl.min_clients = min_clients
    
    click.echo(f"Starting federated learning training")
    click.echo(f"Rounds: {config.fl.rounds}")
    click.echo(f"Min clients: {config.fl.min_clients}")
    
    try:
        import httpx
        
        response = httpx.post(
            f"http://{server_address}/api/v1/federation/start-training",
            json={
                "rounds": config.fl.rounds,
                "min_clients": config.fl.min_clients,
                "fraction_fit": config.fl.fraction_fit,
                "fraction_evaluate": config.fl.fraction_evaluate
            }
        )
        
        if response.status_code == 200:
            click.echo("[OK] Training started successfully")
            result = response.json()
            click.echo(f"Training ID: {result.get('training_id', 'unknown')}")
        else:
            click.echo(f"[ERR] Failed to start training: {response.text}", err=True)
            ctx.exit(1)
            
    except Exception as e:
        click.echo(f"Error starting training: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def models(ctx):
    """List available ML models from the registry."""
    
    try:
        from .ml.models.registry import ModelRegistry, get_model_info
        
        available_models = ModelRegistry.list_models()
        
        if not available_models:
            click.echo("No models registered in the ML registry")
            return
        
        click.echo("Available ML Models")
        click.echo("=" * 50)
        
        for model_name in sorted(available_models):
            try:
                info = get_model_info(model_name)
                config = info.get('default_config', {})
                
                click.echo(f"{model_name}")
                click.echo(f"   Class: {info['class']}")
                
                # Show key configuration parameters
                if config:
                    key_params = []
                    for key, value in config.items():
                        if key in ['input_size', 'input_channels', 'num_classes', 'hidden_size']:
                            key_params.append(f"{key}={value}")
                    
                    if key_params:
                        click.echo(f"   Config: {', '.join(key_params)}")
                
                # Show description if available
                if info.get('docstring'):
                    # Get first line of docstring
                    first_line = info['docstring'].strip().split('\n')[0]
                    if first_line:
                        click.echo(f"   Description: {first_line}")
                
                click.echo()
                
            except Exception as e:
                click.echo(f"[WARN] {model_name}: Error getting info ({e})")
        
        click.echo("[HINT] Usage: dtrust client --model <model_name> ...")
        click.echo("[HINT] Example: dtrust client --model simple_cnn_cifar10 --node-id node1 --enclave-id enc1")
        
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize DTRUST environment and create sample files."""
    
    click.echo("Initializing DTRUST environment...")
    
    try:
        # Create directories
        dirs = ['config', 'data', 'logs', 'models']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
            click.echo(f"[OK] Created directory: {dir_name}")
        
        # Create sample config if it doesn't exist
        config_path = Path('config/dtrust.yaml')
        if not config_path.exists():
            # Copy the default config
            import shutil
            default_config = Path(__file__).parent.parent / 'config' / 'dtrust.yaml'
            if default_config.exists():
                shutil.copy(default_config, config_path)
                click.echo(f"[OK] Created config file: {config_path}")
        
        # Create .env file template
        env_path = Path('.env')
        if not env_path.exists():
            env_content = """# DTRUST Environment Variables
# Copy this file and customize for your environment

# Database
DTRUST_DATABASE_URL=postgresql://dtrust:dtrust@localhost:5432/dtrust

# SGX Settings (for production)
# DTRUST_SGX_MOCK_MODE=false
# DTRUST_SGX_SPID=your_spid_here
# DTRUST_SGX_PRIMARY_KEY=your_primary_key_here

# Trust Management
# DTRUST_TRUST_MIN_SCORE=0.7

# API Settings
# DTRUST_API_HOST=0.0.0.0
# DTRUST_API_PORT=8000
"""
            env_path.write_text(env_content)
            click.echo(f"[OK] Created environment template: {env_path}")
        
        click.echo("\nDTRUST environment initialized successfully!")
        click.echo("\nNext steps:")
        click.echo("1. Review and customize config/dtrust.yaml")
        click.echo("2. Set up your database (PostgreSQL)")
        click.echo("3. Start the server: dtrust server")
        click.echo("4. Register and start clients: dtrust client --node-id <id> --enclave-id <id>")
        
    except Exception as e:
        click.echo(f"Error initializing environment: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option('--dataset', type=click.Choice(['mnist', 'cifar10', 'custom']), required=True, help='Dataset type to prepare')
@click.option('--input-path', type=click.Path(exists=True), help='Path to input data file (for custom dataset)')
@click.option('--output-dir', type=click.Path(), default='./federated_data', help='Output directory for client data')
@click.option('--num-clients', type=int, default=4, help='Number of clients to split data for')
@click.option('--split-method', type=click.Choice(['iid', 'non_iid', 'dirichlet']), default='iid', help='Data split method')
@click.option('--alpha', type=float, default=0.5, help='Alpha parameter for Dirichlet split (lower = more non-IID)')
@click.option('--train-ratio', type=float, default=0.8, help='Ratio of data for training vs testing')
@click.pass_context
def prepare_data(ctx, dataset: str, input_path: Optional[str], output_dir: str, num_clients: int, 
                split_method: str, alpha: float, train_ratio: float):
    """Prepare federated data splits for clients."""
    
    try:
        from pathlib import Path
        import json
        import numpy as np
        from .ml.utils.data_utils import (
            load_mnist_data, load_cifar10_data, load_data_auto,
            create_federated_split, create_data_loaders
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Preparing {dataset} data for {num_clients} clients")
        click.echo(f"Split method: {split_method}")
        click.echo(f"Output directory: {output_path.absolute()}")
        
        # Load the dataset
        if dataset == 'mnist':
            click.echo("Loading MNIST dataset...")
            X_train, y_train, X_test, y_test = load_mnist_data()
            
            # Combine train and test for federated split
            all_features = np.concatenate([X_train, X_test])
            all_targets = np.concatenate([y_train, y_test])
            
        elif dataset == 'cifar10':
            click.echo("Loading CIFAR-10 dataset...")
            X_train, y_train, X_test, y_test = load_cifar10_data()
            
            # Combine train and test for federated split
            all_features = np.concatenate([X_train, X_test])
            all_targets = np.concatenate([y_train, y_test])
            
        elif dataset == 'custom':
            if not input_path:
                click.echo("[ERR] --input-path is required for custom dataset", err=True)
                ctx.exit(1)
            
            click.echo(f"Loading custom data from {input_path}...")
            all_features, all_targets = load_data_auto(input_path)
            
        else:
            click.echo(f"[ERR] Unknown dataset: {dataset}", err=True)
            ctx.exit(1)
        
        click.echo(f"[OK] Loaded {len(all_features)} samples with {all_features.shape[1]} features")
        click.echo(f"Classes: {len(np.unique(all_targets))} ({np.unique(all_targets)})")
        
        # Create federated splits
        click.echo(f"Creating federated splits using {split_method} method...")
        
        client_splits = create_federated_split(
            all_features, all_targets,
            num_clients=num_clients,
            split_method=split_method,
            alpha=alpha,
            min_samples=10
        )
        
        # Save client data
        client_info = []
        
        for client_id, (client_features, client_targets) in enumerate(client_splits):
            client_name = f"client_{client_id + 1}"
            client_dir = output_path / client_name
            client_dir.mkdir(exist_ok=True)
            
            # Save as numpy files
            np.save(client_dir / "features.npy", client_features)
            np.save(client_dir / "targets.npy", client_targets)
            
            # Save as npz file (combined)
            np.savez(
                client_dir / "data.npz",
                data=client_features,
                targets=client_targets
            )
            
            # Create client info
            unique_classes, class_counts = np.unique(client_targets, return_counts=True)
            class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))
            
            info = {
                "client_id": client_name,
                "num_samples": len(client_features),
                "num_features": client_features.shape[1],
                "num_classes": len(unique_classes),
                "class_distribution": class_distribution,
                "data_files": {
                    "features": str(client_dir / "features.npy"),
                    "targets": str(client_dir / "targets.npy"),
                    "combined": str(client_dir / "data.npz")
                }
            }
            
            client_info.append(info)
            
            click.echo(f"   {client_name}: {len(client_features)} samples, classes {list(unique_classes)}")
        
        # Save overall info
        overall_info = {
            "dataset": dataset,
            "split_method": split_method,
            "alpha": alpha if split_method == 'dirichlet' else None,
            "num_clients": num_clients,
            "total_samples": len(all_features),
            "num_features": all_features.shape[1],
            "num_classes": len(np.unique(all_targets)),
            "clients": client_info
        }
        
        with open(output_path / "federated_data_info.json", 'w') as f:
            json.dump(overall_info, f, indent=2)
        
        click.echo(f"\n[OK] Federated data preparation completed!")
        click.echo(f"[STAT] Summary:")
        click.echo(f"   Total samples: {len(all_features)}")
        click.echo(f"   Clients: {num_clients}")
        click.echo(f"   Split method: {split_method}")
        click.echo(f"   Output: {output_path.absolute()}")
        
        click.echo(f"\n[HINT] Usage examples:")
        for i, info in enumerate(client_info):
            client_name = info["client_id"]
            data_file = info["data_files"]["combined"]
            click.echo(f"   Client {i+1}: --data-path {data_file}")
        
        click.echo(f"\nDetailed info saved to: {output_path / 'federated_data_info.json'}")
        
    except Exception as e:
        click.echo(f"[ERR] Error preparing data: {e}", err=True)
        ctx.exit(1)


if __name__ == '__main__':
    cli() 