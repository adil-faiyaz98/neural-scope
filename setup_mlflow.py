#!/usr/bin/env python
"""
Neural-Scope MLflow Setup Script

This script sets up MLflow tracking for Neural-Scope by:
1. Installing MLflow
2. Configuring the tracking server
3. Creating experiments
4. Setting up a local tracking server (optional)
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-mlflow")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural-Scope MLflow Setup")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--backend-store-uri", default="sqlite:///mlflow.db", help="MLflow backend store URI")
    parser.add_argument("--artifacts-uri", default="./mlruns", help="MLflow artifacts URI")
    parser.add_argument("--start-server", action="store_true", help="Start a local MLflow tracking server")
    parser.add_argument("--port", type=int, default=5000, help="Port for the MLflow tracking server")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the MLflow tracking server")
    parser.add_argument("--create-experiments", action="store_true", help="Create default experiments")
    return parser.parse_args()

def install_mlflow():
    """Install MLflow if not already installed."""
    logger.info("Checking MLflow installation...")
    
    try:
        import mlflow
        logger.info(f"MLflow {mlflow.__version__} is already installed")
    except ImportError:
        logger.info("MLflow not found, installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "mlflow"], check=True)
            logger.info("MLflow installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing MLflow: {e}")
            sys.exit(1)

def configure_mlflow(args):
    """Configure MLflow tracking."""
    logger.info(f"Configuring MLflow tracking with URI: {args.tracking_uri}")
    
    # Create .env file with MLflow configuration
    env_content = f"""
# MLflow Configuration
MLFLOW_TRACKING_URI={args.tracking_uri}
MLFLOW_EXPERIMENT_NAME=neural-scope-default
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    # Create a mlflow.yml file with configuration
    os.makedirs(".mlflow", exist_ok=True)
    mlflow_config = f"""
# MLflow Configuration
tracking_uri: {args.tracking_uri}
artifact_location: {args.artifacts_uri}
experiment_name: neural-scope-default
"""
    
    with open(".mlflow/mlflow.yml", "w") as f:
        f.write(mlflow_config)
    
    logger.info("MLflow configuration saved to .env and .mlflow/mlflow.yml")

def create_experiments(args):
    """Create default experiments in MLflow."""
    if not args.create_experiments:
        return
        
    logger.info("Creating default experiments in MLflow...")
    
    try:
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri(args.tracking_uri)
        
        # Create experiments
        experiments = [
            "neural-scope-default",
            "neural-scope-pytorch",
            "neural-scope-tensorflow",
            "neural-scope-huggingface"
        ]
        
        for experiment_name in experiments:
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
                else:
                    logger.info(f"Experiment already exists: {experiment_name} (ID: {experiment.experiment_id})")
            except Exception as e:
                logger.error(f"Error creating experiment {experiment_name}: {e}")
    
    except Exception as e:
        logger.error(f"Error creating experiments: {e}")

def start_mlflow_server(args):
    """Start a local MLflow tracking server."""
    if not args.start_server:
        return
        
    logger.info(f"Starting MLflow tracking server on {args.host}:{args.port}...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.backend_store_uri.replace("sqlite:///", "")), exist_ok=True)
    os.makedirs(args.artifacts_uri, exist_ok=True)
    
    # Build the command
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", args.backend_store_uri,
        "--default-artifact-root", args.artifacts_uri,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    # Start the server
    try:
        logger.info("MLflow server starting...")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Tracking UI: http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start the server in a subprocess
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("MLflow server stopped")
    except Exception as e:
        logger.error(f"Error starting MLflow server: {e}")

def main():
    """Run the MLflow setup script."""
    args = parse_args()
    
    # Install MLflow
    install_mlflow()
    
    # Configure MLflow
    configure_mlflow(args)
    
    # Create experiments
    create_experiments(args)
    
    # Start MLflow server
    start_mlflow_server(args)
    
    if not args.start_server:
        logger.info("\nMLflow setup completed successfully!")
        logger.info("\nTo start the MLflow tracking server:")
        logger.info(f"python setup_mlflow.py --start-server --tracking-uri {args.tracking_uri}")
        logger.info("\nTo use MLflow in your scripts:")
        logger.info("import mlflow")
        logger.info(f"mlflow.set_tracking_uri('{args.tracking_uri}')")
        logger.info("mlflow.set_experiment('neural-scope-default')")

if __name__ == "__main__":
    main()
