#!/usr/bin/env python
"""
Neural-Scope CI/CD Workflow Runner

This script provides a unified interface for running the Neural-Scope CI/CD workflow:
1. Set up the environment
2. Fetch and analyze a model (pre-trained or custom)
3. Track results with MLflow
4. Generate comprehensive reports

Usage:
    python run_neural_scope.py --model resnet18 --source pytorch_hub
    python run_neural_scope.py --model-path models/my_model.pt --config my_config.yaml
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-runner")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural-Scope CI/CD Workflow Runner")
    
    # Model source group (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Pre-trained model name")
    model_group.add_argument("--model-path", help="Path to custom model file")
    
    # Model source options
    parser.add_argument("--source", choices=["pytorch_hub", "tensorflow_hub", "huggingface"], 
                      default="pytorch_hub", help="Source of the pre-trained model")
    
    # Configuration options
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], 
                      default="pytorch", help="Model framework")
    parser.add_argument("--output-dir", default="neural_scope_results", help="Directory to save results")
    
    # Analysis options
    parser.add_argument("--techniques", default="quantization,pruning", 
                      help="Comma-separated list of optimization techniques")
    parser.add_argument("--dataset-path", help="Path to validation dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    
    # Feature flags
    parser.add_argument("--security-check", action="store_true", help="Perform security checks")
    parser.add_argument("--verify-metrics", action="store_true", help="Verify model metrics")
    parser.add_argument("--mlflow", action="store_true", help="Track results with MLflow")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    
    # Setup options
    parser.add_argument("--setup", action="store_true", help="Run setup before analysis")
    parser.add_argument("--ci-system", choices=["github_actions", "gitlab_ci", "jenkins", "azure_devops"], 
                      default="github_actions", help="CI/CD system to use for setup")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file."""
    if not config_path:
        return {}
        
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path}")
                return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

def run_setup(args):
    """Run setup if requested."""
    if not args.setup:
        return
        
    logger.info("Running Neural-Scope setup...")
    
    # Run CI/CD setup
    try:
        cmd = [
            sys.executable, "setup_cicd.py",
            "--ci-system", args.ci_system
        ]
        subprocess.run(cmd, check=True)
        logger.info("CI/CD setup completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running CI/CD setup: {e}")
    
    # Run MLflow setup if requested
    if args.mlflow:
        try:
            cmd = [
                sys.executable, "setup_mlflow.py",
                "--tracking-uri", args.mlflow_tracking_uri,
                "--create-experiments"
            ]
            subprocess.run(cmd, check=True)
            logger.info("MLflow setup completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running MLflow setup: {e}")

def analyze_pretrained_model(args):
    """Analyze a pre-trained model."""
    logger.info(f"Analyzing pre-trained model: {args.model} from {args.source}...")
    
    # Build command
    cmd = [
        sys.executable, "fetch_and_analyze.py",
        "--model", args.model,
        "--source", args.source,
        "--output-dir", args.output_dir
    ]
    
    # Add optional arguments
    if args.security_check:
        cmd.append("--security-check")
    
    if args.verify_metrics:
        cmd.append("--verify-metrics")
    
    if args.mlflow:
        cmd.append("--mlflow")
        cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
    
    if args.config:
        cmd.extend(["--config", args.config])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Pre-trained model analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error analyzing pre-trained model: {e}")
        sys.exit(1)

def analyze_custom_model(args):
    """Analyze a custom model."""
    logger.info(f"Analyzing custom model: {args.model_path}...")
    
    # Build command
    cmd = [
        sys.executable, "neural_scope_cicd.py", "optimize",
        "--model-path", args.model_path,
        "--output-dir", args.output_dir,
        "--framework", args.framework,
        "--techniques", args.techniques
    ]
    
    # Add optional arguments
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Custom model analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error analyzing custom model: {e}")
        sys.exit(1)
    
    # Perform security check if requested
    if args.security_check:
        logger.info("Performing security check...")
        # This would be implemented in a separate script
        # For now, we'll just log a message
        logger.info("Security check not implemented for custom models yet")
    
    # Track results with MLflow if requested
    if args.mlflow:
        logger.info("Tracking results with MLflow...")
        try:
            cmd = [
                sys.executable, "neural_scope_cicd.py", "track",
                "--model-name", os.path.basename(args.model_path),
                "--results-path", os.path.join(args.output_dir, "optimization_results.json"),
                "--tracking-uri", args.mlflow_tracking_uri,
                "--experiment-name", "neural-scope-custom"
            ]
            subprocess.run(cmd, check=True)
            logger.info("Results tracked in MLflow")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error tracking results: {e}")

def main():
    """Run the Neural-Scope CI/CD workflow."""
    args = parse_args()
    
    # Load configuration if provided
    config = load_config(args.config)
    
    # Run setup if requested
    run_setup(args)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze model based on source
    if args.model:
        analyze_pretrained_model(args)
    elif args.model_path:
        analyze_custom_model(args)
    
    logger.info(f"\nNeural-Scope workflow completed successfully!")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Open the results directory
    try:
        if os.name == 'nt':  # Windows
            os.startfile(args.output_dir)
        elif os.name == 'posix':  # macOS or Linux
            import subprocess
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', args.output_dir])
    except:
        pass

if __name__ == "__main__":
    main()
