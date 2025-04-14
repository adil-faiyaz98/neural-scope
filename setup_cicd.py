#!/usr/bin/env python
"""
Neural-Scope CI/CD Setup Script

This script sets up the Neural-Scope CI/CD integration by:
1. Installing required dependencies
2. Creating necessary directories
3. Setting up CI/CD workflows
4. Configuring MLflow tracking (optional)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural-Scope CI/CD Setup")
    parser.add_argument("--ci-system", choices=["github_actions", "gitlab_ci", "jenkins", "azure_devops"], 
                      default="github_actions", help="CI/CD system to use")
    parser.add_argument("--mlflow", action="store_true", help="Set up MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--create-example", action="store_true", help="Create example model and dataset")
    return parser.parse_args()

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Install Neural-Scope with all dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[all]"], check=True)
        print("Neural-Scope installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Neural-Scope: {e}")
        sys.exit(1)
    
    # Install additional dependencies for CI/CD
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)
        print("Additional dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing additional dependencies: {e}")

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("Directories created successfully")

def setup_ci_workflow(args):
    """Set up CI/CD workflow."""
    print(f"Setting up {args.ci_system} workflow...")
    
    # Create the command to run the Neural-Scope CI/CD runner
    cmd = [
        sys.executable, "neural_scope_cicd.py", "create-workflow",
        "--system", args.ci_system,
        "--output-dir", ".github/workflows" if args.ci_system == "github_actions" else ".",
        "--workflow-name", "model_optimization"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"{args.ci_system} workflow set up successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up {args.ci_system} workflow: {e}")

def setup_mlflow(args):
    """Set up MLflow tracking."""
    if not args.mlflow:
        return
    
    print("Setting up MLflow tracking...")
    
    # Install MLflow if not already installed
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "mlflow"], check=True)
        print("MLflow installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing MLflow: {e}")
        return
    
    # Create MLflow configuration
    mlflow_config = f"""
# MLflow Configuration
MLFLOW_TRACKING_URI={args.mlflow_tracking_uri}
MLFLOW_EXPERIMENT_NAME=neural-scope-optimization
"""
    
    # Save configuration to .env file
    with open(".env", "w") as f:
        f.write(mlflow_config)
    
    print(f"MLflow tracking configured with URI: {args.mlflow_tracking_uri}")

def create_example(args):
    """Create example model and dataset."""
    if not args.create_example:
        return
    
    print("Creating example model and dataset...")
    
    # Create the command to run the example ML workflow
    cmd = [
        sys.executable, "examples/ml_workflow_example.py",
        "--output-dir", "example_results",
        "--framework", "pytorch",
        "--epochs", "5"
    ]
    
    if args.mlflow:
        cmd.extend(["--track-with-mlflow", "--mlflow-tracking-uri", args.mlflow_tracking_uri])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Example model and dataset created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error creating example: {e}")

def main():
    """Run the setup script."""
    args = parse_args()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Set up CI/CD workflow
    setup_ci_workflow(args)
    
    # Set up MLflow tracking
    setup_mlflow(args)
    
    # Create example
    create_example(args)
    
    print("\nNeural-Scope CI/CD integration set up successfully!")
    print("\nNext steps:")
    print("1. Review the CI/CD workflow configuration")
    print("2. Add your models to the 'models' directory")
    print("3. Add your datasets to the 'data' directory")
    print("4. Run the CI/CD workflow manually or push changes to trigger it")
    
    if args.mlflow:
        print("\nTo start the MLflow UI:")
        print(f"mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")

if __name__ == "__main__":
    main()
