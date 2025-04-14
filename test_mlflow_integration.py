#!/usr/bin/env python
"""
Test MLflow Integration for Neural-Scope

This script tests the MLflow integration for Neural-Scope by:
1. Setting up MLflow
2. Tracking model analysis results
3. Registering an optimized model
"""

import os
import sys
import argparse
import json
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-mlflow")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test MLflow Integration for Neural-Scope")
    parser.add_argument("--results-dir", default="test_results", help="Directory with analysis results")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="neural-scope-test", help="MLflow experiment name")
    return parser.parse_args()

def setup_mlflow(args):
    """Set up MLflow tracking."""
    logger.info(f"Setting up MLflow tracking with URI: {args.tracking_uri}")

    try:
        import mlflow

        # Set tracking URI
        mlflow.set_tracking_uri(args.tracking_uri)

        # Create experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(args.experiment_name)
            logger.info(f"Created experiment: {args.experiment_name} (ID: {experiment_id})")
        else:
            logger.info(f"Using existing experiment: {args.experiment_name} (ID: {experiment.experiment_id})")

        # Set experiment
        mlflow.set_experiment(args.experiment_name)

        return True
    except ImportError:
        logger.error("MLflow is not installed. Please install it with 'pip install mlflow'.")
        return False
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        return False

def track_analysis_results(args):
    """Track model analysis results in MLflow."""
    logger.info("Tracking model analysis results in MLflow...")

    # Load analysis results
    analysis_path = os.path.join(args.results_dir, "analysis", "model_analysis.json")
    if not os.path.exists(analysis_path):
        logger.error(f"Analysis results not found at {analysis_path}")
        return False

    try:
        with open(analysis_path, 'r') as f:
            analysis_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        return False

    # Load optimization results
    optimization_path = os.path.join(args.results_dir, "optimization", "optimization_results.json")
    if not os.path.exists(optimization_path):
        logger.error(f"Optimization results not found at {optimization_path}")
        return False

    try:
        with open(optimization_path, 'r') as f:
            optimization_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading optimization results: {e}")
        return False

    try:
        import mlflow

        # Start a new run
        with mlflow.start_run() as run:
            # Log model name as a tag
            mlflow.set_tag("model_name", analysis_results["model_name"])
            mlflow.set_tag("model_source", "pytorch_hub")
            mlflow.set_tag("architecture", analysis_results["architecture"])

            # Log analysis results as metrics
            mlflow.log_metric("parameters", analysis_results["parameters"])
            mlflow.log_metric("layers", analysis_results["layers"])
            mlflow.log_metric("memory_usage_mb", analysis_results["memory_usage"])
            mlflow.log_metric("inference_time_ms", analysis_results["inference_time_ms"])

            # Log optimization results as metrics
            mlflow.log_metric("original_size_mb", optimization_results["original_size"])
            mlflow.log_metric("optimized_size_mb", optimization_results["optimized_size"])
            mlflow.log_metric("size_reduction_percentage", optimization_results["size_reduction_percentage"])
            mlflow.log_metric("inference_speedup", optimization_results["inference_speedup"])

            # Log techniques as tags
            mlflow.set_tag("techniques", ",".join(optimization_results["techniques"]))

            # Log analysis results as a JSON artifact
            with open("analysis_results.json", "w") as f:
                json.dump(analysis_results, f, indent=2)
            mlflow.log_artifact("analysis_results.json")

            # Log optimization results as a JSON artifact
            with open("optimization_results.json", "w") as f:
                json.dump(optimization_results, f, indent=2)
            mlflow.log_artifact("optimization_results.json")

            # Log comprehensive report as an artifact
            comprehensive_path = os.path.join(args.results_dir, "comprehensive_report.html")
            if os.path.exists(comprehensive_path):
                mlflow.log_artifact(comprehensive_path)

            logger.info(f"Results tracked in MLflow run: {run.info.run_id}")

        return True
    except ImportError:
        logger.error("MLflow is not installed. Please install it with 'pip install mlflow'.")
        return False
    except Exception as e:
        logger.error(f"Error tracking results in MLflow: {e}")
        return False

def register_optimized_model(args):
    """Register an optimized model in MLflow."""
    logger.info("Registering optimized model in MLflow...")

    # Load the model
    model_path = os.path.join(args.results_dir, "models", "resnet18.pt")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return False

    try:
        # Add safe globals for ResNet
        import torch.serialization
        import torchvision.models.resnet
        torch.serialization.add_safe_globals([torchvision.models.resnet.ResNet])

        # Load the model with weights_only=False for compatibility
        model = torch.load(model_path, weights_only=False)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

    # Load optimization results
    optimization_path = os.path.join(args.results_dir, "optimization", "optimization_results.json")
    if not os.path.exists(optimization_path):
        logger.error(f"Optimization results not found at {optimization_path}")
        return False

    try:
        with open(optimization_path, 'r') as f:
            optimization_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading optimization results: {e}")
        return False

    try:
        import mlflow
        import mlflow.pytorch

        # Start a new run
        with mlflow.start_run() as run:
            # Log model name as a tag
            mlflow.set_tag("model_name", optimization_results["model_name"])
            mlflow.set_tag("model_source", "pytorch_hub")
            mlflow.set_tag("model_type", "optimized")
            mlflow.set_tag("techniques", ",".join(optimization_results["techniques"]))

            # Log optimization results as metrics
            mlflow.log_metric("original_size_mb", optimization_results["original_size"])
            mlflow.log_metric("optimized_size_mb", optimization_results["optimized_size"])
            mlflow.log_metric("size_reduction_percentage", optimization_results["size_reduction_percentage"])
            mlflow.log_metric("inference_speedup", optimization_results["inference_speedup"])

            # Log the model
            mlflow.pytorch.log_model(model, "model")

            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            model_details = mlflow.register_model(model_uri, f"{optimization_results['model_name']}-optimized")

            logger.info(f"Model registered in MLflow with version: {model_details.version}")

        return True
    except ImportError:
        logger.error("MLflow is not installed. Please install it with 'pip install mlflow'.")
        return False
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        return False

def main():
    """Run the MLflow integration test."""
    args = parse_args()

    # Step 1: Set up MLflow
    if not setup_mlflow(args):
        logger.error("Failed to set up MLflow. Exiting...")
        sys.exit(1)

    # Step 2: Track analysis results
    if not track_analysis_results(args):
        logger.error("Failed to track analysis results. Exiting...")
        sys.exit(1)

    # Step 3: Register optimized model
    if not register_optimized_model(args):
        logger.error("Failed to register optimized model. Exiting...")
        sys.exit(1)

    logger.info("\nMLflow integration test completed successfully!")
    logger.info(f"MLflow tracking URI: {args.tracking_uri}")
    logger.info(f"MLflow experiment: {args.experiment_name}")
    logger.info("\nTo view the results, run:")
    logger.info(f"mlflow ui --backend-store-uri {args.tracking_uri}")

if __name__ == "__main__":
    main()
