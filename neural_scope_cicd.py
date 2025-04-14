#!/usr/bin/env python
"""
Neural-Scope CI/CD Runner

This script provides a comprehensive CI/CD integration for Neural-Scope,
enabling automated model analysis, optimization, and validation as part of
ML workflows.

Usage:
    # Run model optimization
    python neural_scope_cicd.py optimize --model-path models/model.pt --output-dir results

    # Create a CI/CD workflow
    python neural_scope_cicd.py create-workflow --system github_actions --output-dir .github/workflows

    # Track results with MLflow
    python neural_scope_cicd.py track --model-name my_model --results-path results/optimization_results.json
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-cicd")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural-Scope CI/CD Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a model")
    optimize_parser.add_argument("--model-path", required=True, help="Path to the model file")
    optimize_parser.add_argument("--output-dir", default="optimization_results", help="Directory to save optimization results")
    optimize_parser.add_argument("--framework", choices=["pytorch", "tensorflow"], default="pytorch", help="Model framework")
    optimize_parser.add_argument("--techniques", default="quantization,pruning", help="Comma-separated list of optimization techniques")
    optimize_parser.add_argument("--dataset-path", help="Path to the validation dataset")
    optimize_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    
    # Create workflow command
    workflow_parser = subparsers.add_parser("create-workflow", help="Create a CI/CD workflow")
    workflow_parser.add_argument("--system", choices=["github_actions", "gitlab_ci", "jenkins", "azure_devops"], 
                               default="github_actions", help="CI/CD system to use")
    workflow_parser.add_argument("--output-dir", default=".github/workflows", help="Directory to save the workflow file")
    workflow_parser.add_argument("--optimization-script", default="neural_scope_cicd.py", 
                               help="Path to the optimization script")
    workflow_parser.add_argument("--test-script", help="Path to the test script")
    workflow_parser.add_argument("--workflow-name", default="model_optimization", help="Name of the workflow")
    workflow_parser.add_argument("--notify", action="store_true", help="Send notifications when the workflow completes")
    
    # Track command
    track_parser = subparsers.add_parser("track", help="Track results with MLflow")
    track_parser.add_argument("--model-name", required=True, help="Name of the model")
    track_parser.add_argument("--results-path", required=True, help="Path to the optimization results")
    track_parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    track_parser.add_argument("--experiment-name", default="neural-scope-optimization", help="MLflow experiment name")
    
    return parser.parse_args()

def optimize_model(args):
    """Optimize a model using Neural-Scope."""
    try:
        from advanced_analysis.analyzer import Analyzer
        from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor
        from advanced_analysis.performance import ModelPerformanceProfiler
    except ImportError:
        logger.error("Neural-Scope is not installed. Please install it with 'pip install neural-scope'.")
        sys.exit(1)
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load the model
    logger.info(f"Loading model from {args.model_path}")
    model = None
    
    try:
        if args.framework == "pytorch":
            import torch
            model = torch.load(args.model_path)
        elif args.framework == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Step 2: Analyze the model
    logger.info("Analyzing model...")
    analyzer = Analyzer()
    analysis_results = analyzer.analyze_model(model)
    
    # Save analysis results
    analysis_file = os.path.join(args.output_dir, "model_analysis.json")
    logger.info(f"Saving analysis results to {analysis_file}")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Step 3: Optimize the model
    logger.info("Optimizing model...")
    compressor = ModelCompressor()
    optimization_techniques = args.techniques.split(',')
    
    start_time = time.time()
    optimized_model, optimization_results = compressor.compress_model(
        model=model,
        techniques=optimization_techniques,
        return_stats=True
    )
    optimization_time = time.time() - start_time
    
    # Save optimized model
    optimized_model_file = os.path.join(args.output_dir, "optimized_model")
    if args.framework == "pytorch":
        import torch
        optimized_model_file += ".pt"
        torch.save(optimized_model, optimized_model_file)
    elif args.framework == "tensorflow":
        optimized_model_file += ".h5"
        optimized_model.save(optimized_model_file)
    
    logger.info(f"Saved optimized model to {optimized_model_file}")
    
    # Save optimization results
    optimization_results["optimization_time"] = optimization_time
    optimization_file = os.path.join(args.output_dir, "optimization_results.json")
    logger.info(f"Saving optimization results to {optimization_file}")
    with open(optimization_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Step 4: Validate the optimized model
    validation_results = {}
    
    if args.dataset_path:
        logger.info(f"Validating optimized model with dataset from {args.dataset_path}")
        
        # Load test data
        test_data = None
        try:
            if args.dataset_path.endswith('.csv'):
                import pandas as pd
                import numpy as np
                df = pd.read_csv(args.dataset_path)
                # Assuming the last column is the target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                test_data = (X, y)
            elif args.dataset_path.endswith('.npz'):
                import numpy as np
                data = np.load(args.dataset_path)
                test_data = (data['X'], data['y'])
            else:
                logger.error(f"Unsupported dataset format: {args.dataset_path}")
                logger.info("Skipping validation...")
                test_data = None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Skipping validation...")
            test_data = None
        
        if test_data is not None:
            # Convert data to the appropriate format
            if args.framework == "pytorch":
                import torch
                X_tensor = torch.tensor(test_data[0], dtype=torch.float32)
                y_tensor = torch.tensor(test_data[1], dtype=torch.float32)
                if len(y_tensor.shape) == 1:
                    y_tensor = y_tensor.unsqueeze(1)
                test_data = (X_tensor, y_tensor)
            
            # Profile the optimized model
            profiler = ModelPerformanceProfiler()
            performance_results = profiler.profile_model(
                model=optimized_model,
                input_data=test_data[0],
                batch_size=args.batch_size,
                framework=args.framework
            )
            
            # Evaluate model accuracy
            accuracy_results = {}
            
            if args.framework == "pytorch":
                import torch
                optimized_model.eval()
                with torch.no_grad():
                    outputs = optimized_model(test_data[0])
                    mse = torch.nn.functional.mse_loss(outputs, test_data[1]).item()
                    accuracy_results['mse'] = mse
            elif args.framework == "tensorflow":
                import tensorflow as tf
                loss = optimized_model.evaluate(test_data[0], test_data[1], verbose=0)
                accuracy_results['loss'] = loss
            
            # Combine results
            validation_results = {
                "performance": performance_results,
                "accuracy": accuracy_results
            }
            
            # Save validation results
            validation_file = os.path.join(args.output_dir, "validation_results.json")
            logger.info(f"Saving validation results to {validation_file}")
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
    
    # Step 5: Generate a summary report
    summary = {
        "model_path": args.model_path,
        "framework": args.framework,
        "optimization_techniques": optimization_techniques,
        "original_size_mb": optimization_results.get("original_size", "N/A"),
        "optimized_size_mb": optimization_results.get("optimized_size", "N/A"),
        "size_reduction_percentage": optimization_results.get("size_reduction_percentage", "N/A"),
        "optimization_time_seconds": optimization_time
    }
    
    if validation_results:
        summary.update({
            "inference_time_ms": validation_results.get("performance", {}).get("inference_time_ms", "N/A"),
            "throughput_samples_per_second": validation_results.get("performance", {}).get("throughput", "N/A"),
            "memory_usage_mb": validation_results.get("performance", {}).get("memory_usage_mb", "N/A")
        })
        
        if "mse" in validation_results.get("accuracy", {}):
            summary["mse"] = validation_results["accuracy"]["mse"]
        elif "loss" in validation_results.get("accuracy", {}):
            summary["loss"] = validation_results["accuracy"]["loss"]
    
    # Save summary report
    summary_file = os.path.join(args.output_dir, "optimization_summary.json")
    logger.info(f"Saving summary report to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\nOptimization Summary:")
    logger.info(f"Original model size: {summary.get('original_size_mb', 'N/A')} MB")
    logger.info(f"Optimized model size: {summary.get('optimized_size_mb', 'N/A')} MB")
    logger.info(f"Size reduction: {summary.get('size_reduction_percentage', 'N/A')}%")
    logger.info(f"Optimization time: {summary.get('optimization_time_seconds', 'N/A'):.2f} seconds")
    
    if "inference_time_ms" in summary:
        logger.info(f"Inference time: {summary.get('inference_time_ms', 'N/A')} ms")
        logger.info(f"Throughput: {summary.get('throughput_samples_per_second', 'N/A')} samples/second")
        logger.info(f"Memory usage: {summary.get('memory_usage_mb', 'N/A')} MB")
    
    if "mse" in summary:
        logger.info(f"Mean Squared Error: {summary['mse']}")
    elif "loss" in summary:
        logger.info(f"Loss: {summary['loss']}")
        
    return 0

def create_workflow(args):
    """Create a CI/CD workflow."""
    try:
        from advanced_analysis.mlops import CICDIntegrator
    except ImportError:
        logger.error("Neural-Scope is not installed. Please install it with 'pip install neural-scope'.")
        sys.exit(1)
        
    # Create the CI/CD integrator
    integrator = CICDIntegrator(system=args.system)
    
    # Create the workflow
    try:
        workflow_file = integrator.create_optimization_workflow(
            optimization_script=args.optimization_script,
            test_script=args.test_script,
            output_dir=args.output_dir,
            workflow_name=args.workflow_name,
            notify_on_completion=args.notify
        )
        
        logger.info(f"Created workflow file: {workflow_file}")
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        return 1
        
    return 0

def track_results(args):
    """Track results with MLflow."""
    try:
        from advanced_analysis.mlops import MLflowIntegrator
    except ImportError:
        logger.error("Neural-Scope is not installed. Please install it with 'pip install neural-scope'.")
        sys.exit(1)
        
    # Load results
    try:
        with open(args.results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return 1
        
    # Create MLflow integrator
    try:
        mlflow_integrator = MLflowIntegrator(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name
        )
        
        # Track results
        run_id = mlflow_integrator.track_model_analysis(
            model_name=args.model_name,
            analysis_results=results
        )
        
        logger.info(f"Results tracked in MLflow run: {run_id}")
    except Exception as e:
        logger.error(f"Error tracking results: {e}")
        return 1
        
    return 0

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "optimize":
        return optimize_model(args)
    elif args.command == "create-workflow":
        return create_workflow(args)
    elif args.command == "track":
        return track_results(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
