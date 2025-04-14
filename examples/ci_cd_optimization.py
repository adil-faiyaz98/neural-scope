#!/usr/bin/env python
"""
Example script for integrating Neural-Scope model optimization into CI/CD pipelines.

This script demonstrates a complete workflow for model optimization that can be
integrated into CI/CD pipelines, including:
1. Model analysis
2. Optimization
3. Validation
4. Reporting

This script can be used as a standalone optimization script or as part of a CI/CD pipeline.
"""

import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor
from advanced_analysis.performance import ModelPerformanceProfiler

def main():
    """Run a complete model optimization workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run a complete model optimization workflow")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model file"
    )
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Directory to save optimization results"
    )
    parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Model framework"
    )
    parser.add_argument(
        "--optimization-techniques",
        default="quantization,pruning",
        help="Comma-separated list of optimization techniques to apply"
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the validation dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load the model
    print(f"Loading model from {args.model_path}")
    model = None
    if args.framework == "pytorch":
        model = torch.load(args.model_path)
    elif args.framework == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model_path)
    
    # Step 2: Analyze the model
    print("Analyzing model...")
    analyzer = Analyzer()
    analysis_results = analyzer.analyze_model(model)
    
    # Save analysis results
    analysis_file = os.path.join(args.output_dir, "model_analysis.json")
    print(f"Saving analysis results to {analysis_file}")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Step 3: Optimize the model
    print("Optimizing model...")
    compressor = ModelCompressor()
    optimization_techniques = args.optimization_techniques.split(',')
    
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
        optimized_model_file += ".pt"
        torch.save(optimized_model, optimized_model_file)
    elif args.framework == "tensorflow":
        optimized_model.save(optimized_model_file)
    
    print(f"Saved optimized model to {optimized_model_file}")
    
    # Save optimization results
    optimization_results["optimization_time"] = optimization_time
    optimization_file = os.path.join(args.output_dir, "optimization_results.json")
    print(f"Saving optimization results to {optimization_file}")
    with open(optimization_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Step 4: Validate the optimized model
    validation_results = {}
    
    if args.dataset_path:
        print(f"Validating optimized model with dataset from {args.dataset_path}")
        
        # Load test data
        test_data = None
        if args.dataset_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.dataset_path)
            # Assuming the last column is the target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            test_data = (X, y)
        elif args.dataset_path.endswith('.npz'):
            data = np.load(args.dataset_path)
            test_data = (data['X'], data['y'])
        
        # Convert data to the appropriate format
        if args.framework == "pytorch":
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
        print(f"Saving validation results to {validation_file}")
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
    print(f"Saving summary report to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nOptimization Summary:")
    print(f"Original model size: {summary.get('original_size_mb', 'N/A')} MB")
    print(f"Optimized model size: {summary.get('optimized_size_mb', 'N/A')} MB")
    print(f"Size reduction: {summary.get('size_reduction_percentage', 'N/A')}%")
    print(f"Optimization time: {summary.get('optimization_time_seconds', 'N/A'):.2f} seconds")
    
    if "inference_time_ms" in summary:
        print(f"Inference time: {summary.get('inference_time_ms', 'N/A')} ms")
        print(f"Throughput: {summary.get('throughput_samples_per_second', 'N/A')} samples/second")
        print(f"Memory usage: {summary.get('memory_usage_mb', 'N/A')} MB")
    
    if "mse" in summary:
        print(f"Mean Squared Error: {summary['mse']}")
    elif "loss" in summary:
        print(f"Loss: {summary['loss']}")
    
if __name__ == "__main__":
    main()
