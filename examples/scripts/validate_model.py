#!/usr/bin/env python
"""
Example script for validating an optimized model using Neural-Scope.

This script demonstrates how to use Neural-Scope to validate an optimized model
by measuring its performance and accuracy.
"""

import os
import argparse
import json
import numpy as np
import torch
from advanced_analysis.performance import ModelPerformanceProfiler

def main():
    """Validate an optimized model using Neural-Scope."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Validate an optimized model using Neural-Scope")
    parser.add_argument(
        "--model-path",
        default="models/optimized_model.pt",
        help="Path to the optimized model file"
    )
    parser.add_argument(
        "--dataset-path",
        default="data/test_data.csv",
        help="Path to the test dataset"
    )
    parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Model framework"
    )
    parser.add_argument(
        "--results-path",
        default="validation_results.json",
        help="Path to save the validation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    args = parser.parse_args()
    
    # Load the model
    model = None
    if args.framework == "pytorch":
        model = torch.load(args.model_path)
        model.eval()
    elif args.framework == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model_path)
    
    # Load or generate test data
    test_data = None
    if os.path.exists(args.dataset_path):
        print(f"Loading test data from {args.dataset_path}")
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
    else:
        print(f"Test data file {args.dataset_path} not found, generating synthetic data")
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000, 1)
        test_data = (X, y)
    
    # Convert data to the appropriate format
    if args.framework == "pytorch":
        X_tensor = torch.tensor(test_data[0], dtype=torch.float32)
        y_tensor = torch.tensor(test_data[1], dtype=torch.float32)
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)
        test_data = (X_tensor, y_tensor)
    
    # Create profiler
    profiler = ModelPerformanceProfiler()
    
    # Profile the model
    print("Profiling model performance...")
    performance_results = profiler.profile_model(
        model=model,
        input_data=test_data[0],
        batch_size=args.batch_size,
        framework=args.framework
    )
    
    # Evaluate model accuracy
    print("Evaluating model accuracy...")
    accuracy_results = {}
    
    if args.framework == "pytorch":
        model.eval()
        with torch.no_grad():
            outputs = model(test_data[0])
            mse = torch.nn.functional.mse_loss(outputs, test_data[1]).item()
            accuracy_results['mse'] = mse
    elif args.framework == "tensorflow":
        import tensorflow as tf
        loss = model.evaluate(test_data[0], test_data[1], verbose=0)
        accuracy_results['loss'] = loss
    
    # Combine results
    validation_results = {
        "performance": performance_results,
        "accuracy": accuracy_results
    }
    
    # Save validation results
    print(f"Saving validation results to {args.results_path}")
    with open(args.results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Inference time: {performance_results.get('inference_time_ms', 'N/A')} ms")
    print(f"Throughput: {performance_results.get('throughput', 'N/A')} samples/second")
    print(f"Memory usage: {performance_results.get('memory_usage_mb', 'N/A')} MB")
    
    if 'mse' in accuracy_results:
        print(f"Mean Squared Error: {accuracy_results['mse']}")
    elif 'loss' in accuracy_results:
        print(f"Loss: {accuracy_results['loss']}")
    
if __name__ == "__main__":
    main()
