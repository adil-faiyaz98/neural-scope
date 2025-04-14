#!/usr/bin/env python
"""
Example script for optimizing a model using Neural-Scope.

This script demonstrates how to use Neural-Scope to optimize a machine learning model
by applying various compression techniques.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor

class SimpleModel(nn.Module):
    """A simple PyTorch model for demonstration purposes."""
    
    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    """Optimize a model using Neural-Scope."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimize a model using Neural-Scope")
    parser.add_argument(
        "--model-path",
        default="models/model.pt",
        help="Path to the model file"
    )
    parser.add_argument(
        "--output-path",
        default="models/optimized_model.pt",
        help="Path to save the optimized model"
    )
    parser.add_argument(
        "--techniques",
        default="quantization,pruning",
        help="Comma-separated list of optimization techniques to apply"
    )
    parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Model framework"
    )
    parser.add_argument(
        "--results-path",
        default="optimization_results.json",
        help="Path to save the optimization results"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load or create the model
    model = None
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        if args.framework == "pytorch":
            model = torch.load(args.model_path)
        elif args.framework == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(args.model_path)
    else:
        print(f"Model file {args.model_path} not found, creating a new model")
        if args.framework == "pytorch":
            model = SimpleModel()
        elif args.framework == "tensorflow":
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(50, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
    
    # Parse optimization techniques
    techniques = args.techniques.split(',')
    
    # Create compressor
    compressor = ModelCompressor()
    
    # Apply optimizations
    print(f"Applying optimization techniques: {techniques}")
    optimized_model, optimization_results = compressor.compress_model(
        model=model,
        techniques=techniques,
        return_stats=True
    )
    
    # Save optimized model
    print(f"Saving optimized model to {args.output_path}")
    if args.framework == "pytorch":
        torch.save(optimized_model, args.output_path)
    elif args.framework == "tensorflow":
        optimized_model.save(args.output_path)
    
    # Save optimization results
    print(f"Saving optimization results to {args.results_path}")
    with open(args.results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Print summary
    print("\nOptimization Summary:")
    print(f"Original model size: {optimization_results.get('original_size', 'N/A')} MB")
    print(f"Optimized model size: {optimization_results.get('optimized_size', 'N/A')} MB")
    print(f"Size reduction: {optimization_results.get('size_reduction_percentage', 'N/A')}%")
    print(f"Inference speedup: {optimization_results.get('inference_speedup', 'N/A')}x")
    
if __name__ == "__main__":
    main()
