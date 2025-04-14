#!/usr/bin/env python
"""
Example script for integrating Neural-Scope with MLflow.

This script demonstrates how to use the Neural-Scope MLOps integration to track
model analysis and optimization results in MLflow.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from advanced_analysis.mlops import MLflowIntegrator
from advanced_analysis.analyzer import Analyzer
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
    """Integrate Neural-Scope with MLflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Integrate Neural-Scope with MLflow")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name",
        default="neural-scope-optimization",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-name",
        default="simple-model",
        help="Name of the model"
    )
    args = parser.parse_args()
    
    # Create a simple model
    model = SimpleModel()
    
    # Create sample data
    np.random.seed(42)
    X = torch.tensor(np.random.randn(100, 10), dtype=torch.float32)
    y = torch.tensor(np.random.randn(100, 1), dtype=torch.float32)
    
    # Create analyzer
    analyzer = Analyzer()
    
    # Analyze the model
    print("Analyzing model...")
    analysis_results = analyzer.analyze_model(model)
    
    # Create MLflow integrator
    mlflow_integrator = MLflowIntegrator(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name
    )
    
    # Track model analysis results
    print("Tracking model analysis results in MLflow...")
    run_id = mlflow_integrator.track_model_analysis(
        model_name=args.model_name,
        analysis_results=analysis_results,
        metrics={
            "parameters": sum(p.numel() for p in model.parameters()),
            "layers": len(list(model.modules()))
        },
        tags={
            "framework": "pytorch",
            "model_type": "simple_nn"
        }
    )
    
    print(f"Analysis results tracked in MLflow run: {run_id}")
    
    # Create compressor
    compressor = ModelCompressor()
    
    # Apply optimizations
    print("Optimizing model...")
    optimized_model, optimization_results = compressor.compress_model(
        model=model,
        techniques=["quantization", "pruning"],
        return_stats=True
    )
    
    # Register optimized model
    print("Registering optimized model in MLflow...")
    model_version = mlflow_integrator.register_optimized_model(
        original_model=model,
        optimized_model=optimized_model,
        optimization_history=optimization_results,
        model_name=f"{args.model_name}-optimized",
        framework="pytorch",
        tags={
            "optimization_techniques": "quantization,pruning",
            "size_reduction": f"{optimization_results.get('size_reduction_percentage', 0)}%"
        }
    )
    
    print(f"Optimized model registered in MLflow with version: {model_version}")
    
if __name__ == "__main__":
    main()
