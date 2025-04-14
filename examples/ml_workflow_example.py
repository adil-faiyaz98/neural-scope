#!/usr/bin/env python
"""
Example ML Workflow with Neural-Scope CI/CD Integration

This script demonstrates a complete ML workflow that integrates Neural-Scope's
model optimization capabilities:

1. Train a simple model
2. Optimize the model using Neural-Scope
3. Evaluate the optimized model
4. Track results with MLflow

This example shows how Neural-Scope can be integrated into existing ML workflows
to automatically optimize models as part of the development process.
"""

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import time
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Example ML Workflow with Neural-Scope")
    parser.add_argument("--data-path", default="data/example_data.npz", help="Path to the dataset")
    parser.add_argument("--output-dir", default="workflow_results", help="Directory to save results")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], default="pytorch", help="ML framework to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--optimization-techniques", default="quantization,pruning", help="Optimization techniques to apply")
    parser.add_argument("--track-with-mlflow", action="store_true", help="Track results with MLflow")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    return parser.parse_args()

def create_dataset(args):
    """Create a synthetic dataset for training and testing."""
    print("Creating synthetic dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randn(200, 1)
    
    # Save dataset
    np.savez(args.data_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Dataset saved to {args.data_path}")
    
    return args.data_path

def train_model(args, data_path):
    """Train a simple model using the specified framework."""
    print(f"Training model using {args.framework}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"model.{'pt' if args.framework == 'pytorch' else 'h5'}")
    
    # Load dataset
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    
    # Train model based on framework
    if args.framework == "pytorch":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # Define a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 20)
                self.fc3 = nn.Linear(20, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Create model, loss function, and optimizer
        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Create data loader
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Train the model
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(dataloader):.4f}")
        
        # Save the model
        torch.save(model, model_path)
        
    elif args.framework == "tensorflow":
        import tensorflow as tf
        
        # Define a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                     loss='mse')
        
        # Train the model
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
        
        # Save the model
        model.save(model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def optimize_model(args, model_path, data_path):
    """Optimize the model using Neural-Scope."""
    print("Optimizing model using Neural-Scope...")
    
    # Create the command to run the Neural-Scope CI/CD runner
    cmd = [
        "python", "neural_scope_cicd.py", "optimize",
        "--model-path", model_path,
        "--output-dir", os.path.join(args.output_dir, "optimization"),
        "--framework", args.framework,
        "--techniques", args.optimization_techniques,
        "--dataset-path", data_path,
        "--batch-size", str(args.batch_size)
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Model optimization completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error optimizing model: {e}")
        sys.exit(1)
    
    # Get the path to the optimized model
    optimized_model_path = os.path.join(
        args.output_dir, 
        "optimization", 
        f"optimized_model.{'pt' if args.framework == 'pytorch' else 'h5'}"
    )
    
    return optimized_model_path

def evaluate_models(args, original_model_path, optimized_model_path, data_path):
    """Evaluate and compare the original and optimized models."""
    print("Evaluating models...")
    
    # Load dataset
    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    
    results = {}
    
    # Evaluate models based on framework
    if args.framework == "pytorch":
        import torch
        import torch.nn as nn
        import time
        
        # Load models
        original_model = torch.load(original_model_path)
        optimized_model = torch.load(optimized_model_path)
        
        # Convert data to tensors
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Evaluate original model
        original_model.eval()
        with torch.no_grad():
            # Measure inference time
            start_time = time.time()
            original_outputs = original_model(X_tensor)
            original_inference_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate MSE
            original_mse = nn.functional.mse_loss(original_outputs, y_tensor).item()
        
        # Evaluate optimized model
        optimized_model.eval()
        with torch.no_grad():
            # Measure inference time
            start_time = time.time()
            optimized_outputs = optimized_model(X_tensor)
            optimized_inference_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate MSE
            optimized_mse = nn.functional.mse_loss(optimized_outputs, y_tensor).item()
        
        # Store results
        results = {
            "original": {
                "mse": original_mse,
                "inference_time_ms": original_inference_time
            },
            "optimized": {
                "mse": optimized_mse,
                "inference_time_ms": optimized_inference_time
            },
            "comparison": {
                "mse_change_percentage": (optimized_mse - original_mse) / original_mse * 100,
                "inference_speedup": original_inference_time / optimized_inference_time
            }
        }
        
    elif args.framework == "tensorflow":
        import tensorflow as tf
        import time
        
        # Load models
        original_model = tf.keras.models.load_model(original_model_path)
        optimized_model = tf.keras.models.load_model(optimized_model_path)
        
        # Evaluate original model
        start_time = time.time()
        original_loss = original_model.evaluate(X_test, y_test, verbose=0)
        original_inference_time = (time.time() - start_time) * 1000  # ms
        
        # Evaluate optimized model
        start_time = time.time()
        optimized_loss = optimized_model.evaluate(X_test, y_test, verbose=0)
        optimized_inference_time = (time.time() - start_time) * 1000  # ms
        
        # Store results
        results = {
            "original": {
                "loss": original_loss,
                "inference_time_ms": original_inference_time
            },
            "optimized": {
                "loss": optimized_loss,
                "inference_time_ms": optimized_inference_time
            },
            "comparison": {
                "loss_change_percentage": (optimized_loss - original_loss) / original_loss * 100,
                "inference_speedup": original_inference_time / optimized_inference_time
            }
        }
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    if args.framework == "pytorch":
        print(f"Original model MSE: {results['original']['mse']:.4f}")
        print(f"Optimized model MSE: {results['optimized']['mse']:.4f}")
        print(f"MSE change: {results['comparison']['mse_change_percentage']:.2f}%")
    else:
        print(f"Original model loss: {results['original']['loss']:.4f}")
        print(f"Optimized model loss: {results['optimized']['loss']:.4f}")
        print(f"Loss change: {results['comparison']['loss_change_percentage']:.2f}%")
    
    print(f"Original inference time: {results['original']['inference_time_ms']:.2f} ms")
    print(f"Optimized inference time: {results['optimized']['inference_time_ms']:.2f} ms")
    print(f"Inference speedup: {results['comparison']['inference_speedup']:.2f}x")
    
    return results_path

def track_with_mlflow(args, results_path):
    """Track results with MLflow."""
    if not args.track_with_mlflow:
        return
    
    print("Tracking results with MLflow...")
    
    # Create the command to run the Neural-Scope CI/CD runner
    cmd = [
        "python", "neural_scope_cicd.py", "track",
        "--model-name", f"example-model-{args.framework}",
        "--results-path", results_path,
        "--tracking-uri", args.mlflow_tracking_uri,
        "--experiment-name", "neural-scope-example"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Results tracked in MLflow")
    except subprocess.CalledProcessError as e:
        print(f"Error tracking results: {e}")

def main():
    """Run the complete ML workflow."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create or load dataset
    if not os.path.exists(args.data_path):
        data_path = create_dataset(args)
    else:
        data_path = args.data_path
        print(f"Using existing dataset: {data_path}")
    
    # Step 2: Train model
    model_path = train_model(args, data_path)
    
    # Step 3: Optimize model using Neural-Scope
    optimized_model_path = optimize_model(args, model_path, data_path)
    
    # Step 4: Evaluate and compare models
    results_path = evaluate_models(args, model_path, optimized_model_path, data_path)
    
    # Step 5: Track results with MLflow
    track_with_mlflow(args, results_path)
    
    print("\nWorkflow completed successfully!")
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
