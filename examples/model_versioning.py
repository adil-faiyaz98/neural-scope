#!/usr/bin/env python
"""
Example of Model Versioning and Promotion

This script demonstrates how to use the Neural-Scope model versioning and promotion system.
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from versioning.model_registry import ModelRegistry
from versioning.version_manager import VersionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-versioning")

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

def create_and_save_model(version, output_dir):
    """Create and save a model."""
    # Create model
    model = SimpleModel()
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_v{version}.pt")
    torch.save(model, model_path)
    
    return model_path

def main():
    """Run the model versioning example."""
    parser = argparse.ArgumentParser(description="Model Versioning Example")
    parser.add_argument("--registry-dir", default="model_registry", help="Directory for the model registry")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI")
    args = parser.parse_args()
    
    # Create model registry and version manager
    registry = ModelRegistry(args.registry_dir)
    version_manager = VersionManager(args.registry_dir, args.mlflow_uri)
    
    # Create and register models
    logger.info("Creating and registering models...")
    
    # Version 1
    model_path_v1 = create_and_save_model("1", "temp_models")
    version1 = registry.register_model(
        model_path=model_path_v1,
        model_name="simple_model",
        version="1.0.0",
        metadata={
            "framework": "pytorch",
            "input_shape": [10],
            "output_shape": [1],
            "description": "Initial model version"
        }
    )
    
    # Register version with metrics
    version_manager.register_version(
        model_name="simple_model",
        version=version1,
        metrics={
            "accuracy": 0.85,
            "f1_score": 0.82,
            "latency_ms": 12.5
        },
        tags={
            "author": "neural-scope",
            "dataset": "example_data"
        }
    )
    
    # Version 2
    model_path_v2 = create_and_save_model("2", "temp_models")
    version2 = registry.register_model(
        model_path=model_path_v2,
        model_name="simple_model",
        version="1.1.0",
        metadata={
            "framework": "pytorch",
            "input_shape": [10],
            "output_shape": [1],
            "description": "Improved model with better accuracy"
        }
    )
    
    # Register version with metrics
    version_manager.register_version(
        model_name="simple_model",
        version=version2,
        metrics={
            "accuracy": 0.88,
            "f1_score": 0.86,
            "latency_ms": 13.2
        },
        tags={
            "author": "neural-scope",
            "dataset": "example_data"
        }
    )
    
    # Version 3
    model_path_v3 = create_and_save_model("3", "temp_models")
    version3 = registry.register_model(
        model_path=model_path_v3,
        model_name="simple_model",
        version="1.2.0",
        metadata={
            "framework": "pytorch",
            "input_shape": [10],
            "output_shape": [1],
            "description": "Optimized model with lower latency"
        }
    )
    
    # Register version with metrics
    version_manager.register_version(
        model_name="simple_model",
        version=version3,
        metrics={
            "accuracy": 0.87,
            "f1_score": 0.85,
            "latency_ms": 10.1
        },
        tags={
            "author": "neural-scope",
            "dataset": "example_data"
        }
    )
    
    # Promote models
    logger.info("Promoting models...")
    
    # Promote version 2 to staging
    registry.promote_model("simple_model", version2, "staging")
    version_manager.promote_version(
        model_name="simple_model",
        version=version2,
        stage="staging",
        reason="Better accuracy than version 1"
    )
    
    # Promote version 3 to production
    registry.promote_model("simple_model", version3, "production")
    version_manager.promote_version(
        model_name="simple_model",
        version=version3,
        stage="production",
        reason="Good accuracy with lower latency"
    )
    
    # Compare versions
    logger.info("Comparing versions...")
    comparison = version_manager.compare_versions("simple_model", version1, version3)
    
    # Print comparison
    print("\nVersion Comparison:")
    print(f"Model: {comparison['model_name']}")
    print(f"Version 1: {comparison['version1']} (Status: {comparison['v1_status']})")
    print(f"Version 2: {comparison['version2']} (Status: {comparison['v2_status']})")
    print("\nMetric Comparison:")
    
    for metric, values in comparison["metric_comparison"].items():
        print(f"  {metric}:")
        print(f"    Version 1: {values['v1_value']}")
        print(f"    Version 3: {values['v2_value']}")
        if values['diff'] is not None:
            print(f"    Difference: {values['diff']:.4f} ({values['pct_change']:.2f}%)")
        else:
            print("    Difference: N/A")
    
    # Get current production version
    production_version = version_manager.get_current_production_version("simple_model")
    production_model_path = registry.get_production_model("simple_model")
    
    print(f"\nCurrent Production Version: {production_version}")
    print(f"Production Model Path: {production_model_path}")
    
    # Get promotion history
    promotion_history = version_manager.get_promotion_history("simple_model")
    
    print("\nPromotion History:")
    for promotion in promotion_history:
        print(f"  Version {promotion['version']} promoted to {promotion['stage']} on {promotion['promoted_at']}")
        print(f"  Reason: {promotion['reason']}")
    
    # Clean up
    logger.info("Cleaning up temporary files...")
    for version in ["1", "2", "3"]:
        os.remove(os.path.join("temp_models", f"model_v{version}.pt"))
    os.rmdir("temp_models")
    
    logger.info("Model versioning example completed successfully!")

if __name__ == "__main__":
    main()
