#!/usr/bin/env python
"""
Example script for integrating Neural-Scope with Kubeflow Pipelines.

This script demonstrates how to use the Neural-Scope MLOps integration to create
a Kubeflow Pipeline for model analysis and optimization.
"""

import os
import argparse
from advanced_analysis.mlops import KubeflowIntegrator

def main():
    """Integrate Neural-Scope with Kubeflow Pipelines."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Integrate Neural-Scope with Kubeflow Pipelines")
    parser.add_argument(
        "--model-path",
        default="models/model.pt",
        help="Path to the model file"
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_outputs",
        help="Directory to save pipeline outputs"
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
        "--validation-dataset",
        default="data/test_data.csv",
        help="Path to the validation dataset"
    )
    parser.add_argument(
        "--pipeline-file",
        default="neural_scope_pipeline.yaml",
        help="Path to save the pipeline file"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the pipeline after creating it"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create Kubeflow integrator
    integrator = KubeflowIntegrator()
    
    # Parse optimization techniques
    optimization_techniques = args.optimization_techniques.split(',')
    
    # Create the pipeline
    print("Creating Kubeflow pipeline...")
    pipeline = integrator.create_neural_scope_pipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        framework=args.framework,
        optimization_techniques=optimization_techniques,
        validation_dataset=args.validation_dataset
    )
    
    print(f"Pipeline created and saved to {args.pipeline_file}")
    
    # Run the pipeline if requested
    if args.run:
        print("Running pipeline...")
        integrator.run_pipeline(
            experiment_name="neural-scope",
            run_name="model-optimization",
            pipeline_file=args.pipeline_file
        )
        print("Pipeline run started")
    
if __name__ == "__main__":
    main()
