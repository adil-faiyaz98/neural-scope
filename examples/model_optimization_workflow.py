#!/usr/bin/env python
"""
Example script for creating a model optimization workflow using Neural-Scope.

This script demonstrates how to use the Neural-Scope MLOps integration to create
a CI/CD workflow for model optimization.
"""

import os
import argparse
from advanced_analysis.mlops import CICDIntegrator

def main():
    """Create a model optimization workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create a model optimization workflow")
    parser.add_argument(
        "--system",
        choices=["github_actions", "gitlab_ci", "jenkins", "azure_devops"],
        default="github_actions",
        help="CI/CD system to use"
    )
    parser.add_argument(
        "--output-dir",
        default=".github/workflows",
        help="Directory to save the workflow file"
    )
    parser.add_argument(
        "--optimization-script",
        default="scripts/optimize_model.py",
        help="Path to the script that performs model optimization"
    )
    parser.add_argument(
        "--test-script",
        default="scripts/validate_model.py",
        help="Path to the script that validates the optimized model"
    )
    parser.add_argument(
        "--workflow-name",
        default="model_optimization",
        help="Name of the workflow"
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notifications when the workflow completes"
    )
    args = parser.parse_args()
    
    # Create the CI/CD integrator
    integrator = CICDIntegrator(system=args.system)
    
    # Create the workflow
    workflow_file = integrator.create_optimization_workflow(
        optimization_script=args.optimization_script,
        test_script=args.test_script,
        output_dir=args.output_dir,
        workflow_name=args.workflow_name,
        notify_on_completion=args.notify
    )
    
    print(f"Created workflow file: {workflow_file}")
    
if __name__ == "__main__":
    main()
