# Neural-Scope CI/CD Integration Guide

This guide provides comprehensive documentation for integrating Neural-Scope's analysis and optimization capabilities into CI/CD pipelines.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [CI/CD Integration](#cicd-integration)
  - [GitHub Actions](#github-actions)
  - [GitLab CI](#gitlab-ci)
  - [Jenkins](#jenkins)
  - [Azure DevOps](#azure-devops)
- [MLOps Integration](#mlops-integration)
  - [MLflow Integration](#mlflow-integration)
  - [Kubeflow Integration](#kubeflow-integration)
- [Example Workflows](#example-workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Neural-Scope provides tools for integrating its analysis and optimization capabilities into CI/CD pipelines, enabling automated model optimization and validation as part of your ML workflow. This integration allows you to:

1. Automatically analyze models for performance, memory usage, and complexity
2. Apply optimization techniques like quantization and pruning
3. Validate optimized models against test datasets
4. Track optimization results and model performance over time

## Getting Started

To use Neural-Scope's CI/CD integration capabilities, install the package with the MLOps dependencies:

```bash
pip install neural-scope[all]
```

Or, if you have the source, install it with the MLOps dependencies:

```bash
pip install -e ".[all]"
```

The package includes a comprehensive CI/CD runner script (`neural_scope_cicd.py`) that provides a command-line interface for model optimization, workflow creation, and result tracking:

## CI/CD Integration

Neural-Scope provides a `CICDIntegrator` class that makes it easy to create CI/CD workflows for model optimization:

```python
from advanced_analysis.mlops import CICDIntegrator

# Create a CI/CD integrator for GitHub Actions
integrator = CICDIntegrator(system="github_actions")

# Create a model optimization workflow
workflow_file = integrator.create_optimization_workflow(
    optimization_script="scripts/optimize_model.py",
    test_script="scripts/validate_model.py",
    output_dir=".github/workflows",
    workflow_name="model_optimization",
    trigger_on=["push", "pull_request"],
    notify_on_completion=True
)
```

### GitHub Actions

To integrate Neural-Scope with GitHub Actions, create a workflow file in the `.github/workflows` directory:

```python
from advanced_analysis.mlops import CICDIntegrator

integrator = CICDIntegrator(system="github_actions")
workflow_file = integrator.create_optimization_workflow(
    optimization_script="scripts/optimize_model.py",
    test_script="scripts/validate_model.py",
    output_dir=".github/workflows",
    workflow_name="model_optimization"
)
```

This will create a workflow file with the following steps:

1. Analyze the model
2. Optimize the model
3. Validate the optimized model
4. Generate a performance report

### GitLab CI

To integrate Neural-Scope with GitLab CI, create a `.gitlab-ci.yml` file:

```python
from advanced_analysis.mlops import CICDIntegrator

integrator = CICDIntegrator(system="gitlab_ci")
workflow_file = integrator.create_optimization_workflow(
    optimization_script="scripts/optimize_model.py",
    test_script="scripts/validate_model.py",
    output_dir=".",
    workflow_name="model_optimization"
)
```

### Jenkins

To integrate Neural-Scope with Jenkins, create a `Jenkinsfile`:

```python
from advanced_analysis.mlops import CICDIntegrator

integrator = CICDIntegrator(system="jenkins")
workflow_file = integrator.create_optimization_workflow(
    optimization_script="scripts/optimize_model.py",
    test_script="scripts/validate_model.py",
    output_dir=".",
    workflow_name="model_optimization"
)
```

### Azure DevOps

To integrate Neural-Scope with Azure DevOps, create an `azure-pipelines.yml` file:

```python
from advanced_analysis.mlops import CICDIntegrator

integrator = CICDIntegrator(system="azure_devops")
workflow_file = integrator.create_optimization_workflow(
    optimization_script="scripts/optimize_model.py",
    test_script="scripts/validate_model.py",
    output_dir=".",
    workflow_name="model_optimization"
)
```

## MLOps Integration

Neural-Scope also provides integration with MLOps platforms like MLflow and Kubeflow.

### MLflow Integration

To integrate Neural-Scope with MLflow for experiment tracking and model registry:

```python
from advanced_analysis.mlops import MLflowIntegrator

# Create an MLflow integrator
mlflow_integrator = MLflowIntegrator(
    tracking_uri="http://localhost:5000",
    experiment_name="neural-scope-optimization"
)

# Track model analysis results
run_id = mlflow_integrator.track_model_analysis(
    model_name="my_model",
    analysis_results=analysis_results,
    metrics={"accuracy": 0.95, "inference_time_ms": 45}
)

# Register optimized model
model_version = mlflow_integrator.register_optimized_model(
    original_model=model,
    optimized_model=optimized_model,
    optimization_history=optimization_results,
    model_name="my_model_optimized"
)
```

### Kubeflow Integration

To integrate Neural-Scope with Kubeflow Pipelines for orchestrating ML workflows:

```python
from advanced_analysis.mlops import KubeflowIntegrator

# Create a Kubeflow integrator
kubeflow_integrator = KubeflowIntegrator()

# Create a Neural-Scope pipeline
pipeline = kubeflow_integrator.create_neural_scope_pipeline(
    model_path="models/model.pt",
    output_dir="pipeline_outputs",
    framework="pytorch",
    optimization_techniques=["quantization", "pruning"],
    validation_dataset="data/test_data.csv"
)

# Run the pipeline
kubeflow_integrator.run_pipeline(
    experiment_name="neural-scope",
    run_name="model-optimization",
    pipeline_file="neural_scope_pipeline.yaml"
)
```

## Example Workflows

Neural-Scope provides a comprehensive CI/CD runner script (`neural_scope_cicd.py`) that can be used in various ways:

### Optimizing a Model

```bash
# Optimize a PyTorch model
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --output-dir results \
    --framework pytorch \
    --techniques quantization,pruning \
    --dataset-path data/test_data.csv

# Optimize a TensorFlow model
python neural_scope_cicd.py optimize \
    --model-path models/model.h5 \
    --output-dir results \
    --framework tensorflow \
    --techniques quantization,pruning
```

### Creating a CI/CD Workflow

```bash
# Create a GitHub Actions workflow
python neural_scope_cicd.py create-workflow \
    --system github_actions \
    --output-dir .github/workflows \
    --workflow-name model_optimization

# Create a GitLab CI workflow
python neural_scope_cicd.py create-workflow \
    --system gitlab_ci \
    --output-dir .
```

### Tracking Results with MLflow

```bash
# Track optimization results
python neural_scope_cicd.py track \
    --model-name my_model \
    --results-path results/optimization_results.json \
    --tracking-uri http://localhost:5000 \
    --experiment-name neural-scope-optimization
```

## Best Practices

1. **Automate the entire workflow**: Analyze, optimize, validate, and deploy models automatically.
2. **Track optimization results**: Use MLflow or a similar platform to track optimization results over time.
3. **Validate optimized models**: Always validate optimized models against test datasets to ensure they still perform well.
4. **Set performance thresholds**: Define performance thresholds for optimized models to ensure they meet your requirements.
5. **Use version control for models**: Store models in version control or a model registry to track changes.

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   ```
   Error: Required dependency 'torch' is not installed
   ```
   Solution: Install the required dependency
   ```bash
   pip install torch
   ```

2. **File not found**:
   ```
   Error: File not found: model.pt
   ```
   Solution: Check the file path and ensure the file exists

3. **Unsupported framework**:
   ```
   Error: Unsupported framework: keras
   ```
   Solution: Use one of the supported frameworks (pytorch, tensorflow, sklearn)

4. **CI/CD integration issues**:
   ```
   Error: Failed to create workflow file
   ```
   Solution: Check that you have write permissions to the output directory
