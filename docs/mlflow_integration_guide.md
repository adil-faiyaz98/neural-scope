# Neural-Scope MLflow Integration Guide

This guide provides detailed instructions for integrating Neural-Scope with MLflow to track model analysis, optimization, security, and robustness metrics.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting Up MLflow](#setting-up-mlflow)
4. [Tracking Model Analysis](#tracking-model-analysis)
5. [Tracking Model Optimization](#tracking-model-optimization)
6. [Tracking Security Analysis](#tracking-security-analysis)
7. [Tracking Robustness Testing](#tracking-robustness-testing)
8. [Using the Model Registry](#using-the-model-registry)
9. [Comparing Models](#comparing-models)
10. [CI/CD Integration](#cicd-integration)
11. [Troubleshooting](#troubleshooting)

## Introduction

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. Neural-Scope integrates with MLflow to provide comprehensive tracking of model analysis, optimization, security, and robustness metrics. This integration enables data scientists and ML engineers to compare different models, track improvements over time, and make informed decisions about model selection and deployment.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.7 or later
- Neural-Scope installed (`pip install neural-scope`)
- MLflow installed (`pip install mlflow`)
- Access to a model for analysis (PyTorch, TensorFlow, or other supported formats)

## Setting Up MLflow

### Starting the MLflow Server

To start the MLflow server:

```bash
# Start MLflow server on default port (5000)
mlflow ui

# Start MLflow server on a specific port
mlflow ui --port 8080

# Start MLflow server with a specific backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start MLflow server with a specific artifact store
mlflow ui --default-artifact-root ./mlflow-artifacts
```

### Configuring Neural-Scope to Use MLflow

You can configure Neural-Scope to use MLflow in two ways:

1. **Environment Variables**:

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Set experiment name
export MLFLOW_EXPERIMENT_NAME=neural-scope-analysis
```

2. **Configuration File**:

Create a file named `neural_scope_config.yaml`:

```yaml
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: neural-scope-analysis
  register_models: true
```

3. **Programmatic Configuration**:

```python
from neural_scope import NeuralScope

# Configure Neural-Scope with MLflow
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="neural-scope-analysis",
    mlflow_register_models=True
)
```

## Tracking Model Analysis

Neural-Scope automatically tracks the following model analysis metrics in MLflow:

- Number of parameters
- Number of layers
- Model architecture
- Memory usage (MB)
- Inference time (ms)

### Example: Analyzing a Model with MLflow Tracking

```python
from neural_scope import NeuralScope
import torch

# Load a model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="model-analysis"
)

# Analyze the model
analysis_results = neural_scope.analyze_model(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Results are automatically tracked in MLflow
print(f"Analysis results tracked in MLflow run: {neural_scope.mlflow_run_id}")
```

## Tracking Model Optimization

Neural-Scope tracks the following optimization metrics in MLflow:

- Original model size (MB)
- Optimized model size (MB)
- Size reduction percentage
- Inference speedup
- Optimization techniques used

### Example: Optimizing a Model with MLflow Tracking

```python
from neural_scope import NeuralScope
import torch

# Load a model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="model-optimization"
)

# Optimize the model
optimized_model, optimization_results = neural_scope.optimize_model(
    model=model,
    model_name="resnet18",
    framework="pytorch",
    techniques=["quantization", "pruning"]
)

# Results are automatically tracked in MLflow
print(f"Optimization results tracked in MLflow run: {neural_scope.mlflow_run_id}")
```

## Tracking Security Analysis

Neural-Scope tracks the following security metrics in MLflow:

- Security score
- Number of vulnerabilities by severity (critical, high, medium, low)
- Vulnerability types
- Security recommendations

### Example: Security Analysis with MLflow Tracking

```python
from neural_scope import NeuralScope
import torch

# Load a model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="security-analysis"
)

# Analyze model security
security_results = neural_scope.analyze_security(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Results are automatically tracked in MLflow
print(f"Security results tracked in MLflow run: {neural_scope.mlflow_run_id}")
```

## Tracking Robustness Testing

Neural-Scope tracks the following robustness metrics in MLflow:

- Robustness score
- Robustness level (very low, low, medium, high, very high)
- Attack results (original accuracy, adversarial accuracy, robustness)
- Attack parameters (epsilon, alpha, iterations)

### Example: Robustness Testing with MLflow Tracking

```python
from neural_scope import NeuralScope
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Load a model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="robustness-testing"
)

# Test adversarial robustness
robustness_results = neural_scope.test_robustness(
    model=model,
    model_name="resnet18",
    framework="pytorch",
    test_data=test_loader,
    attack_types=["fgsm", "pgd"]
)

# Results are automatically tracked in MLflow
print(f"Robustness results tracked in MLflow run: {neural_scope.mlflow_run_id}")
```

## Using the Model Registry

Neural-Scope can register models in the MLflow Model Registry, enabling version tracking and promotion.

### Registering a Model

```python
from neural_scope import NeuralScope
import torch

# Load a model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="model-registry",
    mlflow_register_models=True
)

# Analyze and register the model
analysis_results = neural_scope.analyze_model(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Get the registered model version
model_version = neural_scope.mlflow_model_version
print(f"Model registered as: resnet18 version {model_version}")
```

### Promoting a Model

```python
from neural_scope import NeuralScope
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create MLflow client
client = MlflowClient()

# Promote model to staging
client.transition_model_version_stage(
    name="resnet18",
    version=1,
    stage="Staging"
)

# Promote model to production
client.transition_model_version_stage(
    name="resnet18",
    version=1,
    stage="Production"
)
```

## Comparing Models

MLflow provides a powerful UI for comparing different models and versions.

### Steps to Compare Models

1. Open the MLflow UI (e.g., http://localhost:5000)
2. Navigate to the experiment containing your runs
3. Select multiple runs by checking the boxes next to them
4. Click "Compare" to see a side-by-side comparison of metrics and parameters

### Programmatic Comparison

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create MLflow client
client = MlflowClient()

# Get runs for comparison
run1 = client.get_run("run_id_1")
run2 = client.get_run("run_id_2")

# Compare metrics
metrics1 = run1.data.metrics
metrics2 = run2.data.metrics

# Print comparison
print("Metric Comparison:")
for metric in set(metrics1.keys()) | set(metrics2.keys()):
    value1 = metrics1.get(metric, "N/A")
    value2 = metrics2.get(metric, "N/A")
    print(f"{metric}: {value1} vs {value2}")
```

## CI/CD Integration

Neural-Scope can be integrated into CI/CD pipelines to automatically track model metrics and register models.

### GitHub Actions Example

```yaml
name: Neural-Scope Analysis

on:
  push:
    paths:
      - 'models/**'

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install neural-scope mlflow
      - name: Start MLflow server
        run: |
          mlflow ui --port 5000 &
          sleep 5
      - name: Analyze model
        run: |
          python -c "
          from neural_scope import NeuralScope
          import torch
          
          # Load model
          model = torch.load('models/my_model.pt')
          
          # Create Neural-Scope instance
          neural_scope = NeuralScope(
              mlflow_tracking_uri='http://localhost:5000',
              mlflow_experiment_name='ci-cd-analysis'
          )
          
          # Analyze model
          results = neural_scope.analyze_model(
              model=model,
              model_name='my_model',
              framework='pytorch'
          )
          
          # Print results
          print(f'Analysis results: {results}')
          print(f'MLflow run ID: {neural_scope.mlflow_run_id}')
          "
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install neural-scope mlflow'
                sh 'mlflow ui --port 5000 &'
                sh 'sleep 5'
            }
        }
        stage('Analyze') {
            steps {
                sh '''
                python -c "
                from neural_scope import NeuralScope
                import torch
                
                # Load model
                model = torch.load('models/my_model.pt')
                
                # Create Neural-Scope instance
                neural_scope = NeuralScope(
                    mlflow_tracking_uri='http://localhost:5000',
                    mlflow_experiment_name='jenkins-analysis'
                )
                
                # Analyze model
                results = neural_scope.analyze_model(
                    model=model,
                    model_name='my_model',
                    framework='pytorch'
                )
                
                # Print results
                print(f'Analysis results: {results}')
                print(f'MLflow run ID: {neural_scope.mlflow_run_id}')
                "
                '''
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **MLflow server not running**:
   - Error: `Failed to connect to MLflow server`
   - Solution: Ensure the MLflow server is running and accessible at the specified URI

2. **Model registration fails**:
   - Error: `Failed to register model in MLflow Model Registry`
   - Solution: Check that you have the necessary permissions and that the model name is valid

3. **Missing metrics in MLflow**:
   - Issue: Some metrics are not showing up in the MLflow UI
   - Solution: Ensure that the metrics are being properly logged and that the names are consistent

4. **Model loading errors**:
   - Error: `Error loading model from MLflow`
   - Solution: Ensure that the model is saved in a compatible format and that all dependencies are installed

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the [Neural-Scope documentation](https://neural-scope.readthedocs.io/)
2. Visit the [MLflow documentation](https://mlflow.org/docs/latest/index.html)
3. Open an issue on the [Neural-Scope GitHub repository](https://github.com/adil-faiyaz98/neural-scope/issues)

## Conclusion

Integrating Neural-Scope with MLflow provides a powerful platform for tracking model metrics, comparing different models, and managing the model lifecycle. By following this guide, you can leverage the full potential of both tools to improve your ML workflow and make more informed decisions about model selection and deployment.
