# Neural-Scope

[![PyPI version](https://badge.fury.io/py/neural-scope.svg)](https://badge.fury.io/py/neural-scope)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

Neural-Scope is a comprehensive tool for analyzing and optimizing machine learning models. It provides a robust CI/CD integration that enables automated model optimization as part of ML workflows.

## Features

- **Automated Model Optimization**: Analyze, optimize, and validate ML models automatically
- **CI/CD Integration**: Seamlessly integrate with GitHub Actions, GitLab CI, Jenkins, and Azure DevOps
- **Pre-trained Model Support**: Analyze models from PyTorch Hub, TensorFlow Hub, Hugging Face, SageMaker, and more
- **Advanced Security Analysis**: Detect vulnerabilities with sophisticated detection algorithms
- **Adversarial Robustness**: Test model resilience against FGSM, PGD, and other attacks
- **Model Versioning**: Track model versions and promote models through staging and production
- **MLflow Integration**: Track optimization results and model performance over time

## Installation

```bash
# Basic installation
pip install neural-scope

# With PyTorch support
pip install neural-scope[pytorch]

# With TensorFlow support
pip install neural-scope[tensorflow]

# With MLflow integration
pip install neural-scope[mlflow]

# With security and robustness testing
pip install neural-scope[security]

# With all dependencies
pip install neural-scope[all]
```

## Quick Start

### Analyze a Model

```python
from neural_scope import NeuralScope

# Load your model (PyTorch example)
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance
neural_scope = NeuralScope()

# Analyze the model
results = neural_scope.analyze_model(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Print results
print(f"Parameters: {results['parameters']}")
print(f"Layers: {results['layers']}")
print(f"Memory usage: {results['memory_usage_mb']} MB")
print(f"Inference time: {results['inference_time_ms']} ms")
```

### Optimize a Model

```python
from neural_scope import NeuralScope
import torch

# Load your model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance
neural_scope = NeuralScope()

# Optimize the model
optimized_model, results = neural_scope.optimize_model(
    model=model,
    model_name="resnet18",
    framework="pytorch",
    techniques=["quantization", "pruning"]
)

# Print results
print(f"Original size: {results['original_size']} MB")
print(f"Optimized size: {results['optimized_size']} MB")
print(f"Size reduction: {results['size_reduction_percentage']}%")
print(f"Inference speedup: {results['inference_speedup']}x")
```

### Security Analysis

```python
from neural_scope import NeuralScope
import torch

# Load your model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance
neural_scope = NeuralScope()

# Analyze security
security_results = neural_scope.analyze_security(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Print results
print(f"Security score: {security_results['security_score']}/100")
print(f"Vulnerabilities: {security_results['total_vulnerabilities']}")
for severity in ['critical', 'high', 'medium', 'low']:
    vulns = security_results['vulnerabilities'][severity]
    if vulns:
        print(f"{severity.capitalize()} severity: {len(vulns)}")
```

### MLflow Integration

```python
from neural_scope import NeuralScope
import torch

# Load your model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Create Neural-Scope instance with MLflow tracking
neural_scope = NeuralScope(
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="model-analysis"
)

# Analyze the model
results = neural_scope.analyze_model(
    model=model,
    model_name="resnet18",
    framework="pytorch"
)

# Results are automatically tracked in MLflow
print(f"Results tracked in MLflow run: {neural_scope.mlflow_run_id}")
```

### Command Line Interface

```bash
# Analyze a model
neural-scope analyze \
    --model-path models/model.pt \
    --framework pytorch \
    --output-dir results

# Optimize a model
neural-scope optimize \
    --model-path models/model.pt \
    --framework pytorch \
    --output-dir results \
    --techniques quantization,pruning

# Test security
neural-scope security \
    --model-path models/model.pt \
    --framework pytorch \
    --output-dir results

# Test robustness
neural-scope robustness \
    --model-path models/model.pt \
    --framework pytorch \
    --output-dir results \
    --attack-types fgsm,pgd
```

## Documentation

For more detailed documentation, visit [https://neural-scope.readthedocs.io/](https://neural-scope.readthedocs.io/)

## Why Neural-Scope?

Neural-Scope addresses critical challenges in ML model deployment:

1. **Performance Optimization**: Reduce model size and improve inference speed
2. **Security Analysis**: Identify vulnerabilities with sophisticated detection algorithms
3. **Adversarial Robustness**: Test model resilience against various attack types
4. **CI/CD Integration**: Automate optimization as part of your ML workflow
5. **Model Versioning**: Track model versions and promote them through stages
6. **MLflow Integration**: Track experiments and compare model performance

## Use Cases

- **ML Engineers**: Optimize models for deployment and test robustness
- **DevOps Engineers**: Integrate model optimization into CI/CD pipelines
- **Security Teams**: Assess model vulnerabilities and adversarial robustness
- **Data Scientists**: Analyze model performance and track experiments
- **MLOps Teams**: Implement model versioning and promotion workflows

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
