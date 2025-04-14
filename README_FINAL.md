# Neural-Scope CI/CD Integration

Neural-Scope is a comprehensive Python library for analyzing and optimizing machine learning models. This repository provides a robust CI/CD integration that enables automated model optimization as part of ML workflows.

## Features

- **Automated Model Optimization**: Analyze, optimize, and validate ML models automatically
- **Pre-trained Model Analysis**: Fetch and analyze pre-trained models from popular repositories
- **Security Checks**: Identify potential security vulnerabilities in models
- **Metrics Verification**: Verify model metrics against known values
- **MLflow Integration**: Track optimization results and model performance over time
- **Comprehensive Reporting**: Generate detailed HTML and JSON reports
- **Configuration-based Analysis**: Customize analysis via YAML configuration files
- **CI/CD Integration**: Seamlessly integrate with GitHub Actions, GitLab CI, Jenkins, and Azure DevOps

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/adil-faiyaz98/neural-scope.git
cd neural-scope

# Install dependencies
pip install torch torchvision mlflow
```

### Analyze a Pre-trained Model

```bash
# Run the test implementation
python test_implementation.py --model resnet18 --output-dir results --security-check --verify-metrics
```

### Track Results with MLflow

```bash
# Start MLflow server
mlflow ui --port 5000

# Track results with MLflow
python test_mlflow_integration.py --results-dir results
```

### Customize Analysis with Configuration

```bash
# Create a custom configuration file
cp custom_model_config.yaml my_config.yaml

# Edit the configuration file to match your requirements
# ...

# Run analysis with the custom configuration
python run_neural_scope.py --model-path models/my_model.pt --config my_config.yaml
```

## CI/CD Integration

Neural-Scope can be integrated into CI/CD pipelines to automatically optimize models when they are updated.

### GitHub Actions

A GitHub Actions workflow is provided in `.github/workflows/neural_scope_cicd.yml`. This workflow:

1. Runs when changes are pushed to the repository
2. Fetches or loads a model
3. Analyzes and optimizes the model
4. Tracks results with MLflow
5. Validates optimization results
6. Deploys the optimized model (placeholder for actual deployment)

You can manually trigger the workflow from the GitHub Actions UI with custom parameters.

### Using in Your ML Workflow

To integrate Neural-Scope into your ML workflow:

1. **Add the workflow file** to your repository
2. **Configure the workflow** to match your requirements
3. **Add your models** to the repository
4. **Create a custom configuration file** for your models

## Example Workflow

Here's an example of how Neural-Scope fits into an ML workflow:

1. **Train a model** using your preferred framework
2. **Commit the model** to your repository
3. **Neural-Scope automatically analyzes** the model
4. **Optimization is applied** based on your configuration
5. **Results are tracked** in MLflow
6. **Optimized model is deployed** if it meets your criteria

## Commercial-Grade Integration

Neural-Scope CI/CD integration is designed for commercial-grade ML platforms:

- **Robust Security Checks**: Identify vulnerabilities and provide recommendations
- **Comprehensive Metrics**: Track performance, size, and inference time
- **Flexible Configuration**: Customize analysis for different models and requirements
- **Integration with MLOps Tools**: Work with MLflow, model registries, and deployment platforms
- **Detailed Reporting**: Generate reports for stakeholders and auditing

## Documentation

For more detailed documentation, see:

- [CI/CD Integration Guide](README_CICD.md): Comprehensive documentation for integrating Neural-Scope into CI/CD pipelines
- [MLflow Integration Guide](README_MLFLOW.md): Documentation for integrating with MLflow
- [Configuration Guide](README_CONFIG.md): Documentation for customizing analysis with configuration files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
