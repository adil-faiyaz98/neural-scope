# Neural-Scope CI/CD Integration

Neural-Scope is a comprehensive Python library for analyzing and optimizing machine learning models, code, and data. This repository provides a robust CI/CD integration for Neural-Scope, enabling automated model optimization as part of ML workflows.

## Features

- **Automated Model Optimization**: Analyze, optimize, and validate ML models automatically
- **CI/CD Integration**: Seamlessly integrate with GitHub Actions, GitLab CI, Jenkins, and Azure DevOps
- **MLflow Tracking**: Track optimization results and model performance over time
- **Pre-trained Model Analysis**: Fetch and analyze pre-trained models from popular repositories
- **Security Checks**: Identify potential security vulnerabilities in models
- **Metrics Verification**: Verify model metrics against known values
- **Comprehensive Reporting**: Generate detailed HTML and JSON reports
- **Configuration-based Analysis**: Customize analysis via YAML configuration files

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/adil-faiyaz98/neural-scope.git
cd neural-scope

# Install dependencies
pip install -e .[all]
```

### Setup

Run the setup script to configure the CI/CD integration:

```bash
python setup_cicd.py --ci-system github_actions --create-example
```

Set up MLflow tracking (optional):

```bash
python setup_mlflow.py --create-experiments
```

### Basic Usage

Optimize a model using the Neural-Scope CI/CD runner:

```bash
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --output-dir results \
    --framework pytorch \
    --techniques quantization,pruning \
    --dataset-path data/test_data.csv
```

Analyze a pre-trained model:

```bash
python fetch_and_analyze.py \
    --model resnet18 \
    --source pytorch_hub \
    --output-dir reports \
    --security-check \
    --verify-metrics
```

## Configuration-based Analysis

Neural-Scope supports configuration-based analysis via YAML files:

```bash
# Create a configuration file
cp model_analysis_config.yaml my_config.yaml

# Edit the configuration file
# ...

# Run analysis with the configuration file
python neural_scope_cicd.py optimize --config my_config.yaml
```

## CI/CD Integration

Neural-Scope can be integrated into CI/CD pipelines to automatically optimize models when they are updated:

### GitHub Actions

A GitHub Actions workflow is automatically created in `.github/workflows/model_optimization.yml`. This workflow will:

1. Run when changes are pushed to the `models` directory
2. Optimize the model using Neural-Scope
3. Validate the optimized model
4. Track results with MLflow (if configured)

You can also manually trigger the workflow from the GitHub Actions UI.

### Other CI/CD Systems

Neural-Scope also supports GitLab CI, Jenkins, and Azure DevOps. Run the setup script with the appropriate `--ci-system` option to create the workflow for your CI/CD system.

## MLflow Integration

Neural-Scope integrates with MLflow for experiment tracking and model registry:

```bash
# Start the MLflow tracking server
python setup_mlflow.py --start-server

# Track results with MLflow
python neural_scope_cicd.py track \
    --model-name my_model \
    --results-path results/optimization_results.json \
    --tracking-uri http://localhost:5000 \
    --experiment-name neural-scope-optimization
```

## Pre-trained Model Analysis

Neural-Scope can fetch and analyze pre-trained models from popular repositories:

```bash
# Analyze a pre-trained model from PyTorch Hub
python fetch_and_analyze.py \
    --model resnet18 \
    --source pytorch_hub \
    --output-dir reports \
    --mlflow

# Analyze a pre-trained model from TensorFlow Hub
python fetch_and_analyze.py \
    --model efficientnet/b0 \
    --source tensorflow_hub \
    --output-dir reports

# Analyze a pre-trained model from Hugging Face
python fetch_and_analyze.py \
    --model bert-base-uncased \
    --source huggingface \
    --output-dir reports
```

## Security Checks

Neural-Scope can perform security checks on models:

```bash
python fetch_and_analyze.py \
    --model resnet18 \
    --source pytorch_hub \
    --output-dir reports \
    --security-check
```

The security check will identify:
- Potential vulnerabilities in the model architecture
- Warnings about quantization and other optimization techniques
- Recommendations for improving model security

## Metrics Verification

Neural-Scope can verify model metrics against known values:

```bash
python fetch_and_analyze.py \
    --model resnet18 \
    --source pytorch_hub \
    --output-dir reports \
    --verify-metrics
```

The metrics verification will:
- Compare parameter count, model size, and inference time to expected values
- Identify discrepancies between expected and actual metrics
- Provide insights into model performance

## Comprehensive Reporting

Neural-Scope generates comprehensive reports in HTML and JSON formats:

```bash
python fetch_and_analyze.py \
    --model resnet18 \
    --source pytorch_hub \
    --output-dir reports \
    --security-check \
    --verify-metrics
```

The report includes:
- Model overview
- Analysis results
- Optimization results
- Performance summary
- Security check results
- Metrics verification
- Visualizations (if enabled)

## Example Workflow

The repository includes an example ML workflow that demonstrates how to integrate Neural-Scope into your ML development process:

```bash
python examples/ml_workflow_example.py \
    --output-dir example_results \
    --framework pytorch \
    --epochs 10 \
    --track-with-mlflow
```

This will:
1. Train a simple model
2. Optimize the model using Neural-Scope
3. Evaluate the optimized model
4. Track results with MLflow

## Documentation

For more detailed documentation, see:

- [CI/CD Integration Guide](README_CICD.md): Comprehensive documentation for integrating Neural-Scope into CI/CD pipelines
- [CI/CD Runner Guide](README_RUNNER.md): Documentation for using the Neural-Scope CI/CD runner
- [MLflow Integration Guide](README_MLFLOW.md): Documentation for integrating with MLflow
- [Pre-trained Model Analysis Guide](README_PRETRAINED.md): Documentation for analyzing pre-trained models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
