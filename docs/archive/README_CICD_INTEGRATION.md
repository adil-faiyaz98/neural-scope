# Neural-Scope CI/CD Integration

Neural-Scope is a comprehensive Python library for analyzing and optimizing machine learning models, code, and data. This repository provides a CI/CD integration for Neural-Scope, enabling automated model optimization as part of ML workflows.

## Features

- **Automated Model Optimization**: Analyze, optimize, and validate ML models automatically
- **CI/CD Integration**: Seamlessly integrate with GitHub Actions, GitLab CI, Jenkins, and Azure DevOps
- **MLflow Tracking**: Track optimization results and model performance over time
- **Comprehensive Analysis**: Analyze model architecture, performance, memory usage, and complexity
- **Multiple Optimization Techniques**: Apply quantization, pruning, and other optimization techniques
- **Framework Support**: Works with PyTorch, TensorFlow, and other ML frameworks

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

This will:
1. Install required dependencies
2. Create necessary directories
3. Set up CI/CD workflows
4. Create an example model and dataset

### Usage

Optimize a model using the Neural-Scope CI/CD runner:

```bash
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --output-dir results \
    --framework pytorch \
    --techniques quantization,pruning \
    --dataset-path data/test_data.csv
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
