# Neural-Scope CI/CD Runner

This document provides a quick guide to using the Neural-Scope CI/CD Runner, a comprehensive tool for integrating Neural-Scope's model optimization capabilities into CI/CD pipelines.

## Overview

The Neural-Scope CI/CD Runner (`neural_scope_cicd.py`) is a consolidated script that provides:

1. **Model Optimization**: Analyze, optimize, and validate ML models
2. **Workflow Creation**: Generate CI/CD workflow files for various systems
3. **Result Tracking**: Track optimization results with MLflow

## Installation

```bash
# Install Neural-Scope with all dependencies
pip install neural-scope[all]

# Or, if you have the source
pip install -e ".[all]"
```

## Usage

### Optimizing a Model

```bash
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --output-dir results \
    --framework pytorch \
    --techniques quantization,pruning \
    --dataset-path data/test_data.csv
```

This will:
1. Load the model from `models/model.pt`
2. Analyze the model's architecture and performance
3. Apply quantization and pruning optimizations
4. Validate the optimized model using the test dataset
5. Save the optimized model and results to the `results` directory

### Creating a CI/CD Workflow

```bash
python neural_scope_cicd.py create-workflow \
    --system github_actions \
    --output-dir .github/workflows \
    --workflow-name model_optimization
```

This will create a GitHub Actions workflow file that automates model optimization.

### Tracking Results with MLflow

```bash
python neural_scope_cicd.py track \
    --model-name my_model \
    --results-path results/optimization_results.json \
    --tracking-uri http://localhost:5000 \
    --experiment-name neural-scope-optimization
```

This will track the optimization results in MLflow for experiment tracking and visualization.

## CI/CD Integration

The Neural-Scope CI/CD Runner is designed to be easily integrated into CI/CD pipelines:

### GitHub Actions

```yaml
- name: Optimize Model
  run: |
    python neural_scope_cicd.py optimize \
      --model-path models/model.pt \
      --output-dir results \
      --framework pytorch \
      --techniques quantization,pruning
```

### GitLab CI

```yaml
optimize:
  stage: optimize
  script:
    - python neural_scope_cicd.py optimize \
        --model-path models/model.pt \
        --output-dir results \
        --framework pytorch \
        --techniques quantization,pruning
```

### Jenkins

```groovy
stage('Optimize Model') {
    steps {
        sh 'python neural_scope_cicd.py optimize --model-path models/model.pt --output-dir results --framework pytorch --techniques quantization,pruning'
    }
}
```

## Output

The optimization process generates several output files:

- `model_analysis.json`: Analysis of the original model
- `optimized_model.pt` or `optimized_model.h5`: The optimized model file
- `optimization_results.json`: Detailed optimization results
- `validation_results.json`: Performance and accuracy metrics
- `optimization_summary.json`: Summary of the optimization process

## Advanced Usage

### Custom Optimization Techniques

You can specify custom optimization techniques:

```bash
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --techniques "quantization,pruning,distillation,layer_fusion"
```

### Framework-Specific Options

Different frameworks may have specific optimization options:

```bash
# PyTorch-specific optimizations
python neural_scope_cicd.py optimize \
    --model-path models/model.pt \
    --framework pytorch \
    --techniques "quantization{dtype=qint8},pruning{sparsity=0.7}"

# TensorFlow-specific optimizations
python neural_scope_cicd.py optimize \
    --model-path models/model.h5 \
    --framework tensorflow \
    --techniques "quantization{post_training=true},pruning{block_size=2}"
```

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Ensure the model file exists and is in the correct format
   - Verify that you've specified the correct framework

2. **Optimization fails**:
   - Check that the model architecture is compatible with the optimization techniques
   - Try using fewer or different optimization techniques

3. **Validation fails**:
   - Ensure the dataset is in the correct format
   - Check that the dataset is compatible with the model

### Getting Help

For more detailed information, use the help command:

```bash
python neural_scope_cicd.py --help
python neural_scope_cicd.py optimize --help
python neural_scope_cicd.py create-workflow --help
python neural_scope_cicd.py track --help
```
