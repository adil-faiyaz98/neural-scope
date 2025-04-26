# Neural-Scope CI/CD Integration Guide

This guide provides detailed instructions for integrating Neural-Scope into CI/CD pipelines to automate model analysis, optimization, security testing, and robustness evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [GitHub Actions Integration](#github-actions-integration)
4. [GitLab CI Integration](#gitlab-ci-integration)
5. [Jenkins Integration](#jenkins-integration)
6. [Azure DevOps Integration](#azure-devops-integration)
7. [CircleCI Integration](#circleci-integration)
8. [Travis CI Integration](#travis-ci-integration)
9. [Custom Workflows](#custom-workflows)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Introduction

Integrating Neural-Scope into your CI/CD pipeline enables automated model analysis, optimization, security testing, and robustness evaluation as part of your ML workflow. This integration helps ensure that models meet quality, performance, and security standards before deployment.

## Prerequisites

Before you begin, ensure you have the following:

- Neural-Scope installed (`pip install neural-scope`)
- Access to a CI/CD platform (GitHub Actions, GitLab CI, Jenkins, etc.)
- A machine learning model to analyze
- (Optional) MLflow server for tracking results

## GitHub Actions Integration

### Basic Integration

Create a file named `.github/workflows/neural_scope.yml` in your repository:

```yaml
name: Neural-Scope Analysis

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'
  pull_request:
    branches: [ main ]
    paths:
      - 'models/**'
  workflow_dispatch:
    inputs:
      model_path:
        description: 'Path to the model file'
        required: true
        default: 'models/model.pt'
      framework:
        description: 'Model framework (pytorch, tensorflow)'
        required: true
        default: 'pytorch'

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
          pip install neural-scope torch torchvision
          
      - name: Analyze model
        run: |
          python -m neural_scope.cli analyze \
            --model-path ${{ github.event.inputs.model_path || 'models/model.pt' }} \
            --framework ${{ github.event.inputs.framework || 'pytorch' }} \
            --output-dir results
            
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: neural-scope-results
          path: results/
```

### Advanced Integration with MLflow

```yaml
name: Neural-Scope Analysis with MLflow

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'

jobs:
  analyze:
    runs-on: ubuntu-latest
    services:
      mlflow:
        image: ghcr.io/mlflow/mlflow:latest
        ports:
          - 5000:5000
        options: --entrypoint mlflow
        args: ui --host 0.0.0.0 --port 5000
        
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install neural-scope torch torchvision mlflow
          
      - name: Analyze model
        run: |
          python -m neural_scope.cli analyze \
            --model-path models/model.pt \
            --framework pytorch \
            --output-dir results \
            --mlflow-tracking-uri http://localhost:5000 \
            --mlflow-experiment-name github-actions
            
      - name: Optimize model
        run: |
          python -m neural_scope.cli optimize \
            --model-path models/model.pt \
            --framework pytorch \
            --output-dir results \
            --techniques quantization pruning \
            --mlflow-tracking-uri http://localhost:5000 \
            --mlflow-experiment-name github-actions
            
      - name: Test security
        run: |
          python -m neural_scope.cli security \
            --model-path models/model.pt \
            --framework pytorch \
            --output-dir results \
            --mlflow-tracking-uri http://localhost:5000 \
            --mlflow-experiment-name github-actions
            
      - name: Test robustness
        run: |
          python -m neural_scope.cli robustness \
            --model-path models/model.pt \
            --framework pytorch \
            --output-dir results \
            --attack-types fgsm pgd \
            --mlflow-tracking-uri http://localhost:5000 \
            --mlflow-experiment-name github-actions
            
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: neural-scope-results
          path: results/
```

## GitLab CI Integration

Create a file named `.gitlab-ci.yml` in your repository:

```yaml
image: python:3.8

stages:
  - analyze
  - optimize
  - security
  - robustness

variables:
  MODEL_PATH: "models/model.pt"
  FRAMEWORK: "pytorch"
  OUTPUT_DIR: "results"

before_script:
  - pip install neural-scope torch torchvision

analyze:
  stage: analyze
  script:
    - python -m neural_scope.cli analyze --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR
  artifacts:
    paths:
      - $OUTPUT_DIR/analysis_report.json
      - $OUTPUT_DIR/analysis_report.html

optimize:
  stage: optimize
  script:
    - python -m neural_scope.cli optimize --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR --techniques quantization pruning
  artifacts:
    paths:
      - $OUTPUT_DIR/optimization_report.json
      - $OUTPUT_DIR/optimization_report.html
      - $OUTPUT_DIR/optimized_model.pt

security:
  stage: security
  script:
    - python -m neural_scope.cli security --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR
  artifacts:
    paths:
      - $OUTPUT_DIR/security_report.json
      - $OUTPUT_DIR/security_report.html

robustness:
  stage: robustness
  script:
    - python -m neural_scope.cli robustness --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR --attack-types fgsm pgd
  artifacts:
    paths:
      - $OUTPUT_DIR/robustness_report.json
      - $OUTPUT_DIR/robustness_report.html
```

## Jenkins Integration

Create a `Jenkinsfile` in your repository:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.8'
        }
    }
    
    environment {
        MODEL_PATH = 'models/model.pt'
        FRAMEWORK = 'pytorch'
        OUTPUT_DIR = 'results'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install neural-scope torch torchvision'
            }
        }
        
        stage('Analyze') {
            steps {
                sh 'python -m neural_scope.cli analyze --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR'
            }
        }
        
        stage('Optimize') {
            steps {
                sh 'python -m neural_scope.cli optimize --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR --techniques quantization pruning'
            }
        }
        
        stage('Security') {
            steps {
                sh 'python -m neural_scope.cli security --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR'
            }
        }
        
        stage('Robustness') {
            steps {
                sh 'python -m neural_scope.cli robustness --model-path $MODEL_PATH --framework $FRAMEWORK --output-dir $OUTPUT_DIR --attack-types fgsm pgd'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'results/**', fingerprint: true
        }
    }
}
```

## Azure DevOps Integration

Create a file named `azure-pipelines.yml` in your repository:

```yaml
trigger:
  branches:
    include:
      - main
  paths:
    include:
      - 'models/**'

pool:
  vmImage: 'ubuntu-latest'

variables:
  MODEL_PATH: 'models/model.pt'
  FRAMEWORK: 'pytorch'
  OUTPUT_DIR: 'results'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.8'
    addToPath: true

- script: |
    pip install neural-scope torch torchvision
  displayName: 'Install dependencies'

- script: |
    python -m neural_scope.cli analyze --model-path $(MODEL_PATH) --framework $(FRAMEWORK) --output-dir $(OUTPUT_DIR)
  displayName: 'Analyze model'

- script: |
    python -m neural_scope.cli optimize --model-path $(MODEL_PATH) --framework $(FRAMEWORK) --output-dir $(OUTPUT_DIR) --techniques quantization pruning
  displayName: 'Optimize model'

- script: |
    python -m neural_scope.cli security --model-path $(MODEL_PATH) --framework $(FRAMEWORK) --output-dir $(OUTPUT_DIR)
  displayName: 'Test security'

- script: |
    python -m neural_scope.cli robustness --model-path $(MODEL_PATH) --framework $(FRAMEWORK) --output-dir $(OUTPUT_DIR) --attack-types fgsm pgd
  displayName: 'Test robustness'

- task: PublishBuildArtifacts@1
  inputs:
    pathtoPublish: '$(OUTPUT_DIR)'
    artifactName: 'neural-scope-results'
```

## CircleCI Integration

Create a file named `.circleci/config.yml` in your repository:

```yaml
version: 2.1

jobs:
  analyze:
    docker:
      - image: cimg/python:3.8
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install neural-scope torch torchvision
      - run:
          name: Analyze model
          command: |
            python -m neural_scope.cli analyze \
              --model-path models/model.pt \
              --framework pytorch \
              --output-dir results
      - store_artifacts:
          path: results
          destination: neural-scope-results

  optimize:
    docker:
      - image: cimg/python:3.8
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install neural-scope torch torchvision
      - run:
          name: Optimize model
          command: |
            python -m neural_scope.cli optimize \
              --model-path models/model.pt \
              --framework pytorch \
              --output-dir results \
              --techniques quantization pruning
      - store_artifacts:
          path: results
          destination: neural-scope-results

workflows:
  version: 2
  neural_scope:
    jobs:
      - analyze
      - optimize:
          requires:
            - analyze
```

## Travis CI Integration

Create a file named `.travis.yml` in your repository:

```yaml
language: python
python:
  - "3.8"

install:
  - pip install neural-scope torch torchvision

script:
  - python -m neural_scope.cli analyze --model-path models/model.pt --framework pytorch --output-dir results
  - python -m neural_scope.cli optimize --model-path models/model.pt --framework pytorch --output-dir results --techniques quantization pruning
  - python -m neural_scope.cli security --model-path models/model.pt --framework pytorch --output-dir results
  - python -m neural_scope.cli robustness --model-path models/model.pt --framework pytorch --output-dir results --attack-types fgsm pgd

after_success:
  - tar -czf neural-scope-results.tar.gz results/
```

## Custom Workflows

### Pre-commit Hook

Create a file named `.pre-commit-hooks.yaml` in your repository:

```yaml
- id: neural-scope-analyze
  name: Neural-Scope Analysis
  description: Analyze ML models with Neural-Scope
  entry: python -m neural_scope.cli analyze
  language: python
  files: \.pt$|\.h5$|\.pb$
  args: [--framework, pytorch, --output-dir, results]
```

### Custom Python Script

```python
#!/usr/bin/env python
"""
Custom Neural-Scope CI/CD script.
"""

import argparse
import os
from neural_scope import NeuralScope

def main():
    parser = argparse.ArgumentParser(description="Neural-Scope CI/CD script")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--framework", default="pytorch", help="Model framework")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Neural-Scope
    neural_scope = NeuralScope(
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment_name="custom-cicd"
    )
    
    # Load model
    if args.framework == "pytorch":
        import torch
        model = torch.load(args.model_path)
    elif args.framework == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model_path)
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")
    
    # Analyze model
    analysis_results = neural_scope.analyze_model(
        model=model,
        model_name=os.path.basename(args.model_path),
        framework=args.framework
    )
    
    # Optimize model
    optimized_model, optimization_results = neural_scope.optimize_model(
        model=model,
        model_name=os.path.basename(args.model_path),
        framework=args.framework,
        techniques=["quantization", "pruning"]
    )
    
    # Test security
    security_results = neural_scope.analyze_security(
        model=model,
        model_name=os.path.basename(args.model_path),
        framework=args.framework
    )
    
    # Test robustness
    robustness_results = neural_scope.test_robustness(
        model=model,
        model_name=os.path.basename(args.model_path),
        framework=args.framework,
        attack_types=["fgsm", "pgd"]
    )
    
    # Print summary
    print("\nNeural-Scope Analysis Summary:")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Parameters: {analysis_results['parameters']:,}")
    print(f"Memory Usage: {analysis_results['memory_usage_mb']:.2f} MB")
    print(f"Inference Time: {analysis_results['inference_time_ms']:.2f} ms")
    print(f"Size Reduction: {optimization_results['size_reduction_percentage']:.1f}%")
    print(f"Security Score: {security_results['security_score']}/100")
    print(f"Robustness Score: {robustness_results['robustness_score']:.1f}/100")
    
    if args.mlflow_uri:
        print(f"\nResults tracked in MLflow run: {neural_scope.mlflow_run_id}")
        print(f"View at: {args.mlflow_uri}/#/experiments/{neural_scope.mlflow_experiment_id}/runs/{neural_scope.mlflow_run_id}")

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Version Control for Models

Store your models in version control or a model registry to ensure reproducibility:

```yaml
# .gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
```

### 2. Define Quality Gates

Set up quality gates to ensure models meet minimum standards:

```python
# quality_gates.py
def check_quality_gates(results):
    """Check if results meet quality gates."""
    gates = {
        "max_memory_usage_mb": 100,
        "max_inference_time_ms": 50,
        "min_size_reduction_percentage": 30,
        "min_security_score": 70,
        "min_robustness_score": 40
    }
    
    passed = True
    failures = []
    
    if results["analysis"]["memory_usage_mb"] > gates["max_memory_usage_mb"]:
        passed = False
        failures.append(f"Memory usage too high: {results['analysis']['memory_usage_mb']} MB > {gates['max_memory_usage_mb']} MB")
    
    if results["analysis"]["inference_time_ms"] > gates["max_inference_time_ms"]:
        passed = False
        failures.append(f"Inference time too high: {results['analysis']['inference_time_ms']} ms > {gates['max_inference_time_ms']} ms")
    
    if results["optimization"]["size_reduction_percentage"] < gates["min_size_reduction_percentage"]:
        passed = False
        failures.append(f"Size reduction too low: {results['optimization']['size_reduction_percentage']}% < {gates['min_size_reduction_percentage']}%")
    
    if results["security"]["security_score"] < gates["min_security_score"]:
        passed = False
        failures.append(f"Security score too low: {results['security']['security_score']} < {gates['min_security_score']}")
    
    if results["robustness"]["robustness_score"] < gates["min_robustness_score"]:
        passed = False
        failures.append(f"Robustness score too low: {results['robustness']['robustness_score']} < {gates['min_robustness_score']}")
    
    return passed, failures
```

### 3. Notification and Reporting

Set up notifications for analysis results:

```python
# notifications.py
def send_slack_notification(results, webhook_url):
    """Send results to Slack."""
    import requests
    import json
    
    message = {
        "text": "Neural-Scope Analysis Results",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Neural-Scope Analysis: {results['model_name']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Parameters:* {results['analysis']['parameters']:,}"},
                    {"type": "mrkdwn", "text": f"*Memory Usage:* {results['analysis']['memory_usage_mb']:.2f} MB"},
                    {"type": "mrkdwn", "text": f"*Inference Time:* {results['analysis']['inference_time_ms']:.2f} ms"},
                    {"type": "mrkdwn", "text": f"*Size Reduction:* {results['optimization']['size_reduction_percentage']:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Security Score:* {results['security']['security_score']}/100"},
                    {"type": "mrkdwn", "text": f"*Robustness Score:* {results['robustness']['robustness_score']:.1f}/100"}
                ]
            }
        ]
    }
    
    response = requests.post(webhook_url, json=message)
    return response.status_code == 200
```

### 4. Scheduled Analysis

Set up scheduled analysis to track model drift:

```yaml
# GitHub Actions scheduled analysis
name: Scheduled Neural-Scope Analysis

on:
  schedule:
    - cron: '0 0 * * 1'  # Run every Monday at midnight

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
          pip install neural-scope torch torchvision
      - name: Analyze models
        run: |
          for model in models/*.pt; do
            python -m neural_scope.cli analyze --model-path "$model" --framework pytorch --output-dir "results/$(basename "$model" .pt)"
          done
```

## Troubleshooting

### Common Issues

1. **Model loading errors**:
   - Error: `Error loading model from file`
   - Solution: Ensure the model file exists and is in the correct format for the specified framework

2. **Missing dependencies**:
   - Error: `ModuleNotFoundError: No module named 'torch'`
   - Solution: Install the required dependencies for the model framework (`pip install torch torchvision` for PyTorch)

3. **Permission issues**:
   - Error: `Permission denied: 'results'`
   - Solution: Ensure the CI/CD runner has write permissions to the output directory

4. **Memory issues**:
   - Error: `MemoryError` or `OOM`
   - Solution: Use a runner with more memory or optimize the model loading process

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the [Neural-Scope documentation](https://neural-scope.readthedocs.io/)
2. Open an issue on the [Neural-Scope GitHub repository](https://github.com/adil-faiyaz98/neural-scope/issues)

## Conclusion

Integrating Neural-Scope into your CI/CD pipeline enables automated model analysis, optimization, security testing, and robustness evaluation. By following this guide, you can ensure that your models meet quality, performance, and security standards before deployment.
