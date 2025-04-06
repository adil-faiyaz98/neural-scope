# Neural-Scope CLI Guide

This guide provides comprehensive documentation for using the Neural-Scope command-line interface (CLI).

## Table of Contents

- [Installation](#installation)
- [Command Overview](#command-overview)
- [Global Options](#global-options)
- [Commands](#commands)
  - [analyze-code](#analyze-code)
  - [analyze-model](#analyze-model)
  - [analyze-data](#analyze-data)
  - [compress-model](#compress-model)
  - [version](#version)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Install from PyPI
pip install neural-scope

# Verify installation
neural-scope version
```

## Command Overview

The Neural-Scope CLI provides the following commands:

```
neural-scope [OPTIONS] COMMAND [ARGS]...
```

Available commands:

- `analyze-code`: Analyze Python code for ML inefficiencies and optimization opportunities
- `analyze-model`: Analyze ML model architecture, performance, and optimization potential
- `analyze-data`: Analyze data quality and preprocessing requirements
- `compress-model`: Apply compression techniques to ML models
- `version`: Display version information

## Global Options

These options apply to all commands:

- `-v, --verbose`: Enable verbose logging
- `--log-file PATH`: Path to log file

## Commands

### analyze-code

Analyze Python code for ML inefficiencies and optimization opportunities.

```
neural-scope analyze-code [OPTIONS] FILE
```

Arguments:
- `FILE`: Path to Python file to analyze (use '-' for stdin)

Options:
- `-o, --output PATH`: Path to output file (default: stdout)
- `-f, --format [json|yaml|html|markdown]`: Output format (default: json)

Example:
```bash
# Analyze a Python file and output results to stdout
neural-scope analyze-code model_training.py

# Analyze a Python file and save results to a file
neural-scope analyze-code model_training.py --output analysis_results.json

# Analyze a Python file and generate an HTML report
neural-scope analyze-code model_training.py --format html --output analysis_report.html
```

### analyze-model

Analyze ML model architecture, performance, and optimization potential.

```
neural-scope analyze-model [OPTIONS] MODEL
```

Arguments:
- `MODEL`: Path to model file

Options:
- `--framework [pytorch|tensorflow|sklearn]`: Model framework (default: pytorch)
- `-o, --output PATH`: Path to output file (default: stdout)
- `-f, --format [json|yaml|html|markdown]`: Output format (default: json)

Example:
```bash
# Analyze a PyTorch model
neural-scope analyze-model model.pt --framework pytorch

# Analyze a TensorFlow model and save results to a file
neural-scope analyze-model model.h5 --framework tensorflow --output tf_analysis.json

# Analyze a scikit-learn model and generate an HTML report
neural-scope analyze-model model.joblib --framework sklearn --format html --output sklearn_report.html
```

### analyze-data

Analyze data quality and preprocessing requirements.

```
neural-scope analyze-data [OPTIONS] DATA
```

Arguments:
- `DATA`: Path to data file

Options:
- `--format [csv|json|parquet|pickle]`: Data format (default: csv)
- `-o, --output PATH`: Path to output file (default: stdout)
- `--output-format [json|yaml|html|markdown]`: Output format (default: json)

Example:
```bash
# Analyze a CSV file
neural-scope analyze-data dataset.csv

# Analyze a Parquet file and save results to a file
neural-scope analyze-data dataset.parquet --format parquet --output data_analysis.json

# Analyze a JSON file and generate an HTML report
neural-scope analyze-data dataset.json --format json --output-format html --output data_report.html
```

### compress-model

Apply compression techniques to ML models.

```
neural-scope compress-model [OPTIONS] MODEL
```

Arguments:
- `MODEL`: Path to model file

Options:
- `--framework [pytorch|tensorflow|sklearn]`: Model framework (default: pytorch)
- `--model-type [cnn|rnn|transformer|mlp|other]`: Model type (default: cnn)
- `--hardware [cpu|gpu|mobile|edge]`: Target hardware (default: cpu)
- `--techniques TEXT`: Compression techniques to apply (comma-separated)
- `--quantization-method [dynamic|static|qat|auto]`: Quantization method (default: auto)
- `--prune-amount FLOAT`: Pruning amount (0.0-1.0) (default: 0.3)
- `-o, --output PATH`: Path to output model file
- `--report PATH`: Path to report file (default: stdout)
- `--format [json|yaml|html|markdown]`: Report format (default: json)

Example:
```bash
# Quantize a PyTorch model
neural-scope compress-model model.pt --techniques quantization --output quantized_model.pt

# Apply multiple compression techniques to a TensorFlow model
neural-scope compress-model model.h5 --framework tensorflow --techniques quantization,pruning --output compressed_model.h5

# Compress a model for mobile deployment
neural-scope compress-model model.pt --hardware mobile --techniques quantization,pruning,distillation --output mobile_model.pt
```

### version

Display version information.

```
neural-scope version [OPTIONS]
```

Options:
- `--verbose`: Show detailed version information

Example:
```bash
# Display version
neural-scope version

# Display detailed version information
neural-scope version --verbose
```

## Output Formats

Neural-Scope CLI supports multiple output formats:

- **JSON**: Machine-readable format (default)
- **YAML**: Human-readable structured format
- **HTML**: Interactive HTML report with visualizations
- **Markdown**: Markdown report for documentation

## Examples

### Basic Workflow

```bash
# 1. Analyze code
neural-scope analyze-code model_training.py --output code_analysis.json

# 2. Analyze model
neural-scope analyze-model trained_model.pt --output model_analysis.json

# 3. Compress model
neural-scope compress-model trained_model.pt --techniques quantization,pruning --output optimized_model.pt
```

### Pipeline Integration

```bash
# Analyze and compress in one pipeline
neural-scope analyze-model model.pt > analysis.json && \
neural-scope compress-model model.pt --techniques quantization --output compressed_model.pt
```

### Batch Processing

```bash
# Process multiple files
for file in *.py; do
    neural-scope analyze-code "$file" --output "analysis_$(basename "$file" .py).json"
done
```

## Advanced Usage

### Custom Logging

```bash
# Enable verbose logging to a file
neural-scope analyze-code model.py --verbose --log-file analysis.log
```

### Combining with Other Tools

```bash
# Pipe output to jq for filtering
neural-scope analyze-code model.py | jq '.optimization_suggestions'

# Use with grep to find specific issues
neural-scope analyze-code model.py | grep -i "inefficient"
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```
   Error: Required dependency 'torch' is not installed
   ```
   Solution: Install the required dependency
   ```bash
   pip install torch
   ```

2. **File Not Found**:
   ```
   Error: File not found: model.pt
   ```
   Solution: Check the file path and ensure the file exists

3. **Unsupported Framework**:
   ```
   Error: Unsupported framework: keras
   ```
   Solution: Use one of the supported frameworks (pytorch, tensorflow, sklearn)

### Getting Help

For more help, use the `--help` option:

```bash
neural-scope --help
neural-scope analyze-code --help
```

For bug reports and feature requests, please visit:
[https://github.com/adil-faiyaz98/neural-scope/issues](https://github.com/adil-faiyaz98/neural-scope/issues)
