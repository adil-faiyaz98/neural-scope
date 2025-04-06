# Neural-Scope Library Guide

This guide provides comprehensive documentation for using Neural-Scope as a Python library.

## Table of Contents

- [Installation](#installation)
- [Core Components](#core-components)
- [Code Analysis](#code-analysis)
- [Model Analysis](#model-analysis)
- [Data Quality Assessment](#data-quality-assessment)
- [Model Compression](#model-compression)
- [Performance Profiling](#performance-profiling)
- [Hardware Optimization](#hardware-optimization)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Basic installation
pip install neural-scope

# With PyTorch support
pip install neural-scope[pytorch]

# With TensorFlow support
pip install neural-scope[tensorflow]

# With visualization support
pip install neural-scope[visualization]

# With all optional dependencies
pip install neural-scope[all]
```

## Core Components

Neural-Scope is organized into several core modules:

- `advanced_analysis.analyzer`: Main entry point for analysis
- `advanced_analysis.algorithm_complexity`: Code complexity analysis
- `advanced_analysis.ml_advisor`: ML algorithm recognition and optimization suggestions
- `advanced_analysis.performance`: Performance profiling and optimization
- `advanced_analysis.data_quality`: Data quality assessment and improvement
- `advanced_analysis.algorithm_complexity.model_compression`: Model compression techniques

## Code Analysis

Analyze Python code for ML inefficiencies and optimization opportunities.

```python
from advanced_analysis.analyzer import Analyzer

# Create an analyzer
analyzer = Analyzer()

# Analyze code from a file
results = analyzer.analyze_code("path/to/file.py")

# Analyze code from a string
code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""
results = analyzer.analyze_code(code)

# Access analysis results
print(f"Static analysis: {results['static_analysis']['overall_time_complexity']}")
print(f"Inefficiencies: {len(results['inefficiencies'])}")
print(f"Optimization suggestions: {len(results['optimization_suggestions'])}")

# Generate a report
from advanced_analysis.visualization import ReportGenerator
report_generator = ReportGenerator()
html_report = report_generator.generate_html_report(results)
with open("report.html", "w") as f:
    f.write(html_report)
```

## Model Analysis

Analyze ML model architecture, performance, and optimization potential.

### PyTorch Models

```python
import torch
import torch.nn as nn
from advanced_analysis.algorithm_complexity import ModelAnalyzer

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a model
model = SimpleModel()

# Analyze model
analyzer = ModelAnalyzer()
analysis = analyzer.analyze_model(model, framework="pytorch")

# Print analysis results
print(f"Model type: {analysis['model_type']}")
print(f"Parameters: {analysis['parameters']}")
print(f"Layers: {len(analysis['layers'])}")
print(f"Complexity: {analysis['complexity']}")
```

### TensorFlow Models

```python
import tensorflow as tf
from advanced_analysis.algorithm_complexity import ModelAnalyzer

# Create a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Analyze model
analyzer = ModelAnalyzer()
analysis = analyzer.analyze_model(model, framework="tensorflow")

# Print analysis results
print(f"Model type: {analysis['model_type']}")
print(f"Parameters: {analysis['parameters']}")
print(f"Layers: {len(analysis['layers'])}")
print(f"Complexity: {analysis['complexity']}")
```

## Data Quality Assessment

Evaluate and improve training data quality.

```python
import pandas as pd
from advanced_analysis.data_quality import DataGuardian

# Create a sample DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 5, 7],  # Duplicate value in id
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],  # Missing value in name
    'age': [25, 30, 35, 40, 45, 50, 200],  # Outlier in age
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
    'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
})

# Create a data guardian
guardian = DataGuardian()

# Generate report
report = guardian.generate_report(df)

# Print report
print(f"Missing values: {report.missing_values['total_missing']}")
print(f"Duplicates: {report.duplicates['total_duplicates']}")
print(f"Outliers: {list(report.outliers['outliers_by_column'].keys())}")

# Generate HTML report
html_report = report.to_html()
with open("data_quality_report.html", "w") as f:
    f.write(html_report)

# Optimize data
from advanced_analysis.data_quality import DataOptimizer
optimizer = DataOptimizer()

# Get optimization suggestions
suggestions = optimizer.suggest_optimizations(df)
for suggestion in suggestions:
    print(f"- {suggestion}")

# Apply optimizations
optimized_df = optimizer.optimize(df)
print(f"Original shape: {df.shape}, Optimized shape: {optimized_df.shape}")
```

## Model Compression

Apply compression techniques to ML models.

```python
import torch
import torch.nn as nn
from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor, ProfileInfo

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a model
model = SimpleModel()

# Create a profile for compression
profile = ProfileInfo(
    framework="pytorch",
    model_type="mlp",
    hardware="cpu",
    techniques=["quantization", "pruning"],
    params={
        "quantization_method": "dynamic",
        "prune_amount": 0.3
    }
)

# Create a compressor
compressor = ModelCompressor(profile)

# Compress model
compressed_model = compressor.compress(model)

# Get logs
logs = compressor.get_logs()
for log in logs:
    print(log)

# Save compressed model
torch.save(compressed_model, "compressed_model.pt")
```

## Performance Profiling

Profile ML model performance and identify bottlenecks.

```python
import torch
import torch.nn as nn
from advanced_analysis.performance import PerformanceProfiler

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a model and sample input
model = SimpleModel()
input_data = torch.randn(32, 10)  # Batch size 32, input dimension 10

# Create a profiler
profiler = PerformanceProfiler()

# Profile model
profile_results = profiler.profile_model(model, input_data, framework="pytorch")

# Print results
print(f"Execution time: {profile_results['execution_time']} seconds")
print(f"Memory usage: {profile_results['memory_usage']} MB")
print(f"Throughput: {profile_results['throughput']} samples/second")

# Analyze vectorization opportunities
from advanced_analysis.performance import VectorizationAnalyzer

# Create a function to analyze
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

# Create a vectorization analyzer
vectorization_analyzer = VectorizationAnalyzer()

# Analyze function
vectorization_results = vectorization_analyzer.analyze_function(process_data)

# Print results
print(f"Naive loops: {len(vectorization_results['naive_loops'])}")
print(f"Vectorization suggestions: {len(vectorization_results['vectorization_suggestions'])}")
```

## Hardware Optimization

Optimize models for specific hardware targets.

```python
from advanced_analysis.utils.hardware_utils import detect_hardware, optimize_for_hardware

# Detect available hardware
hardware_info = detect_hardware()
print(f"Platform: {hardware_info['platform']}")
print(f"CPU count: {hardware_info['cpu_count']}")
print(f"GPU available: {hardware_info['gpu_available']}")
if hardware_info['gpu_available']:
    print(f"GPU count: {hardware_info['gpu_count']}")
    for i, gpu in enumerate(hardware_info['gpu_info']):
        print(f"GPU {i}: {gpu['name']}")

# Optimize a PyTorch model for available hardware
import torch
import torch.nn as nn

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a model
model = SimpleModel()

# Optimize model for available hardware
optimized_model = optimize_for_hardware(model)

# Get optimal batch size
from advanced_analysis.utils.hardware_utils import get_optimal_batch_size
model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # Estimate model size in MB
optimal_batch_size = get_optimal_batch_size(model_size_mb)
print(f"Optimal batch size: {optimal_batch_size}")
```

## Advanced Usage

### Combining Multiple Analyses

```python
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import ModelAnalyzer
from advanced_analysis.performance import PerformanceProfiler
from advanced_analysis.data_quality import DataGuardian

# Create analyzers
code_analyzer = Analyzer()
model_analyzer = ModelAnalyzer()
performance_profiler = PerformanceProfiler()
data_guardian = DataGuardian()

# Perform analyses
code_results = code_analyzer.analyze_code("path/to/file.py")
model_results = model_analyzer.analyze_model(model, framework="pytorch")
performance_results = performance_profiler.profile_model(model, input_data, framework="pytorch")
data_results = data_guardian.generate_report(df)

# Combine results
combined_results = {
    "code_analysis": code_results,
    "model_analysis": model_results,
    "performance_analysis": performance_results,
    "data_quality": data_results.__dict__
}

# Generate comprehensive report
from advanced_analysis.visualization import ReportGenerator
report_generator = ReportGenerator()
html_report = report_generator.generate_html_report(combined_results)
with open("comprehensive_report.html", "w") as f:
    f.write(html_report)
```

### Custom Error Handling

```python
from advanced_analysis.utils.error_handling import handle_errors, with_recovery

# Use error handling decorator
@handle_errors(fallback_return=None, log_level="ERROR", reraise=False)
def analyze_model_safely(model, framework):
    analyzer = ModelAnalyzer()
    return analyzer.analyze_model(model, framework=framework)

# Use recovery decorator
@with_recovery
def compress_model_safely(model, profile):
    compressor = ModelCompressor(profile)
    return compressor.compress(model)

# Configure logging
from advanced_analysis.utils.error_handling import configure_file_logging
configure_file_logging("neural_scope.log", level="DEBUG")
```

## API Reference

For detailed API documentation, visit:
[https://neural-scope.readthedocs.io/](https://neural-scope.readthedocs.io/)

## Examples

### End-to-End Workflow

```python
import torch
import torch.nn as nn
import pandas as pd
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import ModelAnalyzer
from advanced_analysis.performance import PerformanceProfiler
from advanced_analysis.data_quality import DataGuardian
from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor, ProfileInfo

# 1. Analyze code
analyzer = Analyzer()
code_results = analyzer.analyze_code("model_training.py")
print(f"Code inefficiencies: {len(code_results['inefficiencies'])}")

# 2. Analyze data
df = pd.read_csv("training_data.csv")
guardian = DataGuardian()
data_report = guardian.generate_report(df)
print(f"Data quality issues: {data_report.missing_values['total_missing'] + data_report.duplicates['total_duplicates']}")

# 3. Analyze model
model = torch.load("model.pt")
model_analyzer = ModelAnalyzer()
model_analysis = model_analyzer.analyze_model(model, framework="pytorch")
print(f"Model parameters: {model_analysis['parameters']}")

# 4. Profile performance
profiler = PerformanceProfiler()
input_data = torch.randn(32, 10)
performance_results = profiler.profile_model(model, input_data, framework="pytorch")
print(f"Inference time: {performance_results['execution_time']} seconds")

# 5. Compress model
profile = ProfileInfo(
    framework="pytorch",
    model_type="mlp",
    hardware="cpu",
    techniques=["quantization", "pruning"],
    params={
        "quantization_method": "dynamic",
        "prune_amount": 0.3
    }
)
compressor = ModelCompressor(profile)
compressed_model = compressor.compress(model)
torch.save(compressed_model, "compressed_model.pt")

# 6. Compare performance
compressed_performance = profiler.profile_model(compressed_model, input_data, framework="pytorch")
speedup = performance_results['execution_time'] / compressed_performance['execution_time']
print(f"Speedup: {speedup:.2f}x")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch'**:
   ```
   ImportError: No module named 'torch'
   ```
   Solution: Install PyTorch
   ```bash
   pip install torch
   ```

2. **CUDA out of memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch size or model size
   ```python
   # Use optimal batch size
   from advanced_analysis.utils.hardware_utils import get_optimal_batch_size
   optimal_batch_size = get_optimal_batch_size(model_size_mb)
   ```

3. **Version mismatch**:
   ```
   ImportError: cannot import name 'X' from 'Y'
   ```
   Solution: Check version compatibility
   ```python
   from advanced_analysis.version import get_version_info
   version_info = get_version_info()
   print(version_info)
   ```

### Getting Help

For bug reports and feature requests, please visit:
[https://github.com/adil-faiyaz98/neural-scope/issues](https://github.com/adil-faiyaz98/neural-scope/issues)
