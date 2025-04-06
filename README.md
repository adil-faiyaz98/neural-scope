# Neural-Scope

**Neural-Scope** is a comprehensive Python library for analyzing and optimizing machine learning code, models, and data. It provides tools for algorithmic complexity analysis, performance profiling, data quality assessment, and ML-specific optimization recommendations.

## Features

- **Algorithm Complexity Analysis**: Analyze code to determine theoretical and empirical time and space complexity.
- **Performance Profiling**: Profile ML models to identify bottlenecks and optimization opportunities.
- **Data Quality Assessment**: Detect and fix issues in datasets, including missing values, duplicates, and outliers.
- **ML Advisor**: Get ML-specific optimization recommendations based on best practices.
- **Inefficiency Detection**: Automatically detect common inefficiencies in ML code.
- **Visualization**: Create interactive dashboards and reports for analysis results.
- **Expanded Model Architecture Support**: Specialized analysis for diffusion models, transformer variants, and other advanced architectures.
- **Advanced Compression Techniques**: Enhanced quantization, pruning, and distillation capabilities for model optimization.
- **Hardware-Specific Optimizations**: Targeted optimizations for TPUs, GPUs, and other specialized hardware accelerators.
- **MLOps Integration**: Seamless integration with MLflow, Kubeflow, and CI/CD pipelines for end-to-end ML lifecycle management.

## Installation

Install the package via pip:

```bash
pip install neural-scope
```

Or, if you have the source, install it and its dependencies:

```bash
pip install -e .
```

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

You can also install optional dependencies:

```bash
# Install PyTorch support
pip install -e ".[pytorch]"

# Install TensorFlow support
pip install -e ".[tensorflow]"

# Install visualization support
pip install -e ".[visualization]"

# Install all optional dependencies
pip install -e ".[all]"
```

## Usage

### Analyzing Code Complexity

```python
from advanced_analysis.analyzer import Analyzer

# Define some code to analyze
code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""

# Create an analyzer and analyze the code
analyzer = Analyzer()
results = analyzer.analyze_code(code)

# Print the results
print(f"Static analysis: {results['static_analysis']['overall_time_complexity']}")
print(f"Inefficiencies: {len(results['inefficiencies'])}")
print(f"Optimization suggestions: {len(results['optimization_suggestions'])}")

# Generate a report
report = analyzer.generate_report(results, format="html")
with open("report.html", "w") as f:
    f.write(report)
```

### Analyzing Data Quality

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

# Create a data guardian and analyze the data
guardian = DataGuardian()
report = guardian.generate_report(df)

# Print the report
print(f"Missing values: {report.missing_values['total_missing']}")
print(f"Duplicates: {report.duplicates['total_duplicates']}")
print(f"Outliers: {list(report.outliers['outliers_by_column'].keys())}")

# Generate an HTML report
html_report = report.to_html()
with open("data_quality_report.html", "w") as f:
    f.write(html_report)
```

### Profiling ML Models

```python
import torch
import torch.nn as nn
from advanced_analysis.performance import ModelPerformanceProfiler

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

# Create a profiler and profile the model
profiler = ModelPerformanceProfiler(model=model, framework="pytorch")
results = profiler.profile_model(input_data)

# Print the results
print(f"Inference time: {results['inference_time']['avg_inference_time']:.6f} seconds")
print(f"Memory usage: {results['memory_usage']['peak_memory_mb']:.2f} MB")
print(f"Throughput: {results['throughput']['throughput']:.2f} samples/second")

# Get optimization recommendations
recommendations = profiler.generate_optimization_recommendations()
for recommendation in recommendations:
    print(f"- {recommendation['description']}")
```

### Advanced Model Architecture Support

```python
from advanced_analysis.model_architectures import DiffusionModelAnalyzer, TransformerAnalyzer

# Analyze a diffusion model
diffusion_model = load_my_diffusion_model()  # Your model loading code
diffusion_analyzer = DiffusionModelAnalyzer(model=diffusion_model)
diffusion_results = diffusion_analyzer.analyze()
print(f"Diffusion model steps efficiency score: {diffusion_results['efficiency_score']}")
print(f"Recommended optimizations: {diffusion_results['recommendations']}")

# Analyze a transformer model
transformer_model = load_my_transformer()  # Your model loading code
transformer_analyzer = TransformerAnalyzer(model=transformer_model)
transformer_results = transformer_analyzer.analyze()
print(f"Attention mechanism efficiency: {transformer_results['attention_efficiency']}")
print(f"Parameter allocation analysis: {transformer_results['parameter_allocation']}")
```

### Advanced Compression Techniques

```python
from advanced_analysis.compression import ModelCompressor

# Load your model
model = load_my_model()  # Your model loading code

# Initialize compressor
compressor = ModelCompressor(model, framework="pytorch")

# Apply advanced quantization
quantized_model = compressor.quantize(
    method="dynamic_aware",
    bits=8,
    preserve_accuracy=True
)

# Apply pruning
pruned_model = compressor.prune(
    method="structured_magnitude",
    sparsity=0.7,
    preserve_layers=["output_layer"]
)

# Apply knowledge distillation
student_model = create_student_model()  # Your student model definition
distilled_model = compressor.distill(
    teacher_model=model,
    student_model=student_model,
    training_data=train_dataloader,
    validation_data=val_dataloader
)

# Evaluate compression results
compression_stats = compressor.evaluate_compression(
    original_model=model,
    compressed_model=quantized_model,
    test_data=test_dataloader
)
print(f"Size reduction: {compression_stats['size_reduction_percentage']}%")
print(f"Speed improvement: {compression_stats['inference_speedup_percentage']}%")
print(f"Accuracy change: {compression_stats['accuracy_change']}%")
```

### Hardware-Specific Optimizations

```python
from advanced_analysis.hardware import HardwareOptimizer

# Initialize hardware optimizer
optimizer = HardwareOptimizer(model, target_hardware="nvidia_a100")

# Get hardware-specific recommendations
recommendations = optimizer.recommend_optimizations()
for rec in recommendations:
    print(f"{rec['type']}: {rec['description']}")

# Apply the optimizations
optimized_model = optimizer.apply_optimizations()

# Profile performance on specific hardware
performance = optimizer.profile_on_hardware(
    model=optimized_model,
    input_shapes={"input": (1, 3, 224, 224)},
    hardware_info={"type": "nvidia_a100", "memory": "40GB"}
)
print(f"Optimized inference time: {performance['inference_time']} ms")
print(f"Memory utilization: {performance['memory_utilization']}%")
print(f"Power efficiency: {performance['power_efficiency']} inferences/watt")
```

### MLOps Integration

```python
from advanced_analysis.mlops import MLflowIntegrator, KubeflowIntegrator, CICDIntegrator

# MLflow integration
mlflow_integrator = MLflowIntegrator()
mlflow_integrator.track_model_analysis(
    model_name="my_production_model",
    analysis_results=analysis_results,
    metrics={"accuracy": 0.95, "inference_time_ms": 45}
)

# Register model with tracking metadata
mlflow_integrator.register_optimized_model(
    original_model=model,
    optimized_model=optimized_model,
    optimization_history=optimizer.history,
    model_name="optimized_production_model"
)

# Kubeflow integration
kubeflow_integrator = KubeflowIntegrator(pipeline_config="kubeflow_pipeline.yaml")
kubeflow_integrator.create_optimization_step(
    optimization_function=optimizer.apply_optimizations,
    resource_requirements={"cpu": 4, "memory": "16Gi"}
)
pipeline = kubeflow_integrator.build_pipeline()
pipeline.run()

# CI/CD integration
cicd = CICDIntegrator(system="github_actions")
cicd.create_optimization_workflow(
    optimization_script="optimization.py",
    test_script="validate_optimized_model.py",
    trigger_on=["push", "pull_request"],
    notify_on_completion=True
)
```

## Command-Line Interface

Neural-Scope provides a command-line interface for easy access to its functionality:

```bash
# Show version information
neural-scope version

# Analyze Python code
neural-scope analyze-code path/to/file.py --output results.json

# Analyze a model
neural-scope analyze-model path/to/model.pt --framework pytorch --output model_analysis.json

# Analyze data quality
neural-scope analyze-data path/to/data.csv --format csv --output data_report.json

# Compress a model
neural-scope compress-model path/to/model.pt --framework pytorch --techniques quantization,pruning --output compressed_model.pt
```

For more options, use the help command:

```bash
neural-scope --help
neural-scope analyze-code --help
neural-scope analyze-model --help
neural-scope analyze-data --help
neural-scope compress-model --help
```

## Documentation

For more detailed documentation, see the [docs](docs/) directory or visit our [documentation website](https://neural-scope.github.io/advanced-analysis/).

## Examples

Check out the [notebooks](notebooks/) directory for example notebooks demonstrating various features of the library.

## Testing

Run the tests with pytest:

```bash
python -m pytest
```

Or use the provided script:

```bash
python tests/run_tests.py
```

### Comprehensive Tests

```bash
# Run all comprehensive tests
python tests/run_comprehensive_tests.py

# Run specific test categories
python tests/run_comprehensive_tests.py --category model_architectures
python tests/run_comprehensive_tests.py --category compression
python tests/run_comprehensive_tests.py --category hardware
python tests/run_comprehensive_tests.py --category mlops

# Run tests with specific hardware targets
python tests/run_comprehensive_tests.py --hardware nvidia_a100,tpu_v4

# Generate detailed test reports
python tests/run_comprehensive_tests.py --report-format html --output-dir test_reports
```

The comprehensive test suite includes:

- Model architecture compatibility tests
- Compression impact assessment
- Hardware-specific performance benchmarks
- MLOps integration validation
- End-to-end workflow tests

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request. Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
