"""
End-to-end tests for the advanced_analysis framework.

This module contains comprehensive end-to-end tests that exercise multiple
components of the framework together, simulating real-world usage scenarios.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the modules to test
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import StaticAnalyzer, DynamicAnalyzer, ComplexityAnalyzer, ModelAnalyzer
from advanced_analysis.data_quality import DataGuardian, DataOptimizer
from advanced_analysis.ml_advisor import AlgorithmRecognizer, InefficiencyDetector
from advanced_analysis.performance import PerformanceProfiler, DistributedTrainingAnalyzer, RooflineAnalyzer, VectorizationAnalyzer

# Skip tests if optional dependencies are not available
pytorch_available = pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not installed"),
    reason="PyTorch not installed"
)

tensorflow_available = pytest.mark.skipif(
    not pytest.importorskip("tensorflow", reason="TensorFlow not installed"),
    reason="TensorFlow not installed"
)

pandas_available = pytest.mark.skipif(
    not pytest.importorskip("pandas", reason="Pandas not installed"),
    reason="Pandas not installed"
)

# Test data and models
@pytest.fixture
def sample_code():
    """Sample ML code for testing."""
    return """
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

def train_linear_model(X, y):
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    return model

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_nn_model(X, y, input_size, hidden_size, output_size, epochs=100):
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def inefficient_function(data):
    # Inefficient implementation with loops
    result = []
    for i in range(len(data)):
        result.append(data[i] ** 2)
    return result

def bubble_sort(arr):
    # Inefficient sorting algorithm
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    import pandas as pd
    import numpy as np
    
    # Create a DataFrame with some issues
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 7],  # Duplicate value in id
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],  # Missing value in name
        'age': [25, 30, 35, 40, 45, 50, 200],  # Outlier in age
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
    })
    
    # Create numpy arrays for ML models
    X = np.random.rand(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 100)
    
    return {
        'dataframe': df,
        'X': X,
        'y': y
    }

@pytest.fixture
def pytorch_model():
    """Create a PyTorch model for testing."""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(5, 10)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    return model

@pytest.fixture
def tensorflow_model():
    """Create a TensorFlow model for testing."""
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    return model

# End-to-end test scenarios
@pytest.mark.end_to_end
class TestEndToEnd:
    """End-to-end tests for the advanced_analysis framework."""
    
    def test_code_analysis_pipeline(self, sample_code):
        """Test the complete code analysis pipeline."""
        # Save sample code to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(sample_code)
            temp_file = f.name
        
        try:
            # Create analyzer
            analyzer = Analyzer()
            
            # Analyze code
            results = analyzer.analyze_code(temp_file)
            
            # Verify results structure
            assert 'static_analysis' in results
            assert 'dynamic_analysis' in results
            assert 'algorithm_recognition' in results
            assert 'inefficiency_detection' in results
            assert 'optimization_suggestions' in results
            
            # Verify static analysis results
            static_results = results['static_analysis']
            assert 'overall_time_complexity' in static_results
            assert 'functions' in static_results
            assert len(static_results['functions']) >= 4  # At least 4 functions in sample code
            
            # Verify algorithm recognition results
            algo_results = results['algorithm_recognition']
            assert len(algo_results) > 0
            assert any('linear_regression' in algo.lower() for algo in algo_results)
            assert any('neural_network' in algo.lower() for algo in algo_results)
            
            # Verify inefficiency detection results
            inefficiency_results = results['inefficiency_detection']
            assert len(inefficiency_results) > 0
            assert any('bubble_sort' in ineff.get('location', '') for ineff in inefficiency_results)
            
            # Verify optimization suggestions
            suggestions = results['optimization_suggestions']
            assert len(suggestions) > 0
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    @pandas_available
    def test_data_quality_pipeline(self, sample_data):
        """Test the data quality analysis pipeline."""
        df = sample_data['dataframe']
        
        # Create data guardian
        guardian = DataGuardian()
        
        # Generate report
        report = guardian.generate_report(df)
        
        # Verify report structure
        assert hasattr(report, 'missing_values')
        assert hasattr(report, 'duplicates')
        assert hasattr(report, 'outliers')
        assert hasattr(report, 'correlations')
        
        # Verify missing values detection
        assert report.missing_values['total_missing'] > 0
        assert 'name' in report.missing_values['missing_by_column']
        
        # Verify duplicate detection
        assert report.duplicates['total_duplicates'] > 0
        assert 'id' in report.duplicates['duplicate_columns']
        
        # Verify outlier detection
        assert 'age' in report.outliers['outliers_by_column']
        
        # Test data optimizer
        optimizer = DataOptimizer()
        
        # Get optimization suggestions
        suggestions = optimizer.suggest_optimizations(df)
        
        # Verify suggestions
        assert len(suggestions) > 0
        assert any('missing' in sugg.lower() for sugg in suggestions)
        assert any('duplicate' in sugg.lower() for sugg in suggestions)
        assert any('outlier' in sugg.lower() for sugg in suggestions)
        
        # Apply optimizations
        optimized_df = optimizer.optimize(df)
        
        # Verify optimized dataframe
        assert len(optimized_df) < len(df)  # Should have removed duplicates
        assert optimized_df.isna().sum().sum() < df.isna().sum().sum()  # Should have handled missing values
    
    @pytorch_available
    def test_pytorch_model_analysis_pipeline(self, pytorch_model, sample_data):
        """Test the PyTorch model analysis pipeline."""
        import torch
        
        X = sample_data['X']
        y = sample_data['y']
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Create model analyzer
        model_analyzer = ModelAnalyzer()
        
        # Analyze model
        analysis = model_analyzer.analyze_model(pytorch_model, framework="pytorch")
        
        # Verify analysis structure
        assert 'model_type' in analysis
        assert 'parameters' in analysis
        assert 'layers' in analysis
        assert 'complexity' in analysis
        
        # Verify model details
        assert analysis['model_type'] == 'neural_network'
        assert analysis['parameters'] > 0
        assert len(analysis['layers']) >= 2  # At least 2 layers
        
        # Create performance profiler
        profiler = PerformanceProfiler()
        
        # Profile model
        profile_results = profiler.profile_model(pytorch_model, X_tensor, framework="pytorch")
        
        # Verify profile results
        assert 'execution_time' in profile_results
        assert 'memory_usage' in profile_results
        assert 'throughput' in profile_results
        
        # Test vectorization analyzer
        vectorization_analyzer = VectorizationAnalyzer()
        
        # Create a simple function to analyze
        def process_data(data):
            result = []
            for item in data:
                result.append(item * 2)
            return result
        
        # Analyze vectorization
        vectorization_results = vectorization_analyzer.analyze_function(process_data)
        
        # Verify vectorization results
        assert 'naive_loops' in vectorization_results
        assert len(vectorization_results['naive_loops']) > 0
        assert 'vectorization_suggestions' in vectorization_results
        assert len(vectorization_results['vectorization_suggestions']) > 0
    
    @tensorflow_available
    def test_tensorflow_model_analysis_pipeline(self, tensorflow_model, sample_data):
        """Test the TensorFlow model analysis pipeline."""
        import tensorflow as tf
        
        X = sample_data['X']
        y = sample_data['y']
        
        # Convert data to TensorFlow tensors
        dataset = tf.data.Dataset.from_tensor_slices((X, y.reshape(-1, 1))).batch(32)
        
        # Create model analyzer
        model_analyzer = ModelAnalyzer()
        
        # Analyze model
        analysis = model_analyzer.analyze_model(tensorflow_model, framework="tensorflow")
        
        # Verify analysis structure
        assert 'model_type' in analysis
        assert 'parameters' in analysis
        assert 'layers' in analysis
        assert 'complexity' in analysis
        
        # Verify model details
        assert analysis['model_type'] == 'neural_network'
        assert analysis['parameters'] > 0
        assert len(analysis['layers']) >= 2  # At least 2 layers
        
        # Create performance profiler
        profiler = PerformanceProfiler()
        
        # Profile model
        profile_results = profiler.profile_model(tensorflow_model, X, framework="tensorflow")
        
        # Verify profile results
        assert 'execution_time' in profile_results
        assert 'memory_usage' in profile_results
        assert 'throughput' in profile_results
        
        # Test distributed training analyzer
        distributed_analyzer = DistributedTrainingAnalyzer()
        
        # Analyze distributed training potential
        distributed_results = distributed_analyzer.analyze_model(tensorflow_model, framework="tensorflow")
        
        # Verify distributed training results
        assert 'parallelizable_operations' in distributed_results
        assert 'distribution_strategy' in distributed_results
        assert 'scaling_estimate' in distributed_results
    
    def test_algorithm_recognition_standalone(self, sample_code):
        """Test the algorithm recognition module as a standalone component."""
        # Create algorithm recognizer
        recognizer = AlgorithmRecognizer()
        
        # Recognize algorithms in code
        algorithms = recognizer.recognize_algorithms(sample_code)
        
        # Verify recognized algorithms
        assert len(algorithms) > 0
        assert any('linear_regression' in algo.lower() for algo in algorithms)
        assert any('neural_network' in algo.lower() for algo in algorithms)
        
        # Get complexity information
        complexity_info = recognizer.get_complexity_info(algorithms)
        
        # Verify complexity information
        assert len(complexity_info) > 0
        assert any('time_complexity' in info for info in complexity_info.values())
        assert any('space_complexity' in info for info in complexity_info.values())
    
    def test_inefficiency_detection_standalone(self, sample_code):
        """Test the inefficiency detection module as a standalone component."""
        # Create inefficiency detector
        detector = InefficiencyDetector()
        
        # Detect inefficiencies in code
        inefficiencies = detector.detect_inefficiencies(sample_code)
        
        # Verify detected inefficiencies
        assert len(inefficiencies) > 0
        assert any('bubble_sort' in ineff.get('location', '') for ineff in inefficiencies)
        assert any('inefficient_function' in ineff.get('location', '') for ineff in inefficiencies)
        
        # Get optimization suggestions
        suggestions = detector.suggest_optimizations(inefficiencies)
        
        # Verify suggestions
        assert len(suggestions) > 0
        assert any('vectorization' in sugg.lower() for sugg in suggestions)
    
    def test_complexity_analyzer_standalone(self, sample_code):
        """Test the complexity analyzer module as a standalone component."""
        # Create complexity analyzer
        analyzer = ComplexityAnalyzer()
        
        # Analyze code complexity
        complexity = analyzer.analyze_code(sample_code)
        
        # Verify complexity analysis
        assert 'overall_time_complexity' in complexity
        assert 'overall_space_complexity' in complexity
        assert 'functions' in complexity
        assert len(complexity['functions']) >= 4  # At least 4 functions in sample code
        
        # Verify function-specific complexity
        assert any('bubble_sort' in func for func in complexity['functions'])
        bubble_sort_complexity = complexity['functions'].get('bubble_sort', {})
        assert bubble_sort_complexity.get('time_complexity') == 'O(n^2)'
    
    def test_roofline_analyzer(self, pytorch_model):
        """Test the roofline analyzer module."""
        # Create roofline analyzer
        analyzer = RooflineAnalyzer()
        
        # Create sample input
        import torch
        sample_input = torch.randn(32, 5)
        
        # Analyze model
        with patch.object(analyzer, '_measure_flops', return_value=1000000):
            with patch.object(analyzer, '_measure_memory_bandwidth', return_value=10000000000):
                with patch.object(analyzer, '_measure_arithmetic_intensity', return_value=0.1):
                    results = analyzer.analyze(pytorch_model, sample_input)
        
        # Verify results
        assert 'flops' in results
        assert 'memory_bandwidth' in results
        assert 'arithmetic_intensity' in results
        assert 'bottleneck' in results
        assert results['bottleneck'] in ['compute_bound', 'memory_bound']
        
        # Get optimization suggestions
        suggestions = analyzer.suggest_optimizations(results)
        
        # Verify suggestions
        assert len(suggestions) > 0
    
    def test_end_to_end_integration(self, sample_code, sample_data, pytorch_model):
        """Test complete end-to-end integration of all components."""
        # Save sample code to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(sample_code)
            temp_file = f.name
        
        try:
            # Create main analyzer
            analyzer = Analyzer()
            
            # Analyze code
            code_results = analyzer.analyze_code(temp_file)
            
            # Analyze data
            data_guardian = DataGuardian()
            data_report = data_guardian.generate_report(sample_data['dataframe'])
            
            # Analyze model
            model_analyzer = ModelAnalyzer()
            model_analysis = model_analyzer.analyze_model(pytorch_model, framework="pytorch")
            
            # Profile model performance
            profiler = PerformanceProfiler()
            import torch
            X_tensor = torch.FloatTensor(sample_data['X'])
            performance_results = profiler.profile_model(pytorch_model, X_tensor, framework="pytorch")
            
            # Combine all results
            combined_results = {
                'code_analysis': code_results,
                'data_quality': data_report.__dict__,
                'model_analysis': model_analysis,
                'performance_analysis': performance_results
            }
            
            # Verify combined results
            assert 'code_analysis' in combined_results
            assert 'data_quality' in combined_results
            assert 'model_analysis' in combined_results
            assert 'performance_analysis' in combined_results
            
            # Verify code analysis results
            assert 'static_analysis' in combined_results['code_analysis']
            assert 'algorithm_recognition' in combined_results['code_analysis']
            
            # Verify data quality results
            assert 'missing_values' in combined_results['data_quality']
            assert 'outliers' in combined_results['data_quality']
            
            # Verify model analysis results
            assert 'model_type' in combined_results['model_analysis']
            assert 'complexity' in combined_results['model_analysis']
            
            # Verify performance results
            assert 'execution_time' in combined_results['performance_analysis']
            assert 'memory_usage' in combined_results['performance_analysis']
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
