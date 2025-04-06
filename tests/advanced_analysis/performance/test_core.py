"""
Tests for the performance core module.
"""

import pytest
import time
import numpy as np
from advanced_analysis.performance.core import ModelPerformanceProfiler

# Skip tests if PyTorch is not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

# Create a simple PyTorch model for testing
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

@pytest.fixture
def sample_model():
    """Create a sample PyTorch model for testing."""
    return SimpleModel()

@pytest.fixture
def sample_input():
    """Create sample input data for testing."""
    return torch.randn(32, 10)  # Batch size 32, input dimension 10

def test_model_performance_profiler_initialization():
    """Test that the ModelPerformanceProfiler can be initialized."""
    profiler = ModelPerformanceProfiler()
    assert profiler is not None

def test_model_performance_profiler_with_model(sample_model):
    """Test ModelPerformanceProfiler with a PyTorch model."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    assert profiler.model is sample_model
    assert profiler.framework == "pytorch"

def test_profile_inference_time(sample_model, sample_input):
    """Test profiling of inference time."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    result = profiler.profile_inference_time(sample_input, num_runs=10)
    
    # Check that the result contains expected fields
    assert "avg_inference_time" in result
    assert "min_inference_time" in result
    assert "max_inference_time" in result
    assert "total_time" in result
    assert "num_runs" in result
    
    # Check that the values are reasonable
    assert result["num_runs"] == 10
    assert result["avg_inference_time"] > 0
    assert result["min_inference_time"] <= result["avg_inference_time"]
    assert result["max_inference_time"] >= result["avg_inference_time"]
    assert result["total_time"] >= result["avg_inference_time"] * 10

def test_profile_memory_usage(sample_model, sample_input):
    """Test profiling of memory usage."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    result = profiler.profile_memory_usage(sample_input)
    
    # Check that the result contains expected fields
    assert "peak_memory_mb" in result
    assert "model_size_mb" in result
    assert "input_size_mb" in result
    assert "output_size_mb" in result
    
    # Check that the values are reasonable
    assert result["peak_memory_mb"] > 0
    assert result["model_size_mb"] > 0
    assert result["input_size_mb"] > 0
    assert result["output_size_mb"] > 0

def test_profile_throughput(sample_model, sample_input):
    """Test profiling of throughput."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    result = profiler.profile_throughput(sample_input, batch_size=32, num_batches=5)
    
    # Check that the result contains expected fields
    assert "throughput" in result
    assert "batch_size" in result
    assert "num_batches" in result
    assert "total_samples" in result
    assert "total_time" in result
    
    # Check that the values are reasonable
    assert result["batch_size"] == 32
    assert result["num_batches"] == 5
    assert result["total_samples"] == 32 * 5
    assert result["throughput"] > 0
    assert result["total_time"] > 0

def test_profile_model(sample_model, sample_input):
    """Test comprehensive model profiling."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    result = profiler.profile_model(sample_input)
    
    # Check that the result contains expected sections
    assert "inference_time" in result
    assert "memory_usage" in result
    assert "throughput" in result
    assert "model_summary" in result
    
    # Check that each section contains expected fields
    assert "avg_inference_time" in result["inference_time"]
    assert "peak_memory_mb" in result["memory_usage"]
    assert "throughput" in result["throughput"]
    assert "total_parameters" in result["model_summary"]

def test_generate_optimization_recommendations(sample_model, sample_input):
    """Test generation of optimization recommendations."""
    profiler = ModelPerformanceProfiler(model=sample_model, framework="pytorch")
    profiler.profile_model(sample_input)
    recommendations = profiler.generate_optimization_recommendations()
    
    # Check that recommendations were generated
    assert len(recommendations) > 0
    
    # Check that recommendations contain expected fields
    for recommendation in recommendations:
        assert "category" in recommendation
        assert "description" in recommendation
        assert "impact" in recommendation
