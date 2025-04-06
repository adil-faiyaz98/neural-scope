"""
Tests for the CPU vectorization analysis module.
"""

import pytest
import time
import numpy as np
from advanced_analysis.performance.vectorization import CPUVectorizationAnalyzer, CPUUtilizationMonitor

# Sample code snippets for testing
VECTORIZABLE_CODE = """
def process_array(arr):
    result = []
    for i in range(len(arr)):
        result.append(arr[i] * 2 + 1)
    return result
"""

NESTED_LOOPS_CODE = """
def compute_matrix(n):
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(i * j)
        result.append(row)
    return result
"""

NUMPY_ITERATION_CODE = """
import numpy as np

def process_numpy_array(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2
    return result
"""

VECTORIZED_CODE = """
import numpy as np

def process_array_vectorized(arr):
    return arr * 2 + 1
"""

def test_cpu_vectorization_analyzer_initialization():
    """Test that the CPUVectorizationAnalyzer can be initialized."""
    analyzer = CPUVectorizationAnalyzer(VECTORIZABLE_CODE)
    assert analyzer is not None

def test_find_naive_loops():
    """Test finding naive loops that could be vectorized."""
    analyzer = CPUVectorizationAnalyzer(VECTORIZABLE_CODE)
    results = analyzer.analyze_code()
    
    # Check that the results contain expected fields
    assert "naive_loops" in results
    assert "recommendations" in results
    
    # Check that naive loops were found
    assert len(results["naive_loops"]) > 0
    
    # Check that the loop is marked as vectorizable
    assert results["naive_loops"][0]["vectorizable"]

def test_find_nested_loops():
    """Test finding nested loops that could be vectorized."""
    analyzer = CPUVectorizationAnalyzer(NESTED_LOOPS_CODE)
    results = analyzer.analyze_code()
    
    # Check that naive loops were found
    assert len(results["naive_loops"]) > 0
    
    # Check that at least one loop has depth > 1
    assert any(loop["depth"] > 1 for loop in results["naive_loops"])

def test_find_numpy_iteration():
    """Test finding numpy array iteration that could be vectorized."""
    analyzer = CPUVectorizationAnalyzer(NUMPY_ITERATION_CODE)
    results = analyzer.analyze_code()
    
    # Check that naive loops were found
    assert len(results["naive_loops"]) > 0
    
    # Check that recommendations were generated
    assert len(results["recommendations"]) > 0

def test_no_loops_in_vectorized_code():
    """Test that no naive loops are found in already vectorized code."""
    analyzer = CPUVectorizationAnalyzer(VECTORIZED_CODE)
    results = analyzer.analyze_code()
    
    # Check that no naive loops were found
    assert len(results["naive_loops"]) == 0

def test_generate_recommendations():
    """Test generation of vectorization recommendations."""
    analyzer = CPUVectorizationAnalyzer(VECTORIZABLE_CODE)
    results = analyzer.analyze_code()
    
    # Check that recommendations were generated
    assert len(results["recommendations"]) > 0
    
    # Check that recommendations contain expected fields
    for recommendation in results["recommendations"]:
        assert "type" in recommendation
        assert "severity" in recommendation
        assert "message" in recommendation
        assert "details" in recommendation
        assert "code_example" in recommendation

def test_cpu_utilization_monitor():
    """Test the CPU utilization monitor."""
    # Skip if psutil is not available
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        pytest.skip("psutil not available")
    
    monitor = CPUUtilizationMonitor()
    
    # Define a CPU-intensive function to monitor
    def cpu_intensive_function():
        result = 0
        for i in range(1000000):
            result += i
        return result
    
    # Monitor the function
    monitoring_results, result = monitor.monitor_function(cpu_intensive_function)
    
    # Check that the monitoring results contain expected fields
    assert "success" in monitoring_results
    assert "execution_time" in monitoring_results
    assert "cpu_utilization" in monitoring_results
    
    # Check that the function executed successfully
    assert monitoring_results["success"]
    assert monitoring_results["execution_time"] > 0
    
    # Check that CPU utilization was measured
    assert "average" in monitoring_results["cpu_utilization"]
    assert "maximum" in monitoring_results["cpu_utilization"]
    assert "samples" in monitoring_results["cpu_utilization"]
    
    # Check that the function result was returned
    assert result == sum(range(1000000))
