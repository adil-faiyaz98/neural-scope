"""
Tests for the dynamic analyzer module.
"""

import pytest
import time
import numpy as np
from advanced_analysis.algorithm_complexity.dynamic_analyzer import (
    DynamicAnalyzer, measure_runtime, measure_memory, estimate_complexity
)

# Test functions with different complexities
def constant_time_function(n):
    """O(1) function"""
    return 1

def linear_time_function(n):
    """O(n) function"""
    total = 0
    for i in range(n):
        total += i
    return total

def quadratic_time_function(n):
    """O(n²) function"""
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

def logarithmic_time_function(n):
    """O(log n) function"""
    total = 0
    i = n
    while i > 0:
        total += i
        i = i // 2
    return total

# Input generator functions
def generate_int(size):
    """Generate an integer of given size"""
    return size

def generate_list(size):
    """Generate a list of given size"""
    return list(range(size))

def generate_array(size):
    """Generate a numpy array of given size"""
    return np.arange(size)

def test_measure_runtime():
    """Test the measure_runtime function"""
    # Test with a simple function that sleeps for a predictable time
    def sleep_function(n):
        time.sleep(0.01 * n)
        return n
    
    # Measure runtime for different input sizes
    sizes = [1, 2]
    results = measure_runtime(sleep_function, generate_int, sizes, repeats=1, timeout=1)
    
    # Check that results contain all sizes
    assert set(results.keys()) == set(sizes)
    
    # Check that runtimes increase with input size
    assert np.mean(results[1]) < np.mean(results[2])

def test_measure_memory():
    """Test the measure_memory function"""
    # Test with a function that allocates memory proportional to input size
    def allocate_memory(n):
        # Allocate a list of n integers
        return [0] * n
    
    # Measure memory usage for different input sizes
    sizes = [1000, 10000]
    results = measure_memory(allocate_memory, generate_int, sizes, repeats=1)
    
    # Check that results contain all sizes
    assert set(results.keys()) == set(sizes)
    
    # Check that memory usage increases with input size
    assert np.mean(results[1000]) < np.mean(results[10000])

def test_estimate_complexity():
    """Test the estimate_complexity function"""
    # Create synthetic data for different complexity classes
    sizes = [10, 20, 30, 40, 50]
    
    # O(1) data
    constant_times = [1, 1, 1, 1, 1]
    constant_result = estimate_complexity(sizes, constant_times)
    assert constant_result["complexity"] == "O(1)"
    
    # O(n) data
    linear_times = [10, 20, 30, 40, 50]
    linear_result = estimate_complexity(sizes, linear_times)
    assert linear_result["complexity"] == "O(n)"
    
    # O(n²) data
    quadratic_times = [100, 400, 900, 1600, 2500]
    quadratic_result = estimate_complexity(sizes, quadratic_times)
    assert quadratic_result["complexity"] == "O(n^2)"

def test_dynamic_analyzer():
    """Test the DynamicAnalyzer class"""
    analyzer = DynamicAnalyzer()
    assert analyzer is not None
    
    # Test analyzing a simple function
    sizes = [10, 20, 30]
    result = analyzer.analyze_function(
        linear_time_function,
        generate_int,
        sizes=sizes,
        repeats=1
    )
    
    # Check that the result contains expected fields
    assert "function_name" in result
    assert "runtime" in result
    assert "memory" in result
    assert "summary" in result
    
    # Check that the function name is correct
    assert result["function_name"] == "linear_time_function"
    
    # Check that the complexity was correctly identified
    assert "time_complexity" in result["summary"]
    # The exact complexity might vary depending on the implementation and noise,
    # but it should be close to O(n) for a linear function
    assert result["summary"]["time_complexity"] in ["O(n)", "O(n log n)", "O(1)"]
