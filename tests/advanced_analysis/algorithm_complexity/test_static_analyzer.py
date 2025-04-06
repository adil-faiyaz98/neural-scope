"""
Tests for the static analyzer module.
"""

import pytest
from advanced_analysis.algorithm_complexity.static_analyzer import StaticAnalyzer

# Sample code for testing
SIMPLE_CODE = """
def simple_function(n):
    result = 0
    for i in range(n):
        result += i
    return result
"""

NESTED_LOOPS_CODE = """
def nested_function(n):
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result
"""

ML_CODE = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
"""

def test_static_analyzer_initialization():
    """Test that the StaticAnalyzer can be initialized."""
    analyzer = StaticAnalyzer()
    assert analyzer is not None

def test_analyze_simple_code():
    """Test analyzing simple code with a single loop."""
    analyzer = StaticAnalyzer()
    results = analyzer.analyze_code(SIMPLE_CODE)
    
    # Check that the analysis results contain expected fields
    assert "functions" in results
    assert "overall_time_complexity" in results
    
    # Check that the function was detected
    assert "simple_function" in results["functions"]
    
    # Check that the complexity was correctly identified (should be O(n) for a single loop)
    function_data = results["functions"]["simple_function"]
    assert function_data["max_loop_depth"] == 1
    assert "O(n)" in function_data["time_complexity"]

def test_analyze_nested_loops():
    """Test analyzing code with nested loops."""
    analyzer = StaticAnalyzer()
    results = analyzer.analyze_code(NESTED_LOOPS_CODE)
    
    # Check that the function was detected
    assert "nested_function" in results["functions"]
    
    # Check that the complexity was correctly identified (should be O(n²) for nested loops)
    function_data = results["functions"]["nested_function"]
    assert function_data["max_loop_depth"] == 2
    assert "O(n^2)" in function_data["time_complexity"] or "O(n²)" in function_data["time_complexity"]

def test_detect_ml_libraries():
    """Test detection of ML libraries in code."""
    analyzer = StaticAnalyzer()
    results = analyzer.analyze_code(ML_CODE)
    
    # Check that ML libraries were detected
    assert "ml_libraries_used" in results
    ml_libs = results["ml_libraries_used"]
    assert "numpy" in ml_libs or "np" in ml_libs
    assert "pandas" in ml_libs or "pd" in ml_libs
    assert "sklearn" in ml_libs

def test_detect_algorithmic_patterns():
    """Test detection of algorithmic patterns."""
    analyzer = StaticAnalyzer()
    
    # Code with linear search pattern
    linear_search_code = """
    def find_element(arr, target):
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    """
    
    results = analyzer.analyze_code(linear_search_code)
    
    # Check that patterns were detected
    assert "detected_patterns" in results
    
    # This might be implementation-dependent, but we expect at least one pattern
    # to be detected in a linear search implementation
    assert len(results["detected_patterns"]) > 0
