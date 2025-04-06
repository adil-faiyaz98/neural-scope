"""
Tests for the complexity analyzer module.
"""

import pytest
from advanced_analysis.algorithm_complexity.complexity_analyzer import ComplexityAnalyzer

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

def recursive_function(n):
    """O(2^n) function (naive recursive fibonacci)"""
    if n <= 1:
        return n
    return recursive_function(n-1) + recursive_function(n-2)

def test_complexity_analyzer_initialization():
    """Test that the ComplexityAnalyzer can be initialized with a function."""
    analyzer = ComplexityAnalyzer(constant_time_function)
    assert analyzer is not None
    assert analyzer.func == constant_time_function
    assert analyzer.func_name == "constant_time_function"

def test_static_complexity_analysis():
    """Test the static complexity analysis method."""
    # Test constant time function
    analyzer = ComplexityAnalyzer(constant_time_function)
    big_o, big_theta, big_omega = analyzer._static_complexity_analysis()
    assert big_o == "O(1)"
    
    # Test linear time function
    analyzer = ComplexityAnalyzer(linear_time_function)
    big_o, big_theta, big_omega = analyzer._static_complexity_analysis()
    assert big_o == "O(n)"
    
    # Test quadratic time function
    analyzer = ComplexityAnalyzer(quadratic_time_function)
    big_o, big_theta, big_omega = analyzer._static_complexity_analysis()
    assert big_o == "O(n^2)"
    
    # Test recursive function
    analyzer = ComplexityAnalyzer(recursive_function)
    big_o, big_theta, big_omega = analyzer._static_complexity_analysis()
    assert "O(2^n)" in big_o or "O(n) or O(2^n)" in big_o

def test_analyze_method():
    """Test the analyze method."""
    # Test with a simple function
    analyzer = ComplexityAnalyzer(linear_time_function)
    result = analyzer.analyze(inputs=[10, 100])
    
    # Check that the result contains expected fields
    assert "function_name" in result
    assert "theoretical_complexity" in result
    assert "empirical_performance" in result
    assert "optimization_suggestions" in result
    
    # Check that the function name is correct
    assert result["function_name"] == "linear_time_function"
    
    # Check that the theoretical complexity was correctly identified
    assert "big_o" in result["theoretical_complexity"]
    assert result["theoretical_complexity"]["big_o"] == "O(n)"
    
    # Check that empirical measurements were performed
    assert "time_measurements" in result["empirical_performance"]
    assert "memory_measurements" in result["empirical_performance"]
    assert len(result["empirical_performance"]["time_measurements"]) == 2
    assert len(result["empirical_performance"]["memory_measurements"]) == 2

def test_generate_suggestions():
    """Test the suggestion generation method."""
    # Test with a quadratic function (should generate suggestions for high complexity)
    analyzer = ComplexityAnalyzer(quadratic_time_function)
    theoretical = analyzer._static_complexity_analysis()
    time_data = [(10, 0.001), (100, 0.1)]  # Mock time data
    profile_stats = {}  # Mock profile stats
    memory_data = [(10, 1.0), (100, 10.0)]  # Mock memory data
    
    suggestions = analyzer._generate_suggestions(theoretical, time_data, profile_stats, memory_data)
    
    # Check that suggestions were generated
    assert len(suggestions) > 0
    
    # Check that at least one suggestion is about complexity
    complexity_suggestions = [s for s in suggestions if s["type"] == "complexity"]
    assert len(complexity_suggestions) > 0
    
    # Check that the suggestion mentions high complexity
    assert any("high" in s["message"].lower() for s in complexity_suggestions)

def test_highest_complexity():
    """Test the highest complexity determination method."""
    analyzer = ComplexityAnalyzer(constant_time_function)
    
    # Test with simple complexities
    assert analyzer._highest_complexity(["O(1)", "O(n)"]) == "O(n)"
    assert analyzer._highest_complexity(["O(n)", "O(n^2)"]) == "O(n^2)"
    assert analyzer._highest_complexity(["O(n^2)", "O(n log n)"]) == "O(n^2)"
    assert analyzer._highest_complexity(["O(n log n)", "O(n)"]) == "O(n log n)"
    
    # Test with uncertain complexities
    assert analyzer._highest_complexity(["O(n) or O(2^n)"]) == "O(2^n)"
