"""
Neural-Scope Algorithm Complexity Analysis Module

This package provides tools for analyzing the algorithmic complexity of code,
including static and dynamic analysis of time and space complexity.

Key features:
1. Static code analysis to identify algorithmic patterns and estimate complexity
2. Dynamic profiling to measure empirical complexity across different input sizes
3. Comprehensive complexity analysis of functions with optimization recommendations

The module is organized into submodules for different aspects of complexity analysis.
"""

from advanced_analysis.algorithm_complexity.static_analyzer import StaticAnalyzer
from advanced_analysis.algorithm_complexity.dynamic_analyzer import DynamicAnalyzer, measure_runtime, measure_memory
from advanced_analysis.algorithm_complexity.complexity_analyzer import ComplexityAnalyzer

__all__ = [
    'StaticAnalyzer',
    'DynamicAnalyzer',
    'ComplexityAnalyzer',
    'measure_runtime',
    'measure_memory'
]
