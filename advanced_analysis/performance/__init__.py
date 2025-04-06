"""
Neural-Scope Performance Optimization Module

This package provides comprehensive tools for analyzing and optimizing
the performance of machine learning models, including:

1. Basic performance profiling
2. Advanced optimization suggestions
3. Distributed training analysis
4. Hardware-specific optimizations

The module is organized into submodules for different aspects of performance optimization.
"""

from advanced_analysis.performance.core import ModelPerformanceProfiler
from advanced_analysis.performance.distributed import DistributedTrainingAnalyzer, ScalingEfficiencyResult, MultiGPUProfiler
from advanced_analysis.performance.roofline import RooflineAnalyzer, RooflineResult
from advanced_analysis.performance.vectorization import CPUVectorizationAnalyzer, CPUUtilizationMonitor
