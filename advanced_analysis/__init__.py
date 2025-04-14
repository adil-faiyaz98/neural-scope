"""
Neural-Scope Advanced Analysis Package

This package provides advanced analysis tools for machine learning models and data,
including performance optimization, data quality assessment, and cloud cost analysis.

The package is organized into submodules:
- performance: Tools for analyzing and optimizing model performance
- data_quality: Tools for assessing and improving data quality
- cloud: Tools for cloud-based profiling and cost optimization
- ml_advisor: ML-based analysis and optimization recommendations
- algorithm_complexity: Tools for analyzing algorithmic complexity
- visualization: Tools for visualizing analysis results
- mlops: Tools for integrating with MLOps platforms and CI/CD pipelines
"""

# Version information
from advanced_analysis.version import __version__, __version_info__, get_version, get_version_info

# Performance module
from advanced_analysis.performance import ModelPerformanceProfiler
from advanced_analysis.performance import DistributedTrainingAnalyzer, ScalingEfficiencyResult, MultiGPUProfiler
from advanced_analysis.performance import RooflineAnalyzer, RooflineResult
from advanced_analysis.performance import CPUVectorizationAnalyzer, CPUUtilizationMonitor

# Data quality module
from advanced_analysis.data_quality import DataGuardian, DataQualityReport
from advanced_analysis.data_quality import DataLoaderOptimizer, DataLoaderProfilingResult

# Cloud module
from advanced_analysis.cloud import CloudProfiler
from advanced_analysis.cloud import CloudCostOptimizer, CloudCostAnalysisResult

# ML advisor module
from advanced_analysis.ml_advisor import MLAdvisor, OptimizationSuggestion
from advanced_analysis.ml_advisor import MLAlgorithmRecognizer, InefficiencyDetector
from advanced_analysis.analyzer import Analyzer

# Algorithm complexity module
from advanced_analysis.algorithm_complexity import StaticAnalyzer, DynamicAnalyzer, ComplexityAnalyzer
from advanced_analysis.algorithm_complexity import measure_runtime, measure_memory

# Visualization module
from advanced_analysis.visualization import PerformanceDashboard, DataQualityDashboard
from advanced_analysis.visualization import ReportGenerator

# MLOps module
from advanced_analysis.mlops import CICDIntegrator, MLflowIntegrator, KubeflowIntegrator

__all__ = [
    # Version information
    '__version__',
    '__version_info__',
    'get_version',
    'get_version_info',

    # Performance
    'ModelPerformanceProfiler',
    'DistributedTrainingAnalyzer',
    'ScalingEfficiencyResult',
    'MultiGPUProfiler',
    'RooflineAnalyzer',
    'RooflineResult',
    'CPUVectorizationAnalyzer',
    'CPUUtilizationMonitor',

    # Data quality
    'DataGuardian',
    'DataQualityReport',
    'DataLoaderOptimizer',
    'DataLoaderProfilingResult',

    # Cloud
    'CloudProfiler',
    'CloudCostOptimizer',
    'CloudCostAnalysisResult',

    # ML advisor
    'MLAdvisor',
    'OptimizationSuggestion',
    'MLAlgorithmRecognizer',
    'InefficiencyDetector',
    'Analyzer',

    # Algorithm complexity
    'StaticAnalyzer',
    'DynamicAnalyzer',
    'ComplexityAnalyzer',
    'measure_runtime',
    'measure_memory',

    # Visualization
    'PerformanceDashboard',
    'DataQualityDashboard',
    'ReportGenerator',
    
    # MLOps
    'CICDIntegrator',
    'MLflowIntegrator',
    'KubeflowIntegrator'
]
