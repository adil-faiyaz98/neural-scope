"""
Neural-Scope Visualization Module

This package provides tools for visualizing analysis results and creating
interactive dashboards for machine learning performance and data quality metrics.

Key features:
1. Interactive dashboards for performance metrics
2. Data quality visualization
3. Complexity analysis visualization
4. Cloud cost visualization
5. Report generation (text and HTML)

The module is organized into submodules for different aspects of visualization.
"""

from advanced_analysis.visualization.dashboards import PerformanceDashboard, DataQualityDashboard
from advanced_analysis.visualization.reports import ReportGenerator

__all__ = [
    'PerformanceDashboard',
    'DataQualityDashboard',
    'ReportGenerator'
]
