"""
Tests for the dashboard visualization module.
"""

import pytest
import datetime
from advanced_analysis.visualization.dashboards import PerformanceDashboard, DataQualityDashboard

# Skip tests if Plotly is not available
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Skip tests if Dash is not available
try:
    import dash
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PLOTLY_AVAILABLE or not DASH_AVAILABLE, 
                               reason="Plotly or Dash not available")

# Sample data for testing
@pytest.fixture
def performance_data():
    """Create sample performance data for testing."""
    return [
        {
            "model_name": "ResNet50",
            "batch_size": 32,
            "framework": "pytorch",
            "device": "cuda",
            "throughput": 120.5,
            "latency_ms": 8.3,
            "memory_mb": 2048.0,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=5)
        },
        {
            "model_name": "ResNet50",
            "batch_size": 64,
            "framework": "pytorch",
            "device": "cuda",
            "throughput": 210.2,
            "latency_ms": 9.5,
            "memory_mb": 3072.0,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=3)
        },
        {
            "model_name": "MobileNetV2",
            "batch_size": 32,
            "framework": "pytorch",
            "device": "cuda",
            "throughput": 350.1,
            "latency_ms": 2.8,
            "memory_mb": 1024.0,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=5)
        },
        {
            "model_name": "MobileNetV2",
            "batch_size": 64,
            "framework": "pytorch",
            "device": "cuda",
            "throughput": 620.5,
            "latency_ms": 3.2,
            "memory_mb": 1536.0,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=3)
        }
    ]

@pytest.fixture
def data_quality_data():
    """Create sample data quality metrics for testing."""
    return [
        {
            "dataset_name": "train",
            "completeness": 0.95,
            "uniqueness": 0.98,
            "consistency": 0.92,
            "accuracy": 0.97,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=5)
        },
        {
            "dataset_name": "train",
            "completeness": 0.96,
            "uniqueness": 0.99,
            "consistency": 0.93,
            "accuracy": 0.98,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=3)
        },
        {
            "dataset_name": "validation",
            "completeness": 0.94,
            "uniqueness": 0.97,
            "consistency": 0.91,
            "accuracy": 0.96,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=5)
        },
        {
            "dataset_name": "validation",
            "completeness": 0.95,
            "uniqueness": 0.98,
            "consistency": 0.92,
            "accuracy": 0.97,
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=3)
        }
    ]

def test_performance_dashboard_initialization():
    """Test that the PerformanceDashboard can be initialized."""
    dashboard = PerformanceDashboard()
    assert dashboard is not None
    assert dashboard.title == "Performance Dashboard"
    
    # Test with custom title
    custom_dashboard = PerformanceDashboard(title="Custom Dashboard")
    assert custom_dashboard.title == "Custom Dashboard"

def test_performance_dashboard_add_data(performance_data):
    """Test adding data to the PerformanceDashboard."""
    dashboard = PerformanceDashboard()
    dashboard.add_data(performance_data)
    
    # Check that data was added
    assert len(dashboard.data) == len(performance_data)
    assert dashboard.data == performance_data

def test_performance_dashboard_create_static_dashboard(performance_data):
    """Test creation of a static performance dashboard."""
    dashboard = PerformanceDashboard()
    dashboard.add_data(performance_data)
    figures = dashboard.create_static_dashboard()
    
    # Check that figures were created
    assert isinstance(figures, dict)
    assert "throughput" in figures
    assert "latency" in figures
    assert "memory" in figures
    
    # Check that each figure is a Plotly figure
    for fig_name, fig in figures.items():
        assert hasattr(fig, "update_layout")

def test_data_quality_dashboard_initialization():
    """Test that the DataQualityDashboard can be initialized."""
    dashboard = DataQualityDashboard()
    assert dashboard is not None
    assert dashboard.title == "Data Quality Dashboard"
    
    # Test with custom title
    custom_dashboard = DataQualityDashboard(title="Custom Dashboard")
    assert custom_dashboard.title == "Custom Dashboard"

def test_data_quality_dashboard_add_data(data_quality_data):
    """Test adding data to the DataQualityDashboard."""
    dashboard = DataQualityDashboard()
    dashboard.add_data(data_quality_data)
    
    # Check that data was added
    assert len(dashboard.data) == len(data_quality_data)
    assert dashboard.data == data_quality_data

def test_data_quality_dashboard_create_static_dashboard(data_quality_data):
    """Test creation of a static data quality dashboard."""
    dashboard = DataQualityDashboard()
    dashboard.add_data(data_quality_data)
    figures = dashboard.create_static_dashboard()
    
    # Check that figures were created
    assert isinstance(figures, dict)
    assert "radar" in figures
    assert "time_series" in figures
    
    # Check that each figure is a Plotly figure
    for fig_name, fig in figures.items():
        assert hasattr(fig, "update_layout")
