"""
Tests for the data quality guardian module.
"""

import pytest
import pandas as pd
import numpy as np
from advanced_analysis.data_quality.guardian import DataGuardian, DataQualityReport

# Create sample data for testing
@pytest.fixture
def sample_data():
    """Create a sample DataFrame with various data quality issues."""
    # Create a DataFrame with missing values, duplicates, and outliers
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 7],  # Duplicate value in id
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],  # Missing value in name
        'age': [25, 30, 35, 40, 45, 50, 200],  # Outlier in age
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
    })
    return df

@pytest.fixture
def reference_data():
    """Create a reference DataFrame for drift detection."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR']
    })
    return df

def test_data_guardian_initialization():
    """Test that the DataGuardian can be initialized."""
    guardian = DataGuardian()
    assert guardian is not None

def test_analyze_basic(sample_data):
    """Test basic analysis functionality."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Check that the report is a DataQualityReport instance
    assert isinstance(report, DataQualityReport)

    # Check basic properties
    assert report.row_count == 7
    assert report.column_count == 5
    assert "id" in report.column_types
    assert "name" in report.column_types
    assert "age" in report.column_types
    assert "salary" in report.column_types
    assert "department" in report.column_types

def test_analyze_completeness(sample_data):
    """Test detection of missing values."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Check completeness (missing values)
    assert report.completeness is not None
    assert "name" in report.completeness
    assert report.completeness["name"] < 100.0  # Should be less than 100% complete

def test_analyze_uniqueness(sample_data):
    """Test detection of duplicate values."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Check uniqueness (duplicates)
    assert report.uniqueness is not None
    assert "id" in report.uniqueness
    assert report.uniqueness["id"] < 100.0  # Should be less than 100% unique

def test_detect_outliers(sample_data):
    """Test detection of outliers."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Check outliers
    assert report.outliers is not None
    assert "age" in report.outliers
    # The outlier is detected by index, not value
    assert len(report.outliers["age"]) > 0

def test_analyze_correlations(sample_data):
    """Test analysis of correlations."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Check correlations
    assert report.correlations is not None
    assert isinstance(report.correlations, dict)

    # Age and salary should have a positive correlation
    assert "age" in report.correlations
    assert "salary" in report.correlations["age"]
    assert report.correlations["age"]["salary"] > 0

def test_analyze_with_target(sample_data):
    """Test analysis with a target column."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data, target="salary")

    # Check that the report was generated successfully
    assert report is not None
    assert isinstance(report, DataQualityReport)

def test_analyze_with_protected_attributes(sample_data):
    """Test analysis with protected attributes."""
    guardian = DataGuardian()
    report = guardian.analyze(sample_data, target="salary", protected_attributes=["department"])

    # Check bias metrics
    assert report.bias_metrics is not None
    assert "department_distribution" in report.bias_metrics
    assert isinstance(report.bias_metrics["department_distribution"], dict)

def test_analyze_drift(sample_data, reference_data):
    """Test drift analysis."""
    guardian = DataGuardian()
    drift_results = guardian.analyze_drift(reference_data, sample_data)

    # Check drift results
    assert drift_results is not None
    assert isinstance(drift_results, dict)
    assert "schema_drift" in drift_results
    assert "distribution_drift" in drift_results

def test_schema_drift(sample_data):
    """Test schema drift detection."""
    guardian = DataGuardian()

    # Create a modified dataset with schema changes
    modified_data = sample_data.copy()
    modified_data["new_column"] = 1
    modified_data = modified_data.drop("department", axis=1)

    drift_results = guardian.analyze_drift(sample_data, modified_data)

    # Check schema drift
    assert drift_results["schema_drift"]["added_columns"] == ["new_column"]
    assert drift_results["schema_drift"]["removed_columns"] == ["department"]

def test_distribution_drift(sample_data):
    """Test distribution drift detection."""
    guardian = DataGuardian()

    # Create a modified dataset with distribution changes
    modified_data = sample_data.copy()
    modified_data["age"] = modified_data["age"] + 20  # Shift age distribution

    drift_results = guardian.analyze_drift(sample_data, modified_data)

    # Check distribution drift
    assert "distribution_drift" in drift_results
    assert isinstance(drift_results["distribution_drift"], dict)

def test_detect_feature_types(sample_data):
    """Test feature type detection."""
    guardian = DataGuardian()
    feature_types = guardian.detect_feature_types(sample_data)

    # Check feature types
    assert feature_types is not None
    assert isinstance(feature_types, dict)
    assert feature_types["id"] in ["numeric", "integer"]
    assert feature_types["name"] == "categorical"
    assert feature_types["age"] in ["numeric", "integer"]
    assert feature_types["salary"] in ["numeric", "integer"]
    assert feature_types["department"] == "categorical"

def test_validate_schema(sample_data):
    """Test schema validation."""
    guardian = DataGuardian()

    # Set schema expectations
    schema = {
        "id": {"type": "integer", "nullable": False},
        "name": {"type": "string", "nullable": True},
        "age": {"type": "integer", "min": 0, "max": 100},
        "salary": {"type": "integer", "min": 0},
        "department": {"type": "categorical", "allowed_values": ["HR", "IT", "Finance"]}
    }
    guardian.set_schema_expectations(schema)

    # Validate schema
    validation_results = guardian.validate_schema(sample_data)

    # Check validation results
    assert validation_results is not None
    assert isinstance(validation_results, dict)

def test_export_report_to_markdown(sample_data):
    """Test exporting report to markdown."""
    # Create a report
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Set the report on a guardian instance
    guardian.last_report = report

    # Export to markdown
    markdown = guardian.export_report_to_markdown()

    # Check markdown
    assert markdown is not None
    assert isinstance(markdown, str)
    assert "# Data Quality Report" in markdown

def test_export_report_to_html(sample_data):
    """Test exporting report to HTML."""
    # Create a report
    guardian = DataGuardian()
    report = guardian.analyze(sample_data)

    # Set the report on a guardian instance
    guardian.last_report = report

    # Export to HTML
    html = guardian.export_report_to_html()

    # Check HTML
    assert html is not None
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "<body" in html.lower()
