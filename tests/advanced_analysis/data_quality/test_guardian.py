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

def test_data_guardian_initialization():
    """Test that the DataGuardian can be initialized."""
    guardian = DataGuardian()
    assert guardian is not None

def test_check_missing_values(sample_data):
    """Test detection of missing values."""
    guardian = DataGuardian()
    report = guardian.check_missing_values(sample_data)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "total_missing" in report
    assert "missing_by_column" in report
    
    # Check that the missing value in 'name' was detected
    assert report["total_missing"] == 1
    assert report["missing_by_column"]["name"] == 1

def test_check_duplicates(sample_data):
    """Test detection of duplicate values."""
    guardian = DataGuardian()
    report = guardian.check_duplicates(sample_data)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "total_duplicates" in report
    assert "duplicate_rows" in report
    
    # Check that the duplicate value in 'id' was detected
    assert report["total_duplicates"] == 1

def test_check_outliers(sample_data):
    """Test detection of outliers."""
    guardian = DataGuardian()
    report = guardian.check_outliers(sample_data, columns=['age', 'salary'])
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "outliers_by_column" in report
    
    # Check that the outlier in 'age' was detected
    assert "age" in report["outliers_by_column"]
    assert len(report["outliers_by_column"]["age"]) > 0
    
    # The outlier value should be 200
    assert 200 in report["outliers_by_column"]["age"].values

def test_check_data_types(sample_data):
    """Test detection of data type issues."""
    guardian = DataGuardian()
    report = guardian.check_data_types(sample_data)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "column_types" in report
    
    # Check that the data types were correctly identified
    assert report["column_types"]["id"] == "int64"
    assert report["column_types"]["name"] == "object"
    assert report["column_types"]["age"] == "int64"
    assert report["column_types"]["salary"] == "int64"
    assert report["column_types"]["department"] == "object"

def test_check_value_distribution(sample_data):
    """Test analysis of value distribution."""
    guardian = DataGuardian()
    report = guardian.check_value_distribution(sample_data, columns=['department'])
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "distribution_by_column" in report
    assert "department" in report["distribution_by_column"]
    
    # Check that the distribution was correctly calculated
    dept_dist = report["distribution_by_column"]["department"]
    assert dept_dist["IT"] == 3
    assert dept_dist["HR"] == 2
    assert dept_dist["Finance"] == 2

def test_generate_report(sample_data):
    """Test generation of a comprehensive data quality report."""
    guardian = DataGuardian()
    report = guardian.generate_report(sample_data)
    
    # Check that the report is a DataQualityReport instance
    assert isinstance(report, DataQualityReport)
    
    # Check that the report contains all expected sections
    assert hasattr(report, "missing_values")
    assert hasattr(report, "duplicates")
    assert hasattr(report, "outliers")
    assert hasattr(report, "data_types")
    assert hasattr(report, "value_distribution")
    
    # Check that the report contains the correct data
    assert report.missing_values["total_missing"] == 1
    assert report.duplicates["total_duplicates"] == 1
    assert "age" in report.outliers["outliers_by_column"]
    assert report.data_types["column_types"]["id"] == "int64"
    assert "department" in report.value_distribution["distribution_by_column"]

def test_data_quality_report_to_dict(sample_data):
    """Test conversion of DataQualityReport to dictionary."""
    guardian = DataGuardian()
    report = guardian.generate_report(sample_data)
    report_dict = report.to_dict()
    
    # Check that the dictionary contains all expected sections
    assert "missing_values" in report_dict
    assert "duplicates" in report_dict
    assert "outliers" in report_dict
    assert "data_types" in report_dict
    assert "value_distribution" in report_dict

def test_data_quality_report_to_json(sample_data):
    """Test conversion of DataQualityReport to JSON."""
    guardian = DataGuardian()
    report = guardian.generate_report(sample_data)
    json_str = report.to_json()
    
    # Check that the JSON string is not empty
    assert json_str is not None
    assert len(json_str) > 0
    
    # Check that the JSON string starts with a curly brace
    assert json_str.startswith("{")
    assert json_str.endswith("}")

def test_data_quality_report_to_html(sample_data):
    """Test conversion of DataQualityReport to HTML."""
    guardian = DataGuardian()
    report = guardian.generate_report(sample_data)
    html_str = report.to_html()
    
    # Check that the HTML string is not empty
    assert html_str is not None
    assert len(html_str) > 0
    
    # Check that the HTML string contains expected tags
    assert "<html" in html_str.lower()
    assert "<body" in html_str.lower()
    assert "<table" in html_str.lower()
