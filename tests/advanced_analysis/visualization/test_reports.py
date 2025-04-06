"""
Tests for the report generation module.
"""

import pytest
import json
from advanced_analysis.visualization.reports import ReportGenerator

# Sample analysis results for testing
@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results for testing."""
    return {
        "complexity_analysis": {
            "theoretical_complexity": {
                "big_o": "O(n^2)",
                "big_theta": "Θ(n^2)",
                "big_omega": "Ω(n^2)",
                "space_complexity": "O(n)"
            },
            "empirical_performance": {
                "time_measurements": [
                    (10, 0.001),
                    (100, 0.01),
                    (1000, 0.1)
                ],
                "memory_measurements": [
                    (10, 1.0),
                    (100, 10.0),
                    (1000, 100.0)
                ]
            },
            "optimization_suggestions": [
                {
                    "type": "complexity",
                    "severity": "high",
                    "message": "Function has high time complexity: O(n^2)",
                    "details": "Consider using more efficient algorithms or data structures to reduce complexity."
                },
                {
                    "type": "memory",
                    "severity": "medium",
                    "message": "High memory usage: 100.00 MB for input size 1000",
                    "details": "Consider using more memory-efficient data structures or processing data in smaller chunks."
                }
            ]
        },
        "performance_analysis": {
            "execution_time": 0.1,
            "throughput": 1000,
            "memory_usage": 100.0,
            "bottlenecks": [
                "Nested loops in function process_data",
                "Inefficient memory allocation in function allocate_buffer"
            ],
            "optimization_suggestions": [
                "Replace nested loops with vectorized operations",
                "Use memory pooling to reduce allocation overhead"
            ]
        },
        "data_quality": {
            "completeness": 95.0,
            "uniqueness": 98.0,
            "consistency": 92.0,
            "accuracy": 97.0,
            "recommendations": [
                "Handle missing values in column 'age'",
                "Remove duplicate entries in column 'id'"
            ]
        }
    }

def test_report_generator_initialization():
    """Test that the ReportGenerator can be initialized."""
    generator = ReportGenerator()
    assert generator is not None
    assert generator.title == "Analysis Report"
    
    # Test with custom title
    custom_generator = ReportGenerator(title="Custom Report")
    assert custom_generator.title == "Custom Report"

def test_generate_text_report(sample_analysis_results):
    """Test generation of a text report."""
    generator = ReportGenerator()
    report = generator.generate_text_report(sample_analysis_results)
    
    # Check that the report is a string
    assert isinstance(report, str)
    
    # Check that the report contains expected sections
    assert "Complexity Analysis" in report
    assert "Performance Analysis" in report
    assert "Data Quality Assessment" in report
    
    # Check that the report contains specific information
    assert "O(n^2)" in report
    assert "execution_time" in report.lower() or "execution time" in report.lower()
    assert "completeness" in report.lower()

def test_generate_html_report(sample_analysis_results):
    """Test generation of an HTML report."""
    generator = ReportGenerator()
    report = generator.generate_html_report(sample_analysis_results)
    
    # Check that the report is a string
    assert isinstance(report, str)
    
    # Check that the report is valid HTML
    assert report.startswith("<!DOCTYPE html>") or report.startswith("<html>") or report.startswith("<html ")
    assert "</html>" in report
    
    # Check that the report contains expected sections
    assert "<h2>Complexity Analysis</h2>" in report or '<h2 class="' in report and "Complexity Analysis" in report
    assert "<h2>Performance Analysis</h2>" in report or '<h2 class="' in report and "Performance Analysis" in report
    assert "<h2>Data Quality Assessment</h2>" in report or '<h2 class="' in report and "Data Quality Assessment" in report
    
    # Check that the report contains specific information
    assert "O(n^2)" in report
    assert "execution_time" in report or "execution time" in report.lower()
    assert "completeness" in report.lower()

def test_save_html_report(sample_analysis_results, tmp_path):
    """Test saving an HTML report to a file."""
    generator = ReportGenerator()
    
    # Create a temporary file path
    file_path = tmp_path / "report.html"
    
    # Save the report
    generator.save_html_report(sample_analysis_results, str(file_path))
    
    # Check that the file exists
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert content.startswith("<!DOCTYPE html>") or content.startswith("<html>") or content.startswith("<html ")
    assert "</html>" in content
