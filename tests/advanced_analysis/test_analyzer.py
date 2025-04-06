"""
Tests for the main analyzer module.
"""

import pytest
import os
import tempfile
from advanced_analysis.analyzer import Analyzer

# Sample code for testing
SAMPLE_CODE = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result

def nested_function(n):
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(i * j)
        result.append(row)
    return result
"""

# Sample function for testing
def sample_function(n):
    """A sample function with O(n) complexity."""
    result = 0
    for i in range(n):
        result += i
    return result

def test_analyzer_initialization():
    """Test that the Analyzer can be initialized."""
    analyzer = Analyzer()
    assert analyzer is not None

def test_analyzer_with_config():
    """Test Analyzer initialization with configuration."""
    config = {"verbose": True, "max_depth": 3}
    analyzer = Analyzer(config=config)
    assert analyzer.config == config

def test_analyze_code():
    """Test analyzing Python code."""
    analyzer = Analyzer()
    results = analyzer.analyze_code(SAMPLE_CODE)
    
    # Check that the results contain expected fields
    assert "context" in results
    assert "static_analysis" in results
    assert "algorithm_recognition" in results
    assert "vectorization_analysis" in results
    assert "inefficiencies" in results
    assert "optimization_suggestions" in results
    
    # Check that static analysis was performed
    assert results["static_analysis"] is not None
    
    # Check that inefficiencies were detected
    assert len(results["inefficiencies"]) > 0
    
    # Check that optimization suggestions were generated
    assert len(results["optimization_suggestions"]) > 0

def test_analyze_file():
    """Test analyzing a Python file."""
    # Create a temporary file with sample code
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file.write(SAMPLE_CODE.encode())
        temp_file_path = temp_file.name
    
    try:
        analyzer = Analyzer()
        results = analyzer.analyze_file(temp_file_path)
        
        # Check that the results contain expected fields
        assert "context" in results
        assert "static_analysis" in results
        assert "algorithm_recognition" in results
        assert "vectorization_analysis" in results
        assert "inefficiencies" in results
        assert "optimization_suggestions" in results
        
        # Check that the context is the file path
        assert results["context"] == temp_file_path
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def test_analyze_function():
    """Test analyzing a Python function."""
    analyzer = Analyzer()
    results = analyzer.analyze_function(sample_function)
    
    # Check that the results contain expected fields
    assert "function_name" in results
    assert "complexity_analysis" in results
    
    # Check that the function name is correct
    assert results["function_name"] == "sample_function"
    
    # Check that complexity analysis was performed
    assert "theoretical_complexity" in results["complexity_analysis"]
    assert "empirical_performance" in results["complexity_analysis"]
    assert "optimization_suggestions" in results["complexity_analysis"]

def test_generate_report():
    """Test generating a report from analysis results."""
    analyzer = Analyzer()
    results = analyzer.analyze_code(SAMPLE_CODE)
    
    # Generate text report
    text_report = analyzer.generate_report(results, format="text")
    assert isinstance(text_report, str)
    assert len(text_report) > 0
    
    # Generate HTML report
    html_report = analyzer.generate_report(results, format="html")
    assert isinstance(html_report, str)
    assert len(html_report) > 0
    assert html_report.startswith("<!DOCTYPE html>") or html_report.startswith("<html>") or html_report.startswith("<html ")

def test_save_report(tmp_path):
    """Test saving a report to a file."""
    analyzer = Analyzer()
    results = analyzer.analyze_code(SAMPLE_CODE)
    
    # Create a temporary file path
    file_path = tmp_path / "report.html"
    
    # Save the report
    analyzer.save_report(results, str(file_path), format="html")
    
    # Check that the file exists
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert content.startswith("<!DOCTYPE html>") or content.startswith("<html>") or content.startswith("<html ")
