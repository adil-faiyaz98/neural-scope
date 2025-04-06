"""
Tests for the error_handling module.
"""

import os
import pytest
import logging
import tempfile
# No mocks needed
from advanced_analysis.utils.error_handling import (
    AdvancedAnalysisError,
    DataError,
    DependencyError,
    FrameworkError,
    ModelError,
    AnalysisError,
    HardwareError,
    handle_errors,
    configure_file_logging,
    LoggingContext
)

def test_advanced_analysis_error():
    """Test AdvancedAnalysisError class."""
    # Test with message only
    error = AdvancedAnalysisError("Test error")
    assert str(error) == "Test error"
    assert error.details == {}

    # Test with details
    error = AdvancedAnalysisError("Test error", {"key": "value"})
    assert str(error) == "Test error"
    assert error.details == {"key": "value"}

    # Test with nested exception
    nested = ValueError("Nested error")
    error = AdvancedAnalysisError("Test error", {"nested": nested})
    assert str(error) == "Test error"
    assert "nested" in error.details
    assert error.details["nested"] == nested

def test_framework_error():
    """Test FrameworkError class."""
    error = FrameworkError("pytorch", "Framework error", {"config_file": "config.json"})
    assert str(error) == "Framework error"
    assert error.details["framework"] == "pytorch"
    assert error.details["config_file"] == "config.json"
    assert isinstance(error, AdvancedAnalysisError)

def test_data_error():
    """Test DataError class."""
    error = DataError("Data error", {"data_file": "data.csv"})
    assert str(error) == "Data error"
    assert error.details == {"data_file": "data.csv"}
    assert isinstance(error, AdvancedAnalysisError)

def test_model_error():
    """Test ModelError class."""
    error = ModelError("cnn", "Model error", {"model_file": "model.pt"})
    assert str(error) == "Model error"
    assert error.details["model_type"] == "cnn"
    assert error.details["model_file"] == "model.pt"
    assert isinstance(error, AdvancedAnalysisError)

def test_analysis_error():
    """Test AnalysisError class."""
    error = AnalysisError("performance", "Analysis error", {"metric": "flops"})
    assert str(error) == "Analysis error"
    assert error.details["analysis_type"] == "performance"
    assert error.details["metric"] == "flops"
    assert isinstance(error, AdvancedAnalysisError)

def test_hardware_error():
    """Test HardwareError class."""
    error = HardwareError("gpu", "Hardware error", {"device": "cuda:0"})
    assert str(error) == "Hardware error"
    assert error.details["hardware_type"] == "gpu"
    assert error.details["device"] == "cuda:0"
    assert isinstance(error, AdvancedAnalysisError)

def test_dependency_error():
    """Test DependencyError class."""
    error = DependencyError("numpy", "NumPy is required for this operation")
    assert str(error) == "NumPy is required for this operation"
    assert error.details == {"dependency": "numpy"}
    assert isinstance(error, AdvancedAnalysisError)

def test_handle_errors_decorator():
    """Test handle_errors decorator."""
    # Test successful function
    @handle_errors()
    def successful_function():
        return "success"

    assert successful_function() == "success"

    # Test function with error and reraise=True (default)
    @handle_errors(reraise=True)
    def error_function_reraise():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        error_function_reraise()

    # Test function with error and reraise=False
    @handle_errors(reraise=False)
    def error_function_no_reraise():
        raise ValueError("Test error")

    assert error_function_no_reraise() is None

    # Test function with expected exceptions
    @handle_errors(expected_exceptions=[ValueError], reraise=False)
    def error_function_expected():
        raise ValueError("Expected error")

    assert error_function_expected() is None

@pytest.mark.skip(reason="File locking issues on Windows")
def test_configure_file_logging():
    """Test configure_file_logging function."""
    # This test is skipped due to file locking issues on Windows
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test.log")

        # Configure logging
        configure_file_logging(log_file)

        # Get the logger
        logger = logging.getLogger("advanced_analysis")

        # Log a message
        logger.info("Test message")

        # Check if the log file exists
        assert os.path.exists(log_file)

        # Check if the message was logged
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Test message" in log_content

        # Remove all handlers to avoid file lock issues
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

def test_logging_context():
    """Test LoggingContext class."""
    # Create a real logger for testing
    test_logger = logging.getLogger("test_logger")
    original_level = test_logger.level

    # Test with default level
    with LoggingContext("test_logger"):
        pass

    # The level should be restored after exiting the context
    assert test_logger.level == original_level

    # Test with custom level
    with LoggingContext("test_logger", level=logging.DEBUG):
        assert test_logger.level == logging.DEBUG

    # The level should be restored after exiting the context
    assert test_logger.level == original_level
