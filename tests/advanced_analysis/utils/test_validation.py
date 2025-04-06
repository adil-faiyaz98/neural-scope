"""
Tests for the validation module.
"""

import os
import pytest
import tempfile
from advanced_analysis.utils.validation import (
    validate_file_exists,
    validate_directory_exists,
    validate_file_extension,
    validate_framework,
    validate_techniques,
    validate_numeric_range,
    validate_string_pattern,
    validate_enum,
    sanitize_input,
    sanitize_file_path,
    ValidationError
)

def test_validate_file_exists():
    """Test validate_file_exists function."""
    # Test with stdin
    assert validate_file_exists("-") is True

    # Test with existing file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        assert validate_file_exists(temp_path) is True

        # Test with non-existent file
        with pytest.raises(ValidationError):
            validate_file_exists("non_existent_file.txt")
    finally:
        os.unlink(temp_path)

def test_validate_directory_exists():
    """Test validate_directory_exists function."""
    # Test with existing directory
    with tempfile.TemporaryDirectory() as temp_dir:
        assert validate_directory_exists(temp_dir) is True

        # Test with non-existent directory
        with pytest.raises(ValidationError):
            validate_directory_exists("non_existent_directory")

def test_validate_file_extension():
    """Test validate_file_extension function."""
    # Test with stdin
    assert validate_file_extension("-", [".py", ".txt"]) is True

    # Test with valid extension
    assert validate_file_extension("file.py", [".py", ".txt"]) is True

    # Test with invalid extension
    with pytest.raises(ValidationError):
        validate_file_extension("file.jpg", [".py", ".txt"])

def test_validate_framework():
    """Test validate_framework function."""
    # Test with valid framework
    assert validate_framework("pytorch", ["pytorch", "tensorflow", "sklearn"]) is True

    # Test with case-insensitive framework
    assert validate_framework("PyTorch", ["pytorch", "tensorflow", "sklearn"]) is True

    # Test with invalid framework
    with pytest.raises(ValidationError):
        validate_framework("keras", ["pytorch", "tensorflow", "sklearn"])

def test_validate_techniques():
    """Test validate_techniques function."""
    # Test with valid techniques
    assert validate_techniques(["quantization", "pruning"], ["quantization", "pruning", "distillation"]) is True

    # Test with case-insensitive techniques
    assert validate_techniques(["Quantization", "Pruning"], ["quantization", "pruning", "distillation"]) is True

    # Test with invalid technique
    with pytest.raises(ValidationError):
        validate_techniques(["quantization", "invalid"], ["quantization", "pruning", "distillation"])

def test_validate_numeric_range():
    """Test validate_numeric_range function."""
    # Test with valid range
    assert validate_numeric_range(5, 0, 10) is True

    # Test with min value
    assert validate_numeric_range(0, 0, 10) is True

    # Test with max value
    assert validate_numeric_range(10, 0, 10) is True

    # Test with value below min
    with pytest.raises(ValidationError):
        validate_numeric_range(-1, 0, 10)

    # Test with value above max
    with pytest.raises(ValidationError):
        validate_numeric_range(11, 0, 10)

def test_validate_string_pattern():
    """Test validate_string_pattern function."""
    # Test with valid pattern
    assert validate_string_pattern("abc123", r"^[a-z0-9]+$") is True

    # Test with invalid pattern
    with pytest.raises(ValidationError):
        validate_string_pattern("abc-123", r"^[a-z0-9]+$")

def test_validate_enum():
    """Test validate_enum function."""
    # Test with valid enum value
    assert validate_enum("a", ["a", "b", "c"]) is True

    # Test with invalid enum value
    with pytest.raises(ValidationError):
        validate_enum("d", ["a", "b", "c"])

def test_sanitize_input():
    """Test sanitize_input function."""
    # Test with safe input
    assert sanitize_input("safe_input") == "safe_input"

    # Test with unsafe input
    assert sanitize_input("unsafe;input") == "unsafeinput"

    # Test with path traversal
    with pytest.raises(ValidationError):
        sanitize_input("../unsafe/path")

def test_sanitize_file_path():
    """Test sanitize_file_path function."""
    # Test with safe path
    safe_path = sanitize_file_path("safe/path")
    assert "safe" in safe_path and ("path" in safe_path)

    # Test with path traversal
    with pytest.raises(ValidationError):
        sanitize_file_path("../unsafe/path")
