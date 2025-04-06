"""
Run all tests for the advanced_analysis module.
"""

import pytest
import sys

if __name__ == "__main__":
    # Run all tests in the tests/advanced_analysis directory
    exit_code = pytest.main(["-xvs", "tests/advanced_analysis"])
    sys.exit(exit_code)
