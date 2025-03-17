"""AI/ML Complexity Analysis Package.

This package provides tools to analyze the complexity and inefficiency of AI/ML code,
estimate computational cost (including mapping to cloud costs), and includes
profiling utilities and result storage functionalities.
"""

__version__ = "1.0.0"

# Set up a default logger for the package
import logging as _logging
logger = _logging.getLogger(__name__)
if not logger.handlers:
    # Configure default logging handler if none exist (avoid duplicate handlers)
    handler = _logging.StreamHandler()
    formatter = _logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(_logging.INFO)

# Expose key components for convenience
from aiml_complexity.analysis import analyze_code, analyze_file, analyze_directory, AnalysisResult
from cloud.aws_costs import estimate_cost
from aiml_complexity.profiling import profile_time, profile_memory
from aiml_complexity.storage import save_analysis, load_analysis
