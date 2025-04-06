"""
Main entry point for analyzing ML code.

This module provides the main entry point for analyzing ML code, including
static analysis, dynamic profiling, and optimization recommendations.
"""

# Version information
from advanced_analysis.version import __version__, get_component_version

import ast
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class Analyzer:
    """
    Main entry point for analyzing ML code.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer with optional configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components based on configuration
        self._init_components()

    def _init_components(self):
        """Initialize analysis components based on configuration."""
        # Import components here to avoid circular imports
        from advanced_analysis.algorithm_complexity import StaticAnalyzer, DynamicAnalyzer, ComplexityAnalyzer
        from advanced_analysis.ml_advisor import MLAdvisor, MLAlgorithmRecognizer
        from advanced_analysis.performance import ModelPerformanceProfiler, CPUVectorizationAnalyzer

        # Initialize static analysis components
        self.static_analyzer = StaticAnalyzer()

        # Initialize ML algorithm recognition
        self.algorithm_recognizer = MLAlgorithmRecognizer("")

        # Initialize ML advisor
        self.ml_advisor = None  # Will be initialized with model when needed

        # Initialize vectorization analyzer
        self.vectorization_analyzer = None  # Will be initialized with code when needed

        # Initialize complexity analyzer
        self.complexity_analyzer = None  # Will be initialized with function when needed

        # Initialize performance profiler
        self.performance_profiler = None  # Will be initialized with model when needed

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file for ML inefficiencies and optimization opportunities.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with analysis results
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Analyze code
        return self.analyze_code(code, context=file_path)

    def analyze_code(self, code: str, context: str = "unknown") -> Dict[str, Any]:
        """
        Analyze Python code for ML inefficiencies and optimization opportunities.

        Args:
            code: Python code as a string
            context: Context information (e.g., file path)

        Returns:
            Dictionary with analysis results
        """
        # Initialize results dictionary
        results = {
            "context": context,
            "static_analysis": None,
            "algorithm_recognition": None,
            "vectorization_analysis": None,
            "inefficiencies": [],
            "optimization_suggestions": [],
            "version": __version__,
            "component_versions": {
                "analyzer": get_component_version("core"),
                "algorithm_complexity": get_component_version("algorithm_complexity"),
                "ml_advisor": get_component_version("ml_advisor"),
                "performance": get_component_version("performance")
            }
        }

        # Validate input
        if not code or not isinstance(code, str):
            logger.error("Invalid code input: code must be a non-empty string")
            results["error"] = "Invalid code input: code must be a non-empty string"
            return results

        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            results["error"] = f"Syntax error: {e}"
            return results
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            results["error"] = f"Error parsing code: {str(e)}"
            return results

        # Perform static analysis
        try:
            static_results = self.static_analyzer.analyze_code(code)
            results["static_analysis"] = static_results
        except Exception as e:
            logger.error(f"Error in static analysis: {e}")
            results["errors"] = results.get("errors", []) + [{"component": "static_analysis", "error": str(e)}]
            # Continue with other analyses

        # Recognize ML algorithms
        try:
            from advanced_analysis.ml_advisor import MLAlgorithmRecognizer
            self.algorithm_recognizer = MLAlgorithmRecognizer(code)
            algorithm_results = self.algorithm_recognizer.analyze_code()
            results["algorithm_recognition"] = algorithm_results
        except Exception as e:
            logger.error(f"Error in algorithm recognition: {e}")
            results["errors"] = results.get("errors", []) + [{"component": "algorithm_recognition", "error": str(e)}]
            # Continue with other analyses

        # Analyze vectorization opportunities
        try:
            from advanced_analysis.performance import CPUVectorizationAnalyzer
            self.vectorization_analyzer = CPUVectorizationAnalyzer(code)
            vectorization_results = self.vectorization_analyzer.analyze_code()
            results["vectorization_analysis"] = vectorization_results
        except Exception as e:
            logger.error(f"Error in vectorization analysis: {e}")
            results["errors"] = results.get("errors", []) + [{"component": "vectorization_analysis", "error": str(e)}]
            # Continue with other analyses

        # Collect inefficiencies
        try:
            if results["static_analysis"] and "detected_patterns" in results["static_analysis"]:
                for pattern in results["static_analysis"]["detected_patterns"]:
                    results["inefficiencies"].append({
                        "type": "algorithmic_pattern",
                        "pattern": pattern["pattern"],
                        "time_complexity": pattern["time_complexity"],
                        "space_complexity": pattern["space_complexity"]
                    })
        except Exception as e:
            logger.error(f"Error collecting inefficiencies: {e}")
            results["errors"] = results.get("errors", []) + [{"component": "inefficiencies", "error": str(e)}]

        # Collect optimization suggestions
        try:
            if results["algorithm_recognition"] and "optimization_suggestions" in results["algorithm_recognition"]:
                results["optimization_suggestions"].extend(results["algorithm_recognition"]["optimization_suggestions"])

            if results["vectorization_analysis"] and "recommendations" in results["vectorization_analysis"]:
                results["optimization_suggestions"].extend(results["vectorization_analysis"]["recommendations"])
        except Exception as e:
            logger.error(f"Error collecting optimization suggestions: {e}")
            results["errors"] = results.get("errors", []) + [{"component": "optimization_suggestions", "error": str(e)}]

        # Add analysis timestamp
        import datetime
        results["timestamp"] = datetime.datetime.now().isoformat()

        return results

    def analyze_function(self, func) -> Dict[str, Any]:
        """
        Analyze a Python function for complexity and performance.

        Args:
            func: Python function to analyze

        Returns:
            Dictionary with analysis results
        """
        from advanced_analysis.algorithm_complexity import ComplexityAnalyzer

        # Initialize complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer(func)

        # Analyze function
        complexity_results = self.complexity_analyzer.analyze()

        return {
            "function_name": func.__name__,
            "complexity_analysis": complexity_results
        }

    def analyze_model(self, model, framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a machine learning model for performance and optimization opportunities.

        Args:
            model: Machine learning model to analyze
            framework: Framework name (e.g., 'pytorch', 'tensorflow')

        Returns:
            Dictionary with analysis results
        """
        from advanced_analysis.performance import ModelPerformanceProfiler
        from advanced_analysis.ml_advisor import MLAdvisor

        # Initialize performance profiler
        self.performance_profiler = ModelPerformanceProfiler(model, framework)

        # Initialize ML advisor
        self.ml_advisor = MLAdvisor(model, framework)

        # Analyze model
        performance_results = None
        try:
            # Note: This would require input data in a real implementation
            # performance_results = self.performance_profiler.profile(input_data)
            performance_results = {"message": "Performance profiling requires input data"}
        except Exception as e:
            logger.error(f"Error profiling model: {e}")
            performance_results = {"error": str(e)}

        # Get optimization suggestions
        advisor_results = self.ml_advisor.analyze_model()

        return {
            "model_analysis": {
                "framework": framework,
                "performance": performance_results,
                "advisor": advisor_results
            }
        }

    def generate_report(self, analysis_results: Dict[str, Any], format: str = "text") -> str:
        """
        Generate a report from analysis results.

        Args:
            analysis_results: Analysis results dictionary
            format: Report format ('text' or 'html')

        Returns:
            Report as a string
        """
        from advanced_analysis.visualization import ReportGenerator

        # Initialize report generator
        report_generator = ReportGenerator(title="ML Code Analysis Report")

        # Generate report
        if format.lower() == "html":
            return report_generator.generate_html_report(analysis_results)
        else:
            return report_generator.generate_text_report(analysis_results)

    def save_report(self, analysis_results: Dict[str, Any], filename: str, format: str = "html") -> None:
        """
        Generate a report and save it to a file.

        Args:
            analysis_results: Analysis results dictionary
            filename: Output filename
            format: Report format ('text' or 'html')
        """
        from advanced_analysis.visualization import ReportGenerator

        # Initialize report generator
        report_generator = ReportGenerator(title="ML Code Analysis Report")

        # Generate and save report
        if format.lower() == "html":
            report_generator.save_html_report(analysis_results, filename)
        else:
            report = report_generator.generate_text_report(analysis_results)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Text report saved to {filename}")
