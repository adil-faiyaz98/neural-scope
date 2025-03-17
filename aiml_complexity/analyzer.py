# prod_ready_aiml_complexity/analyzer.py

import ast
import logging
import inspect
import os
from typing import Any, Dict, List, Optional

from .detectors import AIMLVisitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ComplexityAnalyzer:
    """
    Main entry point for analyzing Python code for AI/ML inefficiencies.
    Handles:
      - AST-based checks (using AIMLVisitor)
      - (Optional) dynamic profiling calls
    """

    def __init__(self, analyze_runtime: bool = False):
        self.analyze_runtime = analyze_runtime
        self.inefficiencies: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze all functions/statements in a Python file.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        logger.info("Analyzing file: %s", file_path)
        tree = ast.parse(code)
        self._run_ast_analysis(tree, context=file_path)

        # (Optional) dynamic analysis
        if self.analyze_runtime:
            logger.info("Runtime analysis placeholder: not fully implemented yet.")

        return {
            "file": file_path,
            "inefficiencies": self.inefficiencies,
            "suggestions": self.suggestions
        }

    def analyze_function(self, func) -> Dict[str, Any]:
        """
        Analyze a specific function object in memory.
        """
        src = inspect.getsource(func)
        context = func.__name__
        logger.info("Analyzing function object: %s", context)
        tree = ast.parse(src)
        self._run_ast_analysis(tree, context=context)

        # (Optional) dynamic analysis
        if self.analyze_runtime:
            logger.info("Runtime analysis for function '%s' not fully implemented.", context)

        return {
            "function": context,
            "inefficiencies": self.inefficiencies,
            "suggestions": self.suggestions
        }

    def _run_ast_analysis(self, tree: ast.AST, context: str):
        visitor = AIMLVisitor(context)
        visitor.visit(tree)
        self.inefficiencies.extend(visitor.inefficiencies)
        self.suggestions.extend(visitor.suggestions)
