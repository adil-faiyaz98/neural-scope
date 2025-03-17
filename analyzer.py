# aiml_complexity/analyzer.py

import ast
import inspect
import time
import os
from typing import Any, Dict, List, Optional


class ComplexityAnalyzer:
    """
    Core AI/ML Complexity Analyzer.
    Parses Python code (AST-based) to detect known inefficiencies:
      - Brute-force kNN patterns
      - Unvectorized matrix ops
      - Slow data preprocessing loops
      - etc.
    Also hooks into runtime to measure basic performance if needed.
    """

    def __init__(self, analyze_runtime: bool = False):
        """
        :param analyze_runtime: If True, we attempt dynamic profiling
                               (e.g., timing runs) in addition to static analysis.
        """
        self.analyze_runtime = analyze_runtime
        self.detections = []  # store discovered inefficiencies or patterns
        self.suggestions = []  # store improvement suggestions

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze all functions in a Python file for AI/ML inefficiencies.
        Returns a structured report dictionary.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Static analysis
        tree = ast.parse(code)
        self._analyze_ast(tree, file_path=file_path)

        # Optionally do runtime analysis (very basic for now)
        # For real usage, we might do a deeper dynamic approach
        if self.analyze_runtime:
            self._analyze_runtime(code, file_path=file_path)

        return {
            "file": file_path,
            "inefficiencies": self.detections,
            "suggestions": self.suggestions
        }

    def analyze_function(self, func) -> Dict[str, Any]:
        """
        Analyze a specific function object (like a kNN or matrix op).
        Useful if the user passes a function directly.
        """
        # 1) Static analysis via AST
        source = inspect.getsource(func)
        tree = ast.parse(source)
        self._analyze_ast(tree, file_path=func.__name__)

        # 2) (Optional) dynamic analysis
        if self.analyze_runtime:
            start = time.time()
            # We might run func with sample data or rely on the user to supply input
            try:
                func()  # naive call with no arguments for demo
            except:
                pass
            elapsed = time.time() - start
            self.detections.append({
                "type": "performance_runtime",
                "detail": f"Executed {func.__name__} in {elapsed:.4f} seconds (no real input)."
            })

        return {
            "function": func.__name__,
            "inefficiencies": self.detections,
            "suggestions": self.suggestions
        }

    def _analyze_ast(self, tree: ast.AST, file_path: str):
        """
        Walk the AST to detect AI/ML patterns.
        For example: nested loops (brute-force kNN),
                     large matrix loops, unvectorized ops, etc.
        """
        # We can create a custom NodeVisitor or do manual walks.
        # For brevity, let's do a simplistic pattern check:
        #  - detect double nested loops as a possible brute-force approach
        #  - detect usage of "range(len(...))" with nested loops
        #  - detect calls to PyTorch/TensorFlow to see if they exist

        analyzer_visitor = _AIMLVisitor(file_path)
        analyzer_visitor.visit(tree)

        # Merge the visitor's findings with the top-level report
        self.detections.extend(analyzer_visitor.inefficiencies)
        self.suggestions.extend(analyzer_visitor.suggestions)

    def _analyze_runtime(self, code: str, file_path: str):
        """
        Basic dynamic analysis hook.
        Could spawn a separate process or interpret usage with cProfile, etc.
        For now, just a placeholder.
        """
        # In a real scenario, we'd compile code, run with sample inputs, measure time & memory
        pass


class _AIMLVisitor(ast.NodeVisitor):
    """
    Custom AST Visitor that hunts for AI/ML inefficiencies.
    Example: Double nested loops for all-pairs distance (brute-force kNN).
    """

    def __init__(self, context: str):
        self.context = context
        self.inefficiencies = []
        self.suggestions = []

    def visit_For(self, node: ast.For):
        # if there's a nested for inside it, that might be O(n^2).
        # We can be naive: check if there's another For inside the body
        nested_for = any(isinstance(child, ast.For) for child in ast.walk(node))
        if nested_for:
            self.inefficiencies.append({
                "type": "potential_brute_force",
                "detail": f"Nested 'for' loops found in {self.context}: possible O(n^2) or worse."
            })
            self.suggestions.append({
                "suggestion": "Consider using a KD-tree/Ball-tree or vectorized approach if this is kNN or distance-based computation."
            })

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check calls to certain libraries or patterns
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # e.g. torch.tensor, tf.matmul, np.dot, etc.
            func_name = node.func.attr

        if func_name in ["matmul", "dot"]:
            # Indicate a matrix operation
            self.inefficiencies.append({
                "type": "matrix_operation",
                "detail": f"Detected matrix op '{func_name}' in {self.context}. Check if shapes are large -> possibly O(n^3)."
            })
            self.suggestions.append({
                "suggestion": "If dealing with large matrices, ensure you're using GPU acceleration or an efficient library (BLAS, cuBLAS)."
            })

        self.generic_visit(node)
