# prod_ready_aiml_complexity/detectors.py

import ast
import logging

logger = logging.getLogger(__name__)

class AIMLVisitor(ast.NodeVisitor):
    """
    AI/ML-specific AST Visitor to detect patterns like:
      - Nested loops => potential brute-force kNN
      - Matrix ops => potential O(n^3)
      - Inefficient data loops => vectorization recommendations
    """

    def __init__(self, context: str):
        super().__init__()
        self.context = context
        self.inefficiencies = []
        self.suggestions = []

    def visit_For(self, node: ast.For):
        """
        Detect nested loops as a sign of potential O(n^2) or O(n^3).
        """
        nested_for = any(isinstance(child, ast.For) for child in ast.iter_child_nodes(node))
        if nested_for:
            detail_msg = f"Nested loops in {self.context}: potential brute-force or O(n^2) approach."
            self.inefficiencies.append({
                "type": "potential_brute_force",
                "detail": detail_msg
            })
            self.suggestions.append({
                "suggestion": "Consider a more efficient data structure (KDTree) or vectorized approach."
            })
            logger.debug(detail_msg)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Detect calls to certain known expensive ops,
        e.g., matrix multiplications or inversions.
        """
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Basic detection for matrix multiplication
        if func_name in ["matmul", "dot"]:
            detail_msg = f"Detected matrix operation '{func_name}' in {self.context}."
            self.inefficiencies.append({
                "type": "matrix_operation",
                "detail": detail_msg
            })
            self.suggestions.append({
                "suggestion": "Use GPU-accelerated libraries if data is large (BLAS, cuBLAS, PyTorch, TensorFlow)."
            })
            logger.debug(detail_msg)

        # Check for possible Pandas loop or DataFrame usage with 'for'
        # For advanced: parse keywords
        self.generic_visit(node)
