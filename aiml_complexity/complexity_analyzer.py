# complexity_analysis.py
import ast

class AdvancedComplexityAnalyzer:
    def __init__(self, code_str):
        self.code_str = code_str
        try:
            self.tree = ast.parse(self.code_str)
        except:
            self.tree = None

    def analyze_complexity(self):
        if not self.tree:
            return {"theoretical_time_complexity": "Unknown", "space_complexity": "Unknown"}

        loop_depth, recursion_count = self._static_analysis_loops_and_recursion()
        time_complex = self._deduce_time_complexity(loop_depth, recursion_count)

        return {
            "theoretical_time_complexity": time_complex,
            "space_complexity": "O(n)" if recursion_count > 0 or loop_depth > 0 else "O(1)"
        }

    def _static_analysis_loops_and_recursion(self):
        """Return (max_loop_depth, recursion_count)."""
        max_depth, recursion_cnt = 0, 0

        class LoopRecursionVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                nonlocal max_depth
                max_depth += 1
                self.generic_visit(node)

            def visit_While(self, node):
                nonlocal max_depth
                max_depth += 1
                self.generic_visit(node)

            def visit_Call(self, node):
                nonlocal recursion_cnt
                if isinstance(node.func, ast.Name):
                    recursion_cnt += 1
                self.generic_visit(node)

        LoopRecursionVisitor().visit(self.tree)
        return max_depth, recursion_cnt

    def _deduce_time_complexity(self, loop_depth, recursion_count):
        if recursion_count > 1:
            return "O(2^n)"
        elif recursion_count == 1:
            return "O(n)"
        else:
            return f"O(n^{loop_depth})" if loop_depth > 1 else "O(n)"
