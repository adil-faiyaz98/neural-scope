"""
CPU and Vectorization Profiler for AI/ML Workloads

This module provides:
1. Static analysis of Python code to detect naive loops and missed vectorization.
2. Dynamic profiling of CPU usage, thread utilization, and basic vectorization checks.
3. Recommendations for improving parallelism (OpenMP, threading configs) and vectorized operations.
"""

import ast
import time
import textwrap
import psutil
import threading
import re

try:
    import cProfile
    import pstats
except ImportError:
    cProfile = None

try:
    import numpy as np
except ImportError:
    np = None

class CPULoopVectorAnalyzer:
    """
    Static Analysis Class to detect naive loops that should be vectorized
    or parallelized on CPU for AI/ML code.
    """
    def __init__(self, code_str):
        self.original_code = code_str
        # attempt to parse AST
        try:
            self.tree = ast.parse(textwrap.dedent(code_str))
        except SyntaxError:
            self.tree = None

    def analyze_code(self):
        """
        Returns a dict:
        {
           "naive_loops": [list of line # or code snippets],
           "recommendations": [string suggestions],
        }
        """
        if not self.tree:
            return {
                "naive_loops": [],
                "recommendations": ["Could not parse code. No AST analysis available."]
            }

        naive_loops = []
        recommendations = []
        # We'll traverse and look for large loops or nested loops
        # that do numeric computations in Python.

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                super().__init__()
                self.current_loop_depth = 0
                self.max_loop_depth = 0
                self.in_numeric_context = False  # if we see numeric ops or calls (like np)
                self.naive_loop_lines = []

            def visit_For(self, node):
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)

                # Check if the body might contain numeric operations
                # (like * + - / on arrays or calls to np without vectorization).
                numeric_ops_found = False
                for child in ast.walk(node):
                    if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        numeric_ops_found = True
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        # e.g., child.func.value.id = 'np' => something
                        if getattr(child.func.value, 'id', '') == 'np':
                            numeric_ops_found = True
                if numeric_ops_found:
                    # We consider this loop a candidate for naive numeric usage
                    self.naive_loop_lines.append(node.lineno)

                self.generic_visit(node)
                self.current_loop_depth -= 1

            def visit_While(self, node):
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                # similarly check if numeric ops inside
                numeric_ops_found = False
                for child in ast.walk(node):
                    if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        numeric_ops_found = True
                    # also check calls
                if numeric_ops_found:
                    self.naive_loop_lines.append(node.lineno)
                self.generic_visit(node)
                self.current_loop_depth -= 1

        lv = LoopVisitor()
        lv.visit(self.tree)
        if lv.naive_loop_lines:
            naive_loops = lv.naive_loop_lines
            recommendations.append(
                f"Detected potential naive numeric loops at lines {lv.naive_loop_lines}. Consider vectorizing with NumPy or using parallelization (OpenMP, multiprocessing, or GPU)."
            )
        # also if we have deep loop nesting:
        if lv.max_loop_depth > 1:
            recommendations.append(
                f"Detected loop nesting depth={lv.max_loop_depth}. Check if there's a chance to reduce nested iteration or use batch operations."
            )

        return {
            "naive_loops": naive_loops,
            "recommendations": recommendations
        }


class CPUUsageProfiler:
    """
    Dynamically track CPU usage, core utilization, and thread usage
    before/during/after a function call.
    Also tries to detect single-core bottlenecks vs. multi-core usage.
    """
    def __init__(self, func):
        """
        :param func: The Python function to profile.
        """
        self.func = func

    def profile(self, *args, **kwargs):
        """
        Execute the function, measure CPU usage over time, and thread usage.
        Return a dictionary with usage stats & recommendations.
        """
        # We'll sample CPU usage on an interval while the function runs in a background thread.
        usage_samples = []
        def target_function():
            if cProfile:
                prof = cProfile.Profile()
                prof.enable()
                self.result = self.func(*args, **kwargs)
                prof.disable()
                stats = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
                self.profile_stats = stats
            else:
                self.result = self.func(*args, **kwargs)
                self.profile_stats = None

        # Start the function in a thread
        t = threading.Thread(target=target_function)
        t.start()

        # While it's running, sample CPU usage
        sample_interval = 0.05
        while t.is_alive():
            # psutil.cpu_percent gives overall CPU usage across all cores
            cpu_percent = psutil.cpu_percent(interval=None)
            # to see if we are single-core bound, we compare cpu_percent to 100/core_count
            core_count = psutil.cpu_count(logical=True)
            usage_samples.append((time.time(), cpu_percent, core_count))
            time.sleep(sample_interval)
        t.join()

        # Once done, let's interpret usage
        # We'll compute average CPU usage, max usage, see if it's near single-core (100/ core_count)
        if not usage_samples:
            return {"cpu_usage_samples": [], "analysis": "No samples collected", "recommendations": []}

        avg_cpu = sum(s[1] for s in usage_samples) / len(usage_samples)
        max_cpu = max(s[1] for s in usage_samples)
        core_count = usage_samples[0][2]

        # Heuristic to detect single-core or multi-core usage
        # e.g., if average CPU usage < (core_count * 30%), we might be underutilizing CPU
        # if average CPU usage ~ 100% but we have multiple cores, might be GIL bound or single-threaded
        recs = []
        analysis_str = (f"Avg CPU usage={avg_cpu:.2f}%, Max CPU usage={max_cpu:.2f}%, System has {core_count} cores.\n")

        if avg_cpu < 30 and core_count > 1:
            recs.append("Low CPU usage - possible I/O bottleneck or small workload. Parallelization might help if the task is CPU-bound.")
        if avg_cpu >= 70 and core_count > 1:
            # Could be saturating multiple cores or single core near 100%. Check ratio
            # if avg_cpu < 100*(core_count-0.5), we might be using multiple cores
            # if avg_cpu ~ 100 or 120 but we have 8 cores => single-core bounding
            single_core_approx = 100.0 / core_count
            if avg_cpu < single_core_approx * 2:
                recs.append("CPU usage suggests single-core saturation; consider multi-threading or vectorization to spread load across cores.")
            else:
                recs.append("High CPU usage - likely using multiple cores. Ensure no thread contention or GIL overhead if purely Python loops.")

        # Check cProfile results if available
        if self.profile_stats:
            # could parse top functions
            pass

        return {
            "cpu_usage_samples": usage_samples,
            "analysis": analysis_str,
            "recommendations": recs
        }


class VectorizationAdvisor:
    """
    Provides additional advice specifically about vectorization
    if we detect numeric loops or partial usage of NumPy, PyTorch, etc.
    """
    def __init__(self, code_str):
        self.code_str = code_str

    def suggest_vector_ops(self):
        """
        If code uses np or numeric loops, provide suggestions like:
         - 'Use np.dot or np.matmul instead of a manual nested loop.'
         - 'Enable MKL/OpenBLAS threading or set OMP_NUM_THREADS.'
        etc.
        """
        suggestions = []
        # If we see 'import numpy as np' or 'from numpy import ...' => we assume user has NumPy
        if re.search(r"import numpy as np", self.code_str):
            # if naive loops found
            suggestions.append("Detected NumPy usage. Ensure large array operations are done with np functions, not Python loops for speed.")
            suggestions.append("Enable BLAS threading: e.g., MKL or OpenBLAS. Set OMP_NUM_THREADS to use multi-core vectorization.")
        # If we see PyTorch references
        if re.search(r"import torch", self.code_str):
            suggestions.append("Detected PyTorch. Consider setting num_workers in DataLoader for CPU parallelism. Use torch.set_num_threads(...) for CPU ops.")
        # If we see TensorFlow references
        if re.search(r"import tensorflow", self.code_str):
            suggestions.append("Detected TensorFlow. Check inter_op and intra_op parallelism settings. E.g., tf.config.threading.set_intra_op_parallelism_threads(...)")
        return suggestions


def analyze_cpu_and_vectorization(code_str, func, func_inputs):
    """
    High-level routine that does:
    1) Static loop & vectorization detection,
    2) CPU usage profiling with the given function & inputs,
    3) Vectorization advice,
    4) Combined suggestions.
    """
    # 1) Static analysis
    static_analyzer = CPULoopVectorAnalyzer(code_str)
    static_result = static_analyzer.analyze_code()

    # 2) CPU usage dynamic profiling
    cpu_profiler = CPUUsageProfiler(func)
    dyn_result = cpu_profiler.profile(*func_inputs)

    # 3) Additional vectorization suggestions
    vect_advisor = VectorizationAdvisor(code_str)
    vect_suggestions = vect_advisor.suggest_vector_ops()

    # Merge suggestions
    all_suggestions = static_result["recommendations"] + dyn_result.get("recommendations", []) + vect_suggestions

    analysis_summary = {
        "static_analysis": static_result,
        "dynamic_profiling": {
            "analysis": dyn_result.get("analysis"),
            "cpu_usage_samples": dyn_result.get("cpu_usage_samples"),
        },
        "recommendations": list(set(all_suggestions))  # unique them
    }
    return analysis_summary


# Simple test / demonstration
if __name__ == "__main__":
    # Example code with naive Python loops using np
    code_example = r"""
import numpy as np

def naive_sum(A, B):
    # naive loop instead of np.add or vectorized approach
    C = []
    for i in range(len(A)):
        C.append(A[i] + B[i])
    return C

dataA = np.random.rand(100000)
dataB = np.random.rand(100000)
res = naive_sum(dataA, dataB)
"""

    # define the actual function we want to profile
    def naive_sum_func(n):
        # simulate naive loop on an array of size n
        import numpy as np
        A = np.random.rand(n)
        B = np.random.rand(n)
        C = []
        for i in range(n):
            C.append(A[i] + B[i])
        return C

    # We'll check code_str for naive loops
    # Then dynamically profile naive_sum_func with input e.g. n=500000
    results = analyze_cpu_and_vectorization(code_example, naive_sum_func, func_inputs=[500000])
    print("Analysis Summary:")
    print("Static analysis naive loops:", results["static_analysis"]["naive_loops"])
    print("CPU usage analysis:", results["dynamic_profiling"]["analysis"])
    print("All Recommendations:")
    for r in results["recommendations"]:
        print(" -", r)
