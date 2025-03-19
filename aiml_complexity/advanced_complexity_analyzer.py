import ast
import inspect
import textwrap
import timeit
import cProfile, pstats
try:
    import psutil
except ImportError:
    psutil = None
try:
    import memory_profiler
except ImportError:
    memory_profiler = None
    
    

class ComplexityAnalyzer:
    """
    A tool for in-depth complexity analysis and optimization recommendations of a given function.
    """
    def __init__(self, func):
        self.func = func
        self.func_name = func.__name__
        # Attempt to get source code of the function for static analysis
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
        except Exception:
            source = None
        self.tree = ast.parse(source) if source else None
        # Containers for last analysis results (for reuse if needed)
        self.last_theoretical = None
        self.last_time_data = None
        self.last_profile = None
        self.last_memory = None
        self.last_suggestions = None

    def _static_complexity_analysis(self):
        """
        Analyze the function's AST to infer theoretical time complexity (Big-O, Big-Theta, Big-Omega).
        Returns a tuple: (big_o, big_theta, big_omega).
        """
        # Default assumptions if static analysis is unavailable
        big_o = "O(1)"
        big_theta = "Θ(1)"
        big_omega = "Ω(1)"
        if not self.tree:
            return big_o, big_theta, big_omega
        # Locate the target function definition node in the AST
        func_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.func_name:
                func_node = node
                break
        if func_node is None:
            return big_o, big_theta, big_omega

        # Visitor class to traverse AST and collect complexity-related info
        class _ComplexityVisitor(ast.NodeVisitor):
            def __init__(self, func_name):
                self.func_name = func_name
                self.max_loop_depth = 0   # tracks maximum nesting depth of loops
                self.current_loop_depth = 0
                self.recursive_calls = 0  # count of recursive calls found
                self.multiple_recursion = False  # whether more than one recursive call in a single frame
                self.divide_and_conquer = False  # whether recursion splits input (e.g. n/2, indicative of n log n)
                self.early_exit = False    # whether there's an early return/break (best-case improvement)
                super().__init__()
            def visit_For(self, node):
                # Entering a for-loop
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                self.generic_visit(node)   # visit loop body
                self.current_loop_depth -= 1
            def visit_While(self, node):
                # Entering a while-loop
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                self.generic_visit(node)
                self.current_loop_depth -= 1
            def visit_If(self, node):
                # Check if there's an early return or break in the if-block (affects best-case)
                for child in node.body:
                    if isinstance(child, ast.Return) or isinstance(child, ast.Break):
                        self.early_exit = True
                # Continue traversing the if (and else) blocks
                self.generic_visit(node)
            def visit_Call(self, node):
                # Check function calls for recursion
                called_name = None
                if isinstance(node.func, ast.Name):
                    called_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle methods (e.g., self.funcName() in class)
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                        called_name = node.func.attr
                if called_name == self.func_name:
                    # A recursive call to the same function
                    self.recursive_calls += 1
                    if self.recursive_calls > 1:
                        self.multiple_recursion = True
                        # Analyze arguments to guess if input size is reduced fractionally
                        for arg in node.args:
                            # e.g., passing n/2 or len(arr)//2 suggests divide-and-conquer
                            if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Div, ast.FloorDiv, ast.RShift)):
                                self.divide_and_conquer = True
                            if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                                # Slicing an array (potentially halving it)
                                self.divide_and_conquer = True
                self.generic_visit(node)

        visitor = _ComplexityVisitor(self.func_name)
        visitor.visit(func_node)

        # Deduce Big-O complexity from collected data
        loop_depth = visitor.max_loop_depth
        rec_calls = visitor.recursive_calls
        multi_recursion = visitor.multiple_recursion
        divide = visitor.divide_and_conquer
        early_exit = visitor.early_exit

        # Worst-case time complexity (Big-O)
        if multi_recursion:
            # Multiple recursive calls in one call frame (e.g. fib(n-1)+fib(n-2))
            big_o = "O(n log n)" if divide else "O(2^n)"
        else:
            if rec_calls == 1:
                # Single recursive call at a time (linear recursion)
                big_o = "O(log n)" if divide else "O(n)"
            else:
                # No recursion, base on loop nesting
                if loop_depth > 1:
                    big_o = f"O(n^{loop_depth})"
                elif loop_depth == 1:
                    big_o = "O(n)"
                else:
                    big_o = "O(1)"
        # Best-case time complexity (Big-Omega)
        if early_exit:
            # If an early break/return exists, best case could be constant
            big_omega = "Ω(1)"
        else:
            # Otherwise, best case grows similarly to worst-case (same order)
            if "2^n" in big_o:
                big_omega = "Ω(2^n)"
            else:
                big_omega = big_o.replace("O", "Ω")
        # Tight-bound notation (Big-Theta)
        if big_o[2:] == big_omega[2:]:
            # If upper and lower bounds match (no variation by input scenario)
            big_theta = big_o.replace("O", "Θ")
        else:
            # If they differ, we indicate a range (not strictly tight)
            big_theta = f"Θ({big_omega[2:]}–{big_o[2:]})"
        return big_o, big_theta, big_omega

    def _time_profile(self, inputs, repeat=3):
        """
        Empirically measure execution time for given inputs using timeit, and profile function calls using cProfile.
        """
        times = []
        # Measure execution time for each input size
        for inp in inputs:
            # Determine a numeric "size" (length) for reporting if possible
            try:
                n = len(inp)
            except Exception:
                n = inp if isinstance(inp, int) else 1
            # Use timeit to average timing over multiple runs for accuracy
            avg_time = timeit.timeit(lambda: self.func(inp), number=repeat) / repeat
            times.append((n, avg_time))
        # Profile the function on the largest input to get function call stats
        profile_stats = {}
        pr = cProfile.Profile()
        pr.enable()
        if inputs:
            self.func(inputs[-1])    # run once on largest input
        pr.disable()
        stats = pstats.Stats(pr).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
        # Collect top functions by cumulative time
        for func, stat in stats.stats.items():
            ncalls, _, _, cumtime, _ = stat  # stats tuple: (call count, reccall count, tot time, cum time, inline time)
            if ncalls == 0:
                continue  # skip entries that were not called
            func_name = f"{func[2]} ({func[0].split('/')[-1]}:{func[1]})"
            profile_stats[func_name] = {"calls": ncalls, "cumtime": cumtime}
        return times, profile_stats

    def _memory_profile(self, inputs):
        """
        Measure peak memory usage for given inputs using memory_profiler (if available) or psutil as fallback.
        """
        mem_usage = []
        for inp in inputs:
            try:
                n = len(inp)
            except Exception:
                n = inp if isinstance(inp, int) else 1
            peak = None
            if memory_profiler:
                # Run the function under memory_profiler to capture usage over time
                try:
                    usage = memory_profiler.memory_usage((self.func, (inp,), {}), max_iterations=1, interval=0.01)
                    if usage:
                        peak = max(usage)
                except Exception:
                    peak = None
            if peak is None and psutil:
                # Fallback: measure memory before and after (note: may miss peak in the middle)
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024**2)  # in MiB
                self.func(inp)
                mem_after = process.memory_info().rss / (1024**2)
                peak = max(mem_before, mem_after)
            mem_usage.append((n, peak))
        return mem_usage

    def suggest_optimizations(self, inputs=None):
        """
        Generate optimization suggestions. If inputs provided, runs a full analysis; otherwise uses last analysis data.
        Returns a list of suggestion strings.
        """
        # Ensure we have recent analysis data
        if inputs is not None:
            self.generate_report(inputs)
        elif not self.last_suggestions:
            # If no prior data, at least do static analysis for some baseline suggestions
            big_o, big_theta, big_omega = self._static_complexity_analysis()
            suggestions = []
            if "n^" in big_o:
                suggestions.append("High nested loop complexity detected; try to reduce nested loops or use more efficient algorithms.")
            if "2^n" in big_o:
                suggestions.append("Exponential recursion detected; use memoization or dynamic programming to cut down redundant computations.")
            if "O(n)" in big_o and "2^n" not in big_o and "log n" not in big_o:
                suggestions.append("Consider adding early break conditions to avoid always doing O(n) work in the best case.")
            # General AI/ML tips:
            suggestions.append("Process data in batches (vectorize operations) instead of one by one to leverage optimized math libraries.")
            suggestions.append("Use efficient data structures (e.g., sets for fast membership tests) to improve algorithmic performance.")
            suggestions.append("Parallelize independent tasks to utilize multiple CPU cores for speedup.")
            return suggestions
        # Return the last generated suggestions list
        return self.last_suggestions

    def generate_report(self, inputs):
        """
        Run the full analysis (theoretical & empirical) and return a formatted report string.
        """
        # 1. Theoretical time complexity analysis
        big_o, big_theta, big_omega = self._static_complexity_analysis()
        self.last_theoretical = (big_o, big_theta, big_omega)
        # 2. Empirical time profiling
        times, profile_stats = self._time_profile(inputs)
        self.last_time_data = times
        self.last_profile = profile_stats
        # 3. Empirical memory profiling
        mem_usage = self._memory_profile(inputs)
        self.last_memory = mem_usage
        # 4. Formulate optimization suggestions
        suggestions = []
        # Check time complexity patterns for optimization
        if "n^" in big_o:
            power = 2
            if '^' in big_o:
                try:
                    power = int(big_o.split('^')[1].rstrip(')'))
                except:
                    power = 2
            suggestions.append(f"Nested loops detected (O(n^{power})); consider using algorithms/data structures to reduce nested iteration.")
        if "2^n" in big_o:
            suggestions.append("Exponential time complexity detected; apply dynamic programming or memoization to avoid repeated sub-computations.")
        if "O(n)" in big_o and "log n" not in big_o and "2^n" not in big_o:
            suggestions.append("Linear iteration over input; include early exits where possible to handle best-case inputs faster.")
        # Recursion vs iteration suggestion
        if self.tree:
            # If recursive calls exist in the AST, advise on recursion optimization
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and node.func.id == self.func_name) or \
                       (isinstance(node.func, ast.Attribute) and node.func.attr == self.func_name):
                        suggestions.append("Recursive pattern detected; consider an iterative approach to reduce function-call overhead (especially if recursion depth is large).")
                        break
        # Memory usage patterns
        if mem_usage:
            base_mem = mem_usage[0][1] or 0
            peak_mem = max(m or 0 for _, m in mem_usage)
            if peak_mem - base_mem > 50:  # if memory grows significantly (threshold 50 MiB here)
                suggestions.append("High memory growth observed; use in-place operations or generators to minimize peak memory footprint.")
        # Vectorization and batching suggestions
        if self.tree:
            code_text = ast.dump(self.tree)
            has_loop = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(self.tree))
            uses_numpy = ('numpy' in code_text or 'np.' in code_text)
            uses_pandas = ('pandas' in code_text or 'pd.' in code_text)
            if has_loop and (uses_numpy or uses_pandas):
                suggestions.append("Loops over large data structures found; leverage NumPy/Pandas vectorized operations to speed up computations.")
            if has_loop:
                suggestions.append("Consider mini-batch processing for large data to utilize vectorized math and parallel hardware (instead of single-item loops).")
        # Profile-based suggestions (identify bottlenecks)
        if profile_stats:
            # Find the function (other than the analyzed function itself) with largest cumulative time
            heaviest_func = None
            max_time = 0.0
            for fname, stats in profile_stats.items():
                if self.func_name in fname or '_lsprof' in fname or 'disable' in fname:
                    continue  # skip the main function and profiler internal calls
                if stats["cumtime"] > max_time:
                    max_time = stats["cumtime"]
                    heaviest_func = fname
            if heaviest_func:
                func_label = heaviest_func.split()[0]
                suggestions.append(f"Frequent calls to {func_label} detected; consider caching its results or optimizing this function to reduce overhead.")
        # General best-practice suggestions
        suggestions.append("Use appropriate data structures (e.g., sets for O(1) lookups, heaps for retrieval) to improve efficiency.")
        suggestions.append("Parallelize independent operations (using multiprocessing or async methods) to utilize multiple cores and speed up execution.")
        # Save suggestions for later access
        self.last_suggestions = suggestions

        # 5. Compile the report text
        report_lines = []
        report_lines.append(f"Function `{self.func_name}` Complexity Analysis:")
        report_lines.append(f"- Theoretical Time Complexity: Worst-case {big_o}, Best-case {big_omega}, Typical {big_theta}.")
        # Empirical time results
        time_results = ", ".join([f"n={n}: {t:.6f}s" for n, t in times])
        report_lines.append(f"- Empirical Time Performance: {time_results}.")
        if len(times) > 1:
            # Indicate growth rate from first to last measured input
            n1, t1 = times[0]; n2, t2 = times[-1]
            if t1 and t2:
                factor_time = t2/t1
                factor_size = n2/n1 if n1 != 0 else float('inf')
                report_lines.append(f"  (Time increased ~{factor_time:.1f}x when input size increased {factor_size:.1f}x.)")
        # Empirical memory results
        mem_results = ", ".join([f"n={n}: {mem:.2f} MiB" if mem is not None else f"n={n}: N/A"
                                 for n, mem in mem_usage])
        report_lines.append(f"- Empirical Memory Usage: {mem_results}.")
        # Hotspots from profiling
        if profile_stats:
            # Take top 3 time-consuming functions (by cumulative time)
            top_funcs = sorted(profile_stats.items(), key=lambda x: x[1]["cumtime"], reverse=True)[:3]
            hotspot_info = "; ".join([f"{fname} (calls={stat['calls']}, time={stat['cumtime']:.4f}s)"
                                       for fname, stat in top_funcs])
            report_lines.append(f"- Performance Hotspots: {hotspot_info}.")
        # Optimization suggestions list
        if suggestions:
            report_lines.append("- Optimization Suggestions:")
            for sug in suggestions:
                report_lines.append(f"  * {sug}")
        return "\n".join(report_lines)
