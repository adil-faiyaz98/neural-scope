"""
Comprehensive complexity analysis for functions.

This module provides tools for in-depth analysis of function complexity,
combining static and dynamic analysis to provide theoretical and empirical
complexity estimates along with optimization recommendations.
"""

import ast
import inspect
import textwrap
import timeit
import cProfile
import pstats
import logging
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
import io
import re

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Some memory profiling features will be limited.")

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not available. Memory profiling will be limited.")

# Model architecture types
class ModelArchitecture:
    UNKNOWN = "unknown"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GAN = "gan"
    DIFFUSION = "diffusion"
    AUTOENCODER = "autoencoder"
    QUANTIZED = "quantized"
    PRUNED = "pruned"

class ComplexityAnalyzer:
    """
    A tool for in-depth complexity analysis and optimization recommendations of a given function.
    """
    def __init__(self, func: Callable):
        """
        Initialize the complexity analyzer with a function to analyze.

        Args:
            func: Function to analyze.
        """
        self.func = func
        self.func_name = func.__name__

        # Attempt to get source code of the function for static analysis
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
        except Exception as e:
            logger.warning(f"Could not get source code for {func.__name__}: {e}")
            source = None

        self.source = source
        self.tree = ast.parse(source) if source else None
                
        # Containers for last analysis results (for reuse if needed)
        self.last_theoretical = None
        self.last_time_data = None
        self.last_profile = None
        self.last_memory = None
        self.last_suggestions = None

        # Detect model architecture
        self.model_architecture = self._detect_model_architecture()
                
    def _detect_model_architecture(self) -> str:
        """
        Analyze the function to detect if it's related to a specific model architecture.
        
        Returns:
            String identifier for the detected model architecture type
        """
        if not self.source:
            return ModelArchitecture.UNKNOWN
            
        # Look for architecture-specific patterns in the source code
        patterns = {
            ModelArchitecture.TRANSFORMER: [
                r'(attention|transformer|mha|multi.?head|bert|gpt|llm|roberta|t5|bart|electra|albert|distilbert|longformer|reformer|xlnet)',
                r'(encoder|decoder).+layers',
                r'self\.attention'
            ],
            ModelArchitecture.CNN: [
                r'(conv[2-3]?d|convolution|kernel|maxpool|avgpool)',
                r'(stride|padding|channels|filters)'
            ],
            ModelArchitecture.RNN: [
                r'(rnn|lstm|gru|recurrent)',
                r'hidden.?state',
                r'cell.?state'
            ],
            ModelArchitecture.GAN: [
                r'(gan|generative.?adversarial|generator|discriminator)',
                r'(real|fake).+(loss|images|data)'
            ],
            ModelArchitecture.DIFFUSION: [
                r'(diffusion|noise|denoise|unet|denoising)',
                r'(timestep|step|scheduler|sampling)',
                r'(reverse|forward).+process',
                r'(score|prediction).+based'
            ],
            ModelArchitecture.AUTOENCODER: [
                r'(autoencoder|encoder|decoder|latent|vae|embedding)',
                r'(reconstruct|bottleneck)'
            ],
            ModelArchitecture.QUANTIZED: [
                r'(quantization|quantize|int8|fp16|bfloat16)',
                r'(qconfig|fake_quant|dynamic_range)',
                r'(observer|scale_factor|zero_point)'
            ],
            ModelArchitecture.PRUNED: [
                r'(mask|threshold)',
                r'(structured|unstructured)',
                r'(weight_decay|regularization|l1|l0)'
            ]
        }

        # Count matches for each architecture type
        match_counts = {arch: 0 for arch in patterns.keys()}
        for arch, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, self.source, re.IGNORECASE)
                match_counts[arch] += len(matches)

        # Get the architecture with the most matches
        best_match = max(match_counts.items(), key=lambda x: x[1])

        # Only consider it a match if we have at least some evidence
        if best_match[1] > 0:
            return best_match[0]
        return ModelArchitecture.UNKNOWN

    def _static_complexity_analysis(self) -> Tuple[str, str, str]:
        """
        Analyze the function's AST to infer theoretical time complexity (Big-O, Big-Theta, Big-Omega).

        Returns:
            Tuple of (big_o, big_theta, big_omega) complexity strings
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
                self.generic_visit(node)   # visit loop body
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

    def _time_profile(self, inputs: List[Any], repeat: int = 3) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        """
        Empirically measure execution time for given inputs using timeit, and profile function calls using cProfile.

        Args:
            inputs: List of inputs to test
            repeat: Number of repetitions for each input

        Returns:
            Tuple of (timing_results, profile_stats)
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

        # Get profile stats
        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)

        # Collect top functions by cumulative time
        for func, stat in stats.stats.items():
            ncalls, _, _, cumtime, _ = stat  # stats tuple: (call count, reccall count, tot time, cum time, inline time)
            if ncalls == 0:
                continue  # skip entries that were not called
            func_name = f"{func[2]} ({func[0].split('/')[-1]}:{func[1]})"
            profile_stats[func_name] = {"calls": ncalls, "cumtime": cumtime}

        return times, profile_stats

    def _memory_profile(self, inputs: List[Any]) -> List[Tuple[int, float]]:
        """
        Measure peak memory usage for given inputs using memory_profiler (if available) or psutil as fallback.

        Args:
            inputs: List of inputs to test

        Returns:
            List of (input_size, memory_usage) tuples
        """
        mem_usage = []

        for inp in inputs:
            try:
                n = len(inp)
            except Exception:
                n = inp if isinstance(inp, int) else 1

            peak = None

            # Apply architecture-specific profiling adjustments
            if self.model_architecture == ModelArchitecture.DIFFUSION:
                # For diffusion models, we need to track memory across all timesteps
                # This is memory-intensive, so we use a specialized approach
                if MEMORY_PROFILER_AVAILABLE:
                    # For diffusion models, use a finer interval to catch peak memory during step process
                    usage = memory_profiler.memory_usage((self.func, (inp,), {}), interval=0.005, max_iterations=1)
                    peak = max(usage) - usage[0]  # peak increase in memory
                elif PSUTIL_AVAILABLE:
                    # For diffusion models, monitor memory before, during and after to catch peak
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                    
                    # Run with periodic memory checks for diffusion process
                    result = self.func(inp)
                    
                    # Check memory after function completes
                    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                    peak = mem_after - mem_before
            else:
                # Standard memory profiling for other function types
                if MEMORY_PROFILER_AVAILABLE:
                    # Production Ranking #3: Highest precision but significant overhead
                    # Best for development/debugging but too slow for production use
                    usage = memory_profiler.memory_usage((self.func, (inp,), {}), interval=0.01, max_iterations=1)
                    peak = max(usage) - usage[0]  # peak increase in memory
                elif PSUTIL_AVAILABLE:
                    # Production Ranking #1: Best balance of accuracy and performance
                    # Low overhead, suitable for production monitoring, but process-wide measurement
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                    self.func(inp)
                    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                    peak = mem_after - mem_before
                else:
                    # Production Ranking #2: Moderate overhead, no external dependencies
                    # Good accuracy for pure Python code, less accurate with C extensions
                    import tracemalloc
                    tracemalloc.start()
                    self.func(inp)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak = peak / (1024 * 1024)  # Convert to MB

            mem_usage.append((n, peak))

        return mem_usage

    def _generate_suggestions(self, theoretical: Tuple[str, str, str], 
                             time_data: List[Tuple[int, float]], 
                             profile_stats: Dict[str, Any], 
                             memory_data: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions based on analysis results.

        Args:
            theoretical: Tuple of (big_o, big_theta, big_omega) complexity strings
            time_data: List of (input_size, execution_time) tuples
            profile_stats: Dictionary with profiling statistics
            memory_data: List of (input_size, memory_usage) tuples

        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        big_o, _, _ = theoretical

        # Add model-specific suggestions based on detected architecture
        if self.model_architecture != ModelArchitecture.UNKNOWN:
            suggestions.append({
                "type": "architecture",
                "severity": "info",
                "message": f"Detected {self.model_architecture.upper()} architecture",
                "details": f"Analysis has been adjusted for {self.model_architecture} model characteristics."
            })

        # Add architecture-specific suggestions
        if self.model_architecture == ModelArchitecture.DIFFUSION:
            suggestions.append({
                "type": "optimization",
                "severity": "medium",
                "message": "Diffusion model optimization opportunities",
                "details": "Consider using these techniques to optimize diffusion models: "
                           "1. Progressive distillation to reduce sampling steps "
                           "2. Low-rank adaptations for parameter efficiency "
                           "3. Half-precision (FP16) computation where possible "
                           "4. Gradient checkpointing to reduce memory usage during training",
                "code_example": """
# Example of fp16 computation for diffusion:
with torch.cuda.amp.autocast():
    model_output = model(x, timesteps)
"""
            })
            # Check for potential memory issues in diffusion models
            if memory_data and memory_data[-1][1] > 50:  # More than 50MB
                suggestions.append({
                    "type": "memory",
                    "severity": "high",
                    "message": f"High memory usage for diffusion model: {memory_data[-1][1]:.2f} MB",
                    "details": "Diffusion models can be memory intensive. Consider:"
                              "1. Reducing batch size "
                              "2. Using gradient checkpointing "
                              "3. Implementing memory-efficient attention "
                              "4. Using lower precision (e.g., FP16, INT8)"
                })

        # Standard suggestions (existing code)
        # Check for high complexity
        if "n^2" in big_o or "n^3" in big_o:
            suggestions.append({
                "type": "complexity",
                "severity": "high",
                "message": f"Function has high time complexity: {big_o}",
                "details": "Consider using more efficient algorithms or data structures to reduce complexity."
            })

        # Check for exponential complexity
        if "2^n" in big_o:
            suggestions.append({
                "type": "complexity",
                "severity": "critical",
                "message": f"Function has exponential time complexity: {big_o}",
                "details": "Exponential algorithms can become extremely slow for even moderately large inputs. Consider using dynamic programming or memoization to avoid redundant calculations."
            })

        # Check for high memory usage
        if memory_data and memory_data[-1][1] > 100:  # More than 100MB
            suggestions.append({
                "type": "memory",
                "severity": "medium",
                "message": f"High memory usage: {memory_data[-1][1]:.2f} MB for input size {memory_data[-1][0]}",
                "details": "Consider using more memory-efficient data structures or processing data in smaller chunks."
            })

        # Check for slow function calls in profile
        if profile_stats:
            slow_funcs = []
            for func_name, stats in profile_stats.items():
                if stats["cumtime"] > 0.1:  # More than 0.1 seconds
                    slow_funcs.append((func_name, stats["cumtime"]))
            if slow_funcs:
                slow_funcs.sort(key=lambda x: x[1], reverse=True)
                suggestions.append({
                    "type": "hotspot",
                    "severity": "medium",
                    "message": f"Slow function calls detected: {slow_funcs[0][0]} ({slow_funcs[0][1]:.3f}s)",
                    "details": "Consider optimizing these functions or reducing the number of calls."
                })

        # Check for potential memoization opportunities
        if "2^n" in big_o and self.source and "return" in self.source:
            suggestions.append({
                "type": "optimization",
                "severity": "high",
                "message": "Potential memoization opportunity",
                "details": "This function appears to have overlapping subproblems. Consider using memoization to cache results and avoid redundant calculations.",
                "code_example": """
# Example of memoization:
from functools import lru_cache

@lru_cache(maxsize=None)
def your_function(n):
    # Your code here
    return result
"""
            })

        # Check for potential parallelization opportunities
        if "n^" in big_o and self.source and "for" in self.source:
            suggestions.append({
                "type": "optimization",
                "severity": "medium",
                "message": "Potential parallelization opportunity",
                "details": "This function contains loops that might benefit from parallelization. Consider using multiprocessing or vectorized operations.",
                "code_example": """
# Example of parallelization:
from multiprocessing import Pool

def process_chunk(chunk):
    # Process a chunk of data
    return result

with Pool() as pool:
    results = pool.map(process_chunk, chunks_of_data)
"""
            })

        return suggestions

    def analyze(self, inputs: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the function's complexity and performance.

        Args:
            inputs: List of inputs to test (if None, will use default inputs)

        Returns:
            Dictionary with analysis results
        """
        # Generate default inputs if none provided
        if inputs is None:
            # Try to generate reasonable inputs based on function signature
            sig = inspect.signature(self.func)
            param_count = len(sig.parameters)
            if param_count == 1:
                # Assume it takes a single numeric or sequence input
                inputs = [10, 100, 1000]
            else:
                # Can't easily generate inputs for multi-parameter functions
                logger.warning(f"No inputs provided and couldn't generate default inputs for {self.func_name} with {param_count} parameters.")
                inputs = []

        # Perform static analysis
        theoretical = self._static_complexity_analysis()
        self.last_theoretical = theoretical

        # Perform dynamic analysis
        time_data, profile_stats = self._time_profile(inputs)
        self.last_time_data = time_data
        self.last_profile = profile_stats

        memory_data = self._memory_profile(inputs)
        self.last_memory = memory_data

        # Generate suggestions
        suggestions = self._generate_suggestions(theoretical, time_data, profile_stats, memory_data)
        self.last_suggestions = suggestions

        # Prepare results
        results = {
            "function_name": self.func_name,
            "model_architecture": self.model_architecture,
            "theoretical_complexity": {
                "big_o": theoretical[0],
                "big_theta": theoretical[1],
                "big_omega": theoretical[2]
            },
            "empirical_performance": {
                "time_measurements": time_data,
                "memory_measurements": memory_data
            },
            "profile_stats": profile_stats,
            "optimization_suggestions": suggestions
        }
        
        return results
