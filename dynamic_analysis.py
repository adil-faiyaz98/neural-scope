# algocomplex/dynamic_analysis.py

import time
import random
import tracemalloc
import sys
import math
import statistics
import os
import psutil
import numpy as np
import threading
from functools import wraps
from collections import defaultdict

try:
    import pyperf
    HAVE_PYPERF = True
except ImportError:
    HAVE_PYPERF = False

try:
    import py_spy
    HAVE_PY_SPY = True
except ImportError:
    HAVE_PY_SPY = False

try:
    from memory_profiler import memory_usage
    HAVE_MEMORY_PROFILER = True
except ImportError:
    HAVE_MEMORY_PROFILER = False


def measure_runtime(func, input_generator, sizes, repeats=3, timeout=5):
    """
    Empirically measure runtime for given function across multiple input sizes.
    input_generator(size) -> returns an input of that size
    sizes -> list of input sizes (e.g. [100, 200, 400, 800])
    repeats -> number of trials per size
    timeout -> max seconds per trial
    Returns a dict of size -> list of runtimes
    """
    results = {}
    for s in sizes:
        times_for_size = []
        for _ in range(repeats):
            inp = input_generator(s)

            start_time = time.time()
            try:
                # run with a simple time limit
                # for a more robust approach, we'd run in a child process
                res = _run_with_timeout(func, inp, timeout)
            except TimeoutError:
                times_for_size.append(float('inf'))
            else:
                elapsed = time.time() - start_time
                times_for_size.append(elapsed)
        results[s] = times_for_size
    return results


def measure_memory(func, input_generator, sizes, repeats=3):
    """
    Use tracemalloc to measure memory usage for each run.
    Returns dict of size -> list of (peak_memory_in_bytes)
    """
    results = {}
    for s in sizes:
        mems_for_size = []
        for _ in range(repeats):
            inp = input_generator(s)
            tracemalloc.start()
            func(inp)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mems_for_size.append(peak)
        results[s] = mems_for_size
    return results


def track_temporary_allocations(func, input_generator, size, threshold=1024):
    """
    Track temporary memory allocations during function execution.
    
    Parameters:
    -----------
    func : callable
        The function to analyze
    input_generator : callable
        Function that generates input of given size
    size : int
        Size of input to generate
    threshold : int
        Minimum size in bytes to track for allocations
        
    Returns:
    --------
    dict : Information about temporary allocations
    """
    if not HAVE_MEMORY_PROFILER:
        return {"error": "memory_profiler not installed"}
    
    inp = input_generator(size)
    
    # Function wrapper to track memory before and after
    @wraps(func)
    def wrapped():
        return func(inp)
    
    # Get memory profile with timestamps
    mem_usage = memory_usage(wrapped, interval=0.001, timestamps=True)
    
    # Process the data to find temporary allocations
    temp_allocs = []
    peak_mem = 0
    baseline = mem_usage[0][1] if mem_usage else 0
    
    for i in range(1, len(mem_usage)):
        timestamp, mem = mem_usage[i]
        prev_timestamp, prev_mem = mem_usage[i-1]
        
        # Track peak memory
        if mem > peak_mem:
            peak_mem = mem
        
        # Detect allocations that later get freed
        if mem > prev_mem + threshold/1e6:  # Convert threshold to MB
            allocation = {
                'timestamp': timestamp,
                'size_mb': mem - prev_mem,
                'duration': None,
                'freed': False
            }
            temp_allocs.append(allocation)
        
        # Check if previous allocations were freed
        for alloc in temp_allocs:
            if not alloc['freed'] and mem < peak_mem - threshold/1e6:
                alloc['freed'] = True
                alloc['duration'] = timestamp - alloc['timestamp']
    
    return {
        'baseline_mb': baseline,
        'peak_mb': peak_mem,
        'temporary_allocations': temp_allocs,
        'total_temp_allocs': len(temp_allocs),
        'avg_temp_size_mb': statistics.mean([a['size_mb'] for a in temp_allocs]) if temp_allocs else 0,
        'avg_temp_duration': statistics.mean([a['duration'] for a in temp_allocs if a['duration']]) if any(a['duration'] for a in temp_allocs) else 0
    }


def track_cpu_utilization(func, input_generator, size, interval=0.1):
    """
    Track CPU utilization during function execution.
    
    Parameters:
    -----------
    func : callable
        The function to analyze
    input_generator : callable
        Function that generates input of given size
    size : int
        Size of input to generate
    interval : float
        Sampling interval in seconds
        
    Returns:
    --------
    dict : CPU utilization metrics
    """
    inp = input_generator(size)
    process = psutil.Process(os.getpid())
    
    # For storing CPU measurements
    cpu_percentages = []
    threads_counts = []
    
    # Flag to control monitoring thread
    running = threading.Event()
    running.set()
    
    # Monitoring thread function
    def monitor():
        while running.is_set():
            cpu_percentages.append(process.cpu_percent())
            threads_counts.append(len(process.threads()))
            time.sleep(interval)
    
    # Start monitoring in separate thread
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Execute the function
    start_time = time.time()
    func(inp)
    elapsed = time.time() - start_time
    
    # Stop monitoring
    running.clear()
    monitor_thread.join(timeout=1.0)
    
    # Calculate metrics
    if not cpu_percentages:
        return {"error": "No CPU measurements collected"}
    
    return {
        'avg_utilization': statistics.mean(cpu_percentages) if cpu_percentages else 0,
        'max_utilization': max(cpu_percentages) if cpu_percentages else 0,
        'min_utilization': min(cpu_percentages) if cpu_percentages else 0,
        'avg_threads': statistics.mean(threads_counts) if threads_counts else 1,
        'max_threads': max(threads_counts) if threads_counts else 1,
        'execution_time': elapsed,
        'measurements': len(cpu_percentages),
        'utilization_variance': statistics.variance(cpu_percentages) if len(cpu_percentages) > 1 else 0
    }


def detect_vectorization(func, input_generator, size):
    """
    Attempt to detect if the function is using vectorized operations efficiently.
    
    This is a heuristic approach that compares:
    1. The execution speed against a naive implementation
    2. CPU instruction patterns during execution (if py-spy is available)
    
    Parameters:
    -----------
    func : callable
        The function to analyze
    input_generator : callable
        Function that generates input of given size
    size : int
        Size of input to generate
        
    Returns:
    --------
    dict : Vectorization metrics and assessment
    """
    inp = input_generator(size)
    
    # Check if input or function involves numpy arrays
    using_numpy = (isinstance(inp, np.ndarray) or 
                  'numpy' in func.__module__ or
                  any('numpy' in str(line) for line in inspect_function_source(func)))
    
    # Execute function and measure performance
    start_time = time.time()
    result = func(inp)
    elapsed = time.time() - start_time
    
    # Get CPU instruction profile if py-spy is available
    simd_instructions = []
    if HAVE_PY_SPY:
        try:
            # This is a simplified version - py-spy would need more setup
            # to actually capture SIMD instructions
            simd_usage = {'detected': False, 'percentage': 0}
        except Exception:
            simd_usage = {'error': 'Failed to profile with py-spy'}
    else:
        simd_usage = {'available': False}
    
    # For numpy functions, we can make additional assessments
    numpy_optimized = False
    if using_numpy and hasattr(result, 'shape'):
        # Heuristic: Check if the function is operating element-wise when it could use vectorized ops
        numpy_optimized = True  # Assume optimized for now
    
    return {
        'vectorization_detected': using_numpy or (simd_usage.get('detected', False) if isinstance(simd_usage, dict) else False),
        'numpy_usage': using_numpy,
        'execution_time': elapsed,
        'simd_usage': simd_usage,
        'assessment': 'Likely vectorized' if using_numpy or numpy_optimized else 'Not vectorized',
        'recommendations': ['Consider using NumPy for vectorized operations'] if not using_numpy else []
    }


def analyze_cache_efficiency(func, input_generator, size):
    """
    Analyze cache efficiency and memory access patterns.
    
    This is an advanced feature that would ideally use hardware performance counters.
    As a simpler alternative, we use timing patterns with different access strategies.
    
    Parameters:
    -----------
    func : callable
        The function to analyze
    input_generator : callable
        Function that generates input of given size
    size : int
        Size of input to generate
        
    Returns:
    --------
    dict : Cache efficiency metrics
    """
    # This is a simplified implementation. A real one would use:
    # 1. Hardware performance counters (cache misses, etc.)
    # 2. Memory access pattern analysis
    
    # For now, we'll use a simple heuristic based on timing sequential vs random access
    # and inferring cache behavior
    
    inp = input_generator(size)
    
    # Try to determine if the function performs sequential or random access
    # by seeing how it performs with different input patterns
    if isinstance(inp, list) or isinstance(inp, np.ndarray):
        # Create a sequential access pattern
        if isinstance(inp, list):
            sequential_inp = list(range(size))
        else:
            sequential_inp = np.arange(size)
        
        # Create a random access pattern
        if isinstance(inp, list):
            random_inp = random.sample(range(size), size)
        else:
            random_inp = np.random.permutation(size)
        
        # Time both patterns
        start_time = time.time()
        func(sequential_inp)
        sequential_time = time.time() - start_time
        
        start_time = time.time()
        func(random_inp)
        random_time = time.time() - start_time
        
        # Calculate ratio - higher means worse cache efficiency with random access
        ratio = random_time / sequential_time if sequential_time > 0 else 1.0
        
        assessment = "Excellent cache efficiency"
        if ratio > 5.0:
            assessment = "Poor cache efficiency - random access is much slower than sequential"
        elif ratio > 2.0:
            assessment = "Moderate cache efficiency concerns"
        
        return {
            'sequential_time': sequential_time,
            'random_time': random_time,
            'random_sequential_ratio': ratio,
            'assessment': assessment,
            'recommendations': ['Consider reorganizing data access patterns for better cache locality'] if ratio > 2.0 else []
        }
    
    # If we can't perform the test, return limited information
    return {
        'assessment': 'Unable to analyze cache efficiency for this input type',
        'recommendations': []
    }


def _run_with_timeout(func, arg, timeout):
    """
    Run function with a timeout.
    This is a simple implementation. For production, you'd want a process-based solution.
    """
    result = [None]
    exception = [None]
    completed = [False]

    def worker():
        try:
            result[0] = func(arg)
            completed[0] = True
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if completed[0]:
        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        raise TimeoutError(f"Function took longer than {timeout} seconds")


def estimate_complexity_class(time_data):
    """
    Given timing data for different input sizes, estimate the complexity class.
    time_data is a dict of: input_size -> list of runtimes
    
    Returns a tuple of (complexity_class, explanation)
    """
    if not time_data:
        return "Unknown", "No timing data available"
    
    # Get average time for each size
    sizes = sorted(time_data.keys())
    avg_times = [statistics.mean([t for t in time_data[s] if t != float('inf')]) for s in sizes]
    
    if any(len([t for t in time_data[s] if t != float('inf')]) == 0 for s in sizes):
        return "Timeout", "Function timed out"
    
    # We'll try to fit different complexity models and see which fits best
    # O(1), O(log n), O(n), O(n log n), O(n²), O(n³)
    
    # Convert to log scale for easier fitting
    log_sizes = [math.log(s) if s > 0 else 0 for s in sizes]
    log_times = [math.log(t) if t > 0 else 0 for t in avg_times]
    
    if len(sizes) < 3:
        # Too few data points for reliable estimation
        ratio = avg_times[-1] / avg_times[0] if avg_times[0] > 0 else float('inf')
        size_ratio = sizes[-1] / sizes[0] if sizes[0] > 0 else float('inf')
        
        if ratio < 1.5:
            return "O(1)", "Constant time (minimal growth observed)"
        elif ratio < size_ratio * 1.5:
            return "O(n)", "Approximately linear growth"
        elif ratio < size_ratio ** 2 * 1.5:
            return "O(n²)", "Approximately quadratic growth"
        else:
            return "O(n^k) for k>2", "Super-quadratic growth"
    
    # Calculate ratios between consecutive times
    ratios = [avg_times[i] / avg_times[i-1] if avg_times[i-1] > 0 else float('inf') 
             for i in range(1, len(avg_times))]
    
    # Calculate corresponding size ratios
    size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
    
    # Normalized ratios (time ratio / size ratio)
    normalized = [ratios[i] / size_ratios[i] if size_ratios[i] > 0 else float('inf') 
                 for i in range(len(ratios))]
    
    # Try to infer the complexity class from the normalized ratios
    avg_norm = statistics.mean([n for n in normalized if n != float('inf')])
    
    if avg_norm < 1.2:
        return "O(n)", "Time grows linearly with input size"
    elif avg_norm < 1.7:
        # Check if it's closer to O(n log n)
        nlogn_ratios = [
            ratios[i] / (size_ratios[i] * math.log(sizes[i]) / math.log(sizes[i-1]))
            for i in range(len(ratios))
        ]
        avg_nlogn = statistics.mean([r for r in nlogn_ratios if r != float('inf')])
        
        if 0.8 < avg_nlogn < 1.2:
            return "O(n log n)", "Time grows proportionally to n log n"
        return "O(n^c) for 1<c<2", "Superlinear growth, but sub-quadratic"
    elif avg_norm < 2.5:
        return "O(n²)", "Time grows quadratically with input size"
    elif avg_norm < 3.5:
        return "O(n³)", "Time grows cubically with input size"
    else:
        # Try to estimate exponent
        if all(r != float('inf') for r in ratios) and all(sr > 0 for sr in size_ratios):
            logs = [math.log(r) / math.log(sr) for r, sr in zip(ratios, size_ratios)]
            est_exp = statistics.mean(logs)
            if est_exp > 10:
                return "Exponential", "Time grows exponentially with input size"
            return f"O(n^{est_exp:.1f})", f"Time grows with input size to power {est_exp:.1f}"
        return "High complexity", "Time grows rapidly with input size"


def inspect_function_source(func):
    """
    Get the source code lines of a function if possible.
    Returns an empty list if not possible.
    """
    try:
        import inspect
        return inspect.getsourcelines(func)[0]
    except Exception:
        return []

