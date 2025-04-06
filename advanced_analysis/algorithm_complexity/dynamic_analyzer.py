"""
Dynamic analysis for algorithmic complexity.

This module provides tools for empirically measuring the runtime and memory usage
of functions across different input sizes to determine their actual complexity.
"""

import time
import tracemalloc
import logging
import math
import statistics
import asyncio
from typing import Dict, List, Tuple, Callable, Any, Optional, Union, Iterable
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Some memory profiling features will be limited.")

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not available. Memory profiling will use tracemalloc instead.")

# Check for plotting dependencies
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.info("matplotlib not available. Plotting features will be disabled.")

def detect_outliers(measurements: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[bool]:
    """
    Detect outliers in measurement data.
    
    Args:
        measurements: List of measurements
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        List of booleans where True indicates an outlier
    """
    if len(measurements) < 4:  # Too few points for reliable outlier detection
        return [False] * len(measurements)
        
    is_outlier = [False] * len(measurements)
    
    if method == 'iqr':
        # Interquartile Range method
        q1 = np.percentile(measurements, 25)
        q3 = np.percentile(measurements, 75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        
        for i, value in enumerate(measurements):
            if value < lower_bound or value > upper_bound:
                is_outlier[i] = True
                
    elif method == 'zscore':
        # Z-score method
        mean = np.mean(measurements)
        std = np.std(measurements)
        
        if std == 0:  # All values are the same
            return is_outlier
            
        for i, value in enumerate(measurements):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                is_outlier[i] = True
                
    return is_outlier

def measure_runtime(func: Callable, input_generator: Callable, sizes: List[int], 
                   repeats: int = 3, timeout: int = 5, 
                   outlier_detection: Optional[str] = None,
                   args_kwargs_generator: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Empirically measure runtime for given function across multiple input sizes.
    
    Args:
        func: Function to measure
        input_generator: Function that generates input of specified size
        sizes: List of input sizes to test
        repeats: Number of trials per size
        timeout: Maximum seconds per trial
        outlier_detection: Method for outlier detection ('iqr', 'zscore', or None)
        args_kwargs_generator: Function that generates additional args and kwargs
        
    Returns:
        Dictionary with runtime results and statistics
    """
    results = {}
    stats = {}
    
    for s in sizes:
        times_for_size = []
        all_inputs = []
        
        for _ in range(repeats):
            if args_kwargs_generator:
                main_input = input_generator(s)
                args, kwargs = args_kwargs_generator(s, main_input)
                all_inputs.append((main_input, args, kwargs))
                
                start_time = time.time()
                func(main_input, *args, **kwargs)
                elapsed = time.time() - start_time
            else:
                inp = input_generator(s)
                all_inputs.append(inp)
                
                start_time = time.time()
                func(inp)
                elapsed = time.time() - start_time
            
            times_for_size.append(elapsed)
            
            # Check if we've exceeded timeout
            if elapsed > timeout:
                logger.warning(f"Function took {elapsed:.2f}s for size {s}, exceeding timeout of {timeout}s. Skipping larger sizes.")
                break
        
        # Calculate statistics before outlier removal
        raw_stats = {
            'min': min(times_for_size) if times_for_size else 0,
            'max': max(times_for_size) if times_for_size else 0,
            'mean': statistics.mean(times_for_size) if times_for_size else 0,
            'median': statistics.median(times_for_size) if times_for_size else 0,
            'stdev': statistics.stdev(times_for_size) if len(times_for_size) > 1 else 0,
            'variance': statistics.variance(times_for_size) if len(times_for_size) > 1 else 0,
            'cv': (statistics.stdev(times_for_size) / statistics.mean(times_for_size)) 
                  if len(times_for_size) > 1 and statistics.mean(times_for_size) > 0 else 0
        }
        
        # Outlier detection and removal if specified
        filtered_times = times_for_size
        outliers = []
        
        if outlier_detection and len(times_for_size) > 3:
            is_outlier = detect_outliers(times_for_size, method=outlier_detection)
            outliers = [times_for_size[i] for i in range(len(times_for_size)) if is_outlier[i]]
            filtered_times = [times_for_size[i] for i in range(len(times_for_size)) if not is_outlier[i]]
            
        # Calculate statistics after outlier removal (if any)
        if filtered_times:
            filtered_stats = {
                'min': min(filtered_times),
                'max': max(filtered_times),
                'mean': statistics.mean(filtered_times),
                'median': statistics.median(filtered_times),
                'stdev': statistics.stdev(filtered_times) if len(filtered_times) > 1 else 0,
                'variance': statistics.variance(filtered_times) if len(filtered_times) > 1 else 0,
                'cv': (statistics.stdev(filtered_times) / statistics.mean(filtered_times))
                      if len(filtered_times) > 1 and statistics.mean(filtered_times) > 0 else 0
            }
        else:
            filtered_stats = raw_stats
            
        results[s] = {
            'raw_measurements': times_for_size,
            'filtered_measurements': filtered_times,
            'outliers': outliers
        }
        
        stats[s] = {
            'raw': raw_stats,
            'filtered': filtered_stats,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(times_for_size)) * 100 if times_for_size else 0
        }
        
        # If the last run exceeded timeout, stop testing larger sizes
        if times_for_size and times_for_size[-1] > timeout:
            break
            
    return {
        'measurements': results,
        'statistics': stats
    }

def measure_memory(func: Callable, input_generator: Callable, sizes: List[int], 
                  repeats: int = 3, outlier_detection: Optional[str] = None,
                  args_kwargs_generator: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Empirically measure memory usage for given function across multiple input sizes.
    
    Args:
        func: Function to measure
        input_generator: Function that generates input of specified size
        sizes: List of input sizes to test
        repeats: Number of trials per size
        outlier_detection: Method for outlier detection ('iqr', 'zscore', or None)
        args_kwargs_generator: Function that generates additional args and kwargs
        
    Returns:
        Dictionary with memory results and statistics
    """
    results = {}
    stats = {}
    
    for s in sizes:
        memory_for_size = []
        
        for _ in range(repeats):
            if args_kwargs_generator:
                main_input = input_generator(s)
                args, kwargs = args_kwargs_generator(s, main_input)
                
                if MEMORY_PROFILER_AVAILABLE:
                    # Use memory_profiler if available (more accurate)
                    mem_usage = memory_usage(
                        (func, (main_input,), {'args': args, **kwargs}),
                        interval=0.01, max_iterations=1
                    )
                    peak_mem = max(mem_usage) if mem_usage else 0
                else:
                    # Fall back to tracemalloc
                    tracemalloc.start()
                    func(main_input, *args, **kwargs)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mem = peak / (1024 * 1024)  # Convert to MB
            else:
                inp = input_generator(s)
                
                if MEMORY_PROFILER_AVAILABLE:
                    # Use memory_profiler if available (more accurate)
                    mem_usage = memory_usage(
                        (func, (inp,), {}),
                        interval=0.01, max_iterations=1
                    )
                    peak_mem = max(mem_usage) if mem_usage else 0
                else:
                    # Fall back to tracemalloc
                    tracemalloc.start()
                    func(inp)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mem = peak / (1024 * 1024)  # Convert to MB
                
            memory_for_size.append(peak_mem)
        
        # Calculate statistics before outlier removal
        raw_stats = {
            'min': min(memory_for_size) if memory_for_size else 0,
            'max': max(memory_for_size) if memory_for_size else 0,
            'mean': statistics.mean(memory_for_size) if memory_for_size else 0,
            'median': statistics.median(memory_for_size) if memory_for_size else 0,
            'stdev': statistics.stdev(memory_for_size) if len(memory_for_size) > 1 else 0,
            'variance': statistics.variance(memory_for_size) if len(memory_for_size) > 1 else 0
        }
        
        # Outlier detection and removal if specified
        filtered_memory = memory_for_size
        outliers = []
        
        if outlier_detection and len(memory_for_size) > 3:
            is_outlier = detect_outliers(memory_for_size, method=outlier_detection)
            outliers = [memory_for_size[i] for i in range(len(memory_for_size)) if is_outlier[i]]
            filtered_memory = [memory_for_size[i] for i in range(len(memory_for_size)) if not is_outlier[i]]
            
        # Calculate statistics after outlier removal (if any)
        if filtered_memory:
            filtered_stats = {
                'min': min(filtered_memory),
                'max': max(filtered_memory),
                'mean': statistics.mean(filtered_memory),
                'median': statistics.median(filtered_memory),
                'stdev': statistics.stdev(filtered_memory) if len(filtered_memory) > 1 else 0,
                'variance': statistics.variance(filtered_memory) if len(filtered_memory) > 1 else 0
            }
        else:
            filtered_stats = raw_stats
            
        results[s] = {
            'raw_measurements': memory_for_size,
            'filtered_measurements': filtered_memory,
            'outliers': outliers
        }
        
        stats[s] = {
            'raw': raw_stats,
            'filtered': filtered_stats,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(memory_for_size)) * 100 if memory_for_size else 0
        }
            
    return {
        'measurements': results,
        'statistics': stats
    }

async def async_measure_runtime(func: Callable, input_generator: Callable, sizes: List[int],
                              repeats: int = 3, timeout: int = 5,
                              outlier_detection: Optional[str] = None,
                              args_kwargs_generator: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Asynchronously measure runtime across multiple input sizes.
    
    Uses a process pool to parallelize measurements for better scalability.
    
    Args:
        func: Function to measure
        input_generator: Function that generates input of specified size
        sizes: List of input sizes to test
        repeats: Number of trials per size
        timeout: Maximum seconds per trial
        outlier_detection: Method for outlier detection ('iqr', 'zscore', or None)
        args_kwargs_generator: Function that generates additional args and kwargs
        
    Returns:
        Dictionary with runtime results and statistics
    """
    # This function runs a single size measurement in a separate process
    def _measure_size(size, func, input_gen, repeats, timeout, outlier_detection, args_kwargs_gen):
        # Create a singleton result dict for this size
        size_result = measure_runtime(
            func, input_gen, [size], repeats, timeout, 
            outlier_detection, args_kwargs_gen
        )
        # Return just the data for this size
        return size, size_result['measurements'].get(size, {}), size_result['statistics'].get(size, {})
    
    # Use ProcessPoolExecutor to run measurements in parallel
    loop = asyncio.get_event_loop()
    
    results = {}
    stats = {}
    
    # Create partial function for each size
    tasks = []
    with ProcessPoolExecutor() as executor:
        for size in sizes:
            task = loop.run_in_executor(
                executor, 
                partial(
                    _measure_size, size, func, input_generator, 
                    repeats, timeout, outlier_detection, args_kwargs_generator
                )
            )
            tasks.append(task)
    
        # Wait for all tasks to complete
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_tasks:
            if isinstance(result, Exception):
                logger.error(f"Error in async measurement: {result}")
                continue
                
            size, measurements, size_stats = result
            results[size] = measurements
            stats[size] = size_stats
    
    return {
        'measurements': results,
        'statistics': stats
    }

def estimate_complexity(sizes: List[int], measurements: List[float]) -> Dict[str, Any]:
    """
    Estimate the complexity class based on empirical measurements.
    
    Args:
        sizes: List of input sizes
        measurements: List of measurements (e.g., runtime or memory usage)
        
    Returns:
        Dictionary with estimated complexity and fit quality
    """
    if len(sizes) < 3 or len(measurements) < 3:
        return {'complexity': 'Unknown', 'quality': 0, 'message': 'Not enough data points'}
    
    # Prepare data
    x = np.array(sizes)
    y = np.array(measurements)
    
    # Define complexity classes to test
    complexity_classes = {
        'O(1)': lambda n: np.ones_like(n),
        'O(log n)': lambda n: np.log(n),
        'O(n)': lambda n: n,
        'O(n log n)': lambda n: n * np.log(n),
        'O(n^2)': lambda n: n**2,
        'O(n^3)': lambda n: n**3,
        'O(2^n)': lambda n: 2**n
    }
    
    # Fit each complexity class
    best_complexity = None
    best_r2 = -float('inf')
    coefficients = {}
    
    for name, func in complexity_classes.items():
        try:
            # Apply the complexity function to input sizes
            transformed_x = func(x)
            
            # Skip if any values are infinite or NaN
            if np.any(np.isinf(transformed_x)) or np.any(np.isnan(transformed_x)):
                continue
                
            # Fit a linear model: y = a * f(x) + b
            coeffs = np.polyfit(transformed_x, y, 1)
            a, b = coeffs
            
            # Calculate predicted values
            y_pred = a * transformed_x + b
            
            # Calculate R^2 (coefficient of determination)
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            coefficients[name] = {'a': a, 'b': b, 'r2': r2}
            
            if r2 > best_r2:
                best_r2 = r2
                best_complexity = name
        except Exception as e:
            logger.warning(f"Error fitting {name}: {e}")
    
    if best_complexity is None:
        return {'complexity': 'Unknown', 'quality': 0, 'message': 'Could not fit any complexity class'}
    
    # Determine quality of fit
    quality = best_r2  # R^2 value (0 to 1)
    
    return {
        'complexity': best_complexity,
        'quality': quality,
        'coefficients': coefficients[best_complexity],
        'all_fits': coefficients
    }

def plot_complexity(sizes: List[int], measurements: List[float], 
                   title: str = "Complexity Analysis", 
                   xlabel: str = "Input Size", 
                   ylabel: str = "Time (seconds)",
                   show_fits: bool = True,
                   complexity_estimate: Optional[Dict[str, Any]] = None,
                   ax = None,
                   save_path: Optional[str] = None) -> Any:
    """
    Plot measurements and estimated complexity curves.
    
    Args:
        sizes: List of input sizes
        measurements: List of measurements
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_fits: Whether to show fits for different complexity classes
        complexity_estimate: Results from estimate_complexity
        ax: Matplotlib axis to plot on (creates new figure if None)
        save_path: Path to save the plot to (doesn't save if None)
        
    Returns:
        Matplotlib figure or None if plotting unavailable
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting requested but matplotlib is not available")
        return None
        
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        
    # Plot actual data points
    ax.plot(sizes, measurements, 'o-', color='blue', linewidth=2, markersize=8, label='Actual measurements')
    
    # Show fits for different complexity classes
    if show_fits and complexity_estimate and 'all_fits' in complexity_estimate:
        fits = complexity_estimate['all_fits']
        complexity_classes = {
            'O(1)': lambda n: np.ones_like(n),
            'O(log n)': lambda n: np.log(n),
            'O(n)': lambda n: n,
            'O(n log n)': lambda n: n * np.log(n),
            'O(n^2)': lambda n: n**2,
            'O(n^3)': lambda n: n**3,
            'O(2^n)': lambda n: 2**n
        }
        
        # Plot the best fit with a thicker line
        best_complexity = complexity_estimate.get('complexity')
        
        # Generate smooth curve for plotting
        x_smooth = np.linspace(min(sizes), max(sizes), 100)
        
        for name, fit_data in fits.items():
            if name in complexity_classes:
                func = complexity_classes[name]
                a, b = fit_data['a'], fit_data['b']
                y_pred = a * func(x_smooth) + b
                
                # Use thicker line and highlight the best fit
                if name == best_complexity:
                    ax.plot(x_smooth, y_pred, '--', linewidth=2.5, alpha=0.8, 
                           label=f"{name} (R²={fit_data['r2']:.3f}) - BEST FIT")
                else:
                    ax.plot(x_smooth, y_pred, '--', linewidth=1, alpha=0.5,
                           label=f"{name} (R²={fit_data['r2']:.3f})")
    
    # Configure plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Use logarithmic scale if range is large
    if max(sizes) / min(sizes) > 100:
        ax.set_xscale('log')
    
    if max(measurements) / min(measurements) > 100:
        ax.set_yscale('log')
        
    # Save plot if path specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig

class DynamicAnalyzer:
    """
    Performs dynamic analysis of functions to empirically determine their complexity.
    """
    
    def __init__(self):
        """Initialize the dynamic analyzer."""
        pass
        
    def analyze_function(self, func: Callable, input_generator: Callable, 
                        sizes: Optional[List[int]] = None, 
                        repeats: int = 3, 
                        timeout: int = 5,
                        outlier_detection: Optional[str] = None,
                        args_kwargs_generator: Optional[Callable] = None,
                        plot_results: bool = False,
                        plot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a function's runtime and memory complexity by measuring performance
        across different input sizes.
        
        Args:
            func: Function to analyze
            input_generator: Function that generates input of specified size
            sizes: List of input sizes to test (default: exponential sequence)
            repeats: Number of trials per size
            timeout: Maximum seconds per trial
            outlier_detection: Method for outlier detection ('iqr', 'zscore', or None)
            args_kwargs_generator: Function that generates additional args and kwargs
            plot_results: Whether to generate plots
            plot_path: Path to save plots (if None, plots are not saved)
            
        Returns:
            Dictionary with analysis results
        """
        # Default sizes if not provided (exponential sequence)
        if sizes is None:
            sizes = [10, 100, 1000, 10000]
            
        # Measure runtime
        runtime_result = measure_runtime(
            func, input_generator, sizes, repeats, timeout, 
            outlier_detection, args_kwargs_generator
        )
        
        # Measure memory usage
        memory_result = measure_memory(
            func, input_generator, sizes, repeats,
            outlier_detection, args_kwargs_generator
        )
        
        # Extract filtered measurements for complexity estimation
        sizes_list = sorted(runtime_result['measurements'].keys())
        
        runtime_list = [
            statistics.mean(runtime_result['measurements'][s]['filtered_measurements'])
            if runtime_result['measurements'][s]['filtered_measurements']
            else statistics.mean(runtime_result['measurements'][s]['raw_measurements'])
            for s in sizes_list
        ]
        
        memory_list = [
            statistics.mean(memory_result['measurements'][s]['filtered_measurements'])
            if memory_result['measurements'][s]['filtered_measurements']
            else statistics.mean(memory_result['measurements'][s]['raw_measurements'])
            for s in sizes_list
        ]
        
        # Estimate complexity
        time_complexity = estimate_complexity(sizes_list, runtime_list)
        space_complexity = estimate_complexity(sizes_list, memory_list)
        
        figures = {}
        if plot_results and PLOTTING_AVAILABLE:
            # Plot runtime complexity
            fig_runtime = plot_complexity(
                sizes_list, runtime_list,
                title=f"Runtime Complexity: {func.__name__}",
                ylabel="Time (seconds)",
                complexity_estimate=time_complexity,
                save_path=f"{plot_path}/{func.__name__}_runtime.png" if plot_path else None
            )
            
            # Plot memory complexity
            fig_memory = plot_complexity(
                sizes_list, memory_list,
                title=f"Memory Complexity: {func.__name__}",
                ylabel="Memory (MB)",
                complexity_estimate=space_complexity,
                save_path=f"{plot_path}/{func.__name__}_memory.png" if plot_path else None
            )
            
            figures = {
                'runtime': fig_runtime,
                'memory': fig_memory
            }
        
        # Prepare results
        results = {
            'function_name': func.__name__,
            'input_sizes': sizes_list,
            'runtime': {
                'data': runtime_result,
                'complexity': time_complexity
            },
            'memory': {
                'data': memory_result,
                'complexity': space_complexity
            },
            'summary': {
                'time_complexity': time_complexity['complexity'],
                'space_complexity': space_complexity['complexity'],
                'time_quality': time_complexity['quality'],
                'space_quality': space_complexity['quality'],
                'runtime_variance': {s: stats['filtered']['variance'] for s, stats in runtime_result['statistics'].items()},
                'memory_variance': {s: stats['filtered']['variance'] for s, stats in memory_result['statistics'].items()},
                'detected_outliers': {
                    'runtime': {s: len(data['outliers']) for s, data in runtime_result['measurements'].items()},
                    'memory': {s: len(data['outliers']) for s, data in memory_result['measurements'].items()}
                }
            }
        }
        
        if figures:
            results['figures'] = figures
            
        return results
    
    async def analyze_function_async(self, func: Callable, input_generator: Callable, 
                                   sizes: Optional[List[int]] = None, 
                                   repeats: int = 3, 
                                   timeout: int = 5,
                                   outlier_detection: Optional[str] = None,
                                   args_kwargs_generator: Optional[Callable] = None,
                                   plot_results: bool = False,
                                   plot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously analyze a function's runtime and memory complexity.
        
        Args:
            func: Function to analyze
            input_generator: Function that generates input of specified size
            sizes: List of input sizes to test (default: exponential sequence)
            repeats: Number of trials per size
            timeout: Maximum seconds per trial
            outlier_detection: Method for outlier detection ('iqr', 'zscore', or None)
            args_kwargs_generator: Function that generates additional args and kwargs
            plot_results: Whether to generate plots
            plot_path: Path to save plots (if None, plots are not saved)
            
        Returns:
            Dictionary with analysis results
        """
        # Default sizes if not provided (exponential sequence)
        if sizes is None:
            sizes = [10, 100, 1000, 10000]
            
        # Measure runtime asynchronously
        runtime_result = await async_measure_runtime(
            func, input_generator, sizes, repeats, timeout, 
            outlier_detection, args_kwargs_generator
        )
        
        # For memory measurement, we'll still use the synchronous version
        # since memory profiling often requires tight control over process state
        memory_result = measure_memory(
            func, input_generator, sizes, repeats,
            outlier_detection, args_kwargs_generator
        )
        
        # Extract filtered measurements for complexity estimation
        sizes_list = sorted(runtime_result['measurements'].keys())
        
        # Rest of the analysis is the same as in synchronous version
        # ...same complexity estimation, plotting, and results preparation...
        runtime_list = [
            statistics.mean(runtime_result['measurements'][s]['filtered_measurements'])
            if runtime_result['measurements'][s]['filtered_measurements']
            else statistics.mean(runtime_result['measurements'][s]['raw_measurements'])
            for s in sizes_list
        ]
        
        memory_list = [
            statistics.mean(memory_result['measurements'][s]['filtered_measurements'])
            if memory_result['measurements'][s]['filtered_measurements']
            else statistics.mean(memory_result['measurements'][s]['raw_measurements'])
            for s in sizes_list
        ]
        
        # Estimate complexity
        time_complexity = estimate_complexity(sizes_list, runtime_list)
        space_complexity = estimate_complexity(sizes_list, memory_list)
        
        figures = {}
        if plot_results and PLOTTING_AVAILABLE:
            # Same plotting code as in synchronous version
            fig_runtime = plot_complexity(
                sizes_list, runtime_list,
                title=f"Runtime Complexity: {func.__name__}",
                ylabel="Time (seconds)",
                complexity_estimate=time_complexity,
                save_path=f"{plot_path}/{func.__name__}_runtime.png" if plot_path else None
            )
            
            fig_memory = plot_complexity(
                sizes_list, memory_list,
                title=f"Memory Complexity: {func.__name__}",
                ylabel="Memory (MB)",
                complexity_estimate=space_complexity,
                save_path=f"{plot_path}/{func.__name__}_memory.png" if plot_path else None
            )
            
            figures = {
                'runtime': fig_runtime,
                'memory': fig_memory
            }
        
        # Prepare results
        results = {
            'function_name': func.__name__,
            'input_sizes': sizes_list,
            'runtime': {
                'data': runtime_result,
                'complexity': time_complexity
            },
            'memory': {
                'data': memory_result,
                'complexity': space_complexity
            },
            'summary': {
                'time_complexity': time_complexity['complexity'],
                'space_complexity': space_complexity['complexity'],
                'time_quality': time_complexity['quality'],
                'space_quality': space_complexity['quality'],
                'runtime_variance': {s: stats['filtered']['variance'] for s, stats in runtime_result['statistics'].items()},
                'memory_variance': {s: stats['filtered']['variance'] for s, stats in memory_result['statistics'].items()},
                'detected_outliers': {
                    'runtime': {s: len(data['outliers']) for s, data in runtime_result['measurements'].items()},
                    'memory': {s: len(data['outliers']) for s, data in memory_result['measurements'].items()}
                }
            }
        }
        
        if figures:
            results['figures'] = figures
            
        return results
