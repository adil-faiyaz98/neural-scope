# algocomplex/analyzer.py

import os
import tempfile
import numpy as np
import multiprocessing
from .static_analysis import analyze_source_code
from .dynamic_analysis import (
    measure_runtime, 
    measure_memory, 
    estimate_complexity_class,
    track_cpu_utilization,
    detect_vectorization,
    analyze_cache_efficiency,
    track_temporary_allocations
)
from .patterns import find_optimizations, detect_ml_inefficiencies
from .ml_optimizations import analyze_ml_pipeline, suggest_ml_improvements
from . import output
import statistics

class Analyzer:
    def __init__(self, 
                 gui=False, 
                 html=False, 
                 plot=True, 
                 adaptive_sampling=True,
                 track_cpu=True,
                 track_memory=True,
                 track_cache=False,
                 ml_optimization=False,
                 parallelism_analysis=True,
                 differential_analysis=False):
        """
        Initialize the analyzer with various options.
        
        Parameters:
        -----------
        gui: bool
            Whether to launch the GUI interface
        html: bool
            Whether to generate an HTML report
        plot: bool
            Whether to generate or embed runtime plots
        adaptive_sampling: bool
            Use adaptive sampling to find inflection points in algorithm behavior
        track_cpu: bool
            Track CPU utilization during execution
        track_memory: bool
            Track memory usage including peak and temporary allocations
        track_cache: bool
            Track cache efficiency and memory access patterns
        ml_optimization: bool
            Enable ML-specific optimizations and suggestions
        parallelism_analysis: bool
            Analyze and suggest multi-threading opportunities
        differential_analysis: bool
            Compare before/after optimizations or different implementations
        """
        self.gui = gui
        self.html = html
        self.plot = plot
        self.adaptive_sampling = adaptive_sampling
        self.track_cpu = track_cpu
        self.track_memory = track_memory
        self.track_cache = track_cache
        self.ml_optimization = ml_optimization
        self.parallelism_analysis = parallelism_analysis
        self.differential_analysis = differential_analysis
        
        # Store baseline measurements for differential analysis
        self.baseline_measurements = {}
        
        # CPU cores for parallelism recommendations
        self.cpu_cores = multiprocessing.cpu_count()

    def analyze_file(self, filepath):
        """Analyze a Python file for algorithmic complexity."""
        with open(filepath, 'r', encoding='utf-8') as f:
            code_str = f.read()
        return self.analyze_code(code_str)

    def analyze_code(self, code_str):
        """
        Analyze code string for time/space complexity and optimization opportunities.
        """
        # 1) Static analysis
        static_results = analyze_source_code(code_str)

        # Execute code in an isolated namespace
        local_ns = {}
        exec(code_str, local_ns)

        # Collect analysis results
        dynamic_results = {}
        memory_results = {}
        cpu_results = {}
        cache_results = {}
        ml_results = {}
        parallelism_results = {}
        vectorization_results = {}
        
        # Determine appropriate test sizes based on adaptive sampling if enabled
        if self.adaptive_sampling:
            test_sizes = self._determine_adaptive_test_sizes(static_results)
        else:
            test_sizes = [100, 200, 400, 800, 1600]  # Default test sizes
            
        # For each discovered function, perform dynamic analysis
        for func_name in static_results.keys():
            func_obj = local_ns.get(func_name, None)
            if not callable(func_obj):
                continue
                
            # Create appropriate input generator
            input_gen = self._create_input_generator(static_results[func_name])

            # Measure runtime performance
            time_data = measure_runtime(func_obj, input_gen, test_sizes)
            time_class, time_details = estimate_complexity_class(time_data)
            dynamic_results[func_name] = {
                'time_data': time_data,
                'time_class': time_class,
                'time_details': time_details
            }

            # Memory analysis
            if self.track_memory:
                mem_data = measure_memory(func_obj, input_gen, test_sizes)
                temp_allocs = track_temporary_allocations(func_obj, input_gen, test_sizes[-1])
                memory_results[func_name] = {
                    'peak_memory': mem_data,
                    'temp_allocations': temp_allocs
                }
            
            # CPU utilization
            if self.track_cpu:
                cpu_data = track_cpu_utilization(func_obj, input_gen, test_sizes[-1])
                cpu_results[func_name] = cpu_data
                
                # Check vectorization efficiency
                vec_data = detect_vectorization(func_obj, input_gen, test_sizes[-1])
                vectorization_results[func_name] = vec_data
            
            # Cache efficiency
            if self.track_cache:
                cache_data = analyze_cache_efficiency(func_obj, input_gen, test_sizes[-1])
                cache_results[func_name] = cache_data
                
            # ML-specific analysis
            if self.ml_optimization and 'ml_patterns' in static_results[func_name]:
                ml_analysis = analyze_ml_pipeline(func_obj, static_results[func_name])
                ml_improvements = suggest_ml_improvements(ml_analysis, self.cpu_cores)
                ml_results[func_name] = {
                    'analysis': ml_analysis,
                    'improvements': ml_improvements
                }
                
            # Parallelism analysis
            if self.parallelism_analysis:
                parallelism_data = self._analyze_parallelism_potential(
                    func_obj, static_results[func_name], cpu_results.get(func_name, None)
                )
                parallelism_results[func_name] = parallelism_data

        # Generate optimization suggestions
        for func_name, res in static_results.items():
            # General optimizations
            res['suggestions'] = find_optimizations(res['detail'])
            
            # ML-specific optimizations
            if self.ml_optimization:
                ml_suggestions = detect_ml_inefficiencies(res['detail'], 
                                                        dynamic_results.get(func_name, {}),
                                                        memory_results.get(func_name, {}))
                res['ml_suggestions'] = ml_suggestions

        # Produce final report
        if self.gui:
            output.show_interactive_report(
                static_results, dynamic_results, memory_results,
                cpu_results, cache_results, ml_results, 
                vectorization_results, parallelism_results,
                self.differential_analysis, self.baseline_measurements
            )
            return None
        elif self.html:
            html = output.generate_interactive_html(
                static_results, dynamic_results, memory_results,
                cpu_results, cache_results, ml_results,
                vectorization_results, parallelism_results,
                self.differential_analysis, self.baseline_measurements
            )
            return html
        else:
            text_report = output.generate_text_report(
                static_results, dynamic_results, memory_results,
                cpu_results, cache_results, ml_results,
                vectorization_results, parallelism_results
            )
            return text_report
            
    def _determine_adaptive_test_sizes(self, static_results):
        """
        Determine appropriate test sizes based on the estimated complexity.
        For efficient algorithms, we test with larger inputs.
        For inefficient algorithms, we use smaller inputs to prevent timeouts.
        """
        # Analyze static complexity to guess appropriate sizes
        worst_complexity = "O(1)"  # Default
        
        for func_name, res in static_results.items():
            if res['time']:
                # Take the worst complexity estimate
                for complexity in res['time']:
                    if 'n^3' in complexity or 'n³' in complexity:
                        worst_complexity = "O(n^3)"
                    elif 'n^2' in complexity or 'n²' in complexity and worst_complexity != "O(n^3)":
                        worst_complexity = "O(n^2)"
                    elif 'n log n' in complexity and worst_complexity not in ["O(n^3)", "O(n^2)"]:
                        worst_complexity = "O(n log n)"
                    elif 'n' in complexity and worst_complexity not in ["O(n^3)", "O(n^2)", "O(n log n)"]:
                        worst_complexity = "O(n)"
        
        # Determine sizes based on worst complexity
        if worst_complexity == "O(n^3)":
            return [10, 20, 40, 80, 160]
        elif worst_complexity == "O(n^2)":
            return [100, 200, 400, 800, 1600]
        elif worst_complexity == "O(n log n)":
            return [1000, 2000, 4000, 8000, 16000]
        elif worst_complexity == "O(n)":
            return [10000, 20000, 40000, 80000, 160000]
        else:
            return [100000, 200000, 400000, 800000, 1600000]
            
    def _create_input_generator(self, func_static_info):
        """
        Create an appropriate input generator based on function properties.
        """
        # Analyze parameters and function body to guess input types
        param_types = func_static_info.get('param_types', {})
        
        # Default generator - produces lists of integers
        def default_gen(size):
            return list(range(size))
            
        # Matrix generator for matrix operations
        def matrix_gen(size):
            return np.random.rand(size, size)
            
        # Graph generator for graph algorithms
        def graph_gen(size):
            # Create a simple adjacency list
            graph = {}
            for i in range(size):
                graph[i] = [j for j in range(size) if np.random.random() < 0.1]
            return graph
            
        # Check for common patterns to determine appropriate generator
        patterns = func_static_info.get('patterns', set())
        
        if 'matrix_operation' in patterns:
            return matrix_gen
        elif 'graph_algorithm' in patterns:
            return graph_gen
        else:
            return default_gen
            
    def _analyze_parallelism_potential(self, func_obj, static_info, cpu_data):
        """
        Analyze the potential for parallelizing the function.
        """
        result = {
            'parallelizable': False,
            'reasons': [],
            'suggestions': []
        }
        
        # Check for parallelizable patterns
        if 'loops' in static_info and static_info['loops'] > 0:
            result['parallelizable'] = True
            result['reasons'].append("Contains independent loops")
            result['suggestions'].append("Consider using OpenMP or multiprocessing for loop parallelization")
            
        # Check for CPU utilization
        if cpu_data and cpu_data['avg_utilization'] < 50 and self.cpu_cores > 1:
            result['reasons'].append(f"Low CPU utilization ({cpu_data['avg_utilization']}%) with {self.cpu_cores} cores available")
            result['suggestions'].append(f"Consider parallel execution to utilize available {self.cpu_cores} cores")
            
        # Check for vectorizable operations
        if 'numerical_operations' in static_info.get('patterns', set()):
            result['suggestions'].append("Consider using NumPy vectorization or Numba for numerical operations")
            
        return result
        
    def set_baseline(self, results):
        """
        Set baseline measurements for differential analysis.
        """
        self.baseline_measurements = results
        
    def compare_with_baseline(self, current_results):
        """
        Compare current results with baseline for differential analysis.
        """
        if not self.baseline_measurements:
            return {
                'error': 'No baseline measurements available for comparison'
            }
            
        comparison = {}
        
        # Compare runtime performance
        for func_name in current_results['dynamic']:
            if func_name in self.baseline_measurements.get('dynamic', {}):
                baseline_time = self.baseline_measurements['dynamic'][func_name]['time_class']
                current_time = current_results['dynamic'][func_name]['time_class']
                
                comparison[func_name] = {
                    'time_improvement': self._compare_complexity(baseline_time, current_time),
                    'memory_improvement': self._compare_memory(
                        self.baseline_measurements['memory'].get(func_name, {}),
                        current_results['memory'].get(func_name, {})
                    )
                }
                
        return comparison
        
    def _compare_complexity(self, baseline, current):
        """Compare complexity classes and return improvement estimate."""
        complexity_rank = {
            'O(1)': 1,
            'O(log n)': 2,
            'O(n)': 3,
            'O(n log n)': 4,
            'O(n^2)': 5,
            'O(n^3)': 6,
            'O(2^n)': 7
        }
        
        if baseline in complexity_rank and current in complexity_rank:
            baseline_rank = complexity_rank[baseline]
            current_rank = complexity_rank[current]
            
            if current_rank < baseline_rank:
                return f"Improved from {baseline} to {current}"
            elif current_rank > baseline_rank:
                return f"Regressed from {baseline} to {current}"
            else:
                return f"No change in complexity class: {current}"
        
        return "Unable to compare complexity classes"
        
    def _compare_memory(self, baseline_memory, current_memory):
        """Compare memory usage and return improvement estimate."""
        if not baseline_memory or not current_memory:
            return "No memory data available for comparison"
            
        try:
            baseline_peak = baseline_memory.get('peak_memory', {}).get(max(baseline_memory.get('peak_memory', {}).keys()), 0)
            current_peak = current_memory.get('peak_memory', {}).get(max(current_memory.get('peak_memory', {}).keys()), 0)
            
            if baseline_peak > 0 and current_peak > 0:
                improvement = (baseline_peak - current_peak) / baseline_peak * 100
                if improvement > 0:
                    return f"Memory usage reduced by {improvement:.1f}%"
                elif improvement < 0:
                    return f"Memory usage increased by {abs(improvement):.1f}%"
                else:
                    return "No significant change in memory usage"
        except (KeyError, ValueError):
            pass
            
        return "Unable to compare memory usage"
