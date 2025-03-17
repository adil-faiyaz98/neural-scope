# algocomplex/patterns.py

def find_optimizations(static_result):
    """
    Given the static result data for a function, identify potential improvements or common pitfalls.
    Returns a list of suggestions.
    """
    suggestions = []
    loops = static_result['loops']
    recursion = static_result['recursion']
    library_calls = static_result['library_calls']
    calls = static_result['calls']
    nested_loops_depth = static_result.get('nested_loops_depth', 0)

    # Complexity-related heuristics
    if loops > 1 and nested_loops_depth >= 2:
        suggestions.append("Nested loops detected (depth: {}); consider if there's a more efficient approach than O(n^{}).".format(
            nested_loops_depth, nested_loops_depth
        ))
    elif loops > 1 and not recursion:
        suggestions.append("Multiple loops detected; check if they can be combined or optimized.")
        
    if nested_loops_depth >= 3:
        suggestions.append("Deep nested loops (depth: {}) may indicate cubic O(n³) or worse complexity.".format(nested_loops_depth))
        
    if recursion and static_result['recursion_calls'] > 1:
        suggestions.append("Multiple recursive calls detected ({}); check if memoization or dynamic programming can reduce exponential blowup.".format(
            static_result['recursion_calls']
        ))
        
    # Algorithm-specific heuristics
    if 'sort' in calls:
        suggestions.append("Detected usage of sort inside function; ensure it's not called repeatedly in a bigger loop.")
        
    if 'in' in library_calls and 'list' in library_calls:
        suggestions.append("Using 'in' with a list can be O(n); consider a set or dict for membership tests if working with large data.")
        
    if any(call in calls for call in ['append', 'extend', 'insert']) and any(call in calls for call in ['pop', 'remove']):
        suggestions.append("Frequent insertions and deletions in lists can be expensive; consider using collections.deque for O(1) operations at both ends.")
        
    # Data structure recommendations
    if 'dict' in library_calls and any(call in calls for call in ['get', 'setdefault']):
        suggestions.append("Consider using collections.defaultdict or collections.Counter for cleaner and faster code with dictionaries.")
        
    # Memory efficiency
    if 'list' in library_calls and ('range' in calls or 'enumerate' in calls) and nested_loops_depth >= 2:
        suggestions.append("Consider using generators instead of materializing large lists for memory efficiency.")
        
    # Algorithmic patterns
    if 'fibonacci' in static_result.get('patterns', set()):
        suggestions.append("Fibonacci implementation detected; ensure you're using dynamic programming or iteration instead of naive recursion.")
        
    if 'matrix_multiply' in static_result.get('patterns', set()):
        suggestions.append("Matrix multiplication detected; consider using numpy or optimized linear algebra libraries.")
    
    return suggestions


def detect_ml_inefficiencies(static_info, dynamic_results, memory_results):
    """
    Detect ML-specific inefficiencies based on static analysis and runtime behavior.
    
    Parameters:
    -----------
    static_info : dict
        Static analysis results
    dynamic_results : dict
        Dynamic analysis results including time complexity
    memory_results : dict
        Memory profiling results
        
    Returns:
    --------
    list : ML-specific optimization suggestions
    """
    ml_suggestions = []
    
    # Check if this appears to be ML-related code
    ml_libraries = ['tensorflow', 'torch', 'sklearn', 'numpy', 'pandas']
    library_calls = static_info.get('library_calls', set())
    detected_ml = any(lib in str(library_calls).lower() for lib in ml_libraries)
    
    if not detected_ml:
        return ml_suggestions
    
    # Check for specific ML patterns
    patterns = static_info.get('patterns', set())
    
    # Check for matrix operations
    if 'matrix_operation' in patterns:
        ml_suggestions.append("Matrix operations detected; ensure you're using optimized linear algebra libraries.")
        
        # Check time complexity
        if dynamic_results.get('time_class', '').startswith('O(n^3)') or 'n³' in dynamic_results.get('time_class', ''):
            ml_suggestions.append("O(n³) complexity detected, which might be due to matrix inversion or naive matrix multiplication. Consider alternatives.")
    
    # Check for potentially inefficient ML algorithms
    if 'knn' in patterns or 'KNeighbors' in str(library_calls):
        ml_suggestions.append("KNN algorithm detected (O(n²) for prediction). For large datasets, consider approximate nearest neighbors or dimensionality reduction.")
    
    # Check for inefficient data processing
    if 'pandas' in str(library_calls) and static_info.get('loops', 0) > 0:
        ml_suggestions.append("Detected loops with pandas; prefer vectorized operations (apply/map) over explicit loops.")
    
    # Memory efficiency in ML
    if memory_results and memory_results.get('peak_memory') and any(mem > 500*1024*1024 for mem in memory_results.get('peak_memory', {}).values()):
        ml_suggestions.append("High memory usage detected. Consider batch processing or more memory-efficient algorithms.")
    
    # Tensorflow-specific suggestions
    if 'tensorflow' in str(library_calls).lower():
        ml_suggestions.append("For TensorFlow code, consider using tf.function decoration for improved performance.")
        
    # PyTorch-specific suggestions
    if 'torch' in str(library_calls).lower():
        ml_suggestions.append("For PyTorch code, ensure you're using model.eval() during inference and properly managing gradients.")
        
    return ml_suggestions


def detect_algorithmic_patterns(ast_node):
    """
    Analyze AST to detect common algorithmic patterns like sorting, searching, etc.
    Returns a set of detected patterns.
    """
    patterns = set()
    
    # TODO: Implement pattern recognition based on AST structure
    # This would look for things like:
    # - Sorting algorithms
    # - Search algorithms (binary search, linear search)
    # - Dynamic programming patterns
    # - Graph algorithms
    # - Matrix operations
    
    return patterns
