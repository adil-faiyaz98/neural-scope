# algocomplex/ml_optimizations.py

import importlib.util
import ast
import re
from collections import defaultdict

# Check for common ML libraries
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

# Define common inefficient ML patterns
ML_INEFFICIENT_PATTERNS = {
    'knn_large_dataset': {
        'pattern': r'KNeighbors(Classifier|Regressor)',
        'complexity': 'O(n²)',
        'recommendation': 'For large datasets, consider using approximate nearest neighbors (ANN) libraries like Annoy or FAISS'
    },
    'matrix_inversion': {
        'pattern': r'(np\.linalg\.inv|scipy\.linalg\.inv|torch\.inverse|tf\.linalg\.inv)',
        'complexity': 'O(n³)',
        'recommendation': 'Matrix inversion is O(n³). Consider alternatives like solving linear systems directly or using iterative methods'
    },
    'naive_bayes_large_features': {
        'pattern': r'GaussianNB',
        'complexity': 'O(nd)',
        'recommendation': 'With very high-dimensional data, consider feature selection or dimensionality reduction first'
    },
    'for_loop_tensor_ops': {
        'pattern': r'for .+ in .+:\s+.*(tensor|tf\.|torch\.|np\.)',
        'complexity': 'Varies',
        'recommendation': 'Replace Python loops with vectorized operations for better performance'
    },
    'non_batched_training': {
        'pattern': r'\.fit\([^,]+\)',
        'complexity': 'Varies',
        'recommendation': 'Use mini-batch training with appropriate batch sizes for better memory efficiency and parallelism'
    },
    'non_sparse_embeddings': {
        'pattern': r'Embedding\(',
        'complexity': 'O(vocabulary_size * embedding_dim)',
        'recommendation': 'For large vocabularies, consider using sparse embeddings or pruning techniques'
    },
    'sklearn_grid_search': {
        'pattern': r'GridSearchCV',
        'complexity': 'O(n_params * n_folds * model_complexity)',
        'recommendation': 'Consider more efficient hyperparameter tuning approaches like Bayesian optimization or Hyperband'
    }
}

def analyze_ml_pipeline(func_obj, static_info):
    """
    Analyze an ML pipeline for efficiency issues.
    
    Parameters:
    -----------
    func_obj : callable
        The function to analyze
    static_info : dict
        Static analysis information about the function
        
    Returns:
    --------
    dict : Analysis results containing detected ML patterns and issues
    """
    # Initialize results
    results = {
        'ml_framework': detect_ml_framework(func_obj, static_info),
        'detected_issues': [],
        'complexity_hotspots': [],
        'parallelism_opportunities': []
    }
    
    # Check for function source code
    source_code = get_function_source(func_obj)
    if not source_code:
        results['error'] = "Could not extract function source code"
        return results
    
    # Look for inefficient patterns in the code
    for pattern_name, pattern_info in ML_INEFFICIENT_PATTERNS.items():
        if re.search(pattern_info['pattern'], source_code):
            results['detected_issues'].append({
                'issue': pattern_name,
                'complexity': pattern_info['complexity'],
                'recommendation': pattern_info['recommendation']
            })
    
    # Check for parallelism opportunities
    parallelism_opportunities = detect_parallelism_opportunities(source_code, results['ml_framework'])
    results['parallelism_opportunities'] = parallelism_opportunities
    
    # Check for framework-specific patterns
    if 'tensorflow' in results['ml_framework']:
        tf_issues = check_tensorflow_optimizations(source_code)
        results['detected_issues'].extend(tf_issues)
    elif 'pytorch' in results['ml_framework']:
        torch_issues = check_pytorch_optimizations(source_code)
        results['detected_issues'].extend(torch_issues)
    elif 'sklearn' in results['ml_framework']:
        sklearn_issues = check_sklearn_optimizations(source_code)
        results['detected_issues'].extend(sklearn_issues)
    
    return results

def suggest_ml_improvements(ml_analysis, cpu_cores):
    """
    Generate specific improvement suggestions based on ML analysis results.
    
    Parameters:
    -----------
    ml_analysis : dict
        Results from analyze_ml_pipeline
    cpu_cores : int
        Number of available CPU cores
        
    Returns:
    --------
    list : Specific improvement suggestions
    """
    suggestions = []
    
    # Add basic suggestions from detected issues
    for issue in ml_analysis.get('detected_issues', []):
        suggestions.append(issue['recommendation'])
    
    # Add framework-specific optimization suggestions
    ml_framework = ml_analysis.get('ml_framework', [])
    
    if 'tensorflow' in ml_framework:
        suggestions.append(f"Enable TensorFlow's XLA compilation for faster execution")
        suggestions.append(f"Set intra_op_parallelism_threads={cpu_cores} in tf.config for better CPU utilization")
    
    elif 'pytorch' in ml_framework:
        suggestions.append("Enable PyTorch's torch.compile() for faster execution")
        if cpu_cores > 1:
            suggestions.append(f"Set num_workers={cpu_cores-1} in DataLoader for parallel data loading")
    
    elif 'sklearn' in ml_framework:
        suggestions.append(f"Set n_jobs={cpu_cores} for scikit-learn estimators that support parallelism")
    
    # Add parallelism suggestions
    for opportunity in ml_analysis.get('parallelism_opportunities', []):
        suggestions.append(opportunity['recommendation'])
    
    # Add general ML performance suggestions
    suggestions.append("Consider using mixed precision training for faster computation")
    suggestions.append("Profile memory usage during training to optimize batch sizes")
    
    return suggestions

def detect_ml_framework(func_obj, static_info):
    """
    Detect which ML frameworks are being used in the function.
    
    Parameters:
    -----------
    func_obj : callable
        The function to analyze
    static_info : dict
        Static analysis information about the function
        
    Returns:
    --------
    list : ML frameworks detected
    """
    # Get function source
    source_code = get_function_source(func_obj)
    frameworks = []
    
    # Check for imports in static analysis
    libraries = static_info.get('library_calls', set())
    
    # Check for framework usage patterns in the source code
    if re.search(r'import\s+tensorflow|import\s+tf|from\s+tensorflow\s+import|tf\.', source_code):
        frameworks.append('tensorflow')
    if re.search(r'import\s+torch|from\s+torch\s+import', source_code):
        frameworks.append('pytorch')
    if re.search(r'import\s+sklearn|from\s+sklearn\s+import', source_code):
        frameworks.append('sklearn')
    if re.search(r'import\s+numpy|import\s+np|from\s+numpy\s+import|np\.', source_code):
        frameworks.append('numpy')
    
    return frameworks

def detect_parallelism_opportunities(source_code, frameworks):
    """
    Detect opportunities for parallelism in ML code.
    
    Parameters:
    -----------
    source_code : str
        Source code of the function
    frameworks : list
        Detected ML frameworks
        
    Returns:
    --------
    list : Detected parallelism opportunities
    """
    opportunities = []
    
    # Check for loops that could be vectorized
    if re.search(r'for\s+.+\s+in\s+.+:.+\s+for\s+.+\s+in\s+.+:', source_code):
        opportunities.append({
            'type': 'nested_loops',
            'recommendation': 'Replace nested loops with vectorized operations or consider numba.jit decorators'
        })
    
    # Framework-specific parallelism checks
    if 'tensorflow' in frameworks:
        if not re.search(r'tf\.distribute\.MirroredStrategy', source_code):
            opportunities.append({
                'type': 'tf_distribution',
                'recommendation': 'Use tf.distribute.MirroredStrategy for multi-GPU training or tf.distribute.TPUStrategy for TPUs'
            })
    
    if 'pytorch' in frameworks:
        if not re.search(r'torch\.nn\.DataParallel|torch\.nn\.parallel\.DistributedDataParallel', source_code):
            opportunities.append({
                'type': 'torch_parallelism',
                'recommendation': 'Use torch.nn.DataParallel or DistributedDataParallel for multi-GPU training'
            })
    
    if 'sklearn' in frameworks:
        if re.search(r'\.fit\(', source_code) and not re.search(r'n_jobs\s*=', source_code):
            opportunities.append({
                'type': 'sklearn_parallelism',
                'recommendation': 'Set n_jobs=-1 in scikit-learn estimators to use all available cores'
            })
    
    return opportunities

def check_tensorflow_optimizations(source_code):
    """Check for TensorFlow-specific optimizations."""
    issues = []
    
    # Check for eager execution without compilation
    if re.search(r'tf\.function', source_code) is None and re.search(r'model\.fit|model\.predict', source_code):
        issues.append({
            'issue': 'tf_no_function',
            'complexity': 'N/A',
            'recommendation': 'Use @tf.function decorator to enable graph compilation for better performance'
        })
    
    # Check for mixed precision training
    if re.search(r'model\.fit', source_code) and not re.search(r'mixed_precision\.set_global_policy', source_code):
        issues.append({
            'issue': 'tf_no_mixed_precision',
            'complexity': 'N/A',
            'recommendation': 'Enable mixed precision training for faster computation and lower memory usage'
        })
    
    return issues

def check_pytorch_optimizations(source_code):
    """Check for PyTorch-specific optimizations."""
    issues = []
    
    # Check for missing model evaluation mode
    if re.search(r'model\(.+\)|net\(.+\)', source_code) and not re.search(r'\.eval\(\)|\.train\(\)', source_code):
        issues.append({
            'issue': 'torch_missing_eval',
            'complexity': 'N/A',
            'recommendation': 'Set model.eval() mode during inference to disable dropout and batchnorm updates'
        })
    
    # Check for manual gradient zeroing
    if re.search(r'loss\.backward\(\)', source_code) and not re.search(r'optimizer\.zero_grad\(\)', source_code):
        issues.append({
            'issue': 'torch_missing_zero_grad',
            'complexity': 'N/A',
            'recommendation': 'Call optimizer.zero_grad() before loss.backward() to prevent gradient accumulation'
        })
    
    return issues

def check_sklearn_optimizations(source_code):
    """Check for scikit-learn specific optimizations."""
    issues = []
    
    # Check for scalability issues in clustering
    if re.search(r'KMeans', source_code) and not re.search(r'MiniBatchKMeans', source_code):
        issues.append({
            'issue': 'sklearn_kmeans_scalability',
            'complexity': 'O(n²)',
            'recommendation': 'Use MiniBatchKMeans for large datasets to improve scalability'
        })
    
    # Check for tree-based ensembles without parallelism
    if re.search(r'Random(Forest|Trees)|GradientBoosting', source_code) and not re.search(r'n_jobs\s*=', source_code):
        issues.append({
            'issue': 'sklearn_missing_parallelism',
            'complexity': 'N/A',
            'recommendation': 'Set n_jobs parameter to utilize multiple cores in tree-based ensembles'
        })
    
    return issues

def get_function_source(func_obj):
    """Get the source code of a function."""
    try:
        import inspect
        return inspect.getsource(func_obj)
    except Exception:
        return "" 