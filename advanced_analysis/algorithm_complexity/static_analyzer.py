"""
Static code analysis for algorithmic complexity.

This module provides tools for analyzing Python code to identify algorithmic patterns
and estimate theoretical time and space complexity.
"""

import ast
import logging
import inspect
from typing import Dict, List, Set, Optional, Any, Union, Tuple
import importlib
from collections import defaultdict
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

logger = logging.getLogger(__name__)

# Known ML libraries and their common imports
ML_LIBRARIES = {
    'tensorflow': ['tf', 'tensorflow'],
    'pytorch': ['torch'],
    'sklearn': ['sklearn', 'scikit-learn'],
    'numpy': ['np', 'numpy'],
    'pandas': ['pd', 'pandas']
}

# Common algorithmic patterns and their typical complexity
ALGORITHMIC_PATTERNS = {
    'linear_search': {'time': ['O(n)'], 'space': ['O(1)']},
    'binary_search': {'time': ['O(log n)'], 'space': ['O(1)']},
    'quicksort': {'time': ['O(n log n)', 'O(n²)'], 'space': ['O(log n)']},
    'mergesort': {'time': ['O(n log n)'], 'space': ['O(n)']},
    'bubble_sort': {'time': ['O(n²)'], 'space': ['O(1)']},
    'bfs': {'time': ['O(V+E)'], 'space': ['O(V)']},
    'dfs': {'time': ['O(V+E)'], 'space': ['O(V)']},
    'dijkstra': {'time': ['O(V² + E)', 'O((V+E)log V)'], 'space': ['O(V)']},
    'dynamic_programming': {'time': ['O(n²)', 'O(n)'], 'space': ['O(n)', 'O(n²)']},
    'matrix_multiply': {'time': ['O(n³)'], 'space': ['O(n²)']},
    'knn': {'time': ['O(n²)'], 'space': ['O(n)']},
    'kmeans': {'time': ['O(n*k*i)'], 'space': ['O(n+k)']}  # n=points, k=clusters, i=iterations
}

# ML-specific algorithmic patterns
ML_ALGORITHMIC_PATTERNS = {
    'linear_regression': {'time': ['O(n*d²)'], 'space': ['O(d²)']},  # n=samples, d=features
    'logistic_regression': {'time': ['O(n*d*i)'], 'space': ['O(d)']},  # i=iterations
    'decision_tree': {'time': ['O(n*d*log(n))'], 'space': ['O(n)']},
    'random_forest': {'time': ['O(t*n*d*log(n))'], 'space': ['O(t*n)']},  # t=trees
    'svm': {'time': ['O(n²*d)'], 'space': ['O(n)']},
    'neural_network': {'time': ['O(n*e*l*h²)'], 'space': ['O(l*h²)']},  # e=epochs, l=layers, h=hidden units
    'cnn': {'time': ['O(n*e*c*f²*k²)'], 'space': ['O(c*f²)']},  # c=channels, f=feature map size, k=kernel size
    'rnn': {'time': ['O(n*e*s*h²)'], 'space': ['O(s*h²)']},  # s=sequence length
    'transformer': {'time': ['O(n*e*s²*h)'], 'space': ['O(s²*h)']},
    'kmeans': {'time': ['O(n*k*i*d)'], 'space': ['O(n+k)']},  # k=clusters, i=iterations
    'pca': {'time': ['O(n*d²)'], 'space': ['O(d²)']}
}

# Additional imports for model analysis
try:
    import thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
        
except ImportError:
    TORCH_AVAILABLE = False
    FVCORE_AVAILABLE = False
    THOP_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configurations for analysis
ML_OPERATIONS = {
    'conv': ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'],
    'norm': ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm'],
    'act': ['ReLU', 'LeakyReLU', 'PReLU', 'ReLU6', 'ELU', 'GELU', 'Sigmoid', 'Tanh'],
    'pool': ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d'],
    'recurrent': ['LSTM', 'GRU', 'RNN'],
    'attention': ['MultiheadAttention'],
    'linear': ['Linear'],
    'dropout': ['Dropout'],
}

# Training data for optimization recommendations
OPTIMIZATION_PATTERNS = {
    'nested_loop_with_numpy': {
        'pattern': 'nested_loops_with_numpy_usage',
        'recommendation': 'Replace nested loops with vectorized operations using numpy.einsum or matrix operations',
        'historical_impact': 'Typical speedup: 10-100x'
    },
    'pytorch_lstm_loop': {
        'pattern': 'for_loop_with_lstm_call',
        'recommendation': 'Use packed sequences and torch.nn.utils.rnn.pack_padded_sequence for variable length sequences',
        'historical_impact': 'Memory reduction: 30-60%, Speed improvement: 2-5x'
    },
    'repeated_tensor_allocation': {
        'pattern': 'repeated_torch_zeros_in_loop',
        'recommendation': 'Pre-allocate tensors outside loop and reuse',
        'historical_impact': 'Reduces memory fragmentation and improves speed by 5-20%'
    },
    'transformer_sequential_attention': {
        'pattern': 'sequential_attention_computation',
        'recommendation': 'Use torch.nn.MultiheadAttention with batch_first=True option',
        'historical_impact': 'Speed improvement: 2-3x on large batches'
    }
}

class VariableTracker:
    """Track variables and their relationships throughout the code"""
    
    def __init__(self):
        self.variables = {}  # name -> info
        self.constants = {}  # name -> value
        self.variable_relationships = defaultdict(list)  # name -> list of related vars
        
    def register_assignment(self, target_name, value_node):
        """Register a variable assignment"""
        self.variables[target_name] = {
            'assigned_at': getattr(value_node, 'lineno', None),
            'node_type': type(value_node).__name__,
            'is_constant': isinstance(value_node, (ast.Num, ast.Str, ast.NameConstant))
        }
        
        # Track actual constant values
        if isinstance(value_node, ast.Num):
            self.constants[target_name] = value_node.n
        elif isinstance(value_node, ast.Str):
            self.constants[target_name] = value_node.s
        elif isinstance(value_node, ast.NameConstant):
            self.constants[target_name] = value_node.value
            
        # Track variable relationships
        if isinstance(value_node, ast.Name):
            self.variable_relationships[target_name].append(value_node.id)
            self.variable_relationships[value_node.id].append(target_name)
        
    def is_constant(self, var_name):
        """Check if a variable is a constant"""
        return var_name in self.variables and self.variables[var_name].get('is_constant', False)
    
    def get_constant_value(self, var_name):
        """Get the value of a constant"""
        return self.constants.get(var_name, None)
    
    def get_related_variables(self, var_name):
        """Get variables related to the given variable"""
        return self.variable_relationships.get(var_name, [])

class StaticAnalyzer(ast.NodeVisitor):
    """
    Performs AST-based analysis of Python code to estimate theoretical time/space complexity
    and detect known algorithmic patterns.
    """
    def __init__(self):
        self.function_data = {}  # store structure info for each function
        self.current_function = None
        self.imported_modules = set()
        self.imported_functions = set()  # e.g. from known libs, so we can detect library calls
        self.loop_stack = []  # Track nesting depth
        self.loop_vars = set()  # Track variables used in loops
        self.current_loop_depth = 0
        self.max_loop_depth = 0
        self.call_graph = {}  # Track function call relationships
        self.ml_libraries_used = set()  # Track which ML libraries are used
        self.variable_tracker = VariableTracker()
        self.loop_bounds = {}  # Track loop bounds data
        self.slicing_ops = []  # Track slicing operations
        self.detected_ml_ops = defaultdict(list)  # Detected ML operations by category
        self.ml_layers = {}  # Track defined layers within models
        self.ml_model_classes = set()  # Track defined model classes
        self.attention_blocks = []  # Track attention blocks
        
        # For historical optimization recommendations
        self.optimization_patterns = {}
        self.found_optimization_patterns = []

    def visit_ImportFrom(self, node):
        """Track if known library calls are used, e.g. bisect, math, etc."""
        module_name = node.module if node.module else ""
        self.imported_modules.add(module_name)
        
        # Check if this is an ML library import
        for lib, aliases in ML_LIBRARIES.items():
            if module_name in aliases or module_name.startswith(f"{lib}."):
                self.ml_libraries_used.add(lib)
                
        for name in node.names:
            self.imported_functions.add(name.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        """Track imported modules"""
        for alias in node.names:
            module_name = alias.name
            self.imported_modules.add(module_name)
            
            # Check if this is an ML library import
            for lib, aliases in ML_LIBRARIES.items():
                if module_name in aliases:
                    self.ml_libraries_used.add(lib)
                    
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze function definitions"""
        prev_function = self.current_function
        self.current_function = node.name
        
        # Initialize function data
        self.function_data[node.name] = {
            'loops': [],
            'nested_loops': [],
            'max_loop_depth': 0,
            'calls': set(),
            'recursion': False,
            'time_complexity': 'O(1)',  # Default
            'space_complexity': 'O(1)',  # Default
            'line_number': node.lineno,
            'patterns': []
        }
        
        # Reset loop tracking for this function
        prev_loop_depth = self.current_loop_depth
        self.current_loop_depth = 0
        prev_max_depth = self.max_loop_depth
        self.max_loop_depth = 0
        
        # Visit function body
        self.generic_visit(node)
        
        # Update function data with loop info
        self.function_data[node.name]['max_loop_depth'] = self.max_loop_depth
        
        # Estimate complexity based on loop nesting
        if self.max_loop_depth > 0:
            # Basic complexity estimation based on loop nesting
            if self.max_loop_depth == 1:
                self.function_data[node.name]['time_complexity'] = 'O(n)'
            else:
                self.function_data[node.name]['time_complexity'] = f'O(n^{self.max_loop_depth})'
        
        # Check for recursion (function calls itself)
        if node.name in self.function_data[node.name]['calls']:
            self.function_data[node.name]['recursion'] = True
            # Simple recursion often implies O(n) or O(log n) but could be exponential
            # This is a simplification; actual analysis would be more complex
            self.function_data[node.name]['time_complexity'] = 'O(n) or O(2^n)'
        
        # Restore previous state
        self.current_function = prev_function
        self.current_loop_depth = prev_loop_depth
        self.max_loop_depth = prev_max_depth

    def visit_For(self, node):
        """Enhanced For loop analysis"""
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        
        # Track loop bounds information
        loop_info = {
            'type': 'for',
            'line': node.lineno,
            'depth': self.current_loop_depth,
            'constant_bounds': False,
            'is_slice': False,
            'is_divide_conquer': False,
            'is_sliding_window': False
        }
        
        # Check if iterating over a slice (potential divide & conquer)
        if isinstance(node.iter, ast.Subscript):
            loop_info['is_slice'] = True
            
            # Check if it's a slice operation
            if isinstance(node.iter.slice, ast.Slice):
                # For Python 3.9+
                if hasattr(node.iter.slice, 'lower') and node.iter.slice.lower and \
                   hasattr(node.iter.slice, 'upper') and node.iter.slice.upper:
                    
                    # Check for divide & conquer patterns - e.g., arr[mid:], arr[:mid]
                    if isinstance(node.iter.slice.lower, ast.Name) and 'mid' in node.iter.slice.lower.id.lower():
                        loop_info['is_divide_conquer'] = True
                    elif isinstance(node.iter.slice.upper, ast.Name) and 'mid' in node.iter.slice.upper.id.lower():
                        loop_info['is_divide_conquer'] = True
                        
                    # Check for sliding window patterns - e.g., i:i+window_size
                    if isinstance(node.iter.slice.lower, ast.Name) and \
                       isinstance(node.iter.slice.upper, ast.BinOp) and \
                       isinstance(node.iter.slice.upper.left, ast.Name) and \
                       node.iter.slice.lower.id == node.iter.slice.upper.left.id:
                        loop_info['is_sliding_window'] = True
                        
            # Record slicing operation for later analysis
            self.slicing_ops.append({
                'line': node.lineno,
                'slice': node.iter,
                'in_function': self.current_function
            })
        
        # Check if bounds are constant
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == 'range':
                # Check if all range args are constants
                is_constant = True
                constant_values = []
                
                for arg in node.iter.args:
                    if isinstance(arg, ast.Num):
                        constant_values.append(arg.n)
                    elif isinstance(arg, ast.Name) and self.variable_tracker.is_constant(arg.id):
                        constant_values.append(self.variable_tracker.get_constant_value(arg.id))
                    else:
                        is_constant = False
                        break
                        
                loop_info['constant_bounds'] = is_constant
                if is_constant:
                    loop_info['bounds_values'] = constant_values
        
        if self.current_function:
            self.function_data[self.current_function]['loops'].append(loop_info)
            
            if self.current_loop_depth > 1:
                self.function_data[self.current_function]['nested_loops'].append(loop_info)
            
            # Update function properties based on loop characteristics
            if loop_info['is_divide_conquer']:
                if 'algorithmic_patterns' not in self.function_data[self.current_function]:
                    self.function_data[self.current_function]['algorithmic_patterns'] = []
                self.function_data[self.current_function]['algorithmic_patterns'].append('divide_and_conquer')
            
            if loop_info['is_sliding_window']:
                if 'algorithmic_patterns' not in self.function_data[self.current_function]:
                    self.function_data[self.current_function]['algorithmic_patterns'] = []
                self.function_data[self.current_function]['algorithmic_patterns'].append('sliding_window')
        
        # Store loop bounds info
        self.loop_bounds[node.lineno] = loop_info
        
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        self.current_loop_depth -= 1

    def visit_While(self, node):
        """Analyze while loops"""
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        
        if self.current_function:
            loop_info = {
                'type': 'while',
                'line': node.lineno,
                'depth': self.current_loop_depth
            }
            self.function_data[self.current_function]['loops'].append(loop_info)
            
            if self.current_loop_depth > 1:
                self.function_data[self.current_function]['nested_loops'].append(loop_info)
        
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        self.current_loop_depth -= 1

    def visit_Call(self, node):
        """Enhanced function call analysis"""
        # Track function calls for call graph
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if self.current_function:
                self.function_data[self.current_function]['calls'].add(func_name)
                
                # Update call graph
                if self.current_function not in self.call_graph:
                    self.call_graph[self.current_function] = set()
                self.call_graph[self.current_function].add(func_name)
        
        # Check for ML library calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module = node.func.value.id
            func = node.func.attr
            
            # Check if this is an ML library call
            for lib, aliases in ML_LIBRARIES.items():
                if module in aliases:
                    self.ml_libraries_used.add(lib)
                    
                    # If we're in a function, record the pattern
                    if self.current_function:
                        pattern = f"{module}.{func}"
                        self.function_data[self.current_function]['patterns'].append(pattern)
        
        # Handle attribute calls (e.g., torch.nn.Linear)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            module = None
            submodule = None
            func = None
            
            # Handle case: module.submodule.function()
            if isinstance(node.func.value.value, ast.Name):
                module = node.func.value.value.id
                submodule = node.func.value.attr
                func = node.func.attr
                
                # Detect ML framework operations
                if module in ('torch', 'tf', 'tensorflow'):
                    if submodule == 'nn' and func in sum(ML_OPERATIONS.values(), []):
                        # Find which category this operation belongs to
                        for category, ops in ML_OPERATIONS.items():
                            if func in ops:
                                self.detected_ml_ops[category].append({
                                    'name': f"{module}.{submodule}.{func}",
                                    'line': node.lineno,
                                    'args': [self._extract_arg_info(arg) for arg in node.args],
                                    'kwargs': {kw.arg: self._extract_arg_info(kw.value) for kw in node.keywords}
                                })
                                break
                                
                        # If we're in a function, record the ML operation
                        if self.current_function:
                            if 'ml_operations' not in self.function_data[self.current_function]:
                                self.function_data[self.current_function]['ml_operations'] = []
                            
                            self.function_data[self.current_function]['ml_operations'].append({
                                'op': f"{module}.{submodule}.{func}",
                                'line': node.lineno
                            })
                            
                        # Detect transformer attention blocks
                        if func == 'MultiheadAttention' or 'attention' in func.lower():
                            self.attention_blocks.append({
                                'line': node.lineno,
                                'function': self.current_function,
                                'details': f"{module}.{submodule}.{func}"
                            })
        
        # Detect optimization patterns in code
        if self.current_function and self.loop_stack:
            # Check for numpy usage in nested loops
            if len(self.loop_stack) > 1 and isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                self.found_optimization_patterns.append({
                    'pattern': 'nested_loop_with_numpy',
                    'line': node.lineno,
                    'function': self.current_function
                })
                
            # Check for LSTM in loops
            if isinstance(node.func, ast.Attribute) and \
               (('lstm' in node.func.attr.lower()) or ('rnn' in node.func.attr.lower())):
                self.found_optimization_patterns.append({
                    'pattern': 'pytorch_lstm_loop',
                    'line': node.lineno,
                    'function': self.current_function
                })
        
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Detect ML model class definitions"""
        parent_classes = []
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                parent_name = f"{base.value.id}.{base.attr}"
                parent_classes.append(parent_name)
            elif isinstance(base, ast.Name):
                parent_classes.append(base.name)
        
        # Check if this is a torch.nn.Module or tf.keras.Model subclass
        is_torch_module = any('nn.Module' in parent for parent in parent_classes)
        is_tf_model = any('keras.Model' in parent for parent in parent_classes)
        
        if is_torch_module or is_tf_model:
            self.ml_model_classes.add(node.name)
            
            # Initialize layers dict for this model
            self.ml_layers[node.name] = []
            
            # Track the original class for visiting its methods
            prev_function = self.current_function
            self.current_function = node.name
            
            # Visit class body
            self.generic_visit(node)
            
            # Restore previous function context
            self.current_function = prev_function
        else:
            self.generic_visit(node)
            
    def _extract_arg_info(self, arg_node):
        """Extract information from argument nodes"""
        if isinstance(arg_node, ast.Num):
            return arg_node.n
        elif isinstance(arg_node, ast.Str):
            return arg_node.s
        elif isinstance(arg_node, ast.Name):
            if self.variable_tracker.is_constant(arg_node.id):
                return self.variable_tracker.get_constant_value(arg_node.id)
            return arg_node.id
        elif isinstance(arg_node, ast.List) or isinstance(arg_node, ast.Tuple):
            return [self._extract_arg_info(elt) for elt in arg_node.elts]
        else:
            return "complex_expression"

    def visit_Assign(self, node):
        """Track variable assignments for constant propagation"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_tracker.register_assignment(target.id, node.value)
        self.generic_visit(node)

    def analyze_code(self, code_str: str) -> Dict[str, Any]:
        """
        Analyze Python code to estimate complexity and detect patterns.
        
        Args:
            code_str: Python code as a string
            
        Returns:
            Dictionary with analysis results
        """
        try:
            tree = ast.parse(code_str)
            self.visit(tree)
            
            # Analyze overall complexity
            overall_complexity = self._determine_overall_complexity()
            
            # Detect algorithmic patterns (improved with AST-based detection)
            patterns = self._detect_algorithmic_patterns(code_str)
            
            # Generate optimization recommendations based on found patterns
            optimization_recommendations = self._generate_optimization_recommendations()
            
            # Prepare results with enhanced data
            results = {
                'functions': self.function_data,
                'call_graph': self.call_graph,
                'overall_time_complexity': overall_complexity['time'],
                'overall_space_complexity': overall_complexity['space'],
                'ml_libraries_used': list(self.ml_libraries_used),
                'detected_patterns': patterns,
                'ml_operations': dict(self.detected_ml_ops),
                'ml_models': list(self.ml_model_classes),
                'attention_blocks': self.attention_blocks,
                'optimization_recommendations': optimization_recommendations,
                'loop_characteristics': self.loop_bounds
            }
            
            return results
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return {
                'error': f"Syntax error: {e}",
                'functions': {},
                'overall_time_complexity': 'Unknown',
                'overall_space_complexity': 'Unknown'
            }
    
    def _determine_overall_complexity(self) -> Dict[str, str]:
        """
        Determine the overall complexity of the code based on function complexities.
        
        Returns:
            Dictionary with time and space complexity
        """
        if not self.function_data:
            return {'time': 'O(1)', 'space': 'O(1)'}
        
        # Find the highest complexity among functions
        time_complexities = []
        space_complexities = []
        
        for func, data in self.function_data.items():
            time_complexities.append(data['time_complexity'])
            space_complexities.append(data.get('space_complexity', 'O(1)'))
        
        # Simple heuristic: take the highest complexity
        # This is a simplification; actual analysis would be more complex
        time_complexity = self._highest_complexity(time_complexities)
        space_complexity = self._highest_complexity(space_complexities)
        
        return {
            'time': time_complexity,
            'space': space_complexity
        }
    
    def _highest_complexity(self, complexities: List[str]) -> str:
        """
        Determine the highest complexity from a list of complexity strings.
        
        Args:
            complexities: List of complexity strings (e.g., ['O(1)', 'O(n)', 'O(n^2)'])
            
        Returns:
            Highest complexity string
        """
        # Order of complexity (from lowest to highest)
        complexity_order = [
            'O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 
            'O(n^2)', 'O(n^3)', 'O(2^n)', 'O(n!)'
        ]
        
        highest_idx = -1
        highest_complexity = 'O(1)'
        
        for complexity in complexities:
            # Handle special cases
            if 'or' in complexity:
                # Take the higher complexity when there's uncertainty
                parts = complexity.split(' or ')
                sub_highest = self._highest_complexity(parts)
                idx = complexity_order.index(sub_highest) if sub_highest in complexity_order else -1
            else:
                idx = complexity_order.index(complexity) if complexity in complexity_order else -1
            
            if idx > highest_idx:
                highest_idx = idx
                highest_complexity = complexity_order[idx]
        
        return highest_complexity
    
    def _detect_algorithmic_patterns(self, code_str: str) -> List[Dict[str, Any]]:
        """
        Detect known algorithmic patterns in the code using AST-based detection 
        and string matching as fallback.
        """
        patterns = []
        
        # Check for general algorithmic patterns
        for pattern_name, complexity in ALGORITHMIC_PATTERNS.items():
            if pattern_name.lower() in code_str.lower():
                patterns.append({
                    'pattern': pattern_name,
                    'time_complexity': complexity['time'],
                    'space_complexity': complexity['space']
                })
        
        # Check for ML-specific patterns
        for pattern_name, complexity in ML_ALGORITHMIC_PATTERNS.items():
            if pattern_name.lower() in code_str.lower():
                patterns.append({
                    'pattern': pattern_name,
                    'time_complexity': complexity['time'],
                    'space_complexity': complexity['space'],
                    'type': 'ml'
                })
        
        # Enhance with AST-based detection results from our traversal
        for function_name, data in self.function_data.items():
            if 'algorithmic_patterns' in data:
                for pattern_name in data['algorithmic_patterns']:
                    if pattern_name in ALGORITHMIC_PATTERNS:
                        complexity = ALGORITHMIC_PATTERNS[pattern_name]
                        patterns.append({
                            'pattern': pattern_name,
                            'time_complexity': complexity['time'],
                            'space_complexity': complexity['space'],
                            'in_function': function_name,
                            'detection_method': 'ast'
                        })
        
        return patterns
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on detected patterns"""
        recommendations = []
        
        for pattern in self.found_optimization_patterns:
            pattern_id = pattern['pattern']
            if pattern_id in OPTIMIZATION_PATTERNS:
                recommendations.append({
                    'location': f"Line {pattern['line']} in function {pattern['function']}",
                    'recommendation': OPTIMIZATION_PATTERNS[pattern_id]['recommendation'],
                    'historical_impact': OPTIMIZATION_PATTERNS[pattern_id]['historical_impact'],
                })
        
        return recommendations
    
    def analyze_model(self, model: Any) -> Dict[str, Any]:
        """
        Analyze a machine learning model's structure and performance characteristics.
        
        Args:
            model: A PyTorch or TensorFlow model
            
        Returns:
            Dictionary with model analysis results
        """
        if TORCH_AVAILABLE and hasattr(model, 'modules'):
            # Handle PyTorch models
            return self._analyze_pytorch_model(model)
        elif TF_AVAILABLE and hasattr(model, 'layers'):
            # Handle TensorFlow models
            return self._analyze_tensorflow_model(model)
        else:
            logger.error("Unsupported model type or required libraries not available")
            return {'error': 'Unsupported model type or required libraries not available'}
    
    def _analyze_pytorch_model(self, model: Any) -> Dict[str, Any]:
        """Analyze PyTorch model structure and estimate performance"""
        model_info = {
            'type': 'pytorch',
            'name': model.__class__.__name__,
            'layers': [],
            'param_count': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'complexity': {},
            'layer_breakdown': {}
        }
        
        # Collect layer information
        for name, module in model.named_modules():
            if name == '':  # Skip the top-level module
                continue
                
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters()),
                'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            model_info['layers'].append(layer_info)
        
        # Estimate FLOPs if possible
        if FVCORE_AVAILABLE and hasattr(model, 'eval'):
            try:
                model.eval()  # Set to evaluation mode
                # Create dummy input based on model's expected input
                # This is a simplified approach; may need adaptation for complex models
                dummy_input = None
                
                # Try to find expected input shape from model's first parameter
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    if len(first_param.shape) >= 2:
                        # For CNN models, create a 4D tensor [batch, channels, height, width]
                        dummy_input = torch.zeros(1, first_param.shape[1], 224, 224)
                    else:
                        # For MLP models, create a 2D tensor [batch, features]
                        dummy_input = torch.zeros(1, first_param.shape[0])
                
                if dummy_input is not None:
                    flops = FlopCountAnalysis(model, dummy_input)
                    model_info['complexity']['flops'] = flops.total()
                    model_info['layer_breakdown']['flops'] = flops.by_module()
                    model_info['complexity']['flops_table'] = flop_count_table(flops)
            except Exception as e:
                logger.warning(f"Error estimating FLOPs: {e}")
                
        elif THOP_AVAILABLE:
            try:
                dummy_input = torch.zeros(1, 3, 224, 224)  # Assume standard image input
                macs, params = thop.profile(model, inputs=(dummy_input,))
                model_info['complexity']['macs'] = macs
                model_info['complexity']['params'] = params
            except Exception as e:
                logger.warning(f"Error estimating MACs with THOP: {e}")
        
        return model_info
    
    def _analyze_tensorflow_model(self, model: Any) -> Dict[str, Any]:
        """Analyze TensorFlow model structure and estimate performance"""
        model_info = {
            'type': 'tensorflow',
            'name': model.__class__.__name__,
            'layers': [],
            'param_count': model.count_params(),
            'trainable_params': sum(w.numpy().size for w in model.trainable_weights),
            'complexity': {},
            'layer_breakdown': {}
        }
        
        # Collect layer information
        for i, layer in enumerate(model.layers):
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params(),
                'trainable_params': sum(w.numpy().size for w in layer.trainable_weights) 
                                  if hasattr(layer, 'trainable_weights') else 0,
                'output_shape': str(layer.output_shape)
            }
            model_info['layers'].append(layer_info)
        
        # Estimate FLOPs if possible
        if TF_AVAILABLE and hasattr(tf, 'profiler'):
            try:
                # Create dummy input
                input_shape = model.input_shape
                if input_shape:
                    import numpy as np
                    dummy_input = np.zeros([1] + list(input_shape[1:]))
                    
                    # Profile model
                    with tf.profiler.experimental.Profile('logdir'):
                        model(dummy_input)
                    # Extract profiling results
                    # This is simplified; actual implementation would extract data from TF profiler
                    model_info['complexity']['profiled'] = True
            except Exception as e:
                logger.warning(f"Error profiling TensorFlow model: {e}")
        
        return model_info
    
    def generate_report(self, analysis_results: Dict[str, Any], format: str = 'text') -> str:
        """
        Generate a formatted report of analysis results.
        
        Args:
            analysis_results: Results from analyze_code or analyze_model
            format: Output format ('text', 'html', 'markdown')
            
        Returns:
            Formatted report string
        """
        if format == 'text':
            return self._generate_text_report(analysis_results)
        elif format == 'html':
            return self._generate_html_report(analysis_results)
        elif format == 'markdown':
            return self._generate_markdown_report(analysis_results)
        else:
            return self._generate_text_report(analysis_results)  # Default to text
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate a plain text report"""
        report = ["==== Algorithm Complexity Analysis Report ====\n"]
        
        # Overall complexity
        report.append(f"Overall Time Complexity: {results.get('overall_time_complexity', 'Unknown')}")
        report.append(f"Overall Space Complexity: {results.get('overall_space_complexity', 'Unknown')}\n")
        
        # ML libraries used
        if 'ml_libraries_used' in results and results['ml_libraries_used']:
            report.append("ML Libraries Used:")
            for lib in results['ml_libraries_used']:
                report.append(f"  - {lib}")
            report.append("")
        
        # Detected patterns
        if 'detected_patterns' in results and results['detected_patterns']:
            report.append("Detected Algorithmic Patterns:")
            for pattern in results['detected_patterns']:
                report.append(f"  - {pattern['pattern']}: {', '.join(pattern['time_complexity'])}")
            report.append("")
        
        # Function details
        if 'functions' in results:
            report.append("Function Analysis:")
            for func_name, func_data in results['functions'].items():
                report.append(f"  {func_name}:")
                report.append(f"    - Time Complexity: {func_data.get('time_complexity', 'Unknown')}")
                report.append(f"    - Space Complexity: {func_data.get('space_complexity', 'Unknown')}")
                report.append(f"    - Max Loop Depth: {func_data.get('max_loop_depth', 0)}")
                if func_data.get('recursion', False):
                    report.append(f"    - Recursive: Yes")
                report.append("")
        
        # Optimization recommendations
        if 'optimization_recommendations' in results and results['optimization_recommendations']:
            report.append("Optimization Recommendations:")
            for rec in results['optimization_recommendations']:
                report.append(f"  - {rec['location']}: {rec['recommendation']}")
                report.append(f"    Expected impact: {rec['historical_impact']}")
            report.append("")
        
        # ML model specifics (if available)
        if 'layers' in results:
            report.append("Model Analysis:")
            report.append(f"  - Model Type: {results.get('type', 'Unknown')}")
            report.append(f"  - Total Parameters: {results.get('param_count', 0):,}")
            report.append(f"  - Trainable Parameters: {results.get('trainable_params', 0):,}")
            
            if 'complexity' in results and 'flops' in results['complexity']:
                report.append(f"  - Estimated FLOPs: {results['complexity']['flops']:,}")
            
            report.append("\n  Layers:")
            for layer in results.get('layers', []):
                report.append(f"    - {layer['name']} ({layer['type']}): {layer.get('params', 0):,} params")
            report.append("")
            
        return "\n".join(report)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate a markdown report"""
        # Similar to text report but with markdown formatting
        report = ["# Algorithm Complexity Analysis Report\n"]
        
        # Overall complexity
        report.append(f"**Overall Time Complexity:** {results.get('overall_time_complexity', 'Unknown')}")
        report.append(f"**Overall Space Complexity:** {results.get('overall_space_complexity', 'Unknown')}\n")
        
        # ML libraries used
        if 'ml_libraries_used' in results and results['ml_libraries_used']:
            report.append("## ML Libraries Used")
            for lib in results['ml_libraries_used']:
                report.append(f"- {lib}")
            report.append("")
        
        # Rest of the implementation similar to text report but with markdown syntax
        # ...
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an HTML report"""
        # HTML report implementation with proper formatting
        # ...
        
        return "<html>...</html>"  # Placeholder
