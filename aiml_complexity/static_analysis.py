# algocomplex/static_analysis.py

import ast
import builtins

from patterns import detect_algorithmic_patterns

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

    def visit_ImportFrom(self, node):
        """Track if known library calls are used, e.g. bisect, math, etc."""
        module_name = node.module if node.module else ''
        self.imported_modules.add(module_name)
        
        # Track imported ML libraries
        for names in node.names:
            import_name = f"{module_name}.{names.name}"
            self.imported_functions.add(import_name)
            
            # Check if it's an ML library import
            for lib, aliases in ML_LIBRARIES.items():
                if module_name in aliases:
                    if self.current_function:
                        self.function_data[self.current_function]['ml_libraries'].add(lib)
                    break
        
        self.generic_visit(node)

    def visit_Import(self, node):
        """Track import usage"""
        for name in node.names:
            self.imported_modules.add(name.name)
            
            # Check if it's an ML library import
            for lib, aliases in ML_LIBRARIES.items():
                if name.name in aliases:
                    if self.current_function:
                        self.function_data[self.current_function]['ml_libraries'].add(lib)
                    break
                    
            # Handle aliased imports
            if name.asname:
                for lib, aliases in ML_LIBRARIES.items():
                    if name.name in aliases:
                        self.imported_functions.add(f"{name.asname}")
                        if self.current_function:
                            self.function_data[self.current_function]['ml_libraries'].add(lib)
        
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """
        Called when a function is defined. We'll gather data like #loops, recursion usage, etc.
        """
        func_name = node.name
        self.current_function = func_name
        self.loop_stack = []
        self.current_loop_depth = 0
        self.max_loop_depth = 0
        self.loop_vars = set()
        
        self.function_data[func_name] = {
            'loops': 0,
            'nested_loops_depth': 0,
            'recursion': False,
            'recursion_calls': 0,
            'calls': {},
            'library_calls': set(),
            'ml_libraries': set(),
            'lines_of_code': len(node.body),
            'patterns': set(),  # recognized algorithmic patterns
            'param_types': {},  # parameter type hints if available
            'docstring': ast.get_docstring(node),
            'complexity_drivers': [],  # what drives the complexity
            'time': [],  # time complexity estimates
            'space': []  # space complexity estimates
        }
        
        # Extract parameter type hints if available
        for arg in node.args.args:
            if hasattr(arg, 'annotation') and arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    self.function_data[func_name]['param_types'][arg.arg] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle generics like List[int]
                    if hasattr(arg.annotation.value, 'id'):
                        container_type = arg.annotation.value.id
                        self.function_data[func_name]['param_types'][arg.arg] = container_type
        
        # Recursively visit the contents of the function
        self.generic_visit(node)
        
        # Apply heuristics to detect algorithmic patterns
        self._detect_patterns(func_name)
        
        # Estimate time and space complexity
        self._estimate_complexity(func_name)
        
        self.current_function = None

    def visit_For(self, node):
        """Analyze for loops and nesting depth"""
        if self.current_function:
            self.function_data[self.current_function]['loops'] += 1
            
            # Track loop variables
            if isinstance(node.target, ast.Name):
                self.loop_vars.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self.loop_vars.add(elt.id)
            
            # Track nesting depth
            self.current_loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
            self.loop_stack.append(node)
            
            # Update nested_loops_depth
            self.function_data[self.current_function]['nested_loops_depth'] = max(
                self.function_data[self.current_function]['nested_loops_depth'],
                self.current_loop_depth
            )
            
            # Check for ML-specific loop patterns (e.g., loops over batches, epochs)
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range':
                    # Look for common ML training loop patterns
                    args = [arg for arg in node.iter.args]
                    if args and isinstance(args[0], ast.Name) and args[0].id == 'epochs':
                        self.function_data[self.current_function]['patterns'].add('training_loop')
                
                # Check for data loading loops
                if node.iter.func.id in ('DataLoader', 'batch_loader', 'get_batch'):
                    self.function_data[self.current_function]['patterns'].add('data_loading')
        
        self.generic_visit(node)
        
        if self.current_function:
            self.current_loop_depth -= 1
            if self.loop_stack:
                self.loop_stack.pop()

    def visit_While(self, node):
        """Analyze while loops and nesting depth"""
        if self.current_function:
            self.function_data[self.current_function]['loops'] += 1
            
            # Track nesting depth
            self.current_loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
            self.loop_stack.append(node)
            
            # Update nested_loops_depth
            self.function_data[self.current_function]['nested_loops_depth'] = max(
                self.function_data[self.current_function]['nested_loops_depth'],
                self.current_loop_depth
            )
        
        self.generic_visit(node)
        
        if self.current_function:
            self.current_loop_depth -= 1
            if self.loop_stack:
                self.loop_stack.pop()

    def visit_Call(self, node):
        """Analyze function calls, including recursive calls and library usage"""
        if not self.current_function:
            self.generic_visit(node)
            return
            
        # Get the function being called
        func_name = None
        is_method = False
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # This is a method call or a module.function call
            if isinstance(node.func.value, ast.Name):
                # Could be a module.function or object.method
                module_or_obj = node.func.value.id
                method_name = node.func.attr
                func_name = f"{module_or_obj}.{method_name}"
                
                # Check if it's an ML library call
                for lib, aliases in ML_LIBRARIES.items():
                    if module_or_obj in aliases:
                        self.function_data[self.current_function]['ml_libraries'].add(lib)
                        
                        # Add specific ML pattern detection
                        if lib == 'tensorflow' and method_name in ('keras', 'nn', 'layers'):
                            self.function_data[self.current_function]['patterns'].add('neural_network')
                        elif lib == 'torch' and method_name in ('nn', 'optim'):
                            self.function_data[self.current_function]['patterns'].add('neural_network')
                        elif lib == 'sklearn':
                            if 'KNeighbors' in method_name:
                                self.function_data[self.current_function]['patterns'].add('knn')
                            elif 'KMeans' in method_name:
                                self.function_data[self.current_function]['patterns'].add('kmeans')
                
                is_method = True
            
            # Check for typical ML method calls
            if node.func.attr in ('fit', 'predict', 'transform', 'fit_transform'):
                self.function_data[self.current_function]['patterns'].add('ml_pipeline')
                
            # Check for matrix operations
            if node.func.attr in ('dot', 'matmul', 'mm', 'bmm', 'einsum'):
                self.function_data[self.current_function]['patterns'].add('matrix_operation')
                
            # Check for tensor reshaping operations
            if node.func.attr in ('reshape', 'view', 'permute', 'transpose'):
                self.function_data[self.current_function]['patterns'].add('tensor_reshape')
        
        # If we couldn't determine the function name, skip further analysis
        if not func_name:
            self.generic_visit(node)
            return
        
        # Track the call
        if func_name not in self.function_data[self.current_function]['calls']:
            self.function_data[self.current_function]['calls'][func_name] = 0
        self.function_data[self.current_function]['calls'][func_name] += 1
        
        # Check for recursion
        if func_name == self.current_function or func_name.endswith(f".{self.current_function}"):
            self.function_data[self.current_function]['recursion'] = True
            self.function_data[self.current_function]['recursion_calls'] += 1
        
        # Track library calls
        if not is_method and func_name in dir(builtins):
            self.function_data[self.current_function]['library_calls'].add(func_name)
        elif func_name in self.imported_functions or any(func_name.startswith(f"{mod}.") for mod in self.imported_modules):
            self.function_data[self.current_function]['library_calls'].add(func_name)
            
            # Update call graph
            if func_name not in self.call_graph:
                self.call_graph[func_name] = set()
            self.call_graph[func_name].add(self.current_function)
        
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        """Analyze list comprehensions, which are implicit loops"""
        if self.current_function:
            # Implicit loop
            self.function_data[self.current_function]['loops'] += 1
            
            # Track nesting depth for nested comprehensions
            for generator in node.generators:
                # Check if the iterable is itself a comprehension
                if isinstance(generator.iter, ast.ListComp):
                    self.function_data[self.current_function]['nested_loops_depth'] += 1
        
        self.generic_visit(node)
        
    def visit_DictComp(self, node):
        """Analyze dictionary comprehensions, which are implicit loops"""
        if self.current_function:
            # Implicit loop
            self.function_data[self.current_function]['loops'] += 1
            
            # Check for nested comprehensions in the generators
            for generator in node.generators:
                if isinstance(generator.iter, (ast.ListComp, ast.DictComp, ast.SetComp)):
                    self.function_data[self.current_function]['nested_loops_depth'] += 1
        
        self.generic_visit(node)
        
    def visit_SetComp(self, node):
        """Analyze set comprehensions, which are implicit loops"""
        if self.current_function:
            # Implicit loop
            self.function_data[self.current_function]['loops'] += 1
            
            # Check for nested comprehensions in the generators
            for generator in node.generators:
                if isinstance(generator.iter, (ast.ListComp, ast.DictComp, ast.SetComp)):
                    self.function_data[self.current_function]['nested_loops_depth'] += 1
        
        self.generic_visit(node)
    
    def _detect_patterns(self, func_name):
        """
        Use heuristics to detect common algorithmic patterns based on collected data.
        """
        func_data = self.function_data[func_name]
        
        # Check for matrix operations
        if any('matrix' in call.lower() or 'mat' in call.lower() for call in func_data['calls']):
            func_data['patterns'].add('matrix_operation')
            
        # Check for sorting
        if 'sort' in func_data['calls'] or 'sorted' in func_data['calls']:
            func_data['patterns'].add('sorting')
            
        # Check for graph algorithms
        if ('graph' in func_name.lower() or 
            any('graph' in call.lower() for call in func_data['calls']) or
            any('edge' in call.lower() and 'node' in other_call.lower() 
                for call in func_data['calls'] for other_call in func_data['calls'])):
            func_data['patterns'].add('graph_algorithm')
            
        # Check for dynamic programming by looking for memoization patterns
        if func_data['recursion'] and (
            '@lru_cache' in func_data['docstring'] if func_data['docstring'] else False or
            '@memoize' in func_data['docstring'] if func_data['docstring'] else False or
            any('memo' in call.lower() or 'cache' in call.lower() for call in func_data['calls'])):
            func_data['patterns'].add('dynamic_programming')
            
        # Check for search algorithms
        if 'search' in func_name.lower() or 'find' in func_name.lower():
            if func_data['recursion'] or func_data['nested_loops_depth'] == 0:
                func_data['patterns'].add('binary_search')
            else:
                func_data['patterns'].add('linear_search')
                
        # Check for common ML algorithm patterns
        if any(lib in func_data['ml_libraries'] for lib in ['tensorflow', 'pytorch', 'sklearn']):
            # Neural network pattern detection
            if (any('layer' in call.lower() or 'conv' in call.lower() or 'pool' in call.lower() 
                   or 'relu' in call.lower() or 'sigmoid' in call.lower() for call in func_data['calls'])):
                func_data['patterns'].add('neural_network')
                
            # Training loop pattern detection
            if (func_data['loops'] > 0 and 
                any('train' in call.lower() or 'fit' in call.lower() or 'loss' in call.lower() 
                    or 'optim' in call.lower() for call in func_data['calls'])):
                func_data['patterns'].add('training_loop')
                
            # Data preprocessing pattern detection
            if any('preprocess' in call.lower() or 'transform' in call.lower() 
                  or 'normalize' in call.lower() or 'scale' in call.lower() for call in func_data['calls']):
                func_data['patterns'].add('data_preprocessing')
        
        # Use AST-based pattern detection
        detected_patterns = detect_algorithmic_patterns(None)  # Placeholder for future implementation
        func_data['patterns'].update(detected_patterns)

    def _estimate_complexity(self, func_name):
        """
        Estimate time and space complexity based on collected data.
        """
        func_data = self.function_data[func_name]
        
        # Start with default complexity estimates
        time_complexity = []
        space_complexity = []
        complexity_drivers = []
        
        # Check patterns first for known complexities
        for pattern in func_data['patterns']:
            if pattern in ALGORITHMIC_PATTERNS:
                time_complexity.extend(ALGORITHMIC_PATTERNS[pattern]['time'])
                space_complexity.extend(ALGORITHMIC_PATTERNS[pattern]['space'])
                complexity_drivers.append(f"Detected {pattern} pattern")
        
        # Heuristic complexity assessment based on loop nesting and recursion
        if not time_complexity:  # If no pattern-based estimate, use heuristics
            if func_data['nested_loops_depth'] == 0 and not func_data['recursion']:
                # Constant time operations
                time_complexity.append('O(1)')
                complexity_drivers.append("No loops or recursion")
            elif func_data['nested_loops_depth'] == 1 and not func_data['recursion']:
                # Linear time
                time_complexity.append('O(n)')
                complexity_drivers.append("Single loop level")
            elif func_data['nested_loops_depth'] == 2 and not func_data['recursion']:
                # Quadratic time
                time_complexity.append('O(n²)')
                complexity_drivers.append("Double nested loops")
            elif func_data['nested_loops_depth'] == 3 and not func_data['recursion']:
                # Cubic time
                time_complexity.append('O(n³)')
                complexity_drivers.append("Triple nested loops")
            elif func_data['nested_loops_depth'] > 3 and not func_data['recursion']:
                # Polynomial time with high degree
                time_complexity.append(f'O(n^{func_data["nested_loops_depth"]})')
                complexity_drivers.append(f"{func_data['nested_loops_depth']} levels of nested loops")
            elif func_data['recursion'] and func_data['recursion_calls'] == 1:
                # Simple recursion often indicates logarithmic or linear time
                # But this is a very rough heuristic
                if any('divide' in call.lower() or 'mid' in call.lower() or 'half' in call.lower() 
                       for call in func_data['calls']):
                    time_complexity.append('O(log n)')
                    complexity_drivers.append("Single recursive call with division/halving")
                else:
                    time_complexity.append('O(n)')
                    complexity_drivers.append("Single recursive call")
            elif func_data['recursion'] and func_data['recursion_calls'] > 1:
                # Multiple recursion calls often indicate exponential time
                time_complexity.append(f'O({func_data["recursion_calls"]}^n)')
                complexity_drivers.append(f"{func_data['recursion_calls']} recursive calls per level")
        
        # Space complexity estimates
        if not space_complexity:  # If no pattern-based estimate, use heuristics
            if func_data['recursion']:
                # Recursive algorithms typically use stack space proportional to depth
                if any('divide' in call.lower() or 'mid' in call.lower() or 'half' in call.lower() 
                       for call in func_data['calls']):
                    space_complexity.append('O(log n)')  # For divide-and-conquer
                else:
                    space_complexity.append('O(n)')  # For linear recursion
            else:
                # For iterative algorithms, check if we're building data structures
                if any('append' in call or 'extend' in call or 'add' in call 
                       or 'insert' in call for call in func_data['calls']):
                    if func_data['nested_loops_depth'] <= 1:
                        space_complexity.append('O(n)')
                    else:
                        space_complexity.append(f'O(n^{func_data["nested_loops_depth"]})')
                else:
                    space_complexity.append('O(1)')  # Constant extra space
        
        # ML-specific complexity estimates
        if 'neural_network' in func_data['patterns']:
            time_complexity.append('O(batch_size * neurons * iterations)')
            space_complexity.append('O(parameters + activations)')
            complexity_drivers.append("Neural network training/inference")
        
        if 'data_preprocessing' in func_data['patterns']:
            time_complexity.append('O(n * features)')
            space_complexity.append('O(n * features)')
            complexity_drivers.append("Data preprocessing operations")
            
        # Update function data
        func_data['time'] = list(set(time_complexity))  # Remove duplicates
        func_data['space'] = list(set(space_complexity))  # Remove duplicates
        func_data['complexity_drivers'] = complexity_drivers
        
        # Create a combined detail field for display
        func_data['detail'] = {
            'loops': func_data['loops'],
            'nested_loops_depth': func_data['nested_loops_depth'],
            'recursion': func_data['recursion'],
            'recursion_calls': func_data['recursion_calls'],
            'patterns': list(func_data['patterns']),
            'complexity_drivers': complexity_drivers,
            'library_calls': list(func_data['library_calls']),
            'ml_libraries': list(func_data['ml_libraries']),
            'calls': func_data['calls']
        }


def analyze_source_code(code_str: str):
    """
    Analyze Python source code to extract complexity information.
    Returns a dict mapping function names to their analysis results.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return {"error": f"Syntax error in code: {str(e)}"}
    
    analyzer = StaticAnalyzer()
    analyzer.visit(tree)
    
    results = {}
    
    # Process the raw function data into a more readable format
    for func_name, func_data in analyzer.function_data.items():
        results[func_name] = {
            'time': func_data['time'],
            'space': func_data['space'],
            'detail': func_data['detail'],
            'patterns': list(func_data['patterns']),
            'ml_libraries': list(func_data['ml_libraries']),
            'complexity_drivers': func_data['complexity_drivers']
        }
    
    return results
