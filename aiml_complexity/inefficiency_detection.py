"""
Module: aiml_pattern_recognition
--------------------------------
This module is part of the AI/ML Complexity Analysis tool. It provides functionality
to analyze Python code for known AI/ML algorithm patterns and detect common inefficiencies
in AI/ML code, along with recommendations for optimization.
"""
import ast

class AIMLPatternRecognizer:
    """
    AIMLPatternRecognizer analyzes Python code to identify AI/ML algorithms and inefficiencies.
    """
    # Define known algorithm patterns for identification
    ALGORITHM_PATTERNS = {
        "Decision Trees": ["DecisionTree", "RandomForest", "XGB", "LightGBM", "CART", "ID3", "GradientBoosting"],
        "Deep Learning - CNN": ["Conv2D", "Conv1D", "Conv3D", "Convolution", "Conv "],
        "Deep Learning - RNN": ["LSTM", "GRU", "SimpleRNN", "recurrent"],
        "Deep Learning - Transformer": ["Transformer", "MultiHeadAttention", "BERT", "GPT"],
        "Deep Learning - GAN": ["GAN", "Generator", "Discriminator"],
        "Optimization - Gradient Descent": ["gradient descent", "learning_rate", " lr", "backpropagation", "backward(", "optimizer.step"],
        "Optimization - Adam": ["Adam("],
        "Optimization - RMSProp": ["RMSProp"],
        "Optimization - SGD": ["SGD("],
        "Clustering - KMeans": ["KMeans", "kmeans"],
        "Clustering - DBSCAN": ["DBSCAN"],
        "Clustering - Hierarchical": ["AgglomerativeClustering", "linkage", "dendrogram"],
        "Graph-based ML - GNN": ["GraphConv", "GraphSAGE", "GraphAttention", "GAT", "GCN", "DGL", "pyg"],
        "Graph-based ML - PageRank": ["PageRank", "pagerank"],
        "Graph-based ML - Spectral": ["SpectralClustering", "SpectralEmbedding"]
    }
    def __init__(self):
        """Initialize the pattern recognizer (no internal state needed)."""
        pass

    def identify_algorithms(self, code_str):
        """
        Identify AI/ML algorithms present in the code based on known patterns.
        :param code_str: The source code as a string.
        :return: A list of detected algorithm types (e.g., 'CNN', 'Decision Trees', 'Gradient Descent').
        """
        detected_algorithms = []
        lower_code = code_str.lower()
        for alg_type, patterns in AIMLPatternRecognizer.ALGORITHM_PATTERNS.items():
            for pattern in patterns:
                # Use case-insensitive search for lowercase patterns (likely general terms)
                if pattern.islower():
                    if pattern in lower_code:
                        detected_algorithms.append(alg_type)
                        break
                else:
                    # Case-sensitive search for CamelCase or specific function/class names
                    if pattern in code_str:
                        detected_algorithms.append(alg_type)
                        break
        # Remove duplicates
        return list(set(detected_algorithms))

    def detect_inefficiencies(self, code_str):
        """
        Detect common inefficient code patterns in AI/ML workflows within the given code.
        :param code_str: The source code as a string.
        :return: A list of descriptions of inefficiencies found.
        """
        inefficiencies = []
        # Pattern 1: Unoptimized Batch Processing (training loop without batching)
        if "range(len(" in code_str or "enumerate(" in code_str:
            inefficiencies.append("Unoptimized Batch Processing - training loop processes samples individually without batching.")
        # Pattern 2: Inefficient Data Loading (data loaded inside a loop)
        if "pd.read_csv" in code_str or "open(" in code_str or "Image.open" in code_str:
            if "for " in code_str:  # likely inside a loop context
                inefficiencies.append("Inefficient Data Loading - data is loaded inside a Python loop instead of using optimized pipelines.")
        # Pattern 3: Redundant Matrix Computations inside loops (invariant computations repeated)
        try:
            tree = ast.parse(code_str)
            loop_invariant_issue_found = False
            # Visitor to identify loop-invariant heavy computations
            class LoopInvariantDetector(ast.NodeVisitor):
                def __init__(self):
                    self.current_loop_vars = []      # Stack of sets of loop variables for nested loops
                    self.current_loop_assigned = []  # Stack of sets of vars assigned inside the current loop
                def visit_For(self, node):
                    # Enter a for-loop: record loop variable(s)
                    if isinstance(node.target, ast.Name):
                        loop_vars = {node.target.id}
                    elif isinstance(node.target, ast.Tuple):
                        loop_vars = {elt.id for elt in node.target.elts if isinstance(elt, ast.Name)}
                    else:
                        loop_vars = set()
                    self.current_loop_vars.append(loop_vars)
                    self.current_loop_assigned.append(set())
                    # Visit the body of the loop
                    for n in node.body:
                        self.visit(n)
                    # Exit the loop
                    self.current_loop_vars.pop()
                    self.current_loop_assigned.pop()
                def visit_Assign(self, node):
                    # Record any assigned variable names inside a loop
                    if self.current_loop_vars:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.current_loop_assigned[-1].add(target.id)
                    self.generic_visit(node)
                def visit_AugAssign(self, node):
                    # Record augmented assignments (like x += ...) inside a loop
                    if self.current_loop_vars:
                        if isinstance(node.target, ast.Name):
                            self.current_loop_assigned[-1].add(node.target.id)
                    self.generic_visit(node)
                def visit_Call(self, node):
                    nonlocal loop_invariant_issue_found
                    if self.current_loop_vars:
                        # Collect all name identifiers used in the function call (function name and arguments)
                        names_in_call = set()
                        # Function name or attribute
                        if isinstance(node.func, ast.Name):
                            names_in_call.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            names_in_call.add(node.func.attr)
                            if isinstance(node.func.value, ast.Name):
                                names_in_call.add(node.func.value.id)
                        # Names in arguments
                        for arg in node.args:
                            for subnode in ast.walk(arg):
                                if isinstance(subnode, ast.Name):
                                    names_in_call.add(subnode.id)
                        # Determine if the call is loop-invariant (doesn't use loop vars or variables modified in loop)
                        all_loop_vars = set().union(*self.current_loop_vars) if self.current_loop_vars else set()
                        assigned_vars = set().union(*self.current_loop_assigned) if self.current_loop_assigned else set()
                        if names_in_call and names_in_call.isdisjoint(all_loop_vars) and names_in_call.isdisjoint(assigned_vars):
                            # Check for heavy computation patterns (common math/ML ops or library calls)
                            heavy_ops = ["dot", "matmul", "multiply", "inv", "inverse", "svd", "eig", "det", "solve", "transform", "predict", "fit"]
                            heavy_libs = ["np", "numpy", "tensorflow", "tf", "torch", "sklearn", "pandas", "pd"]
                            func_name = ""
                            if isinstance(node.func, ast.Name):
                                func_name = node.func.id
                            elif isinstance(node.func, ast.Attribute):
                                func_name = node.func.attr
                                # If it's an attribute call, also include base object name (to catch e.g. np.dot)
                                if isinstance(node.func.value, ast.Name):
                                    names_in_call.add(node.func.value.id)
                            # Flag as issue if it matches heavy operation or library usage
                            if any(lib in names_in_call for lib in heavy_libs) or (func_name and any(func_name.lower().startswith(op) for op in heavy_ops)):
                                loop_invariant_issue_found = True
                    # Continue traversing inside the call
                    self.generic_visit(node)
            # Run the visitor on the AST
            LoopInvariantDetector().visit(tree)
            if loop_invariant_issue_found:
                inefficiencies.append("Redundant Computation - a heavy operation inside a loop is repeated without variation (could be computed once outside).")
        except Exception:
            # If AST parsing fails (e.g., incomplete code), skip this analysis
            pass
        # Pattern 4: Inefficient Feature Engineering (non-vectorized DataFrame operations)
        if "iterrows" in code_str or "itertuples" in code_str:
            inefficiencies.append("Inefficient Feature Engineering - iterating over DataFrame rows (consider vectorized operations instead).")
        return inefficiencies

    def optimization_recommendations(self, inefficiencies):
        """
        Given a list of detected inefficiencies, provide corresponding optimization recommendations.
        :param inefficiencies: list of inefficiency descriptions.
        :return: list of suggested optimizations corresponding to each inefficiency.
        """
        recommendations = []
        for issue in inefficiencies:
            if "Batch Processing" in issue:
                recommendations.append("Use mini-batch processing for training. Process data in batches rather than one sample at a time to exploit vectorized operations.")
            elif "Data Loading" in issue:
                recommendations.append("Utilize optimized data pipelines (e.g., tf.data.Dataset or torch.utils.data.DataLoader) for data loading instead of reading files in each iteration.")
            elif "Redundant Computation" in issue:
                recommendations.append("Cache or move invariant computations outside the loop. Compute constant results once and reuse them, or vectorize the loop to eliminate repeated work.")
            elif "Feature Engineering" in issue:
                recommendations.append("Apply vectorized operations for feature engineering. Leverage Pandas or NumPy to transform entire columns or datasets at once instead of row-by-row processing.")
            else:
                recommendations.append("Refactor this code section for better efficiency.")
        return recommendations

    def analyze_code(self, code_str):
        """
        Analyze the given code string and return a structured summary of detected patterns and improvements.
        :param code_str: Python source code as string.
        :return: A dictionary with detected algorithms, inefficiencies, and recommendations.
        """
        algorithms = self.identify_algorithms(code_str)
        ineffs = self.detect_inefficiencies(code_str)
        recos = self.optimization_recommendations(ineffs)
        return {
            "algorithms_detected": algorithms,
            "inefficiencies_found": ineffs,
            "optimization_recommendations": recos
        }



analyzer = AIMLPatternRecognizer()
code_string = """
# Example code snippet
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Unbatched training loop
for i in range(len(X_train)):
    model.train_on_sample(X_train[i])

# Inefficient data loading
for file in files_list:
    data = pd.read_csv(file)
"""
result = analyzer.analyze_code(code_string)
print(result["algorithms_detected"])
print(result["inefficiencies_found"])
print(result["optimization_recommendations"])
