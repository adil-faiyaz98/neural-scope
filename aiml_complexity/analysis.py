"""
Module: analysis.py

Provides code analysis functionality to identify potential inefficiencies in AI/ML code.
It parses Python code and detects patterns like nested loops or inefficient library usage (e.g., pandas iterrows).
Also estimates a relative complexity score and approximate operation count.
"""

import ast
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import multiprocessing

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    file: Optional[str]
    complexity_score: int
    estimated_operations: int
    inefficiencies: List[str]


def _estimate_operations(node: ast.AST, multiplier: int = 1) -> int:
    """
    Recursively estimate the number of operations represented by the AST node.
    Loops multiply the operations in their body by the number of iterations (estimated if constant).
    """
    ops = 0
    # If node is a loop, handle separately to multiply inner operations by iteration count
    if isinstance(node, ast.For) or isinstance(node, ast.While):
        iter_count = 50  # default assumption for unknown loop iterations
        if isinstance(node, ast.For):
            # Try to estimate iterations for 'for' loops with range or constant iterables
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func,
                                                              ast.Name) and node.iter.func.id == 'range':
                # If range has a constant stop value
                if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant) and isinstance(
                        node.iter.args[0].value, int):
                    iter_count = node.iter.args[0].value
                elif len(node.iter.args) >= 2:
                    # range(start, stop) or range(start, stop, step)
                    if isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[1], ast.Constant):
                        start_val = node.iter.args[0].value
                        stop_val = node.iter.args[1].value
                        if isinstance(start_val, int) and isinstance(stop_val, int):
                            iter_count = max(stop_val - start_val, 0)
            elif isinstance(node.iter, ast.Constant) and isinstance(node.iter.value, int):
                iter_count = node.iter.value  # unlikely scenario: for i in 100 (not typical, but just in case)
        elif isinstance(node, ast.While):
            # For while loops, we can't easily predict iteration count; assume default
            pass
        # Ensure non-negative iteration count
        if iter_count < 0:
            iter_count = 0
        # Cap iteration count to avoid insane multipliers for static analysis
        if iter_count > 1000000:
            iter_count = 1000000  # limit to 1e6 for safety
        # Compute operations in loop body
        for child in node.body:
            ops += _estimate_operations(child, multiplier * iter_count)
        # We do not traverse further down this node outside its body because we've handled it
        return ops
    # If not a loop:
    # Consider certain node types as contributing an operation count of 1 (multiplied by current multiplier)
    if isinstance(node, (ast.Call, ast.BinOp, ast.BoolOp, ast.Compare, ast.Assign, ast.AugAssign, ast.UnaryOp)):
        ops += multiplier
    # Traverse children for further operations
    for child in ast.iter_child_nodes(node):
        ops += _estimate_operations(child, multiplier)
    return ops


def _find_inefficiencies(node: ast.AST, inside_loop: bool = False) -> List[str]:
    """
    Recursively traverse AST and collect inefficiency warnings as strings.
    """
    issues = []
    # Check for nested loop
    if isinstance(node, (ast.For, ast.While)):
        if inside_loop:
            issues.append("Nested loops detected - consider reducing nested iterations or using vectorized operations.")
        # Check for large range in for loop
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func,
                                                                                        ast.Name) and node.iter.func.id == 'range':
            if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant) and isinstance(
                    node.iter.args[0].value, int):
                if node.iter.args[0].value >= 100:
                    issues.append(
                        f"Loop iterating {node.iter.args[0].value} times - consider optimizing or using bulk operations.")
            elif len(node.iter.args) >= 2:
                # range(start, stop)
                if isinstance(node.iter.args[1], ast.Constant) and isinstance(node.iter.args[1].value, int):
                    if node.iter.args[1].value - (
                    node.iter.args[0].value if isinstance(node.iter.args[0], ast.Constant) else 0) >= 100:
                        issues.append("Loop with very large range - consider optimizing or using batch processing.")
        # Recurse into loop body with inside_loop=True
        for child in node.body:
            issues.extend(_find_inefficiencies(child, inside_loop=True))
        # Additionally, check the loop iterable for inefficiencies (e.g., usage of iterrows)
        if isinstance(node, ast.For) and node.iter:
            issues.extend(_find_inefficiencies(node.iter, inside_loop))
        # No need to traverse further down this node (target handled by default recursion if needed)
        return issues
    # Check for inefficient pandas usage
    if isinstance(node, ast.Attribute):
        # If accessing an attribute named 'iterrows' (commonly pandas DataFrame.iterrows, which is slow)
        if node.attr == "iterrows":
            issues.append(
                "Pandas iterrows usage detected - consider using vectorized operations (itertuples or direct dataframe ops).")
    # Recurse for children
    for child in ast.iter_child_nodes(node):
        issues.extend(_find_inefficiencies(child, inside_loop))
    return issues


def analyze_code(code: str, filename: Optional[str] = None) -> AnalysisResult:
    """
    Analyze a string of Python code and return an AnalysisResult with complexity metrics and inefficiencies.

    :param code: The source code as a string.
    :param filename: Optional name of the code source (for reporting).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error while parsing code: {e}")
        raise
    # Perform analysis
    inefficiencies = _find_inefficiencies(tree)
    estimated_ops = _estimate_operations(tree)
    # Derive a simple complexity score (for now, use estimated_ops as the base, with a cap or scale for readability)
    complexity_score = estimated_ops
    # Optionally scale down complexity_score if too large to a simpler range or categorize (not needed at this stage)
    result = AnalysisResult(file=filename, complexity_score=complexity_score, estimated_operations=estimated_ops,
                            inefficiencies=inefficiencies)
    return result


def analyze_file(file_path: str) -> AnalysisResult:
    """
    Analyze a Python file at the given path and return an AnalysisResult.
    """
    logger.info(f"Analyzing file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise
    result = analyze_code(code, filename=file_path)
    logger.info(f"Completed analysis for {file_path}")
    return result


def analyze_directory(directory: str, recursive: bool = True, use_multiprocessing: bool = False) -> List[
    AnalysisResult]:
    """
    Analyze all Python files in a directory. Can be done in parallel if use_multiprocessing is True.

    :param directory: Path to directory containing Python files.
    :param recursive: Whether to search subdirectories recursively.
    :param use_multiprocessing: If True, analyze files in parallel processes for speed (useful for large codebases).
    :return: A list of AnalysisResult for each file.
    """
    import os
    file_paths = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    # Gather .py files
    if recursive:
        for root, _, files in os.walk(directory):
            for name in files:
                if name.endswith(".py"):
                    file_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            if name.endswith(".py"):
                file_paths.append(os.path.join(directory, name))
    results: List[AnalysisResult] = []
    if not file_paths:
        return results
    logger.info(
        f"Analyzing {len(file_paths)} files in directory {directory} (recursive={recursive}, parallel={use_multiprocessing})")
    if use_multiprocessing and len(file_paths) > 1:
        try:
            with multiprocessing.Pool() as pool:
                results = pool.map(analyze_file, file_paths)
        except Exception as e:
            logger.error(f"Parallel analysis failed, falling back to sequential. Error: {e}")
            results = [analyze_file(fp) for fp in file_paths]
    else:
        for fp in file_paths:
            try:
                res = analyze_file(fp)
                results.append(res)
            except Exception as e:
                logger.error(f"Analysis failed for {fp}: {e}")
                # Continue to next file on error
                continue
    return results
