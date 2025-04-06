#!/usr/bin/env python
"""
Example usage of the advanced_analysis module.

This script demonstrates how to use the advanced_analysis module to analyze code,
data, and models.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to the path to import advanced_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import StaticAnalyzer
from advanced_analysis.data_quality import DataGuardian

def main():
    """Run example analyses."""
    print("Advanced Analysis Example Usage")
    print("==============================\n")
    
    # Example 1: Analyze code complexity
    print("Example 1: Analyze code complexity")
    print("----------------------------------")
    
    # Define some sample code
    sample_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    
    # Create a static analyzer and analyze the code
    static_analyzer = StaticAnalyzer()
    results = static_analyzer.analyze_code(sample_code)
    
    # Display the results
    print("\nFunctions detected:")
    for func_name, func_data in results["functions"].items():
        print(f"  {func_name}: {func_data['time_complexity']}")
    
    print("\nOverall complexity:")
    print(f"  Time complexity: {results['overall_time_complexity']}")
    print(f"  Space complexity: {results['overall_space_complexity']}")
    
    # Example 2: Analyze data quality
    print("\n\nExample 2: Analyze data quality")
    print("------------------------------")
    
    # Create a sample DataFrame with various data quality issues
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 7],  # Duplicate value in id
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],  # Missing value in name
        'age': [25, 30, 35, 40, 45, 50, 200],  # Outlier in age
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
    })
    
    # Display the DataFrame
    print("\nSample DataFrame:")
    print(df)
    
    # Create a data guardian and analyze the data
    guardian = DataGuardian()
    report = guardian.generate_report(df)
    
    # Display the report
    print("\nData Quality Report:")
    print(f"  Missing values: {report.missing_values['total_missing']}")
    print(f"  Duplicates: {report.duplicates['total_duplicates']}")
    
    print("\nOutliers:")
    for column, outliers in report.outliers['outliers_by_column'].items():
        print(f"  {column}: {outliers}")
    
    # Example 3: Use the main Analyzer
    print("\n\nExample 3: Use the main Analyzer")
    print("-------------------------------")
    
    # Create an analyzer
    analyzer = Analyzer()
    
    # Analyze the sample code
    results = analyzer.analyze_code(sample_code)
    
    # Display the results
    print("\nStatic analysis:")
    print(f"  Overall time complexity: {results['static_analysis']['overall_time_complexity']}")
    
    print("\nVectorization analysis:")
    print(f"  Naive loops: {len(results['vectorization_analysis']['naive_loops'])}")
    
    print("\nOptimization suggestions:")
    for suggestion in results["optimization_suggestions"]:
        if isinstance(suggestion, dict) and "message" in suggestion:
            print(f"  - {suggestion['message']}")
        elif isinstance(suggestion, dict) and "suggestion" in suggestion:
            print(f"  - {suggestion['suggestion']}")
        else:
            print(f"  - {suggestion}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
