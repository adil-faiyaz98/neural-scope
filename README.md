# AI/ML Complexity Analysis

**AI/ML Complexity Analysis** is a Python library and CLI tool designed to help identify inefficiencies in AI/ML code and estimate the computational cost, including providing insights into potential AWS cloud costs. It analyzes Python code (especially data science and machine learning code) to detect patterns like nested loops or inefficient DataFrame operations, and suggests where performance might be improved.

## Features

- **Complexity Analysis**: Parse Python code to find potential inefficiencies (e.g., nested loops, use of pandas `.iterrows()`).
- **Complexity Scoring**: Compute a relative complexity score and estimate the number of operations in the code.
- **AWS Cost Insights**: Estimate the cost of running the workload on various AWS instance types (e.g., `m5.large`).
- **Profiling Utilities**: Functions to measure execution time and memory usage of code segments.
- **Storage**: Save and load analysis results to/from JSON files for later review.
- **CLI Tool**: Command-line interface to analyze files or entire projects, with results and cost estimates.

## Installation

Install the package via pip:

    pip install aiml_complexity

Or, if you have the source, install it and its dependencies:

    pip install -r requirements.txt
    pip install .

## Usage

### As a Python Library

Analyze a code snippet or file and get a report programmatically:

    from aiml_complexity import analyze_code, estimate_cost

    code = """
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3]})
    for idx, row in df.iterrows():
        print(row['a'])
    """
    result = analyze_code(code)
    print("Complexity Score:", result.complexity_score)
    print("Estimated Operations:", result.estimated_operations)
    print("Inefficiencies:", result.inefficiencies)
    # Estimate AWS cost for the operations on a default instance (m5.large)
    cost = estimate_cost(result.estimated_operations, instance_type="m5.large")
    print(f"Estimated cost on m5.large: ${cost:.6f}")

Typical output might be:

    Complexity Score: 3
    Estimated Operations: 3
    Inefficiencies: ['Pandas iterrows usage detected - consider using vectorized operations (itertuples or direct dataframe ops).']
    Estimated cost on m5.large: $0.000000

In this example, the tool flagged the use of `iterrows()` as an inefficiency and estimated the computational cost.

You can also profile specific functions:

    from aiml_complexity.profiling import profile_time, profile_memory

    def my_function(n):
        import time
        time.sleep(0.1)
        return [i for i in range(n)]

    result, elapsed = profile_time(my_function, 1000)
    print(f"Function result length = {len(result)}, time = {elapsed:.4f} seconds")

    result, peak_mem = profile_memory(my_function, 100000)
    print(f"Function peak memory usage = {peak_mem} bytes")

### Using the CLI Tool

After installing, you can use the CLI tool `aiml-complexity` to analyze files or directories. 

Examples:

- Analyze a single file:

      aiml-complexity path/to/script.py

  This will output the complexity score, estimated operations, any detected inefficiencies, and an estimated AWS cost for running the script on the default instance type.

- Analyze an entire project directory:

      aiml-complexity path/to/project

  The tool will find all `.py` files in the directory (recursively), analyze each, and print a summary for each file.

- Specify a different AWS instance for cost estimation:

      aiml-complexity --instance t2.micro path/to/script.py

  This will use the characteristics of a smaller instance (`t2.micro`) for cost estimation.

For more verbose output (including debug logs), use the `-v`/`--verbose` flag.

## Understanding the Output

For each analyzed file, the output will include:
- **Complexity Score**: a relative measure of complexity based on estimated operations. Higher means more computational work.
- **Estimated Operations**: approximate count of basic operations (like arithmetic or function calls) the code might perform.
- **Inefficiencies**: a list of human-readable warnings about patterns that could be optimized (if none are found, it will say "None").
- **Estimated Cost**: an approximate cost in USD to run those operations on the specified AWS instance type. This is calculated using predefined throughput and cost for that instance type.

## Best Practices for AI/ML Code Optimization

- **Avoid Nested Loops on Large Data**: Try to use vectorized operations (e.g., NumPy or pandas operations) instead of deeply nested Python loops, especially on large datasets.
- **Minimize Pandas `.iterrows()`**: Iterating over DataFrame rows is slow; prefer methods like `apply`, `itertuples()`, or vectorized column operations.
- **Use Efficient Data Structures**: Choose appropriate data structures (e.g., use sets for membership tests, use numpy arrays for numeric computations).
- **Profile Critical Sections**: Use the profiling utilities or other tools to measure where your code spends time and memory, and focus optimization efforts there.
- **Leverage Parallelism**: For independent tasks, use multiprocessing or vectorized hardware (GPUs) to speed up execution.

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request. Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
