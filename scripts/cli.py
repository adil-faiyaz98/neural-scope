"""
CLI tool interface for the AI/ML Complexity Analysis package.
Allows users to analyze files or directories and get complexity and cost insights via command line.
"""
import argparse
import sys
import logging
from aiml_complexity import analysis, aws_costs

def main(args=None):
    parser = argparse.ArgumentParser(prog="aiml-complexity", description="Analyze complexity of AI/ML code and get cost insights.")
    parser.add_argument("path", help="Path to a Python file or a directory to analyze")
    parser.add_argument("--instance", "-i", default="m5.large", help="AWS instance type for cost estimation (default: m5.large)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging output")
    parsed_args = parser.parse_args(args=args)
    # Configure logging
    if parsed_args.verbose:
        logging.getLogger("aiml_complexity").setLevel(logging.DEBUG)
    else:
        # Only errors by default on CLI (to not clutter output)
        logging.getLogger("aiml_complexity").setLevel(logging.ERROR)
    path = parsed_args.path
    instance = parsed_args.instance
    # Determine if path is file or directory
    import os
    if not os.path.exists(path):
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        results = []
        if os.path.isfile(path):
            # Single file
            res = analysis.analyze_file(path)
            results = [res]
        elif os.path.isdir(path):
            # Directory
            results = analysis.analyze_directory(path, recursive=True, use_multiprocessing=False)
        else:
            print(f"Path is not a file or directory: {path}", file=sys.stderr)
            sys.exit(1)
        # Output results
        for res in results:
            file = res.file if res.file else "(input)"
            score = res.complexity_score
            ops = res.estimated_operations
            issues = res.inefficiencies
            issue_count = len(issues)
            print(f"File: {file}")
            print(f"  Complexity Score: {score}")
            print(f"  Estimated Operations: {ops}")
            if issue_count:
                print(f"  Inefficiencies ({issue_count}):")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"  Inefficiencies: None")
            # Estimate cost for this file's operations
            try:
                cost = aws_costs.estimate_cost(ops, instance_type=instance)
                print(f"  Estimated cost on {instance}: ${cost:.6f}")
            except Exception as e:
                print(f"  Cost estimation not available: {e}")
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)
