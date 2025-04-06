#!/usr/bin/env python
"""
Command-line interface for the advanced_analysis package.

This module provides a command-line interface for analyzing code, data, and models
using the advanced_analysis package.
"""

import os
import sys
import click
import json
from pathlib import Path

from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import StaticAnalyzer
from advanced_analysis.data_quality import DataGuardian
from advanced_analysis.ml_advisor import MLAlgorithmRecognizer, InefficiencyDetector


@click.group()
def cli():
    """Advanced Analysis CLI for analyzing ML code, data, and models."""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'html', 'json']), default='text',
              help='Output format for the report')
@click.option('--output', type=click.Path(), help='Output file path for the report')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze_code(file_path, format, output, verbose):
    """Analyze Python code for complexity and inefficiencies."""
    analyzer = Analyzer()

    try:
        # Check if the path is a file or directory
        path = Path(file_path)
        if path.is_file() and path.suffix == '.py':
            # Analyze a single file
            click.echo(f"Analyzing file: {file_path}")
            results = analyzer.analyze_file(file_path)
        elif path.is_dir():
            # Analyze all Python files in the directory
            click.echo(f"Analyzing directory: {file_path}")
            results = {'files': {}}
            for root, _, files in os.walk(file_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if verbose:
                            click.echo(f"Analyzing file: {file_path}")
                        file_results = analyzer.analyze_file(file_path)
                        results['files'][file_path] = file_results
        else:
            click.echo(f"Error: {file_path} is not a Python file or directory")
            sys.exit(1)

        # Generate report
        if format == 'json':
            report = json.dumps(results, indent=2)
        else:
            report = analyzer.generate_report(results, format=format)

        # Output report
        if output:
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(report)

    except Exception as e:
        click.echo(f"Error: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'html', 'json']), default='text',
              help='Output format for the report')
@click.option('--output', type=click.Path(), help='Output file path for the report')
def analyze_data(data_path, format, output):
    """Analyze data quality in CSV or Excel files."""
    try:
        import pandas as pd

        # Load data
        path = Path(data_path)
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            click.echo(f"Error: Unsupported file format: {path.suffix}")
            click.echo("Supported formats: .csv, .xlsx, .xls")
            sys.exit(1)

        # Analyze data
        click.echo(f"Analyzing data: {data_path}")
        guardian = DataGuardian()
        report = guardian.generate_report(df)

        # Generate report
        if format == 'json':
            output_report = report.to_json()
        elif format == 'html':
            output_report = report.to_html()
        else:
            output_report = str(report)

        # Output report
        if output:
            with open(output, 'w') as f:
                f.write(output_report)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(output_report)

    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--framework', type=click.Choice(['pytorch', 'tensorflow']), required=True,
              help='ML framework used for the model')
@click.option('--batch-size', type=int, default=32, help='Batch size for profiling')
@click.option('--input-shape', type=str, help='Input shape as comma-separated values (e.g., "3,224,224")')
@click.option('--output', type=click.Path(), help='Output file path for the report')
def profile_model(model_path, framework, batch_size, input_shape, output):
    """Profile a machine learning model for performance metrics."""
    try:
        if framework == 'pytorch':
            import torch
            from advanced_analysis.performance import ModelPerformanceProfiler

            # Load model
            model = torch.load(model_path)

            # Create profiler
            profiler = ModelPerformanceProfiler(model=model, framework=framework)

            # Generate dummy input
            if input_shape:
                shape = [int(dim) for dim in input_shape.split(',')]
                dummy_input = torch.randn(batch_size, *shape)
            else:
                click.echo("Error: --input-shape is required for PyTorch models")
                sys.exit(1)

            # Run profiling
            click.echo(f"Profiling model: {model_path}")
            results = profiler.profile_model(dummy_input)

            # Generate report
            report = json.dumps(results, indent=2)

            # Output report
            if output:
                with open(output, 'w') as f:
                    f.write(report)
                click.echo(f"Report saved to {output}")
            else:
                click.echo(report)

        elif framework == 'tensorflow':
            click.echo("TensorFlow profiling is not yet implemented")
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()