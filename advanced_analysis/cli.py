"""
Command-line interface for the advanced_analysis package.

This module provides a command-line interface for using the advanced_analysis
package to analyze ML code, models, and data.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Union

from advanced_analysis.version import __version__, get_version_info
from advanced_analysis.utils.error_handling import configure_file_logging, LoggingContext
from advanced_analysis.analyzer import Analyzer
from advanced_analysis.algorithm_complexity import StaticAnalyzer, DynamicAnalyzer, ComplexityAnalyzer
from advanced_analysis.ml_advisor import MLAlgorithmRecognizer, InefficiencyDetector
from advanced_analysis.performance import ModelPerformanceProfiler
from advanced_analysis.data_quality import DataGuardian
from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor, ProfileInfo

# Configure logger
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
        log_file: Path to log file
    """
    # Set log level based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure file logging if specified
    if log_file:
        configure_file_logging(log_file, log_level)
        
    logger.info(f"Logging initialized (level: {logging.getLevelName(log_level)})")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

def analyze_code_command(args):
    """
    Handle the 'analyze-code' command.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Analyzing code: {args.file}")
    
    # Create analyzer
    analyzer = Analyzer()
    
    # Read code from file or stdin
    if args.file == "-":
        logger.info("Reading code from stdin")
        code = sys.stdin.read()
        file_path = "stdin"
    else:
        file_path = args.file
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
            
        logger.info(f"Reading code from file: {file_path}")
        with open(file_path, "r") as f:
            code = f.read()
    
    # Analyze code
    start_time = time.time()
    results = analyzer.analyze_code(code, context=file_path)
    end_time = time.time()
    
    # Add execution time
    results["execution_time"] = end_time - start_time
    
    # Output results
    output_results(results, args.output, args.format)

def analyze_model_command(args):
    """
    Handle the 'analyze-model' command.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Analyzing model: {args.model}")
    
    # Validate model file
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model, args.framework)
    
    # Create analyzer
    analyzer = Analyzer()
    
    # Analyze model
    start_time = time.time()
    results = analyzer.analyze_model(model, framework=args.framework)
    end_time = time.time()
    
    # Add execution time
    results["execution_time"] = end_time - start_time
    
    # Output results
    output_results(results, args.output, args.format)

def analyze_data_command(args):
    """
    Handle the 'analyze-data' command.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Analyzing data: {args.data}")
    
    # Validate data file
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Load data
    data = load_data(args.data, args.format)
    
    # Create data guardian
    guardian = DataGuardian()
    
    # Analyze data
    start_time = time.time()
    results = guardian.generate_report(data)
    end_time = time.time()
    
    # Convert results to dictionary
    results_dict = {
        "data_quality": results.__dict__,
        "execution_time": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__
    }
    
    # Output results
    output_results(results_dict, args.output, args.output_format)

def compress_model_command(args):
    """
    Handle the 'compress-model' command.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Compressing model: {args.model}")
    
    # Validate model file
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model, args.framework)
    
    # Parse techniques
    techniques = args.techniques.split(",") if args.techniques else ["quantization"]
    
    # Create profile
    profile = ProfileInfo(
        framework=args.framework,
        model_type=args.model_type,
        hardware=args.hardware,
        techniques=techniques,
        params={
            "quantization_method": args.quantization_method,
            "prune_amount": args.prune_amount
        }
    )
    
    # Create compressor
    compressor = ModelCompressor(profile)
    
    # Compress model
    start_time = time.time()
    compressed_model = compressor.compress(model)
    end_time = time.time()
    
    # Get logs
    logs = compressor.get_logs()
    
    # Save compressed model
    if args.output:
        save_model(compressed_model, args.output, args.framework)
        logger.info(f"Compressed model saved to: {args.output}")
    
    # Output results
    results = {
        "logs": logs,
        "execution_time": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__,
        "techniques": techniques,
        "framework": args.framework,
        "model_type": args.model_type,
        "hardware": args.hardware
    }
    
    output_results(results, args.report, args.format)

def version_command(args):
    """
    Handle the 'version' command.
    
    Args:
        args: Command-line arguments
    """
    if args.verbose:
        version_info = get_version_info()
        print(json.dumps(version_info, indent=2))
    else:
        print(f"advanced_analysis version {__version__}")

def load_model(model_path: str, framework: str) -> Any:
    """
    Load a model from a file.
    
    Args:
        model_path: Path to model file
        framework: Model framework (pytorch, tensorflow, sklearn)
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {framework} model from {model_path}")
    
    if framework.lower() == "pytorch":
        try:
            import torch
            model = torch.load(model_path)
            return model
        except ImportError:
            logger.error("PyTorch not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            sys.exit(1)
    elif framework.lower() in ["tensorflow", "tf"]:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model
        except ImportError:
            logger.error("TensorFlow not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            sys.exit(1)
    elif framework.lower() == "sklearn":
        try:
            import joblib
            model = joblib.load(model_path)
            return model
        except ImportError:
            logger.error("joblib not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unsupported framework: {framework}")
        sys.exit(1)

def save_model(model: Any, output_path: str, framework: str):
    """
    Save a model to a file.
    
    Args:
        model: Model to save
        output_path: Path to save model
        framework: Model framework (pytorch, tensorflow, sklearn)
    """
    logger.info(f"Saving {framework} model to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if framework.lower() == "pytorch":
        try:
            import torch
            torch.save(model, output_path)
        except ImportError:
            logger.error("PyTorch not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error saving PyTorch model: {e}")
            sys.exit(1)
    elif framework.lower() in ["tensorflow", "tf"]:
        try:
            if hasattr(model, "save"):
                model.save(output_path)
            else:
                import tensorflow as tf
                tf.saved_model.save(model, output_path)
        except ImportError:
            logger.error("TensorFlow not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error saving TensorFlow model: {e}")
            sys.exit(1)
    elif framework.lower() == "sklearn":
        try:
            import joblib
            joblib.dump(model, output_path)
        except ImportError:
            logger.error("joblib not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error saving scikit-learn model: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unsupported framework: {framework}")
        sys.exit(1)

def load_data(data_path: str, format: str) -> Any:
    """
    Load data from a file.
    
    Args:
        data_path: Path to data file
        format: Data format (csv, json, parquet, pickle)
        
    Returns:
        Loaded data
    """
    logger.info(f"Loading {format} data from {data_path}")
    
    if format.lower() == "csv":
        try:
            import pandas as pd
            data = pd.read_csv(data_path)
            return data
        except ImportError:
            logger.error("pandas not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            sys.exit(1)
    elif format.lower() == "json":
        try:
            import pandas as pd
            data = pd.read_json(data_path)
            return data
        except ImportError:
            logger.error("pandas not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            sys.exit(1)
    elif format.lower() == "parquet":
        try:
            import pandas as pd
            data = pd.read_parquet(data_path)
            return data
        except ImportError:
            logger.error("pandas and pyarrow not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading Parquet data: {e}")
            sys.exit(1)
    elif format.lower() == "pickle":
        try:
            import pandas as pd
            data = pd.read_pickle(data_path)
            return data
        except ImportError:
            logger.error("pandas not installed")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading Pickle data: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unsupported format: {format}")
        sys.exit(1)

def output_results(results: Dict[str, Any], output_path: Optional[str] = None, format: str = "json"):
    """
    Output results to file or stdout.
    
    Args:
        results: Results to output
        output_path: Path to output file (None for stdout)
        format: Output format (json, yaml, html, markdown)
    """
    if format.lower() == "json":
        output = json.dumps(results, indent=2, default=str)
    elif format.lower() == "yaml":
        try:
            import yaml
            output = yaml.dump(results, default_flow_style=False)
        except ImportError:
            logger.error("PyYAML not installed, falling back to JSON")
            output = json.dumps(results, indent=2, default=str)
    elif format.lower() == "html":
        try:
            from advanced_analysis.visualization import ReportGenerator
            report_generator = ReportGenerator()
            output = report_generator.generate_html_report(results)
        except ImportError:
            logger.error("Visualization dependencies not installed, falling back to JSON")
            output = json.dumps(results, indent=2, default=str)
    elif format.lower() == "markdown":
        try:
            from advanced_analysis.visualization import ReportGenerator
            report_generator = ReportGenerator()
            output = report_generator.generate_markdown_report(results)
        except ImportError:
            logger.error("Visualization dependencies not installed, falling back to JSON")
            output = json.dumps(results, indent=2, default=str)
    else:
        logger.error(f"Unsupported format: {format}, falling back to JSON")
        output = json.dumps(results, indent=2, default=str)
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(output)
        logger.info(f"Results written to {output_path}")
    else:
        # Write to stdout
        print(output)

def main():
    """Main entry point for the CLI."""
    # Create parser
    parser = argparse.ArgumentParser(
        description="Advanced Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed version information"
    )
    version_parser.set_defaults(func=version_command)
    
    # Analyze code command
    analyze_code_parser = subparsers.add_parser(
        "analyze-code",
        help="Analyze ML code"
    )
    analyze_code_parser.add_argument(
        "file",
        help="Path to Python file to analyze (use '-' for stdin)"
    )
    analyze_code_parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: stdout)"
    )
    analyze_code_parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml", "html", "markdown"],
        default="json",
        help="Output format"
    )
    analyze_code_parser.set_defaults(func=analyze_code_command)
    
    # Analyze model command
    analyze_model_parser = subparsers.add_parser(
        "analyze-model",
        help="Analyze ML model"
    )
    analyze_model_parser.add_argument(
        "model",
        help="Path to model file"
    )
    analyze_model_parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow", "sklearn"],
        default="pytorch",
        help="Model framework"
    )
    analyze_model_parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: stdout)"
    )
    analyze_model_parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml", "html", "markdown"],
        default="json",
        help="Output format"
    )
    analyze_model_parser.set_defaults(func=analyze_model_command)
    
    # Analyze data command
    analyze_data_parser = subparsers.add_parser(
        "analyze-data",
        help="Analyze data quality"
    )
    analyze_data_parser.add_argument(
        "data",
        help="Path to data file"
    )
    analyze_data_parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet", "pickle"],
        default="csv",
        help="Data format"
    )
    analyze_data_parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: stdout)"
    )
    analyze_data_parser.add_argument(
        "--output-format",
        choices=["json", "yaml", "html", "markdown"],
        default="json",
        help="Output format"
    )
    analyze_data_parser.set_defaults(func=analyze_data_command)
    
    # Compress model command
    compress_model_parser = subparsers.add_parser(
        "compress-model",
        help="Compress ML model"
    )
    compress_model_parser.add_argument(
        "model",
        help="Path to model file"
    )
    compress_model_parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow", "sklearn"],
        default="pytorch",
        help="Model framework"
    )
    compress_model_parser.add_argument(
        "--model-type",
        choices=["cnn", "rnn", "transformer", "mlp", "other"],
        default="cnn",
        help="Model type"
    )
    compress_model_parser.add_argument(
        "--hardware",
        choices=["cpu", "gpu", "mobile", "edge"],
        default="cpu",
        help="Target hardware"
    )
    compress_model_parser.add_argument(
        "--techniques",
        help="Compression techniques to apply (comma-separated)"
    )
    compress_model_parser.add_argument(
        "--quantization-method",
        choices=["dynamic", "static", "qat", "auto"],
        default="auto",
        help="Quantization method"
    )
    compress_model_parser.add_argument(
        "--prune-amount",
        type=float,
        default=0.3,
        help="Pruning amount (0.0-1.0)"
    )
    compress_model_parser.add_argument(
        "-o", "--output",
        help="Path to output model file"
    )
    compress_model_parser.add_argument(
        "--report",
        help="Path to report file (default: stdout)"
    )
    compress_model_parser.add_argument(
        "--format",
        choices=["json", "yaml", "html", "markdown"],
        default="json",
        help="Report format"
    )
    compress_model_parser.set_defaults(func=compress_model_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose, args.log_file)
    
    # Execute command
    if hasattr(args, "func"):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
