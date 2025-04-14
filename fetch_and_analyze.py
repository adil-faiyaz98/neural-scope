#!/usr/bin/env python
"""
Fetch and Analyze Pre-trained Models

This script fetches pre-trained models from popular repositories (PyTorch Hub, TensorFlow Hub, 
or Hugging Face), analyzes them using Neural-Scope, and generates comprehensive reports.

Usage:
    python fetch_and_analyze.py --model resnet18 --source pytorch_hub --output-dir reports
"""

import os
import sys
import argparse
import json
import yaml
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neural-scope-fetch")

# Dictionary of available pre-trained models
AVAILABLE_MODELS = {
    "pytorch_hub": [
        "resnet18", "resnet50", "mobilenet_v2", "densenet121", 
        "efficientnet_b0", "efficientnet_b1", "vgg16", "inception_v3"
    ],
    "tensorflow_hub": [
        "efficientnet/b0", "efficientnet/b3", "mobilenet_v2_100_224",
        "inception_v3", "resnet_50"
    ],
    "huggingface": [
        "bert-base-uncased", "distilbert-base-uncased", "gpt2", "gpt2-medium",
        "t5-small", "t5-base", "roberta-base"
    ]
}

# Model metadata with expected performance metrics
MODEL_METADATA = {
    "resnet18": {
        "parameters": 11689512,
        "top1_accuracy": 69.758,
        "top5_accuracy": 89.078,
        "inference_time_range": [5, 20],  # ms, depends on hardware
        "size_mb": 44.7,
        "paper_url": "https://arxiv.org/abs/1512.03385",
        "description": "ResNet-18 is an 18-layer residual network trained on ImageNet"
    },
    "mobilenet_v2": {
        "parameters": 3504872,
        "top1_accuracy": 71.878,
        "top5_accuracy": 90.286,
        "inference_time_range": [3, 15],  # ms, depends on hardware
        "size_mb": 13.6,
        "paper_url": "https://arxiv.org/abs/1801.04381",
        "description": "MobileNetV2 is a lightweight CNN architecture designed for mobile devices"
    },
    "efficientnet_b0": {
        "parameters": 5288548,
        "top1_accuracy": 77.1,
        "top5_accuracy": 93.3,
        "inference_time_range": [4, 18],  # ms, depends on hardware
        "size_mb": 20.2,
        "paper_url": "https://arxiv.org/abs/1905.11946",
        "description": "EfficientNet-B0 is a convolutional network that scales depth/width/resolution"
    }
}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch and analyze pre-trained models")
    parser.add_argument("--model", required=True, help="Model name to fetch and analyze")
    parser.add_argument("--source", choices=["pytorch_hub", "tensorflow_hub", "huggingface"], 
                      default="pytorch_hub", help="Source of the pre-trained model")
    parser.add_argument("--output-dir", default="model_reports", help="Directory to save reports")
    parser.add_argument("--mlflow", action="store_true", help="Track results with MLflow")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--security-check", action="store_true", help="Perform security checks")
    parser.add_argument("--verify-metrics", action="store_true", help="Verify model metrics against known values")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file."""
    if not config_path:
        return {}
        
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path}")
                return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

def fetch_model(args, config):
    """Fetch a pre-trained model from the specified source."""
    logger.info(f"Fetching {args.model} from {args.source}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the framework based on the source
    framework = "pytorch" if args.source == "pytorch_hub" else "tensorflow"
    
    # Path to save the model
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.model}.{'pt' if framework == 'pytorch' else 'h5'}")
    
    # Check if model is in the list of available models
    if args.model not in AVAILABLE_MODELS.get(args.source, []):
        logger.warning(f"Model {args.model} not in the list of known models for {args.source}.")
        logger.info(f"Available models for {args.source}: {', '.join(AVAILABLE_MODELS.get(args.source, []))}")
    
    # Fetch the model based on the source
    try:
        if args.source == "pytorch_hub":
            import torch
            
            # Fetch model from PyTorch Hub
            model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True)
            model.eval()
            
            # Save the model
            torch.save(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
        elif args.source == "tensorflow_hub":
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Fetch model from TensorFlow Hub
            model_url = f"https://tfhub.dev/google/{args.model}/1"
            model = hub.KerasLayer(model_url)
            
            # Create a simple model with the hub layer
            inputs = tf.keras.Input(shape=(224, 224, 3))
            outputs = model(inputs)
            keras_model = tf.keras.Model(inputs, outputs)
            
            # Save the model
            keras_model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
        elif args.source == "huggingface":
            from transformers import AutoModel, AutoTokenizer
            
            # Fetch model from Hugging Face
            model = AutoModel.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            # Save the model and tokenizer
            model_dir = os.path.join(model_dir, args.model)
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # For compatibility with the analyzer, save a reference to the model directory
            with open(model_path, 'w') as f:
                f.write(model_dir)
                
            logger.info(f"Model saved to {model_dir}")
            
        return model_path
        
    except Exception as e:
        logger.error(f"Error fetching model: {e}")
        sys.exit(1)

def analyze_model(args, config, model_path):
    """Analyze the model using Neural-Scope."""
    logger.info(f"Analyzing model {args.model}...")
    
    # Determine the framework based on the source
    framework = "pytorch" if args.source == "pytorch_hub" else "tensorflow"
    
    # Create the command to run the Neural-Scope CI/CD runner
    cmd = [
        "python", "neural_scope_cicd.py", "optimize",
        "--model-path", model_path,
        "--output-dir", os.path.join(args.output_dir, "analysis"),
        "--framework", framework
    ]
    
    # Add techniques from config if available
    if "techniques" in config:
        techniques = ",".join(config["techniques"])
        cmd.extend(["--techniques", techniques])
    
    # Add batch size from config if available
    if "batch_size" in config:
        cmd.extend(["--batch-size", str(config["batch_size"])])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Model analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error analyzing model: {e}")
        sys.exit(1)
    
    # Path to the analysis results
    analysis_path = os.path.join(args.output_dir, "analysis", "model_analysis.json")
    optimization_path = os.path.join(args.output_dir, "analysis", "optimization_results.json")
    summary_path = os.path.join(args.output_dir, "analysis", "optimization_summary.json")
    
    # Load the analysis results
    try:
        with open(analysis_path, 'r') as f:
            analysis_results = json.load(f)
        
        with open(optimization_path, 'r') as f:
            optimization_results = json.load(f)
            
        with open(summary_path, 'r') as f:
            summary_results = json.load(f)
            
        return {
            "analysis": analysis_results,
            "optimization": optimization_results,
            "summary": summary_results
        }
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        return {}

def perform_security_check(args, model_path, results):
    """Perform security checks on the model."""
    if not args.security_check:
        return {}
        
    logger.info("Performing security checks...")
    
    security_results = {
        "vulnerabilities": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check for known vulnerabilities in the model architecture
    if "architecture" in results.get("analysis", {}):
        architecture = results["analysis"]["architecture"]
        
        # Check for outdated architectures
        outdated_architectures = ["vgg", "alexnet"]
        for arch in outdated_architectures:
            if arch in architecture.lower():
                security_results["warnings"].append(
                    f"Using potentially outdated architecture: {architecture}. "
                    f"Consider using a more modern architecture with better security properties."
                )
    
    # Check for quantization issues
    if "quantization" in results.get("optimization", {}).get("techniques", []):
        security_results["warnings"].append(
            "Quantized models may be more susceptible to adversarial attacks. "
            "Consider implementing adversarial training or defensive distillation."
        )
    
    # Check model size for potential over-parameterization
    if "parameters" in results.get("analysis", {}):
        parameters = results["analysis"]["parameters"]
        if parameters > 100000000:  # 100M parameters
            security_results["warnings"].append(
                f"Model has a large number of parameters ({parameters:,}), "
                f"which may increase attack surface. Consider model pruning or distillation."
            )
    
    # Add general security recommendations
    security_results["recommendations"] = [
        "Implement input validation to prevent adversarial examples",
        "Consider using model encryption for sensitive deployments",
        "Regularly update the model with new training data to prevent concept drift",
        "Implement monitoring for detecting unusual model behavior in production"
    ]
    
    # Save security results
    security_path = os.path.join(args.output_dir, "security_check.json")
    with open(security_path, 'w') as f:
        json.dump(security_results, f, indent=2)
        
    logger.info(f"Security check results saved to {security_path}")
    
    return security_results

def verify_metrics(args, results):
    """Verify model metrics against known values."""
    if not args.verify_metrics or args.model not in MODEL_METADATA:
        return {}
        
    logger.info(f"Verifying metrics for {args.model}...")
    
    verification_results = {
        "model": args.model,
        "expected": MODEL_METADATA[args.model],
        "actual": {},
        "discrepancies": []
    }
    
    # Extract actual metrics from results
    if "parameters" in results.get("analysis", {}):
        verification_results["actual"]["parameters"] = results["analysis"]["parameters"]
        
        # Check parameter count
        expected_params = MODEL_METADATA[args.model]["parameters"]
        actual_params = results["analysis"]["parameters"]
        if abs(actual_params - expected_params) / expected_params > 0.05:  # 5% tolerance
            verification_results["discrepancies"].append(
                f"Parameter count discrepancy: expected {expected_params:,}, got {actual_params:,}"
            )
    
    # Check model size
    if "original_size" in results.get("optimization", {}):
        verification_results["actual"]["size_mb"] = results["optimization"]["original_size"]
        
        expected_size = MODEL_METADATA[args.model]["size_mb"]
        actual_size = results["optimization"]["original_size"]
        if abs(actual_size - expected_size) / expected_size > 0.1:  # 10% tolerance
            verification_results["discrepancies"].append(
                f"Model size discrepancy: expected {expected_size:.1f} MB, got {actual_size:.1f} MB"
            )
    
    # Check inference time
    if "inference_time_ms" in results.get("summary", {}):
        verification_results["actual"]["inference_time_ms"] = results["summary"]["inference_time_ms"]
        
        min_time, max_time = MODEL_METADATA[args.model]["inference_time_range"]
        actual_time = results["summary"]["inference_time_ms"]
        if actual_time < min_time or actual_time > max_time:
            verification_results["discrepancies"].append(
                f"Inference time outside expected range: expected {min_time}-{max_time} ms, got {actual_time:.1f} ms"
            )
    
    # Save verification results
    verification_path = os.path.join(args.output_dir, "metrics_verification.json")
    with open(verification_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
        
    logger.info(f"Metrics verification results saved to {verification_path}")
    
    return verification_results

def track_with_mlflow(args, results):
    """Track results with MLflow."""
    if not args.mlflow:
        return
        
    logger.info("Tracking results with MLflow...")
    
    # Create the command to run the Neural-Scope CI/CD runner
    cmd = [
        "python", "neural_scope_cicd.py", "track",
        "--model-name", args.model,
        "--results-path", os.path.join(args.output_dir, "analysis", "optimization_results.json"),
        "--tracking-uri", args.mlflow_tracking_uri,
        "--experiment-name", f"neural-scope-{args.source}"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Results tracked in MLflow")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error tracking results: {e}")

def generate_report(args, results, security_results, verification_results):
    """Generate a comprehensive report."""
    logger.info("Generating comprehensive report...")
    
    # Create report data
    report = {
        "model": {
            "name": args.model,
            "source": args.source,
            "metadata": MODEL_METADATA.get(args.model, {})
        },
        "analysis": results.get("analysis", {}),
        "optimization": results.get("optimization", {}),
        "summary": results.get("summary", {}),
        "security": security_results,
        "verification": verification_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report as JSON
    report_path = os.path.join(args.output_dir, "comprehensive_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    html_report = generate_html_report(report)
    html_path = os.path.join(args.output_dir, "comprehensive_report.html")
    with open(html_path, 'w') as f:
        f.write(html_report)
        
    logger.info(f"Comprehensive report saved to {report_path} and {html_path}")
    
    return report_path, html_path

def generate_html_report(report):
    """Generate an HTML report from the report data."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Neural-Scope Analysis Report: {report['model']['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .metric {{ margin-bottom: 10px; }}
        .metric-name {{ font-weight: bold; }}
        .warning {{ color: #e74c3c; }}
        .recommendation {{ color: #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
        .footer {{ margin-top: 30px; text-align: center; font-size: 0.9em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Neural-Scope Analysis Report</h1>
            <p>Model: {report['model']['name']} | Source: {report['model']['source']} | Generated: {report['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Model Overview</h2>
            <div class="metric">
                <span class="metric-name">Model Name:</span> {report['model']['name']}
            </div>
            <div class="metric">
                <span class="metric-name">Source:</span> {report['model']['source']}
            </div>
"""
    
    # Add metadata if available
    if report['model']['metadata']:
        html += f"""
            <div class="metric">
                <span class="metric-name">Parameters:</span> {report['model']['metadata'].get('parameters', 'N/A'):,}
            </div>
            <div class="metric">
                <span class="metric-name">Top-1 Accuracy:</span> {report['model']['metadata'].get('top1_accuracy', 'N/A')}%
            </div>
            <div class="metric">
                <span class="metric-name">Top-5 Accuracy:</span> {report['model']['metadata'].get('top5_accuracy', 'N/A')}%
            </div>
            <div class="metric">
                <span class="metric-name">Size:</span> {report['model']['metadata'].get('size_mb', 'N/A')} MB
            </div>
            <div class="metric">
                <span class="metric-name">Description:</span> {report['model']['metadata'].get('description', 'N/A')}
            </div>
            <div class="metric">
                <span class="metric-name">Paper:</span> <a href="{report['model']['metadata'].get('paper_url', '#')}" target="_blank">{report['model']['metadata'].get('paper_url', 'N/A')}</a>
            </div>
"""
    
    # Add analysis results
    html += f"""
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
"""
    
    if report['analysis']:
        # Add key metrics from analysis
        if 'parameters' in report['analysis']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Parameters:</span> {report['analysis']['parameters']:,}
            </div>
"""
        
        if 'layers' in report['analysis']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Layers:</span> {report['analysis']['layers']}
            </div>
"""
        
        if 'architecture' in report['analysis']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Architecture:</span> {report['analysis']['architecture']}
            </div>
"""
        
        if 'memory_usage' in report['analysis']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Memory Usage:</span> {report['analysis']['memory_usage']} MB
            </div>
"""
    else:
        html += "<p>No analysis results available.</p>"
    
    # Add optimization results
    html += f"""
        </div>
        
        <div class="section">
            <h2>Optimization Results</h2>
"""
    
    if report['optimization']:
        # Add key metrics from optimization
        if 'original_size' in report['optimization'] and 'optimized_size' in report['optimization']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Original Size:</span> {report['optimization']['original_size']} MB
            </div>
            <div class="metric">
                <span class="metric-name">Optimized Size:</span> {report['optimization']['optimized_size']} MB
            </div>
            <div class="metric">
                <span class="metric-name">Size Reduction:</span> {report['optimization'].get('size_reduction_percentage', 'N/A')}%
            </div>
"""
        
        if 'techniques' in report['optimization']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Techniques Applied:</span> {', '.join(report['optimization']['techniques'])}
            </div>
"""
        
        if 'inference_speedup' in report['optimization']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Inference Speedup:</span> {report['optimization']['inference_speedup']}x
            </div>
"""
    else:
        html += "<p>No optimization results available.</p>"
    
    # Add summary results
    html += f"""
        </div>
        
        <div class="section">
            <h2>Performance Summary</h2>
"""
    
    if report['summary']:
        # Add key metrics from summary
        if 'inference_time_ms' in report['summary']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Inference Time:</span> {report['summary']['inference_time_ms']} ms
            </div>
"""
        
        if 'throughput_samples_per_second' in report['summary']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Throughput:</span> {report['summary']['throughput_samples_per_second']} samples/second
            </div>
"""
        
        if 'memory_usage_mb' in report['summary']:
            html += f"""
            <div class="metric">
                <span class="metric-name">Memory Usage:</span> {report['summary']['memory_usage_mb']} MB
            </div>
"""
    else:
        html += "<p>No performance summary available.</p>"
    
    # Add security results
    html += f"""
        </div>
        
        <div class="section">
            <h2>Security Check</h2>
"""
    
    if report['security']:
        # Add vulnerabilities
        if report['security']['vulnerabilities']:
            html += "<h3>Vulnerabilities</h3><ul>"
            for vuln in report['security']['vulnerabilities']:
                html += f"<li class='warning'>{vuln}</li>"
            html += "</ul>"
        
        # Add warnings
        if report['security']['warnings']:
            html += "<h3>Warnings</h3><ul>"
            for warning in report['security']['warnings']:
                html += f"<li class='warning'>{warning}</li>"
            html += "</ul>"
        
        # Add recommendations
        if report['security']['recommendations']:
            html += "<h3>Recommendations</h3><ul>"
            for rec in report['security']['recommendations']:
                html += f"<li class='recommendation'>{rec}</li>"
            html += "</ul>"
    else:
        html += "<p>No security check results available.</p>"
    
    # Add verification results
    html += f"""
        </div>
        
        <div class="section">
            <h2>Metrics Verification</h2>
"""
    
    if report['verification']:
        # Add expected vs actual metrics
        html += "<h3>Expected vs Actual Metrics</h3>"
        html += "<table><tr><th>Metric</th><th>Expected</th><th>Actual</th></tr>"
        
        if 'parameters' in report['verification']['expected'] and 'parameters' in report['verification']['actual']:
            html += f"""<tr>
                <td>Parameters</td>
                <td>{report['verification']['expected']['parameters']:,}</td>
                <td>{report['verification']['actual']['parameters']:,}</td>
            </tr>"""
        
        if 'size_mb' in report['verification']['expected'] and 'size_mb' in report['verification']['actual']:
            html += f"""<tr>
                <td>Size (MB)</td>
                <td>{report['verification']['expected']['size_mb']}</td>
                <td>{report['verification']['actual']['size_mb']}</td>
            </tr>"""
        
        if 'inference_time_range' in report['verification']['expected'] and 'inference_time_ms' in report['verification']['actual']:
            min_time, max_time = report['verification']['expected']['inference_time_range']
            html += f"""<tr>
                <td>Inference Time (ms)</td>
                <td>{min_time}-{max_time}</td>
                <td>{report['verification']['actual']['inference_time_ms']}</td>
            </tr>"""
        
        html += "</table>"
        
        # Add discrepancies
        if report['verification']['discrepancies']:
            html += "<h3>Discrepancies</h3><ul>"
            for disc in report['verification']['discrepancies']:
                html += f"<li class='warning'>{disc}</li>"
            html += "</ul>"
    else:
        html += "<p>No metrics verification results available.</p>"
    
    # Close the HTML
    html += f"""
        </div>
        
        <div class="footer">
            <p>Generated by Neural-Scope CI/CD Integration | {report['timestamp']}</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def main():
    """Run the fetch and analyze script."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Fetch the model
    model_path = fetch_model(args, config)
    
    # Analyze the model
    results = analyze_model(args, config, model_path)
    
    # Perform security check
    security_results = perform_security_check(args, model_path, results)
    
    # Verify metrics
    verification_results = verify_metrics(args, results)
    
    # Track results with MLflow
    track_with_mlflow(args, results)
    
    # Generate report
    report_path, html_path = generate_report(args, results, security_results, verification_results)
    
    logger.info(f"\nAnalysis completed successfully!")
    logger.info(f"JSON Report: {report_path}")
    logger.info(f"HTML Report: {html_path}")
    
    # Open the HTML report in the default browser
    if os.path.exists(html_path):
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()
