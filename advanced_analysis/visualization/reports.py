"""
Report generation module for analysis results.

This module provides tools for generating text and HTML reports from analysis results,
including complexity analysis, performance profiling, and data quality assessment.
"""

import logging
import io
import base64
from typing import Dict, List, Any, Optional, Union
import datetime

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Some visualization features will be limited.")


class ReportGenerator:
    """
    Generates text and HTML reports from analysis results.
    """
    
    def __init__(self, title: str = "Analysis Report"):
        """
        Initialize the report generator.
        
        Args:
            title: Report title
        """
        self.title = title
        
    def generate_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a text report from analysis results.
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            Text report as a string
        """
        lines = []
        lines.append(f"=== {self.title} ===\n")
        lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add sections based on available data
        if "complexity_analysis" in analysis_results:
            lines.append("\n** Complexity Analysis **\n")
            complexity = analysis_results["complexity_analysis"]
            
            if "theoretical_complexity" in complexity:
                theoretical = complexity["theoretical_complexity"]
                lines.append(f"Theoretical Time Complexity: {theoretical.get('big_o', 'Unknown')}\n")
                lines.append(f"Theoretical Space Complexity: {theoretical.get('space_complexity', 'Unknown')}\n")
                
            if "empirical_performance" in complexity:
                empirical = complexity["empirical_performance"]
                lines.append("\nEmpirical Performance:\n")
                
                if "time_measurements" in empirical:
                    lines.append("Time Measurements:\n")
                    for size, time in empirical["time_measurements"]:
                        lines.append(f"  Input Size: {size}, Time: {time:.6f} seconds\n")
                        
                if "memory_measurements" in empirical:
                    lines.append("\nMemory Measurements:\n")
                    for size, memory in empirical["memory_measurements"]:
                        lines.append(f"  Input Size: {size}, Memory: {memory:.2f} MB\n")
                        
            if "optimization_suggestions" in complexity:
                lines.append("\nOptimization Suggestions:\n")
                for suggestion in complexity["optimization_suggestions"]:
                    lines.append(f"  - {suggestion.get('message', '')}: {suggestion.get('details', '')}\n")
                    
        if "performance_analysis" in analysis_results:
            lines.append("\n** Performance Analysis **\n")
            performance = analysis_results["performance_analysis"]
            
            if "execution_time" in performance:
                lines.append(f"Execution Time: {performance['execution_time']:.6f} seconds\n")
                
            if "throughput" in performance:
                lines.append(f"Throughput: {performance['throughput']:.2f} samples/second\n")
                
            if "memory_usage" in performance:
                lines.append(f"Memory Usage: {performance['memory_usage']:.2f} MB\n")
                
            if "bottlenecks" in performance:
                lines.append("\nBottlenecks:\n")
                for bottleneck in performance["bottlenecks"]:
                    lines.append(f"  - {bottleneck}\n")
                    
            if "optimization_suggestions" in performance:
                lines.append("\nOptimization Suggestions:\n")
                for suggestion in performance["optimization_suggestions"]:
                    lines.append(f"  - {suggestion}\n")
                    
        if "data_quality" in analysis_results:
            lines.append("\n** Data Quality Assessment **\n")
            quality = analysis_results["data_quality"]
            
            if "completeness" in quality:
                lines.append(f"Completeness: {quality['completeness']:.2f}%\n")
                
            if "uniqueness" in quality:
                lines.append(f"Uniqueness: {quality['uniqueness']:.2f}%\n")
                
            if "consistency" in quality:
                lines.append(f"Consistency: {quality['consistency']:.2f}%\n")
                
            if "accuracy" in quality:
                lines.append(f"Accuracy: {quality['accuracy']:.2f}%\n")
                
            if "recommendations" in quality:
                lines.append("\nRecommendations:\n")
                for recommendation in quality["recommendations"]:
                    lines.append(f"  - {recommendation}\n")
                    
        return "".join(lines)
        
    def generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate an HTML report from analysis results.
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            HTML report as a string
        """
        html_parts = []
        html_parts.append(f"<!DOCTYPE html>")
        html_parts.append(f"<html>")
        html_parts.append(f"<head>")
        html_parts.append(f"  <title>{self.title}</title>")
        html_parts.append(f"  <style>")
        html_parts.append(f"    body {{ font-family: Arial, sans-serif; margin: 20px; }}")
        html_parts.append(f"    h1 {{ color: #2c3e50; }}")
        html_parts.append(f"    h2 {{ color: #3498db; }}")
        html_parts.append(f"    h3 {{ color: #2980b9; }}")
        html_parts.append(f"    .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}")
        html_parts.append(f"    .suggestion {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 3px solid #3498db; }}")
        html_parts.append(f"    .warning {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 3px solid #ffc107; }}")
        html_parts.append(f"    .error {{ background-color: #f8d7da; padding: 10px; margin: 5px 0; border-left: 3px solid #dc3545; }}")
        html_parts.append(f"    table {{ border-collapse: collapse; width: 100%; }}")
        html_parts.append(f"    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}")
        html_parts.append(f"    th {{ background-color: #f2f2f2; }}")
        html_parts.append(f"  </style>")
        html_parts.append(f"</head>")
        html_parts.append(f"<body>")
        html_parts.append(f"  <h1>{self.title}</h1>")
        html_parts.append(f"  <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Add sections based on available data
        if "complexity_analysis" in analysis_results:
            html_parts.append(f"  <div class='section'>")
            html_parts.append(f"    <h2>Complexity Analysis</h2>")
            complexity = analysis_results["complexity_analysis"]
            
            if "theoretical_complexity" in complexity:
                theoretical = complexity["theoretical_complexity"]
                html_parts.append(f"    <h3>Theoretical Complexity</h3>")
                html_parts.append(f"    <p>Time Complexity: <strong>{theoretical.get('big_o', 'Unknown')}</strong></p>")
                html_parts.append(f"    <p>Space Complexity: <strong>{theoretical.get('space_complexity', 'Unknown')}</strong></p>")
                
            if "empirical_performance" in complexity:
                empirical = complexity["empirical_performance"]
                html_parts.append(f"    <h3>Empirical Performance</h3>")
                
                if "time_measurements" in empirical:
                    html_parts.append(f"    <h4>Time Measurements</h4>")
                    html_parts.append(f"    <table>")
                    html_parts.append(f"      <tr><th>Input Size</th><th>Time (seconds)</th></tr>")
                    for size, time in empirical["time_measurements"]:
                        html_parts.append(f"      <tr><td>{size}</td><td>{time:.6f}</td></tr>")
                    html_parts.append(f"    </table>")
                    
                    # Add plot if matplotlib is available
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            plt.figure(figsize=(8, 6))
                            sizes = [size for size, _ in empirical["time_measurements"]]
                            times = [time for _, time in empirical["time_measurements"]]
                            plt.plot(sizes, times, 'o-', label='Execution Time')
                            plt.xlabel('Input Size')
                            plt.ylabel('Time (seconds)')
                            plt.title('Execution Time vs. Input Size')
                            plt.grid(True)
                            
                            # Save plot to base64 string
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            img_str = base64.b64encode(buf.read()).decode('utf-8')
                            plt.close()
                            
                            html_parts.append(f"    <div>")
                            html_parts.append(f"      <img src='data:image/png;base64,{img_str}' alt='Execution Time Plot'>")
                            html_parts.append(f"    </div>")
                        except Exception as e:
                            logger.warning(f"Failed to generate plot: {e}")
                        
                if "memory_measurements" in empirical:
                    html_parts.append(f"    <h4>Memory Measurements</h4>")
                    html_parts.append(f"    <table>")
                    html_parts.append(f"      <tr><th>Input Size</th><th>Memory (MB)</th></tr>")
                    for size, memory in empirical["memory_measurements"]:
                        html_parts.append(f"      <tr><td>{size}</td><td>{memory:.2f}</td></tr>")
                    html_parts.append(f"    </table>")
                    
                    # Add plot if matplotlib is available
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            plt.figure(figsize=(8, 6))
                            sizes = [size for size, _ in empirical["memory_measurements"]]
                            memories = [memory for _, memory in empirical["memory_measurements"]]
                            plt.plot(sizes, memories, 'o-', label='Memory Usage')
                            plt.xlabel('Input Size')
                            plt.ylabel('Memory (MB)')
                            plt.title('Memory Usage vs. Input Size')
                            plt.grid(True)
                            
                            # Save plot to base64 string
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            img_str = base64.b64encode(buf.read()).decode('utf-8')
                            plt.close()
                            
                            html_parts.append(f"    <div>")
                            html_parts.append(f"      <img src='data:image/png;base64,{img_str}' alt='Memory Usage Plot'>")
                            html_parts.append(f"    </div>")
                        except Exception as e:
                            logger.warning(f"Failed to generate plot: {e}")
                    
            if "optimization_suggestions" in complexity:
                html_parts.append(f"    <h3>Optimization Suggestions</h3>")
                for suggestion in complexity["optimization_suggestions"]:
                    severity = suggestion.get('severity', 'medium')
                    css_class = 'suggestion'
                    if severity == 'high':
                        css_class = 'warning'
                    elif severity == 'critical':
                        css_class = 'error'
                        
                    html_parts.append(f"    <div class='{css_class}'>")
                    html_parts.append(f"      <h4>{suggestion.get('message', '')}</h4>")
                    html_parts.append(f"      <p>{suggestion.get('details', '')}</p>")
                    
                    if 'code_example' in suggestion:
                        html_parts.append(f"      <pre>{suggestion['code_example']}</pre>")
                        
                    html_parts.append(f"    </div>")
                    
            html_parts.append(f"  </div>")
            
        if "performance_analysis" in analysis_results:
            html_parts.append(f"  <div class='section'>")
            html_parts.append(f"    <h2>Performance Analysis</h2>")
            performance = analysis_results["performance_analysis"]
            
            html_parts.append(f"    <table>")
            if "execution_time" in performance:
                html_parts.append(f"      <tr><td>Execution Time</td><td>{performance['execution_time']:.6f} seconds</td></tr>")
                
            if "throughput" in performance:
                html_parts.append(f"      <tr><td>Throughput</td><td>{performance['throughput']:.2f} samples/second</td></tr>")
                
            if "memory_usage" in performance:
                html_parts.append(f"      <tr><td>Memory Usage</td><td>{performance['memory_usage']:.2f} MB</td></tr>")
                
            html_parts.append(f"    </table>")
            
            if "bottlenecks" in performance:
                html_parts.append(f"    <h3>Bottlenecks</h3>")
                html_parts.append(f"    <ul>")
                for bottleneck in performance["bottlenecks"]:
                    html_parts.append(f"      <li>{bottleneck}</li>")
                html_parts.append(f"    </ul>")
                
            if "optimization_suggestions" in performance:
                html_parts.append(f"    <h3>Optimization Suggestions</h3>")
                html_parts.append(f"    <ul>")
                for suggestion in performance["optimization_suggestions"]:
                    html_parts.append(f"      <li>{suggestion}</li>")
                html_parts.append(f"    </ul>")
                
            html_parts.append(f"  </div>")
            
        if "data_quality" in analysis_results:
            html_parts.append(f"  <div class='section'>")
            html_parts.append(f"    <h2>Data Quality Assessment</h2>")
            quality = analysis_results["data_quality"]
            
            html_parts.append(f"    <table>")
            if "completeness" in quality:
                html_parts.append(f"      <tr><td>Completeness</td><td>{quality['completeness']:.2f}%</td></tr>")
                
            if "uniqueness" in quality:
                html_parts.append(f"      <tr><td>Uniqueness</td><td>{quality['uniqueness']:.2f}%</td></tr>")
                
            if "consistency" in quality:
                html_parts.append(f"      <tr><td>Consistency</td><td>{quality['consistency']:.2f}%</td></tr>")
                
            if "accuracy" in quality:
                html_parts.append(f"      <tr><td>Accuracy</td><td>{quality['accuracy']:.2f}%</td></tr>")
                
            html_parts.append(f"    </table>")
            
            if "recommendations" in quality:
                html_parts.append(f"    <h3>Recommendations</h3>")
                html_parts.append(f"    <ul>")
                for recommendation in quality["recommendations"]:
                    html_parts.append(f"      <li>{recommendation}</li>")
                html_parts.append(f"    </ul>")
                
            html_parts.append(f"  </div>")
            
        html_parts.append(f"</body>")
        html_parts.append(f"</html>")
        
        return "\n".join(html_parts)
        
    def save_html_report(self, analysis_results: Dict[str, Any], filename: str) -> None:
        """
        Generate an HTML report and save it to a file.
        
        Args:
            analysis_results: Dictionary with analysis results
            filename: Output filename
        """
        html = self.generate_html_report(analysis_results)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"HTML report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save HTML report to {filename}: {e}")
            raise
