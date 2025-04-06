"""
Jupyter extension for ML inefficiency detection and suggestions.

This module provides a Jupyter magic that analyzes code cells for inefficiencies
and provides inline suggestions for improvement.
"""

import sys
import ast
import inspect
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.display import display, HTML
from typing import Any, Dict, List, Optional, Union
import re

# Import the inefficiency detector
from .inefficiency_detection import InefficiencyDetector, Config, Severity, __version__

@magics_class
class MLAdvisorMagics(Magics):
    """
    Magic commands for ML code inefficiency detection and optimization.
    
    Usage:
        %ml_advisor [options]          - Analyze the current cell
        %%ml_advisor [options]         - Analyze the cell content
        
    Options:
        -q, --quiet                    - Suppress output
        -s, --severity=low|medium|high - Minimum severity level (default: low)
        -f, --fix                      - Apply suggested auto-fixes
        -c, --config=path              - Path to config file
    """
    
    def __init__(self, shell):
        super().__init__(shell)
        self.config = Config()
    
    @line_magic
    def ml_advisor(self, line):
        """Analyze the last executed cell for inefficiencies."""
        # Get options from the line
        opts = self._parse_options(line)
        
        # Get the last executed cell
        cell = self.shell.user_ns['_oh'].get(self.shell.execution_count-1, "")
        if isinstance(cell, str):
            code = cell
        else:
            # If it's not a string, try to get the source code
            try:
                code = inspect.getsource(cell)
            except (TypeError, OSError):
                print("Error: Could not get source code from last result.")
                return
                
        # Run the analysis
        self._analyze_and_display(code, opts)
    
    @cell_magic
    def ml_advisor(self, line, cell):
        """Analyze the cell content for inefficiencies."""
        # Get options from the line
        opts = self._parse_options(line)
        
        # Run the analysis
        self._analyze_and_display(cell, opts)
        
        # Also execute the cell
        self.shell.ex(cell)
    
    def _parse_options(self, line):
        """Parse the magic options."""
        opts = {
            'quiet': False,
            'severity': 'low',
            'fix': False,
            'config_path': None
        }
        
        if line:
            parts = line.split()
            for part in parts:
                if part in ['-q', '--quiet']:
                    opts['quiet'] = True
                elif part.startswith('-s=') or part.startswith('--severity='):
                    sev = part.split('=', 1)[1].lower()
                    if sev in ['low', 'medium', 'high', 'critical']:
                        opts['severity'] = sev
                elif part in ['-f', '--fix']:
                    opts['fix'] = True
                elif part.startswith('-c=') or part.startswith('--config='):
                    opts['config_path'] = part.split('=', 1)[1]
        
        return opts
    
    def _analyze_and_display(self, code, opts):
        """Analyze the code and display results."""
        if opts['quiet']:
            return
            
        # Configure the detector
        if opts['config_path']:
            detector = InefficiencyDetector.from_config_file(code, opts['config_path'])
        else:
            config = Config()
            config.min_severity = Severity(opts['severity'])
            detector = InefficiencyDetector(code, config)
            
        # Run the analysis
        results = detector.analyze_code()
        
        # Display the results
        self._display_results(results, opts['fix'])
    
    def _display_results(self, results, apply_fixes=False):
        """Display the analysis results as HTML."""
        if not results['optimization_suggestions']:
            display(HTML("<div style='color: green; font-weight: bold;'>✓ No inefficiencies detected</div>"))
            return
            
        html = ["<div style='background-color: #f8f9fa; padding: 10px; border-left: 5px solid #ff9800;'>"]
        html.append("<h3 style='color: #e65100;'>⚠️ ML Code Inefficiencies Detected</h3>")
        html.append("<ul>")
        
        for suggestion in results['optimization_suggestions']:
            severity_color = {
                'low': '#4caf50',
                'medium': '#ff9800',
                'high': '#f44336',
                'critical': '#b71c1c'
            }.get(suggestion['severity'], '#ff9800')
            
            html.append(f"<li><span style='color: {severity_color}; font-weight: bold;'>{suggestion['severity'].upper()}</span>: {suggestion['message']}")
            html.append(f"<p><i>{suggestion['details']}</i></p>")
            html.append("<pre style='background-color: #f1f1f1; padding: 8px;'>")
            html.append(suggestion['code_example'])
            html.append("</pre>")
            html.append("</li>")
            
        html.append("</ul>")
        
        # Display auto-fix options if available
        if results.get('auto_fixes') and apply_fixes:
            html.append("<h4>🔧 Auto-fixes Applied:</h4>")
            html.append("<ul>")
            for name, fix in results['auto_fixes'].items():
                html.append(f"<li><b>{name}</b>")
                html.append(f"<pre style='background-color: #e3f2fd; padding: 8px;'>{fix}</pre>")
                html.append("</li>")
            html.append("</ul>")
            
        html.append(f"<p style='font-size: 0.8em; color: #666;'>ML Advisor v{results['version']}</p>")
        html.append("</div>")
        
        display(HTML("".join(html)))


def load_ipython_extension(ipython):
    """
    Load the extension in IPython.
    
    This function is called when the extension is loaded.
    """
    ipython.register_magics(MLAdvisorMagics)


def unload_ipython_extension(ipython):
    """
    Unload the extension from IPython.
    
    This function is called when the extension is unloaded.
    """
    # Nothing to do here at the moment
    pass
