"""
Neural-Scope CI/CD Templates

This module provides templates for various CI/CD systems to integrate
Neural-Scope's analysis and optimization capabilities into CI/CD pipelines.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Template directory
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_template_path(template_name: str) -> str:
    """
    Get the path to a template file.
    
    Args:
        template_name: Name of the template (e.g., 'github_actions', 'gitlab_ci')
        
    Returns:
        Path to the template file
    """
    template_map = {
        'github_actions': 'github_actions_workflow.yml',
        'gitlab_ci': 'gitlab_ci_workflow.yml',
        'jenkins': 'jenkins_workflow.groovy',
        'azure_devops': 'azure_pipelines_workflow.yml'
    }
    
    if template_name not in template_map:
        raise ValueError(f"Unknown template: {template_name}")
        
    return os.path.join(TEMPLATE_DIR, template_map[template_name])

def load_template(template_name: str) -> str:
    """
    Load a template file.
    
    Args:
        template_name: Name of the template (e.g., 'github_actions', 'gitlab_ci')
        
    Returns:
        Template content as a string
    """
    template_path = get_template_path(template_name)
    
    with open(template_path, 'r') as f:
        return f.read()
        
def save_workflow(template_name: str, 
                output_path: str, 
                variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a workflow file based on a template.
    
    Args:
        template_name: Name of the template (e.g., 'github_actions', 'gitlab_ci')
        output_path: Path to save the workflow file
        variables: Variables to replace in the template
        
    Returns:
        Path to the saved workflow file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Load the template
    template = load_template(template_name)
    
    # Replace variables if provided
    if variables:
        for key, value in variables.items():
            template = template.replace(f"${{{key}}}", str(value))
            
    # Save the workflow file
    with open(output_path, 'w') as f:
        f.write(template)
        
    return output_path
