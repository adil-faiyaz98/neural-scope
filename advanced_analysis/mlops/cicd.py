"""
CI/CD Integration for Neural-Scope

This module provides tools for integrating Neural-Scope analysis and optimization
capabilities into CI/CD pipelines, including GitHub Actions, GitLab CI, Jenkins,
and Azure DevOps.
"""

import os
import logging
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CICDSystem(Enum):
    """Supported CI/CD systems."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLE_CI = "circle_ci"
    TRAVIS_CI = "travis_ci"
    CUSTOM = "custom"

class CICDIntegrator:
    """
    Integrates Neural-Scope analysis and optimization capabilities into CI/CD pipelines.

    This class provides tools for creating CI/CD workflows that incorporate Neural-Scope's
    analysis and optimization capabilities, making it easy to automate model optimization
    and validation as part of the CI/CD process.
    """

    def __init__(self, system: Union[str, CICDSystem] = CICDSystem.GITHUB_ACTIONS,
                 config_path: Optional[str] = None):
        """
        Initialize the CI/CD integrator.

        Args:
            system: The CI/CD system to use (github_actions, gitlab_ci, jenkins, azure_devops)
            config_path: Path to a configuration file for the CI/CD system
        """
        if isinstance(system, str):
            try:
                self.system = CICDSystem(system)
            except ValueError:
                logger.warning(f"Unknown CI/CD system: {system}. Using CUSTOM.")
                self.system = CICDSystem.CUSTOM
        else:
            self.system = system

        self.config = {}
        if config_path:
            self._load_config(config_path)

        # Templates module
        from advanced_analysis.mlops import templates

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    self.config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def create_optimization_workflow(self,
                                    optimization_script: str,
                                    test_script: Optional[str] = None,
                                    output_dir: str = ".github/workflows",
                                    workflow_name: str = "model_optimization",
                                    trigger_on: List[str] = None,
                                    notify_on_completion: bool = False,
                                    requirements: List[str] = None,
                                    python_version: str = "3.9",
                                    timeout_minutes: int = 60) -> str:
        """
        Create a CI/CD workflow for model optimization.

        Args:
            optimization_script: Path to the script that performs model optimization
            test_script: Path to the script that validates the optimized model
            output_dir: Directory to save the workflow file
            workflow_name: Name of the workflow
            trigger_on: List of events to trigger the workflow (e.g., ["push", "pull_request"])
            notify_on_completion: Whether to send notifications when the workflow completes
            requirements: List of additional requirements to install
            python_version: Python version to use
            timeout_minutes: Timeout for the workflow in minutes

        Returns:
            Path to the created workflow file
        """
        if trigger_on is None:
            trigger_on = ["push", "pull_request"]

        if requirements is None:
            requirements = []

        # Create the workflow based on the CI/CD system
        if self.system == CICDSystem.GITHUB_ACTIONS:
            return self._create_github_actions_workflow(
                optimization_script=optimization_script,
                test_script=test_script,
                output_dir=output_dir,
                workflow_name=workflow_name,
                trigger_on=trigger_on,
                notify_on_completion=notify_on_completion,
                requirements=requirements,
                python_version=python_version,
                timeout_minutes=timeout_minutes
            )
        elif self.system == CICDSystem.GITLAB_CI:
            return self._create_gitlab_ci_workflow(
                optimization_script=optimization_script,
                test_script=test_script,
                output_dir=output_dir,
                workflow_name=workflow_name,
                trigger_on=trigger_on,
                notify_on_completion=notify_on_completion,
                requirements=requirements,
                python_version=python_version,
                timeout_minutes=timeout_minutes
            )
        elif self.system == CICDSystem.JENKINS:
            return self._create_jenkins_workflow(
                optimization_script=optimization_script,
                test_script=test_script,
                output_dir=output_dir,
                workflow_name=workflow_name,
                requirements=requirements,
                python_version=python_version,
                timeout_minutes=timeout_minutes
            )
        elif self.system == CICDSystem.AZURE_DEVOPS:
            return self._create_azure_devops_workflow(
                optimization_script=optimization_script,
                test_script=test_script,
                output_dir=output_dir,
                workflow_name=workflow_name,
                trigger_on=trigger_on,
                notify_on_completion=notify_on_completion,
                requirements=requirements,
                python_version=python_version,
                timeout_minutes=timeout_minutes
            )
        else:
            raise ValueError(f"Unsupported CI/CD system: {self.system}")

    def _create_github_actions_workflow(self,
                                      optimization_script: str,
                                      test_script: Optional[str],
                                      output_dir: str,
                                      workflow_name: str,
                                      trigger_on: List[str],
                                      notify_on_completion: bool,
                                      requirements: List[str],
                                      python_version: str,
                                      timeout_minutes: int) -> str:
        """
        Create a GitHub Actions workflow for model optimization.

        Args:
            optimization_script: Path to the script that performs model optimization
            test_script: Path to the script that validates the optimized model
            output_dir: Directory to save the workflow file
            workflow_name: Name of the workflow
            trigger_on: List of events to trigger the workflow
            notify_on_completion: Whether to send notifications when the workflow completes
            requirements: List of additional requirements to install
            python_version: Python version to use
            timeout_minutes: Timeout for the workflow in minutes

        Returns:
            Path to the created workflow file
        """
        # Create the workflow directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create the workflow file
        workflow_file = os.path.join(output_dir, f"{workflow_name}.yml")

        # Create the workflow content
        workflow = {
            "name": f"Neural-Scope {workflow_name.replace('_', ' ').title()}",
            "on": {},
            "jobs": {
                "optimize": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": timeout_minutes,
                    "steps": [
                        {
                            "name": "Checkout Repository",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": python_version
                            }
                        },
                        {
                            "name": "Install Dependencies",
                            "run": "\n".join([
                                "python -m pip install --upgrade pip",
                                "pip install -e .[all]",
                                *[f"pip install {req}" for req in requirements]
                            ])
                        },
                        {
                            "name": "Run Optimization",
                            "run": f"python {optimization_script}"
                        }
                    ]
                }
            }
        }

        # Add triggers
        for trigger in trigger_on:
            if trigger == "push":
                workflow["on"]["push"] = {"branches": ["main"]}
            elif trigger == "pull_request":
                workflow["on"]["pull_request"] = {"branches": ["main"]}
            elif trigger == "schedule":
                workflow["on"]["schedule"] = [{"cron": "0 0 * * 0"}]  # Weekly on Sunday

        # Add test step if a test script is provided
        if test_script:
            workflow["jobs"]["optimize"]["steps"].append({
                "name": "Validate Optimized Model",
                "run": f"python {test_script}"
            })

        # Add notification step if requested
        if notify_on_completion:
            workflow["jobs"]["optimize"]["steps"].append({
                "name": "Send Notification",
                "if": "always()",
                "uses": "rtCamp/action-slack-notify@v2",
                "env": {
                    "SLACK_WEBHOOK": "${{ secrets.SLACK_WEBHOOK }}",
                    "SLACK_CHANNEL": "model-optimization",
                    "SLACK_TITLE": f"Neural-Scope {workflow_name.replace('_', ' ').title()} Workflow",
                    "SLACK_MESSAGE": "Model optimization workflow completed",
                    "SLACK_COLOR": "${{ job.status }}",
                    "SLACK_FOOTER": "Neural-Scope CI/CD Integration"
                }
            })

        # Write the workflow to a file
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow, f, sort_keys=False)

        logger.info(f"Created GitHub Actions workflow: {workflow_file}")
        return workflow_file

    def _create_gitlab_ci_workflow(self,
                                 optimization_script: str,
                                 test_script: Optional[str],
                                 output_dir: str,
                                 workflow_name: str,
                                 trigger_on: List[str],
                                 notify_on_completion: bool,
                                 requirements: List[str],
                                 python_version: str,
                                 timeout_minutes: int) -> str:
        """
        Create a GitLab CI workflow for model optimization.

        Args:
            optimization_script: Path to the script that performs model optimization
            test_script: Path to the script that validates the optimized model
            output_dir: Directory to save the workflow file
            workflow_name: Name of the workflow
            trigger_on: List of events to trigger the workflow
            notify_on_completion: Whether to send notifications when the workflow completes
            requirements: List of additional requirements to install
            python_version: Python version to use
            timeout_minutes: Timeout for the workflow in minutes

        Returns:
            Path to the created workflow file
        """
        # Create the workflow file
        workflow_file = os.path.join(output_dir, ".gitlab-ci.yml")

        # Create the workflow content
        workflow = {
            "stages": ["optimize", "test"],
            "variables": {
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.pip-cache"
            },
            "cache": {
                "paths": [".pip-cache/"]
            },
            "optimize": {
                "stage": "optimize",
                "image": f"python:{python_version}",
                "script": [
                    "python -m pip install --upgrade pip",
                    "pip install -e .[all]",
                    *[f"pip install {req}" for req in requirements],
                    f"python {optimization_script}"
                ],
                "artifacts": {
                    "paths": ["optimized_model.pt", "optimization_results.json"],
                    "expire_in": "1 week"
                },
                "timeout": f"{timeout_minutes}m"
            }
        }

        # Add test job if a test script is provided
        if test_script:
            workflow["test"] = {
                "stage": "test",
                "image": f"python:{python_version}",
                "script": [
                    "python -m pip install --upgrade pip",
                    "pip install -e .[all]",
                    *[f"pip install {req}" for req in requirements],
                    f"python {test_script}"
                ],
                "dependencies": ["optimize"],
                "timeout": f"{timeout_minutes}m"
            }

        # Add notification job if requested
        if notify_on_completion:
            workflow["notify"] = {
                "stage": "test",
                "script": [
                    'curl -X POST -H "Content-Type: application/json" -d "{\\"text\\":\\"Model optimization workflow completed\\"}" $WEBHOOK_URL'
                ],
                "when": "always",
                "dependencies": ["optimize"]
            }
            if test_script:
                workflow["notify"]["dependencies"].append("test")

        # Write the workflow to a file
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow, f, sort_keys=False)

        logger.info(f"Created GitLab CI workflow: {workflow_file}")
        return workflow_file

    def _create_jenkins_workflow(self,
                               optimization_script: str,
                               test_script: Optional[str],
                               output_dir: str,
                               workflow_name: str,
                               requirements: List[str],
                               python_version: str,
                               timeout_minutes: int) -> str:
        """
        Create a Jenkins workflow for model optimization.

        Args:
            optimization_script: Path to the script that performs model optimization
            test_script: Path to the script that validates the optimized model
            output_dir: Directory to save the workflow file
            workflow_name: Name of the workflow
            requirements: List of additional requirements to install
            python_version: Python version to use
            timeout_minutes: Timeout for the workflow in minutes

        Returns:
            Path to the created workflow file
        """
        # Create the workflow file
        workflow_file = os.path.join(output_dir, "Jenkinsfile")

        # Create the workflow content
        workflow = f"""pipeline {{
    agent {{
        docker {{
            image 'python:{python_version}'
        }}
    }}

    options {{
        timeout(time: {timeout_minutes}, unit: 'MINUTES')
    }}

    stages {{
        stage('Install Dependencies') {{
            steps {{
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -e .[all]'
                {' '.join([f"sh 'pip install {req}'" for req in requirements])}
            }}
        }}

        stage('Run Optimization') {{
            steps {{
                sh 'python {optimization_script}'
            }}
        }}
        """

        # Add test stage if a test script is provided
        if test_script:
            workflow += f"""
        stage('Validate Optimized Model') {{
            steps {{
                sh 'python {test_script}'
            }}
        }}
        """

        # Close the workflow
        workflow += """
    }

    post {
        always {
            archiveArtifacts artifacts: 'optimized_model.pt,optimization_results.json', allowEmptyArchive: true
        }
    }
}
"""

        # Write the workflow to a file
        with open(workflow_file, 'w') as f:
            f.write(workflow)

        logger.info(f"Created Jenkins workflow: {workflow_file}")
        return workflow_file

    def _create_azure_devops_workflow(self,
                                    optimization_script: str,
                                    test_script: Optional[str],
                                    output_dir: str,
                                    workflow_name: str,
                                    trigger_on: List[str],
                                    notify_on_completion: bool,
                                    requirements: List[str],
                                    python_version: str,
                                    timeout_minutes: int) -> str:
        """
        Create an Azure DevOps workflow for model optimization.

        Args:
            optimization_script: Path to the script that performs model optimization
            test_script: Path to the script that validates the optimized model
            output_dir: Directory to save the workflow file
            workflow_name: Name of the workflow
            trigger_on: List of events to trigger the workflow
            notify_on_completion: Whether to send notifications when the workflow completes
            requirements: List of additional requirements to install
            python_version: Python version to use
            timeout_minutes: Timeout for the workflow in minutes

        Returns:
            Path to the created workflow file
        """
        # Create the workflow file
        workflow_file = os.path.join(output_dir, "azure-pipelines.yml")

        # Create the workflow content
        workflow = {
            "trigger": [],
            "pool": {
                "vmImage": "ubuntu-latest"
            },
            "jobs": [
                {
                    "job": "Optimize",
                    "timeoutInMinutes": timeout_minutes,
                    "steps": [
                        {
                            "task": "UsePythonVersion@0",
                            "inputs": {
                                "versionSpec": python_version,
                                "addToPath": "true"
                            }
                        },
                        {
                            "script": "python -m pip install --upgrade pip",
                            "displayName": "Install pip"
                        },
                        {
                            "script": "pip install -e .[all]",
                            "displayName": "Install neural-scope"
                        },
                        *[{
                            "script": f"pip install {req}",
                            "displayName": f"Install {req}"
                        } for req in requirements],
                        {
                            "script": f"python {optimization_script}",
                            "displayName": "Run optimization"
                        }
                    ]
                }
            ]
        }

        # Add triggers
        for trigger in trigger_on:
            if trigger == "push":
                workflow["trigger"].append("main")
            elif trigger == "pull_request":
                if "pr" not in workflow:
                    workflow["pr"] = []
                workflow["pr"].append("main")

        # Add test step if a test script is provided
        if test_script:
            workflow["jobs"][0]["steps"].append({
                "script": f"python {test_script}",
                "displayName": "Validate optimized model"
            })

        # Add publish artifacts step
        workflow["jobs"][0]["steps"].append({
            "task": "PublishPipelineArtifact@1",
            "inputs": {
                "targetPath": "$(System.DefaultWorkingDirectory)",
                "artifact": "optimization-results",
                "publishLocation": "pipeline"
            }
        })

        # Write the workflow to a file
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow, f, sort_keys=False)

        logger.info(f"Created Azure DevOps workflow: {workflow_file}")
        return workflow_file

    def create_model_analysis_step(self,
                                  model_path: str,
                                  framework: str = "pytorch",
                                  output_path: str = "model_analysis.json",
                                  analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Create a step for analyzing a model in a CI/CD pipeline.

        Args:
            model_path: Path to the model file
            framework: Model framework (pytorch, tensorflow, sklearn)
            output_path: Path to save the analysis results
            analysis_types: Types of analysis to perform (performance, memory, complexity)

        Returns:
            Dictionary with the step configuration
        """
        if analysis_types is None:
            analysis_types = ["performance", "memory", "complexity"]

        # Create the command
        command = f"neural-scope analyze-model {model_path} --framework {framework} --output {output_path}"

        # Add analysis types
        if analysis_types:
            command += f" --analysis-types {','.join(analysis_types)}"

        # Create the step based on the CI/CD system
        if self.system == CICDSystem.GITHUB_ACTIONS:
            return {
                "name": "Analyze Model",
                "run": command
            }
        elif self.system == CICDSystem.GITLAB_CI:
            return {
                "script": [command]
            }
        elif self.system == CICDSystem.JENKINS:
            return f"sh '{command}'"
        elif self.system == CICDSystem.AZURE_DEVOPS:
            return {
                "script": command,
                "displayName": "Analyze Model"
            }
        else:
            return {"command": command}

    def create_code_analysis_step(self,
                                code_path: str,
                                output_path: str = "code_analysis.json",
                                analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Create a step for analyzing code in a CI/CD pipeline.

        Args:
            code_path: Path to the code file or directory
            output_path: Path to save the analysis results
            analysis_types: Types of analysis to perform (complexity, patterns, inefficiencies)

        Returns:
            Dictionary with the step configuration
        """
        if analysis_types is None:
            analysis_types = ["complexity", "patterns", "inefficiencies"]

        # Create the command
        command = f"neural-scope analyze-code {code_path} --output {output_path}"

        # Add analysis types
        if analysis_types:
            command += f" --analysis-types {','.join(analysis_types)}"

        # Create the step based on the CI/CD system
        if self.system == CICDSystem.GITHUB_ACTIONS:
            return {
                "name": "Analyze Code",
                "run": command
            }
        elif self.system == CICDSystem.GITLAB_CI:
            return {
                "script": [command]
            }
        elif self.system == CICDSystem.JENKINS:
            return f"sh '{command}'"
        elif self.system == CICDSystem.AZURE_DEVOPS:
            return {
                "script": command,
                "displayName": "Analyze Code"
            }
        else:
            return {"command": command}

    def create_data_analysis_step(self,
                                data_path: str,
                                output_path: str = "data_analysis.json",
                                format: str = "json") -> Dict[str, Any]:
        """
        Create a step for analyzing data in a CI/CD pipeline.

        Args:
            data_path: Path to the data file
            output_path: Path to save the analysis results
            format: Output format (json, html, text)

        Returns:
            Dictionary with the step configuration
        """
        # Create the command
        command = f"neural-scope analyze-data {data_path} --output {output_path} --format {format}"

        # Create the step based on the CI/CD system
        if self.system == CICDSystem.GITHUB_ACTIONS:
            return {
                "name": "Analyze Data",
                "run": command
            }
        elif self.system == CICDSystem.GITLAB_CI:
            return {
                "script": [command]
            }
        elif self.system == CICDSystem.JENKINS:
            return f"sh '{command}'"
        elif self.system == CICDSystem.AZURE_DEVOPS:
            return {
                "script": command,
                "displayName": "Analyze Data"
            }
        else:
            return {"command": command}

    def create_model_compression_step(self,
                                    model_path: str,
                                    output_path: str,
                                    framework: str = "pytorch",
                                    techniques: List[str] = None) -> Dict[str, Any]:
        """
        Create a step for compressing a model in a CI/CD pipeline.

        Args:
            model_path: Path to the model file
            output_path: Path to save the compressed model
            framework: Model framework (pytorch, tensorflow)
            techniques: Compression techniques to apply (quantization, pruning, distillation)

        Returns:
            Dictionary with the step configuration
        """
        if techniques is None:
            techniques = ["quantization", "pruning"]

        # Create the command
        command = f"neural-scope compress-model {model_path} --output {output_path} --framework {framework}"

        # Add techniques
        if techniques:
            command += f" --techniques {','.join(techniques)}"

        # Create the step based on the CI/CD system
        if self.system == CICDSystem.GITHUB_ACTIONS:
            return {
                "name": "Compress Model",
                "run": command
            }
        elif self.system == CICDSystem.GITLAB_CI:
            return {
                "script": [command]
            }
        elif self.system == CICDSystem.JENKINS:
            return f"sh '{command}'"
        elif self.system == CICDSystem.AZURE_DEVOPS:
            return {
                "script": command,
                "displayName": "Compress Model"
            }
        else:
            return {"command": command}
