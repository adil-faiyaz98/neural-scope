"""
Kubeflow Integration for Neural-Scope

This module provides tools for integrating Neural-Scope analysis and optimization
capabilities with Kubeflow Pipelines for orchestrating ML workflows.
"""

import os
import logging
import yaml
import json
from typing import Dict, List, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

# Check if Kubeflow Pipelines SDK is available
try:
    import kfp
    from kfp import dsl
    from kfp.components import func_to_container_op
    KUBEFLOW_AVAILABLE = True
except ImportError:
    KUBEFLOW_AVAILABLE = False
    logger.warning("Kubeflow Pipelines SDK not available. Kubeflow integration features will be disabled.")

class KubeflowIntegrator:
    """
    Integrates Neural-Scope analysis and optimization capabilities with Kubeflow Pipelines.
    
    This class provides tools for creating Kubeflow Pipeline components and pipelines
    that incorporate Neural-Scope's analysis and optimization capabilities.
    """
    
    def __init__(self, pipeline_config: Optional[str] = None):
        """
        Initialize the Kubeflow integrator.
        
        Args:
            pipeline_config: Path to a pipeline configuration file
        """
        if not KUBEFLOW_AVAILABLE:
            raise ImportError("Kubeflow Pipelines SDK is not available. Please install it with 'pip install kfp'.")
            
        self.pipeline_name = "neural-scope-pipeline"
        self.pipeline_description = "Neural-Scope analysis and optimization pipeline"
        self.steps = []
        
        # Load configuration if provided
        if pipeline_config:
            self._load_config(pipeline_config)
            
    def _load_config(self, config_path: str) -> None:
        """
        Load pipeline configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
                    
                # Set pipeline properties
                self.pipeline_name = config.get("pipeline_name", self.pipeline_name)
                self.pipeline_description = config.get("pipeline_description", self.pipeline_description)
                
                # Load steps if defined
                if "steps" in config:
                    for step in config["steps"]:
                        self.steps.append(step)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
            
    def create_analysis_step(self,
                           analysis_function: Callable,
                           input_path: str,
                           output_path: str,
                           analysis_type: str = "model",
                           resource_requirements: Optional[Dict[str, str]] = None) -> None:
        """
        Create a step for analyzing a model or code in a Kubeflow pipeline.
        
        Args:
            analysis_function: Function that performs the analysis
            input_path: Path to the input file
            output_path: Path to save the analysis results
            analysis_type: Type of analysis (model, code, data)
            resource_requirements: Resource requirements for the step
        """
        # Create a component from the function
        step = {
            "name": f"{analysis_type}_analysis",
            "function": analysis_function,
            "input_path": input_path,
            "output_path": output_path,
            "resource_requirements": resource_requirements or {}
        }
        
        self.steps.append(step)
        
    def create_optimization_step(self,
                               optimization_function: Callable,
                               resource_requirements: Optional[Dict[str, str]] = None) -> None:
        """
        Create a step for optimizing a model in a Kubeflow pipeline.
        
        Args:
            optimization_function: Function that performs the optimization
            resource_requirements: Resource requirements for the step
        """
        # Create a component from the function
        step = {
            "name": "model_optimization",
            "function": optimization_function,
            "resource_requirements": resource_requirements or {}
        }
        
        self.steps.append(step)
        
    def create_validation_step(self,
                             validation_function: Callable,
                             resource_requirements: Optional[Dict[str, str]] = None) -> None:
        """
        Create a step for validating an optimized model in a Kubeflow pipeline.
        
        Args:
            validation_function: Function that performs the validation
            resource_requirements: Resource requirements for the step
        """
        # Create a component from the function
        step = {
            "name": "model_validation",
            "function": validation_function,
            "resource_requirements": resource_requirements or {}
        }
        
        self.steps.append(step)
        
    def build_pipeline(self) -> Any:
        """
        Build a Kubeflow pipeline from the defined steps.
        
        Returns:
            Kubeflow pipeline
        """
        # Define the pipeline
        @dsl.pipeline(
            name=self.pipeline_name,
            description=self.pipeline_description
        )
        def neural_scope_pipeline():
            """Neural-Scope analysis and optimization pipeline."""
            # Create operations for each step
            ops = {}
            
            for i, step in enumerate(self.steps):
                # Convert function to container op
                op = func_to_container_op(step["function"])
                
                # Set resource requirements if specified
                if step.get("resource_requirements"):
                    op.container.set_cpu_request(step["resource_requirements"].get("cpu", "1"))
                    op.container.set_memory_request(step["resource_requirements"].get("memory", "1Gi"))
                    
                    if "gpu" in step["resource_requirements"]:
                        op.container.set_gpu_limit(step["resource_requirements"]["gpu"])
                        
                # Add the operation to the pipeline
                ops[step["name"]] = op()
                
                # Set dependencies
                if i > 0:
                    ops[step["name"]].after(ops[self.steps[i-1]["name"]])
                    
            return ops
            
        # Compile the pipeline
        pipeline_func = neural_scope_pipeline
        pipeline = kfp.compiler.Compiler().compile(pipeline_func, "neural_scope_pipeline.yaml")
        
        return pipeline
        
    def run_pipeline(self, 
                   experiment_name: str = "neural-scope",
                   run_name: Optional[str] = None,
                   pipeline_file: str = "neural_scope_pipeline.yaml") -> None:
        """
        Run a Kubeflow pipeline.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run
            pipeline_file: Path to the pipeline file
        """
        # Create a client
        client = kfp.Client()
        
        # Create an experiment
        experiment = client.create_experiment(name=experiment_name)
        
        # Run the pipeline
        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name or f"{self.pipeline_name}-{int(time.time())}",
            pipeline_package_path=pipeline_file
        )
        
        logger.info(f"Pipeline run started: {run.id}")
        
    def create_neural_scope_pipeline(self,
                                   model_path: str,
                                   output_dir: str,
                                   framework: str = "pytorch",
                                   optimization_techniques: List[str] = None,
                                   validation_dataset: Optional[str] = None) -> Any:
        """
        Create a complete Neural-Scope pipeline for model analysis and optimization.
        
        Args:
            model_path: Path to the model file
            output_dir: Directory to save outputs
            framework: Model framework (pytorch, tensorflow, sklearn)
            optimization_techniques: List of optimization techniques to apply
            validation_dataset: Path to the validation dataset
            
        Returns:
            Kubeflow pipeline
        """
        if optimization_techniques is None:
            optimization_techniques = ["quantization", "pruning"]
            
        # Define the analysis function
        def analyze_model(model_path: str, output_path: str, framework: str) -> str:
            """Analyze a model using Neural-Scope."""
            from advanced_analysis.analyzer import Analyzer
            
            # Load the model
            model = None
            if framework == "pytorch":
                import torch
                model = torch.load(model_path)
            elif framework == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
            # Create analyzer
            analyzer = Analyzer()
            
            # Analyze the model
            results = analyzer.analyze_model(model)
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            return output_path
            
        # Define the optimization function
        def optimize_model(model_path: str, 
                         output_path: str, 
                         framework: str, 
                         techniques: List[str]) -> str:
            """Optimize a model using Neural-Scope."""
            from advanced_analysis.algorithm_complexity.model_compression import ModelCompressor
            
            # Load the model
            model = None
            if framework == "pytorch":
                import torch
                model = torch.load(model_path)
            elif framework == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
            # Create compressor
            compressor = ModelCompressor()
            
            # Apply optimizations
            optimized_model = compressor.compress_model(
                model=model,
                techniques=techniques
            )
            
            # Save optimized model
            if framework == "pytorch":
                import torch
                torch.save(optimized_model, output_path)
            elif framework == "tensorflow":
                optimized_model.save(output_path)
                
            return output_path
            
        # Define the validation function
        def validate_model(model_path: str, 
                         dataset_path: str, 
                         output_path: str,
                         framework: str) -> str:
            """Validate a model using Neural-Scope."""
            from advanced_analysis.performance import ModelPerformanceProfiler
            
            # Load the model
            model = None
            if framework == "pytorch":
                import torch
                model = torch.load(model_path)
            elif framework == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
            # Load the dataset
            dataset = None
            if dataset_path.endswith('.csv'):
                import pandas as pd
                dataset = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.npz'):
                import numpy as np
                dataset = np.load(dataset_path)
                
            # Create profiler
            profiler = ModelPerformanceProfiler()
            
            # Profile the model
            results = profiler.profile_model(model, dataset)
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            return output_path
            
        # Create the pipeline steps
        analysis_output = os.path.join(output_dir, "model_analysis.json")
        optimized_model = os.path.join(output_dir, "optimized_model")
        validation_output = os.path.join(output_dir, "validation_results.json")
        
        # Add analysis step
        self.create_analysis_step(
            analysis_function=analyze_model,
            input_path=model_path,
            output_path=analysis_output,
            analysis_type="model",
            resource_requirements={"cpu": "2", "memory": "4Gi"}
        )
        
        # Add optimization step
        self.create_optimization_step(
            optimization_function=lambda: optimize_model(
                model_path=model_path,
                output_path=optimized_model,
                framework=framework,
                techniques=optimization_techniques
            ),
            resource_requirements={"cpu": "4", "memory": "8Gi", "gpu": "1"}
        )
        
        # Add validation step if dataset is provided
        if validation_dataset:
            self.create_validation_step(
                validation_function=lambda: validate_model(
                    model_path=optimized_model,
                    dataset_path=validation_dataset,
                    output_path=validation_output,
                    framework=framework
                ),
                resource_requirements={"cpu": "2", "memory": "4Gi"}
            )
            
        # Build the pipeline
        return self.build_pipeline()
