"""
MLflow Integration for Neural-Scope

This module provides tools for integrating Neural-Scope analysis and optimization
capabilities with MLflow for experiment tracking and model registry.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. MLflow integration features will be disabled.")

class MLflowIntegrator:
    """
    Integrates Neural-Scope analysis and optimization capabilities with MLflow.
    
    This class provides tools for tracking Neural-Scope analysis results and
    optimization metrics in MLflow, as well as registering optimized models
    in the MLflow model registry.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "neural-scope"):
        """
        Initialize the MLflow integrator.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not available. Please install it with 'pip install mlflow'.")
            
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        # Set up experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        
    def track_model_analysis(self, 
                           model_name: str,
                           analysis_results: Dict[str, Any],
                           metrics: Optional[Dict[str, float]] = None,
                           tags: Optional[Dict[str, str]] = None,
                           artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Track model analysis results in MLflow.
        
        Args:
            model_name: Name of the model
            analysis_results: Analysis results from Neural-Scope
            metrics: Additional metrics to track
            tags: Tags to add to the run
            artifacts: Artifacts to log (path_name -> local_path)
            
        Returns:
            MLflow run ID
        """
        # Start a new run
        with mlflow.start_run() as run:
            # Log model name as a tag
            mlflow.set_tag("model_name", model_name)
            
            # Log analysis results as metrics
            self._log_analysis_metrics(analysis_results)
            
            # Log additional metrics
            if metrics:
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                    
            # Log tags
            if tags:
                for name, value in tags.items():
                    mlflow.set_tag(name, value)
                    
            # Log analysis results as a JSON artifact
            with open("analysis_results.json", "w") as f:
                json.dump(analysis_results, f, indent=2)
            mlflow.log_artifact("analysis_results.json")
            
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
                    
            return run.info.run_id
            
    def _log_analysis_metrics(self, analysis_results: Dict[str, Any]) -> None:
        """
        Log analysis results as metrics in MLflow.
        
        Args:
            analysis_results: Analysis results from Neural-Scope
        """
        # Extract metrics from analysis results
        metrics = {}
        
        # Performance metrics
        if "performance" in analysis_results:
            perf = analysis_results["performance"]
            if isinstance(perf, dict):
                for key, value in perf.items():
                    if isinstance(value, (int, float)):
                        metrics[f"performance.{key}"] = value
                        
        # Memory metrics
        if "memory" in analysis_results:
            mem = analysis_results["memory"]
            if isinstance(mem, dict):
                for key, value in mem.items():
                    if isinstance(value, (int, float)):
                        metrics[f"memory.{key}"] = value
                        
        # Complexity metrics
        if "complexity" in analysis_results:
            comp = analysis_results["complexity"]
            if isinstance(comp, dict):
                for key, value in comp.items():
                    if isinstance(value, (int, float)):
                        metrics[f"complexity.{key}"] = value
                        
        # Log all metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
    def register_optimized_model(self,
                               original_model: Any,
                               optimized_model: Any,
                               optimization_history: Dict[str, Any],
                               model_name: str,
                               framework: str = "pytorch",
                               tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register an optimized model in the MLflow model registry.
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
            optimization_history: Optimization history
            model_name: Name for the registered model
            framework: Model framework (pytorch, tensorflow, sklearn)
            tags: Tags to add to the model
            
        Returns:
            Model version
        """
        # Start a new run
        with mlflow.start_run() as run:
            # Log optimization history as metrics
            self._log_optimization_metrics(optimization_history)
            
            # Log optimization history as a JSON artifact
            with open("optimization_history.json", "w") as f:
                json.dump(optimization_history, f, indent=2)
            mlflow.log_artifact("optimization_history.json")
            
            # Log tags
            if tags:
                for name, value in tags.items():
                    mlflow.set_tag(name, value)
                    
            # Log the model based on the framework
            if framework.lower() == "pytorch":
                import torch
                mlflow.pytorch.log_model(optimized_model, "model")
            elif framework.lower() == "tensorflow":
                import tensorflow as tf
                mlflow.tensorflow.log_model(optimized_model, "model")
            elif framework.lower() == "sklearn":
                import sklearn
                mlflow.sklearn.log_model(optimized_model, "model")
            else:
                raise ValueError(f"Unsupported framework: {framework}")
                
            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            model_details = mlflow.register_model(model_uri, model_name)
            
            return model_details.version
            
    def _log_optimization_metrics(self, optimization_history: Dict[str, Any]) -> None:
        """
        Log optimization history as metrics in MLflow.
        
        Args:
            optimization_history: Optimization history from Neural-Scope
        """
        # Extract metrics from optimization history
        metrics = {}
        
        # Compression metrics
        if "compression" in optimization_history:
            comp = optimization_history["compression"]
            if isinstance(comp, dict):
                for key, value in comp.items():
                    if isinstance(value, (int, float)):
                        metrics[f"compression.{key}"] = value
                        
        # Performance metrics
        if "performance" in optimization_history:
            perf = optimization_history["performance"]
            if isinstance(perf, dict):
                for key, value in perf.items():
                    if isinstance(value, (int, float)):
                        metrics[f"performance.{key}"] = value
                        
        # Accuracy metrics
        if "accuracy" in optimization_history:
            acc = optimization_history["accuracy"]
            if isinstance(acc, dict):
                for key, value in acc.items():
                    if isinstance(value, (int, float)):
                        metrics[f"accuracy.{key}"] = value
                        
        # Log all metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
    def create_optimization_experiment(self, 
                                     model_name: str,
                                     optimization_techniques: List[str],
                                     dataset_name: Optional[str] = None,
                                     description: Optional[str] = None) -> str:
        """
        Create a new MLflow experiment for model optimization.
        
        Args:
            model_name: Name of the model
            optimization_techniques: List of optimization techniques to apply
            dataset_name: Name of the dataset
            description: Experiment description
            
        Returns:
            MLflow experiment ID
        """
        # Create experiment name
        experiment_name = f"neural-scope-optimization-{model_name}"
        
        # Create experiment
        experiment = mlflow.set_experiment(experiment_name)
        
        # Set experiment tags
        with mlflow.start_run() as run:
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("optimization_techniques", ",".join(optimization_techniques))
            
            if dataset_name:
                mlflow.set_tag("dataset_name", dataset_name)
                
            if description:
                mlflow.set_tag("description", description)
                
        return experiment.experiment_id
        
    def log_optimization_step(self,
                            technique: str,
                            metrics: Dict[str, float],
                            parameters: Optional[Dict[str, Any]] = None,
                            artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Log an optimization step in MLflow.
        
        Args:
            technique: Optimization technique applied
            metrics: Metrics to track
            parameters: Parameters used for the optimization
            artifacts: Artifacts to log (path_name -> local_path)
            
        Returns:
            MLflow run ID
        """
        # Start a new run
        with mlflow.start_run() as run:
            # Log technique as a tag
            mlflow.set_tag("technique", technique)
            
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
            # Log parameters
            if parameters:
                for name, value in parameters.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(name, value)
                    else:
                        mlflow.log_param(name, str(value))
                        
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
                    
            return run.info.run_id
