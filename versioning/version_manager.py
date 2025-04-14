"""
Version Manager for Neural-Scope

This module provides a version manager for tracking model versions and promotions.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class VersionManager:
    """
    Manager for tracking model versions and promotions.
    """
    
    def __init__(self, registry_dir: str, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize the version manager.
        
        Args:
            registry_dir: Directory for the model registry
            mlflow_tracking_uri: URI for MLflow tracking server
        """
        self.registry_dir = registry_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.history_file = os.path.join(registry_dir, "version_history.json")
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)
        
        # Load or create version history
        self.history = self._load_history()
        
    def _load_history(self) -> Dict[str, Any]:
        """
        Load version history from file.
        
        Returns:
            Dictionary with version history
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version history: {e}")
                return {"models": {}}
        else:
            return {"models": {}}
            
    def _save_history(self) -> None:
        """Save version history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version history: {e}")
            
    def register_version(self, model_name: str, version: str, metrics: Dict[str, float], 
                       tags: Optional[Dict[str, str]] = None) -> None:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            metrics: Performance metrics for the model
            tags: Tags for the model
        """
        # Create model entry if it doesn't exist
        if model_name not in self.history["models"]:
            self.history["models"][model_name] = {
                "versions": {},
                "promotions": []
            }
            
        # Add version entry
        self.history["models"][model_name]["versions"][version] = {
            "created_at": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "tags": tags or {},
            "status": "registered"
        }
        
        # Save history
        self._save_history()
        
        logger.info(f"Registered version {version} for model {model_name}")
        
        # Register with MLflow if available
        if self.mlflow_tracking_uri:
            self._register_with_mlflow(model_name, version, metrics, tags)
            
    def _register_with_mlflow(self, model_name: str, version: str, metrics: Dict[str, float], 
                            tags: Optional[Dict[str, str]] = None) -> None:
        """
        Register a model version with MLflow.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            metrics: Performance metrics for the model
            tags: Tags for the model
        """
        try:
            import mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Start a new run
            with mlflow.start_run() as run:
                # Log metrics
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                    
                # Log tags
                if tags:
                    for name, value in tags.items():
                        mlflow.set_tag(name, value)
                        
                # Set model name and version tags
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("model_version", version)
                
                logger.info(f"Registered model {model_name} version {version} with MLflow")
        except ImportError:
            logger.warning("MLflow not available. Skipping MLflow registration.")
        except Exception as e:
            logger.error(f"Error registering with MLflow: {e}")
            
    def promote_version(self, model_name: str, version: str, stage: str, 
                      reason: Optional[str] = None) -> None:
        """
        Promote a model version to a stage.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Stage to promote the model to (staging, production)
            reason: Reason for the promotion
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        if version not in self.history["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in version history")
            
        if stage not in ["staging", "production"]:
            raise ValueError(f"Invalid stage: {stage}. Must be 'staging' or 'production'")
            
        # Update version status
        self.history["models"][model_name]["versions"][version]["status"] = stage
        
        # Add promotion entry
        promotion = {
            "version": version,
            "stage": stage,
            "promoted_at": datetime.datetime.now().isoformat(),
            "reason": reason or "Manual promotion"
        }
        
        self.history["models"][model_name]["promotions"].append(promotion)
        
        # Save history
        self._save_history()
        
        logger.info(f"Promoted model {model_name} version {version} to {stage}")
        
        # Promote in MLflow if available
        if self.mlflow_tracking_uri:
            self._promote_in_mlflow(model_name, version, stage)
            
    def _promote_in_mlflow(self, model_name: str, version: str, stage: str) -> None:
        """
        Promote a model version in MLflow.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Stage to promote the model to (staging, production)
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Create client
            client = MlflowClient()
            
            # Get model version
            model_versions = client.search_model_versions(f"name='{model_name}'")
            for mv in model_versions:
                if mv.version == version:
                    # Transition model version stage
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage=stage.capitalize()
                    )
                    
                    logger.info(f"Promoted model {model_name} version {version} to {stage} in MLflow")
                    return
                    
            logger.warning(f"Model {model_name} version {version} not found in MLflow")
        except ImportError:
            logger.warning("MLflow not available. Skipping MLflow promotion.")
        except Exception as e:
            logger.error(f"Error promoting in MLflow: {e}")
            
    def get_version_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get version history for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with version history
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        return self.history["models"][model_name]
        
    def get_current_production_version(self, model_name: str) -> Optional[str]:
        """
        Get the current production version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Production version or None if no production version exists
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        # Find the latest production promotion
        promotions = self.history["models"][model_name]["promotions"]
        for promotion in reversed(promotions):
            if promotion["stage"] == "production":
                return promotion["version"]
                
        return None
        
    def get_current_staging_version(self, model_name: str) -> Optional[str]:
        """
        Get the current staging version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Staging version or None if no staging version exists
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        # Find the latest staging promotion
        promotions = self.history["models"][model_name]["promotions"]
        for promotion in reversed(promotions):
            if promotion["stage"] == "staging":
                return promotion["version"]
                
        return None
        
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison results
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        if version1 not in self.history["models"][model_name]["versions"]:
            raise ValueError(f"Version {version1} of model {model_name} not found in version history")
            
        if version2 not in self.history["models"][model_name]["versions"]:
            raise ValueError(f"Version {version2} of model {model_name} not found in version history")
            
        # Get version data
        v1_data = self.history["models"][model_name]["versions"][version1]
        v2_data = self.history["models"][model_name]["versions"][version2]
        
        # Compare metrics
        metric_comparison = {}
        for metric in set(v1_data["metrics"].keys()) | set(v2_data["metrics"].keys()):
            v1_value = v1_data["metrics"].get(metric)
            v2_value = v2_data["metrics"].get(metric)
            
            if v1_value is not None and v2_value is not None:
                diff = v2_value - v1_value
                pct_change = (diff / v1_value) * 100 if v1_value != 0 else float('inf')
                
                metric_comparison[metric] = {
                    "v1_value": v1_value,
                    "v2_value": v2_value,
                    "diff": diff,
                    "pct_change": pct_change
                }
            else:
                metric_comparison[metric] = {
                    "v1_value": v1_value,
                    "v2_value": v2_value,
                    "diff": None,
                    "pct_change": None
                }
                
        return {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "metric_comparison": metric_comparison,
            "v1_created_at": v1_data["created_at"],
            "v2_created_at": v2_data["created_at"],
            "v1_status": v1_data["status"],
            "v2_status": v2_data["status"]
        }
        
    def get_version_metrics(self, model_name: str, version: str) -> Dict[str, float]:
        """
        Get metrics for a model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Dictionary with metrics
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        if version not in self.history["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in version history")
            
        return self.history["models"][model_name]["versions"][version]["metrics"]
        
    def get_promotion_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get promotion history for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of promotion entries
        """
        if model_name not in self.history["models"]:
            raise ValueError(f"Model {model_name} not found in version history")
            
        return self.history["models"][model_name]["promotions"]
