"""
Model Registry for Neural-Scope

This module provides a model registry for tracking and managing models.
"""

import os
import json
import logging
import shutil
import datetime
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for tracking and managing models.
    """
    
    def __init__(self, registry_dir: str):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory for the model registry
        """
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.metadata_file = os.path.join(registry_dir, "registry_metadata.json")
        
        # Create registry directories if they don't exist
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load or create registry metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load registry metadata from file.
        
        Returns:
            Dictionary with registry metadata
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry metadata: {e}")
                return {"models": {}}
        else:
            return {"models": {}}
            
    def _save_metadata(self) -> None:
        """Save registry metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry metadata: {e}")
            
    def register_model(self, model_path: str, model_name: str, version: Optional[str] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model file or directory
            model_name: Name of the model
            version: Version of the model (if None, a version will be generated)
            metadata: Additional metadata for the model
            
        Returns:
            Version of the registered model
        """
        # Generate version if not provided
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file or directory
        if os.path.isfile(model_path):
            shutil.copy2(model_path, os.path.join(model_dir, os.path.basename(model_path)))
            model_file = os.path.basename(model_path)
        else:
            shutil.copytree(model_path, os.path.join(model_dir, "model"), dirs_exist_ok=True)
            model_file = "model"
            
        # Create or update model metadata
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {
                "versions": {},
                "latest_version": version,
                "production_version": None,
                "staging_version": None
            }
            
        # Add version metadata
        self.metadata["models"][model_name]["versions"][version] = {
            "created_at": datetime.datetime.now().isoformat(),
            "model_file": model_file,
            "status": "registered",
            "metadata": metadata or {}
        }
        
        # Update latest version
        self.metadata["models"][model_name]["latest_version"] = version
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Registered model {model_name} version {version}")
        
        return version
        
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Get the path to a model in the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, the latest version will be used)
            
        Returns:
            Path to the model
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        # Get version
        if version is None:
            version = self.metadata["models"][model_name]["latest_version"]
        elif version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in registry")
            
        # Get model file
        model_file = self.metadata["models"][model_name]["versions"][version]["model_file"]
        
        # Return path to model
        return os.path.join(self.models_dir, model_name, version, model_file)
        
    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        return list(self.metadata["models"][model_name]["versions"].keys())
        
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, the latest version will be used)
            
        Returns:
            Dictionary with model metadata
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        # Get version
        if version is None:
            version = self.metadata["models"][model_name]["latest_version"]
        elif version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in registry")
            
        return self.metadata["models"][model_name]["versions"][version]["metadata"]
        
    def update_model_metadata(self, model_name: str, version: str, metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            metadata: New metadata for the model
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in registry")
            
        # Update metadata
        self.metadata["models"][model_name]["versions"][version]["metadata"].update(metadata)
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Updated metadata for model {model_name} version {version}")
        
    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """
        Promote a model to a stage.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Stage to promote the model to (staging, production)
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} of model {model_name} not found in registry")
            
        if stage not in ["staging", "production"]:
            raise ValueError(f"Invalid stage: {stage}. Must be 'staging' or 'production'")
            
        # Update stage
        if stage == "staging":
            self.metadata["models"][model_name]["staging_version"] = version
        elif stage == "production":
            self.metadata["models"][model_name]["production_version"] = version
            
        # Update status
        self.metadata["models"][model_name]["versions"][version]["status"] = stage
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Promoted model {model_name} version {version} to {stage}")
        
    def get_production_model(self, model_name: str) -> str:
        """
        Get the production version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the production model
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        production_version = self.metadata["models"][model_name]["production_version"]
        if production_version is None:
            raise ValueError(f"No production version for model {model_name}")
            
        return self.get_model_path(model_name, production_version)
        
    def get_staging_model(self, model_name: str) -> str:
        """
        Get the staging version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the staging model
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        staging_version = self.metadata["models"][model_name]["staging_version"]
        if staging_version is None:
            raise ValueError(f"No staging version for model {model_name}")
            
        return self.get_model_path(model_name, staging_version)
        
    def list_models(self) -> List[str]:
        """
        List all models in the registry.
        
        Returns:
            List of model names
        """
        return list(self.metadata["models"].keys())
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        return self.metadata["models"][model_name]
