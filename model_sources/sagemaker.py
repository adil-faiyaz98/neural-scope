"""
SageMaker Model Source

This module provides utilities for fetching pre-trained models from Amazon SageMaker.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SageMakerSource:
    """
    Source for Amazon SageMaker models.
    """
    
    def __init__(self):
        """Initialize the SageMaker source."""
        self.name = "sagemaker"
        self.available_models = {
            # Free pre-trained models available in SageMaker
            "jumpstart-pytorch-resnet50": {
                "framework": "pytorch",
                "model_id": "pytorch-ic-resnet50-imagenet",
                "version": "1"
            },
            "jumpstart-pytorch-efficientnet": {
                "framework": "pytorch",
                "model_id": "pytorch-ic-efficientnet-b0-imagenet",
                "version": "1"
            },
            "jumpstart-tensorflow-resnet50": {
                "framework": "tensorflow",
                "model_id": "tensorflow-ic-imagenet-resnet-50-classification",
                "version": "1"
            },
            "jumpstart-tensorflow-mobilenet": {
                "framework": "tensorflow",
                "model_id": "tensorflow-ic-imagenet-mobilenet-v2-100-224-classification",
                "version": "1"
            },
            "jumpstart-huggingface-bert": {
                "framework": "huggingface",
                "model_id": "huggingface-tc-bert-base-uncased",
                "version": "1"
            },
            "jumpstart-huggingface-distilbert": {
                "framework": "huggingface",
                "model_id": "huggingface-tc-distilbert-base-uncased",
                "version": "1"
            }
        }
        
    def get_available_models(self):
        """
        Get a list of available models.
        
        Returns:
            List of available model names
        """
        return list(self.available_models.keys())
        
    def fetch_model(self, model_name, output_dir):
        """
        Fetch a pre-trained model from Amazon SageMaker.
        
        Args:
            model_name: Name of the model to fetch
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        try:
            import boto3
            import sagemaker
        except ImportError:
            logger.error("boto3 and sagemaker libraries are required to fetch models from SageMaker")
            raise ImportError("boto3 and sagemaker libraries are required to fetch models from SageMaker")
            
        if model_name not in self.available_models:
            available_models = self.get_available_models()
            logger.error(f"Model {model_name} not found in SageMaker. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model {model_name} not found in SageMaker")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model info
        model_info = self.available_models[model_name]
        framework = model_info["framework"]
        model_id = model_info["model_id"]
        version = model_info["version"]
        
        # Fetch model from SageMaker
        logger.info(f"Fetching {model_name} from SageMaker JumpStart...")
        try:
            # Initialize SageMaker session
            session = sagemaker.Session()
            region = session.boto_region_name
            
            # Create model directory
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # For demonstration purposes, we'll just save the model info
            # In a real implementation, you would use the SageMaker SDK to download the model
            with open(os.path.join(model_dir, "model_info.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "framework": framework,
                    "model_id": model_id,
                    "version": version,
                    "region": region,
                    "note": "This is a placeholder. In a real implementation, you would download the actual model."
                }, f, indent=2)
            
            logger.info(f"Model info saved to {model_dir}")
            logger.warning("Note: This is a placeholder. In a real implementation, you would download the actual model.")
            
            return model_dir
        except Exception as e:
            logger.error(f"Error fetching model from SageMaker: {e}")
            raise
            
    def get_model_metadata(self, model_name):
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            "jumpstart-pytorch-resnet50": {
                "parameters": 25557032,
                "top1_accuracy": 76.130,
                "top5_accuracy": 92.862,
                "inference_time_range": [10, 40],  # ms, depends on hardware
                "size_mb": 97.8,
                "paper_url": "https://arxiv.org/abs/1512.03385",
                "description": "ResNet-50 is a 50-layer residual network trained on ImageNet"
            },
            "jumpstart-pytorch-efficientnet": {
                "parameters": 5288548,
                "top1_accuracy": 77.1,
                "top5_accuracy": 93.3,
                "inference_time_range": [4, 18],  # ms, depends on hardware
                "size_mb": 20.2,
                "paper_url": "https://arxiv.org/abs/1905.11946",
                "description": "EfficientNet-B0 is a convolutional network that scales depth/width/resolution"
            }
        }
        
        return metadata.get(model_name, {})
