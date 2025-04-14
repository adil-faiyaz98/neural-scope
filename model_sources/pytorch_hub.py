"""
PyTorch Hub Model Source

This module provides utilities for fetching pre-trained models from PyTorch Hub.
"""

import os
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class PyTorchHubSource:
    """
    Source for PyTorch Hub models.
    """
    
    def __init__(self):
        """Initialize the PyTorch Hub source."""
        self.name = "pytorch_hub"
        self.available_models = {
            # Vision models
            "resnet18": {"repo": "pytorch/vision:v0.10.0", "model": "resnet18"},
            "resnet50": {"repo": "pytorch/vision:v0.10.0", "model": "resnet50"},
            "mobilenet_v2": {"repo": "pytorch/vision:v0.10.0", "model": "mobilenet_v2"},
            "densenet121": {"repo": "pytorch/vision:v0.10.0", "model": "densenet121"},
            "efficientnet_b0": {"repo": "pytorch/vision:v0.10.0", "model": "efficientnet_b0"},
            "efficientnet_b1": {"repo": "pytorch/vision:v0.10.0", "model": "efficientnet_b1"},
            "vgg16": {"repo": "pytorch/vision:v0.10.0", "model": "vgg16"},
            "inception_v3": {"repo": "pytorch/vision:v0.10.0", "model": "inception_v3"},
            
            # NLP models
            "bert_base": {"repo": "huggingface/pytorch-transformers", "model": "bertModel"},
            "gpt2": {"repo": "huggingface/pytorch-transformers", "model": "gpt2Model"},
            
            # Audio models
            "wav2vec2": {"repo": "pytorch/fairseq", "model": "wav2vec2_base"},
            
            # Segmentation models
            "fcn_resnet50": {"repo": "pytorch/vision:v0.10.0", "model": "fcn_resnet50"},
            "deeplabv3_resnet50": {"repo": "pytorch/vision:v0.10.0", "model": "deeplabv3_resnet50"}
        }
        
    def get_available_models(self):
        """
        Get a list of available models.
        
        Returns:
            List of available model names
        """
        return list(self.available_models.keys())
        
    def fetch_model(self, model_name, output_dir, pretrained=True):
        """
        Fetch a pre-trained model from PyTorch Hub.
        
        Args:
            model_name: Name of the model to fetch
            output_dir: Directory to save the model
            pretrained: Whether to use pre-trained weights
            
        Returns:
            Path to the saved model
        """
        if model_name not in self.available_models:
            available_models = self.get_available_models()
            logger.error(f"Model {model_name} not found in PyTorch Hub. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model {model_name} not found in PyTorch Hub")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model info
        model_info = self.available_models[model_name]
        repo = model_info["repo"]
        model_func = model_info["model"]
        
        # Fetch model from PyTorch Hub
        logger.info(f"Fetching {model_name} from PyTorch Hub ({repo})...")
        try:
            model = torch.hub.load(repo, model_func, pretrained=pretrained)
            model.eval()
            
            # Save the model
            model_path = os.path.join(output_dir, f"{model_name}.pt")
            torch.save(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error fetching model from PyTorch Hub: {e}")
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
            "resnet18": {
                "parameters": 11689512,
                "top1_accuracy": 69.758,
                "top5_accuracy": 89.078,
                "inference_time_range": [5, 20],  # ms, depends on hardware
                "size_mb": 44.7,
                "paper_url": "https://arxiv.org/abs/1512.03385",
                "description": "ResNet-18 is an 18-layer residual network trained on ImageNet"
            },
            "resnet50": {
                "parameters": 25557032,
                "top1_accuracy": 76.130,
                "top5_accuracy": 92.862,
                "inference_time_range": [10, 40],  # ms, depends on hardware
                "size_mb": 97.8,
                "paper_url": "https://arxiv.org/abs/1512.03385",
                "description": "ResNet-50 is a 50-layer residual network trained on ImageNet"
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
        
        return metadata.get(model_name, {})
