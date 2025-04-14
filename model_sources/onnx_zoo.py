"""
ONNX Model Zoo Source

This module provides utilities for fetching pre-trained models from the ONNX Model Zoo.
"""

import os
import json
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class ONNXModelZooSource:
    """
    Source for ONNX Model Zoo models.
    """
    
    def __init__(self):
        """Initialize the ONNX Model Zoo source."""
        self.name = "onnx_zoo"
        self.base_url = "https://github.com/onnx/models/raw/main/vision"
        self.available_models = {
            # Classification models
            "resnet50-v1": {
                "url": f"{self.base_url}/classification/resnet/model/resnet50-v1-7.onnx",
                "category": "classification"
            },
            "resnet50-v2": {
                "url": f"{self.base_url}/classification/resnet/model/resnet50-v2-7.onnx",
                "category": "classification"
            },
            "mobilenet-v2": {
                "url": f"{self.base_url}/classification/mobilenet/model/mobilenetv2-7.onnx",
                "category": "classification"
            },
            "squeezenet": {
                "url": f"{self.base_url}/classification/squeezenet/model/squeezenet1.1-7.onnx",
                "category": "classification"
            },
            "vgg16": {
                "url": f"{self.base_url}/classification/vgg/model/vgg16-7.onnx",
                "category": "classification"
            },
            
            # Object detection models
            "ssd-mobilenetv1": {
                "url": f"{self.base_url}/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
                "category": "object_detection"
            },
            "yolov3": {
                "url": f"{self.base_url}/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
                "category": "object_detection"
            },
            
            # Segmentation models
            "fcn-resnet50": {
                "url": f"{self.base_url}/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx",
                "category": "segmentation"
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
        Fetch a pre-trained model from the ONNX Model Zoo.
        
        Args:
            model_name: Name of the model to fetch
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if model_name not in self.available_models:
            available_models = self.get_available_models()
            logger.error(f"Model {model_name} not found in ONNX Model Zoo. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model {model_name} not found in ONNX Model Zoo")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model info
        model_info = self.available_models[model_name]
        model_url = model_info["url"]
        category = model_info["category"]
        
        # Fetch model from ONNX Model Zoo
        logger.info(f"Fetching {model_name} from ONNX Model Zoo ({model_url})...")
        try:
            # Create model directory
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Download the model
            model_path = os.path.join(model_dir, f"{model_name}.onnx")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Save model info
            with open(os.path.join(model_dir, "model_info.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "category": category,
                    "url": model_url
                }, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error fetching model from ONNX Model Zoo: {e}")
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
            "resnet50-v1": {
                "parameters": 25557032,
                "top1_accuracy": 76.130,
                "top5_accuracy": 92.862,
                "inference_time_range": [10, 40],  # ms, depends on hardware
                "size_mb": 97.8,
                "paper_url": "https://arxiv.org/abs/1512.03385",
                "description": "ResNet-50 v1 is a 50-layer residual network trained on ImageNet"
            },
            "mobilenet-v2": {
                "parameters": 3504872,
                "top1_accuracy": 71.878,
                "top5_accuracy": 90.286,
                "inference_time_range": [3, 15],  # ms, depends on hardware
                "size_mb": 13.6,
                "paper_url": "https://arxiv.org/abs/1801.04381",
                "description": "MobileNetV2 is a lightweight CNN architecture designed for mobile devices"
            },
            "squeezenet": {
                "parameters": 1248424,
                "top1_accuracy": 58.0,
                "top5_accuracy": 81.0,
                "inference_time_range": [2, 10],  # ms, depends on hardware
                "size_mb": 4.8,
                "paper_url": "https://arxiv.org/abs/1602.07360",
                "description": "SqueezeNet is a small CNN architecture that achieves AlexNet-level accuracy with 50x fewer parameters"
            }
        }
        
        return metadata.get(model_name, {})
