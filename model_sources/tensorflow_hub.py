"""
TensorFlow Hub Model Source

This module provides utilities for fetching pre-trained models from TensorFlow Hub.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TensorFlowHubSource:
    """
    Source for TensorFlow Hub models.
    """
    
    def __init__(self):
        """Initialize the TensorFlow Hub source."""
        self.name = "tensorflow_hub"
        self.available_models = {
            # Vision models
            "efficientnet_b0": {"url": "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"},
            "efficientnet_b3": {"url": "https://tfhub.dev/tensorflow/efficientnet/b3/classification/1"},
            "mobilenet_v2_100_224": {"url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"},
            "inception_v3": {"url": "https://tfhub.dev/google/imagenet/inception_v3/classification/5"},
            "resnet_50": {"url": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5"},
            
            # NLP models
            "bert_en_uncased_L-12_H-768_A-12": {"url": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"},
            "albert_en_base": {"url": "https://tfhub.dev/tensorflow/albert_en_base/3"},
            "universal_sentence_encoder": {"url": "https://tfhub.dev/google/universal-sentence-encoder/4"},
            
            # Audio models
            "yamnet": {"url": "https://tfhub.dev/google/yamnet/1"},
            "speech_commands": {"url": "https://tfhub.dev/google/speech_commands/1"}
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
        Fetch a pre-trained model from TensorFlow Hub.
        
        Args:
            model_name: Name of the model to fetch
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            logger.error("TensorFlow and TensorFlow Hub are required to fetch models from TensorFlow Hub")
            raise ImportError("TensorFlow and TensorFlow Hub are required to fetch models from TensorFlow Hub")
            
        if model_name not in self.available_models:
            available_models = self.get_available_models()
            logger.error(f"Model {model_name} not found in TensorFlow Hub. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model {model_name} not found in TensorFlow Hub")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model info
        model_info = self.available_models[model_name]
        model_url = model_info["url"]
        
        # Fetch model from TensorFlow Hub
        logger.info(f"Fetching {model_name} from TensorFlow Hub ({model_url})...")
        try:
            # Load the model
            model = hub.KerasLayer(model_url)
            
            # Create a simple model with the hub layer
            if "classification" in model_url or "imagenet" in model_url:
                # For image classification models
                inputs = tf.keras.Input(shape=(224, 224, 3))
                outputs = model(inputs)
                keras_model = tf.keras.Model(inputs, outputs)
            elif "bert" in model_url or "albert" in model_url:
                # For BERT-like models
                inputs = {
                    "input_word_ids": tf.keras.Input(shape=(128,), dtype=tf.int32),
                    "input_mask": tf.keras.Input(shape=(128,), dtype=tf.int32),
                    "input_type_ids": tf.keras.Input(shape=(128,), dtype=tf.int32)
                }
                outputs = model(inputs)
                keras_model = tf.keras.Model(inputs, outputs)
            else:
                # Generic model
                inputs = tf.keras.Input(shape=(None,))
                outputs = model(inputs)
                keras_model = tf.keras.Model(inputs, outputs)
            
            # Save the model
            model_path = os.path.join(output_dir, model_name)
            keras_model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error fetching model from TensorFlow Hub: {e}")
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
            "efficientnet_b0": {
                "parameters": 5330564,
                "top1_accuracy": 77.1,
                "top5_accuracy": 93.3,
                "inference_time_range": [4, 18],  # ms, depends on hardware
                "size_mb": 20.4,
                "paper_url": "https://arxiv.org/abs/1905.11946",
                "description": "EfficientNet-B0 is a convolutional network that scales depth/width/resolution"
            },
            "mobilenet_v2_100_224": {
                "parameters": 3538984,
                "top1_accuracy": 71.8,
                "top5_accuracy": 90.6,
                "inference_time_range": [3, 15],  # ms, depends on hardware
                "size_mb": 13.5,
                "paper_url": "https://arxiv.org/abs/1801.04381",
                "description": "MobileNetV2 is a lightweight CNN architecture designed for mobile devices"
            },
            "resnet_50": {
                "parameters": 25636712,
                "top1_accuracy": 76.0,
                "top5_accuracy": 92.9,
                "inference_time_range": [10, 40],  # ms, depends on hardware
                "size_mb": 97.8,
                "paper_url": "https://arxiv.org/abs/1512.03385",
                "description": "ResNet-50 is a 50-layer residual network trained on ImageNet"
            }
        }
        
        return metadata.get(model_name, {})
