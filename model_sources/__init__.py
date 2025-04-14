"""
Neural-Scope Model Sources

This module provides utilities for fetching pre-trained models from various sources:
- PyTorch Hub
- TensorFlow Hub
- Hugging Face
- SageMaker Registry
- ONNX Model Zoo
- TensorFlow Model Garden
"""

from model_sources.pytorch_hub import PyTorchHubSource
from model_sources.tensorflow_hub import TensorFlowHubSource
from model_sources.huggingface import HuggingFaceSource
from model_sources.sagemaker import SageMakerSource
from model_sources.onnx_zoo import ONNXModelZooSource
from model_sources.tf_garden import TensorFlowModelGardenSource

__all__ = [
    'PyTorchHubSource',
    'TensorFlowHubSource',
    'HuggingFaceSource',
    'SageMakerSource',
    'ONNXModelZooSource',
    'TensorFlowModelGardenSource',
    'get_model_source'
]

def get_model_source(source_name):
    """
    Get a model source by name.
    
    Args:
        source_name: Name of the model source
        
    Returns:
        Model source instance
    """
    sources = {
        'pytorch_hub': PyTorchHubSource(),
        'tensorflow_hub': TensorFlowHubSource(),
        'huggingface': HuggingFaceSource(),
        'sagemaker': SageMakerSource(),
        'onnx_zoo': ONNXModelZooSource(),
        'tf_garden': TensorFlowModelGardenSource()
    }
    
    if source_name not in sources:
        raise ValueError(f"Unknown model source: {source_name}. Available sources: {', '.join(sources.keys())}")
        
    return sources[source_name]
