"""
Tests for the model compression module.

This module contains comprehensive tests for the ModelCompressor class and its
compression techniques: quantization, pruning, and knowledge distillation.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from advanced_analysis.algorithm_complexity.model_compression import ProfileInfo, ModelCompressor

# Skip tests if optional dependencies are not available
pytorch_available = pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not installed"),
    reason="PyTorch not installed"
)

tensorflow_available = pytest.mark.skipif(
    not pytest.importorskip("tensorflow", reason="TensorFlow not installed"),
    reason="TensorFlow not installed"
)

# Test ProfileInfo class
class TestProfileInfo:
    def test_init_with_defaults(self):
        """Test ProfileInfo initialization with default values."""
        profile = ProfileInfo(framework="pytorch", model_type="cnn", hardware="cpu")
        assert profile.framework == "pytorch"
        assert profile.model_type == "cnn"
        assert profile.hardware == "cpu"
        assert profile.techniques == []
        assert profile.params == {}

    def test_init_with_custom_values(self):
        """Test ProfileInfo initialization with custom values."""
        techniques = ["quantization", "pruning"]
        params = {"quantization_method": "dynamic", "prune_amount": 0.3}
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="transformer", 
            hardware="gpu",
            techniques=techniques,
            params=params
        )
        assert profile.framework == "tensorflow"
        assert profile.model_type == "transformer"
        assert profile.hardware == "gpu"
        assert profile.techniques == techniques
        assert profile.params == params

    def test_case_insensitivity(self):
        """Test that framework and model_type are case-insensitive."""
        profile = ProfileInfo(framework="PyTorch", model_type="CNN", hardware="GPU")
        assert profile.framework == "pytorch"
        assert profile.model_type == "cnn"
        assert profile.hardware == "gpu"

# Test ModelCompressor class
class TestModelCompressor:
    def test_init_and_decide_techniques(self):
        """Test ModelCompressor initialization and technique decision."""
        # Test with CPU hardware
        profile = ProfileInfo(framework="pytorch", model_type="cnn", hardware="cpu")
        compressor = ModelCompressor(profile)
        assert "quantization" in compressor.profile.techniques
        assert "pruning" in compressor.profile.techniques
        
        # Test with GPU hardware and transformer model
        profile = ProfileInfo(framework="pytorch", model_type="transformer", hardware="gpu")
        compressor = ModelCompressor(profile)
        assert "quantization" in compressor.profile.techniques
        assert "pruning" in compressor.profile.techniques
        assert "distillation" in compressor.profile.techniques

    def test_get_logs(self):
        """Test log retrieval functionality."""
        profile = ProfileInfo(framework="pytorch", model_type="cnn", hardware="cpu")
        compressor = ModelCompressor(profile)
        # Add a test log
        compressor.logs.append("Test log message")
        logs = compressor.get_logs()
        assert "Test log message" in logs
        assert len(logs) > 0

# PyTorch-specific tests
@pytorch_available
class TestPyTorchCompression:
    @pytest.fixture
    def pytorch_model(self):
        """Create a simple PyTorch model for testing."""
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(5, 2)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
                
        return SimpleModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        import torch
        return [torch.randn(2, 10) for _ in range(5)]
    
    def test_quantization_dynamic(self, pytorch_model, sample_data):
        """Test dynamic quantization for PyTorch models."""
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="mlp", 
            hardware="cpu",
            techniques=["quantization"],
            params={"quantization_method": "dynamic"}
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock torch.quantization.quantize_dynamic to avoid actual quantization
        with patch('torch.quantization.quantize_dynamic', return_value=pytorch_model):
            compressed_model = compressor.compress(pytorch_model)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("dynamic quantization" in log.lower() for log in logs)
        assert any("pytorch" in log.lower() for log in logs)
    
    def test_quantization_static(self, pytorch_model, sample_data):
        """Test static quantization for PyTorch models."""
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="mlp", 
            hardware="cpu",
            techniques=["quantization"],
            params={"quantization_method": "static"}
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock prepare and convert functions
        with patch('torch.quantization.prepare', return_value=pytorch_model), \
             patch('torch.quantization.convert', return_value=pytorch_model):
            compressed_model = compressor.compress(pytorch_model, calibration_data=sample_data)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("static" in log.lower() for log in logs)
        assert any("pytorch" in log.lower() for log in logs)
    
    def test_pruning(self, pytorch_model):
        """Test pruning for PyTorch models."""
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="mlp", 
            hardware="cpu",
            techniques=["pruning"],
            params={"prune_amount": 0.3}
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock prune functions
        with patch('torch.nn.utils.prune.l1_unstructured', return_value=None), \
             patch('torch.nn.utils.prune.remove', return_value=None):
            compressed_model = compressor.compress(pytorch_model)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("pruning" in log.lower() for log in logs)
        assert any("pytorch" in log.lower() for log in logs)
    
    def test_distillation(self, pytorch_model, sample_data):
        """Test knowledge distillation for PyTorch models."""
        import torch
        import torch.nn as nn
        
        # Create a student model
        student_model = nn.Sequential(
            nn.Linear(10, 2)
        )
        
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="mlp", 
            hardware="cpu",
            techniques=["distillation"],
            params={"distill_epochs": 1, "distill_temperature": 2.0}
        )
        
        compressor = ModelCompressor(profile)
        
        # Create mock data with inputs and targets
        mock_data = [(torch.randn(2, 10), torch.tensor([0, 1]))]
        
        # Mock the training process
        with patch.object(torch.optim, 'Adam', return_value=MagicMock()), \
             patch.object(student_model, 'train', return_value=None), \
             patch.object(pytorch_model, 'eval', return_value=None):
            compressed_model = compressor.compress(
                pytorch_model, 
                student_model=student_model,
                data=mock_data
            )
            
        # Check logs
        logs = compressor.get_logs()
        assert any("distillation" in log.lower() for log in logs)
        assert any("pytorch" in log.lower() for log in logs)

# TensorFlow-specific tests
@tensorflow_available
class TestTensorFlowCompression:
    @pytest.fixture
    def tensorflow_model(self):
        """Create a simple TensorFlow model for testing."""
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        import tensorflow as tf
        return [tf.random.normal([2, 10]) for _ in range(5)]
    
    def test_quantization_ptq(self, tensorflow_model, sample_data):
        """Test post-training quantization for TensorFlow models."""
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="mlp", 
            hardware="mobile",
            techniques=["quantization"],
            params={"quantization_method": "ptq"}
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock TFLite converter
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b'mock_tflite_model'
        
        with patch('tensorflow.lite.TFLiteConverter.from_keras_model', return_value=mock_converter):
            compressed_model = compressor.compress(tensorflow_model, calibration_data=sample_data)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("post-training quantization" in log.lower() for log in logs)
        assert any("tensorflow" in log.lower() for log in logs)
    
    def test_quantization_dynamic(self, tensorflow_model):
        """Test dynamic range quantization for TensorFlow models."""
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="mlp", 
            hardware="mobile",
            techniques=["quantization"],
            params={"quantization_method": "dynamic"}
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock TFLite converter
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b'mock_tflite_model'
        
        with patch('tensorflow.lite.TFLiteConverter.from_keras_model', return_value=mock_converter):
            compressed_model = compressor.compress(tensorflow_model)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("dynamic range" in log.lower() for log in logs)
        assert any("tensorflow" in log.lower() for log in logs)
    
    def test_pruning(self, tensorflow_model):
        """Test pruning for TensorFlow models."""
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="mlp", 
            hardware="mobile",
            techniques=["pruning"],
            params={"prune_amount": 0.2}
        )
        
        compressor = ModelCompressor(profile)
        
        # Test without tfmot
        with patch.dict('sys.modules', {'tensorflow_model_optimization': None}):
            compressed_model = compressor.compress(tensorflow_model)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("pruning" in log.lower() for log in logs)
        assert any("tensorflow" in log.lower() for log in logs)
    
    def test_distillation(self, tensorflow_model, sample_data):
        """Test knowledge distillation for TensorFlow models."""
        import tensorflow as tf
        
        # Create a student model
        student_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        student_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="mlp", 
            hardware="mobile",
            techniques=["distillation"],
            params={"distill_epochs": 1, "distill_temperature": 2.0}
        )
        
        compressor = ModelCompressor(profile)
        
        # Create mock data with inputs and targets
        import numpy as np
        x = tf.random.normal([5, 10])
        y = tf.constant([0, 1, 0, 1, 0])
        mock_data = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        
        # Mock the training process
        with patch.object(tf.keras.optimizers, 'Adam', return_value=MagicMock()):
            compressed_model = compressor.compress(
                tensorflow_model, 
                student_model=student_model,
                data=mock_data
            )
            
        # Check logs
        logs = compressor.get_logs()
        assert any("distillation" in log.lower() for log in logs)
        assert any("tensorflow" in log.lower() for log in logs)

# Test error handling and edge cases
class TestErrorHandling:
    def test_unsupported_framework(self):
        """Test handling of unsupported frameworks."""
        profile = ProfileInfo(
            framework="unsupported", 
            model_type="cnn", 
            hardware="cpu",
            techniques=["quantization"]
        )
        
        compressor = ModelCompressor(profile)
        model = MagicMock()
        
        # Should not raise an exception
        compressed_model = compressor.compress(model)
        
        # Check logs
        logs = compressor.get_logs()
        assert any("not supported" in log.lower() for log in logs)
    
    def test_missing_student_model(self):
        """Test handling of missing student model for distillation."""
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="cnn", 
            hardware="cpu",
            techniques=["distillation"]
        )
        
        compressor = ModelCompressor(profile)
        model = MagicMock()
        
        # Should not raise an exception
        compressed_model = compressor.compress(model)
        
        # Check logs
        logs = compressor.get_logs()
        assert any("no student model provided" in log.lower() for log in logs)
    
    def test_quantization_error_recovery(self):
        """Test recovery from quantization errors."""
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="cnn", 
            hardware="cpu",
            techniques=["quantization", "pruning"]
        )
        
        compressor = ModelCompressor(profile)
        model = MagicMock()
        
        # Mock quantization to raise an exception
        with patch.object(compressor, '_apply_quantization', side_effect=Exception("Quantization error")):
            # Should not raise an exception and continue with pruning
            compressed_model = compressor.compress(model)
            
        # Check logs
        logs = compressor.get_logs()
        assert any("quantization" in log.lower() for log in logs)
        assert any("pruning" in log.lower() for log in logs)

# Integration tests
@pytest.mark.integration
class TestIntegration:
    @pytorch_available
    def test_pytorch_full_pipeline(self):
        """Test the full compression pipeline with PyTorch."""
        import torch
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Create a student model
        student_model = nn.Sequential(
            nn.Linear(10, 2)
        )
        
        # Create sample data
        sample_data = [(torch.randn(2, 10), torch.tensor([0, 1])) for _ in range(3)]
        
        # Create profile with all techniques
        profile = ProfileInfo(
            framework="pytorch", 
            model_type="mlp", 
            hardware="cpu",
            techniques=["quantization", "pruning", "distillation"],
            params={
                "quantization_method": "dynamic",
                "prune_amount": 0.1,
                "distill_epochs": 1
            }
        )
        
        compressor = ModelCompressor(profile)
        
        # Apply all compression techniques
        with patch('torch.quantization.quantize_dynamic', return_value=model), \
             patch('torch.nn.utils.prune.l1_unstructured', return_value=None), \
             patch('torch.nn.utils.prune.remove', return_value=None):
            compressed_model = compressor.compress(
                model, 
                student_model=student_model,
                data=sample_data
            )
            
        # Check logs
        logs = compressor.get_logs()
        assert any("quantization" in log.lower() for log in logs)
        assert any("pruning" in log.lower() for log in logs)
        assert any("distillation" in log.lower() for log in logs)
        assert any("compression completed" in log.lower() for log in logs)
    
    @tensorflow_available
    def test_tensorflow_full_pipeline(self):
        """Test the full compression pipeline with TensorFlow."""
        import tensorflow as tf
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Create a student model
        student_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Create sample data
        x = tf.random.normal([6, 10])
        y = tf.constant([0, 1, 0, 1, 0, 1])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        
        # Create profile with all techniques
        profile = ProfileInfo(
            framework="tensorflow", 
            model_type="mlp", 
            hardware="mobile",
            techniques=["quantization", "pruning", "distillation"],
            params={
                "quantization_method": "dynamic",
                "prune_amount": 0.1,
                "distill_epochs": 1
            }
        )
        
        compressor = ModelCompressor(profile)
        
        # Mock TFLite converter
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b'mock_tflite_model'
        
        # Apply all compression techniques
        with patch('tensorflow.lite.TFLiteConverter.from_keras_model', return_value=mock_converter), \
             patch.dict('sys.modules', {'tensorflow_model_optimization': None}):
            compressed_model = compressor.compress(
                model, 
                student_model=student_model,
                data=dataset
            )
            
        # Check logs
        logs = compressor.get_logs()
        assert any("quantization" in log.lower() for log in logs)
        assert any("pruning" in log.lower() for log in logs)
        assert any("distillation" in log.lower() for log in logs)
        assert any("compression completed" in log.lower() for log in logs)
