"""
ML algorithm recognition module.

This module provides advanced tools for identifying machine learning algorithms in code
based on sophisticated pattern matching, AST analysis, and framework-specific detection.
It can recognize algorithms across multiple ML frameworks including scikit-learn, TensorFlow,
PyTorch, XGBoost, and custom implementations.
"""

import ast
import re
import inspect
import importlib
import logging
import os
import sys
import json
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Any, Union, Tuple, Callable
import numpy as np

# Additional imports for advanced profiling
from contextlib import contextmanager
import time
import traceback
import math
from enum import Enum
import functools

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Hardware platforms for optimization
HARDWARE_PLATFORMS = {
    "CPU": {
        "optimizations": [
            "Use MKL-optimized libraries",
            "Consider ONNX Runtime for inference",
            "Enable multi-threading for data loading"
        ]
    },
    "GPU": {
        "optimizations": [
            "Use mixed precision training (FP16)",
            "Optimize memory transfers between CPU and GPU",
            "Adjust batch size to maximize GPU utilization"
        ]
    },
    "TPU": {
        "optimizations": [
            "Use TPU-specific optimizers and layers",
            "Ensure model is compatible with XLA compilation",
            "Use bfloat16 precision"
        ]
    },
    "Edge": {
        "optimizations": [
            "Quantize model to INT8",
            "Prune model to reduce size",
            "Use model distillation for smaller footprint"
        ]
    }
}

# Cloud provider specific optimizations
CLOUD_COST_OPTIMIZATIONS = {
    "AWS": {
        "instance_types": {
            "compute_optimized": ["c5", "c6g", "c7g"],
            "memory_optimized": ["r5", "r6g", "x2gd"],
            "accelerated_computing": ["p3", "p4d", "g4dn"]
        },
        "cost_reduction": [
            "Use spot instances for training",
            "Consider Savings Plans for consistent workloads",
            "Enable auto-scaling for inference endpoints"
        ]
    },
    "GCP": {
        "instance_types": {
            "compute_optimized": ["c2", "c2d"],
            "memory_optimized": ["m1", "m2"],
            "accelerated_computing": ["a2", "g2"]
        },
        "cost_reduction": [
            "Use preemptible VMs for training",
            "Consider committed use discounts",
            "Use Vertex AI for managed training"
        ]
    },
    "Azure": {
        "instance_types": {
            "compute_optimized": ["F-series", "Fsv2-series"],
            "memory_optimized": ["E-series", "Esv4-series"],
            "accelerated_computing": ["NC-series", "ND-series"]
        },
        "cost_reduction": [
            "Use spot instances for training",
            "Consider reserved instances for consistent workloads",
            "Enable autoscaling for inference endpoints"
        ]
    }
}

# Model compression techniques
MODEL_COMPRESSION_TECHNIQUES = {
    "Quantization": {
        "patterns": [r"quantize", r"quantization", r"int8", r"QAT", r"post_training_quantization"],
        "frameworks": {
            "PyTorch": ["torch.quantization", "torch.ao.quantization"],
            "TensorFlow": ["tf.quantization", "TFLiteConverter.optimizations"],
            "ONNX": ["onnxruntime.quantization"]
        },
        "recommendations": [
            "Apply post-training quantization for inference speedup",
            "Use quantization-aware training for better accuracy preservation",
            "Consider dynamic quantization for RNNs and transformers"
        ]
    },
    "Pruning": {
        "patterns": [r"prune", r"sparsity", r"sparse\."], 
        "frameworks": {
            "PyTorch": ["torch.nn.utils.prune", "torch.nn.utils.sparse"],
            "TensorFlow": ["tensorflow_model_optimization.sparsity"],
            "Keras": ["keras_pruning"]
        },
        "recommendations": [
            "Start with gradual pruning for minimal accuracy loss",
            "Consider structured pruning for hardware acceleration compatibility",
            "Apply magnitude pruning to remove the smallest weights"
        ]
    },
    "Knowledge Distillation": {
        "patterns": [r"distill", r"teacher.*student", r"knowledge.*transfer"],
        "frameworks": {
            "PyTorch": ["torch.nn.KLDivLoss"],
            "TensorFlow": ["tf.keras.losses.KLDivergence"],
            "Hugging Face": ["transformers.DistillationTrainer"]
        },
        "recommendations": [
            "Use a larger teacher model with your smaller student model",
            "Match intermediate representations for deeper knowledge transfer",
            "Combine hard and soft targets for better performance"
        ]
    },
    "Low-Rank Factorization": {
        "patterns": [r"svd", r"factorization", r"decomposition"],
        "frameworks": {
            "PyTorch": ["torch.svd"],
            "TensorFlow": ["tf.linalg.svd"],
            "NumPy": ["np.linalg.svd"]
        },
        "recommendations": [
            "Apply to fully connected layers with many parameters",
            "Consider Tucker decomposition for convolutional layers",
            "Use SVD to find optimal low-rank approximation"
        ]
    }
}

# Expanded ML algorithm patterns database
ML_ALGORITHM_PATTERNS = {
    "Linear Regression": {
        "patterns": [r"LinearRegression", r"np\.linalg\.lst", r"gradient descent", r"normal equation"],
        "std_complexity": "O(n^3) for normal eq, O(n*m) per iteration for GD",
        "optimizations": ["Use mini-batch gradient descent or normal eq on smaller data, L-BFGS for large dims"]
    },
    # ... [existing patterns kept] ...
    "Transformer": {
        "patterns": [r"Transformer", r"attention", r"multi.?head", r"encoder", r"decoder"],
        "std_complexity": "O(n*e*s^2*h) where s=sequence length, h=hidden size",
        "optimizations": ["Use efficient attention (linear, sparse), gradient checkpointing"]
    },
    # New patterns for modern architectures
    "Vision Transformer": {
        "patterns": [r"ViT", r"vision.?transformer", r"patch.?embed"],
        "std_complexity": "O(n*p^2*h) where p=patches, h=hidden size",
        "optimizations": ["Use hierarchical design, token pruning, efficient attention variants"]
    },
    "Graph Neural Network": {
        "patterns": [r"GNN", r"GraphConv", r"MessagePassing", r"graph.?neural", r"node_features"],
        "std_complexity": "O(n*e*f) where e=edges, f=features",
        "optimizations": ["Use neighbor sampling, cluster-GCN for large graphs"]
    },
    "Diffusion Model": {
        "patterns": [r"diffusion", r"DDPM", r"DDIM", r"score.?based", r"noise.?prediction"],
        "std_complexity": "O(n*t*s) where t=timesteps, s=sample size",
        "optimizations": ["Use fewer sampling steps, classifier-free guidance tuning"]
    },
    "Reinforcement Learning": {
        "patterns": [r"PPO", r"DQN", r"A2C", r"DDPG", r"reinforcement", r"reward", r"action.*space"],
        "std_complexity": "Varies by algorithm, typically O(n*a*s) where a=actions, s=states",
        "optimizations": ["Use parallel environments, prioritized experience replay"]
    }
}

# Framework-specific patterns and optimizations
FRAMEWORK_OPTIMIZATIONS = {
    "PyTorch": {
        "patterns": [r"import torch", r"nn\.Module", r"torch\."],
        "inefficiencies": [
            {"pattern": r"for.*in.*dataloader", "suggestion": "Use DataParallel or DistributedDataParallel"},
            {"pattern": r"\.cuda\(\)", "suggestion": "Use device-agnostic code with .to(device)"},
            {"pattern": r"\.item\(\).*for loop", "suggestion": "Avoid .item() in training loops, batch operations"}
        ],
        "optimizations": [
            "Use torch.compile() for PyTorch 2.0+ performance gains",
            "Enable AMP (Automatic Mixed Precision) with torch.cuda.amp",
            "Use torch.jit.script for TorchScript compilation",
            "Consider using torch.fx for graph transformations"
        ]
    },
    "TensorFlow": {
        "patterns": [r"import tensorflow", r"import tf", r"tf\."],
        "inefficiencies": [
            {"pattern": r"Session\(", "suggestion": "Use tf.function for TF2.x performance"},
            {"pattern": r"for.*in.*dataset", "suggestion": "Ensure dataset is properly prefetched/cached"},
            {"pattern": r"placeholder", "suggestion": "Use TF2.x eager execution with tf.function"}
        ],
        "optimizations": [
            "Use tf.function to create graph-compiled functions",
            "Enable mixed precision with tf.keras.mixed_precision",
            "Use TF-XLA compilation for accelerated linear algebra",
            "Consider TFLite for deployment optimization"
        ]
    },
    "JAX": {
        "patterns": [r"import jax", r"jnp\.", r"jax\."],
        "inefficiencies": [
            {"pattern": r"for.*in.*range", "suggestion": "Use jax.vmap for vectorization"},
            {"pattern": r"jnp\..*inside loop", "suggestion": "Move computations outside loops with jax.vmap or lax.scan"}
        ],
        "optimizations": [
            "Use jit compilation with jax.jit",
            "Apply function transformations with jax.vmap and jax.pmap",
            "Use lax.scan for loop optimization",
            "Consider checkpointing with jax.checkpoint for memory efficiency"
        ]
    }
}

# Define profiling constants
MODEL_COMPLEXITY_METRICS = {
    "FLOPs": "Floating Point Operations",
    "MACs": "Multiply-Accumulate Operations",
    "Parameters": "Total trainable parameters",
    "Memory": "Peak memory usage (estimated)",
    "Latency": "Inference time per sample (estimated)"
}

# Layer complexity mapping for common operations
LAYER_COMPLEXITY = {
    "Linear/Dense": {
        "flops": lambda in_features, out_features, batch_size=1: 2 * batch_size * in_features * out_features,
        "params": lambda in_features, out_features: in_features * out_features + out_features,
        "memory": lambda in_features, out_features, batch_size=1, dtype_bytes=4: batch_size * (in_features + out_features) * dtype_bytes
    },
    "Conv2D": {
        "flops": lambda batch_size, in_channels, out_channels, kernel_h, kernel_w, output_h, output_w: 
            batch_size * out_channels * in_channels * kernel_h * kernel_w * output_h * output_w,
        "params": lambda in_channels, out_channels, kernel_h, kernel_w, bias=True: 
            in_channels * out_channels * kernel_h * kernel_w + (out_channels if bias else 0),
        "memory": lambda batch_size, in_channels, h, w, out_channels, out_h, out_w, dtype_bytes=4: 
            batch_size * dtype_bytes * (in_channels * h * w + out_channels * out_h * out_w)
    },
    "Attention": {
        "flops": lambda batch_size, seq_len, embed_dim, heads: 
            batch_size * seq_len * (4 * embed_dim * embed_dim + seq_len * embed_dim),
        "params": lambda embed_dim, heads: 4 * embed_dim * embed_dim,
        "memory": lambda batch_size, seq_len, embed_dim, dtype_bytes=4: 
            batch_size * seq_len * embed_dim * 4 * dtype_bytes
    }
}

# Framework-specific profilers configuration
FRAMEWORK_PROFILERS = {
    "PyTorch": {
        "profiler_import": "torch.profiler",
        "profiler_setup": "with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:",
        "memory_tracking": "torch.cuda.memory_summary()",
        "profiling_apis": ["torch.profiler", "torch.autograd.profiler"],
        "implementation": [
            "from torch.profiler import profile, record_function, ProfilerActivity",
            "with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:",
            "    with record_function('model_inference'):",
            "        output = model(input_data)",
            "print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))"
        ]
    },
    "TensorFlow": {
        "profiler_import": "tf.profiler",
        "profiler_setup": "tf.profiler.experimental.start('logdir')",
        "memory_tracking": "tf.config.experimental.get_memory_info('GPU:0')",
        "profiling_apis": ["tf.profiler", "tf.summary.trace_on"],
        "implementation": [
            "import tensorflow as tf",
            "tf.profiler.experimental.start('logdir')",
            "# Model execution",
            "tf.profiler.experimental.stop()",
            "# For detailed analysis use TensorBoard: tensorboard --logdir=logdir"
        ]
    },
    "JAX": {
        "profiler_import": "jax.profiler",
        "profiler_setup": "with jax.profiler.trace('logdir'):",
        "memory_tracking": "jax.profiler.device_memory_profile()",
        "profiling_apis": ["jax.profiler", "jax.xla_computation"],
        "implementation": [
            "from jax import profiler",
            "with profiler.trace('logdir'):",
            "    # Model execution",
            "# For detailed analysis: tensorboard --logdir=logdir"
        ]
    }
}

# Hardware specific profiling metrics
HARDWARE_PROFILING_METRICS = {
    "CPU": {
        "metrics": ["core_utilization", "memory_bandwidth", "cache_hits", "vectorization_efficiency"],
        "tools": ["Intel VTune", "perf", "AMD uProf"],
        "env_vars": {
            "OMP_NUM_THREADS": "Set to control thread count",
            "MKL_NUM_THREADS": "For Intel MKL optimization"
        }
    },
    "GPU": {
        "metrics": ["sm_utilization", "memory_bandwidth", "occupancy", "tensor_core_utilization"],
        "tools": ["NVIDIA Nsight", "CUDA Profiler", "PyTorch Profiler"],
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "Control GPU visibility",
            "TF_FORCE_GPU_ALLOW_GROWTH": "For TensorFlow memory growth"
        }
    },
    "TPU": {
        "metrics": ["mxu_utilization", "memory_bandwidth", "flops_utilization"],
        "tools": ["Cloud TPU Profiler", "TensorBoard"],
        "env_vars": {
            "TPU_CHIPS_PER_HOST": "TPU configuration",
            "TPU_HOST_BOUNDS": "TPU pod configuration"
        }
    }
}

# New model architecture patterns with detailed profiling capabilities
MODEL_ARCHITECTURE_PATTERNS = {
    "Convolutional Network": {
        "patterns": [r"Conv[2-3]D", r"MaxPool", r"convolutional", r"CNN"],
        "estimations": {
            "inference_latency": {
                "CPU": "O(batch_size * channels * height * width * kernel_size^2)",
                "GPU": "O(batch_size * channels * height * width * kernel_size^2 / cores)"
            },
            "memory_usage": "O(batch_size * sum(feature_map_sizes) * precision)",
            "profiling_focus": [
                "Convolution operation efficiency",
                "Memory transfer bottlenecks",
                "Layer fusion opportunities"
            ]
        }
    },
    "Transformer": {
        "patterns": [r"Transformer", r"attention", r"multi.?head", r"encoder", r"decoder"],
        "estimations": {
            "inference_latency": {
                "CPU": "O(batch_size * seq_length^2 * embed_dim)",
                "GPU": "O(batch_size * seq_length^2 * embed_dim / cores)"
            },
            "memory_usage": "O(batch_size * seq_length * embed_dim * layers)",
            "profiling_focus": [
                "Attention mechanism efficiency",
                "KV cache utilization",
                "Parallel attention computation",
                "Kernel fusion opportunities"
            ]
        }
    },
    "Diffusion Model": {
        "patterns": [r"diffusion", r"DDPM", r"DDIM", r"score.?based", r"noise.?prediction"],
        "estimations": {
            "inference_latency": {
                "CPU": "O(batch_size * steps * image_resolution^2 * channels)",
                "GPU": "O(batch_size * steps * image_resolution^2 * channels / cores)"
            },
            "memory_usage": "O(batch_size * image_resolution^2 * channels * precision)",
            "profiling_focus": [
                "Diffusion step efficiency", 
                "U-Net backbone optimization",
                "Parallel sampling strategies",
                "Scheduler optimization"
            ]
        }
    }
}

# Add memory estimations for common datatypes
DTYPE_MEMORY = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8
}

class ModelProfilingLevel(Enum):
    """Enumeration for model profiling detail levels."""
    BASIC = 1        # Simple analysis with high-level metrics
    INTERMEDIATE = 2 # More detailed with layer-by-layer breakdown
    COMPREHENSIVE = 3 # Full analysis with timeline and bottleneck detection
    HARDWARE_AWARE = 4 # Includes hardware-specific optimizations

class ModelProfiler:
    """
    Advanced model profiler for detailed performance analysis and optimization.
    """
    
    def __init__(self, code_str: str, level: ModelProfilingLevel = ModelProfilingLevel.INTERMEDIATE):
        """
        Initialize the model profiler.
        
        Args:
            code_str: Python code as a string containing model definitions
            level: Profiling detail level
        """
        self.code_str = code_str
        self.level = level
        self.detected_frameworks = []
        self.detected_architectures = []
        self.memory_estimation = {}
        self.complexity_estimation = {}
        self.bottlenecks = []
        
        try:
            self.tree = ast.parse(code_str)
            # Detect frameworks and model architectures
            self.detected_frameworks = self._detect_frameworks()
            self.detected_architectures = self._detect_model_architectures()
            
            # Analyze model structure
            self.model_structure = self._analyze_model_structure()
            
            # Perform complexity estimations based on level
            if level.value >= ModelProfilingLevel.INTERMEDIATE.value:
                self.complexity_estimation = self._estimate_model_complexity()
            
            # Find potential bottlenecks
            if level.value >= ModelProfilingLevel.COMPREHENSIVE.value:
                self.bottlenecks = self._identify_bottlenecks()
                
        except SyntaxError as e:
            logger.error(f"Syntax error in code during profiling: {e}")
            self.tree = None
        except Exception as e:
            logger.error(f"Error in model profiling: {e}")
            traceback.print_exc()
    
    def _detect_frameworks(self) -> List[str]:
        """
        Detect ML frameworks used in the code.
        
        Returns:
            List of detected frameworks
        """
        frameworks = []
        # Check for PyTorch
        if re.search(r"import torch|from torch|import pytorch", self.code_str, re.IGNORECASE):
            frameworks.append("PyTorch")
        
        # Check for TensorFlow/Keras
        if re.search(r"import tensorflow|import tf|from tensorflow|import keras|from keras", 
                    self.code_str, re.IGNORECASE):
            frameworks.append("TensorFlow")
        
        # Check for JAX
        if re.search(r"import jax|from jax", self.code_str, re.IGNORECASE):
            frameworks.append("JAX")
            
        # Check for ONNX
        if re.search(r"import onnx|from onnx", self.code_str, re.IGNORECASE):
            frameworks.append("ONNX")
            
        return frameworks
    
    def _detect_model_architectures(self) -> List[Dict[str, Any]]:
        """
        Detect model architectures used in the code.
        
        Returns:
            List of detected architectures with details
        """
        architectures = []
        
        for arch_name, arch_info in MODEL_ARCHITECTURE_PATTERNS.items():
            matched_patterns = [pat for pat in arch_info["patterns"] 
                              if re.search(pat, self.code_str, re.IGNORECASE)]
            if matched_patterns:
                architectures.append({
                    "architecture": arch_name,
                    "matched_patterns": matched_patterns,
                    "estimations": arch_info["estimations"],
                    "profiling_focus": arch_info["estimations"]["profiling_focus"]
                })
                
        return architectures
    
    def _analyze_model_structure(self) -> Dict[str, Any]:
        """
        Extract and analyze model structure from the code.
        
        Returns:
            Dictionary with model structure information
        """
        if not self.tree:
            return {}
            
        structure = {
            "layers": [],
            "has_sequential": False,
            "has_functional": False,
            "has_subclassing": False,
            "params_count_estimated": 0,
            "input_shapes_detected": []
        }
        
        # Find model definition patterns
        visitor = ModelStructureVisitor()
        visitor.visit(self.tree)
        
        # Extract layers from the visitor
        structure["layers"] = visitor.layers
        structure["has_sequential"] = visitor.has_sequential_model
        structure["has_functional"] = visitor.has_functional_api
        structure["has_subclassing"] = visitor.has_model_subclass
        structure["params_count_estimated"] = self._estimate_parameters(visitor.layers)
        structure["input_shapes_detected"] = visitor.input_shapes
        
        return structure
    
    def _estimate_parameters(self, layers: List[Dict]) -> int:
        """
        Estimate the parameter count for the model based on detected layers.
        
        Args:
            layers: List of layer dictionaries with type and config info
            
        Returns:
            Estimated parameter count
        """
        total_params = 0
        
        for layer in layers:
            layer_type = layer.get("type", "")
            config = layer.get("config", {})
            
            # Handle common layer types
            if "Linear" in layer_type or "Dense" in layer_type:
                in_features = config.get("in_features") or config.get("input_dim", 0)
                out_features = config.get("out_features") or config.get("units", 0)
                
                if in_features and out_features:
                    # Parameters: weights + biases
                    params = in_features * out_features
                    if config.get("bias", True) or "use_bias" not in config or config.get("use_bias", True):
                        params += out_features
                    total_params += params
                    
            elif "Conv" in layer_type:
                in_channels = config.get("in_channels") or config.get("input_channels") or config.get("filters_in", 0)
                out_channels = config.get("out_channels") or config.get("filters") or config.get("filters_out", 0)
                kernel_size = config.get("kernel_size", 0)
                
                # Handle tuple or single value kernel size
                if isinstance(kernel_size, (list, tuple)):
                    kernel_area = 1
                    for k in kernel_size:
                        kernel_area *= k
                else:
                    kernel_area = kernel_size ** 2
                
                if in_channels and out_channels and kernel_area:
                    # Parameters: weights + biases
                    params = in_channels * out_channels * kernel_area
                    if config.get("bias", True) or "use_bias" not in config or config.get("use_bias", True):
                        params += out_channels
                    total_params += params
        
        return total_params
    
    def _estimate_model_complexity(self) -> Dict[str, Any]:
        """
        Estimate model computational complexity.
        
        Returns:
            Dictionary with complexity estimations
        """
        complexity = {
            "flops_per_inference": 0,
            "memory_usage_bytes": 0,
            "estimated_inference_time": {
                "cpu": None,
                "gpu": None
            },
            "bottlenecks": []
        }
        
        # Calculate complexity based on model structure
        if self.model_structure and self.model_structure.get("layers"):
            # Iterate through layers to calculate complexity
            total_flops = 0
            total_memory = 0
            bottlenecks = []
            
            for layer in self.model_structure.get("layers", []):
                layer_type = layer.get("type", "")
                config = layer.get("config", {})
                
                # Calculate FLOPs and memory for each layer
                layer_flops = 0
                layer_memory = 0
                
                # Calculate based on layer type
                if "Linear" in layer_type or "Dense" in layer_type:
                    in_features = config.get("in_features") or config.get("input_dim", 0)
                    out_features = config.get("out_features") or config.get("units", 0)
                    batch_size = self._estimate_batch_size()
                    
                    if in_features and out_features:
                        # Use the complexity formulas from LAYER_COMPLEXITY
                        layer_flops = LAYER_COMPLEXITY["Linear/Dense"]["flops"](in_features, out_features, batch_size)
                        layer_params = LAYER_COMPLEXITY["Linear/Dense"]["params"](in_features, out_features)
                        layer_memory = LAYER_COMPLEXITY["Linear/Dense"]["memory"](in_features, out_features, batch_size)
                        
                elif "Conv2D" in layer_type:
                    # Extract or estimate the needed parameters
                    in_channels = config.get("in_channels", 0)
                    out_channels = config.get("out_channels", 0)
                    kernel_size = config.get("kernel_size", 0)
                    kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
                    
                    # Try to infer input/output dimensions
                    input_shape = self._infer_input_shape(layer)
                    if input_shape and len(input_shape) >= 4:
                        batch_size, _, input_h, input_w = input_shape
                        # Estimate output dimensions (simplified)
                        stride = config.get("stride", 1)
                        padding = config.get("padding", 0)
                        output_h = ((input_h + 2 * padding - kernel_h) // stride) + 1
                        output_w = ((input_w + 2 * padding - kernel_w) // stride) + 1
                        
                        layer_flops = LAYER_COMPLEXITY["Conv2D"]["flops"](
                            batch_size, in_channels, out_channels, kernel_h, kernel_w, output_h, output_w)
                        layer_params = LAYER_COMPLEXITY["Conv2D"]["params"](
                            in_channels, out_channels, kernel_h, kernel_w)
                        layer_memory = LAYER_COMPLEXITY["Conv2D"]["memory"](
                            batch_size, in_channels, input_h, input_w, out_channels, output_h, output_w)
                
                elif "Attention" in layer_type or "MultiHeadAttention" in layer_type:
                    # Extract or estimate attention parameters
                    embed_dim = config.get("embed_dim") or config.get("hidden_size") or config.get("d_model", 512)
                    num_heads = config.get("num_heads") or config.get("n_head", 8)
                    seq_len = self._estimate_sequence_length()
                    batch_size = self._estimate_batch_size()
                    
                    layer_flops = LAYER_COMPLEXITY["Attention"]["flops"](batch_size, seq_len, embed_dim, num_heads)
                    layer_params = LAYER_COMPLEXITY["Attention"]["params"](embed_dim, num_heads)
                    layer_memory = LAYER_COMPLEXITY["Attention"]["memory"](batch_size, seq_len, embed_dim)
                    
                    # Check if this could be a bottleneck
                    if seq_len > 512:
                        bottlenecks.append({
                            "layer": layer_type,
                            "issue": "Long sequence attention calculation",
                            "suggestion": "Consider efficient attention variants or sequence chunking"
                        })
                
                # Add to total complexity
                total_flops += layer_flops
                total_memory += layer_memory
                
                # Track layer complexity for bottleneck detection
                layer["complexity"] = {
                    "flops": layer_flops,
                    "memory_bytes": layer_memory
                }
                
                # Identify potential bottlenecks per layer
                if layer_flops > total_flops * 0.3:  # If layer takes >30% of compute
                    bottlenecks.append({
                        "layer": layer_type,
                        "issue": "High computational load",
                        "suggestion": f"Optimize this layer; it uses {layer_flops/total_flops:.1%} of FLOPs"
                    })
            
            # Update complexity metrics
            complexity["flops_per_inference"] = total_flops
            complexity["memory_usage_bytes"] = total_memory
            complexity["bottlenecks"] = bottlenecks
            
            # Estimate inference time based on FLOPs
            # These are rough estimates and should be adjusted based on actual hardware
            if total_flops > 0:
                # Assuming CPU speed of 10 GFLOPS and GPU speed of 10 TFLOPS (very rough)
                complexity["estimated_inference_time"]["cpu"] = total_flops / (10 * 10**9)  # seconds
                complexity["estimated_inference_time"]["gpu"] = total_flops / (10 * 10**12)  # seconds
        
        return complexity
    
    def _estimate_batch_size(self) -> int:
        """
        Estimate batch size from code context.
        
        Returns:
            Estimated batch size
        """
        # Look for batch_size variable in code
        batch_size_pattern = r"batch_size\s*=\s*(\d+)"
        match = re.search(batch_size_pattern, self.code_str)
        if match:
            return int(match.group(1))
        
        # Check for DataLoader with batch_size
        dataloader_pattern = r"DataLoader\(.*batch_size\s*=\s*(\d+)"
        match = re.search(dataloader_pattern, self.code_str)
        if match:
            return int(match.group(1))
            
        # Default value
        return 1
    
    def _estimate_sequence_length(self) -> int:
        """
        Estimate sequence length for transformer models.
        
        Returns:
            Estimated sequence length
        """
        # Look for seq_len, sequence_length, or max_length variables
        seq_len_pattern = r"(?:seq_len|sequence_length|max_length)\s*=\s*(\d+)"
        match = re.search(seq_len_pattern, self.code_str)
        if match:
            return int(match.group(1))
        
        # Check for common sequence lengths in the code
        if re.search(r"512|768|1024|2048", self.code_str):
            for length in [512, 768, 1024, 2048]:
                if re.search(rf"\b{length}\b", self.code_str):
                    return length
                    
        # Default value for transformers
        return 512
    
    def _infer_input_shape(self, layer: Dict) -> Optional[Tuple]:
        """
        Try to infer the input shape for a layer.
        
        Args:
            layer: Layer dictionary with configurations
            
        Returns:
            Tuple of input shape if detected, None otherwise
        """
        # First check if there's direct input_shape info
        if "input_shape" in layer:
            return layer["input_shape"]
            
        # Check the model structure for input shapes
        if self.model_structure.get("input_shapes_detected"):
            return self.model_structure["input_shapes_detected"][0]
            
        # Look for common patterns in the code
        # This is a simplified approach and would need to be expanded
        # based on the specific framework and model definition style
        
        # Default batch, channels, height, width for CNNs
        return (self._estimate_batch_size(), 3, 224, 224)  
    
    def _identify_bottlenecks(self) -> List[Dict[str, str]]:
        """
        Identify performance bottlenecks in the model.
        
        Returns:
            List of dictionaries with bottleneck information
        """
        bottlenecks = []
        
        # Include existing bottlenecks from complexity estimation
        if self.complexity_estimation and "bottlenecks" in self.complexity_estimation:
            bottlenecks.extend(self.complexity_estimation["bottlenecks"])
        
        # Check for known inefficient patterns
        for framework in self.detected_frameworks:
            # Check framework-specific inefficiencies
            if framework in FRAMEWORK_OPTIMIZATIONS:
                for issue in FRAMEWORK_OPTIMIZATIONS[framework]["inefficiencies"]:
                    if re.search(issue["pattern"], self.code_str, re.IGNORECASE):
                        bottlenecks.append({
                            "type": "framework_inefficiency",
                            "framework": framework,
                            "issue": issue["pattern"],
                            "suggestion": issue["suggestion"]
                        })
        
        # Check for architecture-specific bottlenecks
        for arch in self.detected_architectures:
            arch_name = arch["architecture"]
            if arch_name == "Transformer" and self._estimate_sequence_length() > 1024:
                bottlenecks.append({
                    "type": "architecture_limitation",
                    "architecture": arch_name,
                    "issue": "Long sequence length in Transformer",
                    "suggestion": "Consider sparse attention, linear attention, or sequence pruning"
                })
            elif arch_name == "Convolutional Network" and re.search(r"kernel_size=\(7,\s*7\)", self.code_str):
                bottlenecks.append({
                    "type": "architecture_inefficiency",
                    "architecture": arch_name,
                    "issue": "Large kernel convolution",
                    "suggestion": "Consider replacing 7x7 convs with stacked 3x3 convs for efficiency"
                })
        
        # Memory-related bottlenecks
        if re.search(r"cuda\.empty_cache\(\)|gc\.collect\(", self.code_str, re.IGNORECASE):
            bottlenecks.append({
                "type": "memory_management",
                "issue": "Manual memory management suggests memory pressure",
                "suggestion": "Consider gradient checkpointing, model sharding, or mixed precision"
            })
            
        return bottlenecks
    
    def get_profiling_recommendations(self) -> Dict[str, Any]:
        """
        Generate profiling recommendations based on detected frameworks and architecture.
        
        Returns:
            Dictionary with profiling recommendations
        """
        recommendations = {
            "general": [
                "Start with end-to-end profiling to identify macro bottlenecks",
                "Focus on the most time-consuming operations first",
                "Measure both training and inference performance separately"
            ],
            "framework_specific": [],
            "architecture_specific": [],
            "hardware_specific": []
        }
        
        # Framework-specific recommendations
        for framework in self.detected_frameworks:
            if framework in FRAMEWORK_PROFILERS:
                profiler_info = FRAMEWORK_PROFILERS[framework]
                recommendations["framework_specific"].append({
                    "framework": framework,
                    "profiler": profiler_info["profiler_import"],
                    "setup": profiler_info["profiler_setup"],
                    "memory_tracking": profiler_info["memory_tracking"],
                    "implementation": profiler_info["implementation"]
                })
        
        # Architecture-specific recommendations
        for arch in self.detected_architectures:
            recommendations["architecture_specific"].append({
                "architecture": arch["architecture"],
                "focus_areas": arch["profiling_focus"]
            })
            
        # Hardware-specific recommendations
        for hardware in self._infer_hardware_usage():
            if hardware in HARDWARE_PROFILING_METRICS:
                metrics = HARDWARE_PROFILING_METRICS[hardware]
                recommendations["hardware_specific"].append({
                    "hardware": hardware,
                    "metrics_to_monitor": metrics["metrics"],
                    "recommended_tools": metrics["tools"],
                    "environment_variables": metrics["env_vars"]
                })
                
        return recommendations
    
    def _infer_hardware_usage(self) -> List[str]:
        """
        Infer hardware platforms used based on code patterns.
        
        Returns:
            List of inferred hardware platforms
        """
        platforms = []
        
        # Check for GPU usage
        if re.search(r"cuda|gpu|device|to\(['\"]cuda", self.code_str, re.IGNORECASE):
            platforms.append("GPU")
        
        # Check for TPU usage
        if re.search(r"tpu|xla|pjit", self.code_str, re.IGNORECASE):
            platforms.append("TPU")
            
        # Default to CPU if nothing else detected
        if not platforms:
            platforms.append("CPU")
            
        return platforms
    
    def estimate_memory_requirements(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Estimate memory requirements for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to estimate for
            
        Returns:
            Dictionary with memory requirement estimations
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            
        estimates = {
            "base_memory": 0,  # Fixed memory regardless of batch size
            "per_sample_memory": 0,  # Additional memory per sample
            "batch_estimates": {}
        }
        
        # Estimate base and per-sample memory
        if self.complexity_estimation and "memory_usage_bytes" in self.complexity_estimation:
            current_memory = self.complexity_estimation["memory_usage_bytes"]
            current_batch = self._estimate_batch_size()
            
            # Simple linear model for memory estimation
            if current_batch > 1:
                estimates["base_memory"] = current_memory * 0.2  # Assumption: 20% is fixed overhead
                estimates["per_sample_memory"] = (current_memory - estimates["base_memory"]) / current_batch
            else:
                estimates["base_memory"] = current_memory * 0.5  # Higher ratio for batch_size=1
                estimates["per_sample_memory"] = current_memory * 0.5
                
            # Calculate for different batch sizes
            for batch_size in batch_sizes:
                estimates["batch_estimates"][batch_size] = {
                    "estimated_memory_bytes": estimates["base_memory"] + (batch_size * estimates["per_sample_memory"]),
                    "estimated_memory_human": self._format_bytes(estimates["base_memory"] + (batch_size * estimates["per_sample_memory"]))
                }
                
        return estimates
        
    def _format_bytes(self, size_bytes: float) -> str:
        """
        Format bytes into human-readable string.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable string
        """
        if size_bytes < 1024:
            return f"{size_bytes:.2f} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.2f} MB"
        elif size_bytes < 1024**4:
            return f"{size_bytes/1024**3:.2f} GB"
        else:
            return f"{size_bytes/1024**4:.2f} TB"
            
    def generate_profiling_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive profiling report.
        
        Returns:
            Dictionary with the complete profiling report
        """
        memory_estimates = self.estimate_memory_requirements()
        profiling_recommendations = self.get_profiling_recommendations()
        
        report = {
            "model_summary": {
                "detected_frameworks": self.detected_frameworks,
                "detected_architectures": [arch["architecture"] for arch in self.detected_architectures],
                "estimated_parameters": self.model_structure.get("params_count_estimated", 0),
                "model_type": self._infer_model_type()
            },
            "complexity_analysis": self.complexity_estimation,
            "memory_analysis": memory_estimates,
            "bottlenecks": self.bottlenecks,
            "profiling_recommendations": profiling_recommendations,
            "optimization_opportunities": self._identify_optimization_opportunities()
        }
        
        return report
    
    def _infer_model_type(self) -> str:
        """
        Infer the general type of model.
        
        Returns:
            String describing the model type
        """
        # Check architectures first
        for arch in self.detected_architectures:
            return arch["architecture"]
            
        # Check common model types from code patterns
        if re.search(r"classifier|classification", self.code_str, re.IGNORECASE):
            return "Classification Model"
        elif re.search(r"detection|detector|yolo|rcnn", self.code_str, re.IGNORECASE):
            return "Object Detection Model"
        elif re.search(r"segment|mask|unet", self.code_str, re.IGNORECASE):
            return "Segmentation Model"
        elif re.search(r"generate|generation|gpt|llm|language", self.code_str, re.IGNORECASE):
            return "Language Model"
        elif re.search(r"gan|generator|discriminator", self.code_str, re.IGNORECASE):
            return "Generative Adversarial Network"
        
        # Default
        return "Neural Network (type unspecified)"
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, str]]:
        """
        Identify optimization opportunities based on profiling.
        
        Returns:
            List of dictionaries with optimization suggestions
        """
        opportunities = []
        
        # Framework-specific optimizations
        for framework in self.detected_frameworks:
            if framework in FRAMEWORK_OPTIMIZATIONS:
                for opt in FRAMEWORK_OPTIMIZATIONS[framework]["optimizations"]:
                    opportunities.append({
                        "type": "framework_optimization",
                        "framework": framework,
                        "suggestion": opt
                    })
        
        # Hardware-specific optimizations
        for hardware in self._infer_hardware_usage():
            if hardware in HARDWARE_PLATFORMS:
                for opt in HARDWARE_PLATFORMS[hardware]["optimizations"]:
                    opportunities.append({
                        "type": "hardware_optimization",
                        "hardware": hardware,
                        "suggestion": opt
                    })
        
        # Architecture-specific optimizations
        for arch in self.detected_architectures:
            if "profiling_focus" in arch:
                for focus in arch["profiling_focus"]:
                    opportunities.append({
                        "type": "architecture_optimization",
                        "architecture": arch["architecture"],
                        "focus_area": focus
                    })
        
        # Memory optimization if large model detected
        if self.model_size > LARGE_MODEL_THRESHOLD:
            suggestions.append({
                "type": "memory_optimization",
                "architecture": arch["architecture"],
                "focus_area": focus
            })
        
class MLAlgorithmRecognizer:
    """
    Recognizes machine learning algorithms in code based on pattern matching.
    """

    def __init__(self, code_str: str):
        """
        Initialize the algorithm recognizer with code to analyze.

        Args:
            code_str: Python code as a string
        """
        self.code_str = code_str
        try:
            self.tree = ast.parse(code_str)
            self.framework_used = self._detect_framework()
            self.hardware_inferences = self._infer_hardware_usage()
            self.cloud_provider = self._infer_cloud_provider()
            
            # Run AST visitors
            self.pattern_visitor = AlgorithmPatternVisitor({
                algo: info["patterns"] for algo, info in ML_ALGORITHM_PATTERNS.items()
            })
            if self.tree:
                self.pattern_visitor.visit(self.tree)
            
            self.model_visitor = ModelStructureVisitor()
            if self.tree:
                self.model_visitor.visit(self.tree)
                
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            self.tree = None
            self.framework_used = None
            self.hardware_inferences = []
            self.cloud_provider = None
            self.pattern_visitor = None
            self.model_visitor = None
    
    def _detect_framework(self) -> List[str]:
        """
        Detect ML frameworks used in the code.
        
        Returns:
            List of detected frameworks
        """
        frameworks = []
        for framework, info in FRAMEWORK_OPTIMIZATIONS.items():
            if any(re.search(pattern, self.code_str, re.IGNORECASE) for pattern in info["patterns"]):
                frameworks.append(framework)
        return frameworks
    
    def _infer_hardware_usage(self) -> List[str]:
        """
        Infer hardware platforms used based on code patterns.
        
        Returns:
            List of inferred hardware platforms
        """
        platforms = []
        
        # Check for GPU usage
        if re.search(r"cuda|gpu|device|to\(['\"]cuda", self.code_str, re.IGNORECASE):
            platforms.append("GPU")
        
        # Check for TPU usage
        if re.search(r"tpu|xla|pjit", self.code_str, re.IGNORECASE):
            platforms.append("TPU")
            
        # Check for edge optimization
        if re.search(r"quantize|tflite|coreml|onnx|mobile|edge", self.code_str, re.IGNORECASE):
            platforms.append("Edge")
            
        # Default to CPU if nothing else detected
        if not platforms:
            platforms.append("CPU")
            
        return platforms
    
    def _infer_cloud_provider(self) -> Optional[str]:
        """
        Infer cloud provider based on code patterns.
        
        Returns:
            Inferred cloud provider or None
        """
        if re.search(r"aws|amazon|ec2|s3|sagemaker", self.code_str, re.IGNORECASE):
            return "AWS"
        elif re.search(r"gcp|google cloud|gcs|vertex|gke", self.code_str, re.IGNORECASE):
            return "GCP"
        elif re.search(r"azure|microsoft|blob", self.code_str, re.IGNORECASE):
            return "Azure"
        return None

    def identify_algorithms(self) -> List[Dict[str, Any]]:
        """
        Identify ML algorithms in the code with confidence scores and explanations.

        Returns:
            List of dictionaries with algorithm information including confidence and explanations
        """
        found_algos = []
        
        # If we have AST visitor results, use them for more accurate detection
        if self.pattern_visitor and hasattr(self.pattern_visitor, 'matches'):
            # Calculate confidence based on number and quality of matches
            for algo_name, matches in self.pattern_visitor.matches.items():
                if algo_name in ML_ALGORITHM_PATTERNS:
                    info = ML_ALGORITHM_PATTERNS[algo_name]
                    total_patterns = len(info["patterns"])
                    matched_count = len(set(match[0] for match in matches))  # Count unique pattern matches
                    
                    # Calculate confidence score (0.0-1.0)
                    # Base confidence on pattern match ratio and presence of key imports/patterns
                    base_confidence = min(0.7, matched_count / total_patterns * 0.7)
                    
                    # Bonus for multiple distinct matches
                    match_bonus = min(0.2, 0.05 * (len(matches) - matched_count))
                    
                    # Bonus for framework alignment
                    framework_bonus = 0.0
                    framework_match = self._check_framework_alignment(algo_name, self.framework_used)
                    if framework_match:
                        framework_bonus = 0.1
                    
                    confidence = min(1.0, base_confidence + match_bonus + framework_bonus)
                    
                    # Generate explanations for each matched pattern
                    explanations = []
                    for pattern, match_text in matches:
                        if algo_name in ALGORITHM_EXPLANATION_PATTERNS and pattern in ALGORITHM_EXPLANATION_PATTERNS[algo_name]:
                            explanation = ALGORITHM_EXPLANATION_PATTERNS[algo_name][pattern]
                            explanations.append({
                                "matched_text": match_text,
                                "explanation": explanation
                            })
                        else:
                            explanations.append({
                                "matched_text": match_text,
                                "explanation": f"Matched pattern: {pattern}"
                            })
                    
                    found_algos.append({
                        "algorithm": algo_name,
                        "matched_patterns": [m[0] for m in matches],
                        "std_complexity": info["std_complexity"],
                        "optimizations": info["optimizations"],
                        "confidence": round(confidence * 100),  # Convert to percentage
                        "explanations": explanations
                    })
        else:
            # Fall back to simple regex-based detection
            for algo_name, info in ML_ALGORITHM_PATTERNS.items():
                matched_patterns = [pat for pat in info["patterns"] if re.search(pat, self.code_str, flags=re.IGNORECASE)]
                if matched_patterns:
                    # Simple confidence calculation for regex-only approach
                    confidence = min(80, 40 + 10 * len(matched_patterns))  # Base 40% + 10% per match, max 80%
                    
                    explanations = []
                    for pattern in matched_patterns:
                        if algo_name in ALGORITHM_EXPLANATION_PATTERNS and pattern in ALGORITHM_EXPLANATION_PATTERNS[algo_name]:
                            explanation = ALGORITHM_EXPLANATION_PATTERNS[algo_name][pattern]
                            explanations.append({
                                "matched_text": f"Pattern: {pattern}",
                                "explanation": explanation
                            })
                
                    found_algos.append({
                        "algorithm": algo_name,
                        "matched_patterns": matched_patterns,
                        "std_complexity": info["std_complexity"],
                        "optimizations": info["optimizations"],
                        "confidence": confidence,
                        "explanations": explanations
                    })

        return found_algos
    
    def _check_framework_alignment(self, algo_name: str, frameworks: List[str]) -> bool:
        """
        Check if the algorithm aligns with detected frameworks.
        
        Args:
            algo_name: Name of the algorithm
            frameworks: List of detected frameworks
            
        Returns:
            True if there's framework alignment, False otherwise
        """
        if not frameworks:
            return False
            
        # Define framework-algorithm alignment pairs
        framework_algo_alignment = {
            "PyTorch": ["Transformer", "Vision Transformer", "Graph Neural Network", "Diffusion Model"],
            "TensorFlow": ["Transformer", "Vision Transformer", "Diffusion Model"],
            "JAX": ["Transformer", "Diffusion Model"]
        }
        
        for framework in frameworks:
            if framework in framework_algo_alignment and algo_name in framework_algo_alignment[framework]:
                return True
                
        return False
    
    def suggest_model_variants(self) -> List[Dict[str, Any]]:
        """
        Suggest model variants optimized for edge or cloud deployment.
        
        Returns:
            List of dictionaries with model variant suggestions
        """
        suggestions = []
        
        # First determine deployment target
        is_edge = "Edge" in self.hardware_inferences
        deployment_target = "edge" if is_edge else "cloud"
        
        # Get detected algorithms and architectures
        detected_algorithms = self.identify_algorithms()
        detected_architectures = [algo["algorithm"] for algo in detected_algorithms]
        
        # Add suggestions based on detected architectures
        for arch_name, variants in MODEL_ARCHITECTURE_VARIANTS.items():
            # Check if this architecture was detected or is similar to detected ones
            is_related = False
            for detected in detected_architectures:
                similarity = self._calculate_architecture_similarity(detected, arch_name)
                if similarity > 0.5:  # Threshold for considering architectures related
                    is_related = True
                    break
            
            if arch_name in detected_architectures or is_related:
                target_variants = variants[deployment_target]
                opposite_variants = variants["cloud" if is_edge else "edge"]
                
                suggestions.append({
                    "architecture": arch_name,
                    "deployment_target": deployment_target,
                    "recommended_variants": target_variants,
                    "alternative_deployment": opposite_variants,
                    "explanation": (
                        f"For {deployment_target} deployment of {arch_name}, "
                        f"consider using {', '.join(v['name'] for v in target_variants)}"
                    )
                })
        
        # If no specific architecture detected, provide general suggestions
        if not suggestions:
            if is_edge:
                suggestions.append({
                    "architecture": "General",
                    "deployment_target": "edge",
                    "recommended_variants": [
                        {"name": "MobileNetV3", "description": "Lightweight CNN for mobile devices"},
                        {"name": "EfficientNet-Lite", "description": "Edge-optimized EfficientNet variant"},
                        {"name": "DistilBERT", "description": "Compressed BERT for edge deployment"}
                    ],
                    "explanation": "For edge deployment, consider lightweight architectures optimized for mobile/edge devices"
                })
            else:
                suggestions.append({
                    "architecture": "General",
                    "deployment_target": "cloud",
                    "recommended_variants": [
                        {"name": "ResNet", "description": "Deep residual network with strong performance"},
                        {"name": "BERT-Large", "description": "Full-sized BERT model for maximum accuracy"},
                        {"name": "EfficientNet-B7", "description": "Larger, high-performance EfficientNet variant"}
                    ],
                    "explanation": "For cloud deployment, leverage larger architectures that prioritize accuracy over efficiency"
                })
        
        return suggestions
    
    def _calculate_architecture_similarity(self, arch1: str, arch2: str) -> float:
        """
        Calculate similarity between two architecture names.
        
        Args:
            arch1: First architecture name
            arch2: Second architecture name
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Define architecture groups
        architecture_groups = {
            "vision": ["Convolutional Network", "Vision Transformer", "CNN"],
            "language": ["Transformer", "Language Model", "BERT", "GPT"],
            "graph": ["Graph Neural Network", "GNN"],
            "generative": ["Diffusion Model", "GAN", "VAE"]
        }
        
        # Check if architectures are in the same group
        for group, archs in architecture_groups.items():
            arch1_in_group = any(a.lower() in arch1.lower() for a in archs)
            arch2_in_group = any(a.lower() in arch2.lower() for a in archs)
            if arch1_in_group and arch2_in_group:
                return 0.8  # High similarity if in same group
        
        # Check for shared keywords
        keywords1 = set(re.findall(r'\w+', arch1.lower()))
        keywords2 = set(re.findall(r'\w+', arch2.lower()))
        shared = keywords1.intersection(keywords2)
        
        if shared:
            return len(shared) / max(len(keywords1), len(keywords2))
        
        return 0.0
    
    def identify_framework_inefficiencies(self) -> List[Dict[str, Any]]:
        """
        Identify framework-specific inefficiencies in the code.
        
        Returns:
            List of dictionaries with inefficiency information
        """
        inefficiencies = []
        
        for framework in self.framework_used or []:
            if framework in FRAMEWORK_OPTIMIZATIONS:
                for issue in FRAMEWORK_OPTIMIZATIONS[framework]["inefficiencies"]:
                    if re.search(issue["pattern"], self.code_str, re.IGNORECASE):
                        inefficiencies.append({
                            "framework": framework,
                            "pattern_found": issue["pattern"],
                            "suggestion": issue["suggestion"]
                        })
        
        return inefficiencies
    
    def identify_compression_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify opportunities for model compression.
        
        Returns:
            List of dictionaries with compression opportunities
        """
        opportunities = []
        
        # First check if compression is already applied
        applied_techniques = []
        for technique, info in MODEL_COMPRESSION_TECHNIQUES.items():
            if any(re.search(pattern, self.code_str, re.IGNORECASE) for pattern in info["patterns"]):
                applied_techniques.append(technique)
        
        # Now recommend techniques not already applied
        all_techniques = set(MODEL_COMPRESSION_TECHNIQUES.keys())
        missing_techniques = all_techniques - set(applied_techniques)
        
        # Framework-specific recommendations
        for framework in self.framework_used or []:
            for technique in missing_techniques:
                tech_info = MODEL_COMPRESSION_TECHNIQUES[technique]
                if framework in tech_info["frameworks"]:
                    opportunities.append({
                        "technique": technique,
                        "framework": framework,
                        "implementation": tech_info["frameworks"][framework],
                        "recommendations": tech_info["recommendations"]
                    })
        
        # If no specific framework detected, give general recommendations
        if not self.framework_used:
            for technique in missing_techniques:
                opportunities.append({
                    "technique": technique,
                    "framework": "any",
                    "recommendations": MODEL_COMPRESSION_TECHNIQUES[technique]["recommendations"]
                })
        
        return opportunities
    
    def get_cloud_cost_optimizations(self) -> Dict[str, Any]:
        """
        Get cloud-specific cost optimizations.
        
        Returns:
            Dictionary with cloud cost optimization recommendations
        """
        if not self.cloud_provider:
            # Provide general recommendations
            return {
                "provider": "general",
                "recommendations": [
                    "Consider spot/preemptible instances for training",
                    "Set up auto-scaling for inference endpoints",
                    "Monitor and terminate unused resources",
                    "Use managed ML services for automatic resource optimization"
                ]
            }
        
        return {
            "provider": self.cloud_provider,
            "instance_types": CLOUD_COST_OPTIMIZATIONS[self.cloud_provider]["instance_types"],
            "cost_reduction": CLOUD_COST_OPTIMIZATIONS[self.cloud_provider]["cost_reduction"]
        }
    
    def get_hardware_optimizations(self) -> List[Dict[str, Any]]:
        """
        Get hardware-specific optimization recommendations.
        
        Returns:
            List of dictionaries with hardware optimization recommendations
        """
        optimizations = []
        
        for platform in self.hardware_inferences:
            if platform in HARDWARE_PLATFORMS:
                optimizations.append({
                    "platform": platform,
                    "optimizations": HARDWARE_PLATFORMS[platform]["optimizations"]
                })
        
        return optimizations

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions for identified algorithms.

        Returns:
            List of dictionaries with optimization suggestions
        """
        algorithms = self.identify_algorithms()
        suggestions = []

        for algo in algorithms:
            for opt in algo["optimizations"]:
                suggestions.append({
                    "algorithm": algo["algorithm"],
                    "suggestion": opt,
                    "complexity": algo["std_complexity"]
                })

        return suggestions

    def analyze_code(self) -> Dict[str, Any]:
        """
        Analyze code to identify ML algorithms and provide optimization suggestions.

        Returns:
            Dictionary with analysis results
        """
        try:
            algorithms = self.identify_algorithms()
            algorithm_suggestions = self.get_optimization_suggestions()
            framework_inefficiencies = self.identify_framework_inefficiencies()
            compression_opportunities = self.identify_compression_opportunities()
            cloud_cost_optimizations = self.get_cloud_cost_optimizations()
            hardware_optimizations = self.get_hardware_optimizations()
            model_variants = self.suggest_model_variants()
            
            return {
                "identified_algorithms": algorithms,
                "algorithm_optimization_suggestions": algorithm_suggestions,
                "framework_inefficiencies": framework_inefficiencies,
                "compression_opportunities": compression_opportunities,
                "cloud_cost_optimizations": cloud_cost_optimizations,
                "hardware_optimizations": hardware_optimizations,
                "model_variants": model_variants,
                "detected_frameworks": self.framework_used,
                "inferred_hardware": self.hardware_inferences,
                "inferred_cloud_provider": self.cloud_provider
            }
        except Exception as e:
            logger.error(f"Error during code analysis: {str(e)}")
            traceback.print_exc()
            # Return partial results if available
            return {
                "error": str(e),
                "identified_algorithms": self.identify_algorithms() if self.tree else [],
                "partial_results": True
            }

# Add model architecture variants for edge vs. cloud deployment
MODEL_ARCHITECTURE_VARIANTS = {
    "Convolutional Network": {
        "edge": [
            {"name": "MobileNetV3", "description": "Lightweight CNN designed for mobile devices"},
            {"name": "EfficientNet-Lite", "description": "Edge-optimized version of EfficientNet"},
            {"name": "MnasNet", "description": "Mobile Neural Architecture Search Network"}
        ],
        "cloud": [
            {"name": "ResNet", "description": "Deep residual network with strong performance"},
            {"name": "EfficientNet-B7", "description": "Large, high-performance EfficientNet variant"},
            {"name": "RegNet", "description": "Designed for efficient scaling on cloud hardware"}
        ]
    },
    "Transformer": {
        "edge": [
            {"name": "MobileBERT", "description": "Compressed BERT for mobile devices"},
            {"name": "TinyBERT", "description": "Lightweight BERT through knowledge distillation"},
            {"name": "DistilBERT", "description": "Smaller, faster BERT with similar performance"}
        ],
        "cloud": [
            {"name": "BERT-Large", "description": "Full-sized BERT model for maximum accuracy"},
            {"name": "GPT-3", "description": "Large language model requiring significant compute"},
            {"name": "T5", "description": "Text-To-Text Transfer Transformer with strong performance"}
        ]
    },
    "Diffusion Model": {
        "edge": [
            {"name": "Tiny-DDPM", "description": "Compact diffusion model with reduced parameters"},
            {"name": "FastDiff", "description": "Accelerated sampling for edge devices"},
            {"name": "MobileDiffusion", "description": "Optimized for mobile GPU inference"}
        ],
        "cloud": [
            {"name": "Stable Diffusion XL", "description": "Large high-quality image generation model"},
            {"name": "DALL-E", "description": "OpenAI's high-performance diffusion model"},
            {"name": "Imagen", "description": "Google's text-to-image diffusion model"}
        ]
    },
    "Graph Neural Network": {
        "edge": [
            {"name": "SGC", "description": "Simplified Graph Convolution for edge devices"},
            {"name": "LightGCN", "description": "Lightweight graph convolution network"},
            {"name": "FastGCN", "description": "Efficient sampling-based GNN"}
        ],
        "cloud": [
            {"name": "GraphSAGE", "description": "Full-feature graph sampling and aggregation"},
            {"name": "GAT", "description": "Graph Attention Network with multi-head attention"},
            {"name": "GIN", "description": "Graph Isomorphism Network for expressive graph representations"}
        ]
    }
}

# Add algorithm detection explanation patterns
ALGORITHM_EXPLANATION_PATTERNS = {
    "Linear Regression": {
        "LinearRegression": "Direct instantiation of sklearn's LinearRegression class",
        "np\\.linalg\\.lst": "Using NumPy's least squares solver, a common method for linear regression",
        "gradient descent": "Implementing gradient descent, which is commonly used to optimize linear regression",
        "normal equation": "Using the normal equation method to solve linear regression analytically"
    },
    "Transformer": {
        "Transformer": "Direct usage of a Transformer class or model",
        "attention": "Using attention mechanisms, a core component of Transformer models",
        "multi.?head": "Implementing multi-head attention, a key feature of Transformers",
        "encoder": "Building encoder layers typical in Transformer architectures",
        "decoder": "Creating decoder layers used in Transformer architectures"
    },
    "Vision Transformer": {
        "ViT": "Direct reference to ViT (Vision Transformer) class or model",
        "vision.?transformer": "Explicit mention of Vision Transformer architecture",
        "patch.?embed": "Using patch embedding, a characteristic feature of Vision Transformers"
    },
    "Graph Neural Network": {
        "GNN": "Direct reference to GNN (Graph Neural Network)",
        "GraphConv": "Using graph convolution operations",
        "MessagePassing": "Implementing message passing, a fundamental GNN concept",
        "graph.?neural": "Explicit mention of graph neural networks",
        "node_features": "Processing node features in a graph structure"
    },
    "Diffusion Model": {
        "diffusion": "Direct reference to diffusion processes or models",
        "DDPM": "Using Denoising Diffusion Probabilistic Models",
        "DDIM": "Implementing Denoising Diffusion Implicit Models",
        "score.?based": "Using score-based generative modeling",
        "noise.?prediction": "Implementing noise prediction/removal, core to diffusion models"
    },
    "Reinforcement Learning": {
        "PPO": "Using Proximal Policy Optimization algorithm",
        "DQN": "Implementing Deep Q-Network",
        "A2C": "Using Advantage Actor-Critic method",
        "DDPG": "Using Deep Deterministic Policy Gradient",
        "reinforcement": "Direct reference to reinforcement learning",
        "reward": "Working with reward signals, fundamental to RL",
        "action.*space": "Defining action spaces for RL environments"
    }
}

# Define AST visitors for code parsing
class ModelStructureVisitor(ast.NodeVisitor):
    """
    AST visitor to extract model architecture structure from code.
    """
    
    def __init__(self):
        self.layers = []
        self.has_sequential_model = False
        self.has_functional_api = False
        self.has_model_subclass = False
        self.input_shapes = []
        self.current_class = None
        self.class_definitions = {}
        
    def visit_ClassDef(self, node):
        """Visit class definitions to find model subclasses"""
        self.current_class = node.name
        # Check if class inherits from nn.Module or keras.Model
        for base in node.bases:
            base_str = ast.unparse(base).strip() if hasattr(ast, 'unparse') else self._get_base_str(base)
            if "nn.Module" in base_str or "torch.nn.Module" in base_str:
                self.has_model_subclass = True
                self.class_definitions[node.name] = {"type": "PyTorch", "bases": [base_str]}
            elif "keras.Model" in base_str or "tf.keras.Model" in base_str:
                self.has_model_subclass = True
                self.class_definitions[node.name] = {"type": "TensorFlow", "bases": [base_str]}
        
        # Visit the class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self._extract_layers_from_init(item)
            self.visit(item)
        
        self.current_class = None
    
    def _get_base_str(self, node):
        """Get string representation of base class for Python < 3.9"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_str(node.value)}.{node.attr}"
        return str(node)
    
    def _extract_layers_from_init(self, init_func):
        """Extract layer definitions from __init__ method"""
        for stmt in init_func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        # This is a self.attribute assignment
                        if isinstance(stmt.value, ast.Call):
                            layer_type = self._get_call_name(stmt.value)
                            if any(layer_key in layer_type for layer_key in ["Conv", "Linear", "Dense", "Pool", "Dropout", "BatchNorm", "LayerNorm", "Attention"]):
                                # This is likely a layer assignment
                                self.layers.append({
                                    "name": target.attr,
                                    "type": layer_type,
                                    "config": self._extract_call_args(stmt.value)
                                })
    
    def _get_call_name(self, node):
        """Get the name of a function/class being called"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attribute_name(node.func.value)}.{node.func.attr}"
        return "unknown"
    
    def _get_attribute_name(self, node):
        """Get full attribute name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _extract_call_args(self, call_node):
        """Extract arguments from a function call"""
        config = {}
        # Handle positional args based on common layer signatures
        if hasattr(call_node, 'args') and call_node.args:
            func_name = self._get_call_name(call_node)
            if "Conv" in func_name:
                if len(call_node.args) >= 2:
                    config["in_channels"] = self._get_arg_value(call_node.args[0])
                    config["out_channels"] = self._get_arg_value(call_node.args[1])
                if len(call_node.args) >= 3:
                    config["kernel_size"] = self._get_arg_value(call_node.args[2])
            elif "Linear" in func_name or "Dense" in func_name:
                if len(call_node.args) >= 2:
                    config["in_features"] = self._get_arg_value(call_node.args[0])
                    config["out_features"] = self._get_arg_value(call_node.args[1])
        
        # Handle keyword args
        for kw in call_node.keywords:
            config[kw.arg] = self._get_arg_value(kw.value)
        
        return config
    
    def _get_arg_value(self, node):
        """Get value of an argument node"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id  # Just return the variable name
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_arg_value(elt) for elt in node.elts)
        elif isinstance(node, ast.List):
            return [self._get_arg_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Constant):
            return node.value
        # For more complex expressions, we'll just return a placeholder
        return "complex_expression"
    
    def visit_Assign(self, node):
        """Visit assignments to find model definitions"""
        # Check for Sequential model definition
        if isinstance(node.value, ast.Call):
            call_name = self._get_call_name(node.value)
            if "Sequential" in call_name:
                self.has_sequential_model = True
                # Extract layers from Sequential model
                for arg in node.value.args:
                    if isinstance(arg, ast.List):
                        for elt in arg.elts:
                            if isinstance(elt, ast.Call):
                                layer_type = self._get_call_name(elt)
                                self.layers.append({
                                    "name": f"layer_{len(self.layers)}",
                                    "type": layer_type,
                                    "config": self._extract_call_args(elt)
                                })
            # Check for functional API model definition
            elif "Model" in call_name:
                self.has_functional_api = True
        
        # Look for input shape definitions
        try:
            shape_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)'
            node_str = ast.unparse(node).strip() if hasattr(ast, 'unparse') else str(node)
            if "input_shape" in node_str or "shape" in node_str:
                matches = re.search(shape_pattern, node_str)
                if matches:
                    dims = [int(d) for d in matches.groups() if d]
                    if len(dims) >= 2:
                        # Add batch dimension if not present
                        if len(dims) == 2 or (len(dims) == 3 and dims[0] <= 3):  # Likely channels, height, width
                            dims = [1] + dims
                        self.input_shapes.append(tuple(dims))
        except Exception as e:
            pass  # Silently ignore parsing errors in shape detection
        
        # Continue visiting children
        self.generic_visit(node)

class AlgorithmPatternVisitor(ast.NodeVisitor):
    """
    AST visitor to detect algorithm patterns in code.
    """
    
    def __init__(self, patterns_dict):
        self.patterns_dict = patterns_dict
        self.matches = defaultdict(list)
        self.function_calls = []
        self.imports = []
        self.class_definitions = []
        self.variable_assignments = []
    
    def visit_Call(self, node):
        """Visit function/method calls"""
        call_str = ""
        try:
            call_str = ast.unparse(node) if hasattr(ast, 'unparse') else self._get_call_str(node)
        except:
            call_str = self._get_call_str(node)
            
        self.function_calls.append(call_str)
        
        # Check for algorithm pattern matches
        for algo_name, patterns in self.patterns_dict.items():
            for pattern in patterns:
                if re.search(pattern, call_str, re.IGNORECASE):
                    self.matches[algo_name].append((pattern, call_str))
        
        self.generic_visit(node)
    
    def _get_call_str(self, node):
        """Get string representation of a call for Python < 3.9"""
        func_str = ""
        if isinstance(node.func, ast.Name):
            func_str = node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj_name = self._get_attribute_name(node.func.value)
            func_str = f"{obj_name}.{node.func.attr}"
        else:
            func_str = "unknown"
            
        args_str = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args_str.append(arg.id)
            elif isinstance(arg, ast.Constant):
                args_str.append(str(arg.value))
            else:
                args_str.append("...")
                
        kw_str = []
        for kw in node.keywords:
            kw_str.append(f"{kw.arg}=...")
            
        return f"{func_str}({', '.join(args_str + kw_str)})"
    
    def _get_attribute_name(self, node):
        """Get full attribute name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return "unknown"
    
    def visit_Import(self, node):
        """Visit import statements"""
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements"""
        for name in node.names:
            if node.module:
                self.imports.append(f"{node.module}.{name.name}")
            else:
                self.imports.append(name.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions"""
        self.class_definitions.append({
            'name': node.name,
            'bases': [self._get_name(base) for base in node.bases]
        })
        self.generic_visit(node)
    
    def _get_name(self, node):
        """Get name from a node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"
    
    def visit_Assign(self, node):
        """Visit assignments"""
        try:
            assign_str = ast.unparse(node) if hasattr(ast, 'unparse') else "assignment"
            self.variable_assignments.append(assign_str)
            
            # Check for algorithm pattern matches in assignments
            for algo_name, patterns in self.patterns_dict.items():
                for pattern in patterns:
                    if re.search(pattern, assign_str, re.IGNORECASE):
                        self.matches[algo_name].append((pattern, assign_str))
        except:
            pass
        
        self.generic_visit(node)