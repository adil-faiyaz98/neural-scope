"""
Core performance profiling functionality for ML models.

This module provides the base ModelPerformanceProfiler class that handles
basic profiling of ML models, including execution time, memory usage,
and hardware utilization metrics.
"""

import time
import logging
import warnings
import traceback
import contextlib
from typing import Dict, List, Optional, Union, Any, Generator, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Framework-specific imports with proper error handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorFlow-specific features will be disabled.")

@dataclass
class ProfilingResult:
    """Stores the results of model profiling"""
    execution_time: float
    memory_usage: Dict[str, float]
    throughput: float
    hardware_utilization: Dict[str, float]
    bottlenecks: List[str]
    optimization_suggestions: List[str]
    
    def __post_init__(self):
        """Validate the profiling results after initialization"""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        for key, value in self.memory_usage.items():
            if value < 0:
                raise ValueError(f"Memory usage for {key} cannot be negative")
        if self.throughput < 0:
            raise ValueError("Throughput cannot be negative")

class ModelPerformanceProfiler:
    """Enhanced performance profiler with multi-GPU, cloud and energy tracking capabilities"""
    
    def __init__(self, model=None, input_shape=None):
        self.model = model
        self.input_shape = input_shape
        self.monitoring = False
        self.metrics = {}
        
    def profile_memory(self):
        """Profile model memory usage."""
        import torch
        import psutil
        
        memory_stats = {
            "total_memory": 0,
            "parameters_memory": 0,
            "buffers_memory": 0,
            "activation_memory": 0
        }
        
        # Calculate parameters memory
        parameters_memory = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        memory_stats["parameters_memory"] = parameters_memory
        
        # Calculate buffers memory
        buffers_memory = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        memory_stats["buffers_memory"] = buffers_memory
        
        # Estimate activation memory (rough estimate)
        with torch.no_grad():
            sample_input = torch.randn(self.input_shape)
            if torch.cuda.is_available():
                sample_input = sample_input.cuda()
                self.model = self.model.cuda()
                torch.cuda.reset_peak_memory_stats()
                _ = self.model(sample_input)
                activation_memory = torch.cuda.max_memory_allocated()
            else:
                process = psutil.Process()
                memory_before = process.memory_info().rss
                _ = self.model(sample_input)
                memory_after = process.memory_info().rss
                activation_memory = memory_after - memory_before
        
        memory_stats["activation_memory"] = activation_memory
        memory_stats["total_memory"] = parameters_memory + buffers_memory + activation_memory
        
        return memory_stats
    
    def profile_compute(self):
        """Profile model compute requirements."""
        import torch
        
        compute_stats = {
            "total_flops": 0,
            "flops_per_input": 0,
            "by_layer": {}
        }
        
        def count_conv2d(m, x, y):
            x = x[0]
            out_h = y.size(2)
            out_w = y.size(3)
            kernel_ops = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
            bias_ops = 1 if m.bias is not None else 0
            ops_per_element = kernel_ops + bias_ops
            output_elements = y.nelement()
            total_ops = ops_per_element * output_elements
            
            m.total_ops = torch.DoubleTensor([int(total_ops)])
        
        def count_linear(m, x, y):
            total_ops = m.in_features * m.out_features
            if m.bias is not None:
                total_ops += m.out_features
            m.total_ops = torch.DoubleTensor([int(total_ops)])
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(count_conv2d))
            elif isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(count_linear))
        
        # Run model
        sample_input = torch.randn(self.input_shape)
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
            self.model = self.model.cuda()
        
        with torch.no_grad():
            _ = self.model(sample_input)
        
        # Collect stats
        total_flops = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'total_ops'):
                flops = int(module.total_ops.item())
                compute_stats["by_layer"][name] = flops
                total_flops += flops
                delattr(module, 'total_ops')
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        compute_stats["total_flops"] = total_flops
        compute_stats["flops_per_input"] = total_flops / self.input_shape[0]
        
        return compute_stats
    
    def profile_latency(self, num_iterations=50):
        """Profile model latency."""
        import torch
        import time
        import numpy as np
        
        latency_stats = {
            "mean_latency": 0,
            "std_latency": 0,
            "min_latency": 0,
            "max_latency": 0,
            "p90_latency": 0,
            "p95_latency": 0,
            "p99_latency": 0
        }
        
        # Warmup
        sample_input = torch.randn(self.input_shape)
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
            self.model = self.model.cuda()
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(sample_input)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = self.model(sample_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        latency_stats["mean_latency"] = float(np.mean(latencies))
        latency_stats["std_latency"] = float(np.std(latencies))
        latency_stats["min_latency"] = float(np.min(latencies))
        latency_stats["max_latency"] = float(np.max(latencies))
        latency_stats["p90_latency"] = float(np.percentile(latencies, 90))
        latency_stats["p95_latency"] = float(np.percentile(latencies, 95))
        latency_stats["p99_latency"] = float(np.percentile(latencies, 99))
        
        return latency_stats
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [] if torch.cuda.is_available() else None,
            "gpu_memory": [] if torch.cuda.is_available() else None
        }
    
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        self.monitoring = False
        return self.metrics
    
    def profile_model(self, model, input_tensor, batch_size=32):
        """Profile model performance comprehensively."""
        self.model = model
        self.input_shape = input_tensor.shape
        
        results = {
            "memory": self.profile_memory(),
            "compute": self.profile_compute(),
            "latency": self.profile_latency(),
            "batch_size": batch_size
        }
        
        return results
