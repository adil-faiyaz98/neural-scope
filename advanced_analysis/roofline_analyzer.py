import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComputeMetrics:
    """Store compute intensiveness metrics for operations"""
    flops: float  # Total floating point operations
    bytes_accessed: float  # Total memory bytes accessed
    compute_intensity: float  # FLOPS/Byte
    theoretical_peak: float  # Theoretical peak performance
    achieved_performance: float  # Actual achieved performance
    efficiency: float  # Achieved/Theoretical efficiency
    kernel_name: str  # Operation kernel name


@dataclass
class RooflineResult:
    """Store roofline model analysis results"""
    operations: List[ComputeMetrics]  # Metrics for each operation
    peak_compute: float  # Peak compute capability (TFLOPS)
    peak_memory_bandwidth: float  # Peak memory bandwidth (GB/s)
    memory_bound_ops: List[str]  # Memory-bound operations
    compute_bound_ops: List[str]  # Compute-bound operations
    recommendations: List[Dict[str, Any]]  # Optimization recommendations


class RooflineAnalyzer:
    """
    Analyze model computations using roofline model to identify 
    memory-bound vs. compute-bound operations
    """
    
    def __init__(self, model, framework="pytorch"):
        self.model = model
        self.framework = framework.lower()
        self.device_info = self._get_device_info()
        
    def _get_device_info(self):
        """Get hardware capability information"""
        device_info = {}
        
        if self.framework == "pytorch" and torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            
            # Calculate theoretical peak performance
            # FP32 TFLOPS = cores * 2 ops per cycle * clock_rate / 10^12
            sm_count = properties.multi_processor_count
            # Estimate for most architectures - adjust if needed
            cores_per_sm = 64  # Typical for recent NVIDIA architectures
            clock_mhz = properties.clock_rate / 1000
            
            device_info['peak_fp32_tflops'] = (sm_count * cores_per_sm * 2 * clock_mhz) / 1e6
            device_info['peak_fp16_tflops'] = device_info['peak_fp32_tflops'] * 2  # Typical for recent GPUs
            
            # Memory bandwidth (GB/s)
            memory_clock_mhz = properties.memory_clock_rate / 1000
            bus_width = properties.memory_bus_width
            device_info['peak_memory_bandwidth'] = (memory_clock_mhz * bus_width * 2) / 8 / 1000  # GB/s
            
            device_info['name'] = properties.name
            device_info['total_memory'] = properties.total_memory / (1024**3)  # GB
            
        else:
            # Default values if hardware detection fails
            device_info['peak_fp32_tflops'] = 10.0  # Default assumption: 10 TFLOPS
            device_info['peak_fp16_tflops'] = 20.0
            device_info['peak_memory_bandwidth'] = 500.0  # Default: 500 GB/s
            device_info['name'] = "Unknown"
            device_info['total_memory'] = 16.0  # GB
            
        return device_info
        
    def analyze(self, input_data, use_fp16=False) -> RooflineResult:
        """Perform roofline model analysis on the model"""
        # Prepare operation metrics collection
        operation_metrics = []
        
        if self.framework == "pytorch":
            # Use PyTorch profiler to collect operation metrics
            try:
                from torch.profiler import profile, tensorboard_trace_handler
                from torch.profiler import ProfilerActivity
                
                # Run with profiler
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True,
                    with_modules=True,
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    # Perform forward pass
                    with torch.no_grad():
                        _ = self.model(input_data)
                        torch.cuda.synchronize()
                
                # Collect metrics for significant operations
                for event in prof.key_averages():
                    if event.flops > 0:
                        # Estimate memory accessed (imperfect estimation)
                        input_shape_str = str(event.input_shapes)
                        output_shape_str = str(event.output_shapes)
                        
                        # Calculate input size
                        input_bytes = 0
                        if hasattr(event, 'input_shapes') and event.input_shapes:
                            for shape in event.input_shapes:
                                if shape and len(shape) > 0:
                                    elements = np.prod([dim for dim in shape if isinstance(dim, int) and dim > 0])
                                    input_bytes += elements * (2 if use_fp16 else 4)  # bytes per element
                        
                        # Calculate output size
                        output_bytes = 0
                        if hasattr(event, 'output_shapes') and event.output_shapes:
                            for shape in event.output_shapes:
                                if shape and len(shape) > 0:
                                    elements = np.prod([dim for dim in shape if isinstance(dim, int) and dim > 0])
                                    output_bytes += elements * (2 if use_fp16 else 4)  # bytes per element
                        
                        # Total memory accessed
                        memory_bytes = input_bytes + output_bytes
                        
                        # Skip operations with abnormal values
                        if memory_bytes <= 0:
                            continue
                            
                        # Compute intensity (FLOPS/Byte)
                        compute_intensity = event.flops / memory_bytes
                        
                        # Achieved performance (TFLOPS)
                        achieved_tflops = event.flops / (event.cuda_time * 1e3) / 1e12 if event.cuda_time > 0 else 0
                        
                        # Calculate theoretical peak
                        roofline_peak = min(
                            self.device_info['peak_fp16_tflops'] if use_fp16 else self.device_info['peak_fp32_tflops'],
                            compute_intensity * self.device_info['peak_memory_bandwidth']
                        )
                        
                        # Calculate efficiency
                        efficiency = (achieved_tflops / roofline_peak) if roofline_peak > 0 else 0
                        
                        # Create and store operation metrics
                        metrics = ComputeMetrics(
                            flops=event.flops,
                            bytes_accessed=memory_bytes,
                            compute_intensity=compute_intensity,
                            theoretical_peak=roofline_peak * 1e12,  # Back to FLOPS
                            achieved_performance=achieved_tflops * 1e12,  # Back to FLOPS
                            efficiency=efficiency,
                            kernel_name=f"{event.key} ({event.node_name})"
                        )
                        
                        operation_metrics.append(metrics)
                        
            except Exception as e:
                logger.error(f"Failed to perform roofline analysis: {e}")
                
        # Sort operations by FLOPS for importance
        operation_metrics.sort(key=lambda x: x.flops, reverse=True)
        
        # Get top operations (at most 30)
        top_operations = operation_metrics[:30]
        
        # Identify memory-bound vs. compute-bound operations
        ridge_point = self.device_info['peak_fp16_tflops' if use_fp16 else 'peak_fp32_tflops'] / self.device_info['peak_memory_bandwidth']
        
        memory_bound_ops = [op.kernel_name for op in top_operations if op.compute_intensity < ridge_point]
        compute_bound_ops = [op.kernel_name for op in top_operations if op.compute_intensity >= ridge_point]
        
        # Generate recommendations based on analysis
        recommendations = self._generate_roofline_recommendations(
            top_operations, ridge_point, memory_bound_ops, compute_bound_ops, use_fp16
        )
        
        # Create result
        result = RooflineResult(
            operations=top_operations,
            peak_compute=self.device_info['peak_fp16_tflops' if use_fp16 else 'peak_fp32_tflops'],
            peak_memory_bandwidth=self.device_info['peak_memory_bandwidth'],
            memory_bound_ops=memory_bound_ops,
            compute_bound_ops=compute_bound_ops,
            recommendations=recommendations
        )
        
        return result
    
    def visualize_roofline(self, result: RooflineResult):
        """Generate roofline model visualization"""
        plt.figure(figsize=(12, 8))
        
        # Create x-axis range based on operation characteristics
        min_intensity = min([op.compute_intensity for op in result.operations])
        max_intensity = max([op.compute_intensity for op in result.operations])
        
        x_min = max(0.1, min_intensity / 10)
        x_max = max(100, max_intensity * 10)
        
        x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
        
        # Memory bandwidth ceiling
        y_mem = x * result.peak_memory_bandwidth
        
        # Compute ceiling
        y_compute = np.ones_like(x) * result.peak_compute * 1e12
        
        # Draw roofline ceilings
        plt.loglog(x, np.minimum(y_mem, y_compute), 'k-', linewidth=2, label='Hardware Roofline')
        
        # Ridge point
        ridge_point = result.peak_compute / result.peak_memory_bandwidth
        plt.axvline(x=ridge_point, color='r', linestyle='--', alpha=0.7, 
                    label=f'Ridge Point ({ridge_point:.1f} FLOP/B)')
        
        # Plot each operation
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, op in enumerate(result.operations):
            color_idx = i % len(colors)
            plt.scatter(op.compute_intensity, op.achieved_performance, 
                       s=100 * np.log10(1 + op.flops/1e9),  # Size based on GFLOPS
                       alpha=0.7, color=colors[color_idx], 
                       edgecolors='black', linewidths=1)
            
            # Add labels for top 10 operations
            if i < 10:
                plt.annotate(op.kernel_name.split('(')[0],
                            xy=(op.compute_intensity, op.achieved_performance),
                            xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
        plt.ylabel('Performance (FLOPS)')
        plt.title(f'Roofline Model Analysis on {self.device_info["name"]}')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        
        # Add explanatory text
        plt.figtext(0.02, 0.02, 
                   f"Peak Performance: {result.peak_compute:.2f} TFLOPS\n"
                   f"Peak Memory Bandwidth: {result.peak_memory_bandwidth:.0f} GB/s\n"
                   f"Memory-Bound Ops: {len(result.memory_bound_ops)}\n"
                   f"Compute-Bound Ops: {len(result.compute_bound_ops)}", 
                   fontsize=9)
                   
        return plt
        
    def _generate_roofline_recommendations(self, operations, ridge_point, memory_bound_ops, compute_bound_ops, use_fp16):
        """Generate optimization recommendations based on roofline analysis"""
        recommendations = []
        
        # 1. Overall recommendation based on dominant bottleneck type
        if len(memory_bound_ops) > len(compute_bound_ops):
            recommendations.append({
                'type': 'memory_bandwidth',
                'severity': 'high',
                'description': "Model is primarily memory-bandwidth bound",
                'recommendation': "Focus on optimizations that reduce memory traffic: operator fusion, "
                                 "memory layout optimization, and bandwidth optimization."
            })
        else:
            recommendations.append({
                'type': 'compute_optimization',
                'severity': 'high',
                'description': "Model is primarily compute-bound",
                'recommendation': "Focus on compute optimizations: kernel tuning, tensor core utilization, "
                                 "and algorithm selection."
            })
            
        # 2. Check for operations with poor efficiency
        inefficient_ops = [op for op in operations if op.efficiency < 0.3]
        if inefficient_ops:
            low_efficiency_names = [op.kernel_name for op in inefficient_ops[:5]]
            recommendations.append({
                'type': 'kernel_efficiency',
                'severity': 'high',
                'description': f"Low compute efficiency detected in {len(inefficient_ops)} operations",
                'recommendation': f"Consider optimizing: {', '.join(low_efficiency_names[:3])}",
                'code_example': """
# Consider using specialized kernels or optimized implementations
# For example, if using attention mechanisms:
from flash_attn import flash_attn_func
# Replace standard attention with Flash Attention
attn_output = flash_attn_func(q, k, v, dropout_p=0.0)
"""
            })
            
        # 3. Mixed precision recommendation if not already using
        if not use_fp16:
            recommendations.append({
                'type': 'mixed_precision',
                'severity': 'medium',
                'description': "Model running in FP32 mode",
                'recommendation': "Switch to mixed precision (FP16/BF16) to increase arithmetic intensity "
                                 "and better utilize tensor cores on compatible hardware.",
                'code_example': """
# PyTorch mixed precision# filepath: c:\Users\adilm\repositories\Python\neural-scope\advanced_analysis\roofline_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComputeMetrics:
    """Store compute intensiveness metrics for operations"""
    flops: float  # Total floating point operations
    bytes_accessed: float  # Total memory bytes accessed
    compute_intensity: float  # FLOPS/Byte
    theoretical_peak: float  # Theoretical peak performance
    achieved_performance: float  # Actual achieved performance
    efficiency: float  # Achieved/Theoretical efficiency
    kernel_name: str  # Operation kernel name


@dataclass
class RooflineResult:
    """Store roofline model analysis results"""
    operations: List[ComputeMetrics]  # Metrics for each operation
    peak_compute: float  # Peak compute capability (TFLOPS)
    peak_memory_bandwidth: float  # Peak memory bandwidth (GB/s)
    memory_bound_ops: List[str]  # Memory-bound operations
    compute_bound_ops: List[str]  # Compute-bound operations
    recommendations: List[Dict[str, Any]]  # Optimization recommendations


class RooflineAnalyzer:
    """
    Analyze model computations using roofline model to identify 
    memory-bound vs. compute-bound operations
    """
    
    def __init__(self, model, framework="pytorch"):
        self.model = model
        self.framework = framework.lower()
        self.device_info = self._get_device_info()
        
    def _get_device_info(self):
        """Get hardware capability information"""
        device_info = {}
        
        if self.framework == "pytorch" and torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            
            # Calculate theoretical peak performance
            # FP32 TFLOPS = cores * 2 ops per cycle * clock_rate / 10^12
            sm_count = properties.multi_processor_count
            # Estimate for most architectures - adjust if needed
            cores_per_sm = 64  # Typical for recent NVIDIA architectures
            clock_mhz = properties.clock_rate / 1000
            
            device_info['peak_fp32_tflops'] = (sm_count * cores_per_sm * 2 * clock_mhz) / 1e6
            device_info['peak_fp16_tflops'] = device_info['peak_fp32_tflops'] * 2  # Typical for recent GPUs
            
            # Memory bandwidth (GB/s)
            memory_clock_mhz = properties.memory_clock_rate / 1000
            bus_width = properties.memory_bus_width
            device_info['peak_memory_bandwidth'] = (memory_clock_mhz * bus_width * 2) / 8 / 1000  # GB/s
            
            device_info['name'] = properties.name
            device_info['total_memory'] = properties.total_memory / (1024**3)  # GB
            
        else:
            # Default values if hardware detection fails
            device_info['peak_fp32_tflops'] = 10.0  # Default assumption: 10 TFLOPS
            device_info['peak_fp16_tflops'] = 20.0
            device_info['peak_memory_bandwidth'] = 500.0  # Default: 500 GB/s
            device_info['name'] = "Unknown"
            device_info['total_memory'] = 16.0  # GB
            
        return device_info
        
    def analyze(self, input_data, use_fp16=False) -> RooflineResult:
        """Perform roofline model analysis on the model"""
        # Prepare operation metrics collection
        operation_metrics = []
        
        if self.framework == "pytorch":
            # Use PyTorch profiler to collect operation metrics
            try:
                from torch.profiler import profile, tensorboard_trace_handler
                from torch.profiler import ProfilerActivity
                
                # Run with profiler
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True,
                    with_modules=True,
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    # Perform forward pass
                    with torch.no_grad():
                        _ = self.model(input_data)
                        torch.cuda.synchronize()
                
                # Collect metrics for significant operations
                for event in prof.key_averages():
                    if event.flops > 0:
                        # Estimate memory accessed (imperfect estimation)
                        input_shape_str = str(event.input_shapes)
                        output_shape_str = str(event.output_shapes)
                        
                        # Calculate input size
                        input_bytes = 0
                        if hasattr(event, 'input_shapes') and event.input_shapes:
                            for shape in event.input_shapes:
                                if shape and len(shape) > 0:
                                    elements = np.prod([dim for dim in shape if isinstance(dim, int) and dim > 0])
                                    input_bytes += elements * (2 if use_fp16 else 4)  # bytes per element
                        
                        # Calculate output size
                        output_bytes = 0
                        if hasattr(event, 'output_shapes') and event.output_shapes:
                            for shape in event.output_shapes:
                                if shape and len(shape) > 0:
                                    elements = np.prod([dim for dim in shape if isinstance(dim, int) and dim > 0])
                                    output_bytes += elements * (2 if use_fp16 else 4)  # bytes per element
                        
                        # Total memory accessed
                        memory_bytes = input_bytes + output_bytes
                        
                        # Skip operations with abnormal values
                        if memory_bytes <= 0:
                            continue
                            
                        # Compute intensity (FLOPS/Byte)
                        compute_intensity = event.flops / memory_bytes
                        
                        # Achieved performance (TFLOPS)
                        achieved_tflops = event.flops / (event.cuda_time * 1e3) / 1e12 if event.cuda_time > 0 else 0
                        
                        # Calculate theoretical peak
                        roofline_peak = min(
                            self.device_info['peak_fp16_tflops'] if use_fp16 else self.device_info['peak_fp32_tflops'],
                            compute_intensity * self.device_info['peak_memory_bandwidth']
                        )
                        
                        # Calculate efficiency
                        efficiency = (achieved_tflops / roofline_peak) if roofline_peak > 0 else 0
                        
                        # Create and store operation metrics
                        metrics = ComputeMetrics(
                            flops=event.flops,
                            bytes_accessed=memory_bytes,
                            compute_intensity=compute_intensity,
                            theoretical_peak=roofline_peak * 1e12,  # Back to FLOPS
                            achieved_performance=achieved_tflops * 1e12,  # Back to FLOPS
                            efficiency=efficiency,
                            kernel_name=f"{event.key} ({event.node_name})"
                        )
                        
                        operation_metrics.append(metrics)
                        
            except Exception as e:
                logger.error(f"Failed to perform roofline analysis: {e}")
                
        # Sort operations by FLOPS for importance
        operation_metrics.sort(key=lambda x: x.flops, reverse=True)
        
        # Get top operations (at most 30)
        top_operations = operation_metrics[:30]
        
        # Identify memory-bound vs. compute-bound operations
        ridge_point = self.device_info['peak_fp16_tflops' if use_fp16 else 'peak_fp32_tflops'] / self.device_info['peak_memory_bandwidth']
        
        memory_bound_ops = [op.kernel_name for op in top_operations if op.compute_intensity < ridge_point]
        compute_bound_ops = [op.kernel_name for op in top_operations if op.compute_intensity >= ridge_point]
        
        # Generate recommendations based on analysis
        recommendations = self._generate_roofline_recommendations(
            top_operations, ridge_point, memory_bound_ops, compute_bound_ops, use_fp16
        )
        
        # Create result
        result = RooflineResult(
            operations=top_operations,
            peak_compute=self.device_info['peak_fp16_tflops' if use_fp16 else 'peak_fp32_tflops'],
            peak_memory_bandwidth=self.device_info['peak_memory_bandwidth'],
            memory_bound_ops=memory_bound_ops,
            compute_bound_ops=compute_bound_ops,
            recommendations=recommendations
        )
        
        return result
    
    def visualize_roofline(self, result: RooflineResult):
        """Generate roofline model visualization"""
        plt.figure(figsize=(12, 8))
        
        # Create x-axis range based on operation characteristics
        min_intensity = min([op.compute_intensity for op in result.operations])
        max_intensity = max([op.compute_intensity for op in result.operations])
        
        x_min = max(0.1, min_intensity / 10)
        x_max = max(100, max_intensity * 10)
        
        x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
        
        # Memory bandwidth ceiling
        y_mem = x * result.peak_memory_bandwidth
        
        # Compute ceiling
        y_compute = np.ones_like(x) * result.peak_compute * 1e12
        
        # Draw roofline ceilings
        plt.loglog(x, np.minimum(y_mem, y_compute), 'k-', linewidth=2, label='Hardware Roofline')
        
        # Ridge point
        ridge_point = result.peak_compute / result.peak_memory_bandwidth
        plt.axvline(x=ridge_point, color='r', linestyle='--', alpha=0.7, 
                    label=f'Ridge Point ({ridge_point:.1f} FLOP/B)')
        
        # Plot each operation
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, op in enumerate(result.operations):
            color_idx = i % len(colors)
            plt.scatter(op.compute_intensity, op.achieved_performance, 
                       s=100 * np.log10(1 + op.flops/1e9),  # Size based on GFLOPS
                       alpha=0.7, color=colors[color_idx], 
                       edgecolors='black', linewidths=1)
            
            # Add labels for top 10 operations
            if i < 10:
                plt.annotate(op.kernel_name.split('(')[0],
                            xy=(op.compute_intensity, op.achieved_performance),
                            xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
        plt.ylabel('Performance (FLOPS)')
        plt.title(f'Roofline Model Analysis on {self.device_info["name"]}')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        
        # Add explanatory text
        plt.figtext(0.02, 0.02, 
                   f"Peak Performance: {result.peak_compute:.2f} TFLOPS\n"
                   f"Peak Memory Bandwidth: {result.peak_memory_bandwidth:.0f} GB/s\n"
                   f"Memory-Bound Ops: {len(result.memory_bound_ops)}\n"
                   f"Compute-Bound Ops: {len(result.compute_bound_ops)}", 
                   fontsize=9)
                   
        return plt
        
    def _generate_roofline_recommendations(self, operations, ridge_point, memory_bound_ops, compute_bound_ops, use_fp16):
        """Generate optimization recommendations based on roofline analysis"""
        recommendations = []
        
        # 1. Overall recommendation based on dominant bottleneck type
        if len(memory_bound_ops) > len(compute_bound_ops):
            recommendations.append({
                'type': 'memory_bandwidth',
                'severity': 'high',
                'description': "Model is primarily memory-bandwidth bound",
                'recommendation': "Focus on optimizations that reduce memory traffic: operator fusion, "
                                 "memory layout optimization, and bandwidth optimization."
            })
        else:
            recommendations.append({
                'type': 'compute_optimization',
                'severity': 'high',
                'description': "Model is primarily compute-bound",
                'recommendation': "Focus on compute optimizations: kernel tuning, tensor core utilization, "
                                 "and algorithm selection."
            })
            
        # 2. Check for operations with poor efficiency
        inefficient_ops = [op for op in operations if op.efficiency < 0.3]
        if inefficient_ops:
            low_efficiency_names = [op.kernel_name for op in inefficient_ops[:5]]
            recommendations.append({
                'type': 'kernel_efficiency',
                'severity': 'high',
                'description': f"Low compute efficiency detected in {len(inefficient_ops)} operations",
                'recommendation': f"Consider optimizing: {', '.join(low_efficiency_names[:3])}",
                'code_example': """
# Consider using specialized kernels or optimized implementations
# For example, if using attention mechanisms:
from flash_attn import flash_attn_func
# Replace standard attention with Flash Attention
attn_output = flash_attn_func(q, k, v, dropout_p=0.0)
"""
            })
            
        # 3. Mixed precision recommendation if not already using
        if not use_fp16:
            recommendations.append({
                'type': 'mixed_precision',
                'severity': 'medium',
                'description': "Model running in FP32 mode",
                'recommendation': "Switch to mixed precision (FP16/BF16) to increase arithmetic intensity "
                                 "and better utilize tensor cores on compatible hardware.",
                'code_example': """
# PyTorch mixed precision'
                'code_example': """
# PyTorch mixed precision
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for gradient scaling
scaler = GradScaler()

# Training loop
for inputs, labels in dataloader:
    # Forward pass with mixed precision
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""
            })
            
        # 4. Memory bound optimizations
        if len(memory_bound_ops) >= 3:
            # Get the top 3 memory-bound operations by FLOPS
            top_memory_ops = [op for op in operations if op.kernel_name in memory_bound_ops]
            top_memory_ops = sorted(top_memory_ops, key=lambda x: x.flops, reverse=True)[:3]
            
            if top_memory_ops:
                op_names = [op.kernel_name.split('(')[0] for op in top_memory_ops]
                recommendations.append({
                    'type': 'memory_optimization',
                    'severity': 'high',
                    'description': f"Top memory-bound operations: {', '.join(op_names)}",
                    'recommendation': "Apply memory-access optimizations: data layout transformation, "
                                     "operator fusion, memory pooling, or gradient checkpointing.",
                    'code_example': """
# Example of gradient checkpointing to reduce memory usage
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Split model into segments for checkpointing
        # This example uses sequential layers, adjust for your architecture
        self.segment1 = nn.Sequential(*list(model.children())[:5])
        self.segment2 = nn.Sequential(*list(model.children())[5:10])
        self.segment3 = nn.Sequential(*list(model.children())[10:])
        
    def forward(self, x):
        # Use checkpointing for memory-intensive segments
        x = checkpoint(self.segment1, x, use_reentrant=False)
        x = checkpoint(self.segment2, x, use_reentrant=False)
        x = self.segment3(x)  # Last segment without checkpointing
        return x

# Wrap model with checkpointing
model = CheckpointedModel(model)

# Alternative memory optimization: batch sharding
def forward_backward_shard(model, batch, loss_fn, optimizer, n_chunks=2):
    '''Manually shard a batch for lower memory usage'''
    optimizer.zero_grad()
    batch_size = batch[0].size(0)
    chunk_size = batch_size // n_chunks
    accumulated_loss = 0
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else batch_size
        
        inputs_chunk = batch[0][start_idx:end_idx]
        labels_chunk = batch[1][start_idx:end_idx]
        
        outputs = model(inputs_chunk)
        loss = loss_fn(outputs, labels_chunk) / n_chunks
        loss.backward()
        accumulated_loss += loss.item()
    
    optimizer.step()
    return accumulated_loss * n_chunks
"""
                })
                
        # 5. Compute bound optimizations
        if len(compute_bound_ops) >= 3:
            # Get the top 3 compute-bound operations by FLOPS
            top_compute_ops = [op for op in operations if op.kernel_name in compute_bound_ops]
            top_compute_ops = sorted(top_compute_ops, key=lambda x: x.flops, reverse=True)[:3]
            
            if top_compute_ops:
                op_names = [op.kernel_name.split('(')[0] for op in top_compute_ops]
                
                # Check for common compute-bound operations
                has_attention = any("attention" in op.lower() for op in op_names)
                has_conv = any("conv" in op.lower() for op in op_names)
                has_matmul = any("matmul" in op.lower() or "bmm" in op.lower() for op in op_names)
                
                # Customize recommendation based on operation types
                example_code = ""
                if has_attention:
                    example_code += """
# Optimize attention operations with Flash Attention
from flash_attn import flash_attn_func

# Replace standard attention with optimized version
# q, k, v are query, key, value tensors with shape [batch, seq_len, num_heads, head_dim]
def optimized_attention(q, k, v, mask=None):
    # Flash Attention requires specific tensor format and works best on A100/H100 GPUs
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Flash Attention with dramatically better compute efficiency
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    return output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
"""

                if has_conv:
                    example_code += """
# Optimize convolution operations
# 1. Use NHWC memory format instead of NCHW for Tensor Cores
x = x.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)

# 2. Use cudnn benchmarking for faster convolutions
torch.backends.cudnn.benchmark = True

# 3. Consider replacing some convolutions with depthwise separable convs
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
"""

                if has_matmul:
                    example_code += """
# Optimize matrix multiplication operations
# 1. Use torch.compile (PyTorch 2.0+)
import torch._dynamo as dynamo
model = torch.compile(model)

# 2. For custom matmul operations, use specialized libraries
try:
    from cutlass import Gemm
    # Example of using CUTLASS for optimized matrix multiplication
    def optimized_matmul(a, b):
        # This is simplified - actual CUTLASS usage requires more setup
        m, k = a.shape
        _, n = b.shape
        c = torch.empty(m, n, device='cuda')
        gemm = Gemm(a.shape, b.shape, c.shape)
        gemm(a, b, c)
        return c
except ImportError:
    pass  # CUTLASS not available
"""

                if example_code:
                    recommendations.append({
                        'type': 'compute_optimization',
                        'severity': 'medium',
                        'description': f"Performance optimization for compute-bound operations: {', '.join(op_names)}",
                        'recommendation': "Apply compute-specific optimizations for maximum GPU utilization.",
                        'code_example': example_code
                    })
                    
        # 6. Add recommendation for operator fusion
        if len(operations) > 15:  # If model has many operations, fusion might help
            recommendations.append({
                'type': 'operator_fusion',
                'severity': 'medium',
                'description': "Many separate operations detected - consider operator fusion",
                'recommendation': "Fuse multiple operations to reduce kernel launch overhead and memory traffic.",
                'code_example': """
# Method 1: Use torch.compile (PyTorch 2.0+) for automatic fusion
import torch
optimized_model = torch.compile(model, mode="reduce-overhead")

# Method 2: Custom fused operations
import torch.nn.functional as F

# Example: Fused GELU activation
def fused_bias_gelu(x, bias):
    return F.gelu(x + bias)
    
# Example: Fused LayerNorm
class FusedLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Fused implementation combining multiple operations
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias
"""
            })
            
        # 7. Add recommendation for specialized hardware if low overall efficiency
        avg_efficiency = sum(op.efficiency for op in operations) / len(operations) if operations else 0
        if avg_efficiency < 0.25:  # Less than 25% overall efficiency
            recommendations.append({
                'type': 'hardware_recommendation',
                'severity': 'medium',
                'description': f"Low overall hardware efficiency ({avg_efficiency:.1%})",
                'recommendation': "Consider using newer GPU architecture with better support for your workload patterns.",
                'additional_info': {
                    'current_gpu': self.device_info.get('name', 'Unknown'),
                    'avg_efficiency': avg_efficiency,
                    'suggested_hardware': [
                        "NVIDIA A100 or H100 for large batch training",
                        "NVIDIA L4 for inference workloads",
                        "AMD MI250 for HPC workloads"
                    ]
                }
            })
        
        return recommendations
        
    def advanced_analysis(self, input_data, use_fp16=False, batch_sizes=None):
        """
        Perform advanced roofline analysis with batch size scaling study
        
        Args:
            input_data: Base input data to analyze
            use_fp16: Whether to use FP16 mode
            batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16])
            
        Returns:
            Dictionary with detailed analysis results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
            
        # Results for each batch size
        batch_results = {}
        
        # Base input shape (without batch dimension)
        if isinstance(input_data, torch.Tensor):
            base_shape = input_data.shape[1:]
        else:
            # Default shape if we can't determine it
            base_shape = (3, 224, 224)  # Typical image input shape
            
        for bs in batch_sizes:
            try:
                # Create input with current batch size
                if isinstance(input_data, torch.Tensor):
                    batch_input = torch.zeros((bs,) + base_shape, device=input_data.device, dtype=input_data.dtype)
                else:
                    # Create dummy input if actual input not provided
                    batch_input = torch.zeros((bs,) + base_shape, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                # Run analysis
                result = self.analyze(batch_input, use_fp16)
                
                # Extract key metrics
                batch_results[bs] = {
                    'throughput': bs / result.operations[0].achieved_performance if result.operations else 0,
                    'efficiency': sum(op.efficiency for op in result.operations) / len(result.operations) if result.operations else 0,
                    'memory_bound_ops': len(result.memory_bound_ops),
                    'compute_bound_ops': len(result.compute_bound_ops),
                    'top_operations': [
                        {
                            'name': op.kernel_name,
                            'flops': op.flops,
                            'bytes': op.bytes_accessed,
                            'efficiency': op.efficiency
                        }
                        for op in result.operations[:5]  # Top 5 operations
                    ] if result.operations else []
                }
            except Exception as e:
                logger.error(f"Failed to analyze batch size {bs}: {e}")
                batch_results[bs] = {'error': str(e)}
                
        # Find optimal batch size based on throughput
        valid_batch_sizes = [bs for bs in batch_sizes if isinstance(batch_results[bs], dict) and 'throughput' in batch_results[bs]]
        if valid_batch_sizes:
            optimal_bs = max(valid_batch_sizes, key=lambda bs: batch_results[bs]['throughput'])
        else:
            optimal_bs = None
            
        # Generate advanced recommendations
        advanced_recommendations = self._generate_advanced_recommendations(batch_results, optimal_bs)
        
        return {
            'batch_analysis': batch_results,
            'optimal_batch_size': optimal_bs,
            'advanced_recommendations': advanced_recommendations
        }
        
    def _generate_advanced_recommendations(self, batch_results, optimal_bs):
        """Generate advanced recommendations based on batch size scaling behavior"""
        recommendations = []
        
        if not optimal_bs:
            return recommendations
            
        # Check if efficiency improves with larger batches
        batch_sizes = sorted(batch_results.keys())
        if len(batch_sizes) >= 2:
            small_bs = batch_sizes[0]
            large_bs = batch_sizes[-1]
            
            if isinstance(batch_results[small_bs], dict) and isinstance(batch_results[large_bs], dict):
                small_eff = batch_results[small_bs].get('efficiency', 0)
                large_eff = batch_results[large_bs].get('efficiency', 0)
                
                # If larger batch sizes improve efficiency significantly
                if large_eff > small_eff * 1.5:
                    recommendations.append({
                        'type': 'batch_size_optimization',
                        'severity': 'high',
                        'description': f"Efficiency scales well with batch size (×{large_eff/small_eff:.2f} improvement)",
                        'recommendation': f"Use larger batch sizes (optimal: {optimal_bs}) with gradient accumulation for better hardware utilization.",
                        'code_example': f"""
# Implement gradient accumulation to effectively use batch size {optimal_bs}
# while keeping memory requirements manageable
virtual_batch_size = {optimal_bs}
actual_batch_size = {max(1, optimal_bs // 4)}  # Adjust based on your memory constraints
accumulation_steps = virtual_batch_size // actual_batch_size

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps  # Scale loss
    
    # Backward pass
    loss.backward()
    
    # Update weights after accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
                    })
                    
                # If efficiency doesn't scale well with batch size
                elif large_eff < small_eff * 1.2 and large_bs > small_bs * 4:
                    recommendations.append({
                        'type': 'parallelism_strategy',
                        'severity': 'medium',
                        'description': "Efficiency doesn't scale well with batch size",
                        'recommendation': "Consider model parallelism instead of larger batches. Split model across multiple GPUs.",
                        'code_example': """
# Example of pipeline parallelism with PyTorch
from torch.distributed.pipeline.sync import Pipe

# Split model into stages
stage1 = nn.Sequential(model.layer1, model.layer2)
stage2 = nn.Sequential(model.layer3, model.layer4)

# Move stages to different devices
stage1 = stage1.to('cuda:0')
stage2 = stage2.to('cuda:1')

# Create pipeline
model = Pipe(nn.Sequential(stage1, stage2), chunks=8)
"""
                    })
        
        # Check if memory-bound or compute-bound profile changes with batch size
        if len(batch_sizes) >= 2:
            small_bs = batch_sizes[0]
            large_bs = batch_sizes[-1]
            
            if (isinstance(batch_results[small_bs], dict) and 
                isinstance(batch_results[large_bs], dict) and
                'memory_bound_ops' in batch_results[small_bs] and
                'compute_bound_ops' in batch_results[large_bs]):
                
                small_mem_bound = batch_results[small_bs]['memory_bound_ops']
                small_compute_bound = batch_results[small_bs]['compute_bound_ops']
                large_mem_bound = batch_results[large_bs]['memory_bound_ops']
                large_compute_bound = batch_results[large_bs]['compute_bound_ops']
                
                # If model shifts from memory-bound to compute-bound with larger batches
                if small_mem_bound > small_compute_bound and large_compute_bound > large_mem_bound:
                    recommendations.append({
                        'type': 'batch_optimization',
                        'severity': 'high',
                        'description': "Model shifts from memory-bound to compute-bound with larger batches",
                        'recommendation': f"Using larger batch sizes (optimal: {optimal_bs}) dramatically improves hardware utilization by amortizing memory access costs.",
                        'additional_info': {
                            'small_batch_profile': f"{small_bs}: {small_mem_bound} memory-bound, {small_compute_bound} compute-bound ops",
                            'large_batch_profile': f"{large_bs}: {large_mem_bound} memory-bound, {large_compute_bound} compute-bound ops"
                        }
                    })
                    
        return recommendations