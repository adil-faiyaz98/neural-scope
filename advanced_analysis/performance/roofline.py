"""
Roofline model analysis for ML models.

This module implements the roofline performance model to identify whether
operations are memory-bound or compute-bound, and provides targeted
optimization recommendations.
"""

import logging
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.autograd.profiler as profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib not available. Plotting features will be disabled.")

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
    ridge_point: float = field(default=0.0)  # Ridge point (compute intensity at peak performance)
    execution_time: float = field(default=0.0)  # Total execution time in ms
    bottleneck: str = field(default="")  # Primary bottleneck: "memory" or "compute"


class RooflineAnalyzer:
    """
    Analyze model computations using roofline model to identify 
    memory-bound vs. compute-bound operations
    """
    
    # GPU specifications database
    GPU_SPECS = {
        # NVIDIA GPUs
        "V100": {"peak_compute_tflops": 15.7, "peak_memory_bandwidth_gbps": 900, "architecture": "Volta"},
        "A100": {"peak_compute_tflops": 19.5, "peak_memory_bandwidth_gbps": 1555, "architecture": "Ampere"},
        "A100-80GB": {"peak_compute_tflops": 19.5, "peak_memory_bandwidth_gbps": 2039, "architecture": "Ampere"},
        "A10": {"peak_compute_tflops": 31.2, "peak_memory_bandwidth_gbps": 600, "architecture": "Ampere"},
        "A30": {"peak_compute_tflops": 10.3, "peak_memory_bandwidth_gbps": 933, "architecture": "Ampere"},
        "A40": {"peak_compute_tflops": 37.4, "peak_memory_bandwidth_gbps": 696, "architecture": "Ampere"},
        "T4": {"peak_compute_tflops": 8.1, "peak_memory_bandwidth_gbps": 320, "architecture": "Turing"},
        "RTX 2080 Ti": {"peak_compute_tflops": 13.4, "peak_memory_bandwidth_gbps": 616, "architecture": "Turing"},
        "RTX 3090": {"peak_compute_tflops": 35.6, "peak_memory_bandwidth_gbps": 936, "architecture": "Ampere"},
        "RTX 4090": {"peak_compute_tflops": 82.6, "peak_memory_bandwidth_gbps": 1008, "architecture": "Ada Lovelace"},
        "RTX 3080": {"peak_compute_tflops": 29.8, "peak_memory_bandwidth_gbps": 760, "architecture": "Ampere"},
        "RTX 4080": {"peak_compute_tflops": 48.7, "peak_memory_bandwidth_gbps": 717, "architecture": "Ada Lovelace"},
        "RTX 3070": {"peak_compute_tflops": 20.3, "peak_memory_bandwidth_gbps": 448, "architecture": "Ampere"},
        "RTX 4070": {"peak_compute_tflops": 29.1, "peak_memory_bandwidth_gbps": 504, "architecture": "Ada Lovelace"},
        "GTX 1080 Ti": {"peak_compute_tflops": 11.3, "peak_memory_bandwidth_gbps": 484, "architecture": "Pascal"},
        # AMD GPUs
        "MI100": {"peak_compute_tflops": 23.1, "peak_memory_bandwidth_gbps": 1229, "architecture": "CDNA"},
        "MI250X": {"peak_compute_tflops": 47.9, "peak_memory_bandwidth_gbps": 3277, "architecture": "CDNA2"},
    }
    
    def __init__(self, model, framework="pytorch", cache_path=None):
        self.model = model
        self.framework = framework.lower()
        self.device_info = self._get_device_info()
        self.cache_path = cache_path
        self.flops_estimates = {}
        self._load_operation_flops_estimates()
        
    def _load_operation_flops_estimates(self):
        """Load pre-computed FLOPS estimates for common operations"""
        # This would ideally be loaded from a comprehensive database
        # For demonstration, using common estimates
        self.flops_estimates = {
            "aten::conv2d": lambda input_shape, output_shape, kernel_size, groups: 
                2 * input_shape[1] * output_shape[1] * output_shape[2] * output_shape[3] * kernel_size[0] * kernel_size[1] / groups,
            "aten::linear": lambda input_features, output_features, batch_size: 
                2 * batch_size * input_features * output_features,
            "aten::batch_norm": lambda input_shape: 
                5 * input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3],
            "aten::layer_norm": lambda input_shape: 
                5 * np.prod(input_shape),
            "aten::relu": lambda elements: elements,
            "aten::gelu": lambda elements: 8 * elements,
            "aten::silu": lambda elements: 4 * elements,
            # Add more operations as needed
        }
    
    def _get_device_info(self):
        """Get hardware capability information"""
        device_info = {
            "peak_compute_tflops": 10.0,  # Default fallback
            "peak_memory_bandwidth_gbps": 500.0  # Default fallback
        }
        
        if self.framework == "pytorch" and TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            device_info["name"] = device_name
            device_info["compute_capability"] = torch.cuda.get_device_capability(device)
            
            # Try to match the GPU name with our database
            matched = False
            for gpu_name, specs in self.GPU_SPECS.items():
                if gpu_name in device_name:
                    device_info.update(specs)
                    matched = True
                    logger.info(f"Detected GPU: {device_name} - {specs['architecture']} architecture")
                    break
            
            if not matched:
                # Try to determine architecture and make a best guess
                if "RTX 40" in device_name:
                    device_info.update({
                        "peak_compute_tflops": 40.0,  # Conservative estimate
                        "peak_memory_bandwidth_gbps": 800.0,
                        "architecture": "Ada Lovelace"
                    })
                elif "RTX 30" in device_name:
                    device_info.update({
                        "peak_compute_tflops": 30.0,  # Conservative estimate
                        "peak_memory_bandwidth_gbps": 700.0,
                        "architecture": "Ampere"
                    })
                elif "RTX 20" in device_name:
                    device_info.update({
                        "peak_compute_tflops": 15.0,  # Conservative estimate
                        "peak_memory_bandwidth_gbps": 550.0,
                        "architecture": "Turing"
                    })
                
                logger.warning(f"GPU {device_name} not found in database. Using estimated values.")
            
            # Try to dynamically measure memory bandwidth
            try:
                bandwidth = self._measure_memory_bandwidth()
                if bandwidth > 0:
                    device_info["measured_memory_bandwidth_gbps"] = bandwidth
                    logger.info(f"Measured memory bandwidth: {bandwidth:.2f} GB/s")
            except Exception as e:
                logger.warning(f"Failed to measure memory bandwidth: {e}")
        
        return device_info
    
    def _measure_memory_bandwidth(self, size_mb=1000):
        """Measure actual memory bandwidth using PyTorch tensors"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        # Allocate tensors
        size = size_mb * 1024 * 1024 // 4  # for float32
        x = torch.empty(size, dtype=torch.float32, device="cuda")
        y = torch.empty(size, dtype=torch.float32, device="cuda")
        
        # Warmup
        y.copy_(x)
        torch.cuda.synchronize()
        
        # Measure
        iterations = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            y.copy_(x)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000  # seconds
        
        # Calculate bandwidth: read + write in GB/s
        bytes_transferred = 2 * size * 4 * iterations  # 2 for read+write, 4 bytes per float
        bandwidth = bytes_transferred / elapsed_time / 1e9
        
        return bandwidth
        
    def analyze(self, input_data, use_cache=False) -> RooflineResult:
        """
        Analyze model using roofline model with the given input data
        
        Args:
            input_data: Input data for the model
            use_cache: Whether to use cached results if available
            
        Returns:
            RooflineResult with detailed analysis
        """
        if self.framework == "pytorch":
            return self._analyze_pytorch(input_data, use_cache)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _estimate_flops_from_kernel(self, event):
        """Estimate FLOPS for a kernel based on its name and shapes"""
        kernel_name = event.name
        flops = 0
        
        # Extract shapes and parameters from kernel name
        # This is a simplified approach - a real implementation would need
        # to parse the kernel information more carefully
        if kernel_name in self.flops_estimates:
            # This would need to extract shapes from the profiling event
            # Here we're just using placeholder estimates
            if "conv2d" in kernel_name:
                # Estimate for conv2d
                input_shape = [1, 64, 32, 32]  # Example shape
                output_shape = [1, 128, 16, 16]
                kernel_size = [3, 3]
                groups = 1
                flops = self.flops_estimates[kernel_name](input_shape, output_shape, kernel_size, groups)
            elif "linear" in kernel_name:
                # Estimate for linear layer
                batch_size = 32
                input_features = 512
                output_features = 1024
                flops = self.flops_estimates[kernel_name](input_features, output_features, batch_size)
            else:
                # Generic estimate
                flops = 1e9  # Placeholder
        
        return flops
            
    def _analyze_pytorch(self, input_data, use_cache=False) -> RooflineResult:
        """Analyze PyTorch model using roofline model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorch model analysis")
        
        cache_file = None
        if self.cache_path and use_cache:
            model_name = type(self.model).__name__
            cache_file = Path(self.cache_path) / f"roofline_{model_name}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        
                    # Convert cached data to RooflineResult
                    operations = [ComputeMetrics(**op) for op in cache_data["operations"]]
                    result = RooflineResult(
                        operations=operations,
                        peak_compute=cache_data["peak_compute"],
                        peak_memory_bandwidth=cache_data["peak_memory_bandwidth"],
                        memory_bound_ops=cache_data["memory_bound_ops"],
                        compute_bound_ops=cache_data["compute_bound_ops"],
                        recommendations=cache_data["recommendations"],
                        ridge_point=cache_data["ridge_point"],
                        execution_time=cache_data["execution_time"],
                        bottleneck=cache_data["bottleneck"]
                    )
                    logger.info(f"Loaded cached roofline analysis from {cache_file}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cached results: {e}")
        
        # Actual profiling using PyTorch profiler
        operations = []
        
        model_device = next(self.model.parameters()).device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(model_device)
        elif isinstance(input_data, (list, tuple)):
            input_data = [x.to(model_device) if isinstance(x, torch.Tensor) else x for x in input_data]
        
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Do a warm-up run
            self.model(input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Actual profiling run
            with profiler.profile(record_shapes=True, with_flops=True, profile_memory=True) as prof:
                output = self.model(input_data)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Process profiler results
        total_execution_time = 0
        for event in prof.key_averages():
            # Skip small or irrelevant kernels
            if event.self_cpu_time_total < 1:
                continue
                
            kernel_name = event.name
            
            # Estimate FLOPS and memory accesses
            if hasattr(event, 'flops') and event.flops > 0:
                flops = event.flops
            else:
                flops = self._estimate_flops_from_kernel(event)
            
            # Estimate bytes accessed
            bytes_accessed = 0
            if hasattr(event, 'input_shapes') and event.input_shapes:
                for shape in event.input_shapes:
                    if shape and all(dim > 0 for dim in shape):
                        bytes_accessed += 4 * np.prod(shape)  # Assuming float32
            
            # If we still can't estimate bytes, use a heuristic based on CPU time
            if bytes_accessed == 0:
                bytes_accessed = event.self_cpu_time_total * 1e-3  # Extremely rough approximation
            
            # Calculate compute intensity (FLOPS/Byte)
            compute_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
            
            # Calculate theoretical peak performance (considering mixed precision if relevant)
            theoretical_peak = self.device_info.get("peak_compute_tflops", 10) * 1e12
            
            # Calculate achieved performance
            execution_time_s = event.self_cpu_time_total * 1e-6  # Convert microseconds to seconds
            achieved_performance = flops / execution_time_s if execution_time_s > 0 else 0
            
            # Calculate efficiency
            efficiency = achieved_performance / theoretical_peak if theoretical_peak > 0 else 0
            
            operations.append(ComputeMetrics(
                flops=flops,
                bytes_accessed=bytes_accessed,
                compute_intensity=compute_intensity,
                theoretical_peak=theoretical_peak,
                achieved_performance=achieved_performance,
                efficiency=efficiency,
                kernel_name=kernel_name
            ))
            
            total_execution_time += event.self_cpu_time_total
        
        # Determine if operations are memory-bound or compute-bound
        peak_compute = self.device_info.get("peak_compute_tflops", 10)  # TFLOPS
        peak_memory_bandwidth = self.device_info.get("peak_memory_bandwidth_gbps", 500)  # GB/s
        
        ridge_point = peak_compute * 1e12 / (peak_memory_bandwidth * 1e9)  # FLOPS/Byte
        
        memory_bound_ops = []
        compute_bound_ops = []
        
        for op in operations:
            if op.compute_intensity < ridge_point:
                memory_bound_ops.append(op.kernel_name)
            else:
                compute_bound_ops.append(op.kernel_name)
        
        # Generate recommendations
        recommendations = []
        
        if len(memory_bound_ops) > len(compute_bound_ops):
            bottleneck = "memory"
            recommendations.append({
                "type": "memory_bound",
                "operations": memory_bound_ops[:5],  # Top 5 memory-bound operations
                "suggestion": "Your model is primarily memory-bound. Consider these optimizations:",
                "optimizations": [
                    "Reduce redundant memory accesses and increase data reuse",
                    "Use memory-efficient operators or fuse operators where possible",
                    "Consider using torch.utils.checkpoint to trade computation for memory",
                    "Optimize data layouts for better memory coalescing",
                    f"For {self.device_info.get('architecture', 'your GPU')}, try to batch matmul operations"
                ],
                "code_example": "# Memory optimization example:\n"
                                "from torch.utils.checkpoint import checkpoint\n\n"
                                "# Replace this:\n"
                                "# output = model(input_data)\n\n"
                                "# With this:\n"
                                "# def custom_forward(x):\n"
                                "#     return model(x)\n"
                                "# output = checkpoint(custom_forward, input_data)"
            })
        else:
            bottleneck = "compute"
            recommendations.append({
                "type": "compute_bound",
                "operations": compute_bound_ops[:5],  # Top 5 compute-bound operations
                "suggestion": "Your model is primarily compute-bound. Consider these optimizations:",
                "optimizations": [
                    "Use optimized implementations (e.g., cuDNN for convolutions)",
                    "Consider mixed precision training with torch.cuda.amp",
                    "Explore quantization to reduce precision requirements",
                    "Optimize operator selection and model architecture",
                    f"For {self.device_info.get('architecture', 'your GPU')}, leverage Tensor Cores with proper dimensions"
                ],
                "code_example": "# Mixed precision example:\n"
                                "from torch.cuda.amp import autocast, GradScaler\n\n"
                                "scaler = GradScaler()\n\n"
                                "# Training loop\n"
                                "with autocast():\n"
                                "    outputs = model(inputs)\n"
                                "    loss = loss_fn(outputs, targets)\n\n"
                                "scaler.scale(loss).backward()\n"
                                "scaler.step(optimizer)\n"
                                "scaler.update()"
            })
        
        # Add hardware-specific recommendations
        if "architecture" in self.device_info:
            arch = self.device_info["architecture"]
            if arch in ["Ampere", "Ada Lovelace"]:
                recommendations.append({
                    "type": "hardware_specific",
                    "suggestion": f"Optimization for {arch} architecture:",
                    "optimizations": [
                        "Use TF32 for faster matrix multiplications",
                        "Ensure tensor dimensions are multiples of 16 for best Tensor Core utilization",
                        "Leverage Sparse Tensor Cores for applicable workloads"
                    ],
                    "code_example": "# Enable TF32 (on by default in PyTorch 1.7+)\n"
                                    "torch.backends.cuda.matmul.allow_tf32 = True\n"
                                    "torch.backends.cudnn.allow_tf32 = True"
                })
            elif arch == "Turing":
                recommendations.append({
                    "type": "hardware_specific",
                    "suggestion": "Optimization for Turing architecture:",
                    "optimizations": [
                        "Ensure tensor dimensions are multiples of 16 for INT8 Tensor Cores",
                        "Consider INT8 quantization for inference workloads"
                    ],
                    "code_example": "# Quantization example\n"
                                    "model_fp32 = model\n"
                                    "model_int8 = torch.quantization.quantize_dynamic(\n"
                                    "    model_fp32,  # the original model\n"
                                    "    {torch.nn.Linear},  # a set of layers to dynamically quantize\n"
                                    "    dtype=torch.qint8)  # the target dtype for quantized weights"
                })
        
        result = RooflineResult(
            operations=operations,
            peak_compute=peak_compute,
            peak_memory_bandwidth=peak_memory_bandwidth,
            memory_bound_ops=memory_bound_ops,
            compute_bound_ops=compute_bound_ops,
            recommendations=recommendations,
            ridge_point=ridge_point,
            execution_time=total_execution_time * 1e-6,  # Convert microseconds to seconds
            bottleneck=bottleneck
        )
        
        # Cache the results if a cache path is provided
        if self.cache_path and cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump({
                        "operations": [
                            {
                                "flops": op.flops,
                                "bytes_accessed": op.bytes_accessed,
                                "compute_intensity": op.compute_intensity,
                                "theoretical_peak": op.theoretical_peak,
                                "achieved_performance": op.achieved_performance,
                                "efficiency": op.efficiency,
                                "kernel_name": op.kernel_name
                            } for op in operations
                        ],
                        "peak_compute": peak_compute,
                        "peak_memory_bandwidth": peak_memory_bandwidth,
                        "memory_bound_ops": memory_bound_ops,
                        "compute_bound_ops": compute_bound_ops,
                        "recommendations": recommendations,
                        "ridge_point": ridge_point,
                        "execution_time": total_execution_time * 1e-6,
                        "bottleneck": bottleneck
                    }, f, indent=2)
                logger.info(f"Saved roofline analysis to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache results: {e}")
        
        return result

    def plot_roofline(self, result: RooflineResult, save_path=None, show=True):
        """
        Generate a roofline plot from the analysis results
        
        Args:
            result: RooflineResult from analyze()
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object if matplotlib is available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib is required for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute intensity range for plot
        min_ci = min([op.compute_intensity for op in result.operations if op.compute_intensity > 0], default=0.1)
        max_ci = max([op.compute_intensity for op in result.operations], default=100)
        
        # Ensure reasonable range
        min_ci = max(0.01, min_ci / 10)
        max_ci = min(1000, max_ci * 10)
        
        # Generate x values for plot
        x = np.logspace(np.log10(min_ci), np.log10(max_ci), 1000)
        
        # Memory bandwidth line
        y_memory = result.peak_memory_bandwidth * 1e9 * x
        
        # Peak compute line
        y_compute = np.ones_like(x) * result.peak_compute * 1e12
        
        # Plot the roof
        ax.loglog(x, np.minimum(y_memory, y_compute), 'k-', linewidth=2, label='Roofline')
        
        # Add ridge point
        ax.scatter([result.ridge_point], [result.peak_compute * 1e12], 
                  marker='o', s=80, color='black', zorder=5)
        ax.text(result.ridge_point * 1.1, result.peak_compute * 1e12 * 0.9, 
                f'Ridge Point: {result.ridge_point:.1f} FLOPS/Byte', 
                fontsize=10, ha='left')
        
        # Plot operations
        colors = plt.cm.tab10(np.linspace(0, 1, len(result.operations)))
        for i, op in enumerate(result.operations):
            if op.compute_intensity <= 0 or op.achieved_performance <= 0:
                continue
                
            # Color by bound type
            color = 'red' if op.kernel_name in result.memory_bound_ops else 'blue'
            
            ax.scatter(op.compute_intensity, op.achieved_performance, s=100, 
                      alpha=0.7, color=color, edgecolors='black', zorder=4)
            
            # Add labels for significant operations
            if op.efficiency > 0.05:  # Only label significant operations
                ax.text(op.compute_intensity * 1.1, op.achieved_performance * 1.1,
                        op.kernel_name.split('::')[-1],
                        fontsize=8, ha='left')
        
        # Add legends
        ax.scatter([], [], s=100, color='red', edgecolors='black', label='Memory Bound')
        ax.scatter([], [], s=100, color='blue', edgecolors='black', label='Compute Bound')
        
        # Labels and title
        ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)')
        ax.set_ylabel('Performance (FLOPS)')
        ax.set_title('Roofline Model Analysis')
        
        # Add ridge point line
        ax.axvline(x=result.ridge_point, color='gray', linestyle='--', alpha=0.7)
        
        # Grid and legend
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Device information
        device_text = f"Device: {self.device_info.get('name', 'Unknown')}\n"
        device_text += f"Peak Compute: {result.peak_compute:.1f} TFLOPS\n"
        device_text += f"Peak Memory Bandwidth: {result.peak_memory_bandwidth:.1f} GB/s\n"
        device_text += f"Primary Bottleneck: {result.bottleneck.capitalize()}"
        
        ax.text(0.02, 0.02, device_text, transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Roofline plot saved to {save_path}")
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig
