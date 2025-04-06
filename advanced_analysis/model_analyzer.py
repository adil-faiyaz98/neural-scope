"""
Model analyzer for deep learning architectures.

This module provides tools for analyzing and benchmarking deep learning models,
including parameter counting, FLOPs estimation, memory usage, inference latency,
and architecture pattern detection.
"""

import time
import logging
import os
import json
import platform
import tempfile
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX and/or ONNX Runtime not available. ONNX-related features disabled.")

try:
    from onnxruntime_tools import optimizer
    ONNXRUNTIME_TOOLS_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_TOOLS_AVAILABLE = False
    logger.warning("onnxruntime-tools not available. Advanced ONNX optimization features disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Some tabular data features will be limited.")

# Check for visualization dependencies
try:
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. Interactive HTML visualizations will be disabled.")

# For CLI interface
try:
    import argparse
    import importlib
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    logger.warning("argparse not available. CLI features will be disabled.")

class ModelAnalyzer:
    """
    Comprehensive analyzer for deep learning model architecture, performance, and efficiency.
    """
    
    def __init__(self, device=None, enable_html_reports=True):
        """
        Initialize ModelAnalyzer with optional configuration.
        
        Args:
            device: Device to run analysis on (uses best available if None)
            enable_html_reports: Whether to enable HTML report generation (requires plotly)
        """
        # Set device
        self.device = device if device is not None else \
                    "cuda" if torch.cuda.is_available() else "cpu"
        
        # Enable/disable HTML reports based on availability and user preference
        self.enable_html_reports = enable_html_reports and PLOTLY_AVAILABLE
        
        # Store analysis results
        self.last_analysis = None
        
    def _count_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """
        Count trainable and non-trainable parameters in the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter statistics
        """
        trainable_params = 0
        non_trainable_params = 0
        layer_params = {}
        layer_types = Counter()
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            if param.requires_grad:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
            
            # Get module name without trailing parameter type (weight/bias)
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            if module_name not in layer_params:
                layer_params[module_name] = {"total": 0, "trainable": 0, "non_trainable": 0}
            
            layer_params[module_name]["total"] += param_count
            if param.requires_grad:
                layer_params[module_name]["trainable"] += param_count
            else:
                layer_params[module_name]["non_trainable"] += param_count
        
        # Count layer types
        for name, module in model.named_modules():
            if list(module.children()):  # Skip containers
                continue
            
            layer_type = module.__class__.__name__
            layer_types[layer_type] += 1
        
        return {
            "total": trainable_params + non_trainable_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
            "layer_parameters": layer_params,
            "layer_types": dict(layer_types)
        }
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple) -> Dict[str, Any]:
        """
        Estimate FLOPs for the model using a rough heuristic approach.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with FLOP estimates
        """
        # Rough FLOP estimation without dependencies
        # In practice, use a library like fvcore, thop, or ptflops
        flops_by_layer = {}
        total_flops = 0
        
        # Create sample input
        sample_input = torch.randn(input_shape).to(self.device)
        
        # Register hooks for each layer
        handles = []
        
        def conv_flop_hook(module, input, output):
            # FLOPs = 2 * C_in * kernel_size^2 * C_out * H_out * W_out
            input_shape = input[0].shape
            output_shape = output.shape
            batch_size, in_channels = input_shape[0], input_shape[1]
            out_channels, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]
            kernel_h, kernel_w = module.kernel_size
            flops = 2 * in_channels * kernel_h * kernel_w * out_channels * out_h * out_w * batch_size
            flops_by_layer[module.__class__.__name__] = flops
            return flops
            
        def linear_flop_hook(module, input, output):
            # FLOPs = 2 * in_features * out_features * batch_size
            input_shape = input[0].shape
            batch_size = input_shape[0]
            flops = 2 * module.in_features * module.out_features * batch_size
            flops_by_layer[module.__class__.__name__] = flops
            return flops
            
        # Attach hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_flop_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_flop_hook))
        
        # Forward pass
        with torch.no_grad():
            model(sample_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        total_flops = sum(flops_by_layer.values())
        return {
            "total": total_flops,
            "by_layer": flops_by_layer,
            "gflops": total_flops / 10**9  # Convert to GFLOPs
        }
    
    def _estimate_memory(self, model: nn.Module, input_shape: Tuple) -> Dict[str, Any]:
        """
        Estimate memory usage for the model.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with memory estimates
        """
        # Initialize counters
        param_memory = 0
        buffer_memory = 0
        activation_memory = 0
        
        # Parameter memory
        for param in model.parameters():
            param_memory += param.nelement() * param.element_size()
        
        # Buffers memory (e.g., running stats in BatchNorm)
        for buffer in model.buffers():
            buffer_memory += buffer.nelement() * buffer.element_size()
        
        # Create sample input
        sample_input = torch.randn(input_shape).to(self.device)
        
        # Forward pass to estimate activation memory (rough estimate)
        with torch.no_grad():
            try:
                # Estimate output size
                output = model(sample_input)
                if isinstance(output, torch.Tensor):
                    activation_memory += output.nelement() * output.element_size()
                elif isinstance(output, (list, tuple)):
                    for tensor in output:
                        if isinstance(tensor, torch.Tensor):
                            activation_memory += tensor.nelement() * tensor.element_size()
            except Exception as e:
                logger.warning(f"Error estimating activation memory: {e}")
        
        total_memory = param_memory + buffer_memory + activation_memory
        
        return {
            "total_bytes": total_memory,
            "total_mb": total_memory / (1024 * 1024),  # Convert to MB
            "parameters_mb": param_memory / (1024 * 1024),
            "buffers_mb": buffer_memory / (1024 * 1024),
            "activations_mb": activation_memory / (1024 * 1024)
        }
    
    def benchmark_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, repeat: int = 10) -> Dict[str, float]:
        """
        Time multiple forward passes and report average latency.
        
        Args:
            model: PyTorch model to benchmark
            input_tensor: Input tensor with correct shape
            repeat: Number of iterations to run
            
        Returns:
            Dictionary with inference time benchmarks
        """
        if self.device.startswith("cuda"):
            # Ensure model is on correct device
            model.to(self.device)
            input_tensor = input_tensor.to(self.device)
            
            # Warmup pass
            with torch.no_grad():
                _ = model(input_tensor)
            
            # Synchronize GPU before timing
            torch.cuda.synchronize()
            
            # Run benchmark
            start_time = time.time()
            
            for _ in range(repeat):
                with torch.no_grad():
                    _ = model(input_tensor)
                torch.cuda.synchronize()  # Ensure GPU execution is complete
                
            end_time = time.time()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            avg_latency = elapsed_time / repeat
            fps = 1000 / avg_latency * input_tensor.size(0)  # Images per second
            
        else:  # CPU timing
            # Ensure model is on correct device
            model.to(self.device)
            input_tensor = input_tensor.to(self.device)
            
            # Warmup pass
            with torch.no_grad():
                _ = model(input_tensor)
            
            # Run benchmark
            start_time = time.time()
            
            for _ in range(repeat):
                with torch.no_grad():
                    _ = model(input_tensor)
                
            end_time = time.time()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            avg_latency = elapsed_time / repeat
            fps = 1000 / avg_latency * input_tensor.size(0)  # Images per second
        
        return {
            "avg_latency_ms": avg_latency,
            "throughput_fps": fps,
            "batch_size": input_tensor.size(0),
            "total_time_ms": elapsed_time,
            "iterations": repeat
        }
    
    def export_to_onnx(self, model: nn.Module, input_tensor: torch.Tensor, 
                     path: Optional[str] = None, optimize: bool = True) -> Optional[str]:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_tensor: Example input tensor
            path: Path to save the ONNX model (if None, uses a temp file)
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Path to the exported ONNX model or None if export failed
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX export requested but ONNX is not available")
            return None
            
        try:
            # Use a temporary file if no path is provided
            if path is None:
                fd, path = tempfile.mkstemp(suffix='.onnx')
                os.close(fd)
            
            # Ensure model and input are on the correct device
            model.to(self.device)
            input_tensor = input_tensor.to(self.device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    input_tensor,
                    path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}}
                )
            
            # Verify ONNX model
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            
            # Optimize model if requested and tools are available
            if optimize and ONNXRUNTIME_TOOLS_AVAILABLE:
                optimized_path = path.replace('.onnx', '_optimized.onnx')
                optimized_model = optimizer.optimize_model(
                    path,
                    model_type='bert',  # generic optimization
                    use_gpu=self.device.startswith("cuda"),
                    opt_level=99
                )
                optimized_model.save_model_to_file(optimized_path)
                return optimized_path
                
            return path
            
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            return None
    
    def estimate_onnx_flops(self, onnx_path: str) -> Dict[str, Any]:
        """
        Estimate FLOPs for an ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            
        Returns:
            Dictionary with FLOP estimates
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX profiling requested but ONNX is not available")
            return {"error": "ONNX not available"}
            
        try:
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Initialize results
            flops_by_op = defaultdict(int)
            total_flops = 0
            
            # Map ops to their computational cost
            flop_mapping = {
                "Conv": lambda node: self._estimate_conv_flops(node),
                "Gemm": lambda node: self._estimate_gemm_flops(node),
                "MatMul": lambda node: self._estimate_matmul_flops(node)
            }
            
            # Process the graph
            for node in model.graph.node:
                op_type = node.op_type
                if op_type in flop_mapping:
                    flops = flop_mapping[op_type](node)
                    flops_by_op[op_type] += flops
                    total_flops += flops
            
            return {
                "total": total_flops,
                "by_op_type": dict(flops_by_op),
                "gflops": total_flops / 10**9
            }
            
        except Exception as e:
            logger.error(f"Error estimating ONNX FLOPs: {e}")
            return {"error": str(e)}
    
    def _estimate_conv_flops(self, node) -> int:
        """Helper to estimate FLOPs for Conv operations in ONNX"""
        # This is a simplified example, a real impl would extract attributes
        return 0  # Placeholder
    
    def _estimate_gemm_flops(self, node) -> int:
        """Helper to estimate FLOPs for Gemm operations in ONNX"""
        return 0  # Placeholder
    
    def _estimate_matmul_flops(self, node) -> int:
        """Helper to estimate FLOPs for MatMul operations in ONNX"""
        return 0  # Placeholder
    
    def _detect_architecture_type(self, model: nn.Module) -> Dict[str, Any]:
        """
        Detect the model architecture type based on layer patterns.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with architecture classification
        """
        # Track layer types
        has_conv = False
        has_linear = False
        has_attention = False
        has_embedding = False
        has_rnn = False
        has_transformer = False
        has_positional_encoding = False
        
        # Check all modules
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            
            # Check for CNN components
            if isinstance(module, nn.Conv2d) or "Conv2d" in module_type:
                has_conv = True
                
            # Check for Linear/MLP components
            if isinstance(module, nn.Linear) or "Linear" in module_type:
                has_linear = True
                
            # Check for attention mechanisms
            if isinstance(module, nn.MultiheadAttention) or "Attention" in module_type:
                has_attention = True
                
            # Check for embeddings
            if isinstance(module, nn.Embedding) or "Embedding" in module_type:
                has_embedding = True
                
            # Check for RNN components
            if any(rnn_type in module_type for rnn_type in ["RNN", "LSTM", "GRU"]):
                has_rnn = True
                
            # Check for transformer components 
            if any(tf_component in module_type for tf_component in ["Transformer", "SelfAttention"]):
                has_transformer = True
                
            # Check for positional encoding
            if "Position" in module_type:
                has_positional_encoding = True
        
        # Determine architecture type
        architecture_type = "Unknown"
        confidence = 0.0
        
        if has_conv and not has_attention:
            architecture_type = "CNN"
            confidence = 0.9 if has_conv and not has_embedding and not has_rnn else 0.7
            
        elif has_transformer or (has_attention and has_positional_encoding):
            architecture_type = "Transformer"
            confidence = 0.9
            
        elif has_embedding and has_rnn:
            architecture_type = "RNN/LSTM"
            confidence = 0.8
            
        elif has_linear and not has_conv and not has_rnn and not has_attention:
            architecture_type = "MLP"
            confidence = 0.8
        
        return {
            "type": architecture_type,
            "confidence": confidence,
            "components": {
                "has_convolution": has_conv,
                "has_linear": has_linear,
                "has_attention": has_attention,
                "has_embedding": has_embedding,
                "has_rnn": has_rnn,
                "has_transformer": has_transformer,
                "has_positional_encoding": has_positional_encoding
            }
        }
    
    def estimate_max_batch_size(self, model: nn.Module, base_input: torch.Tensor, 
                              max_tries: int = 10) -> Dict[str, Any]:
        """
        Estimate the maximum batch size the model can handle on current device.
        
        Args:
            model: PyTorch model
            base_input: Base input tensor (batch size 1)
            max_tries: Maximum number of batch size attempts
            
        Returns:
            Dictionary with batch size scaling results
        """
        if not self.device.startswith("cuda"):
            return {"error": "Batch size estimation requires CUDA device"}
            
        # Ensure model is on correct device
        model.to(self.device)
        
        # Ensure base input has batch dimension of 1
        if base_input.size(0) != 1:
            logger.warning(f"Expected base_input with batch size 1, got {base_input.size(0)}")
            base_input = base_input[:1]
        
        base_input = base_input.to(self.device)
        
        # Try increasingly large batch sizes
        current_batch_size = 1
        max_batch_size = 1
        latencies = {}
        
        torch.cuda.empty_cache()
        
        for i in range(max_tries):
            batch_size = 2**i  # 1, 2, 4, 8, 16, ...
            try:
                # Create input with larger batch size
                batched_input = base_input.repeat(batch_size, *[1 for _ in range(base_input.dim()-1)])
                
                # Time inference
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    _ = model(batched_input)
                end.record()
                
                torch.cuda.synchronize()
                latency_ms = start.elapsed_time(end)
                
                # Store latency
                latencies[batch_size] = latency_ms
                
                # Update max working batch size
                max_batch_size = batch_size
                
                # Check for significant slowdown
                if batch_size > 1 and latencies[batch_size] > 3 * (latencies[batch_size//2] * 2):
                    logger.info(f"Significant slowdown detected at batch size {batch_size}")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"OOM at batch size {batch_size}")
                    break
                else:
                    logger.error(f"Error at batch size {batch_size}: {e}")
                    break
        
        # Calculate optimal batch size (max before significant slowdown)
        optimal_batch_size = max_batch_size
        
        return {
            "max_batch_size": max_batch_size,
            "optimal_batch_size": optimal_batch_size,
            "latencies": latencies
        }
    
    def _check_model_compliance(self, model: nn.Module) -> Dict[str, Any]:
        """
        Check model for common compliance issues.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with compliance check results
        """
        issues = []
        warnings = []
        
        # Check for training mode
        if model.training:
            issues.append("Model is in training mode, should be in eval mode for inference")
            
        # Check for dropout in eval mode
        dropout_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and module.p > 0:
                dropout_layers.append(name)
                
        if dropout_layers and not model.training:
            warnings.append(f"Model has {len(dropout_layers)} dropout layers while in eval mode")
            
        # Check if BatchNorm layers are frozen
        unfrozen_bn = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.training:
                    unfrozen_bn.append(name)
                    
        if unfrozen_bn:
            issues.append(f"Found {len(unfrozen_bn)} unfrozen BatchNorm layers")
            
        # Check for device consistency
        device_count = defaultdict(int)
        for name, param in model.named_parameters():
            device_count[str(param.device)] += 1
            
        if len(device_count) > 1:
            warnings.append(f"Model has parameters on different devices: {dict(device_count)}")
            
        # Check for uninitialized weights
        uninit_params = []
        for name, param in model.named_parameters():
            # Check for common signs of uninitialized weights
            if torch.isnan(param).any():
                uninit_params.append(name)
                
        if uninit_params:
            issues.append(f"Found {len(uninit_params)} parameters with NaN values")
            
        return {
            "issues": issues,
            "warnings": warnings,
            "is_compliant": len(issues) == 0
        }
    
    def calculate_cloud_edge_score(self, model: nn.Module, input_shape: Tuple) -> Dict[str, Any]:
        """
        Calculate a score for cloud/edge deployment readiness.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with readiness scores
        """
        scores = {}
        
        # Memory footprint score (0-25)
        memory_estimate = self._estimate_memory(model, input_shape)
        memory_mb = memory_estimate["total_mb"]
        
        if memory_mb < 10:
            memory_score = 25
        elif memory_mb < 50:
            memory_score = 20
        elif memory_mb < 100:
            memory_score = 15
        elif memory_mb < 500:
            memory_score = 10
        elif memory_mb < 1000:
            memory_score = 5
        else:
            memory_score = 0
            
        scores["memory_footprint"] = memory_score
        
        # Quantization readiness score (0-25)
        quant_score = 25  # Start with perfect score
        
        # Check for layer types that might have quantization issues
        for name, module in model.named_modules():
            if "custom" in name.lower() or "plugin" in name.lower():
                quant_score -= 5  # Custom ops might not quantize well
            
        scores["quantization_readiness"] = max(0, quant_score)
        
        # ONNX exportability score (0-25)
        onnx_score = 25 if ONNX_AVAILABLE else 0
        
        if ONNX_AVAILABLE:
            try:
                # Try to export to ONNX
                sample_input = torch.randn(input_shape).to(self.device)
                onnx_path = self.export_to_onnx(model, sample_input, optimize=False)
                if onnx_path is None:
                    onnx_score = 0
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
                onnx_score = 0
                
        scores["onnx_exportability"] = onnx_score
        
        # Batch size scaling score (0-25)
        if self.device.startswith("cuda"):
            try:
                sample_input = torch.randn(input_shape).to(self.device)
                batch_result = self.estimate_max_batch_size(model, sample_input[:1], max_tries=5)
                
                max_batch = batch_result.get("max_batch_size", 1)
                
                if max_batch >= 64:
                    batch_score = 25
                elif max_batch >= 32:
                    batch_score = 20
                elif max_batch >= 16:
                    batch_score = 15
                elif max_batch >= 8:
                    batch_score = 10
                elif max_batch >= 4:
                    batch_score = 5
                else:
                    batch_score = 0
            except Exception as e:
                logger.warning(f"Batch size estimation failed: {e}")
                batch_score = 10  # Default score
        else:
            batch_score = 10  # Default if not on CUDA
            
        scores["batch_size_scaling"] = batch_score
        
        # Calculate total score
        total_score = sum(scores.values())
        
        return {
            "total_score": total_score,
            "normalized_score": total_score / 100,
            "component_scores": scores,
            "interpretation": self._interpret_edge_score(total_score)
        }
    
    def _interpret_edge_score(self, score: float) -> str:
        """Helper to interpret edge readiness score"""
        if score >= 90:
            return "Excellent - Ready for edge deployment"
        elif score >= 75:
            return "Good - Should work well on edge devices"
        elif score >= 50:
            return "Moderate - May need optimization for edge"
        elif score >= 25:
            return "Poor - Consider model compression or architecture changes"
        else:
            return "Not suitable for edge - Use cloud deployment"
    
    def generate_html_report(self, analysis: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate an HTML report with interactive visualizations.
        
        Args:
            analysis: Model analysis results dictionary
            output_path: Path to save the HTML report
            
        Returns:
            Path to the HTML report or None if generation failed
        """
        if not self.enable_html_reports:
            logger.warning("HTML report generation is disabled or plotly is not available")
            return None
            
        try:
            # Create a plotly figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Parameter Count by Layer Type", 
                    "Memory Usage Breakdown",
                    "Layer-wise FLOPs",
                    "Inference Latency vs Batch Size",
                    "Architecture Detection",
                    "Cloud/Edge Readiness"
                ),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                      [{"type": "bar"}, {"type": "scatter"}],
                      [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # 1. Parameter Count by Layer Type
            if "parameters" in analysis:
                layer_types = analysis["parameters"].get("layer_types", {})
                if layer_types:
                    fig.add_trace(
                        go.Bar(
                            x=list(layer_types.keys()),
                            y=list(layer_types.values()),
                            name="Layer Count"
                        ),
                        row=1, col=1
                    )
            
            # 2. Memory Usage Breakdown
            if "memory" in analysis:
                memory = analysis["memory"]
                fig.add_trace(
                    go.Pie(
                        labels=["Parameters", "Buffers", "Activations"],
                        values=[
                            memory.get("parameters_mb", 0),
                            memory.get("buffers_mb", 0),
                            memory.get("activations_mb", 0)
                        ],
                        name="Memory Usage"
                    ),
                    row=1, col=2
                )
            
            # 3. Layer-wise FLOPs
            if "flops" in analysis and "by_layer" in analysis["flops"]:
                flops_by_layer = analysis["flops"]["by_layer"]
                fig.add_trace(
                    go.Bar(
                        x=list(flops_by_layer.keys()),
                        y=list(flops_by_layer.values()),
                        name="FLOPs"
                    ),
                    row=2, col=1
                )
            
            # 4. Inference Latency vs Batch Size
            if "performance" in analysis and "batch_size_scaling" in analysis["performance"]:
                batch_sizes = list(analysis["performance"]["batch_size_scaling"].get("latencies", {}).keys())
                latencies = list(analysis["performance"]["batch_size_scaling"].get("latencies", {}).values())
                
                if batch_sizes and latencies:
                    fig.add_trace(
                        go.Scatter(
                            x=batch_sizes,
                            y=latencies,
                            mode="lines+markers",
                            name="Inference Latency"
                        ),
                        row=2, col=2
                    )
                    
                    fig.update_xaxes(title_text="Batch Size", row=2, col=2)
                    fig.update_yaxes(title_text="Latency (ms)", row=2, col=2)
            
            # 5. Architecture Detection
            if "architecture" in analysis:
                arch = analysis["architecture"]
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=arch.get("confidence", 0) * 100,
                        title={"text": f"Architecture Type: {arch.get('type', 'Unknown')}"},
                        gauge={"axis": {"range": [0, 100]}}
                    ),
                    row=3, col=1
                )
            
            # 6. Cloud/Edge Readiness
            if "cloud_edge_readiness" in analysis:
                readiness = analysis["cloud_edge_readiness"]
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=readiness.get("normalized_score", 0) * 100,
                        title={"text": "Edge Deployment Readiness"},
                        gauge={"axis": {"range": [0, 100]}}
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1000,
                title_text=f"Model Analysis Report: {analysis.get('model_name', 'Unknown Model')}",
                showlegend=False
            )
            
            # Generate HTML
            html_content = fig.to_html(include_plotlyjs=True, full_html=True)
            
            # Add metadata section
            metadata_html = f"""
            <div style="margin-top: 50px; padding: 20px; background-color: #f5f5f5;">
                <h2>Analysis Metadata</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Date:</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{analysis.get('metadata', {}).get('date', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Device:</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{analysis.get('metadata', {}).get('device', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Model Name:</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{analysis.get('model_name', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total Parameters:</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{analysis.get('parameters', {}).get('total', 0):,}</td>
                    </tr>
                </table>
            </div>
            """
            
            html_content = html_content.replace("</body>", f"{metadata_html}</body>")
            
            # Save to file if path provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                return output_path
            else:
                # Generate temporary file
                fd, path = tempfile.mkstemp(suffix='.html')
                os.write(fd, html_content.encode('utf-8'))
                os.close(fd)
                return path
                
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return None
    
    def generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a markdown report from analysis results.
        
        Args:
            analysis: Model analysis results
            
        Returns:
            Markdown formatted string
        """
        md = [
            f"# Model Analysis Report: {analysis.get('model_name', 'Unknown Model')}\n",
            f"Generated on: {analysis.get('metadata', {}).get('date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}\n",
            
            "## Model Summary\n",
            f"- **Architecture Type**: {analysis.get('architecture', {}).get('type', 'Unknown')}"
            f" (Confidence: {analysis.get('architecture', {}).get('confidence', 0):.2f})\n",
            f"- **Total Parameters**: {analysis.get('parameters', {}).get('total', 0):,}\n",
            f"- **Trainable Parameters**: {analysis.get('parameters', {}).get('trainable', 0):,}\n",
            f"- **Non-Trainable Parameters**: {analysis.get('parameters', {}).get('non_trainable', 0):,}\n",
            
            "## Performance Metrics\n",
            f"- **Total FLOPs**: {analysis.get('flops', {}).get('total', 0):,} "
            f"({analysis.get('flops', {}).get('gflops', 0):.2f} GFLOPs)\n",
            f"- **Memory Usage**: {analysis.get('memory', {}).get('total_mb', 0):.2f} MB\n",
        ]
        
        # Add inference time if available
        if "performance" in analysis and "inference_time_ms" in analysis["performance"]:
            md.extend([
                f"- **Average Inference Latency**: {analysis['performance']['inference_time_ms']:.2f} ms\n",
                f"- **Throughput**: {analysis['performance'].get('throughput_fps', 0):.2f} images/sec\n"
            ])
        
        # Add compliance check results
        if "compliance" in analysis:
            md.append("\n## Compliance Checks\n")
            
            issues = analysis["compliance"].get("issues", [])
            warnings = analysis["compliance"].get("warnings", [])
            
            if not issues and not warnings:
                md.append("✅ No compliance issues detected\n")
            else:
                if issues:
                    md.append("### Issues\n")
                    for issue in issues:
                        md.append(f"- ❌ {issue}\n")
                
                if warnings:
                    md.append("### Warnings\n")
                    for warning in warnings:
                        md.append(f"- ⚠️ {warning}\n")
        
        # Add cloud/edge readiness score
        if "cloud_edge_readiness" in analysis:
            md.extend([
                "\n## Cloud/Edge Deployment Readiness\n",
                f"- **Overall Score**: {analysis['cloud_edge_readiness']['total_score']}/100\n",
                f"- **Interpretation**: {analysis['cloud_edge_readiness']['interpretation']}\n",
                "\n### Component Scores\n"
            ])
            
            for component, score in analysis["cloud_edge_readiness"].get("component_scores", {}).items():
                md.append(f"- **{component}**: {score}/25\n")
        
        # Add metadata
        md.extend([
            "\n## Analysis Metadata\n",
            f"- **Date**: {analysis.get('metadata', {}).get('date', 'Unknown')}\n",
            f"- **Device**: {analysis.get('metadata', {}).get('device', 'Unknown')}\n",
            f"- **Python Version**: {analysis.get('metadata', {}).get('python_version', 'Unknown')}\n",
            f"- **PyTorch Version**: {analysis.get('metadata', {}).get('torch_version', 'Unknown')}\n"
        ])
        
        return "".join(md)
    
    def analyze_model(self, model: nn.Module, input_shape: Tuple, 
                    batch_size: int = 1,
                    benchmark_repeats: int = 10,
                    generate_report: str = "markdown",
                    report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a neural network model.
        
        Args:
            model: PyTorch model to analyze
            input_shape: Input shape tuple, excluding batch dimension
            batch_size: Batch size for performance benchmarking
            benchmark_repeats: Number of iterations for inference benchmarking
            generate_report: Report format ('none', 'markdown', 'html')
            report_path: Path to save the report (if None, returns the report string)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        # Set model to evaluation mode
        model_mode = model.training
        model.eval()
        
        # Move model to device
        model.to(self.device)
        
        # Get model name
        model_name = model.__class__.__name__
        
        # Full input shape with batch dimension
        full_input_shape = (batch_size, *input_shape)
        
        # Create sample input
        sample_input = torch.randn(full_input_shape).to(self.device)
        
        # Gather metadata
        metadata = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "input_shape": full_input_shape
        }
        
        # Start building the analysis
        analysis = {
            "model_name": model_name,
            "metadata": metadata
        }
        
        # Count parameters
        analysis["parameters"] = self._count_parameters(model)
        
        # Estimate FLOPs
        analysis["flops"] = self._estimate_flops(model, full_input_shape)
        
        # Estimate memory usage
        analysis["memory"] = self._estimate_memory(model, full_input_shape)
        
        # Benchmark inference time
        inference_benchmark = self.benchmark_inference_time(model, sample_input, benchmark_repeats)
        
        # Detect architecture type
        analysis["architecture"] = self._detect_architecture_type(model)
        
        # Estimate max batch size
        if self.device.startswith("cuda"):
            batch_scaling = self.estimate_max_batch_size(model, sample_input[:1])
        else:
            batch_scaling = {"error": "Batch size estimation requires CUDA device"}
        
        # Check model compliance
        analysis["compliance"] = self._check_model_compliance(model)
        
        # Cloud/Edge readiness score
        analysis["cloud_edge_readiness"] = self.calculate_cloud_edge_score(model, full_input_shape)
        
        # Add performance metrics
        analysis["performance"] = {
            "inference_time_ms": inference_benchmark["avg_latency_ms"],
            "throughput_fps": inference_benchmark["throughput_fps"],
            "batch_size_scaling": batch_scaling
        }
        
        # Export to ONNX if available
        if ONNX_AVAILABLE:
            onnx_path = self.export_to_onnx(model, sample_input)
            if onnx_path:
                onnx_flops = self.estimate_onnx_flops(onnx_path)
                analysis["onnx"] = {
                    "export_path": onnx_path,
                    "flops": onnx_flops
                }
        
        # Generate reports if requested
        if generate_report == "markdown":
            report = self.generate_markdown_report(analysis)
            analysis["report"] = report
            
            if report_path:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                    
        elif generate_report == "html" and self.enable_html_reports:
            html_path = self.generate_html_report(analysis, report_path)
            analysis["report_path"] = html_path
        
        # Restore model's original mode
        if model_mode:
            model.train()
        else:
            model.eval()
            
        # Save the analysis
        self.last_analysis = analysis
        
        return analysis

def main():
    """Command-line interface for ModelAnalyzer"""
    if not CLI_AVAILABLE:
        print("CLI dependencies not available")
        return
        
    parser = argparse.ArgumentParser(description="Analyze deep learning models")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--framework", default="pytorch", choices=["pytorch"], help="Deep learning framework")
    parser.add_argument("--input-shape", default="1,3,224,224", help="Input shape (comma-separated)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--report", default="markdown", choices=["markdown", "html", "none"], help="Report format")
    parser.add_argument("--output", help="Output path for report")
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))
    
    try:
        # Load model
        if args.framework == "pytorch":
            # Try to import model from torchvision or as a module path
            try:
                import torchvision.models as tv_models
                if hasattr(tv_models, args.model):
                    model = getattr(tv_models, args.model)()
                    print(f"Loaded model {args.model} from torchvision")
                else:
                    # Try to import custom model
                    module_path, model_name = args.model.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    model = getattr(module, model_name)()
                    print(f"Loaded model {model_name} from {module_path}")
            except (ImportError, AttributeError) as e:
                print(f"Error loading model: {e}")
                return
                
            # Initialize analyzer
            analyzer = ModelAnalyzer()
            
            # Run analysis
            result = analyzer.analyze_model(
                model, 
                input_shape=input_shape[1:],  # Remove batch dimension
                batch_size=args.batch_size,
                generate_report=args.report,
                report_path=args.output
            )
            
            # Print basic info to console
            print(f"\nAnalysis completed for {args.model}")
            print(f"Parameters: {result['parameters']['total']:,}")
            print(f"FLOPs: {result['flops']['gflops']:.2f} GFLOPs")
            print(f"Memory: {result['memory']['total_mb']:.2f} MB")
            print(f"Inference time: {result['performance']['inference_time_ms']:.2f} ms")
            
            if args.output:
                print(f"Report saved to {args.output}")
            elif args.report == "markdown":
                print("\n" + result["report"])
                
    except Exception as e:
        print(f"Error analyzing model: {e}")
        
if __name__ == "__main__":
    main()
