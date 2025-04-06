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
    
    def __init__(self, model, framework=None, model_name=None):
        self.model = model
        
        # Auto-detect framework if not specified
        if framework is None:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                framework = "pytorch"
            elif TF_AVAILABLE and isinstance(model, tf.Module):
                framework = "tensorflow"
            else:
                framework = "unknown"
                
        self.framework = framework
        self.model_name = model_name or "unnamed_model"
        self.results = {}
        
        # Add attributes for enhanced profiling
        self.power_measurements = []
        self.emissions_data = None
        self.distributed_stats = {}
        
        # Validate model compatibility
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model is compatible with the profiler"""
        if self.framework == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for profiling PyTorch models")
            if not isinstance(self.model, torch.nn.Module):
                raise TypeError("For PyTorch profiling, model must be a torch.nn.Module instance")
        elif self.framework == "tensorflow":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required for profiling TensorFlow models")
            if not isinstance(self.model, tf.Module):
                raise TypeError("For TensorFlow profiling, model must be a tf.Module instance")
        elif self.framework == "unknown":
            warnings.warn("Model framework could not be auto-detected. Limited profiling capabilities will be available.")
    
    def __enter__(self):
        """Context manager entry point"""
        logger.debug(f"Starting profiling session for {self.model_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point for clean resource management"""
        logger.debug(f"Ending profiling session for {self.model_name}")
        self.cleanup()
        # Don't suppress exceptions
        return False
    
    def cleanup(self):
        """Clean up resources used during profiling"""
        # Free up memory and resources
        if self.framework == "pytorch" and TORCH_AVAILABLE:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def profile(self, input_data, batch_size=None, num_steps=10, 
                track_power=False, track_emissions=False, profile_nccl=False):
        """
        Profile model performance with the given input data
        
        Args:
            input_data: Input data for the model
            batch_size: Batch size to use (if None, inferred from input_data)
            num_steps: Number of steps to profile
            track_power: Whether to track power consumption
            track_emissions: Whether to track carbon emissions
            profile_nccl: Whether to profile NCCL communication (for distributed training)
            
        Returns:
            ProfilingResult with detailed performance metrics
            
        Raises:
            ValueError: For invalid input parameters
            RuntimeError: For errors during profiling execution
        """
        # Input validation
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        
        if input_data is None:
            raise ValueError("input_data cannot be None")
        
        # Validate batch_size or try to infer it
        if batch_size is None:
            try:
                # Try to infer batch size from common input formats
                if hasattr(input_data, "shape"):
                    batch_size = input_data.shape[0]
                elif isinstance(input_data, (list, tuple)):
                    batch_size = len(input_data)
                else:
                    batch_size = 1
                    warnings.warn(f"Could not infer batch size. Using default value: {batch_size}")
            except Exception as e:
                batch_size = 1
                warnings.warn(f"Error inferring batch size: {str(e)}. Using default value: {batch_size}")
        
        # Feature availability checks
        if track_power:
            try:
                # Check if power tracking is available
                from .energy import PowerTracker
                power_tracker = PowerTracker()
            except ImportError:
                warnings.warn("Power tracking dependencies not available. Power tracking disabled.")
                track_power = False
        
        if track_emissions and not track_power:
            warnings.warn("Carbon emissions tracking requires power tracking. Emissions tracking disabled.")
            track_emissions = False
        
        try:
            # Implementation would depend on the framework
            if self.framework == "pytorch":
                result = self._profile_pytorch(input_data, batch_size, num_steps, 
                                               track_power, track_emissions, profile_nccl)
            elif self.framework == "tensorflow":
                result = self._profile_tensorflow(input_data, batch_size, num_steps,
                                                  track_power, track_emissions)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
            
            # Store the result for later reference
            self.results[time.time()] = result
            return result
            
        except Exception as e:
            logger.error(f"Error during profiling: {str(e)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            self.cleanup()  # Ensure cleanup happens even on error
            raise RuntimeError(f"Profiling failed: {str(e)}") from e

    def _profile_pytorch(self, input_data, batch_size, num_steps, 
                        track_power, track_emissions, profile_nccl):
        """Profile PyTorch model with robust error handling"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorch model profiling")
        
        # Guard against common PyTorch-specific errors
        device = next(self.model.parameters()).device
        
        try:
            # Ensure input is on the same device as model
            if hasattr(input_data, 'to') and callable(input_data.to):
                input_data = input_data.to(device)
            
            # Initialize tracking metrics
            execution_times = []
            memory_usages = []
            
            # Set up profiler if available
            if hasattr(torch, 'profiler'):
                # Configure PyTorch profiler
                pass
            
            # Actual profiling loop with proper error handling
            for step in range(num_steps):
                try:
                    # Record start time
                    start_time = time.time()
                    
                    # Record GPU memory usage before forward pass if available
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = self.model(input_data)
                    
                    # Record end time
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Wait for all kernels to finish
                    end_time = time.time()
                    
                    # Record GPU memory usage after forward pass if available
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                        memory_usages.append({'gpu': mem_after, 'diff': mem_after - mem_before})
                    
                    # Record execution time
                    execution_times.append(end_time - start_time)
                    
                except Exception as e:
                    logger.warning(f"Error during profiling step {step}: {str(e)}")
                    # Continue with next step rather than failing completely
            
            if not execution_times:
                raise RuntimeError("No successful profiling steps completed")
            
            # Calculate average metrics and create result
            # ...similar to existing placeholder code but with actual measurements...
            
            return result
            
        except Exception as e:
            # Log the specific PyTorch error
            logger.error(f"PyTorch profiling error: {str(e)}")
            raise
    
    def _profile_tensorflow(self, input_data, batch_size, num_steps,
                           track_power, track_emissions):
        """Profile TensorFlow model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TensorFlow model profiling")
        
        # Implementation details would go here
        # This would include:
        # - Using tf.profiler
        # - Measuring execution time
        # - Tracking GPU memory usage
        # - Calculating throughput
        
        # Placeholder for demonstration
        result = ProfilingResult(
            execution_time=0.1 * num_steps,
            memory_usage={"gpu": 1000, "cpu": 500},
            throughput=batch_size / 0.1,
            hardware_utilization={"gpu": 0.5, "cpu": 0.3},
            bottlenecks=["XLA compilation overhead"],
            optimization_suggestions=["Enable mixed precision with tf.keras.mixed_precision"]
        )
        
        return result
    
    @contextlib.contextmanager
    def profile_section(self, section_name: str) -> Generator[None, None, None]:
        """
        Context manager for profiling specific sections of code
        
        Args:
            section_name: Name of the section being profiled
            
        Yields:
            None
        
        Example:
            ```python
            profiler = ModelPerformanceProfiler(model)
            with profiler.profile_section("data_preprocessing"):
                # Code to profile
                processed_data = preprocess(data)
            ```
        """
        try:
            start_time = time.time()
            logger.debug(f"Starting profiling section: {section_name}")
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(f"Section {section_name} took {duration:.4f} seconds")
            # Store section timing in results
            if 'sections' not in self.results:
                self.results['sections'] = {}
            self.results['sections'][section_name] = duration
