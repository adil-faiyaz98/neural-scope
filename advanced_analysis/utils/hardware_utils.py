"""
Hardware detection and optimization utilities for the advanced_analysis framework.

This module provides functions for detecting hardware capabilities and optimizing
code for specific hardware platforms.
"""

import os
import sys
import logging
import platform
from typing import Dict, List, Optional, Tuple, Any

# Configure logger
logger = logging.getLogger(__name__)

def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware and capabilities.
    
    Returns:
        Dictionary with hardware information
    """
    hardware_info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "gpu_available": False,
        "gpu_info": [],
        "tpu_available": False,
        "tpu_info": [],
        "memory_gb": _get_system_memory_gb()
    }
    
    # Detect CUDA GPUs
    try:
        import torch
        hardware_info["gpu_available"] = torch.cuda.is_available()
        if hardware_info["gpu_available"]:
            hardware_info["gpu_count"] = torch.cuda.device_count()
            hardware_info["gpu_info"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3)
                }
                for i in range(hardware_info["gpu_count"])
            ]
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
        
        # Try with tensorflow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            hardware_info["gpu_available"] = len(gpus) > 0
            if hardware_info["gpu_available"]:
                hardware_info["gpu_count"] = len(gpus)
                hardware_info["gpu_info"] = [{"name": gpu.name} for gpu in gpus]
                
                # Check for TPUs
                tpus = tf.config.list_physical_devices('TPU')
                hardware_info["tpu_available"] = len(tpus) > 0
                if hardware_info["tpu_available"]:
                    hardware_info["tpu_count"] = len(tpus)
                    hardware_info["tpu_info"] = [{"name": tpu.name} for tpu in tpus]
        except ImportError:
            logger.debug("TensorFlow not available for GPU/TPU detection")
    
    return hardware_info

def _get_system_memory_gb() -> float:
    """
    Get total system memory in GB.
    
    Returns:
        Total memory in GB
    """
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        # Extract memory in KB and convert to GB
                        return float(line.split()[1]) / (1024**2)
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'])
            # Convert bytes to GB
            return float(output.strip()) / (1024**3)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullExtendedVirtual', c_ulonglong),
                ]
            
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
            # Convert bytes to GB
            return memoryStatus.ullTotalPhys / (1024**3)
    except Exception as e:
        logger.warning(f"Failed to get system memory: {e}")
    
    return 0.0

def get_optimal_batch_size(model_size_mb: float, hardware_info: Optional[Dict[str, Any]] = None) -> int:
    """
    Calculate optimal batch size based on model size and available hardware.
    
    Args:
        model_size_mb: Model size in MB
        hardware_info: Hardware information (if None, will be detected)
        
    Returns:
        Optimal batch size
    """
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Get available memory
    if hardware_info["gpu_available"] and hardware_info["gpu_info"]:
        # Use GPU memory if available
        available_memory_gb = hardware_info["gpu_info"][0].get("memory_gb", 0)
        if available_memory_gb == 0:
            # Estimate based on GPU name
            gpu_name = hardware_info["gpu_info"][0].get("name", "").lower()
            if "a100" in gpu_name:
                available_memory_gb = 40  # A100 has 40-80GB
            elif "v100" in gpu_name:
                available_memory_gb = 16  # V100 has 16-32GB
            elif "t4" in gpu_name:
                available_memory_gb = 16  # T4 has 16GB
            elif "p100" in gpu_name:
                available_memory_gb = 16  # P100 has 16GB
            elif "k80" in gpu_name:
                available_memory_gb = 12  # K80 has 12GB
            else:
                # Default estimate for unknown GPUs
                available_memory_gb = 8
    else:
        # Use system memory
        available_memory_gb = hardware_info["memory_gb"]
    
    # Convert model size to GB
    model_size_gb = model_size_mb / 1024
    
    # Reserve memory for the framework and other processes
    available_memory_gb *= 0.7
    
    # Calculate batch size (model is loaded once, then replicated for each batch item)
    # We also need memory for activations, gradients, and optimizer states
    # Estimate: 4x model size for training
    memory_per_sample_gb = model_size_gb * 4
    
    if memory_per_sample_gb > 0:
        optimal_batch_size = int(available_memory_gb / memory_per_sample_gb)
        # Ensure batch size is at least 1
        return max(1, optimal_batch_size)
    else:
        # Default batch size if we can't estimate
        return 32

def get_optimal_num_workers(hardware_info: Optional[Dict[str, Any]] = None) -> int:
    """
    Calculate optimal number of data loader workers based on available hardware.
    
    Args:
        hardware_info: Hardware information (if None, will be detected)
        
    Returns:
        Optimal number of workers
    """
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    cpu_count = hardware_info["cpu_count"]
    
    # Use number of CPUs minus 1 for the main process, but at least 1
    return max(1, cpu_count - 1)

def is_gpu_available() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    hardware_info = detect_hardware()
    return hardware_info["gpu_available"]

def is_tpu_available() -> bool:
    """
    Check if TPU is available.
    
    Returns:
        True if TPU is available, False otherwise
    """
    hardware_info = detect_hardware()
    return hardware_info["tpu_available"]

def get_recommended_precision(hardware_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Get recommended precision based on available hardware.
    
    Args:
        hardware_info: Hardware information (if None, will be detected)
        
    Returns:
        Recommended precision (fp32, fp16, bf16)
    """
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    if hardware_info["gpu_available"]:
        # Check for specific GPU capabilities
        try:
            import torch
            if torch.cuda.is_available():
                # Check if GPU supports mixed precision
                if torch.cuda.get_device_capability()[0] >= 7:
                    # Ampere (SM 8.0) and newer support bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8:
                        return "bf16"
                    # Volta (SM 7.0) and newer support float16
                    return "fp16"
        except ImportError:
            pass
    
    # Default to fp32 for CPU or older GPUs
    return "fp32"

def optimize_for_hardware(model: Any, hardware_info: Optional[Dict[str, Any]] = None) -> Any:
    """
    Optimize a model for the available hardware.
    
    Args:
        model: The model to optimize
        hardware_info: Hardware information (if None, will be detected)
        
    Returns:
        Optimized model
    """
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Detect model framework
    framework = _detect_model_framework(model)
    
    if framework == "pytorch":
        return _optimize_pytorch_model(model, hardware_info)
    elif framework == "tensorflow":
        return _optimize_tensorflow_model(model, hardware_info)
    else:
        logger.warning(f"Optimization not supported for framework: {framework}")
        return model

def _detect_model_framework(model: Any) -> str:
    """
    Detect the framework of a model.
    
    Args:
        model: The model to detect
        
    Returns:
        Framework name (pytorch, tensorflow, unknown)
    """
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return "pytorch"
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model) or isinstance(model, tf.Module):
            return "tensorflow"
    except ImportError:
        pass
    
    return "unknown"

def _optimize_pytorch_model(model: Any, hardware_info: Dict[str, Any]) -> Any:
    """
    Optimize a PyTorch model for the available hardware.
    
    Args:
        model: The PyTorch model to optimize
        hardware_info: Hardware information
        
    Returns:
        Optimized model
    """
    try:
        import torch
        
        # Move model to appropriate device
        if hardware_info["gpu_available"]:
            model = model.cuda()
            logger.info("Model moved to CUDA device")
        
        # Apply mixed precision if supported
        precision = get_recommended_precision(hardware_info)
        if precision == "fp16":
            # Use automatic mixed precision
            try:
                from torch.cuda.amp import autocast
                logger.info("Enabling automatic mixed precision (fp16)")
                # We can't directly modify the model here, but we can return a context manager
                # that will be used during training/inference
                return model, autocast
            except ImportError:
                logger.warning("Automatic mixed precision not available")
        elif precision == "bf16":
            # Use bfloat16 precision if available
            try:
                from torch.cuda.amp import autocast
                logger.info("Enabling automatic mixed precision (bf16)")
                # Return model and autocast context with bfloat16 dtype
                return model, lambda: autocast(dtype=torch.bfloat16)
            except (ImportError, AttributeError):
                logger.warning("BFloat16 precision not available")
        
        return model
    except ImportError:
        logger.warning("PyTorch not available for model optimization")
        return model

def _optimize_tensorflow_model(model: Any, hardware_info: Dict[str, Any]) -> Any:
    """
    Optimize a TensorFlow model for the available hardware.
    
    Args:
        model: The TensorFlow model to optimize
        hardware_info: Hardware information
        
    Returns:
        Optimized model
    """
    try:
        import tensorflow as tf
        
        # Configure for GPU if available
        if hardware_info["gpu_available"]:
            # Allow memory growth to avoid allocating all GPU memory
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Configured TensorFlow for GPU usage with memory growth")
        
        # Configure for TPU if available
        if hardware_info["tpu_available"]:
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                strategy = tf.distribute.TPUStrategy(resolver)
                
                # Recreate model with TPU strategy
                with strategy.scope():
                    config = model.get_config()
                    optimized_model = tf.keras.models.model_from_config(config)
                    optimized_model.set_weights(model.get_weights())
                
                logger.info("Model optimized for TPU")
                return optimized_model
            except Exception as e:
                logger.warning(f"Failed to optimize for TPU: {e}")
        
        # Apply mixed precision if supported
        precision = get_recommended_precision(hardware_info)
        if precision in ["fp16", "bf16"]:
            policy_name = 'mixed_float16' if precision == "fp16" else 'mixed_bfloat16'
            try:
                policy = tf.keras.mixed_precision.Policy(policy_name)
                tf.keras.mixed_precision.set_global_policy(policy)
                
                # Recreate model with mixed precision
                config = model.get_config()
                optimized_model = tf.keras.models.model_from_config(config)
                optimized_model.set_weights(model.get_weights())
                
                logger.info(f"Model optimized with {policy_name} precision")
                return optimized_model
            except Exception as e:
                logger.warning(f"Failed to apply mixed precision: {e}")
        
        return model
    except ImportError:
        logger.warning("TensorFlow not available for model optimization")
        return model
