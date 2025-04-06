"""
DataLoaderOptimizer: ML Data Loading Performance Optimization System

A comprehensive system for profiling and optimizing data loaders in machine learning
workflows, supporting PyTorch DataLoaders, TensorFlow Datasets, and other common formats.
Provides automatic benchmarking, optimal parameter selection, and detailed profiling reports.
"""

import time
import json
import os
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import datasets
    from datasets import Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class BatchProfileResult:
    """Results from profiling a specific batch size"""
    batch_size: int
    avg_batch_time: float  # seconds
    throughput: float  # samples/second
    memory_used: Optional[float] = None  # MB
    std_batch_time: float = 0.0
    min_batch_time: float = 0.0
    max_batch_time: float = 0.0


@dataclass
class WorkerProfileResult:
    """Results from profiling different worker counts"""
    num_workers: int
    avg_batch_time: float  # seconds
    throughput: float  # samples/second
    cpu_usage: Optional[float] = None  # percentage
    std_batch_time: float = 0.0


@dataclass
class DataLoaderProfilingResult:
    """Comprehensive profiling results for a data loader"""
    dataset_type: str
    batch_size_profile: Dict[int, BatchProfileResult] = field(default_factory=dict)
    worker_profile: Dict[int, WorkerProfileResult] = field(default_factory=dict)
    optimal_batch_size: Optional[int] = None
    optimal_num_workers: Optional[int] = None
    performance_bottleneck: Optional[str] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    profile_timestamp: float = field(default_factory=time.time)


class DataLoaderOptimizer:
    """
    System for profiling and optimizing data loaders in ML workflows
    """
    
    def __init__(self):
        self.last_profile_result = None
    
    def detect_dataset_type(self, dataset_or_loader):
        """
        Automatically detect the type of dataset or data loader
        
        Args:
            dataset_or_loader: Dataset or DataLoader object to detect
            
        Returns:
            String describing the dataset type
        """
        # PyTorch DataLoader detection
        if TORCH_AVAILABLE:
            if isinstance(dataset_or_loader, DataLoader):
                return "pytorch_dataloader"
                
            # PyTorch Dataset detection
            if hasattr(dataset_or_loader, "__getitem__") and hasattr(dataset_or_loader, "__len__"):
                return "pytorch_dataset"
        
        # TensorFlow Dataset detection
        if TF_AVAILABLE:
            if isinstance(dataset_or_loader, tf.data.Dataset):
                return "tensorflow_dataset"
        
        # HuggingFace Dataset detection
        if DATASETS_AVAILABLE:
            if isinstance(dataset_or_loader, HFDataset):
                return "huggingface_dataset"
        
        # File-based detection
        if hasattr(dataset_or_loader, "filenames"):
            filenames = dataset_or_loader.filenames if callable(dataset_or_loader.filenames) else dataset_or_loader.filenames
            if any(f.endswith('.tfrecord') for f in filenames):
                return "tfrecord_dataset"
            elif any(f.endswith('.parquet') for f in filenames):
                return "parquet_dataset"
            elif any(f.endswith('.csv') for f in filenames):
                return "csv_dataset"
            
        # Fallback
        return "unknown_dataset"
    
    def profile_batch_sizes(self, dataloader, batch_sizes=None, num_batches=10, 
                            use_gpu=False, memory_profiling=False):
        """
        Profile different batch sizes to find optimal throughput vs memory trade-off
        
        Args:
            dataloader: The data loader to profile (will be recreated with different batch sizes)
            batch_sizes: List of batch sizes to test (default: [1, 4, 8, 16, 32, 64, 128])
            num_batches: Number of batches to average over
            use_gpu: Whether to use GPU for memory profiling
            memory_profiling: Whether to measure memory usage
            
        Returns:
            Dict mapping batch sizes to BatchProfileResult objects
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64, 128]
            
        results = {}
        dataset_type = self.detect_dataset_type(dataloader)
        
        for batch_size in batch_sizes:
            try:
                # Recreate data loader with new batch size
                if dataset_type == "pytorch_dataloader":
                    modified_loader = self._recreate_pytorch_loader(dataloader, batch_size)
                elif dataset_type == "tensorflow_dataset":
                    modified_loader = dataloader.batch(batch_size)
                else:
                    logger.warning(f"Batch size profiling not supported for dataset type: {dataset_type}")
                    continue
                
                # Perform benchmark
                batch_times = []
                memory_usage = []
                
                # Warmup
                self._warmup_dataloader(modified_loader, 2, dataset_type)
                
                # Benchmark
                for _ in range(num_batches):
                    if memory_profiling and use_gpu and TORCH_AVAILABLE:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()
                    
                    start_time = time.time()
                    # Load a batch
                    if dataset_type == "pytorch_dataloader":
                        for batch in modified_loader:
                            # Process the batch (just accessing it)
                            if isinstance(batch, (list, tuple)):
                                for item in batch:
                                    if hasattr(item, "shape"):
                                        _ = item.shape
                            elif hasattr(batch, "shape"):
                                _ = batch.shape
                            break  # Just one batch
                    elif dataset_type == "tensorflow_dataset":
                        for batch in modified_loader:
                            _ = batch
                            break
                    
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                    
                    if memory_profiling and use_gpu and TORCH_AVAILABLE:
                        mem_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                        memory_usage.append(mem_usage)
                
                # Calculate statistics
                avg_time = np.mean(batch_times)
                std_time = np.std(batch_times)
                min_time = np.min(batch_times)
                max_time = np.max(batch_times)
                
                # Create result
                result = BatchProfileResult(
                    batch_size=batch_size,
                    avg_batch_time=avg_time,
                    throughput=batch_size / avg_time,
                    memory_used=np.mean(memory_usage) if memory_usage else None,
                    std_batch_time=std_time,
                    min_batch_time=min_time,
                    max_batch_time=max_time
                )
                
                results[batch_size] = result
                
            except Exception as e:
                logger.error(f"Error profiling batch size {batch_size}: {e}")
                
        # Store in profiling result
        if not hasattr(self, "last_profile_result") or self.last_profile_result is None:
            self.last_profile_result = DataLoaderProfilingResult(
                dataset_type=dataset_type,
                batch_size_profile=results
            )
        else:
            self.last_profile_result.batch_size_profile = results
            
        # Determine optimal batch size
        self._determine_optimal_batch_size()
            
        return results
    
    def _recreate_pytorch_loader(self, original_loader, batch_size):
        """Recreate a PyTorch DataLoader with a new batch size"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this operation")
            
        # Extract original arguments
        dataset = original_loader.dataset
        num_workers = original_loader.num_workers
        pin_memory = original_loader.pin_memory
        shuffle = original_loader.shuffle
        drop_last = original_loader.drop_last
        
        # Create new loader
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def _warmup_dataloader(self, dataloader, num_batches, dataset_type):
        """Warmup the dataloader by running a few batches"""
        try:
            batch_count = 0
            if dataset_type == "pytorch_dataloader":
                for _ in dataloader:
                    batch_count += 1
                    if batch_count >= num_batches:
                        break
            elif dataset_type == "tensorflow_dataset":
                for _ in dataloader:
                    batch_count += 1
                    if batch_count >= num_batches:
                        break
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def profile_worker_utilization(self, dataloader, worker_counts=None, num_batches=10):
        """
        Profile different worker counts to find optimal parallelism using time deltas
        
        Args:
            dataloader: PyTorch DataLoader to profile
            worker_counts: List of worker counts to test
            num_batches: Number of batches to average over
            
        Returns:
            Dict mapping worker counts to WorkerProfileResult objects
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for worker utilization profiling")
            return {}
            
        if worker_counts is None:
            # Determine reasonable worker counts based on CPU cores
            try:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                worker_counts = [0, 1, 2, 4]
                if cpu_count > 4:
                    worker_counts.extend([cpu_count//2, cpu_count])
                worker_counts = sorted(set(worker_counts))  # Remove duplicates
            except:
                worker_counts = [0, 1, 2, 4]
        
        results = {}
        dataset_type = self.detect_dataset_type(dataloader)
        
        if dataset_type != "pytorch_dataloader":
            logger.warning("Worker utilization profiling only supports PyTorch DataLoaders")
            return {}
            
        original_batch_size = dataloader.batch_size
        dataset = dataloader.dataset
        shuffle = dataloader.shuffle
        pin_memory = dataloader.pin_memory
        drop_last = dataloader.drop_last
        
        for num_workers in worker_counts:
            try:
                # Create data loader with this worker count
                test_loader = DataLoader(
                    dataset,
                    batch_size=original_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last
                )
                
                # Warmup
                self._warmup_dataloader(test_loader, 2, dataset_type)
                
                # Benchmark
                batch_times = []
                batch_load_start = None
                worker_time_deltas = []
                
                for _ in range(num_batches):
                    # Time the entire iteration
                    start_time = time.time()
                    for batch in test_loader:
                        # Process the batch (just accessing it)
                        if isinstance(batch, (list, tuple)):
                            for item in batch:
                                if hasattr(item, "shape"):
                                    _ = item.shape
                        elif hasattr(batch, "shape"):
                            _ = batch.shape
                        break
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                    
                    # Record time delta between batches (to estimate worker efficiency)
                    if batch_load_start is not None:
                        worker_time_deltas.append(start_time - batch_load_start)
                    batch_load_start = time.time()
                
                # Calculate statistics
                avg_time = np.mean(batch_times)
                std_time = np.std(batch_times)
                
                # Create result
                result = WorkerProfileResult(
                    num_workers=num_workers,
                    avg_batch_time=avg_time,
                    throughput=original_batch_size / avg_time,
                    std_batch_time=std_time
                )
                
                results[num_workers] = result
                
            except Exception as e:
                logger.error(f"Error profiling worker count {num_workers}: {e}")
        
        # Store in profiling result
        if not hasattr(self, "last_profile_result") or self.last_profile_result is None:
            self.last_profile_result = DataLoaderProfilingResult(
                dataset_type=dataset_type,
                worker_profile=results
            )
        else:
            self.last_profile_result.worker_profile = results
            
        # Determine optimal worker count
        self._determine_optimal_worker_count()
            
        return results
    
    def _determine_optimal_batch_size(self):
        """Determine the optimal batch size from profiling results"""
        if not self.last_profile_result or not self.last_profile_result.batch_size_profile:
            return None
            
        batch_profile = self.last_profile_result.batch_size_profile
        
        # Simple heuristic: Choose batch size with highest throughput
        max_throughput = 0
        optimal_batch = None
        
        for batch_size, result in batch_profile.items():
            if result.throughput > max_throughput:
                max_throughput = result.throughput
                optimal_batch = batch_size
                
        self.last_profile_result.optimal_batch_size = optimal_batch
        return optimal_batch
    
    def _determine_optimal_worker_count(self):
        """Determine the optimal worker count from profiling results"""
        if not self.last_profile_result or not self.last_profile_result.worker_profile:
            return None
            
        worker_profile = self.last_profile_result.worker_profile
        
        # Find lowest worker count that's within 5% of the best throughput
        max_throughput = max(result.throughput for result in worker_profile.values())
        threshold = max_throughput * 0.95  # Within 5% of max
        
        candidates = [(num_workers, result.throughput) 
                     for num_workers, result in worker_profile.items() 
                     if result.throughput >= threshold]
        
        if candidates:
            # Choose the lowest worker count meeting the threshold
            optimal_workers = min(candidates, key=lambda x: x[0])[0]
            self.last_profile_result.optimal_num_workers = optimal_workers
            return optimal_workers
        
        return None
    
    def generate_recommendations(self):
        """Generate recommendations based on profiling results"""
        if not self.last_profile_result:
            return []
            
        recommendations = []
        
        # Batch size recommendation
        if self.last_profile_result.optimal_batch_size:
            recommendations.append({
                "type": "batch_size",
                "title": "Optimal Batch Size",
                "description": f"Use batch_size={self.last_profile_result.optimal_batch_size} for best throughput",
                "code_example": f"dataloader = DataLoader(dataset, batch_size={self.last_profile_result.optimal_batch_size}, ...)"
            })
            
        # Worker count recommendation
        if self.last_profile_result.optimal_num_workers:
            recommendations.append({
                "type": "num_workers",
                "title": "Optimal Worker Count",
                "description": f"Use num_workers={self.last_profile_result.optimal_num_workers} for best CPU utilization",
                "code_example": f"dataloader = DataLoader(dataset, num_workers={self.last_profile_result.optimal_num_workers}, ...)"
            })
            
        # Dataset-specific recommendations
        dataset_type = self.last_profile_result.dataset_type
        if dataset_type == "pytorch_dataloader":
            if self.last_profile_result.optimal_num_workers and self.last_profile_result.optimal_num_workers > 0:
                recommendations.append({
                    "type": "pin_memory",
                    "title": "Enable pin_memory",
                    "description": "When using multiple workers with GPU, enable pin_memory=True for better transfer speed",
                    "code_example": "dataloader = DataLoader(dataset, pin_memory=True, ...)"
                })
        
        elif dataset_type == "tensorflow_dataset":
            recommendations.append({
                "type": "prefetch",
                "title": "Enable prefetching",
                "description": "Use tf.data.experimental.AUTOTUNE for optimal prefetching",
                "code_example": "dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)"
            })
            
        # Update the recommendations
        self.last_profile_result.recommendations = recommendations
        return recommendations
    
    def export_profile_to_json(self, filepath=None):
        """
        Export profiling results to JSON format for logs and CI integration
        
        Args:
            filepath: Path to save the JSON file (if None, returns string)
            
        Returns:
            JSON string if filepath is None, else None
        """
        if not self.last_profile_result:
            raise ValueError("No profiling results available")
            
        # Convert dataclass to dict
        def _dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: _dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_dataclass_to_dict(x) for x in obj]
            else:
                return obj
        
        # Prepare data for serialization
        profile_data = _dataclass_to_dict(self.last_profile_result)
        
        # Convert batch_size_profile and worker_profile from dict with int keys to list
        if "batch_size_profile" in profile_data:
            profile_data["batch_size_profile"] = [
                {"batch_size": int(k), **v} 
                for k, v in profile_data["batch_size_profile"].items()
            ]
            
        if "worker_profile" in profile_data:
            profile_data["worker_profile"] = [
                {"num_workers": int(k), **v} 
                for k, v in profile_data["worker_profile"].items()
            ]
        
        # Add metadata
        profile_data["profile_metadata"] = {
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        # Convert to JSON
        json_data = json.dumps(profile_data, indent=2)
        
        # Save to file if path provided
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
            return None
        
        return json_data
    
    def full_profile(self, dataloader, batch_sizes=None, worker_counts=None, 
                    num_batches=10, use_gpu=False):
        """
        Run a full profiling suite on the dataloader
        
        Args:
            dataloader: DataLoader to profile
            batch_sizes: List of batch sizes to test
            worker_counts: List of worker counts to test
            num_batches: Number of batches to average over
            use_gpu: Whether to profile GPU memory usage
            
        Returns:
            DataLoaderProfilingResult with complete profiling information
        """
        # Detect dataset type
        dataset_type = self.detect_dataset_type(dataloader)
        
        # Create profiling result
        self.last_profile_result = DataLoaderProfilingResult(dataset_type=dataset_type)
        
        # Run batch size profiling
        self.profile_batch_sizes(dataloader, batch_sizes, num_batches, use_gpu)
        
        # Run worker count profiling if PyTorch
        if dataset_type == "pytorch_dataloader":
            self.profile_worker_utilization(dataloader, worker_counts, num_batches)
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.last_profile_result
