import torch
import time
import threading
import psutil
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

@dataclass
class DataLoaderProfilingResult:
    """Comprehensive profiling results for data loading"""
    avg_batch_load_time: float  # Average time to load one batch
    io_bound_percentage: float  # Percentage of time spent on I/O
    cpu_bound_percentage: float  # Percentage of time spent on CPU processing
    gpu_transfer_time: float  # Time spent transferring data to GPU
    worker_utilization: List[float]  # Utilization of each worker thread
    memory_consumption: float  # Memory used by the dataloader
    recommendations: List[Dict[str, Any]]  # Optimization recommendations

class DatasetOptimizer:
    """Analyzes and optimizes dataset loading and preprocessing"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.result = None
        
    def profile(self, num_batches=50):
        """Profile dataloader performance"""
        # Initialize metrics
        batch_times = []
        worker_active_times = {}
        io_times = []
        cpu_times = []
        gpu_transfer_times = []
        
        # Track disk I/O
        initial_disk_io = psutil.disk_io_counters()
        
        # Start tracking memory
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        start_time = time.time()
        
        # Profile actual data loading
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
                
            # Measure batch loading time
            batch_end = time.time()
            batch_times.append(batch_end - start_time)
            
            # Measure GPU transfer time if using CUDA
            if isinstance(batch, torch.Tensor) and torch.cuda.is_available():
                transfer_start = time.time()
                batch = batch.cuda()
                torch.cuda.synchronize()
                transfer_end = time.time()
                gpu_transfer_times.append(transfer_end - transfer_start)
            elif isinstance(batch, (list, tuple)) and torch.cuda.is_available():
                transfer_start = time.time()
                if isinstance(batch[0], torch.Tensor):
                    batch = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]
                    torch.cuda.synchronize()
                transfer_end = time.time()
                gpu_transfer_times.append(transfer_end - transfer_start)
            
            # Start timer for next batch
            start_time = time.time()
        
        # Get final memory usage
        final_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        memory_used = final_memory - initial_memory
        
        # Get final disk I/O
        final_disk_io = psutil.disk_io_counters()
        read_bytes = final_disk_io.read_bytes - initial_disk_io.read_bytes
        
        # Calculate metrics
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_gpu_transfer = sum(gpu_transfer_times) / len(gpu_transfer_times) if gpu_transfer_times else 0
        
        # Estimate I/O vs CPU bound
        io_bound_estimate = min(1.0, read_bytes / (1024**2) / (sum(batch_times) * 50))  # Rough estimate
        cpu_bound_estimate = 1.0 - io_bound_estimate - (avg_gpu_transfer / avg_batch_time if avg_batch_time > 0 else 0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_batch_time,
            io_bound_estimate,
            cpu_bound_estimate,
            avg_gpu_transfer,
            memory_used
        )
        
        # Create result
        self.result = DataLoaderProfilingResult(
            avg_batch_load_time=avg_batch_time,
            io_bound_percentage=io_bound_estimate * 100,
            cpu_bound_percentage=cpu_bound_estimate * 100,
            gpu_transfer_time=avg_gpu_transfer,
            worker_utilization=[1.0],  # Will be enhanced with actual worker tracking
            memory_consumption=memory_used,
            recommendations=recommendations
        )
        
        return self.result
        
    def _generate_recommendations(self, batch_time, io_bound, cpu_bound, gpu_transfer, memory_used):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check if DataLoader parameters are accessible
        num_workers = getattr(self.dataloader, 'num_workers', 0)
        prefetch_factor = getattr(self.dataloader, 'prefetch_factor', 2)
        pin_memory = getattr(self.dataloader, 'pin_memory', False)
        batch_size = getattr(self.dataloader, 'batch_size', None)
        
        # I/O bound optimizations
        if io_bound > 0.5:  # If more than 50% time spent on I/O
            recommendations.append({
                'type': 'io_optimization',
                'severity': 'high',
                'description': f"Dataset loading is I/O bound ({io_bound:.1%})",
                'recommendation': "Consider caching datasets, memory mapping, or using faster storage",
                'code_example': """
# Option 1: Cache dataset in memory
cached_dataset = list(dataset)  # For small-medium datasets

# Option 2: Use memory-mapped files for large datasets
import numpy as np
# Convert data to numpy and save as memmap
data = np.asarray(data)
np.save('memmap_data.npy', data)
# Later load with memmap
memmap_data = np.load('memmap_data.npy', mmap_mode='r')

# Option 3: Use a faster storage solution
# Consider moving dataset to SSD or using cloud storage with high throughput
"""
            })
            
            # Increase workers if I/O bound
            if num_workers < os.cpu_count():
                recommendations.append({
                    'type': 'worker_optimization',
                    'severity': 'medium',
                    'description': f"Insufficient worker threads ({num_workers}) for I/O bound loading",
                    'recommendation': f"Increase num_workers from {num_workers} to {min(os.cpu_count(), num_workers + 4)}",
                    'code_example': f"""
# Increase the number of worker threads
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size={batch_size},
    num_workers={min(os.cpu_count(), num_workers + 4)},
    pin_memory={pin_memory},
    prefetch_factor={prefetch_factor}
)
"""
                })
                
        # CPU bound optimizations
        if cpu_bound > 0.5:  # If more than 50% time spent on CPU processing
            recommendations.append({
                'type': 'preprocessing_optimization',
                'severity': 'high',
                'description': f"Dataset loading is CPU bound ({cpu_bound:.1%})",
                'recommendation': "Optimize data preprocessing or consider pre-computing transformations",
                'code_example': """
# Option 1: Use faster transforms
# Replace PIL-based transforms with Torchvision's GPU transforms or NVIDIA DALI

# Example with DALI for faster preprocessing:
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

@pipeline_def
def create_pipeline():
    images, labels = fn.readers.file(file_root='./data')
    images = fn.decoders.image(images, device='mixed')
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(images,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return images, labels

# Option 2: Pre-compute and cache transformations
def pre_process_dataset(dataset_path):
    # Load all images
    # Apply transformations
    # Save processed tensors
    processed_data = []
    for image_path in glob.glob(f"{dataset_path}/*.jpg"):
        img = load_and_process_image(image_path)
        processed_data.append(img)
    torch.save(processed_data, 'processed_dataset.pt')
"""
            })
            
        # GPU transfer optimizations
        if gpu_transfer / batch_time > 0.3:  # If more than 30% time spent on GPU transfer
            if not pin_memory:
                recommendations.append({
                    'type': 'gpu_transfer_optimization',
                    'severity': 'medium',
                    'description': f"Slow GPU transfer time ({gpu_transfer:.4f}s per batch)",
                    'recommendation': "Enable pin_memory to speed up GPU transfers",
                    'code_example': f"""
# Enable pin_memory for faster CPU to GPU transfers
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size={batch_size},
    num_workers={num_workers},
    pin_memory=True,
    prefetch_factor={prefetch_factor}
)
"""
                })
                
        # Batch size optimization
        if batch_time > 0.05:  # If batch loading takes more than 50ms
            recommendations.append({
                'type': 'batch_size_optimization',
                'severity': 'low',
                'description': f"Batch loading time ({batch_time:.4f}s) might impact training speed",
                'recommendation': "Consider adjusting batch size or using batch prefetching",
                'code_example': f"""
# Adjust prefetch factor to hide data loading latency
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size={batch_size},
    num_workers={num_workers},
    pin_memory={pin_memory},
    prefetch_factor={prefetch_factor + 2}  # Increase prefetch factor
)
"""
            })
            
        # Memory usage optimizations
        if memory_used > 4000:  # If using more than 4GB of RAM
            recommendations.append({
                'type': 'memory_optimization',
                'severity': 'medium',
                'description': f"High memory usage during data loading ({memory_used:.1f} MB)",
                'recommendation': "Consider using streaming datasets or reducing redundant caching",
                'code_example': """
# Example of a streaming dataset that doesn't load everything into memory
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def process_data(self, file_path):
        # Load and process data from a single file
        data = load_single_file(file_path)
        yield data
        
    def __iter__(self):
        for file_path in self.file_paths:
            yield from self.process_data(file_path)

# Use WebDataset for efficient streaming from cloud storage
import webdataset as wds
dataset = wds.WebDataset('s3://bucket/training-{000..999}.tar')
    .decode("pil")
    .to_tuple("ppm;jpg;jpeg;png", "cls")
    .map_tuple(transform, identity)
"""
            })
            
        return recommendations

    def visualize_profiling(self):
        """Create visualizations of the profiling results"""
        if not self.result:
            raise ValueError("No profiling results available. Run profile() first.")
            
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time breakdown chart
        breakdown = [
            self.result.io_bound_percentage, 
            self.result.cpu_bound_percentage,
            (self.result.gpu_transfer_time / self.result.avg_batch_load_time * 100) if self.result.avg_batch_load_time > 0 else 0
        ]
        
        labels = ['I/O Operations', 'CPU Processing', 'GPU Transfer']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        axs[0].pie(breakdown, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axs[0].set_title('Batch Loading Time Breakdown')
        
        # Batch loading time
        axs[1].bar(['Average Batch Time'], [self.result.avg_batch_load_time * 1000], color='#66b3ff')
        axs[1].set_ylabel('Time (ms)')
        axs[1].set_title('Average Batch Loading Time')
        
        plt.tight_layout()
        return fig

# Example usage:
# dataloader = torch.utils.data.DataLoader(...)
# optimizer = DatasetOptimizer(dataloader)
# result = optimizer.profile()
# optimizer.visualize_profiling()