import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ScalingEfficiencyResult:
    """Stores scaling efficiency analysis results for distributed training"""
    devices: List[int]  # Number of devices in each test
    throughput: List[float]  # Throughput per configuration
    efficiency: List[float]  # Scaling efficiency (0-1)
    communication_overhead: List[float]  # Percentage of time spent in communication
    memory_utilization: List[float]  # Memory utilization per device
    bottlenecks: List[str]  # Identified bottlenecks per configuration
    parallelism_recommendations: Dict[str, Any]  # Recommendations for effective parallelism


class DistributedTrainingAnalyzer:
    """Analyzes distributed training performance and recommends optimal configurations"""
    
    def __init__(self, model, framework="pytorch"):
        self.model = model
        self.framework = framework.lower()
        self.results = {}
        
    def analyze_scaling_efficiency(self, 
                                  input_generator, 
                                  device_counts=[1, 2, 4, 8], 
                                  batch_sizes=None,
                                  iterations=10) -> ScalingEfficiencyResult:
        """
        Analyze how efficiently the model scales across different device counts
        
        Args:
            input_generator: Function that generates input data of specified batch size
            device_counts: List of device counts to test
            batch_sizes: Optional list of batch sizes per device configuration (auto-scaled if None)
            iterations: Number of iterations to run per configuration
            
        Returns:
            ScalingEfficiencyResult with detailed scaling metrics
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for distributed analysis")
            
        max_devices = torch.cuda.device_count()
        valid_counts = [d for d in device_counts if d <= max_devices]
        if not valid_counts:
            raise ValueError(f"No valid device counts. System has {max_devices} devices available")
            
        if batch_sizes is None:
            # Auto-scale batch size proportionally to device count if not specified
            base_batch = 16  # Default base batch size
            batch_sizes = [base_batch * count for count in valid_counts]
            
        throughputs = []
        efficiencies = []
        comm_overheads = []
        memory_utils = []
        bottleneck_types = []
        
        base_throughput = None
        
        for i, count in enumerate(valid_counts):
            logger.info(f"Testing with {count} devices and batch size {batch_sizes[i]}")
            
            # Configure for multi-GPU if needed
            if count > 1:
                self._setup_distributed_environment(count)
                
            # Create model copy for this configuration
            if self.framework == "pytorch":
                if count > 1:
                    from torch.nn.parallel import DistributedDataParallel
                    model_copy = DistributedDataParallel(
                        self.model.to(f"cuda:0"), 
                        device_ids=[0],
                        output_device=0
                    )
                else:
                    model_copy = self.model.to("cuda:0")
            else:
                # Handle other frameworks here
                model_copy = self.model
                
            # Generate input data
            inputs = input_generator(batch_sizes[i])
            
            # Warm-up
            for _ in range(3):
                self._run_iteration(model_copy, inputs)
            
            # Measure performance
            start_time = time.time()
            comm_time_total = 0
            
            for iter_idx in range(iterations):
                # Run with NCCL timing
                iter_time, comm_time = self._run_iteration_with_timing(model_copy, inputs)
                comm_time_total += comm_time
                
            total_time = time.time() - start_time
            
            # Calculate metrics
            samples_per_iter = batch_sizes[i] * count  # Total samples processed per iteration
            throughput = (samples_per_iter * iterations) / total_time
            throughputs.append(throughput)
            
            # Calculate scaling efficiency (vs. perfect linear scaling)
            if base_throughput is None:
                base_throughput = throughput
                efficiency = 1.0
            else:
                ideal_throughput = base_throughput * (count / valid_counts[0])
                efficiency = throughput / ideal_throughput
                
            efficiencies.append(efficiency)
            
            # Communication overhead calculation
            comm_overhead = comm_time_total / total_time if count > 1 else 0
            comm_overheads.append(comm_overhead)
            
            # Memory utilization
            memory_util = self._measure_memory_utilization(count)
            memory_utils.append(memory_util)
            
            # Identify bottleneck
            if count == 1:
                bottleneck = "N/A"
            elif efficiency < 0.7 and comm_overhead > 0.3:
                bottleneck = "communication_bound"
            elif memory_util > 0.9:
                bottleneck = "memory_bound"
            elif efficiency < 0.7:
                bottleneck = "resource_contention"
            else:
                bottleneck = "balanced"
                
            bottleneck_types.append(bottleneck)
            
            # Clean up for next iteration
            if count > 1:
                self._teardown_distributed_environment()
            
        # Generate parallelism recommendations
        parallelism_recommendations = self._recommend_parallelism_strategy(
            valid_counts, efficiencies, comm_overheads, memory_utils, bottleneck_types
        )
        
        # Create result object
        result = ScalingEfficiencyResult(
            devices=valid_counts,
            throughput=throughputs,
            efficiency=efficiencies,
            communication_overhead=comm_overheads,
            memory_utilization=memory_utils,
            bottlenecks=bottleneck_types,
            parallelism_recommendations=parallelism_recommendations
        )
        
        # Store result
        self.results['scaling_efficiency'] = result
        
        return result
    
    def visualize_scaling_efficiency(self, result=None):
        """Create visualization of scaling efficiency analysis"""
        if result is None:
            if 'scaling_efficiency' not in self.results:
                raise ValueError("No scaling efficiency results available")
            result = self.results['scaling_efficiency']
            
        plt.figure(figsize=(15, 12))
        
        # Plot throughput
        plt.subplot(2, 2, 1)
        plt.plot(result.devices, result.throughput, 'o-', linewidth=2)
        plt.plot(result.devices, [result.throughput[0] * d/result.devices[0] for d in result.devices], 
                '--', label='Linear Scaling')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Throughput Scaling')
        plt.grid(True)
        plt.legend()
        
        # Plot efficiency
        plt.subplot(2, 2, 2)
        plt.plot(result.devices, result.efficiency, 'o-', linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='0.8 Efficiency')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Scaling Efficiency')
        plt.title('Scaling Efficiency')
        plt.grid(True)
        plt.ylim(0, 1.1)
        plt.legend()
        
        # Plot communication overhead
        plt.subplot(2, 2, 3)
        plt.bar(result.devices, result.communication_overhead)
        plt.xlabel('Number of GPUs')
        plt.ylabel('Communication Overhead')
        plt.title('Communication Overhead')
        plt.grid(True, axis='y')
        
        # Plot bottlenecks
        plt.subplot(2, 2, 4)
        bottleneck_df = pd.DataFrame({
            'devices': result.devices,
            'bottleneck': result.bottlenecks
        })
        sns.countplot(x='bottleneck', data=bottleneck_df)
        plt.title('Bottleneck Distribution')
        plt.xlabel('Bottleneck Type')
        plt.ylabel('Count')
        
        plt.tight_layout()
        return plt
        
    def _run_iteration(self, model, inputs):
        """Run a single training iteration"""
        if self.framework == "pytorch":
            with torch.no_grad():
                model(inputs)
                torch.cuda.synchronize()
                
    def _run_iteration_with_timing(self, model, inputs):
        """Run iteration and measure computation vs. communication time"""
        start_time = time.time()
        
        if self.framework == "pytorch":
            with torch.no_grad():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    # Track communication time in DDP
                    torch.cuda.synchronize()
                    comp_start = time.time()
                    outputs = model(inputs)
                    torch.cuda.synchronize()
                    comp_time = time.time() - comp_start
                    
                    # Total time includes communication
                    torch.distributed.barrier()
                    total_time = time.time() - start_time
                    comm_time = total_time - comp_time
                else:
                    outputs = model(inputs)
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    comm_time = 0
        else:
            # Handle other frameworks
            total_time = time.time() - start_time
            comm_time = 0
            
        return total_time, comm_time
        
    def _measure_memory_utilization(self, device_count):
        """Measure GPU memory utilization"""
        total_utilization = 0
        
        for i in range(min(device_count, torch.cuda.device_count())):
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            utilization = reserved / total
            total_utilization += utilization
            
        return total_utilization / device_count  # Average utilization
        
    def _setup_distributed_environment(self, world_size):
        """Set up distributed environment for PyTorch"""
        import os
        import torch.distributed as dist
        
        if self.framework == "pytorch":
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = '0'
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', world_size=world_size, rank=0)
                
    def _teardown_distributed_environment(self):
        """Clean up distributed environment"""
        if self.framework == "pytorch":
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                
    def _recommend_parallelism_strategy(self, devices, efficiencies, comm_overheads, memory_utils, bottlenecks):
        """Generate recommendations for optimal parallelism strategy"""
        recommendations = {}
        
        # Determine optimal device count for data parallelism
        optimal_dp_idx = 0
        for i, efficiency in enumerate(efficiencies):
            if efficiency >= 0.8:  # Consider 80% efficiency as good
                optimal_dp_idx = i
            else:
                break
                
        optimal_dp_count = devices[optimal_dp_idx]
        recommendations['optimal_device_count'] = optimal_dp_count
        
        # Check if model is communication bound
        is_comm_bound = any(o > 0.3 for o in comm_overheads)
        
        # Check if model is memory bound
        is_memory_bound = any(m > 0.9 for m in memory_utils)
        
        # Generate strategy recommendations
        if is_memory_bound:
            recommendations['primary_strategy'] = 'model_parallelism'
            recommendations['description'] = (
                "Model appears memory-bound. Consider using model parallelism or pipeline "
                "parallelism to distribute model state. Alternatively, try gradient checkpointing "
                "to reduce memory usage at the cost of recomputation."
            )
            recommendations['code_example'] = """
# Example of gradient checkpointing to reduce memory usage
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Split model into chunks for checkpointing
        self.seq1 = model.layers[:len(model.layers)//2]
        self.seq2 = model.layers[len(model.layers)//2:]
        
    def forward(self, x):
        x = checkpoint(self._forward_seq1, x)
        x = checkpoint(self._forward_seq2, x)
        return x
        
    def _forward_seq1(self, x):
        for layer in self.seq1:
            x = layer(x)
        return x
        
    def _forward_seq2(self, x):
        for layer in self.seq2:
            x = layer(x)
        return x
        
checkpointed_model = CheckpointedModel(model)
"""
        elif is_comm_bound:
            recommendations['primary_strategy'] = 'gradient_accumulation'
            recommendations['description'] = (
                f"Model appears communication-bound beyond {optimal_dp_count} GPUs. "
                "Consider using gradient accumulation to reduce communication frequency, "
                "or use mixed-precision training to reduce communication volume."
            )
            recommendations['code_example'] = """
# Example of gradient accumulation to reduce communication overhead
accumulation_steps = 4  # Accumulate gradients over 4 batches
model.zero_grad()

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()
"""
        else:
            recommendations['primary_strategy'] = 'data_parallelism'
            recommendations['description'] = (
                f"Model scales efficiently up to {optimal_dp_count} GPUs with standard data parallelism. "
                "Consider increasing batch size and learning rate proportionally to device count."
            )
            recommendations['code_example'] = """
# Example of scaled learning rate with data parallelism
base_lr = 0.01
base_batch_size = 32
devices = torch.cuda.device_count()

# Scale learning rate linearly with batch size
scaled_lr = base_lr * (devices * base_batch_size) / base_batch_size
optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr)

# Create DistributedDataParallel model
model = torch.nn.parallel.DistributedDataParallel(model)
"""

        # Advanced recommendations for very large models
        if is_memory_bound and is_comm_bound:
            recommendations['advanced_strategy'] = 'hybrid_parallelism'
            recommendations['advanced_description'] = (
                "Consider a hybrid parallelism approach using libraries like DeepSpeed or Megatron-LM "
                "that combine data, tensor, and pipeline parallelism for optimal scaling."
            )
            recommendations['advanced_code_example'] = """
# Example using DeepSpeed for hybrid parallelism
import deepspeed

# Define DeepSpeed configuration
ds_config = {
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "pipeline": {
        "enabled": True,
        "stages": 2
    }
}

# Initialize DeepSpeed model
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
"""
        
        return recommendations