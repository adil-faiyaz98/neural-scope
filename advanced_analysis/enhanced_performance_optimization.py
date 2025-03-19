# Add required imports at the top of the file
import subprocess
try:
    import py3nvml
    from py3nvml import py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("py3nvml not available. GPU power tracking will be limited.")

try:
    import codecarbon
    from codecarbon import EmissionsTracker
    CARBON_TRACKING_AVAILABLE = True
except ImportError:
    CARBON_TRACKING_AVAILABLE = False
    logger.warning("codecarbon not available. Carbon footprint analysis will be disabled.")

# Import cloud profiler if available
try:
    from .cloud_profiler import CloudProfiler
    CLOUD_PROFILER_AVAILABLE = True
except ImportError:
    CLOUD_PROFILER_AVAILABLE = False
    logger.warning("CloudProfiler not available. Remote cloud profiling will be disabled.")


class ModelPerformanceProfiler:
    """Enhanced performance profiler with multi-GPU, cloud and energy tracking capabilities"""
    
    def __init__(self, model, framework=None, model_name=None, cloud_credentials_path=None):
        # ... existing initialization code ...
        
        # Add new attributes
        self.power_measurements = []
        self.emissions_data = None
        self.distributed_stats = {}
        self.cloud_profiling_results = {}
        self.cloud_profiler = None
        
        # Initialize cloud profiler if available
        if CLOUD_PROFILER_AVAILABLE and cloud_credentials_path:
            try:
                self.cloud_profiler = CloudProfiler(credentials_path=cloud_credentials_path)
            except Exception as e:
                logger.warning(f"Failed to initialize CloudProfiler: {e}")

    def profile(self, input_data, batch_size=None, num_steps=10, 
                track_power=False, track_emissions=False, profile_nccl=False, 
                use_cloud=False, cloud_provider=None, instance_type=None):
        """
        Enhanced profile method with additional tracking options
        
        Args:
            input_data: Input data for model inference
            batch_size: Batch size for profiling
            num_steps: Number of profiling steps
            track_power: Whether to track power consumption
            track_emissions: Whether to track carbon emissions
            profile_nccl: Whether to profile NCCL communications
            use_cloud: Whether to use cloud profiling
            cloud_provider: Cloud provider to use ('aws', 'gcp', 'azure')
            instance_type: Specific instance type to use
            
        Returns:
            ProfilingResult with detailed performance metrics
        """
        # Handle cloud profiling request
        if use_cloud and self.cloud_profiler:
            return self._profile_on_cloud(input_data, batch_size, cloud_provider, instance_type)
            
        # Start emission tracking if requested
        emissions_tracker = None
        if track_emissions and CARBON_TRACKING_AVAILABLE:
            try:
                emissions_tracker = EmissionsTracker(project_name=self.model_name or "neural_scope_model")
                emissions_tracker.start()
            except Exception as e:
                logger.warning(f"Failed to start emissions tracking: {e}")
        
        # Initialize power tracking if requested
        if track_power and NVML_AVAILABLE and torch.cuda.is_available():
            try:
                nvml.nvmlInit()
                self._start_power_monitoring()
            except Exception as e:
                logger.warning(f"Failed to initialize power monitoring: {e}")
                
        # Profile NCCL if requested and we have a distributed setup
        if profile_nccl and self._is_distributed_setup():
            self._setup_nccl_profiling()
            
        # Run the actual profiling process
        if self.framework == "pytorch":
            result = self._profile_pytorch(input_data, batch_size, num_steps)
        elif self.framework == "tensorflow":
            result = self._profile_tensorflow(input_data, batch_size, num_steps)
        else:
            result = self._profile_generic(input_data, batch_size, num_steps)
            
        # Add NCCL profiling results if available
        if profile_nccl and self._is_distributed_setup():
            result.distributed_stats = self._gather_nccl_metrics()
            
        # Stop power monitoring if active
        if track_power and NVML_AVAILABLE and torch.cuda.is_available():
            try:
                power_stats = self._stop_power_monitoring()
                if power_stats:
                    result.power_metrics = power_stats
            except Exception as e:
                logger.warning(f"Failed to stop power monitoring: {e}")
                
        # Stop emissions tracking if active
        if emissions_tracker:
            try:
                emissions = emissions_tracker.stop()
                result.emissions = emissions
            except Exception as e:
                logger.warning(f"Failed to stop emissions tracking: {e}")
                
        # Augment recommendations with energy and distributed training insights
        self._augment_recommendations(result)
        
        return result
        
    def _profile_on_cloud(self, input_data, batch_size=None, cloud_provider=None, instance_type=None):
        """Profile the model on cloud infrastructure"""
        if not self.cloud_profiler:
            raise ValueError("Cloud profiler not available. Install required dependencies.")
            
        # Save model to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
            torch.save(self.model, model_path)
            
        # Determine input shape from sample input
        if isinstance(input_data, torch.Tensor):
            input_shape = tuple(input_data.shape)
        elif isinstance(input_data, (list, tuple)) and isinstance(input_data[0], torch.Tensor):
            input_shape = tuple(input_data[0].shape)
        else:
            raise ValueError("Couldn't determine input shape for cloud profiling")
            
        try:
            # Run profiling on cloud
            result = self.cloud_profiler.profile_on_cloud(
                model_path=model_path,
                input_shape=input_shape,
                cloud_provider=cloud_provider or 'aws',
                instance_type=instance_type,
                runtime=self.framework
            )
            
            # Convert cloud result to our format
            profiling_result = ProfilingResult(
                execution_time=result.get('execution_time', 0),
                memory_usage={'gpu': result.get('memory_usage', 0)},
                cpu_utilization={'mean': 0, 'max': 0},
                gpu_utilization={'mean': 0, 'max': 0},
                operation_stats={},
                bottlenecks=[],
                recommendations=[{
                    'type': 'cloud_deployment',
                    'severity': 'info',
                    'description': f"Model profiled on {result.get('cloud_provider')} {result.get('instance_type')}",
                    'recommendation': f"Estimated cost: ${result.get('cost'):.2f}, Throughput: {result.get('throughput'):.2f} inferences/sec"
                }]
            )
            
            # Store result for later comparison
            self.cloud_profiling_results[f"{cloud_provider}_{instance_type}"] = result
            
            return profiling_result
            
        except Exception as e:
            logger.error(f"Cloud profiling failed: {e}")
            raise
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def _is_distributed_setup(self):
        """Detect if the model is using distributed training"""
        if self.framework == "pytorch":
            # Check for DDP or other distributed modules
            if torch.cuda.device_count() > 1:
                # Check if model is wrapped in DistributedDataParallel
                from torch.nn.parallel import DistributedDataParallel
                if isinstance(self.model, DistributedDataParallel):
                    return True
                    
                # Check for other distributed modules or environment vars
                import os
                if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
                    return True
                    
        elif self.framework == "tensorflow":
            try:
                # Check TF distribution strategy
                import tensorflow as tf
                if isinstance(tf.distribute.get_strategy(), 
                             (tf.distribute.MirroredStrategy, 
                              tf.distribute.MultiWorkerMirroredStrategy,
                              tf.distribute.experimental.ParameterServerStrategy)):
                    return True
            except:
                pass
                
        return False
        
    def _setup_nccl_profiling(self):
        """Set up NCCL profiling for distributed training"""
        if self.framework == "pytorch":
            # Enable NCCL debugging and profiling
            import os
            os.environ['NCCL_DEBUG'] = 'INFO'
            if NVML_AVAILABLE:
                os.environ['NCCL_P2P_LEVEL'] = 'NVL'
                
            # For PyTorch, set up communication hooks if possible
            try:
                # PyTorch 1.8+ provides communication hook API
                if hasattr(torch.distributed, 'register_comm_hook'):
                    def comm_hook(state, bucket):
                        # Record communication stats
                        tensor_size = sum(p.numel() for p in bucket.get_tensors())
                        key = f"allreduce_{bucket.index}"
                        
                        if key not in self.distributed_stats:
                            self.distributed_stats[key] = {
                                'count': 0,
                                'total_size': 0,
                                'tensor_sizes': []
                            }
                            
                        self.distributed_stats[key]['count'] += 1
                        self.distributed_stats[key]['total_size'] += tensor_size
                        self.distributed_stats[key]['tensor_sizes'].append(tensor_size)
                        
                        # Continue with default all-reduce
                        return torch.distributed.all_reduce(bucket.buffer_)
                        
                    # Register hook if model is DDP
                    import torch.distributed as dist
                    from torch.nn.parallel import DistributedDataParallel
                    if isinstance(self.model, DistributedDataParallel) and dist.is_initialized():
                        self.model.register_comm_hook(state=None, hook=comm_hook)
            except Exception as e:
                logger.warning(f"Failed to register NCCL communication hook: {e}")
                
    def _gather_nccl_metrics(self):
        """Gather and analyze NCCL communication metrics"""
        nccl_stats = {}
        
        # Basic stats we've collected via hooks
        if self.distributed_stats:
            nccl_stats.update(self.distributed_stats)
            
        # Get more detailed stats if possible via NCCL debug logs
        try:
            import re
            import subprocess
            
            # This assumes NCCL_DEBUG=INFO was set earlier
            nccl_log_pattern = r"NCCL INFO.+\s(\w+)\s+\d+ -> \d+ \[(\d+)\]"
            
            # Try to extract NCCL logs from dmesg
            try:
                result = subprocess.run(["dmesg"], capture_output=True, text=True)
                if result.returncode == 0:
                    logs = result.stdout
                    matches = re.finditer(nccl_log_pattern, logs)
                    for match in matches:
                        op_type, size = match.groups()
                        if op_type not in nccl_stats:
                            nccl_stats[op_type] = {'count': 0, 'total_size': 0}
                        nccl_stats[op_type]['count'] += 1
                        nccl_stats[op_type]['total_size'] += int(size)
            except:
                pass
                
            # Add P2P bandwidth information if available
            if NVML_AVAILABLE and torch.cuda.is_available():
                try:
                    device_count = torch.cuda.device_count()
                    if device_count > 1:
                        p2p_matrix = []
                        for i in range(device_count):
                            row = []
                            for j in range(device_count):
                                if i == j:
                                    row.append(None)  # No P2P with self
                                else:
                                    can_access = torch.cuda.can_device_access_peer(i, j)
                                    row.append({"direct_access": can_access})
                            p2p_matrix.append(row)
                        nccl_stats['p2p_connectivity'] = p2p_matrix
                except Exception as e:
                    logger.warning(f"Failed to gather P2P connectivity info: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to extract detailed NCCL metrics: {e}")
            
        # Analyze topology if possible
        try:
            if torch.cuda.is_available():
                # Try to get NVLink info if available
                has_nvlink = False
                nvlink_info = {}
                
                if NVML_AVAILABLE:
                    try:
                        for i in range(torch.cuda.device_count()):
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            links = {}
                            for j in range(torch.cuda.device_count()):
                                if i != j:
                                    try:
                                        link = nvml.nvmlDeviceGetNvLinkState(handle, j)
                                        if link == nvml.NVML_NVLINK_STATE_ACTIVE:
                                            links[j] = True
                                            has_nvlink = True
                                    except:
                                        links[j] = False
                            nvlink_info[i] = links
                    except:
                        pass
                
                if has_nvlink:
                    nccl_stats['nvlink_topology'] = nvlink_info
        except Exception as e:
            logger.warning(f"Failed to analyze GPU topology: {e}")
            
        return nccl_stats
        
    def _start_power_monitoring(self):
        """Start monitoring power usage of GPUs"""
        self.power_measurements = []
        self.power_monitoring_start = time.time()
        
        if NVML_AVAILABLE and torch.cuda.is_available():
            # Get device handles
            self.gpu_handles = []
            try:
                for i in range(torch.cuda.device_count()):
                    self.gpu_handles.append(nvml.nvmlDeviceGetHandleByIndex(i))
                    
                # Start a thread to monitor power
                import threading
                import time
                
                def power_monitor():
                    while getattr(self, 'power_monitoring_active', True):
                        measurement = {"timestamp": time.time()}
                        
                        for i, handle in enumerate(self.gpu_handles):
                            try:
                                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                                measurement[f"gpu_{i}_power"] = power
                            except:
                                pass
                                
                        self.power_measurements.append(measurement)
                        time.sleep(0.1)  # Sample every 100ms
                
                self.power_monitoring_active = True
                self.power_monitor_thread = threading.Thread(target=power_monitor)
                self.power_monitor_thread.daemon = True
                self.power_monitor_thread.start()
            except Exception as e:
                logger.warning(f"Failed to start power monitoring: {e}")
                self.power_measurements = []
                
    def _stop_power_monitoring(self):
        """Stop power monitoring and calculate metrics"""
        if not hasattr(self, 'power_monitoring_active'):
            return None
            
        self.power_monitoring_active = False
        if hasattr(self, 'power_monitor_thread'):
            self.power_monitor_thread.join(timeout=2.0)
            
        if not self.power_measurements:
            return None
            
        # Calculate statistics
        power_stats = {
            'duration': time.time() - self.power_monitoring_start,
            'per_gpu': {},
            'total': {}
        }
        
        # Process per-GPU measurements
        for i in range(torch.cuda.device_count()):
            gpu_key = f"gpu_{i}_power"
            values = [m[gpu_key] for m in self.power_measurements if gpu_key in m]
            
            if values:
                power_stats['per_gpu'][i] = {
                    'mean_watts': sum(values) / len(values),
                    'peak_watts': max(values),
                    'min_watts': min(values),
                    'total_energy_kWh': sum(values) * 0.0000002777  # Convert W-100ms to kWh
                }
                
        # Calculate total power across all GPUs
        total_values = []
        for m in self.power_measurements:
            total = sum(m.get(f"gpu_{i}_power", 0) for i in range(torch.cuda.device_count()))
            total_values.append(total)
            
        if total_values:
            power_stats['total'] = {
                'mean_watts': sum(total_values) / len(total_values),
                'peak_watts': max(total_values),
                'total_energy_kWh': sum(total_values) * 0.0000002777,  # Convert W-100ms to kWh
                'measurement_count': len(total_values)
            }
            
        # Clean up
        if NVML_AVAILABLE:
            try:
                nvml.nvmlShutdown()
            except:
                pass
                
        return power_stats
        
    def _augment_recommendations(self, result):
        """Add energy and distributed training recommendations"""
        # Check if we have power metrics
        if hasattr(result, 'power_metrics') and result.power_metrics:
            # Calculate carbon footprint if we know energy usage
            if 'total' in result.power_metrics and 'total_energy_kWh' in result.power_metrics['total']:
                energy_kWh = result.power_metrics['total']['total_energy_kWh']
                
                # Rough estimate using average grid carbon intensity (500g CO2/kWh)
                carbon_g = energy_kWh * 500
                
                # Add carbon footprint recommendation
                result.recommendations.append({
                    'type': 'energy_efficiency',
                    'severity': 'info',
                    'description': f"Estimated carbon footprint: {carbon_g:.2f}g CO2e",
                    'recommendation': "Consider running training in regions with cleaner energy grids like Quebec, Norway, or France to reduce emissions."
                })
                
                # High power usage recommendation
                if result.power_metrics['total']['mean_watts'] > 300:
                    result.recommendations.append({
                        'type': 'energy_efficiency',
                        'severity': 'medium',
                        'description': f"High power usage detected ({result.power_metrics['total']['mean_watts']:.1f}W avg, {result.power_metrics['total']['peak_watts']:.1f}W peak)",
                        'recommendation': "Consider using more efficient GPU architectures, mixed precision training, or pruning to reduce power consumption.",
                        'code_example': """
# Enable mixed precision training in PyTorch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, labels in data_loader:
    optimizer.zero_grad()
    
    # Run model in mixed precision
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # Scale gradients and optimize
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""
                    })
                
        # Add distributed training recommendations if applicable
        if hasattr(result, 'distributed_stats') and result.distributed_stats:
            # Check for suboptimal NCCL performance
            if 'allreduce' in result.distributed_stats:
                total_bytes = result.distributed_stats['allreduce'].get('total_size', 0)
                
                # If large amount of communication detected
                if total_bytes > 1e9:  # 1 GB
                    result.recommendations.append({
                        'type': 'distributed_training',
                        'severity': 'medium',
                        'description': f"High communication volume in distributed training ({total_bytes/1e9:.2f} GB)",
                        'recommendation': "Consider using gradient accumulation, gradient compression, or tensor parallelism to reduce communication overhead.",
                        'code_example': """
# Example of gradient accumulation to reduce communication frequency
accumulation_steps = 4  # Accumulate gradients for 4 steps before communication
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(data_loader):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
                    })
                    
            # Check for NVLink topology
            if 'nvlink_topology' in result.distributed_stats:
                # Check if any GPUs lack direct NVLink connections
                has_nvlink_issue = False
                for gpu_id, links in result.distributed_stats['nvlink_topology'].items():
                    if not all(links.values()):
                        has_nvlink_issue = True
                        break
                        
                if has_nvlink_issue:
                    result.recommendations.append({
                        'type': 'distributed_training',
                        'severity': 'medium',
                        'description': "Suboptimal GPU interconnect topology detected",
                        'recommendation': "Some GPUs don't have direct NVLink connections. Consider using tensor parallelism only between directly connected GPUs, or pipeline parallelism across nodes.",
                        'code_example': """
# Tensor parallelism example with PyTorch
import torch.nn.parallel as parallel

# Create model split across GPUs with NVLink
device_ids = [0, 1]  # GPUs with direct NVLink
model = parallel.DataParallel(model, device_ids=device_ids)
"""
                    })
                    
            # P2P connectivity issues
            if 'p2p_connectivity' in result.distributed_stats:
                p2p_issues = False
                for i, row in enumerate(result.distributed_stats['p2p_connectivity']):
                    for j, info in enumerate(row):
                        if info is not None and not info.get('direct_access', False):
                            p2p_issues = True
                            break
                            
                if p2p_issues:
                    result.recommendations.append({
                        'type': 'distributed_training',
                        'severity': 'medium',
                        'description': "P2P access not available between some GPUs",
                        'recommendation': "Enable UVA (Unified Virtual Addressing) or consider a different GPU topology. Use CUDA_VISIBLE_DEVICES to pick GPUs with optimal connectivity.",
                        'code_example': """
# Set environment variable before running your script
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'  # Select specific GPUs with better connectivity
"""
                    })