class CloudProfiler:
    """System for real-time model profiling across cloud providers"""
    
    SUPPORTED_CLOUDS = ['aws', 'gcp', 'azure']
    INSTANCE_TYPES = {
        'aws': {
            'gpu': ['p3.2xlarge', 'p3.8xlarge', 'p4d.24xlarge', 'g4dn.xlarge', 'g5.xlarge'],
            'cpu': ['c5.2xlarge', 'c5.9xlarge', 'c6g.4xlarge']
        },
        'gcp': {
            'gpu': ['n1-standard-8-nvidia-tesla-t4', 'n1-standard-8-nvidia-tesla-v100', 'a2-highgpu-1g'],
            'cpu': ['n2-standard-8', 'c2-standard-16', 'n2d-standard-32']
        },
        'azure': {
            'gpu': ['Standard_NC6s_v3', 'Standard_NC24rs_v3', 'Standard_ND40rs_v2'],
            'cpu': ['Standard_F8s_v2', 'Standard_F32s_v2', 'Standard_D32_v3']
        }
    }
    
    def __init__(self, credentials_path=None):
        self.credentials_path = credentials_path
        self.results = {}
        self.auth_clients = {}
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialize cloud clients based on available credentials"""
        try:
            if self.credentials_path:
                # AWS client
                try:
                    import boto3
                    import json
                    aws_creds = json.load(open(f"{self.credentials_path}/aws_credentials.json"))
                    self.auth_clients['aws'] = boto3.Session(
                        aws_access_key_id=aws_creds.get('access_key'),
                        aws_secret_access_key=aws_creds.get('secret_key'),
                        region_name=aws_creds.get('region', 'us-west-2')
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize AWS client: {e}")
                
                # GCP client
                try:
                    from google.cloud import compute_v1
                    import os
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{self.credentials_path}/gcp_credentials.json"
                    self.auth_clients['gcp'] = compute_v1.InstancesClient()
                except Exception as e:
                    logger.warning(f"Failed to initialize GCP client: {e}")
                
                # Azure client
                try:
                    from azure.identity import ClientSecretCredential
                    from azure.mgmt.compute import ComputeManagementClient
                    import json
                    azure_creds = json.load(open(f"{self.credentials_path}/azure_credentials.json"))
                    credential = ClientSecretCredential(
                        tenant_id=azure_creds.get('tenant_id'),
                        client_id=azure_creds.get('client_id'),
                        client_secret=azure_creds.get('client_secret')
                    )
                    self.auth_clients['azure'] = ComputeManagementClient(
                        credential=credential,
                        subscription_id=azure_creds.get('subscription_id')
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Azure client: {e}")
        except Exception as e:
            logger.warning(f"Cloud client initialization failed: {e}")
    
    def profile_on_cloud(self, model_path, input_shape, cloud_provider='aws', instance_type=None, 
                         runtime='pytorch', max_budget=None, region=None):
        """
        Run profiling of a model on specified cloud provider
        
        Args:
            model_path: Path to saved model file
            input_shape: Input tensor shape for profiling
            cloud_provider: Cloud provider (aws, gcp, azure)
            instance_type: Instance type to profile on (default=lowest cost GPU instance)
            runtime: 'pytorch' or 'tensorflow'
            max_budget: Maximum budget for profiling in USD
            region: Cloud provider region
            
        Returns:
            Profiling results and cost information
        """
        if cloud_provider not in self.SUPPORTED_CLOUDS:
            raise ValueError(f"Unsupported cloud provider. Choose from: {self.SUPPORTED_CLOUDS}")
            
        if cloud_provider not in self.auth_clients:
            raise ValueError(f"No credentials available for {cloud_provider}")
            
        # Select instance type if not specified
        if instance_type is None:
            instance_type = self._get_default_instance(cloud_provider)
            
        # Verify budget constraints
        estimated_cost = self._estimate_cost(cloud_provider, instance_type, duration_hours=0.5)
        if max_budget and estimated_cost > max_budget:
            raise ValueError(f"Estimated cost (${estimated_cost:.2f}) exceeds max budget (${max_budget:.2f})")
            
        # Package profiling task
        task_id = f"profile_{runtime}_{int(time.time())}"
        task_package = self._package_profiling_task(model_path, input_shape, runtime, task_id)
        
        # Execute profiling on cloud
        result = self._execute_cloud_profiling(
            cloud_provider=cloud_provider,
            instance_type=instance_type,
            task_package=task_package,
            region=region
        )
        
        # Parse and store results
        if result.get('status') == 'success':
            self.results[task_id] = {
                'cloud_provider': cloud_provider,
                'instance_type': instance_type,
                'runtime': runtime,
                'execution_time': result.get('execution_time'),
                'memory_usage': result.get('memory_usage'),
                'throughput': result.get('throughput'),
                'cost': result.get('cost'),
                'timestamp': time.time()
            }
            
        return self.results[task_id] if task_id in self.results else result
    
    def _package_profiling_task(self, model_path, input_shape, runtime, task_id):
        """Package model and profiling code for cloud execution"""
        # This is a simplified implementation - a real one would:
        # 1. Create a zip/docker container with model and profiling code
        # 2. Generate a cloud-specific deployment script
        # 3. Set up result collection mechanism
        
        package_dir = f"/tmp/neural_scope_cloud/{task_id}"
        os.makedirs(package_dir, exist_ok=True)
        
        # Write profiling script
        profiler_script = f"""
import torch
import time
import json
import os
import numpy as np

# Load model
model = torch.load("{model_path}")
model.eval()

# Create input tensor
input_data = torch.rand({input_shape})
if torch.cuda.is_available():
    model = model.cuda()
    input_data = input_data.cuda()

# Warmup
for _ in range(5):
    with torch.no_grad():
        model(input_data)

# Profiling
start_time = time.time()
iterations = 100
memory_usage = 0

with torch.no_grad():
    for i in range(iterations):
        model(input_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if i == 50:  # Measure halfway through
                memory_usage = torch.cuda.max_memory_allocated() / (1024**2)

execution_time = (time.time() - start_time) / iterations
throughput = iterations / (time.time() - start_time)

# Instance metadata
instance_info = {{}}
try:
    import requests
    if "{cloud_provider}" == "aws":
        r = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
        instance_info["type"] = r.text
    # Add similar blocks for GCP and Azure
except:
    pass

# Save results
results = {{
    "execution_time": execution_time,
    "memory_usage": memory_usage,
    "throughput": throughput,
    "instance_info": instance_info
}}

with open("profiling_results.json", "w") as f:
    json.dump(results, f)
"""
        
        with open(f"{package_dir}/profiler.py", "w") as f:
            f.write(profiler_script)
            
        # Write deployment script (specific to cloud provider)
        # For brevity, this is a placeholder
        
        return {
            'package_path': package_dir,
            'main_script': 'profiler.py',
            'task_id': task_id
        }
    
    def _execute_cloud_profiling(self, cloud_provider, instance_type, task_package, region=None):
        """Execute profiling task on specified cloud provider"""
        # This is a simplified implementation - a real one would:
        # 1. Spin up a cloud instance of the requested type
        # 2. Upload and execute the profiling package
        # 3. Collect results
        # 4. Terminate the instance
        
        # For demonstration, we'll return simulated results
        import random
        
        # Simulated variation by instance type
        instance_perf = {
            # AWS P3 instances (V100)
            'p3.2xlarge': {'base_time': 0.015, 'memory': 12000},
            'p3.8xlarge': {'base_time': 0.014, 'memory': 40000},
            'p4d.24xlarge': {'base_time': 0.008, 'memory': 38000},
            
            # AWS G4/G5 instances (T4/A10G)
            'g4dn.xlarge': {'base_time': 0.025, 'memory': 8000},
            'g5.xlarge': {'base_time': 0.018, 'memory': 10000},
            
            # GCP instances
            'n1-standard-8-nvidia-tesla-t4': {'base_time': 0.024, 'memory': 8000},
            'n1-standard-8-nvidia-tesla-v100': {'base_time': 0.016, 'memory': 14000},
            'a2-highgpu-1g': {'base_time': 0.013, 'memory': 18000},
            
            # Azure instances
            'Standard_NC6s_v3': {'base_time': 0.017, 'memory': 12000},
            'Standard_NC24rs_v3': {'base_time': 0.015, 'memory': 35000},
            'Standard_ND40rs_v2': {'base_time': 0.009, 'memory': 36000},
        }
        
        # Get instance performance baseline
        perf = instance_perf.get(instance_type, {'base_time': 0.02, 'memory': 10000})
        
        # Simulate execution with some randomness
        execution_time = perf['base_time'] * (1 + random.uniform(-0.1, 0.1))
        memory_usage = perf['memory'] * (1 + random.uniform(-0.05, 0.05))
        throughput = 1.0 / execution_time
        
        # Simulate costs
        cost = self._estimate_cost(cloud_provider, instance_type, duration_hours=0.5)
        
        # Return simulated results
        return {
            'status': 'success',
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'cost': cost,
            'cloud_provider': cloud_provider,
            'instance_type': instance_type
        }
    
    def _get_default_instance(self, cloud_provider):
        """Get default instance type for a cloud provider"""
        if cloud_provider in self.INSTANCE_TYPES:
            return self.INSTANCE_TYPES[cloud_provider]['gpu'][0]
        return None
    
    def _estimate_cost(self, cloud_provider, instance_type, duration_hours=1.0):
        """Estimate cost for running an instance for the given duration"""
        # Simplified cost model - actual implementation would use cloud provider APIs
        hourly_rates = {
            # AWS
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24,
            'p4d.24xlarge': 32.77,
            'g4dn.xlarge': 0.526,
            'g5.xlarge': 1.006,
            'c5.2xlarge': 0.34,
            'c5.9xlarge': 1.53,
            'c6g.4xlarge': 0.68,
            
            # GCP
            'n1-standard-8-nvidia-tesla-t4': 0.95,
            'n1-standard-8-nvidia-tesla-v100': 2.48,
            'a2-highgpu-1g': 3.67,
            'n2-standard-8': 0.38,
            'c2-standard-16': 0.87,
            'n2d-standard-32': 1.45,
            
            # Azure
            'Standard_NC6s_v3': 3.06,
            'Standard_NC24rs_v3': 13.5,
            'Standard_ND40rs_v2': 28.7,
            'Standard_F8s_v2': 0.42,
            'Standard_F32s_v2': 1.68,
            'Standard_D32_v3': 1.54
        }
        
        hourly_rate = hourly_rates.get(instance_type, 1.0)
        return hourly_rate * duration_hours
    
    def compare_cloud_providers(self, model_path, input_shape, runtime='pytorch', max_budget=100):
        """Profile model across different cloud providers and instance types within budget"""
        results = []
        
        for provider in self.SUPPORTED_CLOUDS:
            # Skip providers without credentials
            if provider not in self.auth_clients:
                continue
                
            # Try top 2 instances from each provider within budget
            for instance in self.INSTANCE_TYPES[provider]['gpu'][:2]:
                estimated_cost = self._estimate_cost(provider, instance, duration_hours=0.5)
                if estimated_cost <= max_budget:
                    try:
                        class CloudProfiler:
    """System for real-time model profiling across cloud providers"""
    
    SUPPORTED_CLOUDS = ['aws', 'gcp', 'azure']
    INSTANCE_TYPES = {
        'aws': {
            'gpu': ['p3.2xlarge', 'p3.8xlarge', 'p4d.24xlarge', 'g4dn.xlarge', 'g5.xlarge'],
            'cpu': ['c5.2xlarge', 'c5.9xlarge', 'c6g.4xlarge']
        },
        'gcp': {
            'gpu': ['n1-standard-8-nvidia-tesla-t4', 'n1-standard-8-nvidia-tesla-v100', 'a2-highgpu-1g'],
            'cpu': ['n2-standard-8', 'c2-standard-16', 'n2d-standard-32']
        },
        'azure': {
            'gpu': ['Standard_NC6s_v3', 'Standard_NC24rs_v3', 'Standard_ND40rs_v2'],
            'cpu': ['Standard_F8s_v2', 'Standard_F32s_v2', 'Standard_D32_v3']
        }
    }
    
    def __init__(self, credentials_path=None):
        self.credentials_path = credentials_path
        self.results = {}
        self.auth_clients = {}
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialize cloud clients based on available credentials"""
        try:
            if self.credentials_path:
                # AWS client
                try:
                    import boto3
                    import json
                    aws_creds = json.load(open(f"{self.credentials_path}/aws_credentials.json"))
                    self.auth_clients['aws'] = boto3.Session(
                        aws_access_key_id=aws_creds.get('access_key'),
                        aws_secret_access_key=aws_creds.get('secret_key'),
                        region_name=aws_creds.get('region', 'us-west-2')
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize AWS client: {e}")
                
                # GCP client
                try:
                    from google.cloud import compute_v1
                    import os
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{self.credentials_path}/gcp_credentials.json"
                    self.auth_clients['gcp'] = compute_v1.InstancesClient()
                except Exception as e:
                    logger.warning(f"Failed to initialize GCP client: {e}")
                
                # Azure client
                try:
                    from azure.identity import ClientSecretCredential
                    from azure.mgmt.compute import ComputeManagementClient
                    import json
                    azure_creds = json.load(open(f"{self.credentials_path}/azure_credentials.json"))
                    credential = ClientSecretCredential(
                        tenant_id=azure_creds.get('tenant_id'),
                        client_id=azure_creds.get('client_id'),
                        client_secret=azure_creds.get('client_secret')
                    )
                    self.auth_clients['azure'] = ComputeManagementClient(
                        credential=credential,
                        subscription_id=azure_creds.get('subscription_id')
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Azure client: {e}")
        except Exception as e:
            logger.warning(f"Cloud client initialization failed: {e}")
    
    def profile_on_cloud(self, model_path, input_shape, cloud_provider='aws', instance_type=None, 
                         runtime='pytorch', max_budget=None, region=None):
        """
        Run profiling of a model on specified cloud provider
        
        Args:
            model_path: Path to saved model file
            input_shape: Input tensor shape for profiling
            cloud_provider: Cloud provider (aws, gcp, azure)
            instance_type: Instance type to profile on (default=lowest cost GPU instance)
            runtime: 'pytorch' or 'tensorflow'
            max_budget: Maximum budget for profiling in USD
            region: Cloud provider region
            
        Returns:
            Profiling results and cost information
        """
        if cloud_provider not in self.SUPPORTED_CLOUDS:
            raise ValueError(f"Unsupported cloud provider. Choose from: {self.SUPPORTED_CLOUDS}")
            
        if cloud_provider not in self.auth_clients:
            raise ValueError(f"No credentials available for {cloud_provider}")
            
        # Select instance type if not specified
        if instance_type is None:
            instance_type = self._get_default_instance(cloud_provider)
            
        # Verify budget constraints
        estimated_cost = self._estimate_cost(cloud_provider, instance_type, duration_hours=0.5)
        if max_budget and estimated_cost > max_budget:
            raise ValueError(f"Estimated cost (${estimated_cost:.2f}) exceeds max budget (${max_budget:.2f})")
            
        # Package profiling task
        task_id = f"profile_{runtime}_{int(time.time())}"
        task_package = self._package_profiling_task(model_path, input_shape, runtime, task_id)
        
        # Execute profiling on cloud
        result = self._execute_cloud_profiling(
            cloud_provider=cloud_provider,
            instance_type=instance_type,
            task_package=task_package,
            region=region
        )
        
        # Parse and store results
        if result.get('status') == 'success':
            self.results[task_id] = {
                'cloud_provider': cloud_provider,
                'instance_type': instance_type,
                'runtime': runtime,
                'execution_time': result.get('execution_time'),
                'memory_usage': result.get('memory_usage'),
                'throughput': result.get('throughput'),
                'cost': result.get('cost'),
                'timestamp': time.time()
            }
            
        return self.results[task_id] if task_id in self.results else result
    
    def _package_profiling_task(self, model_path, input_shape, runtime, task_id):
        """Package model and profiling code for cloud execution"""
        # This is a simplified implementation - a real one would:
        # 1. Create a zip/docker container with model and profiling code
        # 2. Generate a cloud-specific deployment script
        # 3. Set up result collection mechanism
        
        package_dir = f"/tmp/neural_scope_cloud/{task_id}"
        os.makedirs(package_dir, exist_ok=True)
        
        # Write profiling script
        profiler_script = f"""
import torch
import time
import json
import os
import numpy as np

# Load model
model = torch.load("{model_path}")
model.eval()

# Create input tensor
input_data = torch.rand({input_shape})
if torch.cuda.is_available():
    model = model.cuda()
    input_data = input_data.cuda()

# Warmup
for _ in range(5):
    with torch.no_grad():
        model(input_data)

# Profiling
start_time = time.time()
iterations = 100
memory_usage = 0

with torch.no_grad():
    for i in range(iterations):
        model(input_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if i == 50:  # Measure halfway through
                memory_usage = torch.cuda.max_memory_allocated() / (1024**2)

execution_time = (time.time() - start_time) / iterations
throughput = iterations / (time.time() - start_time)

# Instance metadata
instance_info = {{}}
try:
    import requests
    if "{cloud_provider}" == "aws":
        r = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
        instance_info["type"] = r.text
    # Add similar blocks for GCP and Azure
except:
    pass

# Save results
results = {{
    "execution_time": execution_time,
    "memory_usage": memory_usage,
    "throughput": throughput,
    "instance_info": instance_info
}}

with open("profiling_results.json", "w") as f:
    json.dump(results, f)
"""
        
        with open(f"{package_dir}/profiler.py", "w") as f:
            f.write(profiler_script)
            
        # Write deployment script (specific to cloud provider)
        # For brevity, this is a placeholder
        
        return {
            'package_path': package_dir,
            'main_script': 'profiler.py',
            'task_id': task_id
        }
    
    def _execute_cloud_profiling(self, cloud_provider, instance_type, task_package, region=None):
        """Execute profiling task on specified cloud provider"""
        # This is a simplified implementation - a real one would:
        # 1. Spin up a cloud instance of the requested type
        # 2. Upload and execute the profiling package
        # 3. Collect results
        # 4. Terminate the instance
        
        # For demonstration, we'll return simulated results
        import random
        
        # Simulated variation by instance type
        instance_perf = {
            # AWS P3 instances (V100)
            'p3.2xlarge': {'base_time': 0.015, 'memory': 12000},
            'p3.8xlarge': {'base_time': 0.014, 'memory': 40000},
            'p4d.24xlarge': {'base_time': 0.008, 'memory': 38000},
            
            # AWS G4/G5 instances (T4/A10G)
            'g4dn.xlarge': {'base_time': 0.025, 'memory': 8000},
            'g5.xlarge': {'base_time': 0.018, 'memory': 10000},
            
            # GCP instances
            'n1-standard-8-nvidia-tesla-t4': {'base_time': 0.024, 'memory': 8000},
            'n1-standard-8-nvidia-tesla-v100': {'base_time': 0.016, 'memory': 14000},
            'a2-highgpu-1g': {'base_time': 0.013, 'memory': 18000},
            
            # Azure instances
            'Standard_NC6s_v3': {'base_time': 0.017, 'memory': 12000},
            'Standard_NC24rs_v3': {'base_time': 0.015, 'memory': 35000},
            'Standard_ND40rs_v2': {'base_time': 0.009, 'memory': 36000},
        }
        
        # Get instance performance baseline
        perf = instance_perf.get(instance_type, {'base_time': 0.02, 'memory': 10000})
        
        # Simulate execution with some randomness
        execution_time = perf['base_time'] * (1 + random.uniform(-0.1, 0.1))
        memory_usage = perf['memory'] * (1 + random.uniform(-0.05, 0.05))
        throughput = 1.0 / execution_time
        
        # Simulate costs
        cost = self._estimate_cost(cloud_provider, instance_type, duration_hours=0.5)
        
        # Return simulated results
        return {
            'status': 'success',
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'cost': cost,
            'cloud_provider': cloud_provider,
            'instance_type': instance_type
        }
    
    def _get_default_instance(self, cloud_provider):
        """Get default instance type for a cloud provider"""
        if cloud_provider in self.INSTANCE_TYPES:
            return self.INSTANCE_TYPES[cloud_provider]['gpu'][0]
        return None
    
    def _estimate_cost(self, cloud_provider, instance_type, duration_hours=1.0):
        """Estimate cost for running an instance for the given duration"""
        # Simplified cost model - actual implementation would use cloud provider APIs
        hourly_rates = {
            # AWS
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24,
            'p4d.24xlarge': 32.77,
            'g4dn.xlarge': 0.526,
            'g5.xlarge': 1.006,
            'c5.2xlarge': 0.34,
            'c5.9xlarge': 1.53,
            'c6g.4xlarge': 0.68,
            
            # GCP
            'n1-standard-8-nvidia-tesla-t4': 0.95,
            'n1-standard-8-nvidia-tesla-v100': 2.48,
            'a2-highgpu-1g': 3.67,
            'n2-standard-8': 0.38,
            'c2-standard-16': 0.87,
            'n2d-standard-32': 1.45,
            
            # Azure
            'Standard_NC6s_v3': 3.06,
            'Standard_NC24rs_v3': 13.5,
            'Standard_ND40rs_v2': 28.7,
            'Standard_F8s_v2': 0.42,
            'Standard_F32s_v2': 1.68,
            'Standard_D32_v3': 1.54
        }
        
        hourly_rate = hourly_rates.get(instance_type, 1.0)
        return hourly_rate * duration_hours
    
    def compare_cloud_providers(self, model_path, input_shape, runtime='pytorch', max_budget=100):
        """Profile model across different cloud providers and instance types within budget"""
        results = []
        
        for provider in self.SUPPORTED_CLOUDS:
            # Skip providers without credentials
            if provider not in self.auth_clients:
                continue
                
            # Try top 2 instances from each provider within budget
            for instance in self.INSTANCE_TYPES[provider]['gpu'][:2]:
                estimated_cost = self._estimate_cost(provider, instance, duration_hours=0.5)
                if estimated_cost <= max_budget:
                    try:
                        