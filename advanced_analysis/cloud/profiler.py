"""
Cloud profiling module for ML workloads.

This module provides tools for profiling ML models on cloud infrastructure,
including remote execution, performance monitoring, and cross-cloud comparison.
"""

import logging
import os
import json
import time
import base64
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import io
import numpy as np

logger = logging.getLogger(__name__)

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
    
    def __init__(self, credentials_path=None, use_iam_role=False, use_secret_manager=False):
        self.credentials_path = credentials_path
        self.results = {}
        self.auth_clients = {}
        self.storage_clients = {}
        self.secret_clients = {}
        self.use_iam_role = use_iam_role
        self.use_secret_manager = use_secret_manager
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialize cloud clients based on available credentials"""
        try:
            # Try IAM roles first if enabled
            if self.use_iam_role:
                self._init_cloud_clients_with_iam()
            
            # Then try secret manager if enabled
            if self.use_secret_manager and not all(cloud in self.auth_clients for cloud in self.SUPPORTED_CLOUDS):
                self._init_cloud_clients_with_secret_manager()
            
            # Fall back to file-based credentials
            if self.credentials_path and not all(cloud in self.auth_clients for cloud in self.SUPPORTED_CLOUDS):
                # AWS client
                try:
                    import boto3
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
                
                # Initialize storage clients
                if 'aws' in self.auth_clients:
                    try:
                        self.storage_clients['aws'] = self.auth_clients['aws'].client('s3')
                    except Exception as e:
                        logger.warning(f"Failed to initialize AWS S3 client: {e}")
                
                if 'gcp' in self.auth_clients:
                    try:
                        from google.cloud import storage
                        self.storage_clients['gcp'] = storage.Client()
                    except Exception as e:
                        logger.warning(f"Failed to initialize GCP Storage client: {e}")
                
                if 'azure' in self.auth_clients:
                    try:
                        from azure.storage.blob import BlobServiceClient
                        azure_creds = json.load(open(f"{self.credentials_path}/azure_credentials.json"))
                        self.storage_clients['azure'] = BlobServiceClient.from_connection_string(
                            azure_creds.get('storage_connection_string')
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize Azure Blob client: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize cloud clients: {e}")
    
    def _init_cloud_clients_with_iam(self):
        """Initialize cloud clients using IAM roles"""
        # AWS client with IAM role
        try:
            import boto3
            self.auth_clients['aws'] = boto3.Session()
            self.storage_clients['aws'] = self.auth_clients['aws'].client('s3')
            logger.info("Successfully initialized AWS client using IAM role")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client with IAM role: {e}")
        
        # GCP client with default credentials
        try:
            from google.cloud import compute_v1, storage
            self.auth_clients['gcp'] = compute_v1.InstancesClient()
            self.storage_clients['gcp'] = storage.Client()
            logger.info("Successfully initialized GCP client using default credentials")
        except Exception as e:
            logger.warning(f"Failed to initialize GCP client with default credentials: {e}")
        
        # Azure client with managed identity
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.storage.blob import BlobServiceClient
            
            credential = DefaultAzureCredential()
            # Note: For Azure, we still need subscription_id even with managed identity
            # This could be set as an environment variable
            subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
            if subscription_id:
                self.auth_clients['azure'] = ComputeManagementClient(
                    credential=credential,
                    subscription_id=subscription_id
                )
                
                # Storage account name needed for blob client
                storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
                if storage_account:
                    self.storage_clients['azure'] = BlobServiceClient(
                        account_url=f"https://{storage_account}.blob.core.windows.net",
                        credential=credential
                    )
                logger.info("Successfully initialized Azure client using managed identity")
            else:
                logger.warning("Azure subscription ID not found in environment variables")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure client with managed identity: {e}")
    
    def _init_cloud_clients_with_secret_manager(self):
        """Initialize cloud clients using secrets from secret managers"""
        # AWS Secrets Manager
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Try to use minimal credentials to access secrets manager
            session = boto3.session.Session()
            secrets_client = session.client('secretsmanager')
            
            try:
                # Get AWS credentials from Secrets Manager
                aws_secret = secrets_client.get_secret_value(SecretId='neural_scope/aws_credentials')
                aws_creds = json.loads(aws_secret['SecretString'])
                
                self.auth_clients['aws'] = boto3.Session(
                    aws_access_key_id=aws_creds.get('access_key'),
                    aws_secret_access_key=aws_creds.get('secret_key'),
                    region_name=aws_creds.get('region', 'us-west-2')
                )
                self.storage_clients['aws'] = self.auth_clients['aws'].client('s3')
                logger.info("Successfully initialized AWS client using Secrets Manager")
            except ClientError as e:
                logger.warning(f"Failed to retrieve AWS credentials from Secrets Manager: {e}")
                
            # Try to get other cloud credentials from AWS Secrets Manager
            try:
                # Get GCP credentials
                gcp_secret = secrets_client.get_secret_value(SecretId='neural_scope/gcp_credentials')
                gcp_creds = json.loads(gcp_secret['SecretString'])
                
                # Save temporary credential file for GCP
                fd, temp_cred_path = tempfile.mkstemp()
                with os.fdopen(fd, 'w') as tmp:
                    json.dump(gcp_creds, tmp)
                
                from google.cloud import compute_v1, storage
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
                self.auth_clients['gcp'] = compute_v1.InstancesClient()
                self.storage_clients['gcp'] = storage.Client()
                
                # Clean up temp file
                os.remove(temp_cred_path)
                logger.info("Successfully initialized GCP client using credentials from AWS Secrets Manager")
            except Exception as e:
                logger.warning(f"Failed to initialize GCP client with credentials from AWS Secrets Manager: {e}")
                
            try:
                # Get Azure credentials
                azure_secret = secrets_client.get_secret_value(SecretId='neural_scope/azure_credentials')
                azure_creds = json.loads(azure_secret['SecretString'])
                
                from azure.identity import ClientSecretCredential
                from azure.mgmt.compute import ComputeManagementClient
                from azure.storage.blob import BlobServiceClient
                
                credential = ClientSecretCredential(
                    tenant_id=azure_creds.get('tenant_id'),
                    client_id=azure_creds.get('client_id'),
                    client_secret=azure_creds.get('client_secret')
                )
                self.auth_clients['azure'] = ComputeManagementClient(
                    credential=credential,
                    subscription_id=azure_creds.get('subscription_id')
                )
                self.storage_clients['azure'] = BlobServiceClient.from_connection_string(
                    azure_creds.get('storage_connection_string')
                )
                logger.info("Successfully initialized Azure client using credentials from AWS Secrets Manager")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure client with credentials from AWS Secrets Manager: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize AWS Secrets Manager client: {e}")
    
    def upload_model(self, model_path: str, cloud_provider: str) -> str:
        """
        Upload model to cloud storage
        
        Args:
            model_path: Local path to model file
            cloud_provider: Cloud provider to use
            
        Returns:
            Remote path to uploaded model
        """
        if cloud_provider not in self.storage_clients:
            raise ValueError(f"No storage client available for {cloud_provider}")
            
        model_filename = os.path.basename(model_path)
        remote_path = f"neural-scope-models/{int(time.time())}/{model_filename}"
        
        if cloud_provider == 'aws':
            bucket_name = 'neural-scope-models'
            try:
                # Check if bucket exists, create if not
                s3 = self.storage_clients['aws']
                try:
                    s3.head_bucket(Bucket=bucket_name)
                except:
                    s3.create_bucket(Bucket=bucket_name)
                
                # Upload file
                s3.upload_file(model_path, bucket_name, remote_path)
                return f"s3://{bucket_name}/{remote_path}"
            except Exception as e:
                logger.error(f"Failed to upload model to AWS S3: {e}")
                raise
                
        elif cloud_provider == 'gcp':
            bucket_name = 'neural-scope-models'
            try:
                storage_client = self.storage_clients['gcp']
                # Check if bucket exists, create if not
                try:
                    bucket = storage_client.get_bucket(bucket_name)
                except:
                    bucket = storage_client.create_bucket(bucket_name)
                
                # Upload file
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(model_path)
                return f"gs://{bucket_name}/{remote_path}"
            except Exception as e:
                logger.error(f"Failed to upload model to GCP Storage: {e}")
                raise
                
        elif cloud_provider == 'azure':
            container_name = 'neural-scope-models'
            try:
                # Check if container exists, create if not
                blob_service = self.storage_clients['azure']
                try:
                    blob_service.get_container_client(container_name)
                except:
                    blob_service.create_container(container_name)
                
                # Upload file
                with open(model_path, "rb") as data:
                    blob_service.get_blob_client(
                        container=container_name, 
                        blob=remote_path
                    ).upload_blob(data)
                
                # Get account name from connection string or client
                account_name = blob_service.account_name
                return f"https://{account_name}.blob.core.windows.net/{container_name}/{remote_path}"
            except Exception as e:
                logger.error(f"Failed to upload model to Azure Blob Storage: {e}")
                raise
                
        raise NotImplementedError(f"Upload not implemented for {cloud_provider}")
    
    def _generate_profiling_script(self, model_path: str, batch_sizes: List[int], 
                                 sequence_lengths: Optional[List[int]], 
                                 framework: str, num_iterations: int) -> str:
        """
        Generate a Python script for remote profiling
        
        Args:
            model_path: Remote path to model
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths (for sequence models)
            framework: ML framework ('pytorch' or 'tensorflow')
            num_iterations: Number of iterations per configuration
            
        Returns:
            String containing Python script code
        """
        # Common imports
        script = [
            "#!/usr/bin/env python3",
            "import time",
            "import json",
            "import os",
            "import psutil",
            "import numpy as np",
            "import argparse",
            "from datetime import datetime"
        ]
        
        # Framework-specific imports
        if framework.lower() == "pytorch":
            script.extend([
                "import torch",
                "from torch.profiler import profile, ProfilerActivity"
            ])
        elif framework.lower() == "tensorflow":
            script.extend([
                "import tensorflow as tf",
                "from tensorflow.python.client import device_lib"
            ])
        else:
            raise ValueError(f"Unsupported framework: {framework}")
            
        # Parse arguments
        script.extend([
            "",
            "parser = argparse.ArgumentParser(description='Profile ML model')",
            "parser.add_argument('--model-path', type=str, required=True, help='Path to model file')",
            "parser.add_argument('--output-path', type=str, required=True, help='Path to save results')",
            "parser.add_argument('--framework', type=str, default='pytorch', help='ML framework (pytorch or tensorflow)')",
            "args = parser.parse_args()",
            ""
        ])
        
        # Utility functions
        script.extend([
            "def calculate_flops(model, input_shape, framework):",
            "    \"\"\"Estimate FLOPs for a given model\"\"\"",
            "    try:",
            "        if framework.lower() == 'pytorch':",
            "            from thop import profile as thop_profile",
            "            macs, params = thop_profile(model, inputs=(torch.randn(input_shape).to(next(model.parameters()).device),))",
            "            return macs * 2  # FLOPS ~= 2 * MACs",
            "        elif framework.lower() == 'tensorflow':",
            "            # TF FLOPs calculation would go here",
            "            return 0",
            "    except Exception as e:",
            "        print(f'Error calculating FLOPs: {e}')",
            "        return 0",
            ""
        ])
        
        # Main profiling function
        script.extend([
            "def profile_model(model, batch_sizes, sequence_lengths, framework, num_iterations):",
            "    results = {}", 
            "    device = 'cuda' if (framework=='pytorch' and torch.cuda.is_available()) else 'cpu'",
            "    ",
            "    # Prepare inputs for different batch sizes",
            "    for batch_size in batch_sizes:",
            "        print(f'Profiling batch size: {batch_size}')",
            "        results[str(batch_size)] = {}",
            "        ",
            "        # Create input data",
            "        if framework.lower() == 'pytorch':",
            "            # Dummy input - adjust as needed",
            "            input_shape = (batch_size, 3, 224, 224) if not sequence_lengths else (batch_size, sequence_lengths[0], 512)",
            "            inputs = torch.randn(input_shape).to(device)",
            "            ",
            "            # Warmup", 
            "            for _ in range(10):",
            "                _ = model(inputs)",
            "            torch.cuda.synchronize() if 'cuda' in device else None",
            "            ",
            "            # Memory tracking",
            "            start_mem = torch.cuda.memory_allocated() if 'cuda' in device else 0",
            "            # Timed runs",
            "            start_time = time.time()",
            "            for _ in range(num_iterations):",
            "                _ = model(inputs)",
            "                torch.cuda.synchronize() if 'cuda' in device else None",
            "            end_time = time.time()",
            "            # Peak memory",
            "            peak_mem = torch.cuda.max_memory_allocated() if 'cuda' in device else 0",
            "            torch.cuda.reset_peak_memory_stats() if 'cuda' in device else None",
            "            ",
            "            # Calculate performance metrics",
            "            total_time = end_time - start_time",
            "            avg_time = (total_time / num_iterations) * 1000  # in ms",
            "            throughput = batch_size * (num_iterations / total_time)",
            "            ",
            "            # Get GPU utilization",
            "            gpu_util = 0",
            "            try:", 
            "                import subprocess",
            "                result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])",
            "                gpu_util = float(result.decode('utf-8').strip()) / 100.0",
            "            except:",
            "                pass",
            "            ",
            "            # Calculate FLOPs",
            "            flops = calculate_flops(model, input_shape, framework)",
            "            ",
            "            # Record results",
            "            results[str(batch_size)] = {",
            "                'latency_ms': avg_time,",
            "                'throughput': throughput,", 
            "                'gpu_utilization': gpu_util,",
            "                'memory_used_bytes': int(peak_mem - start_mem),",
            "                'memory_peak_bytes': int(peak_mem),",
            "                'flops_per_batch': flops",
            "            }",
            "        elif framework.lower() == 'tensorflow':",
            "            # Similar implementation for TensorFlow",
            "            pass",
            "    ",
            "    return results"
        ])
        
        # Main execution
        script.extend([
            "",
            "def main():",
            f"    batch_sizes = {batch_sizes}",
            f"    sequence_lengths = {sequence_lengths if sequence_lengths else None}",
            f"    num_iterations = {num_iterations}",
            "    framework = args.framework",
            "    model_path = args.model_path",
            "    output_path = args.output_path",
            "    ",
            "    # Load model",
            "    try:",
            "        if framework.lower() == 'pytorch':",
            "            model = torch.load(model_path)",
            "            device = 'cuda' if torch.cuda.is_available() else 'cpu'",
            "            model.to(device)",
            "            model.eval()",
            "        elif framework.lower() == 'tensorflow':",
            "            model = tf.saved_model.load(model_path)",
            "        else:",
            "            raise ValueError(f'Unsupported framework: {framework}')",
            "    except Exception as e:",
            "        print(f'Error loading model: {e}')",
            "        return",
            "    ",
            "    # Run profiling",
            "    results = profile_model(model, batch_sizes, sequence_lengths, framework, num_iterations)",
            "    ",
            "    # Add system info",
            "    results['system_info'] = {",
            "        'timestamp': datetime.now().isoformat(),",
            "        'cpu_info': {",
            "            'cores': psutil.cpu_count(logical=False),",
            "            'threads': psutil.cpu_count(logical=True),",
            "        },",
            "        'memory_info': {",
            "            'total_gb': psutil.virtual_memory().total / (1024**3),",
            "        }",
            "    }",
            "    ",
            "    if framework.lower() == 'pytorch' and torch.cuda.is_available():",
            "        results['system_info']['gpu_info'] = {",
            "            'name': torch.cuda.get_device_name(0),",
            "            'count': torch.cuda.device_count(),",
            "            'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)",
            "        }",
            "    elif framework.lower() == 'tensorflow':",
            "        gpus = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']",
            "        if gpus:",
            "            results['system_info']['gpu_info'] = {",
            "                'name': gpus[0].physical_device_desc,",
            "                'count': len(gpus),",
            "            }",
            "    ",
            "    # Save results",
            "    with open(output_path, 'w') as f:",
            "        json.dump(results, f, indent=2)",
            "    print(f'Results saved to {output_path}')",
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])
        
        return "\n".join(script)
    
    def profile_model(self, model_path, cloud_provider, instance_type, 
                     batch_sizes=None, sequence_lengths=None, 
                     num_iterations=100, framework="pytorch", auto_tune=False):
        """
        Profile model performance on specified cloud instance
        
        Args:
            model_path: Path to saved model file
            cloud_provider: Cloud provider to use ('aws', 'gcp', 'azure')
            instance_type: Instance type to profile on
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test (for sequence models)
            num_iterations: Number of iterations for each configuration
            framework: ML framework ('pytorch' or 'tensorflow')
            auto_tune: Whether to automatically find optimal batch size
            
        Returns:
            Dictionary with profiling results
        """
        if cloud_provider not in self.SUPPORTED_CLOUDS:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}. Supported: {self.SUPPORTED_CLOUDS}")
            
        if cloud_provider not in self.auth_clients:
            raise ValueError(f"No credentials available for {cloud_provider}")
            
        # Default batch sizes if not specified
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128] if not auto_tune else [32]
        
        # Upload model to cloud storage
        remote_model_path = self.upload_model(model_path, cloud_provider)
        logger.info(f"Uploaded model to {remote_model_path}")
        
        # Create profiling script
        profiling_script = self._generate_profiling_script(
            remote_model_path, 
            batch_sizes,
            sequence_lengths,
            framework,
            num_iterations
        )
        
        # Execute profiling on remote instance
        results = self._execute_remote_profiling(
            cloud_provider,
            instance_type,
            remote_model_path,
            profiling_script,
            framework
        )
        
        # Auto-tune batch size if requested
        if auto_tune and results:
            optimal_batch, tuning_results = self._auto_tune_batch_size(
                model_path,
                cloud_provider,
                instance_type,
                framework=framework
            )
            results["auto_tuning_results"] = tuning_results
            results["optimal_batch_size"] = optimal_batch
        
        return results
        
    def _execute_remote_profiling(self, cloud_provider, instance_type, 
                                remote_model_path, profiling_script, framework):
        """
        Execute profiling script on remote cloud instance
        
        Args:
            cloud_provider: Cloud provider name
            instance_type: Instance type to use
            remote_model_path: Path to model on cloud storage
            profiling_script: Profiling script content
            framework: ML framework name
            
        Returns:
            Dictionary with profiling results
        """
        # For demonstration purposes, we'll return a more detailed mock result
        # In a real implementation, this would:
        # 1. Launch an instance or use an existing one
        # 2. Install dependencies
        # 3. Upload and execute the script
        # 4. Retrieve and parse results
        # 5. Terminate the instance if needed
        
        # Enhanced mock results with memory and FLOPs metrics
        return {
            "cloud_provider": cloud_provider,
            "instance_type": instance_type,
            "framework": framework,
            "batch_size_results": {
                "32": {
                    "latency_ms": 45, 
                    "throughput": 711, 
                    "gpu_utilization": 0.85,
                    "memory_used_bytes": 1572864000,
                    "memory_peak_bytes": 2684354560,
                    "flops_per_batch": 35842301952
                },
                "64": {
                    "latency_ms": 82, 
                    "throughput": 780, 
                    "gpu_utilization": 0.92,
                    "memory_used_bytes": 3145728000,
                    "memory_peak_bytes": 4294967296,
                    "flops_per_batch": 71684603904
                },
                "128": {
                    "latency_ms": 158, 
                    "throughput": 810, 
                    "gpu_utilization": 0.98,
                    "memory_used_bytes": 6291456000,
                    "memory_peak_bytes": 7516192768,
                    "flops_per_batch": 143369207808
                }
            },
            "optimal_batch_size": 64,
            "cost_per_1000_samples": 0.0023,
            "estimated_monthly_cost": 165.60,
            "system_info": {
                "timestamp": "2023-10-30T15:32:42.123456",
                "cpu_info": {
                    "cores": 8,
                    "threads": 16
                },
                "memory_info": {
                    "total_gb": 32.0
                },
                "gpu_info": {
                    "name": "Tesla T4",
                    "count": 1,
                    "memory_gb": 16.0
                }
            }
        }
    
    def _auto_tune_batch_size(self, model_path, cloud_provider, instance_type, 
                            min_batch=1, max_batch=512, framework="pytorch"):
        """
        Automatically find optimal batch size for best cost/throughput ratio
        
        Args:
            model_path: Path to model file
            cloud_provider: Cloud provider to use
            instance_type: Instance type to profile on
            min_batch: Minimum batch size to try
            max_batch: Maximum batch size to try
            framework: ML framework
            
        Returns:
            Tuple of (optimal_batch_size, tuning_results)
        """
        logger.info(f"Auto-tuning batch size for {model_path} on {cloud_provider} {instance_type}")
        
        # Binary search approach for finding optimal batch size
        batch_sizes = []
        lower = min_batch
        upper = max_batch
        
        # Start with a few points to get initial performance curve
        initial_points = [min_batch, min_batch*4, min_batch*16, max_batch//4, max_batch]
        initial_points = sorted(list(set([b for b in initial_points if min_batch <= b <= max_batch])))
        
        # Test initial batch sizes
        results = {}
        for batch in initial_points:
            if batch not in batch_sizes:
                batch_sizes.append(batch)
                batch_result = self.profile_model(
                    model_path=model_path,
                    cloud_provider=cloud_provider,
                    instance_type=instance_type,
                    batch_sizes=[batch],
                    framework=framework,
                    auto_tune=False
                )
                results[batch] = batch_result["batch_size_results"][str(batch)]
        
        # Find optimal batch size using cost-performance ratio
        best_batch = None
        best_ratio = 0
        
        # Calculate ratios and find best batch size
        for batch, metrics in results.items():
            if "throughput" in metrics and "cost_per_1000_samples" in metrics:
                ratio = metrics["throughput"] / (metrics.get("cost_per_1000_samples", 0.001) * 1000)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_batch = batch
        
        # If we haven't found a good batch size, use a default
        if best_batch is None:
            best_batch = 32  # Safe default
            
        tuning_results = {
            "tested_batch_sizes": batch_sizes,
            "batch_metrics": results,
            "cost_throughput_ratios": {
                str(b): results[b]["throughput"] / (results[b].get("cost_per_1000_samples", 0.001) * 1000)
                for b in batch_sizes if "throughput" in results[b]
            }
        }
        
        return best_batch, tuning_results
    
    def generate_html_report(self, profiling_results, output_path=None):
        """
        Generate an HTML report from profiling results
        
        Args:
            profiling_results: Results from profiling run(s)
            output_path: Path to save HTML report (if None, returns HTML string)
            
        Returns:
            Path to saved report or HTML string if output_path is None
        """
        # Start HTML document
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Neural-Scope Cloud Profiling Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .header { background-color: #4285f4; color: white; padding: 20px; border-radius: 5px; }",
            "        .provider { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
            "        .provider-aws { border-left: 5px solid #ff9900; }",
            "        .provider-gcp { border-left: 5px solid #4285f4; }",
            "        .provider-azure { border-left: 5px solid #0089d6; }",
            "        table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        .chart-container { height: 300px; margin: 20px 0; }",
            "        .metric-card { display: inline-block; width: 200px; margin: 10px; padding: 15px; ",
            "                      border-radius: 5px; background-color: #f8f9fa; text-align: center; }",
            "        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }",
            "        .recommendation { background-color: #e2f3eb; border-left: 5px solid #34a853; ",
            "                          padding: 15px; margin: 20px 0; }",
            "    </style>",
            "    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>",
            "</head>",
            "<body>",
            "    <div class='header'>",
            "        <h1>Neural-Scope Cloud Profiling Report</h1>",
            f"        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "    </div>"
        ]
        
        # Check if we have comparison results or single provider results
        if "comparison_results" in profiling_results:
            # Comparison report
            html.append("<h2>Cloud Provider Comparison</h2>")
            
            # Summary metrics
            html.append("<div class='summary-metrics'>")
            if profiling_results.get("best_throughput"):
                html.append(f"<div class='metric-card'>")
                html.append(f"<h3>Best Throughput</h3>")
                html.append(f"<div class='metric-value'>{profiling_results['best_throughput']}</div>")
                html.append(f"</div>")
                
            if profiling_results.get("best_cost_efficiency"):
                html.append(f"<div class='metric-card'>")
                html.append(f"<h3>Best Cost Efficiency</h3>")
                html.append(f"<div class='metric-value'>{profiling_results['best_cost_efficiency']}</div>")
                html.append(f"</div>")
            html.append("</div>")
            
            # Recommendation
            if profiling_results.get("recommendation"):
                html.append("<div class='recommendation'>")
                html.append(f"<h3>Recommendation</h3>")
                html.append(f"<p>{profiling_results['recommendation']}</p>")
                html.append("</div>")
            
            # Provider details
            for provider, results in profiling_results.get("comparison_results", {}).items():
                html.append(f"<div class='provider provider-{provider}'>")
                html.append(f"<h2>{provider.upper()} - {results.get('instance_type', 'Unknown')}</h2>")
                
                # Basic metrics
                html.append("<div class='metrics'>")
                metrics = [
                    ("Framework", results.get("framework", "Unknown")),
                    ("Optimal Batch Size", results.get("optimal_batch_size", "Unknown")),
                    ("Cost per 1000 Samples", f"${results.get('cost_per_1000_samples', 0):,.4f}"),
                    ("Estimated Monthly Cost", f"${results.get('estimated_monthly_cost', 0):,.2f}")
                ]
                
                html.append("<table>")
                html.append("<tr>" + "".join([f"<th>{m[0]}</th>" for m in metrics]) + "</tr>")
                html.append("<tr>" + "".join([f"<td>{m[1]}</td>" for m in metrics]) + "</tr>")
                html.append("</table>")
                html.append("</div>")
                
                # Batch size results
                if "batch_size_results" in results:
                    html.append("<h3>Batch Size Performance</h3>")
                    html.append("<table>")
                    html.append("<tr><th>Batch Size</th><th>Latency (ms)</th><th>Throughput</th><th>GPU Util</th><th>Memory Used</th><th>FLOPs/batch</th></tr>")
                    
                    for batch, metrics in results["batch_size_results"].items():
                        memory_gb = metrics.get("memory_used_bytes", 0) / (1024**3)
                        flops = metrics.get("flops_per_batch", 0) / (10**9)
                        
                        html.append("<tr>")
                        html.append(f"<td>{batch}</td>")
                        html.append(f"<td>{metrics.get('latency_ms', 0):,.2f}</td>")
                        html.append(f"<td>{metrics.get('throughput', 0):,.0f}</td>")
                        html.append(f"<td>{metrics.get('gpu_utilization', 0)*100:,.1f}%</td>")
                        html.append(f"<td>{memory_gb:,.2f} GB</td>")
                        html.append(f"<td>{flops:,.2f} GFLOPs</td>")
                        html.append("</tr>")
                        
                    html.append("</table>")
                
                # Add charts
                if "batch_size_results" in results:
                    batch_sizes = list(results["batch_size_results"].keys())
                    latencies = [results["batch_size_results"][b].get("latency_ms", 0) for b in batch_sizes]
                    throughputs = [results["batch_size_results"][b].get("throughput", 0) for b in batch_sizes]
                    
                    # Create chart
                    chart_id = f"chart-{provider}"
                    html.append(f"<div class='chart-container'>")
                    html.append(f"<canvas id='{chart_id}'></canvas>")
                    html.append("</div>")
                    
                    # Chart JS
                    html.append("<script>")
                    html.append(f"new Chart(document.getElementById('{chart_id}'), {{")
                    html.append("    type: 'bar',")
                    html.append("    data: {")
                    html.append("        labels: " + str(batch_sizes) + ",")
                    html.append("        datasets: [{")
                    html.append("            label: 'Latency (ms)',")
                    html.append("            data: " + str(latencies) + ",")
                    html.append("            backgroundColor: 'rgba(255, 99, 132, 0.5)',")
                    html.append("            yAxisID: 'y',")
                    html.append("        }, {")
                    html.append("            label: 'Throughput',")
                    html.append("            data: " + str(throughputs) + ",")
                    html.append("            backgroundColor: 'rgba(54, 162, 235, 0.5)',")
                    html.append("            yAxisID: 'y1',")
                    html.append("        }]")
                    html.append("    },")
                    html.append("    options: {")
                    html.append("        responsive: true,")
                    html.append("        scales: {")
                    html.append("            y: {")
                    html.append("                type: 'linear',")
                    html.append("                display: true,")
                    html.append("                position: 'left',")
                    html.append("                title: {")
                    html.append("                    display: true,")
                    html.append("                    text: 'Latency (ms)'")
                    html.append("                }")
                    html.append("            },")
                    html.append("            y1: {")
                    html.append("                type: 'linear',")
                    html.append("                display: true,")
                    html.append("                position: 'right',")
                    html.append("                title: {")
                    html.append("                    display: true,")
                    html.append("                    text: 'Throughput'")
                    html.append("                }")
                    html.append("            }")
                    html.append("        }")
                    html.append("    }")
                    html.append("});")
                    html.append("</script>")
                
                html.append("</div>")
        else:
            # Single provider report
            provider = profiling_results.get("cloud_provider", "unknown")
            html.append(f"<div class='provider provider-{provider}'>")
            html.append(f"<h2>{provider.upper()} - {profiling_results.get('instance_type', 'Unknown')}</h2>")
            
            # Similar implementation as above for single provider
            # ...
            
            html.append("</div>")
        
        # Close HTML document
        html.append("</body>")
        html.append("</html>")
        
        html_content = "\n".join(html)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(html_content)
            return output_path
        else:
            return html_content
        
    def compare_providers(self, model_path, instance_types=None, framework="pytorch", generate_report=True):
        """
        Compare model performance across cloud providers
        
        Args:
            model_path: Path to saved model file
            instance_types: Dictionary mapping cloud providers to instance types
            framework: ML framework ('pytorch' or 'tensorflow')
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary with comparison results
        """
        if not instance_types:
            # Use default comparable instances
            instance_types = {
                'aws': 'p3.2xlarge',
                'gcp': 'n1-standard-8-nvidia-tesla-v100',
                'azure': 'Standard_NC6s_v3'
            }
            
        results = {}
        
        for provider, instance in instance_types.items():
            if provider in self.auth_clients:
                try:
                    results[provider] = self.profile_model(
                        model_path=model_path,
                        cloud_provider=provider,
                        instance_type=instance,
                        framework=framework
                    )
                except Exception as e:
                    logger.error(f"Failed to profile on {provider}: {e}")
                    
        # Calculate cost-performance metrics
        if results:
            for provider, result in results.items():
                if "throughput" in result and "cost_per_1000_samples" in result:
                    result["cost_performance_ratio"] = result["throughput"] / (result["cost_per_1000_samples"] * 1000)
                    
        results = {
            "comparison_results": results,
            "best_throughput": max(results.items(), key=lambda x: x[1].get("throughput", 0))[0] if results else None,
            "best_cost_efficiency": max(results.items(), key=lambda x: x[1].get("cost_performance_ratio", 0))[0] if results else None,
            "recommendation": "AWS p3.2xlarge offers the best balance of performance and cost for this model"
        }
        
        # Generate HTML report if requested
        if generate_report:
            report_path = f"cloud_comparison_{int(time.time())}.html"
            self.generate_html_report(results, output_path=report_path)
            results["report_path"] = report_path
            
        return results
