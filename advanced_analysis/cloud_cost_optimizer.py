import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class CloudCostAnalysisResult:
    """Comprehensive cloud cost analysis results"""
    current_instance: str
    current_cost_per_hour: float
    recommended_instance: str
    recommended_cost_per_hour: float
    potential_savings_percentage: float
    utilization_metrics: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    alternative_options: List[Dict[str, Any]]

@dataclass
class DataQualityMetrics:
    completeness: Dict[str, float]
    uniqueness: Dict[str, float]
    consistency: Dict[str, Dict]
    validity: Dict[str, float]
    outlier_scores: Dict[str, List[int]]
    distribution_metrics: Dict[str, Dict]

class CloudCostOptimizer:
    """Analyzes and optimizes cloud costs for ML training and inference"""
    
    # Cloud pricing data (simplified - would need regular updates in production)
    AWS_PRICING = {
        'p3.2xlarge': {'cost': 3.06, 'gpus': 1, 'gpu_type': 'V100', 'ram': 61},
        'p3.8xlarge': {'cost': 12.24, 'gpus': 4, 'gpu_type': 'V100', 'ram': 244},
        'p3.16xlarge': {'cost': 24.48, 'gpus': 8, 'gpu_type': 'V100', 'ram': 488},
        'p4d.24xlarge': {'cost': 32.77, 'gpus': 8, 'gpu_type': 'A100', 'ram': 320},
        'g4dn.xlarge': {'cost': 0.526, 'gpus': 1, 'gpu_type': 'T4', 'ram': 16},
        'g4dn.12xlarge': {'cost': 3.912, 'gpus': 4, 'gpu_type': 'T4', 'ram': 192},
        'g5.xlarge': {'cost': 1.006, 'gpus': 1, 'gpu_type': 'A10G', 'ram': 24},
        'g5.12xlarge': {'cost': 8.208, 'gpus': 4, 'gpu_type': 'A10G', 'ram': 192},
        'inf1.xlarge': {'cost': 0.369, 'inferentia_chips': 1, 'ram': 8},
        'inf1.6xlarge': {'cost': 1.842, 'inferentia_chips': 4, 'ram': 48}
    }
    
    GCP_PRICING = {
        'n1-standard-4-nvidia-tesla-t4': {'cost': 0.95, 'gpus': 1, 'gpu_type': 'T4', 'ram': 15},
        'n1-standard-8-nvidia-tesla-t4': {'cost': 1.21, 'gpus': 1, 'gpu_type': 'T4', 'ram': 30},
        'n1-standard-8-nvidia-tesla-v100': {'cost': 2.48, 'gpus': 1, 'gpu_type': 'V100', 'ram': 30},
        'n1-standard-16-nvidia-tesla-v100': {'cost': 3.38, 'gpus': 2, 'gpu_type': 'V100', 'ram': 60},
        'a2-highgpu-1g': {'cost': 3.67, 'gpus': 1, 'gpu_type': 'A100', 'ram': 85},
        'a2-highgpu-4g': {'cost': 13.31, 'gpus': 4, 'gpu_type': 'A100', 'ram': 340},
        'a2-highgpu-8g': {'cost': 26.62, 'gpus': 8, 'gpu_type': 'A100', 'ram': 680}
    }
    
    AZURE_PRICING = {
        'Standard_NC6s_v3': {'cost': 3.06, 'gpus': 1, 'gpu_type': 'V100', 'ram': 112},
        'Standard_NC12s_v3': {'cost': 6.12, 'gpus': 2, 'gpu_type': 'V100', 'ram': 224},
        'Standard_NC24s_v3': {'cost': 12.24, 'gpus': 4, 'gpu_type': 'V100', 'ram': 448},
        'Standard_ND40rs_v2': {'cost': 28.7, 'gpus': 8, 'gpu_type': 'V100', 'ram': 672},
        'Standard_NV6': {'cost': 0.73, 'gpus': 1, 'gpu_type': 'M60', 'ram': 56}
    }
    
    # Spot discount approximations (actual rates vary)
    SPOT_DISCOUNTS = {
        'aws': 0.7,  # 70% discount
        'gcp': 0.6,  # 60% discount
        'azure': 0.6  # 60% discount
    }
    
    def __init__(self):
        self.results = {}
        
    def analyze_training_costs(self, 
                              provider: str,
                              instance_type: str,
                              utilization_metrics: Dict[str, float],
                              training_hours: float,
                              job_type: str = 'training') -> CloudCostAnalysisResult:
        """
        Analyze ML training costs and suggest optimizations
        
        Args:
            provider: Cloud provider ('aws', 'gcp', or 'azure')
            instance_type: Current instance type
            utilization_metrics: Dict with 'gpu_util', 'memory_util', etc.
            training_hours: Expected total training hours
            job_type: 'training' or 'inference'
            
        Returns:
            CloudCostAnalysisResult with cost optimization recommendations
        """
        # Get current pricing
        pricing_data = self._get_pricing_data(provider)
        if instance_type not in pricing_data:
            raise ValueError(f"Unknown instance type: {instance_type} for provider: {provider}")
            
        current_cost_hour = pricing_data[instance_type]['cost']
        total_cost = current_cost_hour * training_hours
        
        # Analyze utilization
        gpu_util = utilization_metrics.get('gpu_util', 0)
        memory_util = utilization_metrics.get('memory_util', 0)
        cpu_util = utilization_metrics.get('cpu_util', 0)
        
        # Generate recommendations
        recommendations = []
        alternative_options = []
        
        # Check for under-utilization
        is_underutilized = False
        if gpu_util < 50:  # Less than 50% GPU utilization
            is_underutilized = True
            recommendations.append({
                'type': 'gpu_utilization',
                'severity': 'high',
                'description': f"Low GPU utilization detected ({gpu_util}%)",
                'recommendation': "Consider using a smaller/cheaper instance type or optimizing batch size",
                'potential_savings': f"{(1 - gpu_util/100) * 100:.1f}%"
            })
            
        # Suggest spot instances for training
        if job_type == 'training' and training_hours > 1:
            spot_discount = self.SPOT_DISCOUNTS.get(provider.lower(), 0.6)
            spot_savings = total_cost * spot_discount
            recommendations.append({
                'type': 'spot_instance',
                'severity': 'medium',
                'description': f"Training job suitable for spot/preemptible instances",
                'recommendation': f"Switch to spot instances for approximately {spot_discount*100:.0f}% lower costs",
                'potential_savings': f"${spot_savings:.2f} ({spot_discount*100:.0f}%)",
                'code_example': self._get_spot_instance_example(provider, instance_type)
            })
            
        # Find optimal instance for observed utilization
        recommended_instance, saving_pct = self._find_optimal_instance(
            provider, instance_type, utilization_metrics, job_type
        )
        
        # Get alternative options
        alternatives = self._get_alternative_instances(provider, instance_type, utilization_metrics, job_type)
        
        # Generate result object
        result = CloudCostAnalysisResult(
            current_instance=instance_type,
            current_cost_per_hour=current_cost_hour,
            recommended_instance=recommended_instance,
            recommended_cost_per_hour=pricing_data.get(recommended_instance, {}).get('cost', current_cost_hour),
            potential_savings_percentage=saving_pct,
            utilization_metrics=utilization_metrics,
            recommendations=recommendations,
            alternative_options=alternatives
        )
        
        self.results[instance_type] = result
        return result
        
    def _get_pricing_data(self, provider):
        """Get pricing data for specified provider"""
        provider = provider.lower()
        if provider == 'aws':
            return self.AWS_PRICING
        elif provider == 'gcp':
            return self.GCP_PRICING
        elif provider == 'azure':
            return self.AZURE_PRICING
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    def _find_optimal_instance(self, provider, current_instance, metrics, job_type):
        """Find the optimal instance type based on utilization metrics"""
        pricing_data = self._get_pricing_data(provider)
        current_cost = pricing_data.get(current_instance, {}).get('cost', 0)
        
        # Get current instance details
        current_specs = pricing_data.get(current_instance, {})
        
        # Calculate required resources based on utilization
        required_gpu_power = current_specs.get('gpus', 1) * (metrics.get('gpu_util', 100) / 100)
        required_memory = current_specs.get('ram', 16) * (metrics.get('memory_util', 100) / 100)
        
        # Find instances that meet requirements but are cheaper
        suitable_instances = []
        for instance, specs in pricing_data.items():
            # Skip if not suitable for the job type
            if job_type == 'inference' and 'inferentia' not in str(specs) and instance != current_instance:
                continue
                
            gpu_power = specs.get('gpus', 0)
            if gpu_power >= required_gpu_power * 0.9 and specs.get('ram', 0) >= required_memory * 0.9:
                cost_savings = (current_cost - specs.get('cost', 0)) / current_cost if current_cost > 0 else 0
                if cost_savings > 0.1:  # At least 10% savings
                    suitable_instances.append((instance, specs.get('cost', 0), cost_savings))
        
        # Sort by cost savings
        suitable_instances.sort(key=lambda x: x[2], reverse=True)
        
        if suitable_instances:
            recommended = suitable_instances[0][0]
            savings_pct = suitable_instances[0][2] * 100
            return recommended, savings_pct
        
        return current_instance, 0.0
        
    def _get_alternative_instances(self, provider, current_instance, metrics, job_type):
        """Get alternative instance options"""
        pricing_data = self._get_pricing_data(provider)
        current_cost = pricing_data.get(current_instance, {}).get('cost', 0)
        
        alternatives = []
        
        # Check for alternative instance types (cheaper/more powerful)
        for instance, specs in pricing_data.items():
            if instance == current_instance:
                continue
                
            # Calculate relative performance and cost
            relative_cost = specs.get('cost', 0) / current_cost if current_cost > 0 else float('inf')
            
            # For similar GPU types
            if specs.get('gpu_type') == pricing_data.get(current_instance, {}).get('gpu_type'):
                gpu_ratio = specs.get('gpus', 1) / max(1, pricing_data.get(current_instance, {}).get('gpus', 1))
                
                # If good value (more GPUs for relatively less money)
                if gpu_ratio > relative_cost:
                    alternatives.append({
                        'instance': instance,
                        'cost_per_hour': specs.get('cost', 0),
                        'gpus': specs.get('gpus', 1),
                        'gpu_type': specs.get('gpu_type', 'unknown'),
                        'ram': specs.get('ram', 0),
                        'value_proposition': f"{gpu_ratio/relative_cost:.2f}x better price/performance ratio"
                    })
                    
            # Add spot instance alternatives
            spot_discount = self.SPOT_DISCOUNTS.get(provider.lower(), 0.6)
            spot_cost = specs.get('cost', 0) * (1 - spot_discount)
            
            if job_type == 'training' and spot_cost < current_cost * 0.9:  # At least 10% cheaper
                alternatives.append({
                    'instance': f"{instance} (Spot)",
                    'cost_per_hour': spot_cost,
                    'gpus': specs.get('gpus', 1),
                    'gpu_type': specs.get('gpu_type', 'unknown'),
                    'ram': specs.get('ram', 0),
                    'value_proposition': f"{(current_cost/spot_cost):.2f}x cheaper than current instance"
                })
                
        # Sort by cost
        alternatives.sort(key=lambda x: x['cost_per_hour'])
        
        # Return top 5 alternatives
        return alternatives[:5]
    
    def _get_spot_instance_example(self, provider, instance_type):
        """Generate code example for using spot/preemptible instances"""
        provider = provider.lower()
        
        if provider == 'aws':
            return """
# AWS Spot Instance with boto3
import boto3

ec2 = boto3.client('ec2')

# Request spot instances
response = ec2.request_spot_instances(
    InstanceCount=1,
    LaunchSpecification={
        'ImageId': 'ami-0123456789abcdef', # Deep Learning AMI
        'InstanceType': '""" + instance_type + """',
        'SecurityGroupIds': ['sg-0123456789abcdef'],
        'SubnetId': 'subnet-0123456789abcdef',
        'IamInstanceProfile': {
            'Name': 'your-role-with-s3-access'
        },
        'BlockDeviceMappings': [
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': 100,
                    'VolumeType': 'gp3'
                }
            }
        ]
    }
)

# For SageMaker:
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role='SageMakerRole',
    instance_count=1,
    instance_type='""" + instance_type + """',
    framework_version='1.12.0',
    use_spot_instances=True,
    max_run=86400,  # Maximum run time in seconds
    max_wait=86700,  # Maximum time to wait for spot
)
"""
        elif provider == 'gcp':
            return """
# GCP Preemptible Instance with gcloud CLI
# Save this to a file, e.g., create-instance.sh

gcloud compute instances create ml-training-spot \\
    --machine-type=""" + instance_type + """ \\
    --image-family=pytorch-latest-gpu \\
    --image-project=deeplearning-platform-release \\
    --maintenance-policy=TERMINATE \\
    --accelerator="type=nvidia-tesla-t4,count=1" \\
    --metadata="install-nvidia-driver=True" \\
    --preemptible

# Or with Python client:
from google.cloud import compute_v1

def create_preemptible_instance(project_id, zone, instance_name):
    instance_client = compute_v1.InstancesClient()
    
    # Configure the machine
    machine_type = f"zones/{zone}/machineTypes/""" + instance_type + """
    instance = {
        "name": instance_name,
        "machine_type": machine_type,
        "scheduling": {
            "preemptible": True,
            "automatic_restart": False,
        },
        "disks": [
            {
                "boot": True,
                "auto_delete": True,
                "initialize_params": {
                    "source_image": "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu",
                    "disk_size_gb": 100,
                }
            }
        ],
        "network_interfaces": [{"network": "global/networks/default"}],
    }
    
    return instance_client.insert(project=project_id, zone=zone, instance_resource=instance)
"""
        elif provider == 'azure':
            return """
# Azure Low Priority VM with Azure CLI
# Save this to a file, e.g., create-instance.sh

az vm create \\
  --resource-group myResourceGroup \\
  --name ml-training-spot \\
  --image UbuntuLTS \\
  --size """ + instance_type + """ \\
  --priority Spot \\
  --max-price -1 \\
  --eviction-policy Deallocate

# Or with Python SDK:
from azure.mgmt.compute import ComputeManagementClient
from azure.identity import DefaultAzureCredential

def create_spot_vm(resource_group, vm_name, location="eastus"):
    # Authenticate
    credential = DefaultAzureCredential()
    compute_client = ComputeManagementClient(credential, subscription_id)
    
    # Define VM
    vm_parameters = {
        'location': location,
        'priority': 'Spot',
        'eviction_policy': 'Deallocate',
        'billing_profile': {'max_price': -1},  # -1 means cap at on-demand price
        'hardware_profile': {'vm_size': '""" + instance_type + """'},
        'storage_profile': {
            'image_reference': {
                'publisher': 'microsoft-dsvm',
                'offer': 'ubuntu-2004',
                'sku': '2004-gen2',
                'version': 'latest'
            }
        },
        'network_profile': {
            'network_interfaces': [{
                'id': '/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/networkInterfaces/{vm_name}-nic',
            }]
        }
    }
    
    return compute_client.virtual_machines.begin_create_or_update(
        resource_group, vm_name, vm_parameters
    )
"""
        else:
            return "# No template available for this provider"
    
    def generate_multi_region_comparison(self, provider, instance_type, utilization_metrics, training_hours=1000):
        """Compare costs across different regions for the same workload"""
        # Sample regions with pricing variations
        region_pricing_factor = {
            'aws': {
                'us-east-1': 1.0,  # Base reference
                'us-west-1': 1.05,
                'eu-west-1': 1.08, 
                'ap-southeast-1': 1.15,
                'ap-northeast-1': 1.20,
                'sa-east-1': 1.25
            },
            'gcp': {
                'us-central1': 1.0,  # Base reference
                'us-west1': 1.03,
                'us-east1': 1.0,
                'europe-west1': 1.10,
                'asia-east1': 1.18,
                'asia-northeast1': 1.15
            },
            'azure': {
                'eastus': 1.0,  # Base reference
                'westus': 1.0,
                'westeurope': 1.06,
                'southeastasia': 1.12,
                'japaneast': 1.15,
                'brazilsouth': 1.22
            }
        }
        
        # Get base pricing
        pricing_data = self._get_pricing_data(provider)
        base_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        
        # Calculate per-region costs
        region_costs = {}
        for region, factor in region_pricing_factor.get(provider, {}).items():
            region_cost = base_cost * factor
            total_cost = region_cost * training_hours
            
            # Calculate CO2 emissions based on regional carbon intensity
            # Rough estimates based on cloud providers' published data
            carbon_intensity = self._get_region_carbon_intensity(provider, region)
            power_consumption = self._estimate_instance_power(provider, instance_type)
            carbon_emissions = power_consumption * training_hours * carbon_intensity
            
            region_costs[region] = {
                'hourly_cost': region_cost,
                'total_cost': total_cost,
                'carbon_emissions_kg': carbon_emissions,
                'sustainability_score': self._calculate_sustainability_score(carbon_emissions, training_hours)
            }
            
        # Sort by total cost
        sorted_regions = sorted(region_costs.items(), key=lambda x: x[1]['total_cost'])
        
        # Format results
        result = {
            'cheapest_region': sorted_regions[0][0],
            'cost_savings_vs_most_expensive': 
                (region_costs[sorted_regions[-1][0]]['total_cost'] - 
                 region_costs[sorted_regions[0][0]]['total_cost']),
            'savings_percentage': 
                (region_costs[sorted_regions[-1][0]]['total_cost'] - 
                 region_costs[sorted_regions[0][0]]['total_cost']) / 
                    region_costs[sorted_regions[-1][0]]['total_cost'] * 100,
            'greenest_region': min(region_costs.items(), key=lambda x: x[1]['carbon_emissions_kg'])[0],
            'detailed_comparison': region_costs
        }
        
        return result
    
    def _get_region_carbon_intensity(self, provider, region):
        """Get CO2 intensity (kg CO2 per kWh) for a given region"""
        # Based on published cloud provider data and regional grid intensities
        # Values are approximate and should be updated regularly
        carbon_intensity = {
            'aws': {
                'us-west-2': 0.15,      # Oregon - hydro power
                'us-east-1': 0.37,      # Virginia
                'us-west-1': 0.24,      # California
                'eu-west-1': 0.28,      # Ireland
                'eu-central-1': 0.33,   # Frankfurt
                'ap-southeast-1': 0.41, # Singapore
                'ap-northeast-1': 0.55, # Tokyo
                'sa-east-1': 0.09,      # São Paulo - mostly hydroelectric
            },
            'gcp': {
                'us-central1': 0.48,    # Iowa
                'us-west1': 0.12,       # Oregon - very clean
                'us-east1': 0.52,       # South Carolina
                'europe-west1': 0.08,   # Belgium - nuclear heavy
                'europe-west3': 0.35,   # Frankfurt
                'asia-east1': 0.54,     # Taiwan
                'asia-northeast1': 0.48, # Tokyo
            },
            'azure': {
                'eastus': 0.39,         # Virginia
                'westus': 0.25,         # California
                'northeurope': 0.38,    # Ireland
                'westeurope': 0.26,     # Netherlands
                'southeastasia': 0.49,  # Singapore
                'japaneast': 0.57,      # Japan
            }
        }
        
        # Default to average if region not found
        return carbon_intensity.get(provider, {}).get(region, 0.45)  # Global average ~0.45
        
    def _estimate_instance_power(self, provider, instance_type):
        """Estimate power consumption in kW for an instance type"""
        # Rough power consumption estimates based on GPU count and type
        instance_specs = self._get_pricing_data(provider).get(instance_type, {})
        
        # Base power for the instance excluding GPUs
        base_power = 0.15  # ~150W for base system
        
        # GPU power estimates
        gpu_power = {
            'V100': 0.3,    # 300W per GPU
            'A100': 0.4,    # 400W per GPU
            'T4': 0.07,     # 70W per GPU
            'K80': 0.15,    # 150W per GPU
            'M60': 0.18,    # 180W per GPU
            'A10G': 0.15,   # 150W per GPU
        }
        
        # Calculate based on GPU count and type
        gpu_type = instance_specs.get('gpu_type', '')
        gpu_count = instance_specs.get('gpus', 0)
        
        total_gpu_power = gpu_count * gpu_power.get(gpu_type, 0.2)  # Default 200W if unknown
        
        # Add memory power consumption (rough estimate)
        ram_gb = instance_specs.get('ram', 16)
        ram_power = ram_gb * 0.0015  # ~1.5W per GB of RAM
        
        return base_power + total_gpu_power + ram_power
        
    def _calculate_sustainability_score(self, carbon_emissions, training_hours):
        """Calculate a normalized sustainability score (0-10)"""
        # Lower emissions = higher score
        # Base calculation on emissions per hour
        emissions_per_hour = carbon_emissions / max(1, training_hours)
        
        if emissions_per_hour < 0.05:  # Very efficient
            return 10
        elif emissions_per_hour < 0.1:
            return 9
        elif emissions_per_hour < 0.2:
            return 8
        elif emissions_per_hour < 0.3:
            return 7
        elif emissions_per_hour < 0.5:
            return 6
        elif emissions_per_hour < 0.7:
            return 5
        elif emissions_per_hour < 1.0:
            return 4
        elif emissions_per_hour < 1.5:
            return 3
        elif emissions_per_hour < 2.0:
            return 2
        else:
            return 1
            
    def analyze_reserved_vs_ondemand(self, provider, instance_type, 
                                   estimated_usage_months=12, 
                                   utilization_percentage=70):
        """Compare on-demand vs. reserved instance costs"""
        pricing_data = self._get_pricing_data(provider)
        hourly_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        
        # Estimate hours used per month based on utilization
        hours_per_month = 730 * (utilization_percentage / 100)
        
        # Calculate on-demand cost for the period
        on_demand_cost = hourly_cost * hours_per_month * estimated_usage_months
        
        # Reserved instance discount factors (approximate)
        # These would ideally be fetched from cloud provider APIs
        reserved_discount = {
            'aws': {
                'no_upfront_1yr': 0.25,    # 25% savings
                'partial_upfront_1yr': 0.3, # 30% savings 
                'all_upfront_1yr': 0.35,    # 35% savings
                'no_upfront_3yr': 0.4,      # 40% savings
                'partial_upfront_3yr': 0.55, # 55% savings
                'all_upfront_3yr': 0.6      # 60% savings
            },
            'gcp': {
                'commit_1yr': 0.27,         # 27% savings
                'commit_3yr': 0.5          # 50% savings
            },
            'azure': {
                'reserved_1yr': 0.3,        # 30% savings
                'reserved_3yr': 0.48        # 48% savings
            }
        }
        
        # Calculate reserved costs
        reserved_costs = {}
        provider_discounts = reserved_discount.get(provider, {})
        
        for plan, discount in provider_discounts.items():
            plan_cost = on_demand_cost * (1 - discount)
            break_even_months = 0
            
            # Calculate break-even point (for plans with upfront costs)
            if "upfront" in plan:
                # Estimate upfront cost based on plan name
                upfront_percent = 0
                if "partial" in plan:
                    upfront_percent = 0.5
                elif "all" in plan:
                    upfront_percent = 1.0
                    
                upfront_cost = plan_cost * upfront_percent
                monthly_cost = (plan_cost - upfront_cost) / estimated_usage_months
                
                # Break-even calculation
                monthly_on_demand = on_demand_cost / estimated_usage_months
                monthly_savings = monthly_on_demand - monthly_cost
                
                if monthly_savings > 0:
                    break_even_months = upfront_cost / monthly_savings
            
            reserved_costs[plan] = {
                'total_cost': plan_cost,
                'savings_vs_ondemand': on_demand_cost - plan_cost,
                'savings_percentage': (on_demand_cost - plan_cost) / on_demand_cost * 100,
                'break_even_months': break_even_months
            }
        
        # Find best option
        if reserved_costs:
            best_plan = max(reserved_costs.items(), key=lambda x: x[1]['savings_vs_ondemand'])
            
            if best_plan[1]['savings_vs_ondemand'] > 0:
                recommendation = {
                    'type': 'reserved_instance',
                    'severity': 'medium',
                    'description': f"Reserved instance savings available",
                    'recommendation': f"Switch to {best_plan[0]} reserved instance for {best_plan[1]['savings_percentage']:.1f}% savings",
                    'potential_savings': f"${best_plan[1]['savings_vs_ondemand']:.2f}",
                    'break_even': f"{best_plan[1]['break_even_months']:.1f} months" if best_plan[1]['break_even_months'] > 0 else "Immediate"
                }
            else:
                recommendation = {
                    'type': 'on_demand',
                    'severity': 'low',
                    'description': "On-demand instances are most cost-effective",
                    'recommendation': "Maintain current on-demand pricing based on your usage pattern",
                }
        else:
            recommendation = {
                'type': 'unknown',
                'severity': 'low',
                'description': "Unable to calculate reserved instance pricing",
                'recommendation': "Contact cloud provider for custom quote"
            }
            
        return {
            'on_demand_cost': on_demand_cost,
            'reserved_options': reserved_costs,
            'recommendation': recommendation,
            'usage_assumptions': {
                'months': estimated_usage_months,
                'utilization_percentage': utilization_percentage,
                'hours_per_month': hours_per_month,
                'total_hours': hours_per_month * estimated_usage_months
            }
        }

    def analyze_real_time_cost(self, metrics_history, provider, instance_type):
        """
        Analyze real-time cost efficiency based on historical utilization metrics
        
        Args:
            metrics_history: List of dictionaries with timestamps and utilization metrics
            provider: Cloud provider name
            instance_type: Instance type name
        """
        # Extract hourly cost
        pricing_data = self._get_pricing_data(provider)
        hourly_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        
        # Process historical metrics
        timestamps = []
        gpu_utils = []
        costs = []
        wasted_costs = []
        
        for entry in metrics_history:
            timestamp = entry.get('timestamp')
            gpu_util = entry.get('gpu_util', 0)
            
            if timestamp:
                timestamps.append(timestamp)
                gpu_utils.append(gpu_util)
                
                # Cost for this time period
                period_cost = hourly_cost / 60  # Assuming metrics are per-minute
                costs.append(period_cost)
                
                # Estimate wasted cost based on idle GPU
                wasted_costs.append(period_cost * (1 - gpu_util/100))
                
        # Calculate efficiency metrics
        if not costs:
            return {"error": "No valid metrics data provided"}
            
        total_cost = sum(costs)
        total_wasted = sum(wasted_costs)
        efficiency = (total_cost - total_wasted) / total_cost * 100 if total_cost > 0 else 0
        
        # Identify periods of low utilization
        low_util_periods = []
        current_period = None
        
        for i, (timestamp, util) in enumerate(zip(timestamps, gpu_utils)):
            if util < 15:  # Less than 15% GPU utilization is considered wasteful
                if current_period is None:
                    current_period = {'start': timestamp, 'utils': [util]}
                else:
                    current_period['utils'].append(util)
            else:
                if current_period is not None:
                    current_period['end'] = timestamps[i-1]
                    current_period['avg_util'] = sum(current_period['utils']) / len(current_period['utils'])
                    current_period['wasted_cost'] = hourly_cost * len(current_period['utils']) / 60 * (1 - current_period['avg_util']/100)
                    low_util_periods.append(current_period)
                    current_period = None
        
        # Add last period if it exists
        if current_period is not None:
            current_period['end'] = timestamps[-1]
            current_period['avg_util'] = sum(current_period['utils']) / len(current_period['utils'])
            current_period['wasted_cost'] = hourly_cost * len(current_period['utils']) / 60 * (1 - current_period['avg_util']/100)
            low_util_periods.append(current_period)
        
        # Sort by wasted cost to find most wasteful periods
        low_util_periods.sort(key=lambda x: x.get('wasted_cost', 0), reverse=True)
        
        # Generate recommendations
        recommendations = []
        
        if total_wasted > 0.1 * total_cost:  # More than 10% waste
            recommendations.append({
                'type': 'utilization',
                'severity': 'high',
                'description': f"Low GPU utilization detected (efficiency: {efficiency:.1f}%)",
                'recommendation': "Consider implementing autoscaling or spot instance strategy",
                'potential_savings': f"${total_wasted:.2f} ({100-efficiency:.1f}% of costs)",
                'code_example': self._get_autoscaling_example(provider)
            })
            
        # Recommendation for scaling to match workload pattern
        if low_util_periods and len(timestamps) > 60:  # At least an hour of data
            recommendations.append({
                'type': 'workload_scheduling',
                'severity': 'medium',
                'description': f"Identified {len(low_util_periods)} periods of low utilization",
                'recommendation': "Schedule workloads to minimize idle time or implement auto-shutdown",
                'potential_savings': f"Up to ${sum(p['wasted_cost'] for p in low_util_periods):.2f}",
                'most_wasteful_period': f"From {low_util_periods[0]['start']} to {low_util_periods[0]['end']} (avg util: {low_util_periods[0]['avg_util']:.1f}%)"
            })
            
        return {
            'total_cost': total_cost,
            'wasted_cost': total_wasted,
            'efficiency': efficiency,
            'low_utilization_periods': low_util_periods[:5],  # Top 5 most wasteful periods
            'recommendations': recommendations,
            'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        }
        
    def _get_autoscaling_example(self, provider):
        """Get code example for implementing autoscaling"""
        provider = provider.lower()
        
        if provider == 'aws':
            return """
# AWS Autoscaling with SageMaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='code',
    role='SageMakerRole',
    instance_count=1,  # This will scale automatically
    instance_type='ml.p3.2xlarge',
    framework_version='1.12.0',
    py_version='py38',
    max_run=86400,
    
    # Training metrics to enable autoscaling
    metric_definitions=[
        {'Name': 'gpu-utilization', 'Regex': 'GPU Utilization: ([0-9\\.]+)'},
        {'Name': 'epoch-accuracy', 'Regex': 'Epoch accuracy: ([0-9\\.]+)'}
    ],
    
    # Distribution for multiple instances
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
    
    # Warm pool to keep instances ready
    warm_pool_status='Enabled'
)

# Implement custom autoscaling with EC2 Auto Scaling groups
# 1. Create an AMI with your ML environment
# 2. Create a launch template
# 3. Set up an Auto Scaling group with scaling policies based on GPU utilization
"""
        elif provider == 'gcp':
            return """
# GCP Autoscaling with Managed Instance Groups

# 1. Create an instance template
gcloud compute instance-templates create ml-template \\
  --machine-type=n1-standard-8 \\
  --accelerator=type=nvidia-tesla-t4,count=1 \\
  --image-family=pytorch-latest-gpu \\
  --image-project=deeplearning-platform-release \\
  --boot-disk-size=100GB \\
  --metadata="install-nvidia-driver=True" \\
  --metadata-from-file=startup-script=startup.sh

# 2. Create a managed instance group with autoscaling
gcloud compute instance-groups managed create ml-group \\
  --zone=us-central1-a \\
  --template=ml-template \\
  --size=0

# 3. Configure autoscaling
gcloud compute instance-groups managed set-autoscaling ml-group \\
  --zone=us-central1-a \\
  --min-num-replicas=0 \\
  --max-num-replicas=10 \\
  --target-cpu-utilization=0.6 \\
  --cool-down-period=300

# Custom utilization metric with Cloud Monitoring
# Add to startup.sh:
cat << 'EOF' > /usr/local/bin/gpu-metric.sh
#!/bin/bash
while true; do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  echo "gpu_utilization $UTIL"
  curl -X POST -H "Content-Type: application/json" -d "{\"name\": \"gpu_utilization\", \"value\": $UTIL}" http://metadata.google.internal/computeMetadata/v1/instance/monitoring/custom -H "Metadata-Flavor: Google"
  sleep 60
done
EOF
chmod +x /usr/local/bin/gpu-metric.sh
nohup /usr/local/bin/gpu-metric.sh &
"""
        elif provider == 'azure':
            return """
# Azure VMSS Autoscaling
az vmss create \\
  --resource-group myResourceGroup \\
  --name ml-vmss \\
  --image UbuntuLTS \\
  --vm-sku Standard_NC6s_v3 \\
  --instance-count 0 \\
  --authentication-type SSH \\
  --ssh-key-value ~/.ssh/id_rsa.pub \\
  --upgrade-policy-mode Automatic \\
  --custom-data cloud-init.txt

# Configure autoscaling
az monitor autoscale create \\
  --resource-group myResourceGroup \\
  --resource ml-vmss \\
  --resource-type Microsoft.Compute/virtualMachineScaleSets \\
  --name ml-autoscale \\
  --min-count 0 \\
  --max-count 10 \\
  --count 0

# Add a scale out rule
az monitor autoscale rule create \\
  --resource-group myResourceGroup \\
  --autoscale-name ml-autoscale \\
  --condition "Percentage CPU > 70 avg 5m" \\
  --scale out 1

# Add a scale in rule
az monitor autoscale rule create \\
  --resource-group myResourceGroup \\
  --autoscale-name ml-autoscale \\
  --condition "Percentage CPU < 30 avg 5m" \\
  --scale in 1
"""
        else:
            return "# No autoscaling template available for this provider"

    def visualize_cost_efficiency(self, utilization_metrics, provider, instance_type):
        """
        Create a visualization of cost efficiency based on utilization metrics
        
        Args:
            utilization_metrics: Dict with utilization metrics
            provider: Cloud provider name
            instance_type: Instance type name
            
        Returns:
            Matplotlib figure with cost efficiency visualization
        """
        # Extract pricing and metrics
        pricing_data = self._get_pricing_data(provider)
        hourly_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        gpu_util = utilization_metrics.get('gpu_util', 0)
        memory_util = utilization_metrics