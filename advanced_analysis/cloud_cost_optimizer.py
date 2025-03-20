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
        memory_util = utilization_metrics.get('memory_util', 0)
        cpu_util = utilization_metrics.get('cpu_util', 0)
        
        # Create figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Cost Efficiency Analysis: {instance_type} on {provider.upper()}', fontsize=16)
        
        # 1. Cost efficiency gauge (upper left)
        ax = axs[0, 0]
        efficiency = gpu_util / 100.0  # Simple efficiency metric based on GPU utilization
        cost_efficiency = efficiency * 100
        
        # Create gauge chart using matplotlib
        gauge_colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40']
        gauge_boundaries = [0, 30, 60, 80, 100]
        
        # Set up gauge
        theta = np.linspace(0, 180, 100)
        r = 1.0
        
        # Draw gauge background
        for i in range(len(gauge_boundaries) - 1):
            start = gauge_boundaries[i]
            end = gauge_boundaries[i + 1]
            indices = (start <= theta) & (theta <= end)
            color = gauge_colors[i]
            ax.fill_between(np.deg2rad(theta[indices]), 0, r, color=color, alpha=0.7)
        
        # Draw needle
        needle_angle = np.deg2rad(cost_efficiency * 1.8)  # Scale to 180 degrees
        ax.plot([0, np.sin(needle_angle)], [0, np.cos(needle_angle)], 'k-', lw=2)
        ax.add_patch(plt.Circle((0, 0), 0.05, color='k'))
        
        # Set the limits and remove the axes
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
        # Add gauge scale and value
        for i, boundary in enumerate(gauge_boundaries):
            angle = np.deg2rad(boundary * 1.8)
            ax.text(1.1 * np.sin(angle), 1.1 * np.cos(angle), f"{boundary}%", 
                    ha='center', va='center', fontsize=10)
        
        ax.text(0, -0.2, f"Cost Efficiency: {cost_efficiency:.1f}%", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.text(0, -0.3, f"Hourly Cost: ${hourly_cost:.2f}", 
                ha='center', va='center', fontsize=12)
                
        # Add efficiency rating
        if cost_efficiency < 30:
            rating = "Poor"
            color = gauge_colors[0]
        elif cost_efficiency < 60:
            rating = "Fair"
            color = gauge_colors[1]
        elif cost_efficiency < 80:
            rating = "Good"
            color = gauge_colors[2]
        else:
            rating = "Excellent"
            color = gauge_colors[3]
            
        ax.text(0, -0.4, f"Rating: {rating}", 
                ha='center', va='center', fontsize=12, 
                color=color, fontweight='bold')
        
        # 2. Cost breakdown across resources (upper right)
        ax = axs[0, 1]
        
        # Estimate cost breakdown
        instance_specs = pricing_data.get(instance_type, {})
        gpu_count = instance_specs.get('gpus', 1)
        gpu_type = instance_specs.get('gpu_type', 'Unknown')
        
        # Simplified cost breakdown (would be more accurate with provider-specific data)
        gpu_cost_pct = 0.7  # 70% of cost typically goes to GPU
        memory_cost_pct = 0.15  # 15% to memory
        cpu_cost_pct = 0.1  # 10% to CPU
        other_cost_pct = 0.05  # 5% to other resources
        
        resources = ['GPU', 'Memory', 'CPU', 'Other']
        cost_pcts = [gpu_cost_pct, memory_cost_pct, cpu_cost_pct, other_cost_pct]
        utilization = [gpu_util/100, memory_util/100, cpu_util/100, 1.0]  # Normalize to 0-1
        
        # Adjust for missing utilization data
        utilization = [u if u > 0 else 0.5 for u in utilization]  # Default to 50% if missing
        
        # Calculate costs and wasted costs
        costs = [hourly_cost * pct for pct in cost_pcts]
        effective_costs = [cost * util for cost, util in zip(costs, utilization)]
        wasted_costs = [costs[i] - effective_costs[i] for i in range(len(costs))]
        
        # Stacked bar chart
        width = 0.5
        ax.bar(resources, effective_costs, width, label='Effective Cost', color='#2ECC40')
        ax.bar(resources, wasted_costs, width, bottom=effective_costs, label='Wasted Cost', color='#FF4136')
        
        # Add labels and legend
        ax.set_title('Hourly Cost Breakdown', fontsize=12)
        ax.set_ylabel('Cost ($)', fontsize=10)
        ax.legend()
        
        # Add value labels
        for i, resource in enumerate(resources):
            total = costs[i]
            wasted = wasted_costs[i]
            pct_wasted = (wasted / total) * 100 if total > 0 else 0
            
            # Show wasted percentage
            ax.text(i, costs[i] + 0.01, f"{pct_wasted:.1f}% wasted", 
                    ha='center', va='bottom', fontsize=8)
                    
            # Show total cost
            ax.text(i, 0.02, f"${total:.2f}/hr", 
                    ha='center', va='bottom', fontsize=8)
        
        # 3. Alternative instance comparison (lower left)
        ax = axs[1, 0]
        
        # Get alternatives for comparison
        alternatives = self._get_alternative_instances(provider, instance_type, utilization_metrics, 'training')
        
        if alternatives:
            # Prepare data for comparison
            alt_names = [alt['instance'].split()[0] if len(alt['instance'].split()) > 0 else alt['instance'] 
                        for alt in alternatives[:4]]  # Limit to 4 alternatives
            alt_names.insert(0, instance_type)  # Add current instance
            
            alt_costs = [alt['cost_per_hour'] for alt in alternatives[:4]]
            alt_costs.insert(0, hourly_cost)  # Add current instance cost
            
            # Adjust instance names for display (shortened)
            display_names = [name[-12:] if len(name) > 12 else name for name in alt_names]
            
            # Color current instance differently
            colors = ['#3498db' if i == 0 else '#2ecc71' for i in range(len(alt_names))]
            
            # Create bar chart
            ax.bar(display_names, alt_costs, color=colors)
            ax.set_title('Cost Comparison with Alternatives', fontsize=12)
            ax.set_ylabel('Hourly Cost ($)', fontsize=10)
            ax.set_xticklabels(display_names, rotation=45, ha='right')
            
            # Add value and savings labels
            for i, cost in enumerate(alt_costs):
                if i > 0:  # Alternative instance
                    savings = (alt_costs[0] - cost) / alt_costs[0] * 100
                    ax.text(i, cost + 0.1, f"{savings:.1f}% less", 
                            ha='center', va='bottom', fontsize=8, color='green')
                
                # Show cost
                ax.text(i, cost / 2, f"${cost:.2f}/hr", 
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No suitable alternatives found", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # 4. Projected monthly costs with different scenarios (lower right)
        ax = axs[1, 1]
        
        # Calculate monthly costs under different scenarios
        hours_per_month = 730  # Average hours in a month
        
        # Scenario 1: Current utilization with on-demand pricing
        current_monthly = hourly_cost * hours_per_month
        
        # Scenario 2: Optimized utilization (improve by 20%)
        optimized_monthly = hourly_cost * hours_per_month * 0.8
        
        # Scenario 3: Current utilization with spot instances
        spot_discount = self.SPOT_DISCOUNTS.get(provider.lower(), 0.6)
        spot_monthly = hourly_cost * hours_per_month * (1 - spot_discount)
        
        # Scenario 4: Optimized utilization with spot instances
        optimized_spot_monthly = optimized_monthly * (1 - spot_discount)
        
        # Scenario names and values
        scenarios = ['Current\n(On-Demand)', 'Optimized\nUtilization', 'Spot\nInstances', 'Optimized\n+ Spot']
        monthly_costs = [current_monthly, optimized_monthly, spot_monthly, optimized_spot_monthly]
        
        # Use different colors for different scenarios
        scenario_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        # Create bar chart
        ax.bar(scenarios, monthly_costs, color=scenario_colors)
        ax.set_title('Projected Monthly Costs (Different Scenarios)', fontsize=12)
        ax.set_ylabel('Monthly Cost ($)', fontsize=10)
        
        # Add value and savings labels
        for i, cost in enumerate(monthly_costs):
            if i > 0:  # Not the first scenario
                savings = (monthly_costs[0] - cost) / monthly_costs[0] * 100
                ax.text(i, cost + current_monthly * 0.02, f"{savings:.1f}% less", 
                        ha='center', va='bottom', fontsize=8)
            
            # Show monthly cost
            ax.text(i, cost / 2, f"${cost:.0f}/mo", 
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Adjust layout and return figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        return fig

    def forecast_costs(self, historical_metrics, provider, instance_type, forecast_days=30):
        """
        Forecast future costs based on historical usage patterns
        
        Args:
            historical_metrics: List of dictionaries with timestamps and utilization metrics
            provider: Cloud provider name
            instance_type: Instance type name
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast data and visualizations
        """
        if not historical_metrics:
            return {"error": "No historical data provided for forecasting"}
            
        # Extract pricing information
        pricing_data = self._get_pricing_data(provider)
        hourly_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        
        # Convert historical data to time series
        dates = []
        utils = []
        costs = []
        
        for entry in historical_metrics:
            if 'timestamp' in entry and 'gpu_util' in entry:
                # Convert timestamp to datetime if it's a string
                if isinstance(entry['timestamp'], str):
                    timestamp = datetime.datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                else:
                    timestamp = entry['timestamp']
                    
                dates.append(timestamp)
                utils.append(entry['gpu_util'])
                
                # Calculate hourly cost based on utilization
                # Here we're using a simple model where cost scales with time
                # In reality, you might have a more complex cost model
                period_cost = hourly_cost / 60  # Assuming data points are minutes
                costs.append(period_cost)
        
        if not dates:
            return {"error": "Could not parse historical data timestamps"}
            
        # Aggregate by day for forecasting
        df = pd.DataFrame({
            'date': dates,
            'utilization': utils,
            'hourly_cost': costs
        })
        
        # Convert to pandas datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Resample to daily data for forecasting
        daily_data = df.resample('D').agg({
            'utilization': 'mean',
            'hourly_cost': 'sum'
        })
        
        # Calculate daily cost
        daily_data['daily_cost'] = daily_data['hourly_cost'] * 24
        
        # Fill missing values using forward fill
        daily_data = daily_data.fillna(method='ffill')
        
        # Use a simple time series model for forecasting
        # In a production system, you might use ARIMA, Prophet, or other advanced models
        # Simple trailing average forecast
        forecast_start = daily_data.index[-1] + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=forecast_days)
        forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        
        # Create forecast DataFrame
        forecast = pd.DataFrame(index=forecast_index)
        
        # Use average of last 7 days for forecasting
        avg_util = daily_data['utilization'].tail(7).mean()
        avg_cost = daily_data['daily_cost'].tail(7).mean()
        
        # Apply simple growth model (optional)
        # Here we assume a small growth in usage over time
        growth_rate = 0.02  # 2% growth per month
        daily_growth = (1 + growth_rate) ** (1/30) - 1  # Convert to daily growth rate
        
        forecast['utilization'] = [avg_util * ((1 + daily_growth) ** i) for i in range(len(forecast_index))]
        forecast['daily_cost'] = [avg_cost * ((1 + daily_growth) ** i) for i in range(len(forecast_index))]
        
        # Calculate cumulative and total forecast cost
        forecast['cumulative_cost'] = forecast['daily_cost'].cumsum()
        total_forecast_cost = forecast['daily_cost'].sum()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot historical and forecasted utilization
        ax1.plot(daily_data.index, daily_data['utilization'], 'b-', label='Historical Utilization')
        ax1.plot(forecast.index, forecast['utilization'], 'r--', label='Forecasted Utilization')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('Historical and Forecasted GPU Utilization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot historical and forecasted daily cost
        ax2.plot(daily_data.index, daily_data['daily_cost'], 'g-', label='Historical Cost')
        ax2.plot(forecast.index, forecast['daily_cost'], 'r--', label='Forecasted Cost')
        ax2.set_ylabel('Daily Cost ($)')
        ax2.set_title('Historical and Forecasted Daily Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate key metrics
        historical_total = daily_data['daily_cost'].sum()
        historical_avg = daily_data['daily_cost'].mean()
        forecast_avg = forecast['daily_cost'].mean()
        cost_trend = (forecast_avg / historical_avg - 1) * 100
        
        return {
            'forecast_data': forecast.reset_index().to_dict('records'),
            'historical_data': daily_data.reset_index().to_dict('records'),
            'total_forecast_cost': total_forecast_cost,
            'avg_daily_cost': forecast_avg,
            'cost_trend_pct': cost_trend,
            'forecast_period_days': forecast_days,
            'visualization': fig,
            'forecast_summary': {
                'start_date': forecast_start.strftime('%Y-%m-%d'),
                'end_date': forecast_end.strftime('%Y-%m-%d'),
                'total_cost': total_forecast_cost,
                'avg_daily_cost': forecast_avg,
                'cost_trend': f"{cost_trend:.1f}% {'increase' if cost_trend > 0 else 'decrease'} compared to historical average"
            }
        }

    def generate_cost_report(self, provider, instance_type, utilization_metrics, 
                           training_hours=1000, with_alternatives=True, 
                           include_reserved=True, include_regions=True, 
                           sustainability_analysis=True):
        """
        Generate a comprehensive cost report with all analyses and recommendations
        
        Args:
            provider: Cloud provider name
            instance_type: Current instance type
            utilization_metrics: Dict with utilization metrics
            training_hours: Expected total training hours
            with_alternatives: Whether to include alternative instance types
            include_reserved: Whether to include reserved instance analysis
            include_regions: Whether to include multi-region comparison
            sustainability_analysis: Whether to include CO2 emissions analysis
            
        Returns:
            Dictionary with comprehensive cost analysis and recommendations
        """
        # Basic training costs and recommendations
        basic_analysis = self.analyze_training_costs(
            provider, instance_type, utilization_metrics, training_hours
        )
        
        # Create report dictionary
        report = {
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'provider': provider,
            'instance_type': instance_type,
            'training_hours': training_hours,
            'current_cost_per_hour': basic_analysis.current_cost_per_hour,
            'total_cost_estimate': basic_analysis.current_cost_per_hour * training_hours,
            'recommended_instance': basic_analysis.recommended_instance,
            'potential_savings_percentage': basic_analysis.potential_savings_percentage,
            'recommendations': basic_analysis.recommendations,
        }
        
        # Add reserved instance analysis if requested
        if include_reserved:
            reserved_analysis = self.analyze_reserved_vs_ondemand(
                provider, instance_type, estimated_usage_months=training_hours/730
            )
            report['reserved_instance_analysis'] = reserved_analysis
            
            # Add reserved recommendation to main recommendations if it's beneficial
            if reserved_analysis.get('recommendation', {}).get('type') == 'reserved_instance':
                report['recommendations'].append(reserved_analysis['recommendation'])
        
        # Add regional comparison if requested
        if include_regions:
            region_analysis = self.generate_multi_region_comparison(
                provider, instance_type, utilization_metrics, training_hours
            )
            report['regional_analysis'] = region_analysis
            
            # Add regional recommendation if savings are significant
            if region_analysis.get('savings_percentage', 0) > 10:
                report['recommendations'].append({
                    'type': 'region_selection',
                    'severity': 'medium',
                    'description': f"Significant cost difference between regions detected",
                    'recommendation': f"Consider using {region_analysis['cheapest_region']} region for {region_analysis['savings_percentage']:.1f}% lower costs",
                    'potential_savings': f"${region_analysis['cost_savings_vs_most_expensive']:.2f}"
                })
        
        # Add sustainability analysis if requested
        if sustainability_analysis and include_regions:
            greenest_region = region_analysis.get('greenest_region')
            cheapest_region = region_analysis.get('cheapest_region')
            
            if greenest_region and greenest_region != cheapest_region:
                # Compare emissions
                cheapest_emissions = region_analysis['detailed_comparison'][cheapest_region]['carbon_emissions_kg']
                greenest_emissions = region_analysis['detailed_comparison'][greenest_region]['carbon_emissions_kg']
                
                emissions_diff = cheapest_emissions - greenest_emissions
                emissions_pct = (emissions_diff / cheapest_emissions) * 100 if cheapest_emissions > 0 else 0
                
                if emissions_pct > 20:  # If greenest region has 20% lower emissions
                    # Calculate cost difference for using greenest region
                    greenest_cost = region_analysis['detailed_comparison'][greenest_region]['total_cost']
                    cheapest_cost = region_analysis['detailed_comparison'][cheapest_region]['total_cost']
                    
                    cost_diff = greenest_cost - cheapest_cost
                    cost_pct = (cost_diff / cheapest_cost) * 100 if cheapest_cost > 0 else 0
                    
                    report['recommendations'].append({
                        'type': 'sustainability',
                        'severity': 'low',
                        'description': f"Significant CO2 emissions difference between regions",
                        'recommendation': f"For sustainability, consider {greenest_region} with {emissions_pct:.1f}% lower CO2 emissions",
                        'cost_impact': f"{cost_pct:.1f}% higher cost (${cost_diff:.2f})",
                        'emissions_savings': f"{emissions_diff:.2f} kg CO2 ({emissions_pct:.1f}%)"
                    })
            
            # Add overall sustainability metrics
            report['sustainability_metrics'] = {
                'estimated_carbon_emissions_kg': region_analysis['detailed_comparison'][cheapest_region]['carbon_emissions_kg'],
                'sustainability_score': region_analysis['detailed_comparison'][cheapest_region]['sustainability_score'],
                'greenest_region': greenest_region,
                'greenest_region_emissions_kg': region_analysis['detailed_comparison'][greenest_region]['carbon_emissions_kg']
            }
        
        # Add alternative instances if requested
        if with_alternatives:
            report['alternative_options'] = basic_analysis.alternative_options
        
        # Calculate total potential savings across all recommendations
        total_savings = 0
        for rec in report['recommendations']:
            if 'potential_savings' in rec:
                # Extract numeric value from potential savings string (removing $ and % signs)
                savings_str = rec['potential_savings']
                try:
                    if savings_str.startswith('$'):
                        total_savings += float(savings_str.replace('$', '').split()[0])
                except:
                    pass  # Skip if we can't parse the savings amount
        
        report['total_potential_savings'] = total_savings
        
        # Sort recommendations by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        report['recommendations'] = sorted(
            report['recommendations'], 
            key=lambda x: severity_order.get(x.get('severity', 'low'), 99)
        )
        
        return report

    def compare_instance_families(self, provider, workload_type='training', comparison_metric='cost_efficiency'):
        """
        Compare different instance families for specific workload types
        
        Args:
            provider: Cloud provider name
            workload_type: 'training', 'inference', or 'nlp', 'cv', etc.
            comparison_metric: Metric to use for comparison
            
        Returns:
            Dictionary with comparison data and visualizations
        """
        pricing_data = self._get_pricing_data(provider)
        
        # Group instances by family
        instance_families = {}
        
        for instance, specs in pricing_data.items():
            # Extract family from instance name
            if provider.lower() == 'aws':
                # AWS example: p3.2xlarge -> family is p3
                family = instance.split('.')[0]
            elif provider.lower() == 'gcp':
                # GCP example: n1-standard-4-nvidia-tesla-t4 -> we use the GPU type
                family = specs.get('gpu_type', 'unknown')
            elif provider.lower() == 'azure':
                # Azure example: Standard_NC6s_v3 -> family is NC
                parts = instance.split('_')
                if len(parts) > 1:
                    family = ''.join(filter(str.isalpha, parts[1]))
                else:
                    family = 'unknown'
            else:
                family = 'unknown'
                
            # Group by family
            if family not in instance_families:
                instance_families[family] = []
                
            instance_families[family].append({
                'instance': instance,
                'specs': specs
            })
        
        # Calculate comparison metrics for each family
        family_metrics = {}
        
        for family, instances in instance_families.items():
            # Skip families with no instances
            if not instances:
                continue
                
            # Calculate average cost per GPU
            total_cost = sum(inst['specs'].get('cost', 0) for inst in instances)
            total_gpus = sum(inst['specs'].get('gpus', 0) for inst in instances)
            
            cost_per_gpu = total_cost / total_gpus if total_gpus > 0 else float('inf')
            
            # Metric for memory per GPU
            total_memory = sum(inst['specs'].get('ram', 0) for inst in instances)
            memory_per_gpu = total_memory / total_gpus if total_gpus > 0 else 0
            
            # Representative instance from this family
            representative = max(instances, key=lambda x: x['specs'].get('gpus', 0))
            
            # Workload suitability scores (simplified)
            suitability_scores = {
                'training': 0,
                'inference': 0
            }
            
            # Calculate suitability based on hardware specs
            # For training: prefer GPUs with high FLOPS and memory bandwidth
            # For inference: prefer cost-efficient GPUs with good memory
            gpu_type = representative['specs'].get('gpu_type', '')
            
            # Training score factors
            if gpu_type in ['V100', 'A100']:
                suitability_scores['training'] = 9
            elif gpu_type in ['P100', 'M60']:
                suitability_scores['training'] = 6
            elif gpu_type in ['T4', 'A10G']:
                suitability_scores['training'] = 7
            else:
                suitability_scores['training'] = 4
                
            # Inference score factors
            if gpu_type in ['T4', 'A10G']:
                suitability_scores['inference'] = 9
            elif gpu_type in ['V100', 'A100']:
                suitability_scores['inference'] = 7  # Great but expensive for inference
            else:
                suitability_scores['inference'] = 5
                
            # Adjust for cost efficiency
            suitability_scores['training'] *= (1 / (cost_per_gpu ** 0.5)) * 3  # Cost matters less for training
            suitability_scores['inference'] *= (1 / cost_per_gpu) * 5  # Cost matters more for inference
            
            # Normalize scores to 0-10 range
            for key in suitability_scores:
                suitability_scores[key] = min(10, max(0, suitability_scores[key]))
            
            # Store metrics for this family
            family_metrics[family] = {
                'instances': len(instances),
                'cost_per_gpu': cost_per_gpu,
                'memory_per_gpu': memory_per_gpu,
                'representative_instance': representative['instance'],
                'gpu_type': gpu_type,
                'workload_suitability': suitability_scores,
                'instances_list': [inst['instance'] for inst in instances]
            }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Sort families by the requested comparison metric
        if comparison_metric == 'cost_efficiency':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['cost_per_gpu'])
        elif comparison_metric == 'training_suitability':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['training'], reverse=True)
        elif comparison_metric == 'inference_suitability':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['inference'], reverse=True)
        else:
            sorted_families = sorted(family_metrics.items())
            
        # Limit to top 10 families for visualization
        sorted_families = sorted_families[:10]
        
        # Prepare data for visualization
        family_names = [f[0] for f in sorted_families]
        cost_per_gpu = [f[1]['cost_per_gpu'] for f in sorted_families]
        training_scores = [f[1]['workload_suitability']['training'] for f in sorted_families]
        inference_scores = [f[1]['workload_suitability']['inference'] for f in sorted_families]
        
        # Cost per GPU visualization
        ax1.bar(family_names, cost_per_gpu, color='#3498db')
        ax1.set_title('Cost per GPU by Instance Family', fontsize=12)
        ax1.set_ylabel('Hourly Cost per GPU ($)', fontsize=# filepath: c:\Users\adilm\repositories\Python\neural-scope\advanced_analysis\cloud_cost_optimizer.py
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
        memory_util = utilization_metrics.get('memory_util', 0)
        cpu_util = utilization_metrics.get('cpu_util', 0)
        
        # Create figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Cost Efficiency Analysis: {instance_type} on {provider.upper()}', fontsize=16)
        
        # 1. Cost efficiency gauge (upper left)
        ax = axs[0, 0]
        efficiency = gpu_util / 100.0  # Simple efficiency metric based on GPU utilization
        cost_efficiency = efficiency * 100
        
        # Create gauge chart using matplotlib
        gauge_colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40']
        gauge_boundaries = [0, 30, 60, 80, 100]
        
        # Set up gauge
        theta = np.linspace(0, 180, 100)
        r = 1.0
        
        # Draw gauge background
        for i in range(len(gauge_boundaries) - 1):
            start = gauge_boundaries[i]
            end = gauge_boundaries[i + 1]
            indices = (start <= theta) & (theta <= end)
            color = gauge_colors[i]
            ax.fill_between(np.deg2rad(theta[indices]), 0, r, color=color, alpha=0.7)
        
        # Draw needle
        needle_angle = np.deg2rad(cost_efficiency * 1.8)  # Scale to 180 degrees
        ax.plot([0, np.sin(needle_angle)], [0, np.cos(needle_angle)], 'k-', lw=2)
        ax.add_patch(plt.Circle((0, 0), 0.05, color='k'))
        
        # Set the limits and remove the axes
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
        # Add gauge scale and value
        for i, boundary in enumerate(gauge_boundaries):
            angle = np.deg2rad(boundary * 1.8)
            ax.text(1.1 * np.sin(angle), 1.1 * np.cos(angle), f"{boundary}%", 
                    ha='center', va='center', fontsize=10)
        
        ax.text(0, -0.2, f"Cost Efficiency: {cost_efficiency:.1f}%", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.text(0, -0.3, f"Hourly Cost: ${hourly_cost:.2f}", 
                ha='center', va='center', fontsize=12)
                
        # Add efficiency rating
        if cost_efficiency < 30:
            rating = "Poor"
            color = gauge_colors[0]
        elif cost_efficiency < 60:
            rating = "Fair"
            color = gauge_colors[1]
        elif cost_efficiency < 80:
            rating = "Good"
            color = gauge_colors[2]
        else:
            rating = "Excellent"
            color = gauge_colors[3]
            
        ax.text(0, -0.4, f"Rating: {rating}", 
                ha='center', va='center', fontsize=12, 
                color=color, fontweight='bold')
        
        # 2. Cost breakdown across resources (upper right)
        ax = axs[0, 1]
        
        # Estimate cost breakdown
        instance_specs = pricing_data.get(instance_type, {})
        gpu_count = instance_specs.get('gpus', 1)
        gpu_type = instance_specs.get('gpu_type', 'Unknown')
        
        # Simplified cost breakdown (would be more accurate with provider-specific data)
        gpu_cost_pct = 0.7  # 70% of cost typically goes to GPU
        memory_cost_pct = 0.15  # 15% to memory
        cpu_cost_pct = 0.1  # 10% to CPU
        other_cost_pct = 0.05  # 5% to other resources
        
        resources = ['GPU', 'Memory', 'CPU', 'Other']
        cost_pcts = [gpu_cost_pct, memory_cost_pct, cpu_cost_pct, other_cost_pct]
        utilization = [gpu_util/100, memory_util/100, cpu_util/100, 1.0]  # Normalize to 0-1
        
        # Adjust for missing utilization data
        utilization = [u if u > 0 else 0.5 for u in utilization]  # Default to 50% if missing
        
        # Calculate costs and wasted costs
        costs = [hourly_cost * pct for pct in cost_pcts]
        effective_costs = [cost * util for cost, util in zip(costs, utilization)]
        wasted_costs = [costs[i] - effective_costs[i] for i in range(len(costs))]
        
        # Stacked bar chart
        width = 0.5
        ax.bar(resources, effective_costs, width, label='Effective Cost', color='#2ECC40')
        ax.bar(resources, wasted_costs, width, bottom=effective_costs, label='Wasted Cost', color='#FF4136')
        
        # Add labels and legend
        ax.set_title('Hourly Cost Breakdown', fontsize=12)
        ax.set_ylabel('Cost ($)', fontsize=10)
        ax.legend()
        
        # Add value labels
        for i, resource in enumerate(resources):
            total = costs[i]
            wasted = wasted_costs[i]
            pct_wasted = (wasted / total) * 100 if total > 0 else 0
            
            # Show wasted percentage
            ax.text(i, costs[i] + 0.01, f"{pct_wasted:.1f}% wasted", 
                    ha='center', va='bottom', fontsize=8)
                    
            # Show total cost
            ax.text(i, 0.02, f"${total:.2f}/hr", 
                    ha='center', va='bottom', fontsize=8)
        
        # 3. Alternative instance comparison (lower left)
        ax = axs[1, 0]
        
        # Get alternatives for comparison
        alternatives = self._get_alternative_instances(provider, instance_type, utilization_metrics, 'training')
        
        if alternatives:
            # Prepare data for comparison
            alt_names = [alt['instance'].split()[0] if len(alt['instance'].split()) > 0 else alt['instance'] 
                        for alt in alternatives[:4]]  # Limit to 4 alternatives
            alt_names.insert(0, instance_type)  # Add current instance
            
            alt_costs = [alt['cost_per_hour'] for alt in alternatives[:4]]
            alt_costs.insert(0, hourly_cost)  # Add current instance cost
            
            # Adjust instance names for display (shortened)
            display_names = [name[-12:] if len(name) > 12 else name for name in alt_names]
            
            # Color current instance differently
            colors = ['#3498db' if i == 0 else '#2ecc71' for i in range(len(alt_names))]
            
            # Create bar chart
            ax.bar(display_names, alt_costs, color=colors)
            ax.set_title('Cost Comparison with Alternatives', fontsize=12)
            ax.set_ylabel('Hourly Cost ($)', fontsize=10)
            ax.set_xticklabels(display_names, rotation=45, ha='right')
            
            # Add value and savings labels
            for i, cost in enumerate(alt_costs):
                if i > 0:  # Alternative instance
                    savings = (alt_costs[0] - cost) / alt_costs[0] * 100
                    ax.text(i, cost + 0.1, f"{savings:.1f}% less", 
                            ha='center', va='bottom', fontsize=8, color='green')
                
                # Show cost
                ax.text(i, cost / 2, f"${cost:.2f}/hr", 
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No suitable alternatives found", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # 4. Projected monthly costs with different scenarios (lower right)
        ax = axs[1, 1]
        
        # Calculate monthly costs under different scenarios
        hours_per_month = 730  # Average hours in a month
        
        # Scenario 1: Current utilization with on-demand pricing
        current_monthly = hourly_cost * hours_per_month
        
        # Scenario 2: Optimized utilization (improve by 20%)
        optimized_monthly = hourly_cost * hours_per_month * 0.8
        
        # Scenario 3: Current utilization with spot instances
        spot_discount = self.SPOT_DISCOUNTS.get(provider.lower(), 0.6)
        spot_monthly = hourly_cost * hours_per_month * (1 - spot_discount)
        
        # Scenario 4: Optimized utilization with spot instances
        optimized_spot_monthly = optimized_monthly * (1 - spot_discount)
        
        # Scenario names and values
        scenarios = ['Current\n(On-Demand)', 'Optimized\nUtilization', 'Spot\nInstances', 'Optimized\n+ Spot']
        monthly_costs = [current_monthly, optimized_monthly, spot_monthly, optimized_spot_monthly]
        
        # Use different colors for different scenarios
        scenario_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        # Create bar chart
        ax.bar(scenarios, monthly_costs, color=scenario_colors)
        ax.set_title('Projected Monthly Costs (Different Scenarios)', fontsize=12)
        ax.set_ylabel('Monthly Cost ($)', fontsize=10)
        
        # Add value and savings labels
        for i, cost in enumerate(monthly_costs):
            if i > 0:  # Not the first scenario
                savings = (monthly_costs[0] - cost) / monthly_costs[0] * 100
                ax.text(i, cost + current_monthly * 0.02, f"{savings:.1f}% less", 
                        ha='center', va='bottom', fontsize=8)
            
            # Show monthly cost
            ax.text(i, cost / 2, f"${cost:.0f}/mo", 
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Adjust layout and return figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        return fig

    def forecast_costs(self, historical_metrics, provider, instance_type, forecast_days=30):
        """
        Forecast future costs based on historical usage patterns
        
        Args:
            historical_metrics: List of dictionaries with timestamps and utilization metrics
            provider: Cloud provider name
            instance_type: Instance type name
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast data and visualizations
        """
        if not historical_metrics:
            return {"error": "No historical data provided for forecasting"}
            
        # Extract pricing information
        pricing_data = self._get_pricing_data(provider)
        hourly_cost = pricing_data.get(instance_type, {}).get('cost', 0)
        
        # Convert historical data to time series
        dates = []
        utils = []
        costs = []
        
        for entry in historical_metrics:
            if 'timestamp' in entry and 'gpu_util' in entry:
                # Convert timestamp to datetime if it's a string
                if isinstance(entry['timestamp'], str):
                    timestamp = datetime.datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                else:
                    timestamp = entry['timestamp']
                    
                dates.append(timestamp)
                utils.append(entry['gpu_util'])
                
                # Calculate hourly cost based on utilization
                # Here we're using a simple model where cost scales with time
                # In reality, you might have a more complex cost model
                period_cost = hourly_cost / 60  # Assuming data points are minutes
                costs.append(period_cost)
        
        if not dates:
            return {"error": "Could not parse historical data timestamps"}
            
        # Aggregate by day for forecasting
        df = pd.DataFrame({
            'date': dates,
            'utilization': utils,
            'hourly_cost': costs
        })
        
        # Convert to pandas datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Resample to daily data for forecasting
        daily_data = df.resample('D').agg({
            'utilization': 'mean',
            'hourly_cost': 'sum'
        })
        
        # Calculate daily cost
        daily_data['daily_cost'] = daily_data['hourly_cost'] * 24
        
        # Fill missing values using forward fill
        daily_data = daily_data.fillna(method='ffill')
        
        # Use a simple time series model for forecasting
        # In a production system, you might use ARIMA, Prophet, or other advanced models
        # Simple trailing average forecast
        forecast_start = daily_data.index[-1] + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=forecast_days)
        forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        
        # Create forecast DataFrame
        forecast = pd.DataFrame(index=forecast_index)
        
        # Use average of last 7 days for forecasting
        avg_util = daily_data['utilization'].tail(7).mean()
        avg_cost = daily_data['daily_cost'].tail(7).mean()
        
        # Apply simple growth model (optional)
        # Here we assume a small growth in usage over time
        growth_rate = 0.02  # 2% growth per month
        daily_growth = (1 + growth_rate) ** (1/30) - 1  # Convert to daily growth rate
        
        forecast['utilization'] = [avg_util * ((1 + daily_growth) ** i) for i in range(len(forecast_index))]
        forecast['daily_cost'] = [avg_cost * ((1 + daily_growth) ** i) for i in range(len(forecast_index))]
        
        # Calculate cumulative and total forecast cost
        forecast['cumulative_cost'] = forecast['daily_cost'].cumsum()
        total_forecast_cost = forecast['daily_cost'].sum()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot historical and forecasted utilization
        ax1.plot(daily_data.index, daily_data['utilization'], 'b-', label='Historical Utilization')
        ax1.plot(forecast.index, forecast['utilization'], 'r--', label='Forecasted Utilization')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('Historical and Forecasted GPU Utilization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot historical and forecasted daily cost
        ax2.plot(daily_data.index, daily_data['daily_cost'], 'g-', label='Historical Cost')
        ax2.plot(forecast.index, forecast['daily_cost'], 'r--', label='Forecasted Cost')
        ax2.set_ylabel('Daily Cost ($)')
        ax2.set_title('Historical and Forecasted Daily Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate key metrics
        historical_total = daily_data['daily_cost'].sum()
        historical_avg = daily_data['daily_cost'].mean()
        forecast_avg = forecast['daily_cost'].mean()
        cost_trend = (forecast_avg / historical_avg - 1) * 100
        
        return {
            'forecast_data': forecast.reset_index().to_dict('records'),
            'historical_data': daily_data.reset_index().to_dict('records'),
            'total_forecast_cost': total_forecast_cost,
            'avg_daily_cost': forecast_avg,
            'cost_trend_pct': cost_trend,
            'forecast_period_days': forecast_days,
            'visualization': fig,
            'forecast_summary': {
                'start_date': forecast_start.strftime('%Y-%m-%d'),
                'end_date': forecast_end.strftime('%Y-%m-%d'),
                'total_cost': total_forecast_cost,
                'avg_daily_cost': forecast_avg,
                'cost_trend': f"{cost_trend:.1f}% {'increase' if cost_trend > 0 else 'decrease'} compared to historical average"
            }
        }

    def generate_cost_report(self, provider, instance_type, utilization_metrics, 
                           training_hours=1000, with_alternatives=True, 
                           include_reserved=True, include_regions=True, 
                           sustainability_analysis=True):
        """
        Generate a comprehensive cost report with all analyses and recommendations
        
        Args:
            provider: Cloud provider name
            instance_type: Current instance type
            utilization_metrics: Dict with utilization metrics
            training_hours: Expected total training hours
            with_alternatives: Whether to include alternative instance types
            include_reserved: Whether to include reserved instance analysis
            include_regions: Whether to include multi-region comparison
            sustainability_analysis: Whether to include CO2 emissions analysis
            
        Returns:
            Dictionary with comprehensive cost analysis and recommendations
        """
        # Basic training costs and recommendations
        basic_analysis = self.analyze_training_costs(
            provider, instance_type, utilization_metrics, training_hours
        )
        
        # Create report dictionary
        report = {
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'provider': provider,
            'instance_type': instance_type,
            'training_hours': training_hours,
            'current_cost_per_hour': basic_analysis.current_cost_per_hour,
            'total_cost_estimate': basic_analysis.current_cost_per_hour * training_hours,
            'recommended_instance': basic_analysis.recommended_instance,
            'potential_savings_percentage': basic_analysis.potential_savings_percentage,
            'recommendations': basic_analysis.recommendations,
        }
        
        # Add reserved instance analysis if requested
        if include_reserved:
            reserved_analysis = self.analyze_reserved_vs_ondemand(
                provider, instance_type, estimated_usage_months=training_hours/730
            )
            report['reserved_instance_analysis'] = reserved_analysis
            
            # Add reserved recommendation to main recommendations if it's beneficial
            if reserved_analysis.get('recommendation', {}).get('type') == 'reserved_instance':
                report['recommendations'].append(reserved_analysis['recommendation'])
        
        # Add regional comparison if requested
        if include_regions:
            region_analysis = self.generate_multi_region_comparison(
                provider, instance_type, utilization_metrics, training_hours
            )
            report['regional_analysis'] = region_analysis
            
            # Add regional recommendation if savings are significant
            if region_analysis.get('savings_percentage', 0) > 10:
                report['recommendations'].append({
                    'type': 'region_selection',
                    'severity': 'medium',
                    'description': f"Significant cost difference between regions detected",
                    'recommendation': f"Consider using {region_analysis['cheapest_region']} region for {region_analysis['savings_percentage']:.1f}% lower costs",
                    'potential_savings': f"${region_analysis['cost_savings_vs_most_expensive']:.2f}"
                })
        
        # Add sustainability analysis if requested
        if sustainability_analysis and include_regions:
            greenest_region = region_analysis.get('greenest_region')
            cheapest_region = region_analysis.get('cheapest_region')
            
            if greenest_region and greenest_region != cheapest_region:
                # Compare emissions
                cheapest_emissions = region_analysis['detailed_comparison'][cheapest_region]['carbon_emissions_kg']
                greenest_emissions = region_analysis['detailed_comparison'][greenest_region]['carbon_emissions_kg']
                
                emissions_diff = cheapest_emissions - greenest_emissions
                emissions_pct = (emissions_diff / cheapest_emissions) * 100 if cheapest_emissions > 0 else 0
                
                if emissions_pct > 20:  # If greenest region has 20% lower emissions
                    # Calculate cost difference for using greenest region
                    greenest_cost = region_analysis['detailed_comparison'][greenest_region]['total_cost']
                    cheapest_cost = region_analysis['detailed_comparison'][cheapest_region]['total_cost']
                    
                    cost_diff = greenest_cost - cheapest_cost
                    cost_pct = (cost_diff / cheapest_cost) * 100 if cheapest_cost > 0 else 0
                    
                    report['recommendations'].append({
                        'type': 'sustainability',
                        'severity': 'low',
                        'description': f"Significant CO2 emissions difference between regions",
                        'recommendation': f"For sustainability, consider {greenest_region} with {emissions_pct:.1f}% lower CO2 emissions",
                        'cost_impact': f"{cost_pct:.1f}% higher cost (${cost_diff:.2f})",
                        'emissions_savings': f"{emissions_diff:.2f} kg CO2 ({emissions_pct:.1f}%)"
                    })
            
            # Add overall sustainability metrics
            report['sustainability_metrics'] = {
                'estimated_carbon_emissions_kg': region_analysis['detailed_comparison'][cheapest_region]['carbon_emissions_kg'],
                'sustainability_score': region_analysis['detailed_comparison'][cheapest_region]['sustainability_score'],
                'greenest_region': greenest_region,
                'greenest_region_emissions_kg': region_analysis['detailed_comparison'][greenest_region]['carbon_emissions_kg']
            }
        
        # Add alternative instances if requested
        if with_alternatives:
            report['alternative_options'] = basic_analysis.alternative_options
        
        # Calculate total potential savings across all recommendations
        total_savings = 0
        for rec in report['recommendations']:
            if 'potential_savings' in rec:
                # Extract numeric value from potential savings string (removing $ and % signs)
                savings_str = rec['potential_savings']
                try:
                    if savings_str.startswith('$'):
                        total_savings += float(savings_str.replace('$', '').split()[0])
                except:
                    pass  # Skip if we can't parse the savings amount
        
        report['total_potential_savings'] = total_savings
        
        # Sort recommendations by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        report['recommendations'] = sorted(
            report['recommendations'], 
            key=lambda x: severity_order.get(x.get('severity', 'low'), 99)
        )
        
        return report

    def compare_instance_families(self, provider, workload_type='training', comparison_metric='cost_efficiency'):
        """
        Compare different instance families for specific workload types
        
        Args:
            provider: Cloud provider name
            workload_type: 'training', 'inference', or 'nlp', 'cv', etc.
            comparison_metric: Metric to use for comparison
            
        Returns:
            Dictionary with comparison data and visualizations
        """
        pricing_data = self._get_pricing_data(provider)
        
        # Group instances by family
        instance_families = {}
        
        for instance, specs in pricing_data.items():
            # Extract family from instance name
            if provider.lower() == 'aws':
                # AWS example: p3.2xlarge -> family is p3
                family = instance.split('.')[0]
            elif provider.lower() == 'gcp':
                # GCP example: n1-standard-4-nvidia-tesla-t4 -> we use the GPU type
                family = specs.get('gpu_type', 'unknown')
            elif provider.lower() == 'azure':
                # Azure example: Standard_NC6s_v3 -> family is NC
                parts = instance.split('_')
                if len(parts) > 1:
                    family = ''.join(filter(str.isalpha, parts[1]))
                else:
                    family = 'unknown'
            else:
                family = 'unknown'
                
            # Group by family
            if family not in instance_families:
                instance_families[family] = []
                
            instance_families[family].append({
                'instance': instance,
                'specs': specs
            })
        
        # Calculate comparison metrics for each family
        family_metrics = {}
        
        for family, instances in instance_families.items():
            # Skip families with no instances
            if not instances:
                continue
                
            # Calculate average cost per GPU
            total_cost = sum(inst['specs'].get('cost', 0) for inst in instances)
            total_gpus = sum(inst['specs'].get('gpus', 0) for inst in instances)
            
            cost_per_gpu = total_cost / total_gpus if total_gpus > 0 else float('inf')
            
            # Metric for memory per GPU
            total_memory = sum(inst['specs'].get('ram', 0) for inst in instances)
            memory_per_gpu = total_memory / total_gpus if total_gpus > 0 else 0
            
            # Representative instance from this family
            representative = max(instances, key=lambda x: x['specs'].get('gpus', 0))
            
            # Workload suitability scores (simplified)
            suitability_scores = {
                'training': 0,
                'inference': 0
            }
            
            # Calculate suitability based on hardware specs
            # For training: prefer GPUs with high FLOPS and memory bandwidth
            # For inference: prefer cost-efficient GPUs with good memory
            gpu_type = representative['specs'].get('gpu_type', '')
            
            # Training score factors
            if gpu_type in ['V100', 'A100']:
                suitability_scores['training'] = 9
            elif gpu_type in ['P100', 'M60']:
                suitability_scores['training'] = 6
            elif gpu_type in ['T4', 'A10G']:
                suitability_scores['training'] = 7
            else:
                suitability_scores['training'] = 4
                
            # Inference score factors
            if gpu_type in ['T4', 'A10G']:
                suitability_scores['inference'] = 9
            elif gpu_type in ['V100', 'A100']:
                suitability_scores['inference'] = 7  # Great but expensive for inference
            else:
                suitability_scores['inference'] = 5
                
            # Adjust for cost efficiency
            suitability_scores['training'] *= (1 / (cost_per_gpu ** 0.5)) * 3  # Cost matters less for training
            suitability_scores['inference'] *= (1 / cost_per_gpu) * 5  # Cost matters more for inference
            
            # Normalize scores to 0-10 range
            for key in suitability_scores:
                suitability_scores[key] = min(10, max(0, suitability_scores[key]))
            
            # Store metrics for this family
            family_metrics[family] = {
                'instances': len(instances),
                'cost_per_gpu': cost_per_gpu,
                'memory_per_gpu': memory_per_gpu,
                'representative_instance': representative['instance'],
                'gpu_type': gpu_type,
                'workload_suitability': suitability_scores,
                'instances_list': [inst['instance'] for inst in instances]
            }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Sort families by the requested comparison metric
        if comparison_metric == 'cost_efficiency':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['cost_per_gpu'])
        elif comparison_metric == 'training_suitability':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['training'], reverse=True)
        elif comparison_metric == 'inference_suitability':
            sorted_families = sorted(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['inference'], reverse=True)
        else:
            sorted_families = sorted(family_metrics.items())
            
        # Limit to top 10 families for visualization
        sorted_families = sorted_families[:10]
        
        # Prepare data for visualization
        family_names = [f[0] for f in sorted_families]
        cost_per_gpu = [f[1]['cost_per_gpu'] for f in sorted_families]
        training_scores = [f[1]['workload_suitability']['training'] for f in sorted_families]
        inference_scores = [f[1]['workload_suitability']['inference'] for f in sorted_families]
        
    
        # Cost per GPU visualization
        ax1.bar(family_names, cost_per_gpu, color='#3498db')
        ax1.set_title('Cost per GPU by Instance Family', fontsize=12)
        ax1.set_ylabel('Hourly Cost per GPU ($)', fontsize=10)
        ax1.set_xticks(range(len(family_names)))
        ax1.set_xticklabels(family_names, rotation=45, ha='right')
        
        # Add cost labels on each bar
        for i, cost in enumerate(cost_per_gpu):
            ax1.text(i, cost + 0.1, f"${cost:.2f}", ha='center', fontsize=9)
        
        # Workload suitability visualization
        x = np.arange(len(family_names))
        width = 0.35
        
        ax2.bar(x - width/2, training_scores, width, label='Training', color='#2ecc71')
        ax2.bar(x + width/2, inference_scores, width, label='Inference', color='#e74c3c')
        
        ax2.set_title(f'Workload Suitability Scores (0-10)', fontsize=12)
        ax2.set_ylabel('Suitability Score', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(family_names, rotation=45, ha='right')
        ax2.set_ylim(0, 10.5)
        ax2.legend()
        
        # Add score labels on each bar
        for i, score in enumerate(training_scores):
            ax2.text(i - width/2, score + 0.3, f"{score:.1f}", ha='center', fontsize=9)
            
        for i, score in enumerate(inference_scores):
            ax2.text(i + width/2, score + 0.3, f"{score:.1f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Generate recommendations based on workload type
        if workload_type == 'training':
            best_family = max(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['training'])
            top_3_training = sorted(family_metrics.items(), 
                                key=lambda x: x[1]['workload_suitability']['training'], 
                                reverse=True)[:3]
        elif workload_type == 'inference':
            best_family = max(family_metrics.items(), key=lambda x: x[1]['workload_suitability']['inference'])
            top_3_training = sorted(family_metrics.items(), 
                                key=lambda x: x[1]['workload_suitability']['inference'], 
                                reverse=True)[:3]
        else:
            # Default to balanced approach
            best_family = max(family_metrics.items(), 
                            key=lambda x: x[1]['workload_suitability']['training'] + 
                                        x[1]['workload_suitability']['inference'])
            top_3_training = sorted(family_metrics.items(), 
                                key=lambda x: x[1]['workload_suitability']['training'] + 
                                            x[1]['workload_suitability']['inference'], 
                                reverse=True)[:3]
        
        recommendations = {
            'best_family': best_family[0],
            'recommended_instance': family_metrics[best_family[0]]['representative_instance'],
            'top_3_families': [(f[0], f[1]['representative_instance']) for f in top_3_training],
            'explanation': f"For {workload_type} workloads, the {best_family[0]} family offers the best "
                         f"balance of performance and cost efficiency."
        }
        
        return {
            'family_metrics': family_metrics,
            'visualization': fig,
            'recommendations': recommendations,
            'sorted_by': comparison_metric
        }
        
    def compare_providers(self, instance_specs, workload='training'):
        """
        Compare pricing and performance across different cloud providers for similar hardware specs
        
        Args:
            instance_specs: Dict with 'gpus', 'gpu_type', 'ram' etc.
            workload: Type of workload ('training', 'inference', etc.)
            
        Returns:
            Dictionary with cross-provider comparison and recommendations
        """
        # Find matching instances across providers
        provider_matches = {}
        
        # Target GPU type and count from request
        target_gpu_type = instance_specs.get('gpu_type')
        target_gpu_count = instance_specs.get('gpus', 1)
        target_ram = instance_specs.get('ram', 0)
        
        # Search across providers
        for provider in ['aws', 'gcp', 'azure']:
            pricing_data = self._get_pricing_data(provider)
            
            # Find matching instances
            matches = []
            for instance, specs in pricing_data.items():
                # Match GPU type if specified
                if target_gpu_type and specs.get('gpu_type') != target_gpu_type:
                    continue
                    
                # Match GPU count (exact match or closest)
                if specs.get('gpus', 0) != target_gpu_count:
                    if workload == 'training':
                        # For training, we might want to consider instances with more GPUs
                        if specs.get('gpus', 0) < target_gpu_count:
                            continue
                    else:
                        # For inference, prefer exact matches but consider close ones
                        if abs(specs.get('gpus', 0) - target_gpu_count) > 1:
                            continue
                
                # Consider RAM requirements
                if target_ram > 0 and specs.get('ram', 0) < target_ram * 0.8:
                    continue
                    
                # Calculate match score (0-100)
                gpu_type_match = 100 if specs.get('gpu_type') == target_gpu_type else 0
                gpu_count_match = 100 - min(100, abs(specs.get('gpus', 0) - target_gpu_count) * 25)
                ram_match = 100 - min(100, abs(specs.get('ram', 0) - target_ram) / target_ram * 100) if target_ram > 0 else 100
                
                match_score = (gpu_type_match * 0.5 + gpu_count_match * 0.3 + ram_match * 0.2)
                
                if match_score > 50:  # Only consider good matches
                    matches.append({
                        'instance': instance,
                        'specs': specs,
                        'match_score': match_score,
                        'hourly_cost': specs.get('cost', 0),
                        'hourly_cost_per_gpu': specs.get('cost', 0) / max(1, specs.get('gpus', 1))
                    })
            
            # Sort matches by match score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            if matches:
                provider_matches[provider] = matches[:3]  # Top 3 matches
        
        # Create comparison visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for visualization
        providers = []
        instance_names = []
        costs = []
        costs_per_gpu = []
        match_scores = []
        bar_colors = {'aws': '#FF9900', 'gcp': '#4285F4', 'azure': '#00A4EF'}
        
        for provider, matches in provider_matches.items():
            if matches:
                # Use best match for each provider
                best_match = matches[0]
                providers.append(provider.upper())
                instance_names.append(best_match['instance'])
                costs.append(best_match['hourly_cost'])
                costs_per_gpu.append(best_match['hourly_cost_per_gpu'])
                match_scores.append(best_match['match_score'])
        
        # Plot cost comparison
        x = np.arange(len(providers))
        width = 0.35
        
        # Hourly cost bars
        cost_bars = ax.bar(x - width/2, costs, width, label='Total Hourly Cost', 
                         color=[bar_colors.get(p.lower(), '#333333') for p in providers])
        
        # Cost per GPU bars
        cost_per_gpu_bars = ax.bar(x + width/2, costs_per_gpu, width, label='Cost per GPU', 
                                alpha=0.7, color=[bar_colors.get(p.lower(), '#333333') for p in providers])
        
        # Add labels and styling
        ax.set_title('Cross-Provider Cost Comparison for Similar Instances', fontsize=14)
        ax.set_ylabel('Hourly Cost ($)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(providers)
        ax.legend()
        
        # Add instance names and match scores
        for i, provider in enumerate(providers):
            ax.text(i, -0.2, f"{instance_names[i]}\nMatch: {match_scores[i]:.0f}%", 
                   ha='center', va='top', fontsize=10, rotation=0)
            
            # Add cost labels on each bar
            ax.text(i - width/2, costs[i] + 0.2, f"${costs[i]:.2f}", ha='center', fontsize=10)
            ax.text(i + width/2, costs_per_gpu[i] + 0.2, f"${costs_per_gpu[i]:.2f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Find cheapest provider
        if costs:
            cheapest_idx = costs.index(min(costs))
            cheapest_provider = providers[cheapest_idx]
            cheapest_instance = instance_names[cheapest_idx]
            
            best_value_idx = costs_per_gpu.index(min(costs_per_gpu))
            best_value_provider = providers[best_value_idx]
            best_value_instance = instance_names[best_value_idx]
            
            # Calculate savings percentages
            savings = {}
            for i, provider in enumerate(providers):
                if i != cheapest_idx:
                    savings[provider] = (costs[i] - costs[cheapest_idx]) / costs[i] * 100
        else:
            cheapest_provider = "No matches found"
            cheapest_instance = "N/A"
            best_value_provider = "No matches found"
            best_value_instance = "N/A"
            savings = {}
        
        return {
            'provider_matches': provider_matches,
            'cheapest_provider': cheapest_provider,
            'cheapest_instance': cheapest_instance,
            'best_value_provider': best_value_provider,
            'best_value_instance': best_value_instance,
            'visualization': fig,
            'savings_percentages': savings
        }
        
    def analyze_tpu_vs_gpu(self, model_size_b, batch_size, seq_length=512, training_hours=1000):
        """
        Analyze when it's beneficial to switch from GPU to TPU for large-scale training
        
        Args:
            model_size_b: Model size in billions of parameters
            batch_size: Global batch size
            seq_length: Sequence length for transformer models
            training_hours: Expected training duration in hours
            
        Returns:
            Dictionary with cost and performance comparison
        """
        # Approximate pricing for different compute options (hourly)
        tpu_v3_8_cost = 9.92  # TPU v3-8
        tpu_v4_8_cost = 11.20 # TPU v4-8 (newer generation, faster)
        tpu_v4_16_cost = 20.80 # TPU v4-16
        tpu_v4_32_cost = 40.60 # TPU v4-32
        tpu_v5_8_cost = 14.00 # TPU v5-8 (estimate)
        
        gpu_a100_8_cost = 32.77  # 8x A100 GPUs (p4d.24xlarge)
        gpu_a100_16_cost = 65.54  # 16x A100 GPUs (2x p4d.24xlarge)
        
        # Approximate FLOPS for different hardware options
        tpu_v3_8_flops = 420e12  # 420 TFLOPS for TPU v3-8
        tpu_v4_8_flops = 693e12  # TPU v4-8 FLOPS
        tpu_v4_16_flops = 1386e12  # TPU v4-16 FLOPS
        tpu_v4_32_flops = 2772e12  # TPU v4-32 FLOPS
        tpu_v5_8_flops = 1400e12  # TPU v5-8 FLOPS (estimate)
        
        a100_flops = 312e12  # 312 TFLOPS for 8x A100 (mixed precision)
        a100_16_flops = 624e12  # 624 TFLOPS for 16x A100
        
        # Estimated FLOPS needed for different model sizes (very rough approximation)
        # Based on general trend that larger models require more compute
        flops_needed = model_size_b * batch_size * seq_length * 6 * 1e9  # Very rough approximation
        
        # Calculate approximate training time for each option (in arbitrary units)
        tpu_v3_8_time = flops_needed / tpu_v3_8_flops
        tpu_v4_8_time = flops_needed / tpu_v4_8_flops
        tpu_v4_16_time = flops_needed / tpu_v4_16_flops
        tpu_v4_32_time = flops_needed / tpu_v4_32_flops
        tpu_v5_8_time = flops_needed / tpu_v5_8_flops
        
        gpu_a100_8_time = flops_needed / a100_flops
        gpu_a100_16_time = flops_needed / a100_16_flops
        
        # Normalize times to the fastest option
        all_times = [tpu_v3_8_time, tpu_v4_8_time, tpu_v4_16_time, tpu_v4_32_time, 
                    tpu_v5_8_time, gpu_a100_8_time, gpu_a100_16_time]
        fastest_time = min(all_times)
        
        # Calculate normalized times (relative to fastest)
        relative_times = {
            'TPU v3-8': tpu_v3_8_time / fastest_time,
            'TPU v4-8': tpu_v4_8_time / fastest_time,
            'TPU v4-16': tpu_v4_16_time / fastest_time,
            'TPU v4-32': tpu_v4_32_time / fastest_time,
            'TPU v5-8': tpu_v5_8_time / fastest_time,
            'GPU 8xA100': gpu_a100_8_time / fastest_time,
            'GPU 16xA100': gpu_a100_16_time / fastest_time
        }
        
        # Calculate total cost = hourly cost * relative time * training hours
        relative_costs = {
            'TPU v3-8': tpu_v3_8_cost * relative_times['TPU v3-8'] * training_hours,
            'TPU v4-8': tpu_v4_8_cost * relative_times['TPU v4-8'] * training_hours,
            'TPU v4-16': tpu_v4_16_cost * relative_times['TPU v4-16'] * training_hours,
            'TPU v4-32': tpu_v4_32_cost * relative_times['TPU v4-32'] * training_hours,
            'TPU v5-8': tpu_v5_8_cost * relative_times['TPU v5-8'] * training_hours,
            'GPU 8xA100': gpu_a100_8_cost * relative_times['GPU 8xA100'] * training_hours,
            'GPU 16xA100': gpu_a100_16_cost * relative_times['GPU 16xA100'] * training_hours
        }
        
        # Calculate cost efficiency (FLOPS per dollar)
        cost_efficiency = {
            'TPU v3-8': tpu_v3_8_flops / tpu_v3_8_cost,
            'TPU v4-8': tpu_v4_8_flops / tpu_v4_8_cost,
            'TPU v4-16': tpu_v4_16_flops / tpu_v4_16_cost,
            'TPU v4-32': tpu_v4_32_flops / tpu_v4_32_cost,
            'TPU v5-8': tpu_v5_8_flops / tpu_v5_8_cost,
            'GPU 8xA100': a100_flops / gpu_a100_8_cost,
            'GPU 16xA100': a100_16_flops / gpu_a100_16_cost
        }
        
        # Find the most cost-efficient option
        most_efficient = min(relative_costs.items(), key=lambda x: x[1])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hardware options
        hardware_options = list(relative_costs.keys())
        
        # Plot relative costs
        cost_values = list(relative_costs.values())
        colors = ['#4285F4' if 'TPU' in hw else '#FF9900' for hw in hardware_options]
        
        ax1.bar(hardware_options, cost_values, color=colors)
        ax1.set_title('Estimated Total Training Cost', fontsize=12)
        ax1.set_ylabel('Total Cost ($)', fontsize=10)
        ax1.set_xticklabels(hardware_options, rotation=45, ha='right')
        
        # Add cost labels
        for i, cost in enumerate(cost_values):
            ax1.text(i, cost + 1000, f"${cost:.0f}", ha='center', fontsize=9)
        
        # Plot training time comparison 
        time_values = list(relative_times.values())
        
        ax2.bar(hardware_options, time_values, color=colors)
        ax2.set_title('Relative Training Time (Lower is Better)', fontsize=12)
        ax2.set_ylabel('Relative Time (1.0 = Fastest)', fontsize=10)
        ax2.set_xticklabels(hardware_options, rotation=45, ha='right')
        
        # Add time labels
        for i, time in enumerate(time_values):
            ax2.text(i, time + 0.1, f"{time:.1f}x", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Generate recommendations
        if model_size_b >= 10:
            tpu_advantage = "High - TPUs excel for very large models"
        elif model_size_b >= 1:
            tpu_advantage = "Medium - TPUs may be cost-effective"
        else:
            tpu_advantage = "Low - GPUs may be more practical for smaller models"
        
        # Code example for TPU training
        tpu_code_example = """
# JAX/Flax training with TPUs
import jax
from jax import numpy as jnp
from flax import linen as nn

# Initialize TPU devices
devices = jax.devices()
print(f"Training with {len(devices)} TPU devices")

# Create model
model = create_flax_model(...)

# Initialize model parameters
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
params = model.init(rng, dummy_input)

# Training configuration
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)

# Single training step function
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']
        ).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Parallel training function using pmap
p_train_step = jax.pmap(train_step, axis_name='devices')
"""
        
        return {
            'tpu_vs_gpu_comparison': {
                'relative_costs': relative_costs,
                'relative_training_times': relative_times,
                'cost_efficiency': cost_efficiency,
                'most_cost_efficient': most_efficient[0],
                'total_cost_most_efficient': most_efficient[1]
            },
            'visualization': fig,
            'tpu_advantage_level': tpu_advantage,
            'recommendation': f"For a {model_size_b}B parameter model with batch size {batch_size}, "
                             f"the most cost-efficient option is {most_efficient[0]} "
                             f"with an estimated training cost of ${most_efficient[1]:.2f}",
            'tpu_code_example': tpu_code_example
        }