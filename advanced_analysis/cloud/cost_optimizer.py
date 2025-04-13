"""
Cloud cost optimization module for ML workloads.

This module provides tools for analyzing and optimizing cloud costs for
machine learning workloads, including instance selection, scaling strategies,
and cost forecasting.
"""

import logging
import json
import requests
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import boto3
from datetime import datetime
import markdown
import base64

# For visualization support in notebooks/dashboards
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

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
    
    def to_dict(self):
        """Convert result to dictionary format"""
        return asdict(self)


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
        'n1-standard-8-nvidia-tesla-t4': {'cost': 0.752, 'gpus': 1, 'gpu_type': 'T4', 'ram': 30},
        'n1-standard-8-nvidia-tesla-v100': {'cost': 2.48, 'gpus': 1, 'gpu_type': 'V100', 'ram': 30},
        'a2-highgpu-1g': {'cost': 3.67, 'gpus': 1, 'gpu_type': 'A100', 'ram': 85},
        'a2-highgpu-4g': {'cost': 13.92, 'gpus': 4, 'gpu_type': 'A100', 'ram': 340}
    }
    
    AZURE_PRICING = {
        'Standard_NC6s_v3': {'cost': 3.06, 'gpus': 1, 'gpu_type': 'V100', 'ram': 112},
        'Standard_NC24s_v3': {'cost': 12.24, 'gpus': 4, 'gpu_type': 'V100', 'ram': 448},
        'Standard_ND40rs_v2': {'cost': 26.7, 'gpus': 8, 'gpu_type': 'V100', 'ram': 672},
        'Standard_NC4as_T4_v3': {'cost': 0.526, 'gpus': 1, 'gpu_type': 'T4', 'ram': 28}
    }
    
    def __init__(self, cloud_provider="aws", use_realtime_pricing=False):
        self.cloud_provider = cloud_provider.lower()
        self.use_realtime_pricing = use_realtime_pricing
        self.pricing_data = self._get_pricing_data()
        self.last_price_update = None
        
        if use_realtime_pricing:
            self.refresh_pricing_data()
    
    def _get_pricing_data(self):
        """Get pricing data for the selected cloud provider"""
        if self.cloud_provider == "aws":
            return self.AWS_PRICING
        elif self.cloud_provider == "gcp":
            return self.GCP_PRICING
        elif self.cloud_provider == "azure":
            return self.AZURE_PRICING
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud_provider}")
            
    def refresh_pricing_data(self):
        """Fetch real-time pricing data from cloud provider APIs"""
        if self.cloud_provider == "aws":
            self._refresh_aws_pricing()
        elif self.cloud_provider == "gcp":
            self._refresh_gcp_pricing()
        elif self.cloud_provider == "azure":
            self._refresh_azure_pricing()
        else:
            logger.warning(f"Real-time pricing not implemented for {self.cloud_provider}")
            
        self.last_price_update = datetime.now()
        
    def _refresh_aws_pricing(self):
        """Fetch real-time AWS pricing using boto3"""
        try:
            client = boto3.client('pricing', region_name='us-east-1')
            
            # Get EC2 pricing for specific instance types
            for instance_type in self.AWS_PRICING.keys():
                response = client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                        {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                    ]
                )
                
                # Parse response to extract on-demand pricing
                for item in response['PriceList']:
                    product_data = json.loads(item)
                    
                    # Find on-demand pricing
                    terms = product_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        price_dimensions = list(terms.values())[0].get('priceDimensions', {})
                        if price_dimensions:
                            price_str = list(price_dimensions.values())[0].get('pricePerUnit', {}).get('USD')
                            if price_str:
                                self.AWS_PRICING[instance_type]['cost'] = float(price_str)
                                
            logger.info("Updated AWS pricing successfully")
            
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {str(e)}")
    
    def _refresh_gcp_pricing(self):
        """Fetch real-time GCP pricing using Cloud Billing API"""
        try:
            # Using public pricing API endpoint
            url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process and update GCP pricing
                for sku in data.get('skus', []):
                    # Check if this is a compute SKU for GPUs
                    if 'gpu' in sku.get('description', '').lower():
                        # Extract pricing
                        pricing_info = sku.get('pricingInfo', [{}])[0]
                        price_per_unit = float(pricing_info.get('pricingExpression', {}).get('tieredRates', [{}])[0].get('unitPrice', {}).get('units', 0))
                        
                        # Update pricing for matching instance types
                        for instance_type in self.GCP_PRICING.keys():
                            if instance_type in sku.get('description', ''):
                                self.GCP_PRICING[instance_type]['cost'] = price_per_unit
                
                logger.info("Updated GCP pricing successfully")
            else:
                logger.error(f"Failed to fetch GCP pricing: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to fetch GCP pricing: {str(e)}")
    
    def _refresh_azure_pricing(self):
        """Fetch real-time Azure pricing using Retail Prices API"""
        try:
            # Using Azure Retail Prices API
            url = "https://prices.azure.com/api/retail/prices"
            params = {
                'api-version': '2021-10-01-preview',
                '$filter': "serviceFamily eq 'Compute' and serviceName eq 'Virtual Machines'"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process and update Azure pricing
                for item in data.get('Items', []):
                    # Check if this is a GPU VM SKU
                    if any(vm_type in item.get('armSkuName', '') for vm_type in ['NC', 'ND', 'NV']):
                        sku_name = item.get('armSkuName')
                        
                        # Update pricing for matching instance types
                        for instance_type in self.AZURE_PRICING.keys():
                            if instance_type == sku_name:
                                self.AZURE_PRICING[instance_type]['cost'] = float(item.get('retailPrice', 0.0))
                
                logger.info("Updated Azure pricing successfully")
            else:
                logger.error(f"Failed to fetch Azure pricing: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to fetch Azure pricing: {str(e)}")
    
    def analyze_cost(self, current_instance, utilization_metrics, 
                    model_size_gb=None, batch_size=None, 
                    throughput_requirement=None,
                    priority_metrics=None) -> CloudCostAnalysisResult:
        """
        Analyze cloud costs and recommend optimizations with multi-metric prioritization
        
        Args:
            current_instance: Current instance type
            utilization_metrics: Dictionary with utilization metrics (gpu_util, memory_util, etc.)
            model_size_gb: Size of model in GB
            batch_size: Current batch size
            throughput_requirement: Required throughput (samples/second)
            priority_metrics: Dict with metric priorities {'cost': 0.7, 'performance': 0.2, 'stability': 0.1}
            
        Returns:
            CloudCostAnalysisResult with detailed analysis and recommendations
        """
        # Use real-time pricing if enabled
        if self.use_realtime_pricing:
            self.refresh_pricing_data()
        
        if current_instance not in self.pricing_data:
            raise ValueError(f"Unknown instance type: {current_instance}")
        
        # Default priority metrics if none provided
        if not priority_metrics:
            priority_metrics = {'cost': 1.0}
        else:
            # Normalize priorities to sum to 1.0
            total = sum(priority_metrics.values())
            priority_metrics = {k: v/total for k, v in priority_metrics.items()}
            
        # Get current instance details
        current_instance_details = self.pricing_data[current_instance]
        current_cost = current_instance_details['cost']
        
        # Analyze utilization
        gpu_util = utilization_metrics.get('gpu_util', 0.5)  # Default to 50% if not provided
        memory_util = utilization_metrics.get('memory_util', 0.5)  # Default to 50% if not provided
        
        # Determine if instance is over-provisioned
        over_provisioned = gpu_util < 0.7 and memory_util < 0.7
        under_provisioned = gpu_util > 0.9 or memory_util > 0.9
        
        # Find recommended instance
        recommended_instance = current_instance
        recommendations = []
        alternative_options = []
        
        # Score all instances based on multi-metric prioritization
        instance_scores = {}
        for instance, details in self.pricing_data.items():
            # Skip current instance
            if instance == current_instance:
                continue
                
            # Calculate base scores for each metric (0-1 scale where 1 is best)
            cost_score = 1.0 - (details['cost'] / max(i['cost'] for i in self.pricing_data.values()))
            
            # Performance score based on GPU count and type
            gpu_score = 0.0
            if 'A100' in details.get('gpu_type', ''):
                gpu_score = 1.0
            elif 'V100' in details.get('gpu_type', ''):
                gpu_score = 0.8
            elif 'A10G' in details.get('gpu_type', ''):
                gpu_score = 0.7
            elif 'T4' in details.get('gpu_type', ''):
                gpu_score = 0.5
            
            # Scale by number of GPUs
            gpu_score *= min(details.get('gpus', 1) / 8, 1.0)
            
            # Memory score
            memory_score = details.get('ram', 0) / max(i.get('ram', 1) for i in self.pricing_data.values())
            
            # Check if instance meets memory requirements
            if model_size_gb and details.get('ram', 0) < model_size_gb:
                continue  # Instance doesn't have enough RAM
            
            # Apply priorities
            total_score = 0
            if 'cost' in priority_metrics:
                total_score += cost_score * priority_metrics['cost']
            if 'performance' in priority_metrics:
                total_score += gpu_score * priority_metrics['performance']
            if 'memory' in priority_metrics:
                total_score += memory_score * priority_metrics['memory']
                
            # Store score
            instance_scores[instance] = {
                'total_score': total_score,
                'cost_score': cost_score,
                'performance_score': gpu_score,
                'memory_score': memory_score,
                'details': details
            }
        
        # Filter and sort alternatives based on instance scenario
        if over_provisioned:
            # Filter for cheaper instances
            cheaper_instances = {k: v for k, v in instance_scores.items() 
                               if self.pricing_data[k]['cost'] < current_cost}
            
            # Sort by score
            sorted_instances = sorted(cheaper_instances.items(), 
                                     key=lambda x: x[1]['total_score'], 
                                     reverse=True)
            
            for instance, score_data in sorted_instances:
                details = self.pricing_data[instance]
                
                alternative_options.append({
                    'instance': instance,
                    'cost': details['cost'],
                    'savings': current_cost - details['cost'],
                    'savings_percentage': ((current_cost - details['cost']) / current_cost) * 100,
                    'score': score_data['total_score'],
                    'gpu_type': details.get('gpu_type', 'N/A'),
                    'gpus': details.get('gpus', 0),
                    'ram': details.get('ram', 0),
                })
                
            # Select recommended instance
            if alternative_options:
                recommended_instance = alternative_options[0]['instance']
                recommendations.append({
                    'type': 'instance_downsizing',
                    'description': f"Current instance is over-provisioned (GPU: {gpu_util*100:.1f}%, Memory: {memory_util*100:.1f}%)",
                    'recommendation': f"Switch to {recommended_instance} to save ${alternative_options[0]['savings']:.2f}/hour ({alternative_options[0]['savings_percentage']:.1f}%)"
                })
                
        elif under_provisioned:
            # Filter for more powerful instances
            better_instances = {k: v for k, v in instance_scores.items() 
                              if self.pricing_data[k]['cost'] > current_cost}
            
            # Sort by score
            sorted_instances = sorted(better_instances.items(), 
                                     key=lambda x: x[1]['total_score'], 
                                     reverse=True)
            
            for instance, score_data in sorted_instances:
                details = self.pricing_data[instance]
                
                alternative_options.append({
                    'instance': instance,
                    'cost': details['cost'],
                    'additional_cost': details['cost'] - current_cost,
                    'cost_increase_percentage': ((details['cost'] - current_cost) / current_cost) * 100,
                    'score': score_data['total_score'],
                    'gpu_type': details.get('gpu_type', 'N/A'),
                    'gpus': details.get('gpus', 0),
                    'ram': details.get('ram', 0),
                })
                
            # Select recommended instance
            if alternative_options:
                recommended_instance = alternative_options[0]['instance']
                recommendations.append({
                    'type': 'instance_upsizing',
                    'description': f"Current instance is under-provisioned (GPU: {gpu_util*100:.1f}%, Memory: {memory_util*100:.1f}%)",
                    'recommendation': f"Switch to {recommended_instance} for better performance at ${alternative_options[0]['additional_cost']:.2f}/hour more"
                })
        
        # Add spot/reserved instance recommendations
        if self.cloud_provider == "aws":
            spot_savings = current_cost * 0.7  # Approximate 70% savings
            recommendations.append({
                'type': 'pricing_model',
                'description': "Consider using Spot Instances for interruptible workloads",
                'recommendation': f"Using Spot Instances could save approximately ${spot_savings:.2f}/hour (70%)",
                'code_example': "# AWS CLI example\naws ec2 request-spot-instances --instance-count 1 --type one-time --launch-specification file://specification.json"
            })
            
            reserved_savings = current_cost * 0.4  # Approximate 40% savings with 1-year commitment
            recommendations.append({
                'type': 'pricing_model',
                'description': "Consider Reserved Instances for long-term workloads",
                'recommendation': f"1-year Reserved Instances could save approximately ${reserved_savings:.2f}/hour (40%)",
                'code_example': "# Use AWS Management Console to purchase Reserved Instances"
            })
            
        # Create result
        recommended_instance_details = self.pricing_data[recommended_instance]
        recommended_cost = recommended_instance_details['cost']
        
        savings_percentage = ((current_cost - recommended_cost) / current_cost) * 100 if current_cost > recommended_cost else 0
        
        result = CloudCostAnalysisResult(
            current_instance=current_instance,
            current_cost_per_hour=current_cost,
            recommended_instance=recommended_instance,
            recommended_cost_per_hour=recommended_cost,
            potential_savings_percentage=savings_percentage,
            utilization_metrics=utilization_metrics,
            recommendations=recommendations,
            alternative_options=alternative_options
        )
        
        return result
    
    def forecast_cost(self, instance_type, usage_hours_per_day, days_per_month=30):
        """
        Forecast monthly cost for the given instance and usage pattern
        
        Args:
            instance_type: Instance type
            usage_hours_per_day: Hours of usage per day
            days_per_month: Days of usage per month
            
        Returns:
            Dictionary with cost forecast
        """
        if instance_type not in self.pricing_data:
            raise ValueError(f"Unknown instance type: {instance_type}")
            
        hourly_cost = self.pricing_data[instance_type]['cost']
        monthly_hours = usage_hours_per_day * days_per_month
        monthly_cost = hourly_cost * monthly_hours
        
        # Calculate on-demand vs. reserved costs
        reserved_1yr_cost = monthly_cost * 0.6  # Approximate 40% savings
        reserved_3yr_cost = monthly_cost * 0.4  # Approximate 60% savings
        
        return {
            'instance_type': instance_type,
            'hourly_cost': hourly_cost,
            'usage_hours_per_month': monthly_hours,
            'monthly_on_demand_cost': monthly_cost,
            'monthly_reserved_1yr_cost': reserved_1yr_cost,
            'monthly_reserved_3yr_cost': reserved_3yr_cost,
            'annual_on_demand_cost': monthly_cost * 12,
            'annual_reserved_1yr_cost': reserved_1yr_cost * 12,
            'annual_reserved_3yr_cost': reserved_3yr_cost * 12,
            'recommendation': f"For consistent usage of {usage_hours_per_day} hours/day, Reserved Instances offer significant savings"
        }
    
    def get_visualization_data(self, analysis_result: CloudCostAnalysisResult) -> Tuple[Dict, Dict]:
        """
        Get data formatted for visualization of analysis results
        
        Args:
            analysis_result: The CloudCostAnalysisResult to visualize
            
        Returns:
            Tuple of (instance_comparison_data, cost_breakdown_data)
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Install pandas and matplotlib.")
            return {}, {}
            
        # Format data for instance comparison chart
        instance_data = {
            'instances': [analysis_result.current_instance] + 
                        [alt['instance'] for alt in analysis_result.alternative_options[:5]],
            'costs': [analysis_result.current_cost_per_hour] + 
                    [alt['cost'] for alt in analysis_result.alternative_options[:5]],
            'gpus': [self.pricing_data[analysis_result.current_instance].get('gpus', 0)] + 
                   [self.pricing_data[alt['instance']].get('gpus', 0) 
                    for alt in analysis_result.alternative_options[:5]],
            'memory': [self.pricing_data[analysis_result.current_instance].get('ram', 0)] + 
                     [self.pricing_data[alt['instance']].get('ram', 0) 
                      for alt in analysis_result.alternative_options[:5]]
        }
        
        # Format data for cost breakdown chart
        current = analysis_result.current_cost_per_hour
        recommended = analysis_result.recommended_cost_per_hour
        
        cost_breakdown = {
            'labels': ['Current Cost', 'Recommended Cost', 'Potential Savings'],
            'values': [current, recommended, current - recommended if current > recommended else 0]
        }
        
        return instance_data, cost_breakdown
    
    def plot_instance_comparison(self, analysis_result: CloudCostAnalysisResult, 
                               save_path=None, show=True):
        """
        Generate a bar chart comparing instance options
        
        Args:
            analysis_result: The CloudCostAnalysisResult to visualize
            save_path: Path to save the figure (optional)
            show: Whether to display the figure (default: True)
            
        Returns:
            matplotlib figure or None if visualization is not available
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Install pandas and matplotlib.")
            return None
            
        instance_data, _ = self.get_visualization_data(analysis_result)
        
        # Create dataframe
        df = pd.DataFrame({
            'Instance': instance_data['instances'],
            'Cost per Hour ($)': instance_data['costs']
        })
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['Instance'], df['Cost per Hour ($)'])
        
        # Highlight current and recommended instances
        for i, instance in enumerate(df['Instance']):
            if instance == analysis_result.current_instance:
                bars[i].set_color('orange')
            elif instance == analysis_result.recommended_instance:
                bars[i].set_color('green')
        
        # Add labels and title
        ax.set_xlabel('Instance Type')
        ax.set_ylabel('Cost per Hour ($)')
        ax.set_title('Cloud Instance Cost Comparison')
        
        # Add annotations
        for i, value in enumerate(df['Cost per Hour ($)']):
            ax.text(i, value + 0.1, f'${value:.2f}', ha='center')
            
        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return fig
    
    def export_to_markdown(self, analysis_result: CloudCostAnalysisResult) -> str:
        """
        Export analysis results to Markdown format
        
        Args:
            analysis_result: The CloudCostAnalysisResult to export
            
        Returns:
            String containing markdown content
        """
        md = []
        
        # Add header
        md.append("# Cloud Cost Optimization Analysis")
        md.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md.append("")
        
        # Current vs Recommended Summary
        md.append("## Summary")
        md.append(f"**Current Instance:** {analysis_result.current_instance}")
        md.append(f"**Current Cost:** ${analysis_result.current_cost_per_hour:.2f}/hour")
        md.append(f"**Recommended Instance:** {analysis_result.recommended_instance}")
        md.append(f"**Recommended Cost:** ${analysis_result.recommended_cost_per_hour:.2f}/hour")
        
        if analysis_result.potential_savings_percentage > 0:
            md.append(f"**Potential Savings:** {analysis_result.potential_savings_percentage:.2f}%")
        md.append("")
        
        # Utilization metrics
        md.append("## Utilization Metrics")
        for metric, value in analysis_result.utilization_metrics.items():
            md.append(f"- **{metric}:** {value*100:.1f}%")
        md.append("")
        
        # Recommendations
        md.append("## Recommendations")
        for rec in analysis_result.recommendations:
            md.append(f"### {rec['type'].replace('_', ' ').title()}")
            md.append(f"{rec['description']}")
            md.append(f"**Recommendation:** {rec['recommendation']}")
            
            if 'code_example' in rec:
                md.append("```")
                md.append(rec['code_example'])
                md.append("```")
            md.append("")
        
        # Alternative options
        md.append("## Alternative Options")
        md.append("| Instance | Cost/Hour | Savings | Savings % | GPUs | GPU Type | RAM |")
        md.append("|----------|-----------|---------|-----------|------|----------|-----|")
        
        for alt in analysis_result.alternative_options[:10]:  # Show top 10 alternatives
            instance = alt['instance']
            details = self.pricing_data[instance]
            
            savings = alt.get('savings', alt.get('additional_cost', 0))
            savings_pct = alt.get('savings_percentage', alt.get('cost_increase_percentage', 0))
            savings_str = f"${savings:.2f}" if 'savings' in alt else f"-${abs(savings):.2f}"
            savings_pct_str = f"{savings_pct:.1f}%" if 'savings_percentage' in alt else f"-{abs(savings_pct):.1f}%"
            
            md.append(f"| {instance} | ${alt['cost']:.2f} | {savings_str} | {savings_pct_str} | "
                    f"{details.get('gpus', 'N/A')} | {details.get('gpu_type', 'N/A')} | "
                    f"{details.get('ram', 'N/A')} GB |")
        
        return "\n".join(md)
    
    def export_to_html(self, analysis_result: CloudCostAnalysisResult) -> str:
        """
        Export analysis results to HTML format
        
        Args:
            analysis_result: The CloudCostAnalysisResult to export
            
        Returns:
            String containing HTML content
        """
        # Convert markdown to HTML
        md_content = self.export_to_markdown(analysis_result)
        html = markdown.markdown(md_content, extensions=['tables'])
        
        # Add CSS styling
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cloud Cost Optimization Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .savings {{ color: green; }}
                .cost-increase {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        return html
    
    def export_to_json(self, analysis_result: CloudCostAnalysisResult) -> str:
        """
        Export analysis results to JSON format
        
        Args:
            analysis_result: The CloudCostAnalysisResult to export
            
        Returns:
            String containing JSON content
        """
        # Convert to dictionary
        result_dict = analysis_result.to_dict()
        
        # Add metadata
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "cloud_provider": self.cloud_provider,
                "real_time_pricing": self.use_realtime_pricing,
                "price_update_time": self.last_price_update.isoformat() if self.last_price_update else None
            },
            "analysis": result_dict
        }
        
        # Convert to JSON string
        return json.dumps(export_data, indent=2)
    
    def estimate_costs(self, perf_results, requests_per_month=1000000, providers=None):
        """Estimate cloud deployment costs across different providers."""
        if providers is None:
            providers = ["aws", "gcp", "azure"]
            
        # Get baseline metrics with defaults
        avg_latency = max(0.001, perf_results.get("latency_stats", {}).get("mean", 0.001))  # Minimum 1ms
        peak_memory = max(128 * 1024 * 1024, perf_results.get("memory_stats", {}).get("max", 128 * 1024 * 1024))  # Minimum 128MB
        avg_memory = max(64 * 1024 * 1024, perf_results.get("memory_stats", {}).get("mean", 64 * 1024 * 1024))  # Minimum 64MB
        duration_s = avg_latency
        
        # Cost estimates per provider
        estimates = {}
        
        # AWS cost estimation
        if "aws" in providers:
            # AWS Lambda pricing (per GB-second)
            lambda_cost = 0.0000166667
            memory_gb = max(0.128, peak_memory / (1024 * 1024 * 1024))  # Minimum 128MB
            monthly_compute_cost = requests_per_month * memory_gb * duration_s * lambda_cost
            
            estimates["aws"] = {
                "monthly_cost": monthly_compute_cost,
                "service_type": "Lambda",
                "memory_config": f"{memory_gb:.2f}GB",
                "notes": []
            }
            
            # Add recommendations
            if memory_gb < 0.128:
                estimates["aws"]["notes"].append("Consider increasing memory allocation for better performance")
            elif memory_gb > 3:
                estimates["aws"]["notes"].append("Consider using EC2 for high memory workloads")
        
        # GCP cost estimation
        if "gcp" in providers:
            # Cloud Functions pricing (per million invocations)
            base_invocation_cost = 0.40
            memory_gb = max(0.128, peak_memory / (1024 * 1024 * 1024))
            cpu_cost = 0.00001 * duration_s * requests_per_month  # $0.00001 per vCPU-second
            memory_cost = 0.0000025 * memory_gb * duration_s * requests_per_month  # $0.0000025 per GB-second
            
            monthly_cost = (base_invocation_cost * (requests_per_month / 1000000)) + cpu_cost + memory_cost
            
            estimates["gcp"] = {
                "monthly_cost": monthly_cost,
                "service_type": "Cloud Functions",
                "memory_config": f"{memory_gb:.2f}GB",
                "notes": []
            }
            
            if avg_latency > 1:
                estimates["gcp"]["notes"].append("Consider Cloud Run for longer-running workloads")
        
        # Azure cost estimation
        if "azure" in providers:
            # Azure Functions pricing
            base_cost = 0.20  # per million executions
            memory_gb = max(0.128, peak_memory / (1024 * 1024 * 1024))
            gb_second_cost = 0.000016 * memory_gb * duration_s * requests_per_month
            
            monthly_cost = (base_cost * (requests_per_month / 1000000)) + gb_second_cost
            
            estimates["azure"] = {
                "monthly_cost": monthly_cost,
                "service_type": "Functions",
                "memory_config": f"{memory_gb:.2f}GB",
                "notes": []
            }
            
            if memory_gb > 1.5:
                estimates["azure"]["notes"].append("Consider Container Apps for high memory requirements")
        
        # Find optimal provider
        if estimates:
            sorted_providers = sorted(estimates.items(), key=lambda x: x[1]["monthly_cost"])
            optimal_provider = sorted_providers[0][0]
            optimal_cost = max(0.0001, estimates[optimal_provider]["monthly_cost"])  # Avoid division by zero
            
            cost_difference = {
                p: ((estimates[p]["monthly_cost"] / optimal_cost) - 1) * 100
                for p in estimates if p != optimal_provider
            }
            
            recommendations = {
                "optimal_provider": optimal_provider,
                "cost_savings_percentage": cost_difference,
                "general_notes": []
            }
            
            # Add general recommendations
            if avg_latency < 0.1:
                recommendations["general_notes"].append("Serverless functions recommended for low latency")
            elif avg_latency > 5:
                recommendations["general_notes"].append("Consider container-based solutions for long-running tasks")
                
            if requests_per_month > 10000000:
                recommendations["general_notes"].append("High volume - consider reserved instances or committed use discounts")
        else:
            recommendations = {
                "optimal_provider": None,
                "cost_savings_percentage": {},
                "general_notes": ["No valid cost estimates available"]
            }
        
        return {
            "estimates": estimates,
            "recommendations": recommendations,
            "input_params": {
                "requests_per_month": requests_per_month,
                "avg_latency": avg_latency,
                "peak_memory_gb": memory_gb
            }
        }
