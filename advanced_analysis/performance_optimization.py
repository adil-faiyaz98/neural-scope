"""
Neural-Scope ML Advisor: Advanced AI/ML Performance Optimization System

A comprehensive, industry-leading solution for ML model analysis, profiling, and optimization
that bridges the gap between data science and ML engineering. This system provides
actionable, model-specific recommendations based on deep inspection of model architecture,
execution patterns, and hardware utilization.

Key features:
1. Deep Model Inspection
   - Layer-by-layer profiling with operation-specific bottleneck detection
   - Computation vs. memory bottleneck classification
   - Framework-specific optimization opportunities (PyTorch/TensorFlow)
   - Memory access pattern analysis for cache efficiency

2. Execution Profiling
   - Kernel launch statistics and GPU utilization metrics
   - Thread-level CPU analysis with GIL contention detection
   - Data pipeline efficiency evaluation
   - Distributed training communication overhead measurement

3. Intelligent Optimization Recommendations
   - Architecture-specific suggestions (attention mechanisms, embedding tables)
   - Automatic quantization opportunity detection with precision impact analysis
   - JIT/XLA/TorchScript compilation benefits estimation
   - Custom CUDA kernel suggestions for hotspot operations

4. Data Quality Assessment
   - Intersectional bias detection across protected attributes
   - Representation disparity measurement
   - Distribution shift detection between train/test/production
   - Noise and outlier impact quantification

5. Hardware Matching
   - Specific GPU recommendations based on memory/compute bottleneck analysis
   - Cost-performance optimization for cloud providers
   - TPU/IPU/specialized hardware opportunity identification
   - Multi-node scaling efficiency prediction
"""

import os
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
import matplotlib.pyplot as plt
import psycopg2
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Framework-specific imports with graceful degradation
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.python.eager import profiler as tf_profiler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRNESS_METRICS_AVAILABLE = True
except ImportError:
    FAIRNESS_METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NeuralScope')

@dataclass
class LayerProfile:
    """Detailed profile information for a single model layer"""
    name: str
    layer_type: str
    parameters: int
    execution_time: float
    memory_usage: float
    flops: Optional[int] = None
    io_bound: bool = False
    compute_bound: bool = False
    memory_bound: bool = False
    bottleneck_score: float = 0.0
    optimization_candidates: List[str] = None
    
    def __post_init__(self):
        if self.optimization_candidates is None:
            self.optimization_candidates = []

@dataclass
class ModelProfile:
    """Comprehensive profile of an ML model's performance characteristics"""
    model_name: str
    framework: str
    total_parameters: int
    batch_size: int
    input_shape: tuple
    dtype: str
    execution_time: float
    peak_memory: float
    throughput: float
    layers: List[LayerProfile]
    bottleneck_type: str = "unknown"
    data_pipeline_efficiency: float = 1.0
    gpu_utilization: float = 0.0
    parallelizable: bool = False
    
    @property
    def total_execution_time(self) -> float:
        """Sum of all layer execution times"""
        return sum(layer.execution_time for layer in self.layers)
    
    @property
    def bottleneck_layers(self) -> List[LayerProfile]:
        """Return layers sorted by bottleneck score (descending)"""
        return sorted(self.layers, key=lambda x: x.bottleneck_score, reverse=True)

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment results"""
    dataset_name: str
    total_samples: int
    missing_values_pct: Dict[str, float]
    categorical_imbalance: Dict[str, float]
    numerical_outlier_pct: Dict[str, float]
    distribution_skew: Dict[str, float]
    intersectional_bias_score: float = 0.0
    class_imbalance_score: float = 0.0
    label_noise_estimate: float = 0.0
    fairness_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fairness_metrics is None:
            self.fairness_metrics = {}
    
    @property
    def quality_score(self) -> float:
        """Overall data quality score from 0-1"""
        scores = [
            1.0 - max(self.missing_values_pct.values()) if self.missing_values_pct else 1.0,
            1.0 - self.class_imbalance_score,
            1.0 - self.label_noise_estimate,
            1.0 - self.intersectional_bias_score,
        ]
        return sum(scores) / len(scores)

class HardwareRecommender:
    """Recommends optimal hardware based on model profile and constraints"""
    
    # Hardware database with capabilities and costs
    GPU_DATABASE = {
        'T4': {
            'memory': 16, 'compute': 8.1, 'fp16_speedup': 2.0, 'int8_speedup': 4.0, 
            'cost_per_hour': 0.35, 'provider': 'GCP'
        },
        'V100': {
            'memory': 32, 'compute': 15.7, 'fp16_speedup': 8.0, 'int8_speedup': 2.0, 
            'cost_per_hour': 0.90, 'provider': 'AWS'
        },
        'A100': {
            'memory': 80, 'compute': 19.5, 'fp16_speedup': 20.0, 'int8_speedup': 40.0, 
            'cost_per_hour': 2.50, 'provider': 'GCP'
        },
        'H100': {
            'memory': 80, 'compute': 51.0, 'fp16_speedup': 30.0, 'int8_speedup': 60.0, 
            'cost_per_hour': 5.50, 'provider': 'Azure'
        },
        'RTX 4090': {
            'memory': 24, 'compute': 18.0, 'fp16_speedup': 16.0, 'int8_speedup': 32.0, 
            'cost_per_hour': 0.0, 'provider': 'On-prem'
        }
    }
    
    def __init__(self, budget_constraint: Optional[float] = None):
        self.budget_constraint = budget_constraint
    
    def recommend_for_profile(self, profile: ModelProfile) -> Dict[str, Any]:
        """Recommend hardware based on detailed model profile"""
        is_memory_bound = profile.bottleneck_type == "memory"
        is_compute_bound = profile.bottleneck_type == "compute"
        
        required_memory = profile.peak_memory * 1.5  # Add 50% buffer
        
        # Filter GPUs that meet memory requirements
        candidates = {gpu: specs for gpu, specs in self.GPU_DATABASE.items() 
                     if specs['memory'] >= required_memory}
        
        if not candidates:
            # Fall back to multi-GPU configurations if single GPU can't fit model
            multi_gpu_options = []
            for gpu, specs in self.GPU_DATABASE.items():
                num_gpus = max(2, int(np.ceil(required_memory / specs['memory'])))
                if num_gpus < 8:  # Limit to reasonable configurations
                    multi_gpu_options.append({
                        'gpu': gpu,
                        'count': num_gpus,
                        'total_memory': specs['memory'] * num_gpus,
                        'total_compute': specs['compute'] * num_gpus * 0.9,  # 10% overhead
                        'cost': specs['cost_per_hour'] * num_gpus,
                        'efficiency': (specs['compute'] / specs['cost_per_hour']) if specs['cost_per_hour'] > 0 else float('inf')
                    })
                    
            if not multi_gpu_options:
                return {"recommendation": "Model too large for standard GPUs - consider model sharding or optimization"}
            
            # Sort by efficiency or cost constraint
            if self.budget_constraint:
                valid_options = [opt for opt in multi_gpu_options if opt['cost'] <= self.budget_constraint]
                if not valid_options:
                    return {"recommendation": f"No configuration meets budget constraint of ${self.budget_constraint}/hour"}
                best_option = max(valid_options, key=lambda x: x['total_compute'])
            else:
                best_option = max(multi_gpu_options, key=lambda x: x['efficiency'])
                
            return {
                "recommendation": f"{best_option['count']}x {best_option['gpu']}",
                "configuration": "distributed" if best_option['count'] > 1 else "single",
                "reason": "Memory requirements exceed single GPU capacity",
                "estimated_cost": f"${best_option['cost']:.2f}/hour on {self.GPU_DATABASE[best_option['gpu']]['provider']}",
                "optimization_priority": "Model sharding and memory efficiency"
            }
        
        # For memory-bound workloads, prioritize GPUs with more memory
        if is_memory_bound:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
                
            best_gpu = max(candidates.items(), key=lambda x: x[1]['memory'])
            reason = "Memory-bound workload needs maximum GPU memory"
            optimization = "Memory optimization (quantization, attention optimizations)"
        
        # For compute-bound workloads, prioritize computational capacity
        elif is_compute_bound:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
                    
            best_gpu = max(candidates.items(), key=lambda x: x[1]['compute'])
            reason = "Compute-bound workload needs maximum computational power"
            optimization = "Algorithmic optimizations, kernel fusion, TensorRT"
        
        # For balanced or unknown workloads, find best price/performance ratio
        else:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
            
            # Compute efficiency as TFLOPS/dollar
            best_gpu = max(candidates.items(), 
                           key=lambda x: (x[1]['compute']/x[1]['cost_per_hour']) if x[1]['cost_per_hour'] > 0 else float('inf'))
            reason = "Balanced workload benefits from good price-performance ratio"
            optimization = "Mixed precision training, pipeline parallelism"
        
        gpu_name, specs = best_gpu
        
        mixed_precision_speedup = None
        if profile.dtype == 'float32':
            mixed_precision_speedup = specs['fp16_speedup']
        
        return {
            "recommendation": gpu_name,
            "memory": f"{specs['memory']} GB",
            "compute": f"{specs['compute']} TFLOPS",
            "provider": specs['provider'],
            "estimated_cost": f"${specs['cost_per_hour']:.2f}/hour" if specs['cost_per_hour'] > 0 else "On-premises",
            "reason": reason,
            "optimization_priority": optimization,
            "mixed_precision_benefit": f"{mixed_precision_speedup}x speedup possible" if mixed_precision_speedup else None
        }

class DataQualityAnalyzer:
    """Advanced data quality assessment for ML datasets"""
    
    def __init__(self, sensitive_attributes: Optional[List[str]] = None):
        """
        Initialize with a list of columns to treat as sensitive/protected attributes
        for fairness and bias analysis
        """
        self.sensitive_attributes = sensitive_attributes or []
    
    def analyze_dataset(self, data: pd.DataFrame, label_column: Optional[str] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment on a pandas DataFrame
        """
        dataset_name = getattr(data, 'name', 'unknown_dataset')
        total_samples = len(data)
        
        # Compute missing value statistics by column
        missing_pct = {}
        for col in data.columns:
            missing_pct[col] = data[col].isna().mean() * 100
        
        # Analyze categorical columns for imbalance
        categorical_imbalance = {}
        for col in data.select_dtypes(include=['object', 'category']).columns:
            value_counts = data[col].value_counts(normalize=True)
            # Gini impurity as measure of imbalance
            gini = 1 - np.sum(value_counts ** 2)
            categorical_imbalance[col] = gini
        
        # Detect outliers in numerical columns using IQR method
        numerical_outliers = {}
        for col in data.select_dtypes(include=np.number).columns:
            if col == label_column:
                continue
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))
            numerical_outliers[col] = outlier_mask.mean() * 100
        
        # Calculate distribution skewness
        skewness = {}
        for col in data.select_dtypes(include=np.number).columns:
            skewness[col] = abs(data[col].skew())
        
        # Class imbalance for classification tasks
        class_imbalance = 0
        if label_column and label_column in data.columns:
            # Check if it's likely a classification task
            if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                # Normalized entropy as imbalance metric (0=perfect balance, 1=complete imbalance)
                value_counts = data[label_column].value_counts(normalize=True)
                if len(value_counts) > 1:
                    entropy = -np.sum(value_counts * np.log(value_counts))
                    max_entropy = np.log(len(value_counts))
                    class_imbalance = 1 - (entropy / max_entropy)
        
        # Estimate label noise (simplified approach)
        label_noise = 0.0
        if label_column and len(data) > 100:
            try:
                # For classification: use label consistency in feature neighborhoods as proxy for noise
                if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Sample if dataset is very large to speed up computation
                    data_sample = data.sample(min(10000, len(data))) if len(data) > 10000 else data
                    
                    # Select only numeric features
                    numeric_cols = data_sample.select_dtypes(include=np.number).columns
                    numeric_cols = [c for c in numeric_cols if c != label_column]
                    
                    if len(numeric_cols) > 0:
                        X = data_sample[numeric_cols].fillna(0)
                        y = data_sample[label_column]
                        
                        # Normalize features
                        X = (X - X.mean()) / X.std()
                        
                        # Find 5 nearest neighbors for each point
                        nn = NearestNeighbors(n_neighbors=6)  # including the point itself
                        nn.fit(X)
                        _, indices = nn.kneighbors(X)
                        
                        # Calculate label consistency within neighborhoods
                        inconsistent_count = 0
                        for i, neighbors in enumerate(indices):
                            center_label = y.iloc[i]
                            neighbor_labels = [y.iloc[j] for j in neighbors[1:]]  # exclude self
                            inconsistent_count += (1 - neighbor_labels.count(center_label) / len(neighbor_labels))
                        
                        label_noise = inconsistent_count / len(X)
                        label_noise = min(1.0, max(0.0, label_noise))  # clamp to [0,1]
            except Exception as e:
                logger.warning(f"Label noise estimation failed: {e}")
        
        # Intersectional bias analysis
        intersectional_bias = 0.0
        fairness_metrics = {}
        
        if label_column and self.sensitive_attributes and FAIRNESS_METRICS_AVAILABLE:
            try:
                if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                    label_values = data[label_column].unique()
                    if len(label_values) == 2:  # Binary classification case
                        y_true = data[label_column]
                        
                        # Analyze each sensitive attribute
                        dpd_scores = []
                        eod_scores = []
                        for attr in self.sensitive_attributes:
                            if attr in data.columns:
                                # Remove rows with missing values in the sensitive attribute
                                mask = ~data[attr].isna()
                                if mask.sum() > 100:  # Need sufficient data
                                    dpd = demographic_parity_difference(
                                        y_true=y_true[mask],
                                        sensitive_features=data[attr][mask]
                                    )
                                    eod = equalized_odds_difference(
                                        y_true=y_true[mask],
                                        y_pred=y_true[mask],  # Using true labels as proxy
                                        sensitive_features=data[attr][mask]
                                    )
                                    dpd_scores.append(dpd)
                                    eod_scores.append(eod)
                                    fairness_metrics[f"demographic_parity_{attr}"] = dpd
                                    fairness_metrics[f"equalized_odds_{attr}"] = eod
                        
                        # Average the scores
                        if dpd_scores:
                            intersectional_bias = (sum(dpd_scores) / len(dpd_scores) + 
                                                sum(eod_scores) / len(eod_scores)) / 2
            except Exception as e:
                logger.warning(f"Fairness metrics calculation failed: {e}")
        
        return DataQualityMetrics(
            dataset_name=dataset_name,
            total_samples=total_samples,
            missing_values_pct=missing_pct,
            categorical_imbalance=categorical_imbalance,
            numerical_outlier_pct=numerical_outliers,
            distribution_skew=skewness,
            intersectional_bias_score=intersectional_bias,
            class_imbalance_score=class_imbalance,
            label_noise_estimate=label_noise,
            fairness_metrics=fairness_metrics
        )
    
    def generate_recommendations(self, metrics: DataQualityMetrics) -> List[Dict[str, Any]]:
        """Generate specific data quality improvement recommendations"""
        recommendations = []
        
        # High missing value columns
        high_missing = [(col, pct) for col, pct in metrics.missing_values_pct.items() if pct > 5]
        if high_missing:
            for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "High Missing Values",
                    "column": col,
                    "severity": "High" if pct > 20 else "Medium",
                    "details": f"{pct:.1f}% missing values",
                    "suggestions": [
                        f"Impute missing values using {self._suggest_imputation_method(col)}",
                        "Consider if this feature should be dropped if not informative",
                        "Investigate data collection process to reduce missingness"
                    ]
                })
        
        # Highly skewed numerical features
        high_skew = [(col, skew) for col, skew in metrics.distribution_skew.items() if skew > 1.0]
        if high_skew:
            for col, skew in sorted(high_skew, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "Skewed Distribution",
                    "column": col,
                    "severity": "Medium",
                    "details": f"Skewness: {skew:.2f}",
                    "suggestions": [
                        f"Apply log transformation: np.log1p({col})",
                        f"Apply Box-Cox transformation on {col}",
                        f"Standardize or normalize {col} to improve training stability"
                    ]
                })
        
        # High outlier percentage
        high_outliers = [(col, pct) for col, pct in metrics.numerical_outlier_pct.items() if pct > 1.0]
        if high_outliers:
            for col, pct in sorted(high_outliers, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "Outliers Detected",
                    "column": col,
                    "severity": "Medium" if pct > 5 else "Low",
                    "details": f"{pct:.1f}% outliers detected",
                    "suggestions": [
                        f"Cap outliers using winsorization: df['{col}'] = np.clip(df['{col}'], lower, upper)",
                        "Investigate if outliers represent valid but rare cases",
                        f"Create binary flag feature: is_{col}_outlier to preserve information"
                    ]
                })
        
        # Class imbalance
        if metrics.class_imbalance_score > 0.3:
            recommendations.append({
                "issue": "Class Imbalance",
                "column": "target/label",
                "severity": "High" if metrics.class_imbalance_score > 0.6 else "Medium",
                "details": f"Imbalance score: {metrics.class_imbalance_score:.2f}",
                "suggestions": [
                    "Use class weights in model training",
                    "Apply SMOTE or other resampling techniques",
                    "Use focal loss or other imbalance-aware loss functions",
                    "Evaluate models with balanced accuracy or F1-score instead of accuracy"
                ]
            })
        
        # Possible label noise
        if metrics.label_noise_estimate > 0.1:
            recommendations.append({
                "issue": "Potential Label Noise",
                "column": "target/label",
                "severity": "High" if metrics.label_noise_estimate > 0.3 else "Medium",
                "details": f"Estimated noise: {metrics.label_noise_estimate:.2f}",
                "suggestions": [
                    "Manually review a sample of potentially mislabeled instances",
                    "Consider robust loss functions (e.g., MAE instead of MSE)",
                    "Implement label smoothing in classification tasks",
                    "Try data cleaning techniques like Confident Learning (cleanlab library)"
                ]
            })
        
        # Bias/fairness issues
        if metrics.intersectional_bias_score > 0.1 and metrics.fairness_metrics:
            highest_bias = max(metrics.fairness_metrics.items(), key=lambda x: x[1])
            metric_name, value = highest_bias
            attribute = metric_name.split('_')[-1]
            recommendations.append({
                "issue": "Potential Bias Detected",
                "column": attribute,
                "severity": "High" if metrics.intersectional_bias_score > 0.2 else "Medium",
                "details": f"Bias score: {metrics.intersectional_bias_score:.2f}",
                "suggestions": [
                    "Apply fairness constraints during model training",
                    f"Analyze and balance representation across {attribute} groups",
                    "Use post-processing methods to equalize predictions across groups",
                    "Consider collecting additional data from underrepresented groups"
                ]
            })
        
        return recommendations
    
    def _suggest_imputation_method(self, column_name: str) -> str:
        """Suggest appropriate imputation method based on column name heuristics"""
        column_lower = column_name.lower()
        
        if any(word in column_lower for word in ['age', 'year', 'income', 'salary', 'price']):
            return "median imputation"
        elif any(word in column_lower for word in ['count', 'number', 'quantity', 'amt']):
            return "mean or median imputation"
        elif any(word in column_lower for word in ['category', 'type', 'class', 'group']):
            return "mode imputation"
        elif any(word in column_lower for word in ['date', 'time']):
            return "forward fill or interpolation"
        else:
            return "KNN or model-based imputation"

class ModelOptimizer:
    """Recommends and implements model-specific optimizations"""
    
    ATTENTION_PATTERNS = [
        (r'MultiHead', 'Multi-head attention layer'),
        (r'SelfAttention', 'Self-attention mechanism'),
        (r'TransformerEncoderLayer', 'Transformer encoder'),
        (r'TransformerDecoderLayer', 'Transformer decoder')
    ]
    
    OPTIMIZATION_TECHNIQUES = {
        'attention': [
            {
                'name': 'FlashAttention',
                'description': 'IO-aware attention implementation with better memory efficiency',
                'speedup': '2-4x for attention operations',
                'memory_savings': '10-20% of total memory',
                'implementation': 'from flash_attn import flash_attn_qkvpacked_func',
                'constraints': 'CUDA only, SM 8.0+ (Ampere or newer)'
            },
            {
                'name': 'Memory-efficient attention',
                'description': 'Reduces memory usage by recomputing activations',
                'speedup': '5-15% end-to-end',
                'memory_savings': '20-30% of attention memory',
                'implementation': 'from xformers.ops import memory_efficient_attention',
                'constraints': 'Requires xformers library'
            }
        ],
        'quantization': [
            {
                'name': 'Dynamic Quantization',
                'description': 'Quantize weights to INT8 dynamically during inference',
                'speedup': '2-4x on CPU inference',
                'memory_savings': '75% for quantized layers',
                'implementation': 'model_int8 = torch.quantization.quantize_dynamic(model)',
                'constraints': 'Primarily for CPU inference'
            },
            {
                'name': 'Weight-only Quantization',
                'description': 'Int8 weights with fp16/32 activations',
                'speedup': '1.5-2x on inference',
                'memory_savings': '50-75% for model weights',
                'implementation': 'model = AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)',
                'constraints': 'Requires transformers >= 4.30.0 and CUDA'
            },
            {
                'name': 'QLoRA fine-tuning',
                'description': 'Quantized weights with trainable LoRA adapters',
                'speedup': 'Enables fine-tuning with ~80% less memory',
                'memory_savings': '70-80% during fine-tuning',
                'implementation': 'from peft import get_peft_model, LoraConfig',
                'constraints': 'Requires PEFT library and bitsandbytes'
            }
        ],
        'compilation': [
            {
                'name': 'TorchScript Compilation',
                'description': 'Compiles model for optimized inference',
                'speedup': '20-50% for inference',
                'memory_savings': 'Minimal',
                'implementation': 'model_ts = torch.jit.script(model)',
                'constraints': 'Model must be compatible with TorchScript tracing'
            },
            {
                'name': 'TensorRT Export',
                'description': 'NVIDIA inference optimization with kernel fusion',
                'speedup': '2-5x for inference',
                'memory_savings': 'Varies by model',
                'implementation': 'import torch_tensorrt; model_trt = torch_tensorrt.compile(model)',
                'constraints': 'CUDA only, complex models may have compatibility issues'
            },
            {
                'name': 'XLA Compilation',
                'description': 'Linear algebra compiler for accelerated training',
                'speedup': '20-30% for training',
                'memory_savings': 'Minimal',
                'implementation': 'For TF: tf.config.optimizer.set_jit(True)',
                'constraints': 'Primarily for TPUs and TensorFlow'
            }
        ],
        'parallelism': [
            {
                'name': 'Data Parallel Training',
                'description': 'Splits batches across multiple GPUs',
                'speedup': 'Near-linear with number of GPUs',
                'memory# filepath: c:\Users\adilm\repositories\Python\neural-scope\aiml_complexity\ml_based_suggestions_2.py
"""
Neural-Scope ML Advisor: Advanced AI/ML Performance Optimization System

A comprehensive, industry-leading solution for ML model analysis, profiling, and optimization
that bridges the gap between data science and ML engineering. This system provides
actionable, model-specific recommendations based on deep inspection of model architecture,
execution patterns, and hardware utilization.

Key features:
1. Deep Model Inspection
   - Layer-by-layer profiling with operation-specific bottleneck detection
   - Computation vs. memory bottleneck classification
   - Framework-specific optimization opportunities (PyTorch/TensorFlow)
   - Memory access pattern analysis for cache efficiency

2. Execution Profiling
   - Kernel launch statistics and GPU utilization metrics
   - Thread-level CPU analysis with GIL contention detection
   - Data pipeline efficiency evaluation
   - Distributed training communication overhead measurement

3. Intelligent Optimization Recommendations
   - Architecture-specific suggestions (attention mechanisms, embedding tables)
   - Automatic quantization opportunity detection with precision impact analysis
   - JIT/XLA/TorchScript compilation benefits estimation
   - Custom CUDA kernel suggestions for hotspot operations

4. Data Quality Assessment
   - Intersectional bias detection across protected attributes
   - Representation disparity measurement
   - Distribution shift detection between train/test/production
   - Noise and outlier impact quantification

5. Hardware Matching
   - Specific GPU recommendations based on memory/compute bottleneck analysis
   - Cost-performance optimization for cloud providers
   - TPU/IPU/specialized hardware opportunity identification
   - Multi-node scaling efficiency prediction
"""

import os
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
import matplotlib.pyplot as plt
import psycopg2
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Framework-specific imports with graceful degradation
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.python.eager import profiler as tf_profiler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRNESS_METRICS_AVAILABLE = True
except ImportError:
    FAIRNESS_METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NeuralScope')

@dataclass
class LayerProfile:
    """Detailed profile information for a single model layer"""
    name: str
    layer_type: str
    parameters: int
    execution_time: float
    memory_usage: float
    flops: Optional[int] = None
    io_bound: bool = False
    compute_bound: bool = False
    memory_bound: bool = False
    bottleneck_score: float = 0.0
    optimization_candidates: List[str] = None
    
    def __post_init__(self):
        if self.optimization_candidates is None:
            self.optimization_candidates = []

@dataclass
class ModelProfile:
    """Comprehensive profile of an ML model's performance characteristics"""
    model_name: str
    framework: str
    total_parameters: int
    batch_size: int
    input_shape: tuple
    dtype: str
    execution_time: float
    peak_memory: float
    throughput: float
    layers: List[LayerProfile]
    bottleneck_type: str = "unknown"
    data_pipeline_efficiency: float = 1.0
    gpu_utilization: float = 0.0
    parallelizable: bool = False
    
    @property
    def total_execution_time(self) -> float:
        """Sum of all layer execution times"""
        return sum(layer.execution_time for layer in self.layers)
    
    @property
    def bottleneck_layers(self) -> List[LayerProfile]:
        """Return layers sorted by bottleneck score (descending)"""
        return sorted(self.layers, key=lambda x: x.bottleneck_score, reverse=True)

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment results"""
    dataset_name: str
    total_samples: int
    missing_values_pct: Dict[str, float]
    categorical_imbalance: Dict[str, float]
    numerical_outlier_pct: Dict[str, float]
    distribution_skew: Dict[str, float]
    intersectional_bias_score: float = 0.0
    class_imbalance_score: float = 0.0
    label_noise_estimate: float = 0.0
    fairness_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fairness_metrics is None:
            self.fairness_metrics = {}
    
    @property
    def quality_score(self) -> float:
        """Overall data quality score from 0-1"""
        scores = [
            1.0 - max(self.missing_values_pct.values()) if self.missing_values_pct else 1.0,
            1.0 - self.class_imbalance_score,
            1.0 - self.label_noise_estimate,
            1.0 - self.intersectional_bias_score,
        ]
        return sum(scores) / len(scores)

class HardwareRecommender:
    """Recommends optimal hardware based on model profile and constraints"""
    
    # Hardware database with capabilities and costs
    GPU_DATABASE = {
        'T4': {
            'memory': 16, 'compute': 8.1, 'fp16_speedup': 2.0, 'int8_speedup': 4.0, 
            'cost_per_hour': 0.35, 'provider': 'GCP'
        },
        'V100': {
            'memory': 32, 'compute': 15.7, 'fp16_speedup': 8.0, 'int8_speedup': 2.0, 
            'cost_per_hour': 0.90, 'provider': 'AWS'
        },
        'A100': {
            'memory': 80, 'compute': 19.5, 'fp16_speedup': 20.0, 'int8_speedup': 40.0, 
            'cost_per_hour': 2.50, 'provider': 'GCP'
        },
        'H100': {
            'memory': 80, 'compute': 51.0, 'fp16_speedup': 30.0, 'int8_speedup': 60.0, 
            'cost_per_hour': 5.50, 'provider': 'Azure'
        },
        'RTX 4090': {
            'memory': 24, 'compute': 18.0, 'fp16_speedup': 16.0, 'int8_speedup': 32.0, 
            'cost_per_hour': 0.0, 'provider': 'On-prem'
        }
    }
    
    def __init__(self, budget_constraint: Optional[float] = None):
        self.budget_constraint = budget_constraint
    
    def recommend_for_profile(self, profile: ModelProfile) -> Dict[str, Any]:
        """Recommend hardware based on detailed model profile"""
        is_memory_bound = profile.bottleneck_type == "memory"
        is_compute_bound = profile.bottleneck_type == "compute"
        
        required_memory = profile.peak_memory * 1.5  # Add 50% buffer
        
        # Filter GPUs that meet memory requirements
        candidates = {gpu: specs for gpu, specs in self.GPU_DATABASE.items() 
                     if specs['memory'] >= required_memory}
        
        if not candidates:
            # Fall back to multi-GPU configurations if single GPU can't fit model
            multi_gpu_options = []
            for gpu, specs in self.GPU_DATABASE.items():
                num_gpus = max(2, int(np.ceil(required_memory / specs['memory'])))
                if num_gpus < 8:  # Limit to reasonable configurations
                    multi_gpu_options.append({
                        'gpu': gpu,
                        'count': num_gpus,
                        'total_memory': specs['memory'] * num_gpus,
                        'total_compute': specs['compute'] * num_gpus * 0.9,  # 10% overhead
                        'cost': specs['cost_per_hour'] * num_gpus,
                        'efficiency': (specs['compute'] / specs['cost_per_hour']) if specs['cost_per_hour'] > 0 else float('inf')
                    })
                    
            if not multi_gpu_options:
                return {"recommendation": "Model too large for standard GPUs - consider model sharding or optimization"}
            
            # Sort by efficiency or cost constraint
            if self.budget_constraint:
                valid_options = [opt for opt in multi_gpu_options if opt['cost'] <= self.budget_constraint]
                if not valid_options:
                    return {"recommendation": f"No configuration meets budget constraint of ${self.budget_constraint}/hour"}
                best_option = max(valid_options, key=lambda x: x['total_compute'])
            else:
                best_option = max(multi_gpu_options, key=lambda x: x['efficiency'])
                
            return {
                "recommendation": f"{best_option['count']}x {best_option['gpu']}",
                "configuration": "distributed" if best_option['count'] > 1 else "single",
                "reason": "Memory requirements exceed single GPU capacity",
                "estimated_cost": f"${best_option['cost']:.2f}/hour on {self.GPU_DATABASE[best_option['gpu']]['provider']}",
                "optimization_priority": "Model sharding and memory efficiency"
            }
        
        # For memory-bound workloads, prioritize GPUs with more memory
        if is_memory_bound:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
                
            best_gpu = max(candidates.items(), key=lambda x: x[1]['memory'])
            reason = "Memory-bound workload needs maximum GPU memory"
            optimization = "Memory optimization (quantization, attention optimizations)"
        
        # For compute-bound workloads, prioritize computational capacity
        elif is_compute_bound:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
                    
            best_gpu = max(candidates.items(), key=lambda x: x[1]['compute'])
            reason = "Compute-bound workload needs maximum computational power"
            optimization = "Algorithmic optimizations, kernel fusion, TensorRT"
        
        # For balanced or unknown workloads, find best price/performance ratio
        else:
            if self.budget_constraint:
                affordable = {gpu: specs for gpu, specs in candidates.items() 
                             if specs['cost_per_hour'] <= self.budget_constraint}
                if affordable:
                    candidates = affordable
            
            # Compute efficiency as TFLOPS/dollar
            best_gpu = max(candidates.items(), 
                           key=lambda x: (x[1]['compute']/x[1]['cost_per_hour']) if x[1]['cost_per_hour'] > 0 else float('inf'))
            reason = "Balanced workload benefits from good price-performance ratio"
            optimization = "Mixed precision training, pipeline parallelism"
        
        gpu_name, specs = best_gpu
        
        mixed_precision_speedup = None
        if profile.dtype == 'float32':
            mixed_precision_speedup = specs['fp16_speedup']
        
        return {
            "recommendation": gpu_name,
            "memory": f"{specs['memory']} GB",
            "compute": f"{specs['compute']} TFLOPS",
            "provider": specs['provider'],
            "estimated_cost": f"${specs['cost_per_hour']:.2f}/hour" if specs['cost_per_hour'] > 0 else "On-premises",
            "reason": reason,
            "optimization_priority": optimization,
            "mixed_precision_benefit": f"{mixed_precision_speedup}x speedup possible" if mixed_precision_speedup else None
        }

class DataQualityAnalyzer:
    """Advanced data quality assessment for ML datasets"""
    
    def __init__(self, sensitive_attributes: Optional[List[str]] = None):
        """
        Initialize with a list of columns to treat as sensitive/protected attributes
        for fairness and bias analysis
        """
        self.sensitive_attributes = sensitive_attributes or []
    
    def analyze_dataset(self, data: pd.DataFrame, label_column: Optional[str] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment on a pandas DataFrame
        """
        dataset_name = getattr(data, 'name', 'unknown_dataset')
        total_samples = len(data)
        
        # Compute missing value statistics by column
        missing_pct = {}
        for col in data.columns:
            missing_pct[col] = data[col].isna().mean() * 100
        
        # Analyze categorical columns for imbalance
        categorical_imbalance = {}
        for col in data.select_dtypes(include=['object', 'category']).columns:
            value_counts = data[col].value_counts(normalize=True)
            # Gini impurity as measure of imbalance
            gini = 1 - np.sum(value_counts ** 2)
            categorical_imbalance[col] = gini
        
        # Detect outliers in numerical columns using IQR method
        numerical_outliers = {}
        for col in data.select_dtypes(include=np.number).columns:
            if col == label_column:
                continue
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))
            numerical_outliers[col] = outlier_mask.mean() * 100
        
        # Calculate distribution skewness
        skewness = {}
        for col in data.select_dtypes(include=np.number).columns:
            skewness[col] = abs(data[col].skew())
        
        # Class imbalance for classification tasks
        class_imbalance = 0
        if label_column and label_column in data.columns:
            # Check if it's likely a classification task
            if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                # Normalized entropy as imbalance metric (0=perfect balance, 1=complete imbalance)
                value_counts = data[label_column].value_counts(normalize=True)
                if len(value_counts) > 1:
                    entropy = -np.sum(value_counts * np.log(value_counts))
                    max_entropy = np.log(len(value_counts))
                    class_imbalance = 1 - (entropy / max_entropy)
        
        # Estimate label noise (simplified approach)
        label_noise = 0.0
        if label_column and len(data) > 100:
            try:
                # For classification: use label consistency in feature neighborhoods as proxy for noise
                if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Sample if dataset is very large to speed up computation
                    data_sample = data.sample(min(10000, len(data))) if len(data) > 10000 else data
                    
                    # Select only numeric features
                    numeric_cols = data_sample.select_dtypes(include=np.number).columns
                    numeric_cols = [c for c in numeric_cols if c != label_column]
                    
                    if len(numeric_cols) > 0:
                        X = data_sample[numeric_cols].fillna(0)
                        y = data_sample[label_column]
                        
                        # Normalize features
                        X = (X - X.mean()) / X.std()
                        
                        # Find 5 nearest neighbors for each point
                        nn = NearestNeighbors(n_neighbors=6)  # including the point itself
                        nn.fit(X)
                        _, indices = nn.kneighbors(X)
                        
                        # Calculate label consistency within neighborhoods
                        inconsistent_count = 0
                        for i, neighbors in enumerate(indices):
                            center_label = y.iloc[i]
                            neighbor_labels = [y.iloc[j] for j in neighbors[1:]]  # exclude self
                            inconsistent_count += (1 - neighbor_labels.count(center_label) / len(neighbor_labels))
                        
                        label_noise = inconsistent_count / len(X)
                        label_noise = min(1.0, max(0.0, label_noise))  # clamp to [0,1]
            except Exception as e:
                logger.warning(f"Label noise estimation failed: {e}")
        
        # Intersectional bias analysis
        intersectional_bias = 0.0
        fairness_metrics = {}
        
        if label_column and self.sensitive_attributes and FAIRNESS_METRICS_AVAILABLE:
            try:
                if data[label_column].dtype == 'object' or data[label_column].nunique() < 10:
                    label_values = data[label_column].unique()
                    if len(label_values) == 2:  # Binary classification case
                        y_true = data[label_column]
                        
                        # Analyze each sensitive attribute
                        dpd_scores = []
                        eod_scores = []
                        for attr in self.sensitive_attributes:
                            if attr in data.columns:
                                # Remove rows with missing values in the sensitive attribute
                                mask = ~data[attr].isna()
                                if mask.sum() > 100:  # Need sufficient data
                                    dpd = demographic_parity_difference(
                                        y_true=y_true[mask],
                                        sensitive_features=data[attr][mask]
                                    )
                                    eod = equalized_odds_difference(
                                        y_true=y_true[mask],
                                        y_pred=y_true[mask],  # Using true labels as proxy
                                        sensitive_features=data[attr][mask]
                                    )
                                    dpd_scores.append(dpd)
                                    eod_scores.append(eod)
                                    fairness_metrics[f"demographic_parity_{attr}"] = dpd
                                    fairness_metrics[f"equalized_odds_{attr}"] = eod
                        
                        # Average the scores
                        if dpd_scores:
                            intersectional_bias = (sum(dpd_scores) / len(dpd_scores) + 
                                                sum(eod_scores) / len(eod_scores)) / 2
            except Exception as e:
                logger.warning(f"Fairness metrics calculation failed: {e}")
        
        return DataQualityMetrics(
            dataset_name=dataset_name,
            total_samples=total_samples,
            missing_values_pct=missing_pct,
            categorical_imbalance=categorical_imbalance,
            numerical_outlier_pct=numerical_outliers,
            distribution_skew=skewness,
            intersectional_bias_score=intersectional_bias,
            class_imbalance_score=class_imbalance,
            label_noise_estimate=label_noise,
            fairness_metrics=fairness_metrics
        )
    
    def generate_recommendations(self, metrics: DataQualityMetrics) -> List[Dict[str, Any]]:
        """Generate specific data quality improvement recommendations"""
        recommendations = []
        
        # High missing value columns
        high_missing = [(col, pct) for col, pct in metrics.missing_values_pct.items() if pct > 5]
        if high_missing:
            for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "High Missing Values",
                    "column": col,
                    "severity": "High" if pct > 20 else "Medium",
                    "details": f"{pct:.1f}% missing values",
                    "suggestions": [
                        f"Impute missing values using {self._suggest_imputation_method(col)}",
                        "Consider if this feature should be dropped if not informative",
                        "Investigate data collection process to reduce missingness"
                    ]
                })
        
        # Highly skewed numerical features
        high_skew = [(col, skew) for col, skew in metrics.distribution_skew.items() if skew > 1.0]
        if high_skew:
            for col, skew in sorted(high_skew, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "Skewed Distribution",
                    "column": col,
                    "severity": "Medium",
                    "details": f"Skewness: {skew:.2f}",
                    "suggestions": [
                        f"Apply log transformation: np.log1p({col})",
                        f"Apply Box-Cox transformation on {col}",
                        f"Standardize or normalize {col} to improve training stability"
                    ]
                })
        
        # High outlier percentage
        high_outliers = [(col, pct) for col, pct in metrics.numerical_outlier_pct.items() if pct > 1.0]
        if high_outliers:
            for col, pct in sorted(high_outliers, key=lambda x: x[1], reverse=True)[:3]:
                recommendations.append({
                    "issue": "Outliers Detected",
                    "column": col,
                    "severity": "Medium" if pct > 5 else "Low",
                    "details": f"{pct:.1f}% outliers detected",
                    "suggestions": [
                        f"Cap outliers using winsorization: df['{col}'] = np.clip(df['{col}'], lower, upper)",
                        "Investigate if outliers represent valid but rare cases",
                        f"Create binary flag feature: is_{col}_outlier to preserve information"
                    ]
                })
        
        # Class imbalance
        if metrics.class_imbalance_score > 0.3:
            recommendations.append({
                "issue": "Class Imbalance",
                "column": "target/label",
                "severity": "High" if metrics.class_imbalance_score > 0.6 else "Medium",
                "details": f"Imbalance score: {metrics.class_imbalance_score:.2f}",
                "suggestions": [
                    "Use class weights in model training",
                    "Apply SMOTE or other resampling techniques",
                    "Use focal loss or other imbalance-aware loss functions",
                    "Evaluate models with balanced accuracy or F1-score instead of accuracy"
                ]
            })
        
        # Possible label noise
        if metrics.label_noise_estimate > 0.1:
            recommendations.append({
                "issue": "Potential Label Noise",
                "column": "target/label",
                "severity": "High" if metrics.label_noise_estimate > 0.3 else "Medium",
                "details": f"Estimated noise: {metrics.label_noise_estimate:.2f}",
                "suggestions": [
                    "Manually review a sample of potentially mislabeled instances",
                    "Consider robust loss functions (e.g., MAE instead of MSE)",
                    "Implement label smoothing in classification tasks",
                    "Try data cleaning techniques like Confident Learning (cleanlab library)"
                ]
            })
        
        # Bias/fairness issues
        if metrics.intersectional_bias_score > 0.1 and metrics.fairness_metrics:
            highest_bias = max(metrics.fairness_metrics.items(), key=lambda x: x[1])
            metric_name, value = highest_bias
            attribute = metric_name.split('_')[-1]
            recommendations.append({
                "issue": "Potential Bias Detected",
                "column": attribute,
                "severity": "High" if metrics.intersectional_bias_score > 0.2 else "Medium",
                "details": f"Bias score: {metrics.intersectional_bias_score:.2f}",
                "suggestions": [
                    "Apply fairness constraints during model training",
                    f"Analyze and balance representation across {attribute} groups",
                    "Use post-processing methods to equalize predictions across groups",
                    "Consider collecting additional data from underrepresented groups"
                ]
            })
        
        return recommendations
    
    def _suggest_imputation_method(self, column_name: str) -> str:
        """Suggest appropriate imputation method based on column name heuristics"""
        column_lower = column_name.lower()
        
        if any(word in column_lower for word in ['age', 'year', 'income', 'salary', 'price']):
            return "median imputation"
        elif any(word in column_lower for word in ['count', 'number', 'quantity', 'amt']):
            return "mean or median imputation"
        elif any(word in column_lower for word in ['category', 'type', 'class', 'group']):
            return "mode imputation"
        elif any(word in column_lower for word in ['date', 'time']):
            return "forward fill or interpolation"
        else:
            return "KNN or model-based imputation"

class ModelOptimizer:
    """Recommends and implements model-specific optimizations"""
    
    ATTENTION_PATTERNS = [
        (r'MultiHead', 'Multi-head attention layer'),
        (r'SelfAttention', 'Self-attention mechanism'),
        (r'TransformerEncoderLayer', 'Transformer encoder'),
        (r'TransformerDecoderLayer', 'Transformer decoder')
    ]
    
    OPTIMIZATION_TECHNIQUES = {
        'attention': [
            {
                'name': 'FlashAttention',
                'description': 'IO-aware attention implementation with better memory efficiency',
                'speedup': '2-4x for attention operations',
                'memory_savings': '10-20% of total memory',
                'implementation': 'from flash_attn import flash_attn_qkvpacked_func',
                'constraints': 'CUDA only, SM 8.0+ (Ampere or newer)'
            },
            {
                'name': 'Memory-efficient attention',
                'description': 'Reduces memory usage by recomputing activations',
                'speedup': '5-15% end-to-end',
                'memory_savings': '20-30% of attention memory',
                'implementation': 'from xformers.ops import memory_efficient_attention',
                'constraints': 'Requires xformers library'
            }
        ],
        'quantization': [
            {
                'name': 'Dynamic Quantization',
                'description': 'Quantize weights to INT8 dynamically during inference',
                'speedup': '2-4x on CPU inference',
                'memory_savings': '75% for quantized layers',
                'implementation': 'model_int8 = torch.quantization.quantize_dynamic(model)',
                'constraints': 'Primarily for CPU inference'
            },
            {
                'name': 'Weight-only Quantization',
                'description': 'Int8 weights with fp16/32 activations',
                'speedup': '1.5-2x on inference',
                'memory_savings': '50-75% for model weights',
                'implementation': 'model = AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)',
                'constraints': 'Requires transformers >= 4.30.0 and CUDA'
            },
            {
                'name': 'QLoRA fine-tuning',
                'description': 'Quantized weights with trainable LoRA adapters',
                'speedup': 'Enables fine-tuning with ~80% less memory',
                'memory_savings': '70-80% during fine-tuning',
                'implementation': 'from peft import get_peft_model, LoraConfig',
                'constraints': 'Requires PEFT library and bitsandbytes'
            }
        ],
        'compilation': [
            {
                'name': 'TorchScript Compilation',
                'description': 'Compiles model for optimized inference',
                'speedup': '20-50% for inference',
                'memory_savings': 'Minimal',
                'implementation': 'model_ts = torch.jit.script(model)',
                'constraints': 'Model must be compatible with TorchScript tracing'
            },
            {
                'name': 'TensorRT Export',
                'description': 'NVIDIA inference optimization with kernel fusion',
                'speedup': '2-5x for inference',
                'memory_savings': 'Varies by model',
                'implementation': 'import torch_tensorrt; model_trt = torch_tensorrt.compile(model)',
                'constraints': 'CUDA only, complex models may have compatibility issues'
            },
            {
                'name': 'XLA Compilation',
                'description': 'Linear algebra compiler for accelerated training',
                'speedup': '20-30% for training',
                'memory_savings': 'Minimal',
                'implementation': 'For TF: tf.config.optimizer.set_jit(True)',
                'constraints': 'Primarily for TPUs and TensorFlow'
            }
        ],
        'parallelism': [
            {
                'name': 'Data Parallel Training',
                'description': 'Splits batches across multiple GPUs',
                'speedup': 'Near-linear with number of GPUs',
                'memory_savings': 'Minimal',
                'recommendation': "Consider feature hashing, target encoding, or grouping rare categories.",
                'code_example': """
                # Feature hashing for high cardinality features
                from sklearn.feature_extraction import FeatureHasher
                h = FeatureHasher(n_features=50, input_type='string')
                hashed_features = h.transform(df['{col}'].astype(str)).toarray()
                # Add hashed features to dataframe
                for i in range(50):
                    df[f'{col}_hash_{i}'] = hashed_features[:, i]
                # Then drop original column
                df.drop('{col}', axis=1, inplace=True)
                """
                                    })
                
                        return recommendations
                
                
                class ModelPerformanceProfiler:
                    """
                    Advanced model performance profiling with granular layer-wise analysis,
                    memory tracking, CPU/GPU utilization, and targeted optimization recommendations.
                    """
                    def __init__(self, model, framework: str = None):
                        self.model = model
                        
                        # Auto-detect framework if not specified
                        if framework is None:
                            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                                framework = 'pytorch'
                            elif TF_AVAILABLE and 'tensorflow' in str(type(model)):
                                framework = 'tensorflow' 
                            else:
                                raise ValueError("Could not auto-detect framework. Please specify 'pytorch' or 'tensorflow'.")
                        
                        self.framework = framework.lower()
                        self.profiling_result = None
                        self.model_info = self._extract_model_info()
                        self.hardware_info = self._get_hardware_info()
                
                    def _extract_model_info(self) -> Dict[str, Any]:
                        """Extract detailed model architecture information"""
                        model_info = {
                            'parameter_count': 0,
                            'layer_types': {},
                            'architecture_type': 'unknown'
                        }
                        
                        if self.framework == 'pytorch' and TORCH_AVAILABLE:
                            # Get parameter count
                            model_info['parameter_count'] = sum(p.numel() for p in self.model.parameters())
                            
                            # Analyze layer types
                            for name, module in self.model.named_modules():
                                module_type = type(module).__name__
                                if module_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                                    model_info['layer_types'][module_type] = model_info['layer_types'].get(module_type, 0) + 1
                            
                            # Determine architecture type
                            if any(t in str(self.model) for t in ['Transformer', 'MultiheadAttention']):
                                model_info['architecture_type'] = 'transformer'
                            elif any(t in str(self.model) for t in ['Conv', 'conv']):
                                model_info['architecture_type'] = 'cnn'
                            elif any(t in str(self.model) for t in ['RNN', 'LSTM', 'GRU']):
                                model_info['architecture_type'] = 'rnn'
                            elif any(t in str(self.model) for t in ['Policy', 'Actor', 'Critic']):
                                model_info['architecture_type'] = 'reinforcement_learning'
                            
                        elif self.framework == 'tensorflow' and TF_AVAILABLE:
                            # Get parameter count
                            model_info['parameter_count'] = self.model.count_params()
                            
                            # Analyze layer types
                            for layer in self.model.layers:
                                layer_type = type(layer).__name__
                                model_info['layer_types'][layer_type] = model_info['layer_types'].get(layer_type, 0) + 1
                            
                            # Determine architecture type based on layer composition
                            if any('attention' in l.name.lower() for l in self.model.layers if hasattr(l, 'name')):
                                model_info['architecture_type'] = 'transformer'
                            elif any(isinstance(l, tf_layers.Conv1D) or isinstance(l, tf_layers.Conv2D) for l in self.model.layers):
                                model_info['architecture_type'] = 'cnn'
                            elif any(isinstance(l, tf_layers.LSTM) or isinstance(l, tf_layers.GRU) for l in self.model.layers):
                                model_info['architecture_type'] = 'rnn'
                                
                        return model_info
                
                    def _get_hardware_info(self) -> Dict[str, Any]:
                        """Collect detailed hardware information"""
                        hardware_info = {
                            'cpu': {
                                'count': os.cpu_count(),
                                'model': 'Unknown'
                            },
                            'memory': {
                                'total': round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                                'available': round(psutil.virtual_memory().available / (1024**3), 2)  # GB
                            },
                            'gpu': []
                        }
                        
                        # Try to get CPU model
                        try:
                            import platform
                            if platform.system() == "Windows":
                                import winreg
                                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                                hardware_info['cpu']['model'] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                            elif platform.system() == "Linux":
                                with open('/proc/cpuinfo') as f:
                                    for line in f:
                                        if 'model name' in line:
                                            hardware_info['cpu']['model'] = line.split(':')[1].strip()
                                            break
                        except Exception as e:
                            logger.warning(f"Could not determine CPU model: {e}")
                        
                        # Get GPU information
                        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
                            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                            visible_devices = [int(x) for x in cuda_visible.split(',')] if cuda_visible else list(range(torch.cuda.device_count()))
                            
                            for i in visible_devices:
                                hardware_info['gpu'].append({
                                    'name': torch.cuda.get_device_name(i),
                                    'memory_total': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),  # GB
                                    'compute_capability': f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                                })
                                
                        elif self.framework == 'tensorflow' and TF_AVAILABLE:
                            gpus = tf.config.list_physical_devices('GPU')
                            for i, gpu in enumerate(gpus):
                                gpu_details = tf.config.experimental.get_device_details(gpu)
                                hardware_info['gpu'].append({
                                    'name': gpu_details.get('device_name', f"GPU {i}"),
                                    'compute_capability': str(gpu_details.get('compute_capability', 'Unknown'))
                                })
                                
                        return hardware_info
                
                    def profile_execution(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """
                        Profile model execution with detailed metrics:
                        - Layer-wise execution time
                        - Memory usage patterns
                        - CPU/GPU utilization
                        - Operation-level performance stats
                        """
                        if self.framework == 'pytorch':
                            return self._profile_pytorch(input_data, batch_size, num_steps)
                        elif self.framework == 'tensorflow':
                            return self._profile_tensorflow(input_data, batch_size, num_steps)
                        else:
                            raise ValueError(f"Unsupported framework: {self.framework}")
                
                    def _profile_pytorch(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """PyTorch-specific profiling with detailed layer and kernel analysis"""
                        if not TORCH_AVAILABLE:
                            raise ImportError("PyTorch is not available")
                            
                        # Ensure model is in evaluation mode
                        self.model.eval()
                        
                        # Get device
                        device = next(self.model.parameters()).device
                        
                        # Prepare data loader if necessary
                        if hasattr(input_data, '__getitem__') and not isinstance(input_data, torch.Tensor):
                            if batch_size is None:
                                batch_size = 16  # Default batch size
                            data_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size)
                            sample_input = next(iter(data_loader))
                            if isinstance(sample_input, (list, tuple)):
                                # Assume first element is input if it's a tuple/list (input, target)
                                sample_input = sample_input[0].to(device)
                            else:
                                sample_input = sample_input.to(device)
                        else:
                            # Use provided tensor directly
                            sample_input = input_data
                            if not isinstance(sample_input, torch.Tensor):
                                sample_input = torch.tensor(sample_input).to(device)
                        
                        # Initialize metrics containers
                        memory_usage = {'peak': 0, 'active': 0, 'allocated': 0}
                        cpu_utilization = {'mean': 0, 'max': 0}
                        gpu_utilization = {'mean': 0, 'max': 0, 'memory_usage_percent': 0}
                        operation_stats = {}
                        bottlenecks = []
                        
                        # Warmup run
                        with torch.no_grad():
                            self.model(sample_input)
                        
                        # Record CPU usage
                        cpu_percent_values = []
                        start_time = time.time()
                        
                        # Use PyTorch's profiler for detailed analysis
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True
                        ) as prof:
                            for _ in range(num_steps):
                                with record_function("model_inference"):
                                    with torch.no_grad():
                                        self.model(sample_input)
                                
                                # Record CPU usage during execution
                                cpu_percent = psutil.cpu_percent(interval=None)
                                cpu_percent_values.append(cpu_percent)
                                
                                # Record GPU utilization if available
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    memory_usage['allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
                                    memory_usage['peak'] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                                    memory_usage['reserved'] = torch.cuda.memory_reserved() / (1024**2)  # MB
                        
                        execution_time = (time.time() - start_time) / num_steps
                        
                        # Process CPU utilization
                        cpu_utilization['mean'] = sum(cpu_percent_values) / len(cpu_percent_values)
                        cpu_utilization['max'] = max(cpu_percent_values)
                        
                        # Process profiling results
                        events = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
                        
                        # Extract key metrics from profiler events
                        operation_stats = {}
                        for evt in prof.key_averages():
                            if evt.key not in operation_stats:
                                operation_stats[evt.key] = {
                                    'cpu_time': evt.cpu_time_total / 1000,  # convert to ms
                                    'cuda_time': evt.cuda_time_total / 1000 if hasattr(evt, 'cuda_time_total') else 0,
                                    'self_cpu_time': evt.self_cpu_time_total / 1000,
                                    'self_cuda_time': evt.self_cuda_time_total / 1000 if hasattr(evt, 'self_cuda_time_total') else 0,
                                    'cpu_memory_usage': evt.cpu_memory_usage / (1024**2) if hasattr(evt, 'cpu_memory_usage') else 0,  # MB
                                    'cuda_memory_usage': evt.cuda_memory_usage / (1024**2) if hasattr(evt, 'cuda_memory_usage') else 0,  # MB
                                    'occurrences': 1
                                }
                            else:
                                operation_stats[evt.key]['occurrences'] += 1
                        
                        # Identify bottlenecks
                        bottlenecks = self._identify_pytorch_bottlenecks(operation_stats, memory_usage)
                        
                        # Generate recommendations
                        recommendations = self._generate_pytorch_recommendations(
                            bottlenecks, 
                            operation_stats, 
                            memory_usage,
                            self.model_info,
                            batch_size
                        )
                        
                        # Create profiling result
                        self.profiling_result = ProfilingResult(
                            execution_time=execution_time,
                            memory_usage=memory_usage,
                            cpu_utilization=cpu_utilization,
                            gpu_utilization=gpu_utilization if torch.cuda.is_available() else None,
                            operation_stats=operation_stats,
                            bottlenecks=bottlenecks,
                            recommendations=recommendations
                        )
                        
                        return self.profiling_result
                
                    def _identify_pytorch_bottlenecks(self, operation_stats, memory_usage) -> List[Dict[str, Any]]:
                        """Identify performance bottlenecks in PyTorch model execution"""
                        bottlenecks = []
                        
                        # Sort operations by CUDA time (if available) or CPU time
                        sorted_ops = sorted(
                            operation_stats.items(), 
                            key=lambda x: x[1]['cuda_time'] if x[1]['cuda_time'] > 0 else x[1]['cpu_time'],
                            reverse=True
                        )
                        
                        # Top-3 most time-consuming operations
                        for i, (op_name, stats) in enumerate(sorted_ops[:3]):
                            if i == 0:  # Most expensive operation
                                severity = 'high'
                            elif i == 1:
                                severity = 'medium'
                            else:
                                severity = 'low'
                                
                            cuda_time = stats['cuda_time']
                            cpu_time = stats['cpu_time']
                            
                            bottleneck = {
                                'type': 'operation',
                                'name': op_name,
                                'severity': severity,
                                'time_ms': cuda_time if cuda_time > 0 else cpu_time,
                                'device': 'gpu' if cuda_time > 0 else 'cpu',
                                'memory_mb': stats['cuda_memory_usage'] if cuda_time > 0 else stats['cpu_memory_usage']
                            }
                            bottlenecks.append(bottleneck)
                        
                        # Check for memory-related bottlenecks
                        if memory_usage['peak'] > 0.9 * self.hardware_info['gpu'][0]['memory_total'] * 1024:  # Convert GB to MB
                            bottlenecks.append({
                                'type': 'memory',
                                'name': 'GPU Memory Near Capacity',
                                'severity': 'high',
                                'memory_mb': memory_usage['peak'],
                                'details': 'Model is using >90% of available GPU memory'
                            })
                        
                        # Check for operation pattern bottlenecks (e.g., excessive memory copying)
                        mem_copies = [op for op, stats in operation_stats.items() if 'memcpy' in op.lower()]
                        if mem_copies and sum(operation_stats[op]['cuda_time'] for op in mem_copies) > 0.2 * sorted_ops[0][1]['cuda_time']:
                            bottlenecks.append({
                                'type': 'data_movement',
                                'name': 'Excessive Host-Device Memory Transfers',
                                'severity': 'high',
                                'time_ms': sum(operation_stats[op]['cuda_time'] for op in mem_copies),
                                'details': 'Significant time spent on memory transfers between CPU and GPU'
                            })
                            
                        return bottlenecks
                
                    def _generate_pytorch_recommendations(self, bottlenecks, operation_stats, 
                                                         memory_usage, model_info, batch_size) -> List[Dict[str, Any]]:
                        """Generate specific optimization recommendations based on profiling results"""
                        recommendations = []
                        
                        # Check for memory-intensive operations
                        if any(b['type'] == 'memory' for b in bottlenecks):
                            # Recommend gradient checkpointing for transformer models
                            if model_info['architecture_type'] == 'transformer':
                                recommendations.append({
                                    'type': 'memory_optimization',
                                    'severity': 'high',
                                    'description': 'Model is using excessive GPU memory',
                                    'recommendation': 'Implement gradient checkpointing to reduce memory usage',
                                    'estimated_impact': 'Can reduce memory usage by 30-40% with minimal performance impact',
                                    'code_example': """
                # Enable gradient checkpointing
                model.gradient_checkpointing_enable()
                
                # Or manually for specific modules:
                from torch.utils.checkpoint import checkpoint
                def forward(self, *args, **kwargs):
                    # Use checkpoint for transformer layers
                    output = checkpoint(self.transformer_layer, *args, **kwargs)
                    output = self.final_layer(output)  # Non-checkpointed operations
                    return output
                """
                                })
                                
                            # Recommend smaller batch size if memory is the primary issue
                            recommendations.append({
                                'type': 'training_parameter',
                                'severity': 'medium',
                                'description': 'Model exceeds GPU memory capacity',
                                'recommendation': f'Reduce batch size (currently {batch_size})',
                                'estimated_impact': 'Linear reduction in memory usage',
                                'code_example': f"trainer.train(batch_size={max(1, batch_size // 2)})"
                            })
                            
                            # Recommend mixed precision training
                            recommendations.append({
                                'type': 'training_technique',
                                'severity': 'high',
                                'description': 'Model using full precision (FP32) computation',
                                'recommendation': 'Enable mixed precision training (FP16/BF16)',
                                'estimated_impact': 'Can reduce memory usage by up to 50% with potential speedup',
                                'code_example': """
                # For PyTorch native mixed precision
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                
                # In training loop:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                """
                            })
                            
                        # Check for attention-related bottlenecks in transformer models
                        if model_info['architecture_type'] == 'transformer' and any('attention' in b['name'].lower() for b in bottlenecks):
                            recommendations.append({
                                'type': 'algorithm_optimization',
                                'severity': 'high',
                                'description': 'Attention mechanism is a performance bottleneck',
                                'recommendation': 'Implement FlashAttention for faster transformer computation',
                                'estimated_impact': 'Can improve attention computation speed by 2-4x',
                                'code_example': """
                # Install flash-attn package
                # pip install flash-attn
                
                from flash_attn import flash_attn_func
                
                # Replace standard attention with flash attention
                def flash_attention_forward(q, k, v, mask=None):
                    # q, k, v: (batch_size, seq_len, num_heads, head_dim)
                    batch_size, seq_len, num_heads, head_dim = q.shape
                    q, k, v = [x.reshape(-1, seq_len, num_heads, head_dim) for x in (q, k, v)]
                    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None)
                    return out.reshape(batch_size, seq_len, num_heads * head_dim)
                """
                            })
                            
                        # Check for data transfer bottlenecks
                        if any(b['type'] == 'data_movement' for b in bottlenecks):
                            recommendations.append({
                                'type': 'data_pipeline',
                                'severity': 'high',
                                'description': 'Excessive CPU-GPU data transfers',
                                'recommendation': 'Use pinned memory and non-blocking transfers',
                                'estimated_impact': 'Can reduce data transfer overhead by 20-30%',
                                'code_example': """
                # Use pinned memory in DataLoader
                train_loader = DataLoader(dataset, batch_size=32, pin_memory=True)
                
                # Use non-blocking transfers
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                """
                            })
                            
                        # Large model optimization suggestions
                        if model_info['parameter_count'] > 1e9:  # >1B parameters
                            recommendations.append({
                                'type': 'efficient_training',
                                'severity': 'high',
                                'description': 'Very large model (>1B parameters)',
                                'recommendation': 'Implement Parameter-Efficient Fine-Tuning (PEFT)',
                                'estimated_impact': 'Can reduce memory usage by >90% during fine-tuning',
                                'code_example': """
                # Install PEFT library
                # pip install peft
                
                from peft import get_peft_model, LoraConfig, TaskType
                
                # Configure LoRA
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,  # Task type
                    r=8,                         # Rank
                    lora_alpha=32,               # Alpha parameter
                    lora_dropout=0.1,            # Dropout probability
                    target_modules=["q_proj", "v_proj"]  # Target attention projection matrices
                )
                
                # Get PEFT model
                peft_model = get_peft_model(model, peft_config)
                """
                            })
                            
                        # Quantization recommendations for inference
                        recommendations.append({
                            'type': 'quantization',
                            'severity': 'medium',
                            'description': 'Model could benefit from quantization for inference',
                            'recommendation': 'Apply 8-bit or 4-bit quantization for inference',
                            'estimated_impact': '2-4x memory reduction with minimal accuracy impact',
                            'code_example': """
                # 8-bit quantization with bitsandbytes
                # pip install bitsandbytes
                import bitsandbytes as bnb
                
                # Convert linear layers to 8-bit
                model = bnb.nn.convert_to_8bit(model)
                
                # Or with transformers library for specific models:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    'model_name',
                    device_map='auto',
                    load_in_8bit=True
                )
                """
                        })
                        
                        # Multi-GPU recommendations for large models
                        if model_info['parameter_count'] > 5e8 and len(self.hardware_info['gpu']) > 1:  # >500M params and multiple GPUs
                            recommendations.append({
                                'type': 'parallelism',
                                'severity': 'medium',
                                'description': 'Large model could benefit from multi-GPU training',
                                'recommendation': 'Implement model parallelism or distributed training',
                                'estimated_impact': 'Linear scaling with number of GPUs',
                                'code_example': """
                # Using PyTorch DDP
                import torch.distributed as dist
                import torch.multiprocessing as mp
                from torch.nn.parallel import DistributedDataParallel as DDP
                
                def setup(rank, world_size):
                    dist.init_process_group("nccl", rank=rank, world_size=world_size)
                
                def cleanup():
                    dist.destroy_process_group()
                
                def train(rank, world_size):
                    setup(rank, world_size)
                    model = Model().to(rank)
                    ddp_model = DDP(model, device_ids=[rank])
                    # Training loop...
                    cleanup()
                
                world_size = torch.cuda.device_count()
                mp.spawn(train, args=(world_size,), nprocs=world_size)
                """
                            })
                        
                        return recommendations
                        
                    def _profile_tensorflow(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """TensorFlow-specific profiling with detailed kernel analysis"""
                        if not TF_AVAILABLE:
                            raise ImportError("TensorFlow is not available")
                            
                        # Implementation for TensorFlow would follow a similar structure as the PyTorch version
                        # but using TensorFlow-specific profiling tools
                        
                        # For brevity, this is not fully implemented in this example, but would include:
                        # - tf.profiler for low-level kernel analysis
                        # - XLA optimization checks
                        # - TF-specific memory tracking
                        # - TensorFlow-specific recommendations
                        
                        # Placeholder implementation:
                        memory_usage = {'peak': 0, 'active': 0, 'allocated': 0}
                        cpu_utilization = {'mean': 0, 'max': 0}
                        gpu_utilization = {'mean': 0, 'max': 0}
                        operation_stats = {}
                        bottlenecks = []
                        recommendations = []
                        
                        # Create profiling result
                        self.profiling_result = ProfilingResult(
                            execution_time=0.0,
                            memory_usage=memory_usage,
                            cpu_utilization=cpu_utilization,
                            gpu_utilization=gpu_utilization,
                            operation_stats=operation_stats,
                            bottlenecks=bottlenecks,
                            recommendations=recommendations
                        )
                        
                        return self.profiling_result
                
                    def generate_hardware_recommendations(self) -> Dict[str, Any]:
                        """Generate specific hardware recommendations based on profiling results"""
                        if self.profiling_result is None:
                            raise ValueError("Must run profile_execution before generating hardware recommendations")
                            
                        recommendations = {
                            'hardware': {},
                            'cloud_options': {},
                            'estimated_costs': {},
                            'scaling_strategy': ''
                        }
                        
                        parameter_count = self.model_info['parameter_count']
                        memory_usage_gb = self.profiling_result.memory_usage['peak'] / 1024  # Convert MB to GB
                        
                        # Base recommendations on model size, memory usage, and detected bottlenecks
                        bottleneck_types = [b['type'] for b in self.profiling_result.bottlenecks]
                        
                        # Memory-bound models (large transformers, etc)
                        if 'memory' in bottleneck_types or parameter_count > 1e9:
                            recommendations['hardware']['gpu'] = 'NVIDIA A100 (40GB) or newer'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 32):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Consider model parallelism or sharding for distributed training"
                            
                            recommendations['cloud_options']['aws'] = 'p4d.24xlarge or p4de.24xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'A2-highgpu-8g or A3-highgpu-8g instance'
                            recommendations['cloud_options']['azure'] = 'NC A100 v4-series'
                            
                            # Cost estimates (approximate)
                            recommendations['estimated_costs']['aws'] = '$32-40 per hour'
                            recommendations['estimated_costs']['gcp'] = '$30-38 per hour'
                            recommendations['estimated_costs']['azure'] = '$28-35 per hour'
                            
                        # Compute-bound models (CNNs, etc)
                        elif 'operation' in bottleneck_types and not 'memory' in bottleneck_types:
                            recommendations['hardware']['gpu'] = 'NVIDIA V100 or RTX A6000'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 16):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Data parallelism with synchronized gradient updates"
                            
                            recommendations['cloud_options']['aws'] = 'p3.8xlarge or g5.8xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'N1 with V100 GPUs'
                            recommendations['cloud_options']['azure'] = 'NC V3-series'
                            
                            # Cost estimates
                            recommendations['estimated_costs']['aws'] = '$12-25 per hour'
                            recommendations['estimated_costs']['gcp'] = '$10-20 per hour'
                            recommendations['estimated_costs']['azure'] = '$10-18 per hour'
                            
                        # Medium-sized models
                        else:
                            recommendations['hardware']['gpu'] = 'NVIDIA RTX 3090, RTX 4090, or RTX A5000'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 8):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Single GPU training with optimization techniques"
                            
                            recommendations['cloud_options']['aws'] = 'g4dn.xlarge or g5.xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'N1 with T4 GPUs'
                            recommendations['cloud_options']['azure'] = 'NC T4_v3-series'
                            
                            # Cost estimates
                            recommendations['estimated_costs']['aws'] = '$0.5-2.5 per hour'
                            recommendations['estimated_costs']['gcp'] = '$0.45-2.0 per hour'
                            recommendations['estimated_costs']['azure'] = '$0.7-2.2 per hour'
                        
                        # CPU recommendations
                        if self.profiling_result.cpu_utilization['max'] > 80:
                            recommendations['hardware']['cpu'] = '16+ cores with high clock speed'
                            recommendations['hardware']['cpu_notes'] = 'Data preprocessing appears CPU-bound, consider optimizing the data pipeline'
                        
                        return recommendations
                
                
                class NeuralScopeDashboard:
                    """
                    Interactive dashboard for visualizing Neural-Scope results with
                    data quality insights and performance metrics.
                    """
                    def __init__(self, data_quality_results=None, performance_results=None):
                        if not DASH_AVAILABLE:
                            raise ImportError("Plotly Dash not available. Cannot create dashboard.")
                            
                        self.app = Dash(__name__)
                        self.data_quality_results = data_quality_results or []
                        self.performance_results = performance_results or []
                        
                        self._setup_layout()
                        self._setup_callbacks()
                        
                    def _setup_layout(self):
                        """Set up the dashboard layout with tabs and visualization components"""
                        self.app.layout = html.Div([
                            html.H1("Neural-Scope Dashboard"),
                            
                            dcc.Tabs([
                                # Data Quality Tab# filepath: c:\Users\adilm\repositories\Python\neural-scope\aiml_complexity\ml_based_suggestions.py
                                        'recommendation': "Consider feature hashing, target encoding, or grouping rare categories.",
                                        'code_example': """
                # Feature hashing for high cardinality features
                from sklearn.feature_extraction import FeatureHasher
                h = FeatureHasher(n_features=50, input_type='string')
                hashed_features = h.transform(df['{col}'].astype(str)).toarray()
                # Add hashed features to dataframe
                for i in range(50):
                    df[f'{col}_hash_{i}'] = hashed_features[:, i]
                # Then drop original column
                df.drop('{col}', axis=1, inplace=True)
                """
                                    })
                
                        return recommendations
                
                
                class ModelPerformanceProfiler:
                    """
                    Advanced model performance profiling with granular layer-wise analysis,
                    memory tracking, CPU/GPU utilization, and targeted optimization recommendations.
                    """
                    def __init__(self, model, framework: str = None):
                        self.model = model
                        
                        # Auto-detect framework if not specified
                        if framework is None:
                            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                                framework = 'pytorch'
                            elif TF_AVAILABLE and 'tensorflow' in str(type(model)):
                                framework = 'tensorflow' 
                            else:
                                raise ValueError("Could not auto-detect framework. Please specify 'pytorch' or 'tensorflow'.")
                        
                        self.framework = framework.lower()
                        self.profiling_result = None
                        self.model_info = self._extract_model_info()
                        self.hardware_info = self._get_hardware_info()
                
                    def _extract_model_info(self) -> Dict[str, Any]:
                        """Extract detailed model architecture information"""
                        model_info = {
                            'parameter_count': 0,
                            'layer_types': {},
                            'architecture_type': 'unknown'
                        }
                        
                        if self.framework == 'pytorch' and TORCH_AVAILABLE:
                            # Get parameter count
                            model_info['parameter_count'] = sum(p.numel() for p in self.model.parameters())
                            
                            # Analyze layer types
                            for name, module in self.model.named_modules():
                                module_type = type(module).__name__
                                if module_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                                    model_info['layer_types'][module_type] = model_info['layer_types'].get(module_type, 0) + 1
                            
                            # Determine architecture type
                            if any(t in str(self.model) for t in ['Transformer', 'MultiheadAttention']):
                                model_info['architecture_type'] = 'transformer'
                            elif any(t in str(self.model) for t in ['Conv', 'conv']):
                                model_info['architecture_type'] = 'cnn'
                            elif any(t in str(self.model) for t in ['RNN', 'LSTM', 'GRU']):
                                model_info['architecture_type'] = 'rnn'
                            elif any(t in str(self.model) for t in ['Policy', 'Actor', 'Critic']):
                                model_info['architecture_type'] = 'reinforcement_learning'
                            
                        elif self.framework == 'tensorflow' and TF_AVAILABLE:
                            # Get parameter count
                            model_info['parameter_count'] = self.model.count_params()
                            
                            # Analyze layer types
                            for layer in self.model.layers:
                                layer_type = type(layer).__name__
                                model_info['layer_types'][layer_type] = model_info['layer_types'].get(layer_type, 0) + 1
                            
                            # Determine architecture type based on layer composition
                            if any('attention' in l.name.lower() for l in self.model.layers if hasattr(l, 'name')):
                                model_info['architecture_type'] = 'transformer'
                            elif any(isinstance(l, tf_layers.Conv1D) or isinstance(l, tf_layers.Conv2D) for l in self.model.layers):
                                model_info['architecture_type'] = 'cnn'
                            elif any(isinstance(l, tf_layers.LSTM) or isinstance(l, tf_layers.GRU) for l in self.model.layers):
                                model_info['architecture_type'] = 'rnn'
                                
                        return model_info
                
                    def _get_hardware_info(self) -> Dict[str, Any]:
                        """Collect detailed hardware information"""
                        hardware_info = {
                            'cpu': {
                                'count': os.cpu_count(),
                                'model': 'Unknown'
                            },
                            'memory': {
                                'total': round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                                'available': round(psutil.virtual_memory().available / (1024**3), 2)  # GB
                            },
                            'gpu': []
                        }
                        
                        # Try to get CPU model
                        try:
                            import platform
                            if platform.system() == "Windows":
                                import winreg
                                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                                hardware_info['cpu']['model'] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                            elif platform.system() == "Linux":
                                with open('/proc/cpuinfo') as f:
                                    for line in f:
                                        if 'model name' in line:
                                            hardware_info['cpu']['model'] = line.split(':')[1].strip()
                                            break
                        except Exception as e:
                            logger.warning(f"Could not determine CPU model: {e}")
                        
                        # Get GPU information
                        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
                            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                            visible_devices = [int(x) for x in cuda_visible.split(',')] if cuda_visible else list(range(torch.cuda.device_count()))
                            
                            for i in visible_devices:
                                hardware_info['gpu'].append({
                                    'name': torch.cuda.get_device_name(i),
                                    'memory_total': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),  # GB
                                    'compute_capability': f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                                })
                                
                        elif self.framework == 'tensorflow' and TF_AVAILABLE:
                            gpus = tf.config.list_physical_devices('GPU')
                            for i, gpu in enumerate(gpus):
                                gpu_details = tf.config.experimental.get_device_details(gpu)
                                hardware_info['gpu'].append({
                                    'name': gpu_details.get('device_name', f"GPU {i}"),
                                    'compute_capability': str(gpu_details.get('compute_capability', 'Unknown'))
                                })
                                
                        return hardware_info
                
                    def profile_execution(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """
                        Profile model execution with detailed metrics:
                        - Layer-wise execution time
                        - Memory usage patterns
                        - CPU/GPU utilization
                        - Operation-level performance stats
                        """
                        if self.framework == 'pytorch':
                            return self._profile_pytorch(input_data, batch_size, num_steps)
                        elif self.framework == 'tensorflow':
                            return self._profile_tensorflow(input_data, batch_size, num_steps)
                        else:
                            raise ValueError(f"Unsupported framework: {self.framework}")
                
                    def _profile_pytorch(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """PyTorch-specific profiling with detailed layer and kernel analysis"""
                        if not TORCH_AVAILABLE:
                            raise ImportError("PyTorch is not available")
                            
                        # Ensure model is in evaluation mode
                        self.model.eval()
                        
                        # Get device
                        device = next(self.model.parameters()).device
                        
                        # Prepare data loader if necessary
                        if hasattr(input_data, '__getitem__') and not isinstance(input_data, torch.Tensor):
                            if batch_size is None:
                                batch_size = 16  # Default batch size
                            data_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size)
                            sample_input = next(iter(data_loader))
                            if isinstance(sample_input, (list, tuple)):
                                # Assume first element is input if it's a tuple/list (input, target)
                                sample_input = sample_input[0].to(device)
                            else:
                                sample_input = sample_input.to(device)
                        else:
                            # Use provided tensor directly
                            sample_input = input_data
                            if not isinstance(sample_input, torch.Tensor):
                                sample_input = torch.tensor(sample_input).to(device)
                        
                        # Initialize metrics containers
                        memory_usage = {'peak': 0, 'active': 0, 'allocated': 0}
                        cpu_utilization = {'mean': 0, 'max': 0}
                        gpu_utilization = {'mean': 0, 'max': 0, 'memory_usage_percent': 0}
                        operation_stats = {}
                        bottlenecks = []
                        
                        # Warmup run
                        with torch.no_grad():
                            self.model(sample_input)
                        
                        # Record CPU usage
                        cpu_percent_values = []
                        start_time = time.time()
                        
                        # Use PyTorch's profiler for detailed analysis
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True
                        ) as prof:
                            for _ in range(num_steps):
                                with record_function("model_inference"):
                                    with torch.no_grad():
                                        self.model(sample_input)
                                
                                # Record CPU usage during execution
                                cpu_percent = psutil.cpu_percent(interval=None)
                                cpu_percent_values.append(cpu_percent)
                                
                                # Record GPU utilization if available
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    memory_usage['allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
                                    memory_usage['peak'] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                                    memory_usage['reserved'] = torch.cuda.memory_reserved() / (1024**2)  # MB
                        
                        execution_time = (time.time() - start_time) / num_steps
                        
                        # Process CPU utilization
                        cpu_utilization['mean'] = sum(cpu_percent_values) / len(cpu_percent_values)
                        cpu_utilization['max'] = max(cpu_percent_values)
                        
                        # Process profiling results
                        events = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
                        
                        # Extract key metrics from profiler events
                        operation_stats = {}
                        for evt in prof.key_averages():
                            if evt.key not in operation_stats:
                                operation_stats[evt.key] = {
                                    'cpu_time': evt.cpu_time_total / 1000,  # convert to ms
                                    'cuda_time': evt.cuda_time_total / 1000 if hasattr(evt, 'cuda_time_total') else 0,
                                    'self_cpu_time': evt.self_cpu_time_total / 1000,
                                    'self_cuda_time': evt.self_cuda_time_total / 1000 if hasattr(evt, 'self_cuda_time_total') else 0,
                                    'cpu_memory_usage': evt.cpu_memory_usage / (1024**2) if hasattr(evt, 'cpu_memory_usage') else 0,  # MB
                                    'cuda_memory_usage': evt.cuda_memory_usage / (1024**2) if hasattr(evt, 'cuda_memory_usage') else 0,  # MB
                                    'occurrences': 1
                                }
                            else:
                                operation_stats[evt.key]['occurrences'] += 1
                        
                        # Identify bottlenecks
                        bottlenecks = self._identify_pytorch_bottlenecks(operation_stats, memory_usage)
                        
                        # Generate recommendations
                        recommendations = self._generate_pytorch_recommendations(
                            bottlenecks, 
                            operation_stats, 
                            memory_usage,
                            self.model_info,
                            batch_size
                        )
                        
                        # Create profiling result
                        self.profiling_result = ProfilingResult(
                            execution_time=execution_time,
                            memory_usage=memory_usage,
                            cpu_utilization=cpu_utilization,
                            gpu_utilization=gpu_utilization if torch.cuda.is_available() else None,
                            operation_stats=operation_stats,
                            bottlenecks=bottlenecks,
                            recommendations=recommendations
                        )
                        
                        return self.profiling_result
                
                    def _identify_pytorch_bottlenecks(self, operation_stats, memory_usage) -> List[Dict[str, Any]]:
                        """Identify performance bottlenecks in PyTorch model execution"""
                        bottlenecks = []
                        
                        # Sort operations by CUDA time (if available) or CPU time
                        sorted_ops = sorted(
                            operation_stats.items(), 
                            key=lambda x: x[1]['cuda_time'] if x[1]['cuda_time'] > 0 else x[1]['cpu_time'],
                            reverse=True
                        )
                        
                        # Top-3 most time-consuming operations
                        for i, (op_name, stats) in enumerate(sorted_ops[:3]):
                            if i == 0:  # Most expensive operation
                                severity = 'high'
                            elif i == 1:
                                severity = 'medium'
                            else:
                                severity = 'low'
                                
                            cuda_time = stats['cuda_time']
                            cpu_time = stats['cpu_time']
                            
                            bottleneck = {
                                'type': 'operation',
                                'name': op_name,
                                'severity': severity,
                                'time_ms': cuda_time if cuda_time > 0 else cpu_time,
                                'device': 'gpu' if cuda_time > 0 else 'cpu',
                                'memory_mb': stats['cuda_memory_usage'] if cuda_time > 0 else stats['cpu_memory_usage']
                            }
                            bottlenecks.append(bottleneck)
                        
                        # Check for memory-related bottlenecks
                        if memory_usage['peak'] > 0.9 * self.hardware_info['gpu'][0]['memory_total'] * 1024:  # Convert GB to MB
                            bottlenecks.append({
                                'type': 'memory',
                                'name': 'GPU Memory Near Capacity',
                                'severity': 'high',
                                'memory_mb': memory_usage['peak'],
                                'details': 'Model is using >90% of available GPU memory'
                            })
                        
                        # Check for operation pattern bottlenecks (e.g., excessive memory copying)
                        mem_copies = [op for op, stats in operation_stats.items() if 'memcpy' in op.lower()]
                        if mem_copies and sum(operation_stats[op]['cuda_time'] for op in mem_copies) > 0.2 * sorted_ops[0][1]['cuda_time']:
                            bottlenecks.append({
                                'type': 'data_movement',
                                'name': 'Excessive Host-Device Memory Transfers',
                                'severity': 'high',
                                'time_ms': sum(operation_stats[op]['cuda_time'] for op in mem_copies),
                                'details': 'Significant time spent on memory transfers between CPU and GPU'
                            })
                            
                        return bottlenecks
                
                    def _generate_pytorch_recommendations(self, bottlenecks, operation_stats, 
                                                         memory_usage, model_info, batch_size) -> List[Dict[str, Any]]:
                        """Generate specific optimization recommendations based on profiling results"""
                        recommendations = []
                        
                        # Check for memory-intensive operations
                        if any(b['type'] == 'memory' for b in bottlenecks):
                            # Recommend gradient checkpointing for transformer models
                            if model_info['architecture_type'] == 'transformer':
                                recommendations.append({
                                    'type': 'memory_optimization',
                                    'severity': 'high',
                                    'description': 'Model is using excessive GPU memory',
                                    'recommendation': 'Implement gradient checkpointing to reduce memory usage',
                                    'estimated_impact': 'Can reduce memory usage by 30-40% with minimal performance impact',
                                    'code_example': """
                # Enable gradient checkpointing
                model.gradient_checkpointing_enable()
                
                # Or manually for specific modules:
                from torch.utils.checkpoint import checkpoint
                def forward(self, *args, **kwargs):
                    # Use checkpoint for transformer layers
                    output = checkpoint(self.transformer_layer, *args, **kwargs)
                    output = self.final_layer(output)  # Non-checkpointed operations
                    return output
                """
                                })
                                
                            # Recommend smaller batch size if memory is the primary issue
                            recommendations.append({
                                'type': 'training_parameter',
                                'severity': 'medium',
                                'description': 'Model exceeds GPU memory capacity',
                                'recommendation': f'Reduce batch size (currently {batch_size})',
                                'estimated_impact': 'Linear reduction in memory usage',
                                'code_example': f"trainer.train(batch_size={max(1, batch_size // 2)})"
                            })
                            
                            # Recommend mixed precision training
                            recommendations.append({
                                'type': 'training_technique',
                                'severity': 'high',
                                'description': 'Model using full precision (FP32) computation',
                                'recommendation': 'Enable mixed precision training (FP16/BF16)',
                                'estimated_impact': 'Can reduce memory usage by up to 50% with potential speedup',
                                'code_example': """
                # For PyTorch native mixed precision
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                
                # In training loop:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                """
                            })
                            
                        # Check for attention-related bottlenecks in transformer models
                        if model_info['architecture_type'] == 'transformer' and any('attention' in b['name'].lower() for b in bottlenecks):
                            recommendations.append({
                                'type': 'algorithm_optimization',
                                'severity': 'high',
                                'description': 'Attention mechanism is a performance bottleneck',
                                'recommendation': 'Implement FlashAttention for faster transformer computation',
                                'estimated_impact': 'Can improve attention computation speed by 2-4x',
                                'code_example': """
                # Install flash-attn package
                # pip install flash-attn
                
                from flash_attn import flash_attn_func
                
                # Replace standard attention with flash attention
                def flash_attention_forward(q, k, v, mask=None):
                    # q, k, v: (batch_size, seq_len, num_heads, head_dim)
                    batch_size, seq_len, num_heads, head_dim = q.shape
                    q, k, v = [x.reshape(-1, seq_len, num_heads, head_dim) for x in (q, k, v)]
                    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None)
                    return out.reshape(batch_size, seq_len, num_heads * head_dim)
                """
                            })
                            
                        # Check for data transfer bottlenecks
                        if any(b['type'] == 'data_movement' for b in bottlenecks):
                            recommendations.append({
                                'type': 'data_pipeline',
                                'severity': 'high',
                                'description': 'Excessive CPU-GPU data transfers',
                                'recommendation': 'Use pinned memory and non-blocking transfers',
                                'estimated_impact': 'Can reduce data transfer overhead by 20-30%',
                                'code_example': """
                # Use pinned memory in DataLoader
                train_loader = DataLoader(dataset, batch_size=32, pin_memory=True)
                
                # Use non-blocking transfers
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                """
                            })
                            
                        # Large model optimization suggestions
                        if model_info['parameter_count'] > 1e9:  # >1B parameters
                            recommendations.append({
                                'type': 'efficient_training',
                                'severity': 'high',
                                'description': 'Very large model (>1B parameters)',
                                'recommendation': 'Implement Parameter-Efficient Fine-Tuning (PEFT)',
                                'estimated_impact': 'Can reduce memory usage by >90% during fine-tuning',
                                'code_example': """
                # Install PEFT library
                # pip install peft
                
                from peft import get_peft_model, LoraConfig, TaskType
                
                # Configure LoRA
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,  # Task type
                    r=8,                         # Rank
                    lora_alpha=32,               # Alpha parameter
                    lora_dropout=0.1,            # Dropout probability
                    target_modules=["q_proj", "v_proj"]  # Target attention projection matrices
                )
                
                # Get PEFT model
                peft_model = get_peft_model(model, peft_config)
                """
                            })
                            
                        # Quantization recommendations for inference
                        recommendations.append({
                            'type': 'quantization',
                            'severity': 'medium',
                            'description': 'Model could benefit from quantization for inference',
                            'recommendation': 'Apply 8-bit or 4-bit quantization for inference',
                            'estimated_impact': '2-4x memory reduction with minimal accuracy impact',
                            'code_example': """
                # 8-bit quantization with bitsandbytes
                # pip install bitsandbytes
                import bitsandbytes as bnb
                
                # Convert linear layers to 8-bit
                model = bnb.nn.convert_to_8bit(model)
                
                # Or with transformers library for specific models:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    'model_name',
                    device_map='auto',
                    load_in_8bit=True
                )
                """
                        })
                        
                        # Multi-GPU recommendations for large models
                        if model_info['parameter_count'] > 5e8 and len(self.hardware_info['gpu']) > 1:  # >500M params and multiple GPUs
                            recommendations.append({
                                'type': 'parallelism',
                                'severity': 'medium',
                                'description': 'Large model could benefit from multi-GPU training',
                                'recommendation': 'Implement model parallelism or distributed training',
                                'estimated_impact': 'Linear scaling with number of GPUs',
                                'code_example': """
                # Using PyTorch DDP
                import torch.distributed as dist
                import torch.multiprocessing as mp
                from torch.nn.parallel import DistributedDataParallel as DDP
                
                def setup(rank, world_size):
                    dist.init_process_group("nccl", rank=rank, world_size=world_size)
                
                def cleanup():
                    dist.destroy_process_group()
                
                def train(rank, world_size):
                    setup(rank, world_size)
                    model = Model().to(rank)
                    ddp_model = DDP(model, device_ids=[rank])
                    # Training loop...
                    cleanup()
                
                world_size = torch.cuda.device_count()
                mp.spawn(train, args=(world_size,), nprocs=world_size)
                """
                            })
                        
                        return recommendations
                        
                    def _profile_tensorflow(self, input_data, batch_size: int = None, num_steps: int = 10) -> ProfilingResult:
                        """TensorFlow-specific profiling with detailed kernel analysis"""
                        if not TF_AVAILABLE:
                            raise ImportError("TensorFlow is not available")
                            
                        # Implementation for TensorFlow would follow a similar structure as the PyTorch version
                        # but using TensorFlow-specific profiling tools
                        
                        # For brevity, this is not fully implemented in this example, but would include:
                        # - tf.profiler for low-level kernel analysis
                        # - XLA optimization checks
                        # - TF-specific memory tracking
                        # - TensorFlow-specific recommendations
                        
                        # Placeholder implementation:
                        memory_usage = {'peak': 0, 'active': 0, 'allocated': 0}
                        cpu_utilization = {'mean': 0, 'max': 0}
                        gpu_utilization = {'mean': 0, 'max': 0}
                        operation_stats = {}
                        bottlenecks = []
                        recommendations = []
                        
                        # Create profiling result
                        self.profiling_result = ProfilingResult(
                            execution_time=0.0,
                            memory_usage=memory_usage,
                            cpu_utilization=cpu_utilization,
                            gpu_utilization=gpu_utilization,
                            operation_stats=operation_stats,
                            bottlenecks=bottlenecks,
                            recommendations=recommendations
                        )
                        
                        return self.profiling_result
                
                    def generate_hardware_recommendations(self) -> Dict[str, Any]:
                        """Generate specific hardware recommendations based on profiling results"""
                        if self.profiling_result is None:
                            raise ValueError("Must run profile_execution before generating hardware recommendations")
                            
                        recommendations = {
                            'hardware': {},
                            'cloud_options': {},
                            'estimated_costs': {},
                            'scaling_strategy': ''
                        }
                        
                        parameter_count = self.model_info['parameter_count']
                        memory_usage_gb = self.profiling_result.memory_usage['peak'] / 1024  # Convert MB to GB
                        
                        # Base recommendations on model size, memory usage, and detected bottlenecks
                        bottleneck_types = [b['type'] for b in self.profiling_result.bottlenecks]
                        
                        # Memory-bound models (large transformers, etc)
                        if 'memory' in bottleneck_types or parameter_count > 1e9:
                            recommendations['hardware']['gpu'] = 'NVIDIA A100 (40GB) or newer'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 32):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Consider model parallelism or sharding for distributed training"
                            
                            recommendations['cloud_options']['aws'] = 'p4d.24xlarge or p4de.24xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'A2-highgpu-8g or A3-highgpu-8g instance'
                            recommendations['cloud_options']['azure'] = 'NC A100 v4-series'
                            
                            # Cost estimates (approximate)
                            recommendations['estimated_costs']['aws'] = '$32-40 per hour'
                            recommendations['estimated_costs']['gcp'] = '$30-38 per hour'
                            recommendations['estimated_costs']['azure'] = '$28-35 per hour'
                            
                        # Compute-bound models (CNNs, etc)
                        elif 'operation' in bottleneck_types and not 'memory' in bottleneck_types:
                            recommendations['hardware']['gpu'] = 'NVIDIA V100 or RTX A6000'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 16):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Data parallelism with synchronized gradient updates"
                            
                            recommendations['cloud_options']['aws'] = 'p3.8xlarge or g5.8xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'N1 with V100 GPUs'
                            recommendations['cloud_options']['azure'] = 'NC V3-series'
                            
                            # Cost estimates
                            recommendations['estimated_costs']['aws'] = '$12-25 per hour'
                            recommendations['estimated_costs']['gcp'] = '$10-20 per hour'
                            recommendations['estimated_costs']['azure'] = '$10-18 per hour'
                            
                        # Medium-sized models
                        else:
                            recommendations['hardware']['gpu'] = 'NVIDIA RTX 3090, RTX 4090, or RTX A5000'
                            recommendations['hardware']['memory'] = f"Minimum {max(memory_usage_gb * 1.5, 8):.0f} GB GPU memory"
                            recommendations['scaling_strategy'] = "Single GPU training with optimization techniques"
                            
                            recommendations['cloud_options']['aws'] = 'g4dn.xlarge or g5.xlarge instance'
                            recommendations['cloud_options']['gcp'] = 'N1 with T4 GPUs'
                            recommendations['cloud_options']['azure'] = 'NC T4_v3-series'
                            
                            # Cost estimates
                            recommendations['estimated_costs']['aws'] = '$0.5-2.5 per hour'
                            recommendations['estimated_costs']['gcp'] = '$0.45-2.0 per hour'
                            recommendations['estimated_costs']['azure'] = '$0.7-2.2 per hour'
                        
                        # CPU recommendations
                        if self.profiling_result.cpu_utilization['max'] > 80:
                            recommendations['hardware']['cpu'] = '16+ cores with high clock speed'
                            recommendations['hardware']['cpu_notes'] = 'Data preprocessing appears CPU-bound, consider optimizing the data pipeline'
                        
                        return recommendations
                
                
                class NeuralScopeDashboard:
                    """
                    Interactive dashboard for visualizing Neural-Scope results with
                    data quality insights and performance metrics.
                    """
                    def __init__(self, data_quality_results=None, performance_results=None):
                        if not DASH_AVAILABLE:
                            raise ImportError("Plotly Dash not available. Cannot create dashboard.")
                            
                        self.app = Dash(__name__)
                        self.data_quality_results = data_quality_results or []
                        self.performance_results = performance_results or []
                        
                        self._setup_layout()
                        self._setup_callbacks()
                        
                    def _setup_layout(self):
                        """Set up the dashboard layout with tabs and visualization components"""
                        self.app.layout = html.Div([
                            html.H1("Neural-Scope Dashboard"),
                            
                            # Data Quality Tab
                            
                                            # Data Quality Tab
                dcc.Tab(label="Data Quality", children=[
                    html.Div([
                        html.H3("Dataset Quality Insights"),
                        html.Div([
                            dcc.Dropdown(
                                id='dataset-selector',
                                options=[{'label': res.dataset_name, 'value': i} 
                                        for i, res in enumerate(self.data_quality_results)],
                                placeholder="Select a dataset",
                            ),
                        ], style={'width': '30%', 'margin-bottom': '20px'}),
                        
                        # Data Quality Score Card
                        html.Div([
                            html.Div(id='quality-score-card', className='metric-card', children=[
                                html.H4("Data Quality Score"),
                                html.Div(id='quality-score-value', className='metric-value'),
                                html.Div(id='quality-score-gauge')
                            ]),
                            html.Div(id='missing-values-card', className='metric-card', children=[
                                html.H4("Missing Values"),
                                html.Div(id='missing-values-value', className='metric-value')
                            ]),
                            html.Div(id='bias-score-card', className='metric-card', children=[
                                html.H4("Bias Score"),
                                html.Div(id='bias-score-value', className='metric-value')
                            ]),
                        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
                        
                        # Data Distribution & Issues
                        html.Div([
                            html.Div([
                                html.H4("Feature Statistics"),
                                dcc.Graph(id='feature-distribution-graph')
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H4("Quality Issues"),
                                html.Div(id='quality-issues-table')
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justify-content': 'space-between'}),
                        
                        # Recommendations Section
                        html.Div([
                            html.H3("Quality Improvement Recommendations"),
                            html.Div(id='quality-recommendations')
                        ], style={'margin-top': '30px'})
                    ])
                ]),
                
                # Model Performance Tab
                dcc.Tab(label="Model Performance", children=[
                    html.Div([
                        html.H3("Model Performance Analysis"),
                        html.Div([
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': f"{res.model_name} ({res.framework})", 'value': i} 
                                        for i, res in enumerate(self.performance_results)],
                                placeholder="Select a model",
                            ),
                        ], style={'width': '30%', 'margin-bottom': '20px'}),
                        
                        # Performance Metrics
                        html.Div([
                            html.Div(id='execution-time-card', className='metric-card', children=[
                                html.H4("Execution Time"),
                                html.Div(id='execution-time-value', className='metric-value')
                            ]),
                            html.Div(id='memory-usage-card', className='metric-card', children=[
                                html.H4("Memory Usage"),
                                html.Div(id='memory-usage-value', className='metric-value')
                            ]),
                            html.Div(id='cpu-usage-card', className='metric-card', children=[
                                html.H4("CPU Utilization"),
                                html.Div(id='cpu-usage-value', className='metric-value')
                            ]),
                        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
                        
                        # Operation Stats & Bottlenecks
                        html.Div([
                            html.Div([
                                html.H4("Operation Breakdown"),
                                dcc.Graph(id='operation-stats-graph')
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H4("Bottlenecks"),
                                html.Div(id='bottlenecks-table')
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justify-content': 'space-between'}),
                        
                        # Optimization Recommendations
                        html.Div([
                            html.H3("Optimization Recommendations"),
                            html.Div(id='optimization-recommendations')
                        ], style={'margin-top': '30px'}),
                        
                        # Hardware Recommendations
                        html.Div([
                            html.H3("Hardware Recommendations"),
                            html.Div(id='hardware-recommendations')
                        ], style={'margin-top': '30px'})
                    ])
                ]),
                
                # Comparison Tab
                dcc.Tab(label="Historical Comparison", children=[
                    html.Div([
                        html.H3("Performance Over Time"),
                        html.Div([
                            dcc.Dropdown(
                                id='model-history-selector',
                                options=[{'label': f"{res.model_name}", 'value': res.model_name} 
                                        for res in self.performance_results],
                                placeholder="Select model for historical view",
                            ),
                        ], style={'width': '30%', 'margin-bottom': '20px'}),
                        
                        dcc.Graph(id='historical-performance-graph'),
                        
                        html.Div([
                            html.H4("Optimization Progress"),
                            html.Div(id='optimization-progress-table')
                        ], style={'margin-top': '20px'})
                    ])
                ])
            ])
        ])

    def _setup_callbacks(self):
        """Set up interactive dashboard callbacks"""
        
        # Data Quality Tab callbacks
        @self.app.callback(
            [Output('quality-score-value', 'children'),
             Output('quality-score-gauge', 'children'),
             Output('missing-values-value', 'children'),
             Output('bias-score-value', 'children'),
             Output('feature-distribution-graph', 'figure'),
             Output('quality-issues-table', 'children'),
             Output('quality-recommendations', 'children')],
            [Input('dataset-selector', 'value')]
        )
        def update_data_quality_view(dataset_index):
            if dataset_index is None or len(self.data_quality_results) == 0:
                return "N/A", no_data_gauge(), "N/A", "N/A", empty_figure(), "No data selected", "No data selected"
                
            # Get selected dataset results
            result = self.data_quality_results[dataset_index]
            
            # Quality score with gauge
            quality_score = f"{result.data_purity:.2f}"
            gauge_figure = create_gauge_figure(result.data_purity, 
                                              title="Data Quality", 
                                              color_ranges=[(0, 0.5, "red"), 
                                                           (0.5, 0.8, "orange"), 
                                                           (0.8, 1, "green")])
            
            # Missing values summary
            missing_values_pct = sum(result.missing_values.values()) / len(result.missing_values) if result.missing_values else 0
            missing_values_text = f"{missing_values_pct:.2f}% (Columns: {len(result.missing_values)})"
            
            # Bias score summary
            bias_score = result.bias_metrics.get('intersectional_bias_score', 0)
            bias_text = f"{bias_score:.2f}" if bias_score > 0 else "Not detected"
            
            # Feature distribution graph
            fig = create_feature_distribution_figure(result)
            
            # Quality issues table
            issues_table = create_quality_issues_table(result)
            
            # Recommendations
            recommendations_ui = create_recommendations_ui(result.recommendations)
            
            return quality_score, dcc.Graph(figure=gauge_figure), missing_values_text, bias_text, fig, issues_table, recommendations_ui
            
        # Model Performance Tab callbacks
        @self.app.callback(
            [Output('execution-time-value', 'children'),
             Output('memory-usage-value', 'children'),
             Output('cpu-usage-value', 'children'),
             Output('operation-stats-graph', 'figure'),
             Output('bottlenecks-table', 'children'),
             Output('optimization-recommendations', 'children'),
             Output('hardware-recommendations', 'children')],
            [Input('model-selector', 'value')]
        )
        def update_model_performance_view(model_index):
            if model_index is None or len(self.performance_results) == 0:
                return "N/A", "N/A", "N/A", empty_figure(), "No data selected", "No data selected", "No data selected"
                
            # Get selected model results
            result = self.performance_results[model_index]
            
            # Basic metrics
            execution_time = f"{result.execution_time:.4f} seconds"
            memory_usage = f"{result.memory_usage.get('peak', 0):.2f} MB"
            cpu_usage = f"{result.cpu_utilization.get('mean', 0):.1f}% (Max: {result.cpu_utilization.get('max', 0):.1f}%)"
            
            # Operation stats graph
            op_fig = create_operation_stats_figure(result.operation_stats)
            
            # Bottlenecks table
            bottlenecks_table = create_bottlenecks_table(result.bottlenecks)
            
            # Recommendations UI
            recommendations_ui = create_recommendations_ui(result.recommendations)
            
            # Hardware recommendations
            hw_recommendations = create_hardware_recommendations_ui(result.model_info, result.bottlenecks)
            
            return execution_time, memory_usage, cpu_usage, op_fig, bottlenecks_table, recommendations_ui, hw_recommendations
            
        # Historical Comparison Tab callbacks
        @self.app.callback(
            [Output('historical-performance-graph', 'figure'),
             Output('optimization-progress-table', 'children')],
            [Input('model-history-selector', 'value')]
        )
        def update_historical_comparison(model_name):
            if not model_name:
                return empty_figure(), "No model selected"
                
            # Filter performance results for this model
            model_history = [res for res in self.performance_results if res.model_name == model_name]
            
            if not model_history:
                return empty_figure(), "No historical data available"
                
            # Create historical performance graph
            history_fig = create_historical_performance_figure(model_history)
            
            # Create optimization progress table
            progress_table = create_optimization_progress_table(model_history)
            
            return history_fig, progress_table
    
    def _create_gauge_figure(self, value, title, color_ranges):
        """Create a gauge chart for metrics"""
        # Implementation details omitted for brevity
        pass
        
    def _create_feature_distribution_figure(self, quality_result):
        """Create distribution visualization for feature statistics"""
        # Implementation details omitted for brevity
        pass
        
    def _create_operation_stats_figure(self, operation_stats):
        """Create visualization for operation statistics"""
        # Implementation details omitted for brevity
        pass
        
    def _create_quality_issues_table(self, quality_result):
        """Generate HTML table showing data quality issues"""
        # Implementation details omitted for brevity
        pass
        
    def _create_bottlenecks_table(self, bottlenecks):
        """Generate HTML table showing performance bottlenecks"""
        # Implementation details omitted for brevity
        pass
        
    def _create_recommendations_ui(self, recommendations):
        """Create expandable UI components for recommendations"""
        # Implementation details omitted for brevity
        pass
        
    def _create_hardware_recommendations_ui(self, model_info, bottlenecks):
        """Create UI for hardware recommendations"""
        # Implementation details omitted for brevity
        pass
        
    def _create_historical_performance_figure(self, model_history):
        """Create line chart showing performance metrics over time"""
        # Implementation details omitted for brevity
        pass
        
    def _create_optimization_progress_table(self, model_history):
        """Create table showing optimization improvements over time"""
        # Implementation details omitted for brevity
        pass
        
    def run_server(self, debug=False, port=8050):
        """Launch the dashboard server"""
        self.app.run_server(debug=debug, port=port)
        
    def _empty_figure(self):
        """Create an empty placeholder figure"""
        return px.scatter(title="No data available")


class NeuralScopeIntegration:
    """
    Integration class that ties together all components of Neural-Scope into a cohesive system.
    Provides high-level API for end users to analyze and optimize their ML workflows.
    """
    def __init__(self, db_config=None):
        """
        Initialize Neural-Scope with optional database configuration
        
        Args:
            db_config: Optional dictionary with PostgreSQL connection parameters
        """
        self.db = None
        if db_config:
            self.db = PostgresStorage(**db_config)
            self.db.connect()
            self.db.setup_tables()
            
        self.data_quality_results = []
        self.performance_results = []
        
    def analyze_dataset(self, dataframe, dataset_name, sensitive_features=None, label_column=None):
        """
        Perform comprehensive data quality analysis on a dataset
        
        Args:
            dataframe: pandas DataFrame to analyze
            dataset_name: Name identifier for the dataset
            sensitive_features: List of column names to treat as sensitive/protected attributes
            label_column: Name of the target/label column
            
        Returns:
            DataQualityResult object with analysis and recommendations
        """
        analyzer = DataQualityAnalyzer(sensitive_attributes=sensitive_features)
        metrics = analyzer.analyze_dataset(dataframe, label_column)
        recommendations = analyzer.generate_recommendations(metrics)
        
        # Create result object
        result = DataQualityResult(
            dataset_name=dataset_name,
            feature_stats={col: dataframe[col].describe().to_dict() for col in dataframe.select_dtypes(include='number').columns},
            missing_values={col: metrics.missing_values_pct[col] / 100 for col in metrics.missing_values_pct},
            outliers={col: dataframe.index[dataframe[col] > dataframe[col].quantile(0.75) + 1.5 * (dataframe[col].quantile(0.75) - dataframe[col].quantile(0.25))].tolist()
                    for col in dataframe.select_dtypes(include='number').columns},
            bias_metrics={'intersectional_bias_score': metrics.intersectional_bias_score,
                         'class_imbalance_score': metrics.class_imbalance_score},
            fairness_metrics=metrics.fairness_metrics,
            data_purity=metrics.quality_score,
            recommendations=recommendations
        )
        
        self.data_quality_results.append(result)
        
        # Store in database if configured
        if self.db:
            self.db.store_data_quality_result(result)
            
        return result
    
    def profile_model(self, model, input_data, batch_size=None, framework=None, num_steps=10):
        """
        Perform comprehensive performance profiling on a model
        
        Args:
            model: PyTorch or TensorFlow model to profile
            input_data: Sample input data for profiling (tensor, dataset, or numpy array)
            batch_size: Batch size to use for profiling
            framework: 'pytorch' or 'tensorflow' (auto-detected if not specified)
            num_steps: Number of profiling steps to run for averaging
            
        Returns:
            ProfilingResult object with analysis and recommendations
        """
        profiler = ModelPerformanceProfiler(model, framework)
        result = profiler.profile_execution(input_data, batch_size, num_steps)
        hw_recommendations = profiler.generate_hardware_recommendations()
        
        self.performance_results.append(result)
        
        # Store in database if configured
        if self.db:
            model_name = model.__class__.__name__
            model_info = profiler.model_info
            hardware_info = profiler.hardware_info
            self.db.store_model_performance(model_name, framework, batch_size, 
                                          result, hardware_info, model_info)
            self.db.store_hardware_recommendations(len(self.performance_results), hw_recommendations)
            
        return result
    
    def launch_dashboard(self, port=8050):
        """
        Launch the interactive dashboard to explore data quality and performance results
        
        Args:
            port: Port to use for the web server
        """
        dashboard = NeuralScopeDashboard(self.data_quality_results, self.performance_results)
        dashboard.run_server(port=port)
        
    def get_historical_performance(self, model_name, limit=10):
        """
        Retrieve historical performance data for a model
        
        Args:
            model_name: Name of the model to query
            limit: Maximum number of historical entries to retrieve
            
        Returns:
            List of performance entries with timestamps
        """
        if not self.db:
            return []
            
        return self.db.get_historical_performance(model_name, limit)
    
    def close(self):
        """Clean up resources and close database connections"""
        if self.db:
            self.db.disconnect()


# Example usage 
def neural_scope_example():
    """Example demonstrating the Neural-Scope workflow"""
    # Initialize Neural-Scope
    ns = NeuralScopeIntegration()
    
    # Analyze a dataset
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    quality_result = ns.analyze_dataset(
        df, 
        dataset_name="Breast Cancer Dataset",
        sensitive_features=None,  # No sensitive features in this dataset
        label_column="target"
    )
    
    print(f"Data Quality Score: {quality_result.data_purity:.2f}")
    print(f"Found {len(quality_result.recommendations)} data quality recommendations")
    
    # Profile a model (if PyTorch is available)
    if TORCH_AVAILABLE:
        import torch
        import torch.nn as nn
        
        # Create a simple model for demo
        model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        inputs = torch.randn(100, 30)
        
        profile_result = ns.profile_model(
            model,
            inputs,
            batch_size=32
        )
        
        print(f"Model execution time: {profile_result.execution_time:.4f} seconds")
        print(f"Peak memory usage: {profile_result.memory_usage['peak']:.2f} MB")
        print(f"Found {len(profile_result.bottlenecks)} performance bottlenecks")
        print(f"Generated {len(profile_result.recommendations)} optimization recommendations")
    
    # Launch the interactive dashboard
    ns.launch_dashboard()
    
    # Clean up
    ns.close()


if __name__ == "__main__":
    neural_scope_example()
                            
                                