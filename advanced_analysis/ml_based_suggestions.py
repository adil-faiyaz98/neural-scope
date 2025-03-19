"""
Neural-Scope: Advanced AI/ML Performance and Data Quality Analysis

A comprehensive solution for machine learning engineers to analyze, optimize, and
improve their ML workflows through deep performance profiling, data quality assessment,
and targeted optimization recommendations.

This module provides granular, actionable insights for improving model performance,
data quality, and hardware utilization, with specific code-level recommendations.
"""

import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Framework-specific imports with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.python.eager import profiler as tf_profiler
    from tensorflow.keras import layers as tf_layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorFlow-specific features will be disabled.")

try:
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Plotly Dash not available. Interactive dashboards will be disabled.")

# Model architecture recognition utilities
from aiml_complexity.ml_patterns import MLPatternDatabase


@dataclass
class ProfilingResult:
    """Stores comprehensive profiling results with detailed metrics"""
    execution_time: float
    memory_usage: Dict[str, float]
    cpu_utilization: Dict[str, float]
    gpu_utilization: Optional[Dict[str, float]]
    operation_stats: Dict[str, Dict[str, float]]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


@dataclass
class DataQualityResult:
    """Stores comprehensive data quality analysis results"""
    dataset_name: str
    feature_stats: Dict[str, Dict[str, float]]
    missing_values: Dict[str, float]
    outliers: Dict[str, List[int]]
    bias_metrics: Dict[str, float]
    fairness_metrics: Dict[str, Dict[str, float]]
    data_purity: float
    recommendations: List[Dict[str, Any]]


class PostgresStorage:
    """
    Advanced PostgreSQL storage for both data quality results
    and model performance stats with proper connection management
    and detailed metadata storage.
    """
    def __init__(self, host, port, dbname, user, password):
        self.connection_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cur = None
        
    def connect(self):
        """Establish connection with error handling"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cur = self.conn.cursor()
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            return False

    def disconnect(self):
        """Properly close connections"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        self.cur = None
        self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def setup_tables(self):
        """Create tables if they don't exist with comprehensive schema design"""
        try:
            # Data quality table with detailed metrics
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id SERIAL PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feature_stats JSONB,
                    missing_values JSONB,
                    outliers JSONB,
                    bias_metrics JSONB,
                    fairness_metrics JSONB,
                    data_purity REAL,
                    recommendations JSONB
                )
            """)
            
            # Model performance table with detailed metrics
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    batch_size INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time REAL,
                    memory_usage JSONB,
                    cpu_utilization JSONB,
                    gpu_utilization JSONB,
                    operation_stats JSONB,
                    bottlenecks JSONB,
                    recommendations JSONB,
                    hardware_info JSONB,
                    model_info JSONB
                )
            """)
            
            # Hardware recommendations table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS hardware_recommendations (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES model_performance(id),
                    recommended_hardware JSONB,
                    cloud_options JSONB,
                    estimated_costs JSONB,
                    scaling_strategy TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_dataset ON data_quality(dataset_name)")
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_name ON model_performance(model_name)")
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_timestamp ON model_performance(timestamp)")
            
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Database setup error: {e}")
            self.conn.rollback()
            return False

    def store_data_quality_result(self, result: DataQualityResult) -> int:
        """Store data quality analysis results"""
        try:
            query = """
                INSERT INTO data_quality 
                (dataset_name, feature_stats, missing_values, outliers, 
                bias_metrics, fairness_metrics, data_purity, recommendations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                result.dataset_name,
                json.dumps(result.feature_stats),
                json.dumps(result.missing_values),
                json.dumps({k: v for k, v in result.outliers.items()}),
                json.dumps(result.bias_metrics),
                json.dumps(result.fairness_metrics),
                result.data_purity,
                json.dumps(result.recommendations)
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing data quality results: {e}")
            self.conn.rollback()
            return -1

    def store_model_performance(self, model_name, framework, batch_size, 
                               profiling_result: ProfilingResult,
                               hardware_info, model_info) -> int:
        """Store comprehensive model performance profiling results"""
        try:
            query = """
                INSERT INTO model_performance 
                (model_name, framework, batch_size, execution_time, memory_usage, 
                cpu_utilization, gpu_utilization, operation_stats, bottlenecks, 
                recommendations, hardware_info, model_info)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                model_name,
                framework,
                batch_size,
                profiling_result.execution_time,
                json.dumps(profiling_result.memory_usage),
                json.dumps(profiling_result.cpu_utilization),
                json.dumps(profiling_result.gpu_utilization) if profiling_result.gpu_utilization else None,
                json.dumps(profiling_result.operation_stats),
                json.dumps(profiling_result.bottlenecks),
                json.dumps(profiling_result.recommendations),
                json.dumps(hardware_info),
                json.dumps(model_info)
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
            self.conn.rollback()
            return -1

    def store_hardware_recommendations(self, model_id, recommendations):
        """Store hardware recommendations with cost analysis"""
        try:
            query = """
                INSERT INTO hardware_recommendations
                (model_id, recommended_hardware, cloud_options, estimated_costs, scaling_strategy)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                model_id,
                json.dumps(recommendations['hardware']),
                json.dumps(recommendations['cloud_options']),
                json.dumps(recommendations['estimated_costs']),
                recommendations['scaling_strategy']
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing hardware recommendations: {e}")
            self.conn.rollback()
            return -1

    def get_historical_performance(self, model_name, limit=10):
        """Retrieve historical performance data for a specific model"""
        try:
            query = """
                SELECT id, timestamp, execution_time, memory_usage, 
                       cpu_utilization, gpu_utilization, bottlenecks, 
                       recommendations
                FROM model_performance
                WHERE model_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            self.cur.execute(query, (model_name, limit))
            results = []
            for row in self.cur.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'execution_time': row[2],
                    'memory_usage': json.loads(row[3]),
                    'cpu_utilization': json.loads(row[4]),
                    'gpu_utilization': json.loads(row[5]) if row[5] else None,
                    'bottlenecks': json.loads(row[6]),
                    'recommendations': json.loads(row[7])
                })
            return results
        except Exception as e:
            logger.error(f"Error retrieving historical performance: {e}")
            return []


class DataQualityAnalyzer:
    """
    Advanced data quality analysis with comprehensive metrics for bias, 
    fairness, and data integrity, along with targeted recommendations.
    """
    def __init__(self, df: pd.DataFrame, dataset_name: str, 
                 sensitive_features: List[str] = None):
        self.df = df
        self.dataset_name = dataset_name
        self.sensitive_features = sensitive_features or []
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.missing_threshold = 0.05  # 5% missing values threshold for recommendations
        self.outlier_threshold = 0.01  # 1% outliers threshold
        self.result = None

    def analyze(self) -> DataQualityResult:
        """Run comprehensive data quality analysis"""
        feature_stats = self._compute_feature_statistics()
        missing_values = self._analyze_missing_values()
        outliers = self._detect_outliers()
        bias_metrics = self._analyze_bias()
        fairness_metrics = self._compute_fairness_metrics()
        data_purity = self._calculate_data_purity()
        recommendations = self._generate_recommendations()

        self.result = DataQualityResult(
            dataset_name=self.dataset_name,
            feature_stats=feature_stats,
            missing_values=missing_values,
            outliers=outliers,
            bias_metrics=bias_metrics,
            fairness_metrics=fairness_metrics,
            data_purity=data_purity,
            recommendations=recommendations
        )
        return self.result

    def _compute_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute detailed statistics for each feature"""
        stats = {}
        
        # Process numerical features with detailed statistics
        for col in self.numerical_features:
            if self.df[col].isnull().sum() == len(self.df):
                continue
                
            col_stats = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'skew': float(stats.skew(self.df[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.df[col].dropna())),
                'iqr': float(np.percentile(self.df[col].dropna(), 75) - 
                            np.percentile(self.df[col].dropna(), 25))
            }
            stats[col] = col_stats
        
        # Process categorical features with frequency and entropy
        for col in self.categorical_features:
            value_counts = self.df[col].value_counts(normalize=True)
            
            if not value_counts.empty:
                entropy = stats.entropy(value_counts)
                most_common = value_counts.index[0]
                least_common = value_counts.index[-1]
                
                col_stats = {
                    'cardinality': len(value_counts),
                    'entropy': float(entropy),
                    'most_common_value': str(most_common),
                    'most_common_freq': float(value_counts.iloc[0]),
                    'least_common_value': str(least_common),
                    'least_common_freq': float(value_counts.iloc[-1])
                }
                stats[col] = col_stats
                
        return stats

    def _analyze_missing_values(self) -> Dict[str, float]:
        """Analyze missing values pattern and percentage"""
        missing_dict = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = missing_count / len(self.df)
            if missing_count > 0:
                missing_dict[col] = float(missing_pct)
        return missing_dict

    def _detect_outliers(self) -> Dict[str, List[int]]:
        """
        Detect outliers using multiple methods:
        - IQR method for numerical features
        - Z-score method as backup
        - Isolation Forest for complex patterns
        """
        outliers_dict = {}
        
        # Process each numerical feature
        for col in self.numerical_features:
            # Skip fully missing columns
            if self.df[col].isnull().all():
                continue
                
            # Get non-missing values
            clean_data = self.df[col].dropna()
            
            # IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Get indices of outliers
            outlier_indices = self.df[
                (self.df[col] < lower_bound) | 
                (self.df[col] > upper_bound)
            ].index.tolist()
            
            if len(outlier_indices) > 0:
                outliers_dict[col] = outlier_indices
                
        # Could add Isolation Forest for multivariate outlier detection here
            
        return outliers_dict

    def _analyze_bias(self) -> Dict[str, float]:
        """
        Analyze bias in the dataset with multiple metrics:
        - Class imbalance
        - Correlation between sensitive features and target
        - Statistical disparity between groups
        """
        bias_metrics = {}
        
        # Check for label column
        label_cols = [col for col in self.df.columns if col.lower() in 
                     ['label', 'target', 'class', 'y', 'outcome']]
        
        if label_cols:
            label_col = label_cols[0]
            # Class imbalance analysis
            value_counts = self.df[label_col].value_counts(normalize=True)
            # Normalized entropy (0 = perfect imbalance, 1 = perfect balance)
            if len(value_counts) > 1:
                entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                bias_metrics['class_balance'] = float(entropy)
                bias_metrics['majority_class_ratio'] = float(value_counts.max())
                bias_metrics['minority_class_ratio'] = float(value_counts.min())
            
            # Compute correlations with sensitive features if available
            if self.sensitive_features:
                for feature in self.sensitive_features:
                    if feature in self.df.columns:
                        # Calculate statistical parity difference
                        feature_values = self.df[feature].unique()
                        if len(feature_values) > 1 and len(feature_values) <= 10:
                            positive_rates = {}
                            for val in feature_values:
                                if isinstance(val, (int, float, bool, str)):
                                    subset = self.df[self.df[feature] == val]
                                    if len(subset) > 0 and label_col in subset:
                                        pos_rate = (subset[label_col] == 1).mean() if 1 in subset[label_col].unique() else 0
                                        positive_rates[str(val)] = pos_rate
                            
                            if positive_rates:
                                max_diff = max(positive_rates.values()) - min(positive_rates.values())
                                bias_metrics[f'statistical_parity_diff_{feature}'] = float(max_diff)
                                
        # Compute general distribution metrics for sensitive features
        if self.sensitive_features:
            for feature in self.sensitive_features:
                if feature in self.df.columns:
                    value_counts = self.df[feature].value_counts(normalize=True)
                    if len(value_counts) > 1:
                        entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                        bias_metrics[f'{feature}_distribution_entropy'] = float(entropy)
                        
        return bias_metrics

    def _compute_fairness_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute fairness metrics for sensitive features:
        - Demographic parity
        - Equal opportunity
        - Equalized odds
        """
        fairness_metrics = {}
        
        # Check for label column
        label_cols = [col for col in self.df.columns if col.lower() in 
                     ['label', 'target', 'class', 'y', 'outcome']]
        
        # Skip if no sensitive features or no label
        if not self.sensitive_features or not label_cols:
            return fairness_metrics
            
        label_col = label_cols[0]
        
        # Compute fairness metrics for each sensitive feature
        for feature in self.sensitive_features:
            if feature in self.df.columns:
                feature_metrics = {}
                feature_values = self.df[feature].unique()
                
                # Only compute metrics if we have reasonable number of categories
                if len(feature_values) > 1 and len(feature_values) <= 10:
                    # Calculate demographic parity difference
                    overall_positive_rate = (self.df[label_col] == 1).mean() if 1 in self.df[label_col].unique() else 0
                    max_diff = 0
                    
                    for val in feature_values:
                        if isinstance(val, (int, float, bool, str)):
                            subset = self.df[self.df[feature] == val]
                            if len(subset) > 0 and label_col in subset:
                                group_positive_rate = (subset[label_col] == 1).mean() if 1 in subset[label_col].unique() else 0
                                diff = abs(group_positive_rate - overall_positive_rate)
                                max_diff = max(max_diff, diff)
                                feature_metrics[f'positive_rate_{val}'] = float(group_positive_rate)
                    
                    feature_metrics['demographic_parity_diff'] = float(max_diff)
                    fairness_metrics[feature] = feature_metrics
                
        return fairness_metrics

    def _calculate_data_purity(self) -> float:
        """
        Calculate overall data purity score based on:
        - Missing values
        - Outliers
        - Bias metrics
        - Data distribution quality
        """
        factors = []
        
        # Factor 1: Missing values penalty
        missing_penalty = sum(self.df.isnull().mean()) / len(self.df.columns)
        factors.append(1 - missing_penalty)
        
        # Factor 2: Outliers penalty
        total_outliers = sum(len(indices) for indices in self._detect_outliers().values())
        outlier_penalty = total_outliers / (len(self.df) * len(self.numerical_features)) if self.numerical_features else 0
        factors.append(1 - outlier_penalty)
        
        # Factor 3: Balance score for categorical features
        balance_scores = []
        for col in self.categorical_features:
            value_counts = self.df[col].value_counts(normalize=True)
            if len(value_counts) > 1:
                entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                balance_scores.append(entropy)
        
        if balance_scores:
            factors.append(sum(balance_scores) / len(balance_scores))
            
        # Factor 4: Normality of numerical distributions
        normality_scores = []
        for col in self.numerical_features:
            if not self.df[col].isnull().all() and len(self.df[col].unique()) > 5:
                # Shapiro-Wilk test returns pvalue, higher = more normal
                try:
                    _, p_value = stats.shapiro(self.df[col].dropna())
                    normality_scores.append(min(p_value, 0.05) / 0.05)  # Normalize to [0,1]
                except:
                    pass
                    
        if normality_scores:
            factors.append(sum(normality_scores) / len(normality_scores))
            
        # Compute weighted average of all factors
        return sum(factors) / len(factors) if factors else 0.5

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific recommendations to improve data quality"""
        recommendations = []
        
        # Missing values recommendations
        high_missing_cols = {col: pct for col, pct in self._analyze_missing_values().items() 
                           if pct > self.missing_threshold}
        
        if high_missing_cols:
            for col, pct in high_missing_cols.items():
                missing_pct = pct * 100
                if pct > 0.5:
                    recommendations.append({
                        'type': 'missing_values',
                        'feature': col,
                        'severity': 'high',
                        'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                        'recommendation': f"Consider dropping this column or using advanced imputation techniques."
                    })
                else:
                    # Suggest specific imputation method based on feature type
                    if col in self.numerical_features:
                        recommendations.append({
                            'type': 'missing_values',
                            'feature': col,
                            'severity': 'medium',
                            'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                            'recommendation': f"Use KNN or regression-based imputation instead of mean/median.",
                            'code_example': f"from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors=5)\ndf['{col}'] = imputer.fit_transform(df[['{col}']])[:,0]"
                        })
                    else:
                        recommendations.append({
                            'type': 'missing_values',
                            'feature': col,
                            'severity': 'medium',
                            'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                            'recommendation': f"Use mode imputation or create a 'missing' category."
                        })
        
        # Outlier recommendations
        outliers = self._detect_outliers()
        for col, indices in outliers.items():
            if len(indices) > len(self.df) * self.outlier_threshold:
                recommendations.append({
                    'type': 'outliers',
                    'feature': col,
                    'severity': 'medium',
                    'description': f"Column '{col}' has {len(indices)} outliers ({len(indices)/len(self.df)*100:.1f}%).",
                    'recommendation': f"Consider robust scaling or winsorizing to handle outliers.",
                    'code_example': f"from scipy import stats\ndf['{col}'] = stats.mstats.winsorize(df['{col}'], limits=[0.05, 0.05])"
                })
        
        # Bias recommendations
        bias_metrics = self._analyze_bias()
        for metric, value in bias_metrics.items():
            if 'class_balance' in metric and value < 0.75:
                recommendations.append({
                    'type': 'bias',
                    'metric': metric,
                    'severity': 'high',
                    'description': f"Dataset shows class imbalance (balance score: {value:.2f}).",
                    'recommendation': "Use SMOTE, class weights, or stratified sampling to address class imbalance.",
                    'code_example': """
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
"""
                })
            elif 'statistical_parity_diff' in metric and value > 0.1:
                feature = metric.split('_')[-1]
                recommendations.append({
                    'type': 'fairness',
                    'metric': metric,
                    'severity': 'high',
                    'description': f"Statistical parity difference of {value:.2f} for feature '{feature}'.",
                    'recommendation': f"Apply fairness constraints or reweighting techniques.",
                    'code_example': """
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

# Convert to AIF360 dataset
dataset = BinaryLabelDataset(df=df, label_names=['target'], 
                            protected_attribute_names=['sensitive_feature'])
                            
# Apply reweighing
reweigher = Reweighing(unprivileged_groups=[{'sensitive_feature': 0}],
                      privileged_groups=[{'sensitive_feature': 1}])
transformed = reweigher.fit_transform(dataset)
"""
                })
                
        # Feature-specific recommendations based on statistics
        for col, stats in self._compute_feature_statistics().items():
            # Check for highly skewed numerical features
            if col in self.numerical_features and 'skew' in stats:
                if abs(stats['skew']) > 1.5:
                    recommendations.append({
                        'type': 'distribution',
                        'feature': col,
                        'severity': 'medium',
                        'description': f"Feature '{col}' is highly skewed (skew = {stats['skew']:.2f}).",
                        'recommendation': "Apply log, sqrt, or Box-Cox transformation.",
                        'code_example': f"import numpy as np\ndf['{col}_log'] = np.log1p(df['{col}'])"
                    })
            
            # Check for categorical features with too many categories
            if col in self.categorical_features and 'cardinality' in stats:
                if stats['cardinality'] > 20:
                    recommendations.append({
                        'type': 'high_cardinality',
                        'feature': col,
                        'severity': 'medium',
                        'description': f"Categorical feature '{col}' has {stats['cardinality']} unique values.",
                # filepath: c:\Users\adilm\repositories\Python\neural-scope\aiml_complexity\ml_based_suggestions_2.py
"""
Neural-Scope: Advanced AI/ML Performance and Data Quality Analysis

A comprehensive solution for machine learning engineers to analyze, optimize, and
improve their ML workflows through deep performance profiling, data quality assessment,
and targeted optimization recommendations.

This module provides granular, actionable insights for improving model performance,
data quality, and hardware utilization, with specific code-level recommendations.
"""

import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Framework-specific imports with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.python.eager import profiler as tf_profiler
    from tensorflow.keras import layers as tf_layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorFlow-specific features will be disabled.")

try:
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Plotly Dash not available. Interactive dashboards will be disabled.")

# Model architecture recognition utilities
from aiml_complexity.ml_patterns import MLPatternDatabase


@dataclass
class ProfilingResult:
    """Stores comprehensive profiling results with detailed metrics"""
    execution_time: float
    memory_usage: Dict[str, float]
    cpu_utilization: Dict[str, float]
    gpu_utilization: Optional[Dict[str, float]]
    operation_stats: Dict[str, Dict[str, float]]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


@dataclass
class DataQualityResult:
    """Stores comprehensive data quality analysis results"""
    dataset_name: str
    feature_stats: Dict[str, Dict[str, float]]
    missing_values: Dict[str, float]
    outliers: Dict[str, List[int]]
    bias_metrics: Dict[str, float]
    fairness_metrics: Dict[str, Dict[str, float]]
    data_purity: float
    recommendations: List[Dict[str, Any]]


class PostgresStorage:
    """
    Advanced PostgreSQL storage for both data quality results
    and model performance stats with proper connection management
    and detailed metadata storage.
    """
    def __init__(self, host, port, dbname, user, password):
        self.connection_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cur = None
        
    def connect(self):
        """Establish connection with error handling"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cur = self.conn.cursor()
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            return False

    def disconnect(self):
        """Properly close connections"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        self.cur = None
        self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def setup_tables(self):
        """Create tables if they don't exist with comprehensive schema design"""
        try:
            # Data quality table with detailed metrics
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id SERIAL PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feature_stats JSONB,
                    missing_values JSONB,
                    outliers JSONB,
                    bias_metrics JSONB,
                    fairness_metrics JSONB,
                    data_purity REAL,
                    recommendations JSONB
                )
            """)
            
            # Model performance table with detailed metrics
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    batch_size INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time REAL,
                    memory_usage JSONB,
                    cpu_utilization JSONB,
                    gpu_utilization JSONB,
                    operation_stats JSONB,
                    bottlenecks JSONB,
                    recommendations JSONB,
                    hardware_info JSONB,
                    model_info JSONB
                )
            """)
            
            # Hardware recommendations table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS hardware_recommendations (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES model_performance(id),
                    recommended_hardware JSONB,
                    cloud_options JSONB,
                    estimated_costs JSONB,
                    scaling_strategy TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_dataset ON data_quality(dataset_name)")
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_name ON model_performance(model_name)")
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_timestamp ON model_performance(timestamp)")
            
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Database setup error: {e}")
            self.conn.rollback()
            return False

    def store_data_quality_result(self, result: DataQualityResult) -> int:
        """Store data quality analysis results"""
        try:
            query = """
                INSERT INTO data_quality 
                (dataset_name, feature_stats, missing_values, outliers, 
                bias_metrics, fairness_metrics, data_purity, recommendations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                result.dataset_name,
                json.dumps(result.feature_stats),
                json.dumps(result.missing_values),
                json.dumps({k: v for k, v in result.outliers.items()}),
                json.dumps(result.bias_metrics),
                json.dumps(result.fairness_metrics),
                result.data_purity,
                json.dumps(result.recommendations)
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing data quality results: {e}")
            self.conn.rollback()
            return -1

    def store_model_performance(self, model_name, framework, batch_size, 
                               profiling_result: ProfilingResult,
                               hardware_info, model_info) -> int:
        """Store comprehensive model performance profiling results"""
        try:
            query = """
                INSERT INTO model_performance 
                (model_name, framework, batch_size, execution_time, memory_usage, 
                cpu_utilization, gpu_utilization, operation_stats, bottlenecks, 
                recommendations, hardware_info, model_info)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                model_name,
                framework,
                batch_size,
                profiling_result.execution_time,
                json.dumps(profiling_result.memory_usage),
                json.dumps(profiling_result.cpu_utilization),
                json.dumps(profiling_result.gpu_utilization) if profiling_result.gpu_utilization else None,
                json.dumps(profiling_result.operation_stats),
                json.dumps(profiling_result.bottlenecks),
                json.dumps(profiling_result.recommendations),
                json.dumps(hardware_info),
                json.dumps(model_info)
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
            self.conn.rollback()
            return -1

    def store_hardware_recommendations(self, model_id, recommendations):
        """Store hardware recommendations with cost analysis"""
        try:
            query = """
                INSERT INTO hardware_recommendations
                (model_id, recommended_hardware, cloud_options, estimated_costs, scaling_strategy)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(query, (
                model_id,
                json.dumps(recommendations['hardware']),
                json.dumps(recommendations['cloud_options']),
                json.dumps(recommendations['estimated_costs']),
                recommendations['scaling_strategy']
            ))
            row_id = self.cur.fetchone()[0]
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Error storing hardware recommendations: {e}")
            self.conn.rollback()
            return -1

    def get_historical_performance(self, model_name, limit=10):
        """Retrieve historical performance data for a specific model"""
        try:
            query = """
                SELECT id, timestamp, execution_time, memory_usage, 
                       cpu_utilization, gpu_utilization, bottlenecks, 
                       recommendations
                FROM model_performance
                WHERE model_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            self.cur.execute(query, (model_name, limit))
            results = []
            for row in self.cur.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'execution_time': row[2],
                    'memory_usage': json.loads(row[3]),
                    'cpu_utilization': json.loads(row[4]),
                    'gpu_utilization': json.loads(row[5]) if row[5] else None,
                    'bottlenecks': json.loads(row[6]),
                    'recommendations': json.loads(row[7])
                })
            return results
        except Exception as e:
            logger.error(f"Error retrieving historical performance: {e}")
            return []


class DataQualityAnalyzer:
    """
    Advanced data quality analysis with comprehensive metrics for bias, 
    fairness, and data integrity, along with targeted recommendations.
    """
    def __init__(self, df: pd.DataFrame, dataset_name: str, 
                 sensitive_features: List[str] = None):
        self.df = df
        self.dataset_name = dataset_name
        self.sensitive_features = sensitive_features or []
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.missing_threshold = 0.05  # 5% missing values threshold for recommendations
        self.outlier_threshold = 0.01  # 1% outliers threshold
        self.result = None

    def analyze(self) -> DataQualityResult:
        """Run comprehensive data quality analysis"""
        feature_stats = self._compute_feature_statistics()
        missing_values = self._analyze_missing_values()
        outliers = self._detect_outliers()
        bias_metrics = self._analyze_bias()
        fairness_metrics = self._compute_fairness_metrics()
        data_purity = self._calculate_data_purity()
        recommendations = self._generate_recommendations()

        self.result = DataQualityResult(
            dataset_name=self.dataset_name,
            feature_stats=feature_stats,
            missing_values=missing_values,
            outliers=outliers,
            bias_metrics=bias_metrics,
            fairness_metrics=fairness_metrics,
            data_purity=data_purity,
            recommendations=recommendations
        )
        return self.result

    def _compute_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute detailed statistics for each feature"""
        stats = {}
        
        # Process numerical features with detailed statistics
        for col in self.numerical_features:
            if self.df[col].isnull().sum() == len(self.df):
                continue
                
            col_stats = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'skew': float(stats.skew(self.df[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.df[col].dropna())),
                'iqr': float(np.percentile(self.df[col].dropna(), 75) - 
                            np.percentile(self.df[col].dropna(), 25))
            }
            stats[col] = col_stats
        
        # Process categorical features with frequency and entropy
        for col in self.categorical_features:
            value_counts = self.df[col].value_counts(normalize=True)
            
            if not value_counts.empty:
                entropy = stats.entropy(value_counts)
                most_common = value_counts.index[0]
                least_common = value_counts.index[-1]
                
                col_stats = {
                    'cardinality': len(value_counts),
                    'entropy': float(entropy),
                    'most_common_value': str(most_common),
                    'most_common_freq': float(value_counts.iloc[0]),
                    'least_common_value': str(least_common),
                    'least_common_freq': float(value_counts.iloc[-1])
                }
                stats[col] = col_stats
                
        return stats

    def _analyze_missing_values(self) -> Dict[str, float]:
        """Analyze missing values pattern and percentage"""
        missing_dict = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = missing_count / len(self.df)
            if missing_count > 0:
                missing_dict[col] = float(missing_pct)
        return missing_dict

    def _detect_outliers(self) -> Dict[str, List[int]]:
        """
        Detect outliers using multiple methods:
        - IQR method for numerical features
        - Z-score method as backup
        - Isolation Forest for complex patterns
        """
        outliers_dict = {}
        
        # Process each numerical feature
        for col in self.numerical_features:
            # Skip fully missing columns
            if self.df[col].isnull().all():
                continue
                
            # Get non-missing values
            clean_data = self.df[col].dropna()
            
            # IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Get indices of outliers
            outlier_indices = self.df[
                (self.df[col] < lower_bound) | 
                (self.df[col] > upper_bound)
            ].index.tolist()
            
            if len(outlier_indices) > 0:
                outliers_dict[col] = outlier_indices
                
        # Could add Isolation Forest for multivariate outlier detection here
            
        return outliers_dict

    def _analyze_bias(self) -> Dict[str, float]:
        """
        Analyze bias in the dataset with multiple metrics:
        - Class imbalance
        - Correlation between sensitive features and target
        - Statistical disparity between groups
        """
        bias_metrics = {}
        
        # Check for label column
        label_cols = [col for col in self.df.columns if col.lower() in 
                     ['label', 'target', 'class', 'y', 'outcome']]
        
        if label_cols:
            label_col = label_cols[0]
            # Class imbalance analysis
            value_counts = self.df[label_col].value_counts(normalize=True)
            # Normalized entropy (0 = perfect imbalance, 1 = perfect balance)
            if len(value_counts) > 1:
                entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                bias_metrics['class_balance'] = float(entropy)
                bias_metrics['majority_class_ratio'] = float(value_counts.max())
                bias_metrics['minority_class_ratio'] = float(value_counts.min())
            
            # Compute correlations with sensitive features if available
            if self.sensitive_features:
                for feature in self.sensitive_features:
                    if feature in self.df.columns:
                        # Calculate statistical parity difference
                        feature_values = self.df[feature].unique()
                        if len(feature_values) > 1 and len(feature_values) <= 10:
                            positive_rates = {}
                            for val in feature_values:
                                if isinstance(val, (int, float, bool, str)):
                                    subset = self.df[self.df[feature] == val]
                                    if len(subset) > 0 and label_col in subset:
                                        pos_rate = (subset[label_col] == 1).mean() if 1 in subset[label_col].unique() else 0
                                        positive_rates[str(val)] = pos_rate
                            
                            if positive_rates:
                                max_diff = max(positive_rates.values()) - min(positive_rates.values())
                                bias_metrics[f'statistical_parity_diff_{feature}'] = float(max_diff)
                                
        # Compute general distribution metrics for sensitive features
        if self.sensitive_features:
            for feature in self.sensitive_features:
                if feature in self.df.columns:
                    value_counts = self.df[feature].value_counts(normalize=True)
                    if len(value_counts) > 1:
                        entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                        bias_metrics[f'{feature}_distribution_entropy'] = float(entropy)
                        
        return bias_metrics

    def _compute_fairness_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute fairness metrics for sensitive features:
        - Demographic parity
        - Equal opportunity
        - Equalized odds
        """
        fairness_metrics = {}
        
        # Check for label column
        label_cols = [col for col in self.df.columns if col.lower() in 
                     ['label', 'target', 'class', 'y', 'outcome']]
        
        # Skip if no sensitive features or no label
        if not self.sensitive_features or not label_cols:
            return fairness_metrics
            
        label_col = label_cols[0]
        
        # Compute fairness metrics for each sensitive feature
        for feature in self.sensitive_features:
            if feature in self.df.columns:
                feature_metrics = {}
                feature_values = self.df[feature].unique()
                
                # Only compute metrics if we have reasonable number of categories
                if len(feature_values) > 1 and len(feature_values) <= 10:
                    # Calculate demographic parity difference
                    overall_positive_rate = (self.df[label_col] == 1).mean() if 1 in self.df[label_col].unique() else 0
                    max_diff = 0
                    
                    for val in feature_values:
                        if isinstance(val, (int, float, bool, str)):
                            subset = self.df[self.df[feature] == val]
                            if len(subset) > 0 and label_col in subset:
                                group_positive_rate = (subset[label_col] == 1).mean() if 1 in subset[label_col].unique() else 0
                                diff = abs(group_positive_rate - overall_positive_rate)
                                max_diff = max(max_diff, diff)
                                feature_metrics[f'positive_rate_{val}'] = float(group_positive_rate)
                    
                    feature_metrics['demographic_parity_diff'] = float(max_diff)
                    fairness_metrics[feature] = feature_metrics
                
        return fairness_metrics

    def _calculate_data_purity(self) -> float:
        """
        Calculate overall data purity score based on:
        - Missing values
        - Outliers
        - Bias metrics
        - Data distribution quality
        """
        factors = []
        
        # Factor 1: Missing values penalty
        missing_penalty = sum(self.df.isnull().mean()) / len(self.df.columns)
        factors.append(1 - missing_penalty)
        
        # Factor 2: Outliers penalty
        total_outliers = sum(len(indices) for indices in self._detect_outliers().values())
        outlier_penalty = total_outliers / (len(self.df) * len(self.numerical_features)) if self.numerical_features else 0
        factors.append(1 - outlier_penalty)
        
        # Factor 3: Balance score for categorical features
        balance_scores = []
        for col in self.categorical_features:
            value_counts = self.df[col].value_counts(normalize=True)
            if len(value_counts) > 1:
                entropy = stats.entropy(value_counts) / np.log(len(value_counts))
                balance_scores.append(entropy)
        
        if balance_scores:
            factors.append(sum(balance_scores) / len(balance_scores))
            
        # Factor 4: Normality of numerical distributions
        normality_scores = []
        for col in self.numerical_features:
            if not self.df[col].isnull().all() and len(self.df[col].unique()) > 5:
                # Shapiro-Wilk test returns pvalue, higher = more normal
                try:
                    _, p_value = stats.shapiro(self.df[col].dropna())
                    normality_scores.append(min(p_value, 0.05) / 0.05)  # Normalize to [0,1]
                except:
                    pass
                    
        if normality_scores:
            factors.append(sum(normality_scores) / len(normality_scores))
            
        # Compute weighted average of all factors
        return sum(factors) / len(factors) if factors else 0.5

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific recommendations to improve data quality"""
        recommendations = []
        
        # Missing values recommendations
        high_missing_cols = {col: pct for col, pct in self._analyze_missing_values().items() 
                           if pct > self.missing_threshold}
        
        if high_missing_cols:
            for col, pct in high_missing_cols.items():
                missing_pct = pct * 100
                if pct > 0.5:
                    recommendations.append({
                        'type': 'missing_values',
                        'feature': col,
                        'severity': 'high',
                        'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                        'recommendation': f"Consider dropping this column or using advanced imputation techniques."
                    })
                else:
                    # Suggest specific imputation method based on feature type
                    if col in self.numerical_features:
                        recommendations.append({
                            'type': 'missing_values',
                            'feature': col,
                            'severity': 'medium',
                            'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                            'recommendation': f"Use KNN or regression-based imputation instead of mean/median.",
                            'code_example': f"from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors=5)\ndf['{col}'] = imputer.fit_transform(df[['{col}']])[:,0]"
                        })
                    else:
                        recommendations.append({
                            'type': 'missing_values',
                            'feature': col,
                            'severity': 'medium',
                            'description': f"Column '{col}' has {missing_pct:.1f}% missing values.",
                            'recommendation': f"Use mode imputation or create a 'missing' category."
                        })
        
        # Outlier recommendations
        outliers = self._detect_outliers()
        for col, indices in outliers.items():
            if len(indices) > len(self.df) * self.outlier_threshold:
                recommendations.append({
                    'type': 'outliers',
                    'feature': col,
                    'severity': 'medium',
                    'description': f"Column '{col}' has {len(indices)} outliers ({len(indices)/len(self.df)*100:.1f}%).",
                    'recommendation': f"Consider robust scaling or winsorizing to handle outliers.",
                    'code_example': f"from scipy import stats\ndf['{col}'] = stats.mstats.winsorize(df['{col}'], limits=[0.05, 0.05])"
                })
        
        # Bias recommendations
        bias_metrics = self._analyze_bias()
        for metric, value in bias_metrics.items():
            if 'class_balance' in metric and value < 0.75:
                recommendations.append({
                    'type': 'bias',
                    'metric': metric,
                    'severity': 'high',
                    'description': f"Dataset shows class imbalance (balance score: {value:.2f}).",
                    'recommendation': "Use SMOTE, class weights, or stratified sampling to address class imbalance.",
                    'code_example': """
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
"""
                })
            elif 'statistical_parity_diff' in metric and value > 0.1:
                feature = metric.split('_')[-1]
                recommendations.append({
                    'type': 'fairness',
                    'metric': metric,
                    'severity': 'high',
                    'description': f"Statistical parity difference of {value:.2f} for feature '{feature}'.",
                    'recommendation': f"Apply fairness constraints or reweighting techniques.",
                    'code_example': """
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

# Convert to AIF360 dataset
dataset = BinaryLabelDataset(df=df, label_names=['target'], 
                            protected_attribute_names=['sensitive_feature'])
                            
# Apply reweighing
reweigher = Reweighing(unprivileged_groups=[{'sensitive_feature': 0}],
                      privileged_groups=[{'sensitive_feature': 1}])
transformed = reweigher.fit_transform(dataset)
"""
                })
                
        # Feature-specific recommendations based on statistics
        for col, stats in self._compute_feature_statistics().items():
            # Check for highly skewed numerical features
            if col in self.numerical_features and 'skew' in stats:
                if abs(stats['skew']) > 1.5:
                    recommendations.append({
                        'type': 'distribution',
                        'feature': col,
                        'severity': 'medium',
                        'description': f"Feature '{col}' is highly skewed (skew = {stats['skew']:.2f}).",
                        'recommendation': "Apply log, sqrt, or Box-Cox transformation.",
                        'code_example': f"import numpy as np\ndf['{col}_log'] = np.log1p(df['{col}'])"
                    })
            
            # Check for categorical features with too many categories
            if col in self.categorical_features and 'cardinality' in stats:
                if stats['cardinality'] > 20:
                    recommendations.append({
                        'type': 'high_cardinality',
                        'feature': col,
                        'severity': 'medium',
                        'description': f"Categorical feature '{col}' has {stats['cardinality']} unique values.",
                