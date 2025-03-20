"""
DataGuardian: Enterprise-Grade ML Data Quality & Ethical Analysis System

A comprehensive, state-of-the-art data quality assessment framework designed for 
production ML workflows. DataGuardian performs deep inspection of datasets to 
identify quality issues, biases, and ethical concerns while providing actionable 
remediation recommendations with code examples.

Key capabilities:
1. Multi-dimensional bias detection across intersectional protected attributes
2. Advanced data quality assessment with ML-based anomaly detection
3. Distribution drift monitoring for production ML systems
4. Privacy risk assessment and PII detection
5. Fairness constraint implementation with multiple mathematical definitions
6. Explainable AI integration for bias investigation
7. Customizable reporting with interactive visualizations
8. Remediation recommendations with executable code examples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy import stats
import re
import logging
import json
from pathlib import Path
import hashlib
from datetime import datetime
import warnings
import joblib

# Optional dependencies with graceful fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import fairlearn
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataGuardian")


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics container"""
    completeness: Dict[str, float] = field(default_factory=dict)
    uniqueness: Dict[str, float] = field(default_factory=dict)
    consistency: Dict[str, Dict] = field(default_factory=dict)
    validity: Dict[str, float] = field(default_factory=dict)
    outlier_scores: Dict[str, List[int]] = field(default_factory=dict)
    distribution_metrics: Dict[str, Dict] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    quality_score: float = 1.0
    

@dataclass
class BiasMetrics:
    """Container for bias and fairness metrics"""
    # Statistical bias metrics
    statistical_parity: Dict[str, float] = field(default_factory=dict)
    disparate_impact: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, float] = field(default_factory=dict)
    
    # Intersectional bias metrics
    intersectional_bias_scores: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance bias
    feature_importance_bias: Dict[str, float] = field(default_factory=dict)
    
    # Overall bias score (0-1 where 0 is perfectly unbiased)
    overall_bias_score: float = 0.0
    

@dataclass
class PrivacyMetrics:
    """Container for privacy and security risk metrics"""
    pii_detected: Dict[str, List[str]] = field(default_factory=dict)
    uniqueness_risk: Dict[str, float] = field(default_factory=dict)
    k_anonymity: int = 0
    l_diversity: Dict[str, int] = field(default_factory=dict)
    overall_privacy_risk: float = 0.0


@dataclass
class DataDriftMetrics:
    """Container for distribution shift and drift metrics"""
    feature_drift: Dict[str, float] = field(default_factory=dict)
    distribution_shifts: Dict[str, Dict] = field(default_factory=dict)
    overall_drift_score: float = 0.0


@dataclass
class DataRecommendation:
    """Structured recommendation with actionable code examples"""
    issue_type: str  # 'quality', 'bias', 'privacy', 'drift'
    severity: str    # 'low', 'medium', 'high', 'critical'
    feature: Optional[str] = None
    description: str = ""
    impact: str = ""
    recommendation: str = ""
    code_example: str = ""
    reference_url: str = ""


@dataclass
class DataGuardianReport:
    """Comprehensive report containing all analysis results"""
    dataset_name: str
    timestamp: datetime
    dataset_stats: Dict[str, Any]
    quality_metrics: DataQualityMetrics
    bias_metrics: Optional[BiasMetrics] = None
    privacy_metrics: Optional[PrivacyMetrics] = None
    drift_metrics: Optional[DataDriftMetrics] = None
    recommendations: List[DataRecommendation] = field(default_factory=list)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        result = {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "dataset_stats": self.dataset_stats,
            "quality_metrics": {
                "completeness": self.quality_metrics.completeness,
                "uniqueness": self.quality_metrics.uniqueness,
                "distribution_metrics": self.quality_metrics.distribution_metrics,
                "quality_score": self.quality_metrics.quality_score
            },
            "recommendations": [
                {
                    "issue_type": r.issue_type,
                    "severity": r.severity,
                    "feature": r.feature,
                    "description": r.description,
                    "recommendation": r.recommendation,
                    "code_example": r.code_example
                } for r in self.recommendations
            ],
            "execution_time": self.execution_time
        }
        
        # Add optional metrics if available
        if self.bias_metrics:
            result["bias_metrics"] = {
                "statistical_parity": self.bias_metrics.statistical_parity,
                "disparate_impact": self.bias_metrics.disparate_impact,
                "overall_bias_score": self.bias_metrics.overall_bias_score
            }
            
        if self.privacy_metrics:
            result["privacy_metrics"] = {
                "pii_detected": self.privacy_metrics.pii_detected,
                "k_anonymity": self.privacy_metrics.k_anonymity,
                "overall_privacy_risk": self.privacy_metrics.overall_privacy_risk
            }
            
        if self.drift_metrics:
            result["drift_metrics"] = {
                "feature_drift": self.drift_metrics.feature_drift,
                "overall_drift_score": self.drift_metrics.overall_drift_score
            }
            
        return result
    
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save report as JSON file or return JSON string"""
        report_dict = self.to_dict()
        json_str = json.dumps(report_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        
        return json_str
    
    def create_summary(self) -> str:
        """Generate a text summary of the report"""
        summary = [
            f"DataGuardian Report for '{self.dataset_name}'",
            f"Generated on: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset shape: {self.dataset_stats['num_rows']} rows × {self.dataset_stats['num_columns']} columns",
            "",
            "=== DATA QUALITY SUMMARY ===",
            f"Overall data quality score: {self.quality_metrics.quality_score:.2f}/1.00",
            f"Columns with missing values: {sum(1 for v in self.quality_metrics.completeness.values() if v < 1.0)}",
            f"Columns with outliers: {len(self.quality_metrics.outlier_scores)}",
        ]
        
        if self.bias_metrics:
            summary.extend([
                "",
                "=== BIAS ASSESSMENT SUMMARY ===",
                f"Overall bias score: {self.bias_metrics.overall_bias_score:.2f}/1.00",
                f"Protected attributes analyzed: {len(self.bias_metrics.statistical_parity)}"
            ])
        
        if self.privacy_metrics:
            summary.extend([
                "",
                "=== PRIVACY ASSESSMENT SUMMARY ===",
                f"Overall privacy risk: {self.privacy_metrics.overall_privacy_risk:.2f}/1.00",
                f"PII columns detected: {len(self.privacy_metrics.pii_detected)}",
                f"K-anonymity: {self.privacy_metrics.k_anonymity}"
            ])
        
        if self.drift_metrics:
            summary.extend([
                "",
                "=== DATA DRIFT SUMMARY ===",
                f"Overall drift score: {self.drift_metrics.overall_drift_score:.2f}/1.00",
                f"Features with significant drift: {sum(1 for v in self.drift_metrics.feature_drift.values() if v > 0.1)}"
            ])
        
        summary.extend([
            "",
            "=== RECOMMENDATIONS SUMMARY ===",
            f"Total recommendations: {len(self.recommendations)}",
            f"Critical issues: {sum(1 for r in self.recommendations if r.severity == 'critical')}",
            f"High severity issues: {sum(1 for r in self.recommendations if r.severity == 'high')}",
            f"Medium severity issues: {sum(1 for r in self.recommendations if r.severity == 'medium')}",
            f"Low severity issues: {sum(1 for r in self.recommendations if r.severity == 'low')}"
        ])
        
        return "\n".join(summary)


class DataGuardian:
    """
    Advanced data quality, bias detection, and ethical analysis framework for ML datasets.
    
    DataGuardian provides comprehensive assessment of ML datasets to ensure they meet
    quality standards, are free from harmful biases, respect privacy, and remain stable
    over time through distribution drift detection.
    """
    
    def __init__(self, 
                 dataset_name: str = "unnamed_dataset",
                 protected_attributes: Optional[List[str]] = None,
                 sensitive_attributes: Optional[List[str]] = None,
                 categorical_threshold: int = 20,
                 outlier_threshold: float = 0.03,
                 save_reports: bool = True,
                 reports_dir: str = "./data_guardian_reports",
                 verbose: bool = True):
        """
        Initialize DataGuardian with configuration parameters
        
        Args:
            dataset_name: Name identifier for the dataset
            protected_attributes: List of column names to be treated as protected attributes for bias analysis
            sensitive_attributes: List of column names containing sensitive/PII data for privacy analysis
            categorical_threshold: Maximum number of unique values to consider a column categorical
            outlier_threshold: Threshold for outlier detection (ratio of outliers to flag)
            save_reports: Whether to save reports to disk
            reports_dir: Directory to save generated reports
            verbose: Whether to print detailed logs during analysis
        """
        self.dataset_name = dataset_name
        self.protected_attributes = protected_attributes or []
        self.sensitive_attributes = sensitive_attributes or []
        self.categorical_threshold = categorical_threshold
        self.outlier_threshold = outlier_threshold
        self.save_reports = save_reports
        self.reports_dir = reports_dir
        self.verbose = verbose
        
        # Create reports directory if it doesn't exist
        if self.save_reports:
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
        # Configuration for common data types and patterns
        self._setup_type_patterns()
        
        # Set log level based on verbosity
        if not verbose:
            logger.setLevel(logging.WARNING)
        
        logger.info(f"DataGuardian initialized for dataset '{dataset_name}'")
    
    def _setup_type_patterns(self):
        """Setup regex patterns for various data types"""
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d{4}[ -]?){3}\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "address": r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way)\b',
            "name": r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
        }
    
    def analyze_dataset(self, 
                        df: pd.DataFrame, 
                        target_column: Optional[str] = None,
                        reference_df: Optional[pd.DataFrame] = None,
                        protected_attributes: Optional[List[str]] = None,
                        sensitive_attributes: Optional[List[str]] = None) -> DataGuardianReport:
        """
        Perform comprehensive analysis of a dataset
        
        Args:
            df: Pandas DataFrame to analyze
            target_column: Name of the target/label column for bias analysis
            reference_df: Optional reference dataset for drift detection
            protected_attributes: Override class-level protected attributes
            sensitive_attributes: Override class-level sensitive attributes
            
        Returns:
            DataGuardianReport with comprehensive analysis results
        """
        start_time = datetime.now()
        logger.info(f"Starting comprehensive analysis of dataset with shape {df.shape}")
        
        # Use parameters if provided, else use class attributes
        protected_attrs = protected_attributes or self.protected_attributes
        sensitive_attrs = sensitive_attributes or self.sensitive_attributes
        
        # Basic dataset statistics
        dataset_stats = self._compute_dataset_statistics(df)
        
        # Data type inference
        numerical_features, categorical_features = self._infer_column_types(df)
        
        # Comprehensive data quality assessment
        quality_metrics = self._assess_data_quality(df, numerical_features, categorical_features)
        
        # Initialize containers for optional analyses
        bias_metrics = None
        privacy_metrics = None
        drift_metrics = None
        
        # Bias assessment if target column and protected attributes are specified
        if target_column is not None and protected_attrs:
            bias_metrics = self._assess_bias(df, target_column, protected_attrs)
        
        # Privacy assessment if sensitive attributes are specified or auto-detection is enabled
        if sensitive_attrs or self._should_detect_pii():
            privacy_metrics = self._assess_privacy(df, sensitive_attrs)
        
        # Drift assessment if reference dataset is provided
        if reference_df is not None:
            drift_metrics = self._assess_drift(df, reference_df)
        
        # Generate recommendations based on all findings
        recommendations = self._generate_recommendations(
            df, quality_metrics, bias_metrics, privacy_metrics, drift_metrics,
            numerical_features, categorical_features, target_column
        )
        
        # Create comprehensive report
        report = DataGuardianReport(
            dataset_name=self.dataset_name,
            timestamp=datetime.now(),
            dataset_stats=dataset_stats,
            quality_metrics=quality_metrics,
            bias_metrics=bias_metrics,
            privacy_metrics=privacy_metrics,
            drift_metrics=drift_metrics,
            recommendations=recommendations,
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        # Save report if configured
        if self.save_reports:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.reports_dir}/{self.dataset_name}_{timestamp}.json"
            report.to_json(filename)
            logger.info(f"Report saved to {filename}")
        
        logger.info(f"Analysis completed in {report.execution_time:.2f} seconds")
        return report
    
    def _compute_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute general statistics about the dataset"""
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "column_types": df.dtypes.astype(str).to_dict(),
            "column_unique_counts": df.nunique().to_dict(),
            "column_missing_counts": df.isna().sum().to_dict()
        }
        return stats
    
    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Intelligently infer numerical and categorical columns
        Goes beyond simple dtype checking to handle encoded categories
        """
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            # Skip columns with too many missing values
            if df[col].isna().mean() > 0.5:
                continue
                
            # Check if numeric type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (encoded as numbers)
                unique_count = df[col].nunique()
                if unique_count <= self.categorical_threshold:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
                
        logger.info(f"Inferred {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        return numerical_features, categorical_features

    def _assess_data_quality(self, 
                             df: pd.DataFrame, 
                             numerical_features: List[str],
                             categorical_features: List[str]) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment
        
        Checks for:
        - Completeness (missing values)
        - Uniqueness
        - Consistency (format, ranges)
        - Validity (data format compliance)
        - Outliers (statistical and ML-based)
        - Distribution characteristics
        """
        metrics = DataQualityMetrics()
        
        # Assess completeness (missing values)
        metrics.completeness = {col: 1 - df[col].isna().mean() for col in df.columns}
        
        # Assess uniqueness
        metrics.uniqueness = {col: df[col].nunique() / len(df) for col in df.columns}
        
        # Find outliers in numerical features
        metrics.outlier_scores = {}
        for col in numerical_features:
            if df[col].isna().mean() > 0.3:  # Skip columns with too many missing values
                continue
                
            # Use IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            if len(outliers) / len(df) > self.outlier_threshold:
                metrics.outlier_scores[col] = outliers
        
        # Calculate distribution metrics for numerical features
        metrics.distribution_metrics = {}
        for col in numerical_features:
            if df[col].isna().mean() > 0.3:
                continue
                
            metrics.distribution_metrics[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': float(stats.skew(df[col].dropna())),
                'kurtosis': float(stats.kurtosis(df[col].dropna())),
            }
            
        # Calculate distribution metrics for categorical features
        for col in categorical_features:
            if df[col].isna().mean() > 0.3:
                continue
                
            value_counts = df[col].value_counts(normalize=True)
            entropy = stats.entropy(value_counts)
            metrics.distribution_metrics[col] = {
                'entropy': float(entropy),
                'top_value': value_counts.index[0],
                'top_value_pct': float(value_counts.iloc[0]),
                'unique_count': df[col].nunique(),
            }
        
        # Assess consistency and validity
        metrics.consistency = self._check_consistency(df)
        metrics.validity = self._check_validity(df)
        
        # Compute correlation matrix if there are enough numerical features
        if len(numerical_features) > 1:
            metrics.correlation_matrix = df[numerical_features].corr()
        
        # Compute overall quality score based on all metrics
        quality_scores = []
        # Completeness score
        completeness_score = np.mean(list(metrics.completeness.values()))
        quality_scores.append(completeness_score)
        
        # Outlier score (penalize datasets with many outliers)
        outlier_ratio = sum(len(outliers) for outliers in metrics.outlier_scores.values()) / (len(df) * len(numerical_features)) if numerical_features else 0
        outlier_score = 1 - outlier_ratio
        quality_scores.append(outlier_score)
        
        # Distribution score (penalize highly skewed features)
        skewness_scores = []
        for col, stats_dict in metrics.distribution_metrics.items():
            if 'skewness' in stats_dict:
                # Normalize skewness impact: 0 (highly skewed) to 1 (normal)
                skew_score = 1 - min(abs(stats_dict['skewness']), 10) / 10
                skewness_scores.append(skew_score)
        
        distribution_score = np.mean(skewness_scores) if skewness_scores else 1.0
        quality_scores.append(distribution_score)
        
        # Overall quality score
        metrics.quality_score = float(np.mean(quality_scores))
        
        logger.info(f"Data quality assessment completed. Overall score: {metrics.quality_score:.4f}")
        return metrics
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Check data consistency within columns"""
        consistency = {}
        
        # Check date format consistency
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100).astype(str)
                
                # Date format consistency check
                date_formats = {}
                for val in sample:
                    # Skip empty or very long values
                    if not val or len(val) > 30:
                        continue
                        
                    # Check if it might be a date
                    if re.match(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', val):
                        format_key = re.sub(r'\d', '#', val)
                        date_formats[format_key] = date_formats.get(format_key, 0) + 1
                
                if date_formats:
                    consistency[col] = {
                        'potential_date_column': True,
                        'date_formats': date_formats,
                        'consistent_format': len(date_formats) == 1
                    }
        
        return consistency
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check data validity based on expected patterns"""
        validity = {}
        
        # Check for valid email format in string columns that might contain emails
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column name suggests it might contain emails
                if 'email' in col.lower():
                    # Sample values and check against email regex
                    sample = df[col].dropna().astype(str).head(1000)
                    valid_count = sum(1 for val in sample if re.match(self.pii_patterns['email'], val))
                    if valid_count > 0:
                        validity[col] = valid_count / len(sample)
                        
        # Add other validity checks as needed (phone numbers, URLs, etc.)
                        
        return validity
    
    def _assess_bias(self, 
                     df: pd.DataFrame, 
                     target_column: str, 
                     protected_attributes: List[str]) -> BiasMetrics:
        """
        Perform comprehensive bias assessment across protected attributes
        
        Args:
            df: DataFrame to analyze
            target_column: Target/label column name
            protected_attributes: List of protected attribute column names
            
        Returns:
            BiasMetrics with comprehensive bias assessment
        """
        metrics = BiasMetrics()
        
        # Ensure target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return metrics
            
        # Skip if no protected attributes
        if not protected_attributes:
            logger.warning("No protected attributes specified, skipping bias assessment")
            return metrics

        # Analyze each protected attribute
        has_fairlearn = FAIRLEARN_AVAILABLE
        
        # Calculate statistical parity (demographic disparity)
        for attr in protected_attributes:
            if attr not in df.columns:
                logger.warning(f"Protected attribute '{attr}' not found in dataset, skipping")
                continue
                
            # For each unique value in the protected attribute
            attr_values = df[attr].dropna().unique()
            
            if len(attr_values) < 2:
                logger.warning(f"Protected attribute '{attr}' has fewer than 2 unique values, skipping")
                continue
                
            # Calculate overall distribution of positive outcome
            overall_positive_rate = df[target_column].mean()
            
            # Calculate positive rate for each group
            group_rates = {}
            max_disparity = 0
            
            for value in attr_values:
                group_df = df[df[attr] == value]
                if len(group_df) < 10:  # Skip groups that are too small
                    continue
                    
                group_positive_rate = group_df[target_column].mean()
                group_rates[str(value)] = group_positive_rate
                
                # Update maximum disparity
                disparity = abs(group_positive_rate - overall_positive_rate)
                max_disparity = max(max_disparity, disparity)
            
            # Store statistical parity results
            metrics.statistical_parity[attr] = max_disparity
            
            # Calculate disparate impact if binary classification
            if len(df[target_column].unique()) <= 2:
                if len(attr_values) == 2:
                    # Identify privileged and unprivileged groups
                    rates = [(v, r) for v, r in group_rates.items()]
                    rates.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(rates) >= 2 and rates[1][1] > 0:  # Avoid division by zero
                        disparate_impact = rates[0][1] / rates[1][1]
                        metrics.disparate_impact[attr] = disparate_impact
            
            # Use fairlearn for additional metrics if available
            if has_fairlearn and len(df[target_column].unique()) <= 2:
                try:
                    # Convert to binary if needed
                    y_true = df[target_column].astype(int)
                    
                    # Create sensitive feature groups
                    sensitive_features = df[attr].astype(str)
                    
                    # Calculate fairlearn metrics
                    dpd = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)
                    metrics.equal_opportunity[attr] = dpd
                    
                    eod = equalized_odds_difference(y_true, y_true, sensitive_features=sensitive_features)
                    metrics.equalized_odds[attr] = eod
                except Exception as e:
                    logger.warning(f"Error calculating fairlearn metrics for {attr}: {e}")
        
        # Calculate intersectional bias for combinations of protected attributes
        if len(protected_attributes) >= 2 and has_fairlearn:
            # Get valid combinations of protected attributes
            valid_attrs = [attr for attr in protected_attributes if attr in df.columns]
            
            if len(valid_attrs) >= 2:
                # Create an intersectional column
                for i in range(len(valid_attrs)):
                    for j in range(i+1, len(valid_attrs)):
                        attr1, attr2 = valid_attrs[i], valid_attrs[j]
                        intersection = df[attr1].astype(str) + '_' + df[attr2].astype(str)
                        
                        try:
                            # Calculate demographic parity for the intersection
                            y_true = df[target_column].astype(int)
                            dpd = demographic_parity_difference(y_true, y_true, sensitive_features=intersection)
                            metrics.intersectional_bias_scores[f"{attr1}_{attr2}"] = dpd
                        except Exception as e:
                            logger.warning(f"Error calculating intersectional bias: {e}")
        
        # Calculate feature importance bias using SHAP if available
        if SHAP_AVAILABLE and len(protected_attributes) > 0:
            try:
                # Create a simple model to explain
                from sklearn.ensemble import RandomForestClassifier
                
                # Prepare data
                X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
                y = df[target_column]
                
                # Train a simple model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y# filepath: c:\Users\adilm\repositories\Python\neural-scope\data_guardian_2.py
"""
DataGuardian: Enterprise-Grade ML Data Quality & Ethical Analysis System

A comprehensive, state-of-the-art data quality assessment framework designed for 
production ML workflows. DataGuardian performs deep inspection of datasets to 
identify quality issues, biases, and ethical concerns while providing actionable 
remediation recommendations with code examples.

Key capabilities:
1. Multi-dimensional bias detection across intersectional protected attributes
2. Advanced data quality assessment with ML-based anomaly detection
3. Distribution drift monitoring for production ML systems
4. Privacy risk assessment and PII detection
5. Fairness constraint implementation with multiple mathematical definitions
6. Explainable AI integration for bias investigation
7. Customizable reporting with interactive visualizations
8. Remediation recommendations with executable code examples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy import stats
import re
import logging
import json
from pathlib import Path
import hashlib
from datetime import datetime
import warnings
import joblib

# Optional dependencies with graceful fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import fairlearn
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataGuardian")


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics container"""
    completeness: Dict[str, float] = field(default_factory=dict)
    uniqueness: Dict[str, float] = field(default_factory=dict)
    consistency: Dict[str, Dict] = field(default_factory=dict)
    validity: Dict[str, float] = field(default_factory=dict)
    outlier_scores: Dict[str, List[int]] = field(default_factory=dict)
    distribution_metrics: Dict[str, Dict] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    quality_score: float = 1.0
    

@dataclass
class BiasMetrics:
    """Container for bias and fairness metrics"""
    # Statistical bias metrics
    statistical_parity: Dict[str, float] = field(default_factory=dict)
    disparate_impact: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, float] = field(default_factory=dict)
    
    # Intersectional bias metrics
    intersectional_bias_scores: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance bias
    feature_importance_bias: Dict[str, float] = field(default_factory=dict)
    
    # Overall bias score (0-1 where 0 is perfectly unbiased)
    overall_bias_score: float = 0.0
    

@dataclass
class PrivacyMetrics:
    """Container for privacy and security risk metrics"""
    pii_detected: Dict[str, List[str]] = field(default_factory=dict)
    uniqueness_risk: Dict[str, float] = field(default_factory=dict)
    k_anonymity: int = 0
    l_diversity: Dict[str, int] = field(default_factory=dict)
    overall_privacy_risk: float = 0.0


@dataclass
class DataDriftMetrics:
    """Container for distribution shift and drift metrics"""
    feature_drift: Dict[str, float] = field(default_factory=dict)
    distribution_shifts: Dict[str, Dict] = field(default_factory=dict)
    overall_drift_score: float = 0.0


@dataclass
class DataRecommendation:
    """Structured recommendation with actionable code examples"""
    issue_type: str  # 'quality', 'bias', 'privacy', 'drift'
    severity: str    # 'low', 'medium', 'high', 'critical'
    feature: Optional[str] = None
    description: str = ""
    impact: str = ""
    recommendation: str = ""
    code_example: str = ""
    reference_url: str = ""


@dataclass
class DataGuardianReport:
    """Comprehensive report containing all analysis results"""
    dataset_name: str
    timestamp: datetime
    dataset_stats: Dict[str, Any]
    quality_metrics: DataQualityMetrics
    bias_metrics: Optional[BiasMetrics] = None
    privacy_metrics: Optional[PrivacyMetrics] = None
    drift_metrics: Optional[DataDriftMetrics] = None
    recommendations: List[DataRecommendation] = field(default_factory=list)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        result = {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "dataset_stats": self.dataset_stats,
            "quality_metrics": {
                "completeness": self.quality_metrics.completeness,
                "uniqueness": self.quality_metrics.uniqueness,
                "distribution_metrics": self.quality_metrics.distribution_metrics,
                "quality_score": self.quality_metrics.quality_score
            },
            "recommendations": [
                {
                    "issue_type": r.issue_type,
                    "severity": r.severity,
                    "feature": r.feature,
                    "description": r.description,
                    "recommendation": r.recommendation,
                    "code_example": r.code_example
                } for r in self.recommendations
            ],
            "execution_time": self.execution_time
        }
        
        # Add optional metrics if available
        if self.bias_metrics:
            result["bias_metrics"] = {
                "statistical_parity": self.bias_metrics.statistical_parity,
                "disparate_impact": self.bias_metrics.disparate_impact,
                "overall_bias_score": self.bias_metrics.overall_bias_score
            }
            
        if self.privacy_metrics:
            result["privacy_metrics"] = {
                "pii_detected": self.privacy_metrics.pii_detected,
                "k_anonymity": self.privacy_metrics.k_anonymity,
                "overall_privacy_risk": self.privacy_metrics.overall_privacy_risk
            }
            
        if self.drift_metrics:
            result["drift_metrics"] = {
                "feature_drift": self.drift_metrics.feature_drift,
                "overall_drift_score": self.drift_metrics.overall_drift_score
            }
            
        return result
    
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save report as JSON file or return JSON string"""
        report_dict = self.to_dict()
        json_str = json.dumps(report_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        
        return json_str
    
    def create_summary(self) -> str:
        """Generate a text summary of the report"""
        summary = [
            f"DataGuardian Report for '{self.dataset_name}'",
            f"Generated on: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset shape: {self.dataset_stats['num_rows']} rows × {self.dataset_stats['num_columns']} columns",
            "",
            "=== DATA QUALITY SUMMARY ===",
            f"Overall data quality score: {self.quality_metrics.quality_score:.2f}/1.00",
            f"Columns with missing values: {sum(1 for v in self.quality_metrics.completeness.values() if v < 1.0)}",
            f"Columns with outliers: {len(self.quality_metrics.outlier_scores)}",
        ]
        
        if self.bias_metrics:
            summary.extend([
                "",
                "=== BIAS ASSESSMENT SUMMARY ===",
                f"Overall bias score: {self.bias_metrics.overall_bias_score:.2f}/1.00",
                f"Protected attributes analyzed: {len(self.bias_metrics.statistical_parity)}"
            ])
        
        if self.privacy_metrics:
            summary.extend([
                "",
                "=== PRIVACY ASSESSMENT SUMMARY ===",
                f"Overall privacy risk: {self.privacy_metrics.overall_privacy_risk:.2f}/1.00",
                f"PII columns detected: {len(self.privacy_metrics.pii_detected)}",
                f"K-anonymity: {self.privacy_metrics.k_anonymity}"
            ])
        
        if self.drift_metrics:
            summary.extend([
                "",
                "=== DATA DRIFT SUMMARY ===",
                f"Overall drift score: {self.drift_metrics.overall_drift_score:.2f}/1.00",
                f"Features with significant drift: {sum(1 for v in self.drift_metrics.feature_drift.values() if v > 0.1)}"
            ])
        
        summary.extend([
            "",
            "=== RECOMMENDATIONS SUMMARY ===",
            f"Total recommendations: {len(self.recommendations)}",
            f"Critical issues: {sum(1 for r in self.recommendations if r.severity == 'critical')}",
            f"High severity issues: {sum(1 for r in self.recommendations if r.severity == 'high')}",
            f"Medium severity issues: {sum(1 for r in self.recommendations if r.severity == 'medium')}",
            f"Low severity issues: {sum(1 for r in self.recommendations if r.severity == 'low')}"
        ])
        
        return "\n".join(summary)


class DataGuardian:
    """
    Advanced data quality, bias detection, and ethical analysis framework for ML datasets.
    
    DataGuardian provides comprehensive assessment of ML datasets to ensure they meet
    quality standards, are free from harmful biases, respect privacy, and remain stable
    over time through distribution drift detection.
    """
    
    def __init__(self, 
                 dataset_name: str = "unnamed_dataset",
                 protected_attributes: Optional[List[str]] = None,
                 sensitive_attributes: Optional[List[str]] = None,
                 categorical_threshold: int = 20,
                 outlier_threshold: float = 0.03,
                 save_reports: bool = True,
                 reports_dir: str = "./data_guardian_reports",
                 verbose: bool = True):
        """
        Initialize DataGuardian with configuration parameters
        
        Args:
            dataset_name: Name identifier for the dataset
            protected_attributes: List of column names to be treated as protected attributes for bias analysis
            sensitive_attributes: List of column names containing sensitive/PII data for privacy analysis
            categorical_threshold: Maximum number of unique values to consider a column categorical
            outlier_threshold: Threshold for outlier detection (ratio of outliers to flag)
            save_reports: Whether to save reports to disk
            reports_dir: Directory to save generated reports
            verbose: Whether to print detailed logs during analysis
        """
        self.dataset_name = dataset_name
        self.protected_attributes = protected_attributes or []
        self.sensitive_attributes = sensitive_attributes or []
        self.categorical_threshold = categorical_threshold
        self.outlier_threshold = outlier_threshold
        self.save_reports = save_reports
        self.reports_dir = reports_dir
        self.verbose = verbose
        
        # Create reports directory if it doesn't exist
        if self.save_reports:
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
        # Configuration for common data types and patterns
        self._setup_type_patterns()
        
        # Set log level based on verbosity
        if not verbose:
            logger.setLevel(logging.WARNING)
        
        logger.info(f"DataGuardian initialized for dataset '{dataset_name}'")
    
    def _setup_type_patterns(self):
        """Setup regex patterns for various data types"""
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d{4}[ -]?){3}\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "address": r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way)\b',
            "name": r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
        }
    
    def analyze_dataset(self, 
                        df: pd.DataFrame, 
                        target_column: Optional[str] = None,
                        reference_df: Optional[pd.DataFrame] = None,
                        protected_attributes: Optional[List[str]] = None,
                        sensitive_attributes: Optional[List[str]] = None) -> DataGuardianReport:
        """
        Perform comprehensive analysis of a dataset
        
        Args:
            df: Pandas DataFrame to analyze
            target_column: Name of the target/label column for bias analysis
            reference_df: Optional reference dataset for drift detection
            protected_attributes: Override class-level protected attributes
            sensitive_attributes: Override class-level sensitive attributes
            
        Returns:
            DataGuardianReport with comprehensive analysis results
        """
        start_time = datetime.now()
        logger.info(f"Starting comprehensive analysis of dataset with shape {df.shape}")
        
        # Use parameters if provided, else use class attributes
        protected_attrs = protected_attributes or self.protected_attributes
        sensitive_attrs = sensitive_attributes or self.sensitive_attributes
        
        # Basic dataset statistics
        dataset_stats = self._compute_dataset_statistics(df)
        
        # Data type inference
        numerical_features, categorical_features = self._infer_column_types(df)
        
        # Comprehensive data quality assessment
        quality_metrics = self._assess_data_quality(df, numerical_features, categorical_features)
        
        # Initialize containers for optional analyses
        bias_metrics = None
        privacy_metrics = None
        drift_metrics = None
        
        # Bias assessment if target column and protected attributes are specified
        if target_column is not None and protected_attrs:
            bias_metrics = self._assess_bias(df, target_column, protected_attrs)
        
        # Privacy assessment if sensitive attributes are specified or auto-detection is enabled
        if sensitive_attrs or self._should_detect_pii():
            privacy_metrics = self._assess_privacy(df, sensitive_attrs)
        
        # Drift assessment if reference dataset is provided
        if reference_df is not None:
            drift_metrics = self._assess_drift(df, reference_df)
        
        # Generate recommendations based on all findings
        recommendations = self._generate_recommendations(
            df, quality_metrics, bias_metrics, privacy_metrics, drift_metrics,
            numerical_features, categorical_features, target_column
        )
        
        # Create comprehensive report
        report = DataGuardianReport(
            dataset_name=self.dataset_name,
            timestamp=datetime.now(),
            dataset_stats=dataset_stats,
            quality_metrics=quality_metrics,
            bias_metrics=bias_metrics,
            privacy_metrics=privacy_metrics,
            drift_metrics=drift_metrics,
            recommendations=recommendations,
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        # Save report if configured
        if self.save_reports:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.reports_dir}/{self.dataset_name}_{timestamp}.json"
            report.to_json(filename)
            logger.info(f"Report saved to {filename}")
        
        logger.info(f"Analysis completed in {report.execution_time:.2f} seconds")
        return report
    
    def _compute_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute general statistics about the dataset"""
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "column_types": df.dtypes.astype(str).to_dict(),
            "column_unique_counts": df.nunique().to_dict(),
            "column_missing_counts": df.isna().sum().to_dict()
        }
        return stats
    
    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Intelligently infer numerical and categorical columns
        Goes beyond simple dtype checking to handle encoded categories
        """
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            # Skip columns with too many missing values
            if df[col].isna().mean() > 0.5:
                continue
                
            # Check if numeric type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (encoded as numbers)
                unique_count = df[col].nunique()
                if unique_count <= self.categorical_threshold:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
                
        logger.info(f"Inferred {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        return numerical_features, categorical_features

    def _assess_data_quality(self, 
                             df: pd.DataFrame, 
                             numerical_features: List[str],
                             categorical_features: List[str]) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment
        
        Checks for:
        - Completeness (missing values)
        - Uniqueness
        - Consistency (format, ranges)
        - Validity (data format compliance)
        - Outliers (statistical and ML-based)
        - Distribution characteristics
        """
        metrics = DataQualityMetrics()
        
        # Assess completeness (missing values)
        metrics.completeness = {col: 1 - df[col].isna().mean() for col in df.columns}
        
        # Assess uniqueness
        metrics.uniqueness = {col: df[col].nunique() / len(df) for col in df.columns}
        
        # Find outliers in numerical features
        metrics.outlier_scores = {}
        for col in numerical_features:
            if df[col].isna().mean() > 0.3:  # Skip columns with too many missing values
                continue
                
            # Use IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            if len(outliers) / len(df) > self.outlier_threshold:
                metrics.outlier_scores[col] = outliers
        
        # Calculate distribution metrics for numerical features
        metrics.distribution_metrics = {}
        for col in numerical_features:
            if df[col].isna().mean() > 0.3:
                continue
                
            metrics.distribution_metrics[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': float(stats.skew(df[col].dropna())),
                'kurtosis': float(stats.kurtosis(df[col].dropna())),
            }
            
        # Calculate distribution metrics for categorical features
        for col in categorical_features:
            if df[col].isna().mean() > 0.3:
                continue
                
            value_counts = df[col].value_counts(normalize=True)
            entropy = stats.entropy(value_counts)
            metrics.distribution_metrics[col] = {
                'entropy': float(entropy),
                'top_value': value_counts.index[0],
                'top_value_pct': float(value_counts.iloc[0]),
                'unique_count': df[col].nunique(),
            }
        
        # Assess consistency and validity
        metrics.consistency = self._check_consistency(df)
        metrics.validity = self._check_validity(df)
        
        # Compute correlation matrix if there are enough numerical features
        if len(numerical_features) > 1:
            metrics.correlation_matrix = df[numerical_features].corr()
        
        # Compute overall quality score based on all metrics
        quality_scores = []
        # Completeness score
        completeness_score = np.mean(list(metrics.completeness.values()))
        quality_scores.append(completeness_score)
        
        # Outlier score (penalize datasets with many outliers)
        outlier_ratio = sum(len(outliers) for outliers in metrics.outlier_scores.values()) / (len(df) * len(numerical_features)) if numerical_features else 0
        outlier_score = 1 - outlier_ratio
        quality_scores.append(outlier_score)
        
        # Distribution score (penalize highly skewed features)
        skewness_scores = []
        for col, stats_dict in metrics.distribution_metrics.items():
            if 'skewness' in stats_dict:
                # Normalize skewness impact: 0 (highly skewed) to 1 (normal)
                skew_score = 1 - min(abs(stats_dict['skewness']), 10) / 10
                skewness_scores.append(skew_score)
        
        distribution_score = np.mean(skewness_scores) if skewness_scores else 1.0
        quality_scores.append(distribution_score)
        
        # Overall quality score
        metrics.quality_score = float(np.mean(quality_scores))
        
        logger.info(f"Data quality assessment completed. Overall score: {metrics.quality_score:.4f}")
        return metrics
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Check data consistency within columns"""
        consistency = {}
        
        # Check date format consistency
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100).astype(str)
                
                # Date format consistency check
                date_formats = {}
                for val in sample:
                    # Skip empty or very long values
                    if not val or len(val) > 30:
                        continue
                        
                    # Check if it might be a date
                    if re.match(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', val):
                        format_key = re.sub(r'\d', '#', val)
                        date_formats[format_key] = date_formats.get(format_key, 0) + 1
                
                if date_formats:
                    consistency[col] = {
                        'potential_date_column': True,
                        'date_formats': date_formats,
                        'consistent_format': len(date_formats) == 1
                    }
        
        return consistency
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check data validity based on expected patterns"""
        validity = {}
        
        # Check for valid email format in string columns that might contain emails
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column name suggests it might contain emails
                if 'email' in col.lower():
                    # Sample values and check against email regex
                    sample = df[col].dropna().astype(str).head(1000)
                    valid_count = sum(1 for val in sample if re.match(self.pii_patterns['email'], val))
                    if valid_count > 0:
                        validity[col] = valid_count / len(sample)
                        
        # Add other validity checks as needed (phone numbers, URLs, etc.)
                        
        return validity
    
    def _assess_bias(self, 
                     df: pd.DataFrame, 
                     target_column: str, 
                     protected_attributes: List[str]) -> BiasMetrics:
        """
        Perform comprehensive bias assessment across protected attributes
        
        Args:
            df: DataFrame to analyze
            target_column: Target/label column name
            protected_attributes: List of protected attribute column names
            
        Returns:
            BiasMetrics with comprehensive bias assessment
        """
        metrics = BiasMetrics()
        
        # Ensure target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return metrics
            
        # Skip if no protected attributes
        if not protected_attributes:
            logger.warning("No protected attributes specified, skipping bias assessment")
            return metrics

        # Analyze each protected attribute
        has_fairlearn = FAIRLEARN_AVAILABLE
        
        # Calculate statistical parity (demographic disparity)
        for attr in protected_attributes:
            if attr not in df.columns:
                logger.warning(f"Protected attribute '{attr}' not found in dataset, skipping")
                continue
                
            # For each unique value in the protected attribute
            attr_values = df[attr].dropna().unique()
            
            if len(attr_values) < 2:
                logger.warning(f"Protected attribute '{attr}' has fewer than 2 unique values, skipping")
                continue
                
            # Calculate overall distribution of positive outcome
            overall_positive_rate = df[target_column].mean()
            
            # Calculate positive rate for each group
            group_rates = {}
            max_disparity = 0
            
            for value in attr_values:
                group_df = df[df[attr] == value]
                if len(group_df) < 10:  # Skip groups that are too small
                    continue
                    
                group_positive_rate = group_df[target_column].mean()
                group_rates[str(value)] = group_positive_rate
                
                # Update maximum disparity
                disparity = abs(group_positive_rate - overall_positive_rate)
                max_disparity = max(max_disparity, disparity)
            
            # Store statistical parity results
            metrics.statistical_parity[attr] = max_disparity
            
            # Calculate disparate impact if binary classification
            if len(df[target_column].unique()) <= 2:
                if len(attr_values) == 2:
                    # Identify privileged and unprivileged groups
                    rates = [(v, r) for v, r in group_rates.items()]
                    rates.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(rates) >= 2 and rates[1][1] > 0:  # Avoid division by zero
                        disparate_impact = rates[0][1] / rates[1][1]
                        metrics.disparate_impact[attr] = disparate_impact
            
            # Use fairlearn for additional metrics if available
            if has_fairlearn and len(df[target_column].unique()) <= 2:
                try:
                    # Convert to binary if needed
                    y_true = df[target_column].astype(int)
                    
                    # Create sensitive feature groups
                    sensitive_features = df[attr].astype(str)
                    
                    # Calculate fairlearn metrics
                    dpd = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)
                    metrics.equal_opportunity[attr] = dpd
                    
                    eod = equalized_odds_difference(y_true, y_true, sensitive_features=sensitive_features)
                    metrics.equalized_odds[attr] = eod
                except Exception as e:
                    logger.warning(f"Error calculating fairlearn metrics for {attr}: {e}")
        
        # Calculate intersectional bias for combinations of protected attributes
        if len(protected_attributes) >= 2 and has_fairlearn:
            # Get valid combinations of protected attributes
            valid_attrs = [attr for attr in protected_attributes if attr in df.columns]
            
            if len(valid_attrs) >= 2:
                # Create an intersectional column
                for i in range(len(valid_attrs)):
                    for j in range(i+1, len(valid_attrs)):
                        attr1, attr2 = valid_attrs[i], valid_attrs[j]
                        intersection = df[attr1].astype(str) + '_' + df[attr2].astype(str)
                        
                        try:
                            # Calculate demographic parity for the intersection
                            y_true = df[target_column].astype(int)
                            dpd = demographic_parity_difference(y_true, y_true, sensitive_features=intersection)
                            metrics.intersectional_bias_scores[f"{attr1}_{attr2}"] = dpd
                        except Exception as e:
                            logger.warning(f"Error calculating intersectional bias: {e}")
        
        # Calculate feature importance bias using SHAP if available
        if SHAP_AVAILABLE and len(protected_attributes) > 0:
            try:
                # Create a simple model to explain
                from sklearn.ensemble import RandomForestClassifier
                
                # Prepare data
                X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
                y = df[target_column]
                
                # Train a simple model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                    def _recommend_automatic_fixes(self) -> Dict:
        """
        Generate recommendations for which fixes to apply automatically.
        
        Returns:
            Dictionary with recommended fixes and their rationale
        """
        recommendations = {
            "recommended_fixes": [],
            "optional_fixes": [],
            "manual_review_required": [],
            "rationale": {}
        }
        
        # Check for duplicates - safe to remove
        duplicate_count = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicate_count > 0:
            recommendations["recommended_fixes"].append("duplicates")
            recommendations["rationale"]["duplicates"] = f"Found {duplicate_count} exact duplicate rows that can be safely removed."
        
        # Check for missing values
        missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
        if missing_patterns and missing_patterns.get("imputation_recommendations"):
            # Only auto-recommend if missingness is below a threshold
            missing_pcts = missing_patterns.get("missing_percentages", {})
            avg_missing = sum(missing_pcts.values()) / len(missing_pcts) if missing_pcts else 0
            
            if avg_missing > 0:
                if avg_missing < 10:
                    recommendations["recommended_fixes"].append("missing")
                    recommendations["rationale"]["missing"] = "Low missing data percentage can be safely imputed automatically."
                elif avg_missing < 30:
                    recommendations["optional_fixes"].append("missing")
                    recommendations["rationale"]["missing"] = f"Moderate missing data ({avg_missing:.1f}%) - imputation may affect analysis."
                else:
                    recommendations["manual_review_required"].append("missing")
                    recommendations["rationale"]["missing"] = f"High missing data percentage ({avg_missing:.1f}%) - requires careful review before imputation."
        
        # Check for storage optimization - always safe
        storage_info = self.report.get("storage_efficiency", {})
        if storage_info:
            savings_pct = storage_info.get("savings_percentage", 0)
            if savings_pct > 5:
                recommendations["recommended_fixes"].append("storage")
                recommendations["rationale"]["storage"] = f"Potential memory savings of {savings_pct:.1f}% with dtype optimization."
        
        # Check for outliers
        outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
        if outliers:
            outlier_cols = [col for col, info in outliers.items() if info.get("percentage", 0) > 5]
            if outlier_cols:
                recommendations["optional_fixes"].append("outliers")
                recommendations["rationale"]["outliers"] = f"Found significant outliers in {len(outlier_cols)} columns - consider winsorizing if they're not valid data points."
        
        # Check for corrupted values
        corrupt_values = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
        if corrupt_values:
            recommendations["optional_fixes"].append("corrupt")
            recommendations["rationale"]["corrupt"] = f"Found potentially corrupted values in {len(corrupt_values)} columns - review before cleaning."
        
        # Check for class imbalance
        if self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance.get("is_imbalanced", False):
                imbalance_ratio = class_balance.get("imbalance_ratio", 1.0)
                
                if imbalance_ratio > 10:
                    recommendations["manual_review_required"].append("class_imbalance")
                    recommendations["rationale"]["class_imbalance"] = f"Severe class imbalance (ratio: {imbalance_ratio:.1f}) - sampling techniques may significantly alter data distribution."
                elif imbalance_ratio > 3:
                    recommendations["optional_fixes"].append("class_imbalance")
                    recommendations["rationale"]["class_imbalance"] = f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}) - sampling may improve model performance."
        
        return recommendations

    def generate_interactive_visualization(self) -> Dict:
        """
        Generate interactive visualizations for the full dataset analysis.
        
        Returns:
            Dictionary of Plotly figures for interactive exploration
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            figures = {}
            
            # 1. Missing data heatmap
            if "missing_patterns" in self.report["data_integrity"]:
                # Create a binary missing data matrix
                missing_matrix = self.df.isna().astype(int)
                
                # Get columns with missing values
                missing_cols = [col for col in self.df.columns if self.df[col].isna().any()]
                
                if missing_cols:
                    # Sample up to 1000 rows for better visibility
                    sample_size = min(1000, len(missing_matrix))
                    if len(missing_matrix) > sample_size:
                        missing_sample = missing_matrix.sample(sample_size)[missing_cols]
                    else:
                        missing_sample = missing_matrix[missing_cols]
                    
                    # Create heatmap
                    fig = px.imshow(
                        missing_sample.T,
                        labels=dict(x="Row index", y="Features", color="Missing"),
                        color_continuous_scale=["#FFFFFF", "#6E67EB"],
                        title="Missing Data Pattern (1=Missing)",
                        width=900,
                        height=max(400, len(missing_cols) * 20)
                    )
                    fig.update_layout(
                        xaxis_title="Rows (sample)",
                        yaxis_title="Features",
                        coloraxis_showscale=True
                    )
                    figures["missing_heatmap"] = fig
                    
                    # Missing percentage bar chart
                    missing_pcts = self.report["data_integrity"]["missing_patterns"]["missing_percentages"]
                    if missing_pcts:
                        sorted_data = [(k, v) for k, v in sorted(missing_pcts.items(), key=lambda x: x[1], reverse=True) if v > 0]
                        if sorted_data:
                            columns, percentages = zip(*sorted_data)
                            
                            fig = px.bar(
                                x=columns, 
                                y=percentages,
                                labels={"x": "Column", "y": "Missing (%)"},
                                title="Missing Values by Column",
                                color=percentages,
                                color_continuous_scale=px.colors.sequential.Viridis,
                                width=900,
                                height=500
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["missing_percentages"] = fig
            
            # 2. Interactive distribution explorer
            if self.numeric_cols:
                # Create distribution figures for numeric columns
                for col in self.numeric_cols[:10]:  # Limit to 10 for performance
                    fig = px.histogram(
                        self.df, 
                        x=col,
                        marginal="box",  # Add boxplot on top of histogram
                        histnorm="probability density",
                        title=f"Distribution of {col}",
                        width=700,
                        height=500
                    )
                    
                    # Add KDE curve if scipy is available
                    try:
                        from scipy import stats
                        
                        # Calculate KDE
                        values = self.df[col].dropna()
                        if len(values) > 1:  # Need at least 2 points for KDE
                            kde = stats.gaussian_kde(values)
                            x_range = np.linspace(values.min(), values.max(), 1000)
                            kde_values = kde(x_range)
                            
                            # Add KDE trace
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=kde_values,
                                    mode='lines',
                                    name='KDE',
                                    line=dict(color='red', width=2)
                                )
                            )
                    except ImportError:
                        pass
                    
                    figures[f"distribution_{col}"] = fig
                
                # 3. Correlation matrix with customizable threshold
                if len(self.numeric_cols) > 1:
                    corr_matrix = self.df[self.numeric_cols].corr().round(2)
                    
                    # Create mask for upper triangle
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # Set upper triangle to NaN
                    corr_matrix_masked = corr_matrix.copy()
                    corr_matrix_masked.values[mask] = np.nan
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix_masked,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        range_color=[-1, 1],
                        title="Feature Correlation Matrix"
                    )
                    
                    # Add correlation text
                    for i, row in enumerate(corr_matrix_masked.index):
                        for j, col in enumerate(corr_matrix_masked.columns):
                            if not pd.isna(corr_matrix_masked.iloc[i, j]):
                                fig.add_annotation(
                                    x=j, 
                                    y=i,
                                    text=str(corr_matrix.iloc[i, j]),
                                    showarrow=False,
                                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                                )
                    
                    figures["correlation_matrix"] = fig
            
            # 4. Class distribution (if target exists)
            if self.target_column and self.target_column in self.df.columns:
                # Handle differently based on numeric or categorical target
                if self.target_column in self.numeric_cols:
                    # For numeric target, show histogram and boxplot
                    fig = make_subplots(
                        rows=2, 
                        cols=1,
                        subplot_titles=["Distribution", "Boxplot"],
                        vertical_spacing=0.2,
                        specs=[[{"type": "histogram"}], [{"type": "box"}]]
                    )
                    
                    fig.add_trace(
                        go.Histogram(
                            x=self.df[self.target_column],
                            name="Distribution",
                            histnorm="probability density"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Box(
                            x=self.df[self.target_column],
                            name="Boxplot",
                            boxpoints="outliers"
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title_text=f"Target Variable: {self.target_column}",
                        showlegend=False,
                        height=700
                    )
                else:
                    # For categorical target, show bar chart and pie chart
                    value_counts = self.df[self.target_column].value_counts()
                    
                    fig = make_subplots(
                        rows=1, 
                        cols=2,
                        subplot_titles=["Bar Chart", "Pie Chart"],
                        specs=[[{"type": "bar"}, {"type": "pie"}]],
                        column_widths=[0.6, 0.4]
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            y=value_counts.index,
                            x=value_counts.values,
                            orientation='h',
                            name="Count",
                            marker=dict(
                                color=px.colors.qualitative.Plotly[:len(value_counts)]
                            )
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Pie(
                            labels=value_counts.index,
                            values=value_counts.values,
                            name="Percentage",
                            marker=dict(
                                colors=px.colors.qualitative.Plotly[:len(value_counts)]
                            )
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title_text=f"Target Variable: {self.target_column}",
                        height=500
                    )
                
                figures["target_distribution"] = fig
            
            # 5. Feature importance and relationship with target (if available)
            if self.target_column and "feature_analysis" in self.report and len(self.numeric_cols) > 1:
                try:
                    # For categorical target, use mutual information
                    if self.target_column not in self.numeric_cols:
                        from sklearn.feature_selection import mutual_info_classif
                        
                        # Get numeric features
                        features = [col for col in self.numeric_cols if col != self.target_column]
                        
                        if features:
                            # Calculate mutual information
                            X = self.df[features].fillna(0)  # Simple imputation for calculation
                            y = self.df[self.target_column].fillna(self.df[self.target_column].mode().iloc[0])
                            
                            mi_scores = mutual_info_classif(X, y)
                            mi_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': mi_scores
                            }).sort_values('Importance', ascending=False)
                            
                            # Create bar chart of feature importance
                            fig = px.bar(
                                mi_df,
                                x='Feature',
                                y='Importance',
                                title=f"Feature Importance (Mutual Information) with {self.target_column}",
                                color='Importance',
                                labels={'Importance': 'Mutual Information Score'}
                            )
                            
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["feature_importance"] = fig
                    
                    # For numeric target, use correlation
                    else:
                        features = [col for col in self.numeric_cols if col != self.target_column]
                        
                        if features:
                            # Calculate correlations with target
                            correlations = {}
                            for feature in features:
                                correlations[feature] = self.df[[feature, self.target_column]].corr().iloc[0, 1]
                            
                            corr_df = pd.DataFrame({
                                'Feature': list(correlations.keys()),
                                'Correlation': list(correlations.values())
                            }).sort_values('Correlation', key=lambda x: abs(x), ascending=False)
                            
                            # Create bar chart of correlations
                            fig = px.bar(
                                corr_df,
                                x='Feature',
                                y='Correlation',
                                title=f"Feature Correlation with {self.target_column}",
                                color='Correlation',
                                color_continuous_scale=px.colors.diverging.RdBu_r,
                                range_color=[-1, 1]
                            )
                            
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["feature_correlation"] = fig
                except Exception as e:
                    # Skip if calculation fails
                    pass
            
            # 6. Outlier visualization
            outlier_data = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
            if outlier_data:
                # Get columns with highest outlier percentages
                outlier_cols = sorted(
                    [(col, data["percentage"]) for col, data in outlier_data.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5 columns
                
                if outlier_cols:
                    # Create subplot figure
                    fig = make_subplots(
                        rows=len(outlier_cols),
                        cols=1,
                        subplot_titles=[f"{col} ({pct:.1f}% outliers)" for col, pct in outlier_cols],
                        vertical_spacing=0.1
                    )
                    
                    for i, (col, _) in enumerate(outlier_cols):
                        # Add box plot
                        fig.add_trace(
                            go.Box(
                                y=self.df[col].dropna(),
                                name=col,
                                boxpoints='outliers',
                                jitter=0.3,
                                pointpos=-1.8,
                                marker=dict(
                                    color='rgba(255, 0, 0, 0.6)',
                                    size=4
                                ),
                                line=dict(color='rgb(0, 0, 255)')
                            ),
                            row=i+1,
                            col=1
                        )
                        
                        # Add normal range reference if available
                        if "normal_range" in outlier_data[col]:
                            try:
                                lower, upper = map(float, outlier_data[col]["normal_range"].split(" - "))
                                
                                # Add range area
                                fig.add_shape(
                                    type="rect",
                                    xref=f"x{i+1}",
                                    yref=f"y{i+1}",
                                    x0=0,
                                    x1=1,
                                    y0=lower,
                                    y1=upper,
                                    fillcolor="rgba(0, 255, 0, 0.2)",
                                    layer="below",
                                    line_width=0
                                )
                            except:
                                pass
                    
                    fig.update_layout(
                        height=300 * len(outlier_cols),
                        title_text='Top Features with Outliers',
                        showlegend=False
                    )
                    
                    figures["outlier_analysis"] = fig
            
            return figures
            
        except ImportError:
            # Return empty dict if Plotly is not available
            return {"error": "Plotly is required for interactive visualizations. Please install with: pip install plotly"}

    def generate_advanced_data_profile(self) -> Dict:
        """
        Generate an advanced data profile with comprehensive statistical summaries and quality metrics.
        
        Returns:
            Dictionary with detailed profiling information
        """
        profile = {
            "overview": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "duplicated_rows": int(self.df.duplicated().sum()),
                "duplicated_rows_pct": round(100 * self.df.duplicated().sum() / len(self.df), 2) if len(self.df) > 0 else 0
            },
            "quality_metrics": self._calculate_quality_score(),
            "column_profiles": {},
            "correlation_overview": {},
            "missing_data_overview": {}
        }
        
        # Generate detailed column profiles
        for col in self.df.columns:
            col_data = self.df[col]
            col_profile = {
                "type": str(col_data.dtype),
                "count": len(col_data),
                "missing": int(col_data.isna().sum()),
                "missing_pct": round(100 * col_data.isna().sum() / len(col_data), 2),
                "unique_values": int(col_data.nunique()),
                "unique_pct": round(100 * col_data.nunique() / len(col_data.dropna()), 2) if len(col_data.dropna()) > 0 else 0
            }
            
            # Add numeric-specific stats
            if col in self.numeric_cols:
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    col_profile.update({
                        "min": float(numeric_data.min()),
                        "max": float(numeric_data.max()),
                        "mean": float(numeric_data.mean()),
                        "median": float(numeric_data.median()),
                        "std": float(numeric_data.std()),
                        "skew": float(numeric_data.skew()),
                        "kurtosis": float(numeric_data.kurtosis()),
                        "zeros_count": int((numeric_data == 0).sum()),
                        "zeros_pct": round(100 * (numeric_data == 0).sum() / len(numeric_data), 2),
                        "negatives_count": int((numeric_data < 0).sum()),
                        "negatives_pct": round(100 * (numeric_data < 0).sum() / len(numeric_data), 2),
                        "quantiles": {
                            "1%": float(numeric_data.quantile(0.01)),
                            "5%": float(numeric_data.quantile(0.05)),
                            "25%": float(numeric_data.quantile(0.25)),
                            "50%": float(numeric_data.quantile(0.5)),
                            "75%": float(numeric_data.quantile(0.75)),
                            "95%": float(numeric_data.quantile(0.95)),
                            "99%": float(numeric_data.quantile(0.99))
                        }
                    })
                    
                    # Add outlier info if available
                    outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {}).get(col, {})
                    if outliers:
                        col_profile["outliers"] = {
                            "count": outliers.get("count", 0),
                            "percentage": outliers.get("percentage", 0),
                            "normal_range": outliers.get("normal_range", "")
                        }
            
            # Add categorical-specific stats
            elif col in self.categorical_cols:
                cat_data = col_data.dropna()
                if len(cat_data) > 0:
                    # Get frequency of top categories
                    value_counts = cat_data.value_counts(normalize=True)
                    
                    col_profile.update({
                        "top_categories": {
                            str(k): round(v * 100, 2) for k, v in value_counts.head(5).items()
                        },
                        "mode": str(cat_data.mode().iloc[0]) if not cat_data.mode().empty else "",
                        "mode_pct": round(100 * cat_data.value_counts().max() / len(cat_data), 2) if not cat_data.value_counts().empty else 0,
                        "cardinality": "High" if col_profile["unique_pct"] > 50 else "Medium" if col_profile["unique_pct"] > 10 else "Low"
                    })
                    
                    # Special handling for detected types
                    if 'email' in col.lower():
                        # Check for valid email pattern
                        valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        valid_count = cat_data.astype(str).str.match(valid_email_pattern).sum()
                        
                        col_profile["detected_type"] = "Email"
                        col_profile["valid_format_pct"] = round(100 * valid_count / len(cat_data), 2)
                    
                    elif 'date' in col.lower() or 'time' in col.lower():
                        # Try to parse as datetime
                        try:
                            col_profile["detected_type"] = "DateTime"
                            dates = pd.to_datetime(cat_data, errors='coerce')
                            valid_dates = dates.notna().sum()
                            col_profile["valid_format_pct"] = round(100 * valid_dates / len(cat_data), 2)
                            
                            if valid_dates > 0:
                                col_profile["date_range"] = {
                                    "min": str(dates.min().date()),
                                    "max": str(dates.max().date())
                                }
                        except:
                            pass
            
            profile["column_profiles"][col] = col_profile
        
        # Add correlation overview
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.7:
                        high_correlations.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": round(corr_matrix.iloc[i, j], 3),
                            "correlation_strength": "Very Strong" if corr_matrix.iloc[i, j] > 0.9 else "Strong"
                        })
            
            profile["correlation_overview"] = {
                "high_correlation_count": len(high_correlations),
                "high_correlation_pairs": sorted(high_correlations, key=lambda x: x["correlation"], reverse=True)[:10]  # Top 10
            }
        
        # Add missing data patterns
        missing_cols = [col for col in self.df.columns if self.df[col].isna().any()]
        if missing_cols:
            # Calculate overall statistics
            profile["missing_data_overview"] = {
                "columns_with_missing": len(missing_cols),
                "columns_with_missing_pct": round(100 * len(missing_cols) / len(self.df.columns), 2),
                "total_missing_values": int(self.df.isna().sum().sum()),
                "total_missing_pct": round(100 * self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)), 2)
            }
            
            # If we have a pattern analysis, include it
            if "missing_patterns" in self.report.get("data_integrity", {}):
                missing_pattern = self.report["data_integrity"]["missing_patterns"].get("likely_pattern", "Unknown")
                profile["missing_data_overview"]["likely_pattern"] = missing_pattern
        
        return profile
    
    def generate_comprehensive_html_report(self, filename: str = "data_quality_report.html") -> str:
        """
        Generate a comprehensive HTML report with all data quality findings and visualizations.
        
        Args:
            filename: Output HTML file path
            
        Returns:
            Path to the generated HTML report
        """
        try:
            import jinja2
            
            # Generate all needed data
            data_quality_report = self.generate_comprehensive_report(include_figures=False)
            interactive_figures = self.generate_interactive_visualization()
            advanced_profile = self.generate_advanced_data_profile()
            
            # Convert matplotlib figures to base64 for embedding
            import base64
            from io import BytesIO
            
            embedded_figures = {}
            for name, fig in self.visualize_issues().items():
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                embedded_figures[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)  # Close to prevent memory leaks
            
            # Convert Plotly figures to HTML divs
            plotly_divs = {}
            for name, fig in interactive_figures.items():
                if name != "error":
                    try:
                        import plotly.io as pio
                        plotly_divs[name] = pio.to_html(fig, full_html=False)
                    except:
                        pass
            
            # Create HTML template
            template_str = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Comprehensive Data Quality Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .header {
                        text-align: center;
                        padding: 20px 0;
                        border-bottom: 2px solid #eee;
                        margin-bottom: 30px;
                    }
                    .summary-card {
                        background: #f9f9f9;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .metric-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }
                    .metric-box {
                        background: white;
                        padding: 15px;
                        border-radius: 6px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                        text-align: center;
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight# filepath: c:\Users\adilm\repositories\Python\neural-scope\aiml_complexity\data_guardian_enhancer.py
    def _recommend_automatic_fixes(self) -> Dict:
        """
        Generate recommendations for which fixes to apply automatically.
        
        Returns:
            Dictionary with recommended fixes and their rationale
        """
        recommendations = {
            "recommended_fixes": [],
            "optional_fixes": [],
            "manual_review_required": [],
            "rationale": {}
        }
        
        # Check for duplicates - safe to remove
        duplicate_count = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicate_count > 0:
            recommendations["recommended_fixes"].append("duplicates")
            recommendations["rationale"]["duplicates"] = f"Found {duplicate_count} exact duplicate rows that can be safely removed."
        
        # Check for missing values
        missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
        if missing_patterns and missing_patterns.get("imputation_recommendations"):
            # Only auto-recommend if missingness is below a threshold
            missing_pcts = missing_patterns.get("missing_percentages", {})
            avg_missing = sum(missing_pcts.values()) / len(missing_pcts) if missing_pcts else 0
            
            if avg_missing > 0:
                if avg_missing < 10:
                    recommendations["recommended_fixes"].append("missing")
                    recommendations["rationale"]["missing"] = "Low missing data percentage can be safely imputed automatically."
                elif avg_missing < 30:
                    recommendations["optional_fixes"].append("missing")
                    recommendations["rationale"]["missing"] = f"Moderate missing data ({avg_missing:.1f}%) - imputation may affect analysis."
                else:
                    recommendations["manual_review_required"].append("missing")
                    recommendations["rationale"]["missing"] = f"High missing data percentage ({avg_missing:.1f}%) - requires careful review before imputation."
        
        # Check for storage optimization - always safe
        storage_info = self.report.get("storage_efficiency", {})
        if storage_info:
            savings_pct = storage_info.get("savings_percentage", 0)
            if savings_pct > 5:
                recommendations["recommended_fixes"].append("storage")
                recommendations["rationale"]["storage"] = f"Potential memory savings of {savings_pct:.1f}% with dtype optimization."
        
        # Check for outliers
        outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
        if outliers:
            outlier_cols = [col for col, info in outliers.items() if info.get("percentage", 0) > 5]
            if outlier_cols:
                recommendations["optional_fixes"].append("outliers")
                recommendations["rationale"]["outliers"] = f"Found significant outliers in {len(outlier_cols)} columns - consider winsorizing if they're not valid data points."
        
        # Check for corrupted values
        corrupt_values = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
        if corrupt_values:
            recommendations["optional_fixes"].append("corrupt")
            recommendations["rationale"]["corrupt"] = f"Found potentially corrupted values in {len(corrupt_values)} columns - review before cleaning."
        
        # Check for class imbalance
        if self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance.get("is_imbalanced", False):
                imbalance_ratio = class_balance.get("imbalance_ratio", 1.0)
                
                if imbalance_ratio > 10:
                    recommendations["manual_review_required"].append("class_imbalance")
                    recommendations["rationale"]["class_imbalance"] = f"Severe class imbalance (ratio: {imbalance_ratio:.1f}) - sampling techniques may significantly alter data distribution."
                elif imbalance_ratio > 3:
                    recommendations["optional_fixes"].append("class_imbalance")
                    recommendations["rationale"]["class_imbalance"] = f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}) - sampling may improve model performance."
        
        return recommendations

    def generate_interactive_visualization(self) -> Dict:
        """
        Generate interactive visualizations for the full dataset analysis.
        
        Returns:
            Dictionary of Plotly figures for interactive exploration
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            figures = {}
            
            # 1. Missing data heatmap
            if "missing_patterns" in self.report["data_integrity"]:
                # Create a binary missing data matrix
                missing_matrix = self.df.isna().astype(int)
                
                # Get columns with missing values
                missing_cols = [col for col in self.df.columns if self.df[col].isna().any()]
                
                if missing_cols:
                    # Sample up to 1000 rows for better visibility
                    sample_size = min(1000, len(missing_matrix))
                    if len(missing_matrix) > sample_size:
                        missing_sample = missing_matrix.sample(sample_size)[missing_cols]
                    else:
                        missing_sample = missing_matrix[missing_cols]
                    
                    # Create heatmap
                    fig = px.imshow(
                        missing_sample.T,
                        labels=dict(x="Row index", y="Features", color="Missing"),
                        color_continuous_scale=["#FFFFFF", "#6E67EB"],
                        title="Missing Data Pattern (1=Missing)",
                        width=900,
                        height=max(400, len(missing_cols) * 20)
                    )
                    fig.update_layout(
                        xaxis_title="Rows (sample)",
                        yaxis_title="Features",
                        coloraxis_showscale=True
                    )
                    figures["missing_heatmap"] = fig
                    
                    # Missing percentage bar chart
                    missing_pcts = self.report["data_integrity"]["missing_patterns"]["missing_percentages"]
                    if missing_pcts:
                        sorted_data = [(k, v) for k, v in sorted(missing_pcts.items(), key=lambda x: x[1], reverse=True) if v > 0]
                        if sorted_data:
                            columns, percentages = zip(*sorted_data)
                            
                            fig = px.bar(
                                x=columns, 
                                y=percentages,
                                labels={"x": "Column", "y": "Missing (%)"},
                                title="Missing Values by Column",
                                color=percentages,
                                color_continuous_scale=px.colors.sequential.Viridis,
                                width=900,
                                height=500
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["missing_percentages"] = fig
            
            # 2. Interactive distribution explorer
            if self.numeric_cols:
                # Create distribution figures for numeric columns
                for col in self.numeric_cols[:10]:  # Limit to 10 for performance
                    fig = px.histogram(
                        self.df, 
                        x=col,
                        marginal="box",  # Add boxplot on top of histogram
                        histnorm="probability density",
                        title=f"Distribution of {col}",
                        width=700,
                        height=500
                    )
                    
                    # Add KDE curve if scipy is available
                    try:
                        from scipy import stats
                        
                        # Calculate KDE
                        values = self.df[col].dropna()
                        if len(values) > 1:  # Need at least 2 points for KDE
                            kde = stats.gaussian_kde(values)
                            x_range = np.linspace(values.min(), values.max(), 1000)
                            kde_values = kde(x_range)
                            
                            # Add KDE trace
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=kde_values,
                                    mode='lines',
                                    name='KDE',
                                    line=dict(color='red', width=2)
                                )
                            )
                    except ImportError:
                        pass
                    
                    figures[f"distribution_{col}"] = fig
                
                # 3. Correlation matrix with customizable threshold
                if len(self.numeric_cols) > 1:
                    corr_matrix = self.df[self.numeric_cols].corr().round(2)
                    
                    # Create mask for upper triangle
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # Set upper triangle to NaN
                    corr_matrix_masked = corr_matrix.copy()
                    corr_matrix_masked.values[mask] = np.nan
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix_masked,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        range_color=[-1, 1],
                        title="Feature Correlation Matrix"
                    )
                    
                    # Add correlation text
                    for i, row in enumerate(corr_matrix_masked.index):
                        for j, col in enumerate(corr_matrix_masked.columns):
                            if not pd.isna(corr_matrix_masked.iloc[i, j]):
                                fig.add_annotation(
                                    x=j, 
                                    y=i,
                                    text=str(corr_matrix.iloc[i, j]),
                                    showarrow=False,
                                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                                )
                    
                    figures["correlation_matrix"] = fig
            
            # 4. Class distribution (if target exists)
            if self.target_column and self.target_column in self.df.columns:
                # Handle differently based on numeric or categorical target
                if self.target_column in self.numeric_cols:
                    # For numeric target, show histogram and boxplot
                    fig = make_subplots(
                        rows=2, 
                        cols=1,
                        subplot_titles=["Distribution", "Boxplot"],
                        vertical_spacing=0.2,
                        specs=[[{"type": "histogram"}], [{"type": "box"}]]
                    )
                    
                    fig.add_trace(
                        go.Histogram(
                            x=self.df[self.target_column],
                            name="Distribution",
                            histnorm="probability density"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Box(
                            x=self.df[self.target_column],
                            name="Boxplot",
                            boxpoints="outliers"
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title_text=f"Target Variable: {self.target_column}",
                        showlegend=False,
                        height=700
                    )
                else:
                    # For categorical target, show bar chart and pie chart
                    value_counts = self.df[self.target_column].value_counts()
                    
                    fig = make_subplots(
                        rows=1, 
                        cols=2,
                        subplot_titles=["Bar Chart", "Pie Chart"],
                        specs=[[{"type": "bar"}, {"type": "pie"}]],
                        column_widths=[0.6, 0.4]
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            y=value_counts.index,
                            x=value_counts.values,
                            orientation='h',
                            name="Count",
                            marker=dict(
                                color=px.colors.qualitative.Plotly[:len(value_counts)]
                            )
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Pie(
                            labels=value_counts.index,
                            values=value_counts.values,
                            name="Percentage",
                            marker=dict(
                                colors=px.colors.qualitative.Plotly[:len(value_counts)]
                            )
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title_text=f"Target Variable: {self.target_column}",
                        height=500
                    )
                
                figures["target_distribution"] = fig
            
            # 5. Feature importance and relationship with target (if available)
            if self.target_column and "feature_analysis" in self.report and len(self.numeric_cols) > 1:
                try:
                    # For categorical target, use mutual information
                    if self.target_column not in self.numeric_cols:
                        from sklearn.feature_selection import mutual_info_classif
                        
                        # Get numeric features
                        features = [col for col in self.numeric_cols if col != self.target_column]
                        
                        if features:
                            # Calculate mutual information
                            X = self.df[features].fillna(0)  # Simple imputation for calculation
                            y = self.df[self.target_column].fillna(self.df[self.target_column].mode().iloc[0])
                            
                            mi_scores = mutual_info_classif(X, y)
                            mi_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': mi_scores
                            }).sort_values('Importance', ascending=False)
                            
                            # Create bar chart of feature importance
                            fig = px.bar(
                                mi_df,
                                x='Feature',
                                y='Importance',
                                title=f"Feature Importance (Mutual Information) with {self.target_column}",
                                color='Importance',
                                labels={'Importance': 'Mutual Information Score'}
                            )
                            
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["feature_importance"] = fig
                    
                    # For numeric target, use correlation
                    else:
                        features = [col for col in self.numeric_cols if col != self.target_column]
                        
                        if features:
                            # Calculate correlations with target
                            correlations = {}
                            for feature in features:
                                correlations[feature] = self.df[[feature, self.target_column]].corr().iloc[0, 1]
                            
                            corr_df = pd.DataFrame({
                                'Feature': list(correlations.keys()),
                                'Correlation': list(correlations.values())
                            }).sort_values('Correlation', key=lambda x: abs(x), ascending=False)
                            
                            # Create bar chart of correlations
                            fig = px.bar(
                                corr_df,
                                x='Feature',
                                y='Correlation',
                                title=f"Feature Correlation with {self.target_column}",
                                color='Correlation',
                                color_continuous_scale=px.colors.diverging.RdBu_r,
                                range_color=[-1, 1]
                            )
                            
                            fig.update_layout(xaxis_tickangle=-45)
                            figures["feature_correlation"] = fig
                except Exception as e:
                    # Skip if calculation fails
                    pass
            
            # 6. Outlier visualization
            outlier_data = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
            if outlier_data:
                # Get columns with highest outlier percentages
                outlier_cols = sorted(
                    [(col, data["percentage"]) for col, data in outlier_data.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5 columns
                
                if outlier_cols:
                    # Create subplot figure
                    fig = make_subplots(
                        rows=len(outlier_cols),
                        cols=1,
                        subplot_titles=[f"{col} ({pct:.1f}% outliers)" for col, pct in outlier_cols],
                        vertical_spacing=0.1
                    )
                    
                    for i, (col, _) in enumerate(outlier_cols):
                        # Add box plot
                        fig.add_trace(
                            go.Box(
                                y=self.df[col].dropna(),
                                name=col,
                                boxpoints='outliers',
                                jitter=0.3,
                                pointpos=-1.8,
                                marker=dict(
                                    color='rgba(255, 0, 0, 0.6)',
                                    size=4
                                ),
                                line=dict(color='rgb(0, 0, 255)')
                            ),
                            row=i+1,
                            col=1
                        )
                        
                        # Add normal range reference if available
                        if "normal_range" in outlier_data[col]:
                            try:
                                lower, upper = map(float, outlier_data[col]["normal_range"].split(" - "))
                                
                                # Add range area
                                fig.add_shape(
                                    type="rect",
                                    xref=f"x{i+1}",
                                    yref=f"y{i+1}",
                                    x0=0,
                                    x1=1,
                                    y0=lower,
                                    y1=upper,
                                    fillcolor="rgba(0, 255, 0, 0.2)",
                                    layer="below",
                                    line_width=0
                                )
                            except:
                                pass
                    
                    fig.update_layout(
                        height=300 * len(outlier_cols),
                        title_text='Top Features with Outliers',
                        showlegend=False
                    )
                    
                    figures["outlier_analysis"] = fig
            
            return figures
            
        except ImportError:
            # Return empty dict if Plotly is not available
            return {"error": "Plotly is required for interactive visualizations. Please install with: pip install plotly"}

    def generate_advanced_data_profile(self) -> Dict:
        """
        Generate an advanced data profile with comprehensive statistical summaries and quality metrics.
        
        Returns:
            Dictionary with detailed profiling information
        """
        profile = {
            "overview": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "duplicated_rows": int(self.df.duplicated().sum()),
                "duplicated_rows_pct": round(100 * self.df.duplicated().sum() / len(self.df), 2) if len(self.df) > 0 else 0
            },
            "quality_metrics": self._calculate_quality_score(),
            "column_profiles": {},
            "correlation_overview": {},
            "missing_data_overview": {}
        }
        
        # Generate detailed column profiles
        for col in self.df.columns:
            col_data = self.df[col]
            col_profile = {
                "type": str(col_data.dtype),
                "count": len(col_data),
                "missing": int(col_data.isna().sum()),
                "missing_pct": round(100 * col_data.isna().sum() / len(col_data), 2),
                "unique_values": int(col_data.nunique()),
                "unique_pct": round(100 * col_data.nunique() / len(col_data.dropna()), 2) if len(col_data.dropna()) > 0 else 0
            }
            
            # Add numeric-specific stats
            if col in self.numeric_cols:
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    col_profile.update({
                        "min": float(numeric_data.min()),
                        "max": float(numeric_data.max()),
                        "mean": float(numeric_data.mean()),
                        "median": float(numeric_data.median()),
                        "std": float(numeric_data.std()),
                        "skew": float(numeric_data.skew()),
                        "kurtosis": float(numeric_data.kurtosis()),
                        "zeros_count": int((numeric_data == 0).sum()),
                        "zeros_pct": round(100 * (numeric_data == 0).sum() / len(numeric_data), 2),
                        "negatives_count": int((numeric_data < 0).sum()),
                        "negatives_pct": round(100 * (numeric_data < 0).sum() / len(numeric_data), 2),
                        "quantiles": {
                            "1%": float(numeric_data.quantile(0.01)),
                            "5%": float(numeric_data.quantile(0.05)),
                            "25%": float(numeric_data.quantile(0.25)),
                            "50%": float(numeric_data.quantile(0.5)),
                            "75%": float(numeric_data.quantile(0.75)),
                            "95%": float(numeric_data.quantile(0.95)),
                            "99%": float(numeric_data.quantile(0.99))
                        }
                    })
                    
                    # Add outlier info if available
                    outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {}).get(col, {})
                    if outliers:
                        col_profile["outliers"] = {
                            "count": outliers.get("count", 0),
                            "percentage": outliers.get("percentage", 0),
                            "normal_range": outliers.get("normal_range", "")
                        }
            
            # Add categorical-specific stats
            elif col in self.categorical_cols:
                cat_data = col_data.dropna()
                if len(cat_data) > 0:
                    # Get frequency of top categories
                    value_counts = cat_data.value_counts(normalize=True)
                    
                    col_profile.update({
                        "top_categories": {
                            str(k): round(v * 100, 2) for k, v in value_counts.head(5).items()
                        },
                        "mode": str(cat_data.mode().iloc[0]) if not cat_data.mode().empty else "",
                        "mode_pct": round(100 * cat_data.value_counts().max() / len(cat_data), 2) if not cat_data.value_counts().empty else 0,
                        "cardinality": "High" if col_profile["unique_pct"] > 50 else "Medium" if col_profile["unique_pct"] > 10 else "Low"
                    })
                    
                    # Special handling for detected types
                    if 'email' in col.lower():
                        # Check for valid email pattern
                        valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        valid_count = cat_data.astype(str).str.match(valid_email_pattern).sum()
                        
                        col_profile["detected_type"] = "Email"
                        col_profile["valid_format_pct"] = round(100 * valid_count / len(cat_data), 2)
                    
                    elif 'date' in col.lower() or 'time' in col.lower():
                        # Try to parse as datetime
                        try:
                            col_profile["detected_type"] = "DateTime"
                            dates = pd.to_datetime(cat_data, errors='coerce')
                            valid_dates = dates.notna().sum()
                            col_profile["valid_format_pct"] = round(100 * valid_dates / len(cat_data), 2)
                            
                            if valid_dates > 0:
                                col_profile["date_range"] = {
                                    "min": str(dates.min().date()),
                                    "max": str(dates.max().date())
                                }
                        except:
                            pass
            
            profile["column_profiles"][col] = col_profile
        
        # Add correlation overview
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.7:
                        high_correlations.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": round(corr_matrix.iloc[i, j], 3),
                            "correlation_strength": "Very Strong" if corr_matrix.iloc[i, j] > 0.9 else "Strong"
                        })
            
            profile["correlation_overview"] = {
                "high_correlation_count": len(high_correlations),
                "high_correlation_pairs": sorted(high_correlations, key=lambda x: x["correlation"], reverse=True)[:10]  # Top 10
            }
        
        # Add missing data patterns
        missing_cols = [col for col in self.df.columns if self.df[col].isna().any()]
        if missing_cols:
            # Calculate overall statistics
            profile["missing_data_overview"] = {
                "columns_with_missing": len(missing_cols),
                "columns_with_missing_pct": round(100 * len(missing_cols) / len(self.df.columns), 2),
                "total_missing_values": int(self.df.isna().sum().sum()),
                "total_missing_pct": round(100 * self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)), 2)
            }
            
            # If we have a pattern analysis, include it
            if "missing_patterns" in self.report.get("data_integrity", {}):
                missing_pattern = self.report["data_integrity"]["missing_patterns"].get("likely_pattern", "Unknown")
                profile["missing_data_overview"]["likely_pattern"] = missing_pattern
        
        return profile
    
    def generate_comprehensive_html_report(self, filename: str = "data_quality_report.html") -> str:
        """
        Generate a comprehensive HTML report with all data quality findings and visualizations.
        
        Args:
            filename: Output HTML file path
            
        Returns:
            Path to the generated HTML report
        """
        try:
            import jinja2
            
            # Generate all needed data
            data_quality_report = self.generate_comprehensive_report(include_figures=False)
            interactive_figures = self.generate_interactive_visualization()
            advanced_profile = self.generate_advanced_data_profile()
            
            # Convert matplotlib figures to base64 for embedding
            import base64
            from io import BytesIO
            
            embedded_figures = {}
            for name, fig in self.visualize_issues().items():
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                embedded_figures[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)  # Close to prevent memory leaks
            
            # Convert Plotly figures to HTML divs
            plotly_divs = {}
            for name, fig in interactive_figures.items():
                if name != "error":
                    try:
                        import plotly.io as pio
                        plotly_divs[name] = pio.to_html(fig, full_html=False)
                    except:
                        pass
            
            # Create HTML template
            template_str = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Comprehensive Data Quality Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .header {
                        text-align: center;
                        padding: 20px 0;
                        border-bottom: 2px solid #eee;
                        margin-bottom: 30px;
                    }
                    .summary-card {
                        background: #f9f9f9;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .metric-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }
                    .metric-box {
                        background: white;
                        padding: 15px;
                        border-radius: 6px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                        text-align: center;
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight: bold;
                        
                                # Train a simple model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Analyze feature importance for bias
                feature_importances = {}
                protected_columns = []
                
                # Identify protected attribute columns in the one-hot encoded dataframe
                for attr in protected_attributes:
                    if attr in df.columns:
                        cols = [col for col in X.columns if col.startswith(f"{attr}_")]
                        protected_columns.extend(cols)
                
                # Calculate importance of protected vs non-protected features
                if protected_columns:
                    protected_importance = sum(abs(shap_values).mean(axis=0)[X.columns.get_indexer(protected_columns)])
                    total_importance = sum(abs(shap_values).mean(axis=0))
                    
                    # Normalize to calculate bias score
                    protected_ratio = len(protected_columns) / len(X.columns)
                    expected_importance = total_importance * protected_ratio
                    actual_importance = protected_importance
                    
                    # Calculate bias as relative deviation from expected importance
                    if expected_importance > 0:
                        bias_score = actual_importance / expected_importance
                        metrics.feature_importance_bias['shap_ratio'] = bias_score
                
            except Exception as e:
                logger.warning(f"Error calculating SHAP-based feature importance bias: {e}")
        
        # Calculate overall bias score
        bias_scores = []
        
        # Statistical parity contributes to overall bias
        if metrics.statistical_parity:
            max_parity_score = max(metrics.statistical_parity.values())
            bias_scores.append(max_parity_score)
        
        # Disparate impact contributes to overall bias
        if metrics.disparate_impact:
            # Convert disparate impact to a score between 0 and 1
            di_scores = [abs(di - 1) for di in metrics.disparate_impact.values()]
            if di_scores:
                max_di_score = max(di_scores)
                bias_scores.append(max_di_score)
        
        # Intersectional bias contributes to overall bias
        if metrics.intersectional_bias_scores:
            max_intersectional = max(metrics.intersectional_bias_scores.values())
            bias_scores.append(max_intersectional)
        
        # Feature importance bias contributes to overall bias
        if metrics.feature_importance_bias:
            feat_importance_bias = metrics.feature_importance_bias.get('shap_ratio', 1)
            # Convert to 0-1 scale where 1 means maximum bias
            normalized_importance_bias = min(abs(feat_importance_bias - 1), 1)
            bias_scores.append(normalized_importance_bias)
        
        # Calculate overall bias score (0 = unbiased, 1 = maximum bias)
        if bias_scores:
            metrics.overall_bias_score = float(np.mean(bias_scores))
        
        logger.info(f"Bias assessment completed. Overall bias score: {metrics.overall_bias_score:.4f}")
        return metrics

    def _assess_privacy(self, 
                        df: pd.DataFrame, 
                        sensitive_attributes: Optional[List[str]] = None) -> PrivacyMetrics:
        """
        Perform comprehensive privacy risk assessment
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of columns with sensitive information
            
        Returns:
            PrivacyMetrics with comprehensive privacy assessment
        """
        metrics = PrivacyMetrics()
        logger.info("Starting privacy assessment")
        
        # Detect PII in text columns
        metrics.pii_detected = self._detect_pii(df, sensitive_attributes)
        
        # Calculate k-anonymity (minimum group size for quasi-identifiers)
        quasi_identifiers = self._identify_quasi_identifiers(df, sensitive_attributes)
        
        if quasi_identifiers:
            # Count occurrences of each quasi-identifier combination
            if len(quasi_identifiers) > 0:
                try:
                    # Get combination of values for quasi-identifiers
                    grouped = df.groupby(quasi_identifiers).size()
                    # k-anonymity is the minimum group size
                    metrics.k_anonymity = int(grouped.min()) if len(grouped) > 0 else len(df)
                except Exception as e:
                    logger.warning(f"Error calculating k-anonymity: {e}")
                    metrics.k_anonymity = 0
                
                # Calculate l-diversity for sensitive attributes if specified
                if sensitive_attributes:
                    for sensitive_attr in sensitive_attributes:
                        if sensitive_attr in df.columns and sensitive_attr not in quasi_identifiers:
                            try:
                                # For each combination of quasi-identifiers, count distinct values of sensitive attribute
                                l_diversity = df.groupby(quasi_identifiers)[sensitive_attr].nunique()
                                metrics.l_diversity[sensitive_attr] = int(l_diversity.min()) if len(l_diversity) > 0 else 0
                            except Exception as e:
                                logger.warning(f"Error calculating l-diversity for {sensitive_attr}: {e}")
        
        # Calculate uniqueness risk score for each column
        for col in df.columns:
            uniqueness = df[col].nunique() / len(df)
            metrics.uniqueness_risk[col] = uniqueness
        
        # Calculate overall privacy risk score
        privacy_risk_factors = []
        
        # PII detection contributes to privacy risk
        pii_factor = len(metrics.pii_detected) / len(df.columns) if df.columns.size > 0 else 0
        privacy_risk_factors.append(pii_factor)
        
        # k-anonymity contributes to privacy risk (lower k = higher risk)
        if metrics.k_anonymity > 0:
            k_factor = min(1.0, 5 / metrics.k_anonymity)  # 5 is a threshold for reasonable anonymity
            privacy_risk_factors.append(k_factor)
        
        # Uniqueness contributes to privacy risk
        if metrics.uniqueness_risk:
            # Columns with high uniqueness (close to unique identifiers) increase risk
            high_uniqueness_cols = sum(1 for v in metrics.uniqueness_risk.values() if v > 0.8)
            uniqueness_factor = high_uniqueness_cols / len(df.columns) if df.columns.size > 0 else 0
            privacy_risk_factors.append(uniqueness_factor)
        
        # Calculate overall privacy risk (0 = low risk, 1 = high risk)
        if privacy_risk_factors:
            metrics.overall_privacy_risk = float(np.mean(privacy_risk_factors))
        
        logger.info(f"Privacy assessment completed. Overall privacy risk score: {metrics.overall_privacy_risk:.4f}")
        return metrics
    
    def _detect_pii(self, 
                    df: pd.DataFrame, 
                    sensitive_attributes: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Detect personally identifiable information in the dataset
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of columns with known sensitive information
            
        Returns:
            Dictionary mapping column names to detected PII types
        """
        pii_detected = {}
        
        # Process explicitly specified sensitive attributes
        if sensitive_attributes:
            for col in sensitive_attributes:
                if col in df.columns:
                    pii_detected[col] = ["user_specified"]
        
        # Look for PII in string columns using regex patterns
        for col in df.columns:
            if col in pii_detected:
                continue  # Skip already identified columns
                
            if df[col].dtype == 'object':
                # Take a sample to detect patterns
                sample = df[col].dropna().astype(str).head(1000)
                
                # Check each PII pattern
                detected_patterns = []
                for pii_type, pattern in self.pii_patterns.items():
                    # Check if a significant portion of values match the pattern
                    matches = sum(1 for val in sample if re.match(pattern, val))
                    if matches / len(sample) > 0.1:  # More than 10% match
                        detected_patterns.append(pii_type)
                
                if detected_patterns:
                    pii_detected[col] = detected_patterns
                    
        return pii_detected
    
    def _identify_quasi_identifiers(self, 
                                   df: pd.DataFrame, 
                                   sensitive_attributes: Optional[List[str]] = None) -> List[str]:
        """
        Identify columns that could serve as quasi-identifiers
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of sensitive columns to exclude
            
        Returns:
            List of column names that might be quasi-identifiers
        """
        quasi_identifiers = []
        
        # Exclude explicitly marked sensitive attributes
        exclude_cols = set(sensitive_attributes or [])
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Check uniqueness - quasi-identifiers typically have medium uniqueness
            uniqueness = df[col].nunique() / len(df)
            
            # Check for columns with categorical-like distributions but not too many unique values
            if 0.01 < uniqueness < 0.5:
                quasi_identifiers.append(col)
        
        return quasi_identifiers
    
    def _should_detect_pii(self) -> bool:
        """Determine if automatic PII detection should be performed"""
        # In enterprise environments, safer to always detect PII unless explicitly disabled
        return True
    
    def _assess_drift(self, 
                      current_df: pd.DataFrame, 
                      reference_df: pd.DataFrame) -> DataDriftMetrics:
        """
        Detect distribution shifts between current and reference datasets
        
        Args:
            current_df: Current DataFrame to analyze
            reference_df: Reference DataFrame to compare against
            
        Returns:
            DataDriftMetrics with comprehensive drift assessment
        """
        metrics = DataDriftMetrics()
        logger.info("Starting distribution drift assessment")
        
        # Ensure dataframes have compatible columns
        common_columns = set(current_df.columns).intersection(set(reference_df.columns))
        
        if not common_columns:
            logger.warning("No common columns found between current and reference datasets")
            return metrics
        
        # Calculate drift for each common column
        for col in common_columns:
            # Skip columns with too many missing values
            if (current_df[col].isna().mean() > 0.3 or 
                reference_df[col].isna().mean() > 0.3):
                continue
                
            # Different drift detection methods based on column type
            if pd.api.types.is_numeric_dtype(current_df[col]) and pd.api.types.is_numeric_dtype(reference_df[col]):
                # For numeric columns, use statistical tests
                drift_score = self._detect_numeric_drift(current_df[col], reference_df[col])
                
                if drift_score > 0:
                    metrics.feature_drift[col] = drift_score
                    
                    # Add distribution shift details for significant drift
                    if drift_score > 0.1:
                        metrics.distribution_shifts[col] = {
                            'current_mean': float(current_df[col].mean()),
                            'reference_mean': float(reference_df[col].mean()),
                            'current_std': float(current_df[col].std()),
                            'reference_std': float(reference_df[col].std()),
                            'shift_type': 'mean_shift' if abs(current_df[col].mean() - reference_df[col].mean()) > 0.1 * reference_df[col].std() else 'variance_shift'
                        }
            else:
                # For categorical columns, use distribution comparison
                drift_score = self._detect_categorical_drift(current_df[col], reference_df[col])
                
                if drift_score > 0:
                    metrics.feature_drift[col] = drift_score
                    
                    # Add distribution shift details for significant drift
                    if drift_score > 0.1:
                        # Get top categories with biggest shifts
                        current_dist = current_df[col].value_counts(normalize=True).to_dict()
                        reference_dist = reference_df[col].value_counts(normalize=True).to_dict()
                        
                        # Find categories with biggest absolute difference
                        all_categories = set(current_dist.keys()).union(set(reference_dist.keys()))
                        category_shifts = {}
                        
                        for category in all_categories:
                            current_freq = current_dist.get(category, 0)
                            reference_freq = reference_dist.get(category, 0)
                            category_shifts[category] = abs(current_freq - reference_freq)
                        
                        # Get top 5 categories with biggest shifts
                        top_shifts = sorted(category_shifts.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        metrics.distribution_shifts[col] = {
                            'shift_type': 'category_distribution_shift',
                            'top_shifted_categories': dict(top_shifts)
                        }
        
        # Calculate overall drift score as weighted average of feature drifts
        if metrics.feature_drift:
            metrics.overall_drift_score = float(np.mean(list(metrics.feature_drift.values())))
        
        logger.info(f"Drift assessment completed. Overall drift score: {metrics.overall_drift_score:.4f}")
        return metrics
    
    def _detect_numeric_drift(self, current_series: pd.Series, reference_series: pd.Series) -> float:
        """
        Detect drift in numeric features using statistical methods
        
        Returns:
            Drift score between 0 (no drift) and 1 (extreme drift)
        """
        try:
            # Clean data
            current_clean = current_series.dropna()
            reference_clean = reference_series.dropna()
            
            if len(current_clean) < 10 or len(reference_clean) < 10:
                return 0.0
            
            # Use Kolmogorov-Smirnov test to compare distributions
            ks_statistic, p_value = stats.ks_2samp(current_clean, reference_clean)
            
            # Convert p-value to drift score (smaller p-value = higher drift)
            # p-value < 0.05 is statistically significant drift
            if p_value < 0.05:
                # Scale drift score based on KS statistic (0-1)
                return float(ks_statistic)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in numeric drift detection: {e}")
            return 0.0
    
    def _detect_categorical_drift(self, current_series: pd.Series, reference_series: pd.Series) -> float:
        """
        Detect drift in categorical features using distribution comparison
        
        Returns:
            Drift score between 0 (no drift) and 1 (extreme drift)
        """
        try:
            # Convert to categorical if not already
            if not pd.api.types.is_categorical_dtype(current_series):
                current_series = current_series.astype(str)
                
            if not pd.api.types.is_categorical_dtype(reference_series):
                reference_series = reference_series.astype(str)
            
            # Calculate distribution for both series
            current_dist = current_series.value_counts(normalize=True).to_dict()
            reference_dist = reference_series.value_counts(normalize=True).to_dict()
            
            # Get all unique categories
            all_categories = set(current_dist.keys()).union(set(reference_dist.keys()))
            
            # Calculate Jensen-Shannon divergence
            current_probs = np.array([current_dist.get(cat, 0) for cat in all_categories])
            reference_probs = np.array([reference_dist.get(cat, 0) for cat in all_categories])
            
            # Add small epsilon to avoid zeros
            epsilon = 1e-10
            current_probs = current_probs + epsilon
            reference_probs = reference_probs + epsilon
            
            # Normalize
            current_probs = current_probs / current_probs.sum()
            reference_probs = reference_probs / reference_probs.sum()
            
            # Calculate Jensen-Shannon divergence
            m = (current_probs + reference_probs) / 2
            js_divergence = (stats.entropy(current_probs, m) + stats.entropy(reference_probs, m)) / 2
            
            # Normalize to 0-1 range (JS divergence is between 0 and ln(2))
            drift_score = min(1.0, js_divergence / np.log(2))
            
            return float(drift_score)
            
        except Exception as e:
            logger.warning(f"Error in categorical drift detection: {e}")
            return 0.0
    
    def _generate_recommendations(self,
                                 df: pd.DataFrame,
                                 quality_metrics: DataQualityMetrics,
                                 bias_metrics: Optional[BiasMetrics],
                                 privacy_metrics: Optional[PrivacyMetrics],
                                 drift_metrics: Optional[DataDriftMetrics],
                                 numerical_features: List[str],
                                 categorical_features: List[str],
                                 target_column: Optional[str]) -> List[DataRecommendation]:
        """
        Generate actionable recommendations based on all analysis results
        
        Returns:
            List of DataRecommendation objects with severity and code examples
        """
        recommendations = []
        
        # Quality recommendations
        self._add_quality_recommendations(
            df, quality_metrics, numerical_features, categorical_features, recommendations
        )
        
        # Bias recommendations
        if bias_metrics:
            self._add_bias_recommendations(
                df, bias_metrics, target_column, recommendations
            )
        
        # Privacy recommendations
        if privacy_metrics:
            self._add_privacy_recommendations(
                df, privacy_metrics, recommendations
            )
        
        # Drift recommendations
        if drift_metrics:
            self._add_drift_recommendations(
                df, drift_metrics, recommendations
            )
        
        # Sort recommendations by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _add_quality_recommendations(self,
                                    df: pd.DataFrame,
                                    metrics: DataQualityMetrics,
                                    numerical_features: List[str],
                                    categorical_features: List[str],
                                    recommendations: List[DataRecommendation]):
        """Add data quality recommendations"""
        
        # Missing value recommendations
        missing_cols = {col: comp for col, comp in metrics.completeness.items() if comp < 0.98}
        for col, completeness in sorted(missing_cols.items(), key=lambda x: x[1]):
            # Calculate missing percentage
            missing_pct = (1 - completeness) * 100
            
            if missing_pct > 20:
                severity = "high"
            elif missing_pct > 5:
                severity = "medium"
            else:
                severity = "low"
                
            # Suggest different imputation methods based on column type
            if col in numerical_features:
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Missing values detected ({missing_pct:.1f}%) in numerical column '{col}'",
                    impact="May reduce model performance and lead to biased predictions",
                    recommendation="Use advanced imputation techniques for numerical data",
                    code_example=f"""
# Method 1: KNN imputation (better than mean/median for preserving relationships)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['{col}'] = imputer.fit_transform(df[['{col}']])[:,0]

# Method 2: Use iterative imputation (models each feature as a function of others)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=42)
df['{col}'] = imp.fit_transform(df[['{col}']])[:,0]
"""
                ))
            elif col in categorical_features:
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Missing values detected ({missing_pct:.1f}%) in categorical column '{col}'",
                    impact="May reduce model performance and lead to biased predictions",
                    recommendation="Use appropriate imputation techniques for categorical data",
                    code_example=f"""
# Method 1: Mode imputation (most frequent value)
mode_value = df['{col}'].mode()[0]
df['{col}'] = df['{col}'].fillna(mode_value)

# Method 2: Create a 'Missing' category
df['{col}'] = df['{col}'].fillna('Missing')

# Method 3: Use predictive model for imputation
from sklearn.ensemble import RandomForestClassifier
# First create a copy of the dataframe without the column to impute
df_temp = df.drop(columns=['{col}'])
# Split data into rows with and without missing values
known = df[df['{col}'].notna()]
unknown = df[df['{col}'].isna()]
# Train a model on known data
X_train = known.drop(columns=['{col}'])
y_train = known['{col}']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict missing values
if len(unknown) > 0:
    X_pred = unknown.drop(columns=['{col}'])
    predictions = model.predict(X_pred)
    df.loc[df['{col}'].isna(), '{col}'] = predictions
"""
                ))
                
        # Outlier recommendations
        for col, outlier_indices in metrics.outlier_scores.items():
            outlier_pct = len(outlier_indices) / len(df) * 100
            
            if outlier_pct > 5:
                severity = "high"
            elif outlier_pct > 1:
                severity = "medium"
            else:
                severity = "low"
                
            recommendations.append(DataRecommendation(
                issue_type="quality",
                severity=severity,
                feature=col,
                description=f"Outliers detected ({outlier_pct:.1f}%) in column '{col}'",
                impact="May skew model training, affect scaling, and reduce robustness",
                recommendation="Consider robust scaling, capping, or removing outliers",
                code_example=f"""
# Method 1: Cap outliers (winsorization)
from scipy import stats
df['{col}_winsorized'] = stats.mstats.winsorize(df['{col}'], limits=[0.05, 0.05])

# Method 2: Apply robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['{col}_scaled'] = scaler.fit_transform(df[['{col}']])

# Method 3: Log transformation to reduce impact of outliers
import numpy as np
if (df['{col}'] > 0).all():  # Only if all values are positive
    df['{col}_log'] = np.log1p(df['{col}'])
"""
            ))
            
        # Distribution skewness recommendations
        for col, stats_dict in metrics.distribution_metrics.items():
            if 'skewness' in stats_dict and abs(stats_dict['skewness']) > 1.0:
                skewness = stats_dict['skewness']
                
                if abs(skewness) > 3:
                    severity = "high"
                elif abs(skewness) > 2:
                    severity = "medium"
                else:
                    severity = "low"
                    
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Highly skewed distribution (skewness={skewness:.2f}) in column '{col}'",
                    impact="May violate model assumptions and reduce predictive performance",
                    recommendation="Apply appropriate transformation to normalize distribution",
                    code_example=f"""
import numpy as np
from scipy import stats

# Method 1: Box-Cox transformation (data must be positive)
if (df['{col}'] > 0).all():
    df['{col}_boxcox'], lambda_value = stats.boxcox(df['{col}'])
    print(f"Box-Cox lambda: {{lambda_value}}")

# Method 2: Yeo-Johnson transformation (works with negative values too)
df['{col}_yeojohnson'], lambda_value = stats.yeojohnson(df['{col}'])
print(f"Yeo-Johnson lambda: {{lambda_value}}")

# Method 3: Simple transformations
if skewness > 0:  # Right-skewed
    # Log transformation for positive right-skewed data
    if (df['{col}'] > 0).all():
        df['{col}_log'] = np.log1p(df['{col}'])
    # Square root for right-skewed data
    if (df['{col}'] >= 0).all():
        df['{col}_sqrt'] = np.sqrt(df['{col}'])
else:  # Left-skewed
    # Square or cube for left-skewed data
    df['{col}_squared'] = df['{col}'] ** 2
"""
                ))
                
        # High cardinality categorical features
        for col in categorical_features:
            if col in metrics.distribution_metrics and 'unique_count' in metrics.distribution_metrics[col]:
                unique_count = metrics.distribution_metrics[col]['unique_count']
                
                if unique_count > 100:
                    severity = "high"
                elif unique_count > 50:
                    severity = "medium"
                elif unique_count > 20:
                    severity = "low"
                else:
                    continue  # Not a concern
                    
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"High cardinality detected in categorical feature '{col}' ({unique_count} unique values)",
                    impact="May cause overfitting, curse of dimensionality when one-hot encoded, and reduce model performance",
                    recommendation="Use dimensionality reduction techniques for categorical features",
                    code_example=f"""
# Method 1: Group rare categories
value_counts = df['{col}'].value_counts()
# Keep top 10 categories, group the rest as 'Other'
top_categories = value_counts.nlargest(10).index
df['{col}_grouped'] = df['{col}'].apply(lambda x: x if x in top_categories else 'Other')

# Method 2: Target encoding (replace categories with target mean)
# Caution: Requires cross-validation to prevent data leakage
if 'target_column' in df:  # Replace with your actual target column
    global_mean = df['target_column'].mean()
    encoding_map = df.groupby('{col}')['target_column'].mean().to_dict()
    df['{col}_encoded'] = df['{col}'].map(encoding_map).fillna(global_mean)

# Method 3: Use hash encoding 
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=20, input_type='string')
hashed_features = hasher.transform(df['{col}'].astype(str))
hashed_df = pd.DataFrame(hashed_features.toarray())
hashed_df.columns = [f'{col}_hash_{{i}}' for i in range(20)]
df = pd.concat([df, hashed_df], axis=1)
"""
                ))
                
        # Correlation recommendations
        if metrics.correlation_matrix is not None:
            # Find highly correlated features
            corr_matrix = metrics.correlation_matrix
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.9:
                        high_corr_pairs.append((col1, col2, abs(corr)))
            
            if high_corr_pairs:
                cols1 = [pair[0] for pair in high_corr_pairs[:3]]
                cols2 = [pair[1] for pair in high_corr_pairs[:3]]
                corrs = [f"{pair[2]:.2f}" for pair in high_corr_pairs[:3]]
                
                col_pairs_str = ", ".join([f"'{c1}'/'{c2}' ({corr})" for c1, c2, corr in zip(cols1, cols2, corrs)])
                
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity="medium",
                    feature=None,
                    description=f"High multicollinearity detected between features: {col_pairs_str}",
                    impact="May cause instability in model coefficients, reduce interpretability, and make models sensitive to small changes in input",
                    recommendation="Apply dimensionality# filepath: c:\Users\adilm\repositories\Python\neural-scope\advanced_analysis\data_guardian_enterprise.py
                # Train a simple model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Analyze feature importance for bias
                feature_importances = {}
                protected_columns = []
                
                # Identify protected attribute columns in the one-hot encoded dataframe
                for attr in protected_attributes:
                    if attr in df.columns:
                        cols = [col for col in X.columns if col.startswith(f"{attr}_")]
                        protected_columns.extend(cols)
                
                # Calculate importance of protected vs non-protected features
                if protected_columns:
                    protected_importance = sum(abs(shap_values).mean(axis=0)[X.columns.get_indexer(protected_columns)])
                    total_importance = sum(abs(shap_values).mean(axis=0))
                    
                    # Normalize to calculate bias score
                    protected_ratio = len(protected_columns) / len(X.columns)
                    expected_importance = total_importance * protected_ratio
                    actual_importance = protected_importance
                    
                    # Calculate bias as relative deviation from expected importance
                    if expected_importance > 0:
                        bias_score = actual_importance / expected_importance
                        metrics.feature_importance_bias['shap_ratio'] = bias_score
                
            except Exception as e:
                logger.warning(f"Error calculating SHAP-based feature importance bias: {e}")
        
        # Calculate overall bias score
        bias_scores = []
        
        # Statistical parity contributes to overall bias
        if metrics.statistical_parity:
            max_parity_score = max(metrics.statistical_parity.values())
            bias_scores.append(max_parity_score)
        
        # Disparate impact contributes to overall bias
        if metrics.disparate_impact:
            # Convert disparate impact to a score between 0 and 1
            di_scores = [abs(di - 1) for di in metrics.disparate_impact.values()]
            if di_scores:
                max_di_score = max(di_scores)
                bias_scores.append(max_di_score)
        
        # Intersectional bias contributes to overall bias
        if metrics.intersectional_bias_scores:
            max_intersectional = max(metrics.intersectional_bias_scores.values())
            bias_scores.append(max_intersectional)
        
        # Feature importance bias contributes to overall bias
        if metrics.feature_importance_bias:
            feat_importance_bias = metrics.feature_importance_bias.get('shap_ratio', 1)
            # Convert to 0-1 scale where 1 means maximum bias
            normalized_importance_bias = min(abs(feat_importance_bias - 1), 1)
            bias_scores.append(normalized_importance_bias)
        
        # Calculate overall bias score (0 = unbiased, 1 = maximum bias)
        if bias_scores:
            metrics.overall_bias_score = float(np.mean(bias_scores))
        
        logger.info(f"Bias assessment completed. Overall bias score: {metrics.overall_bias_score:.4f}")
        return metrics

    def _assess_privacy(self, 
                        df: pd.DataFrame, 
                        sensitive_attributes: Optional[List[str]] = None) -> PrivacyMetrics:
        """
        Perform comprehensive privacy risk assessment
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of columns with sensitive information
            
        Returns:
            PrivacyMetrics with comprehensive privacy assessment
        """
        metrics = PrivacyMetrics()
        logger.info("Starting privacy assessment")
        
        # Detect PII in text columns
        metrics.pii_detected = self._detect_pii(df, sensitive_attributes)
        
        # Calculate k-anonymity (minimum group size for quasi-identifiers)
        quasi_identifiers = self._identify_quasi_identifiers(df, sensitive_attributes)
        
        if quasi_identifiers:
            # Count occurrences of each quasi-identifier combination
            if len(quasi_identifiers) > 0:
                try:
                    # Get combination of values for quasi-identifiers
                    grouped = df.groupby(quasi_identifiers).size()
                    # k-anonymity is the minimum group size
                    metrics.k_anonymity = int(grouped.min()) if len(grouped) > 0 else len(df)
                except Exception as e:
                    logger.warning(f"Error calculating k-anonymity: {e}")
                    metrics.k_anonymity = 0
                
                # Calculate l-diversity for sensitive attributes if specified
                if sensitive_attributes:
                    for sensitive_attr in sensitive_attributes:
                        if sensitive_attr in df.columns and sensitive_attr not in quasi_identifiers:
                            try:
                                # For each combination of quasi-identifiers, count distinct values of sensitive attribute
                                l_diversity = df.groupby(quasi_identifiers)[sensitive_attr].nunique()
                                metrics.l_diversity[sensitive_attr] = int(l_diversity.min()) if len(l_diversity) > 0 else 0
                            except Exception as e:
                                logger.warning(f"Error calculating l-diversity for {sensitive_attr}: {e}")
        
        # Calculate uniqueness risk score for each column
        for col in df.columns:
            uniqueness = df[col].nunique() / len(df)
            metrics.uniqueness_risk[col] = uniqueness
        
        # Calculate overall privacy risk score
        privacy_risk_factors = []
        
        # PII detection contributes to privacy risk
        pii_factor = len(metrics.pii_detected) / len(df.columns) if df.columns.size > 0 else 0
        privacy_risk_factors.append(pii_factor)
        
        # k-anonymity contributes to privacy risk (lower k = higher risk)
        if metrics.k_anonymity > 0:
            k_factor = min(1.0, 5 / metrics.k_anonymity)  # 5 is a threshold for reasonable anonymity
            privacy_risk_factors.append(k_factor)
        
        # Uniqueness contributes to privacy risk
        if metrics.uniqueness_risk:
            # Columns with high uniqueness (close to unique identifiers) increase risk
            high_uniqueness_cols = sum(1 for v in metrics.uniqueness_risk.values() if v > 0.8)
            uniqueness_factor = high_uniqueness_cols / len(df.columns) if df.columns.size > 0 else 0
            privacy_risk_factors.append(uniqueness_factor)
        
        # Calculate overall privacy risk (0 = low risk, 1 = high risk)
        if privacy_risk_factors:
            metrics.overall_privacy_risk = float(np.mean(privacy_risk_factors))
        
        logger.info(f"Privacy assessment completed. Overall privacy risk score: {metrics.overall_privacy_risk:.4f}")
        return metrics
    
    def _detect_pii(self, 
                    df: pd.DataFrame, 
                    sensitive_attributes: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Detect personally identifiable information in the dataset
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of columns with known sensitive information
            
        Returns:
            Dictionary mapping column names to detected PII types
        """
        pii_detected = {}
        
        # Process explicitly specified sensitive attributes
        if sensitive_attributes:
            for col in sensitive_attributes:
                if col in df.columns:
                    pii_detected[col] = ["user_specified"]
        
        # Look for PII in string columns using regex patterns
        for col in df.columns:
            if col in pii_detected:
                continue  # Skip already identified columns
                
            if df[col].dtype == 'object':
                # Take a sample to detect patterns
                sample = df[col].dropna().astype(str).head(1000)
                
                # Check each PII pattern
                detected_patterns = []
                for pii_type, pattern in self.pii_patterns.items():
                    # Check if a significant portion of values match the pattern
                    matches = sum(1 for val in sample if re.match(pattern, val))
                    if matches / len(sample) > 0.1:  # More than 10% match
                        detected_patterns.append(pii_type)
                
                if detected_patterns:
                    pii_detected[col] = detected_patterns
                    
        return pii_detected
    
    def _identify_quasi_identifiers(self, 
                                   df: pd.DataFrame, 
                                   sensitive_attributes: Optional[List[str]] = None) -> List[str]:
        """
        Identify columns that could serve as quasi-identifiers
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: List of sensitive columns to exclude
            
        Returns:
            List of column names that might be quasi-identifiers
        """
        quasi_identifiers = []
        
        # Exclude explicitly marked sensitive attributes
        exclude_cols = set(sensitive_attributes or [])
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Check uniqueness - quasi-identifiers typically have medium uniqueness
            uniqueness = df[col].nunique() / len(df)
            
            # Check for columns with categorical-like distributions but not too many unique values
            if 0.01 < uniqueness < 0.5:
                quasi_identifiers.append(col)
        
        return quasi_identifiers
    
    def _should_detect_pii(self) -> bool:
        """Determine if automatic PII detection should be performed"""
        # In enterprise environments, safer to always detect PII unless explicitly disabled
        return True
    
    def _assess_drift(self, 
                      current_df: pd.DataFrame, 
                      reference_df: pd.DataFrame) -> DataDriftMetrics:
        """
        Detect distribution shifts between current and reference datasets
        
        Args:
            current_df: Current DataFrame to analyze
            reference_df: Reference DataFrame to compare against
            
        Returns:
            DataDriftMetrics with comprehensive drift assessment
        """
        metrics = DataDriftMetrics()
        logger.info("Starting distribution drift assessment")
        
        # Ensure dataframes have compatible columns
        common_columns = set(current_df.columns).intersection(set(reference_df.columns))
        
        if not common_columns:
            logger.warning("No common columns found between current and reference datasets")
            return metrics
        
        # Calculate drift for each common column
        for col in common_columns:
            # Skip columns with too many missing values
            if (current_df[col].isna().mean() > 0.3 or 
                reference_df[col].isna().mean() > 0.3):
                continue
                
            # Different drift detection methods based on column type
            if pd.api.types.is_numeric_dtype(current_df[col]) and pd.api.types.is_numeric_dtype(reference_df[col]):
                # For numeric columns, use statistical tests
                drift_score = self._detect_numeric_drift(current_df[col], reference_df[col])
                
                if drift_score > 0:
                    metrics.feature_drift[col] = drift_score
                    
                    # Add distribution shift details for significant drift
                    if drift_score > 0.1:
                        metrics.distribution_shifts[col] = {
                            'current_mean': float(current_df[col].mean()),
                            'reference_mean': float(reference_df[col].mean()),
                            'current_std': float(current_df[col].std()),
                            'reference_std': float(reference_df[col].std()),
                            'shift_type': 'mean_shift' if abs(current_df[col].mean() - reference_df[col].mean()) > 0.1 * reference_df[col].std() else 'variance_shift'
                        }
            else:
                # For categorical columns, use distribution comparison
                drift_score = self._detect_categorical_drift(current_df[col], reference_df[col])
                
                if drift_score > 0:
                    metrics.feature_drift[col] = drift_score
                    
                    # Add distribution shift details for significant drift
                    if drift_score > 0.1:
                        # Get top categories with biggest shifts
                        current_dist = current_df[col].value_counts(normalize=True).to_dict()
                        reference_dist = reference_df[col].value_counts(normalize=True).to_dict()
                        
                        # Find categories with biggest absolute difference
                        all_categories = set(current_dist.keys()).union(set(reference_dist.keys()))
                        category_shifts = {}
                        
                        for category in all_categories:
                            current_freq = current_dist.get(category, 0)
                            reference_freq = reference_dist.get(category, 0)
                            category_shifts[category] = abs(current_freq - reference_freq)
                        
                        # Get top 5 categories with biggest shifts
                        top_shifts = sorted(category_shifts.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        metrics.distribution_shifts[col] = {
                            'shift_type': 'category_distribution_shift',
                            'top_shifted_categories': dict(top_shifts)
                        }
        
        # Calculate overall drift score as weighted average of feature drifts
        if metrics.feature_drift:
            metrics.overall_drift_score = float(np.mean(list(metrics.feature_drift.values())))
        
        logger.info(f"Drift assessment completed. Overall drift score: {metrics.overall_drift_score:.4f}")
        return metrics
    
    def _detect_numeric_drift(self, current_series: pd.Series, reference_series: pd.Series) -> float:
        """
        Detect drift in numeric features using statistical methods
        
        Returns:
            Drift score between 0 (no drift) and 1 (extreme drift)
        """
        try:
            # Clean data
            current_clean = current_series.dropna()
            reference_clean = reference_series.dropna()
            
            if len(current_clean) < 10 or len(reference_clean) < 10:
                return 0.0
            
            # Use Kolmogorov-Smirnov test to compare distributions
            ks_statistic, p_value = stats.ks_2samp(current_clean, reference_clean)
            
            # Convert p-value to drift score (smaller p-value = higher drift)
            # p-value < 0.05 is statistically significant drift
            if p_value < 0.05:
                # Scale drift score based on KS statistic (0-1)
                return float(ks_statistic)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in numeric drift detection: {e}")
            return 0.0
    
    def _detect_categorical_drift(self, current_series: pd.Series, reference_series: pd.Series) -> float:
        """
        Detect drift in categorical features using distribution comparison
        
        Returns:
            Drift score between 0 (no drift) and 1 (extreme drift)
        """
        try:
            # Convert to categorical if not already
            if not pd.api.types.is_categorical_dtype(current_series):
                current_series = current_series.astype(str)
                
            if not pd.api.types.is_categorical_dtype(reference_series):
                reference_series = reference_series.astype(str)
            
            # Calculate distribution for both series
            current_dist = current_series.value_counts(normalize=True).to_dict()
            reference_dist = reference_series.value_counts(normalize=True).to_dict()
            
            # Get all unique categories
            all_categories = set(current_dist.keys()).union(set(reference_dist.keys()))
            
            # Calculate Jensen-Shannon divergence
            current_probs = np.array([current_dist.get(cat, 0) for cat in all_categories])
            reference_probs = np.array([reference_dist.get(cat, 0) for cat in all_categories])
            
            # Add small epsilon to avoid zeros
            epsilon = 1e-10
            current_probs = current_probs + epsilon
            reference_probs = reference_probs + epsilon
            
            # Normalize
            current_probs = current_probs / current_probs.sum()
            reference_probs = reference_probs / reference_probs.sum()
            
            # Calculate Jensen-Shannon divergence
            m = (current_probs + reference_probs) / 2
            js_divergence = (stats.entropy(current_probs, m) + stats.entropy(reference_probs, m)) / 2
            
            # Normalize to 0-1 range (JS divergence is between 0 and ln(2))
            drift_score = min(1.0, js_divergence / np.log(2))
            
            return float(drift_score)
            
        except Exception as e:
            logger.warning(f"Error in categorical drift detection: {e}")
            return 0.0
    
    def _generate_recommendations(self,
                                 df: pd.DataFrame,
                                 quality_metrics: DataQualityMetrics,
                                 bias_metrics: Optional[BiasMetrics],
                                 privacy_metrics: Optional[PrivacyMetrics],
                                 drift_metrics: Optional[DataDriftMetrics],
                                 numerical_features: List[str],
                                 categorical_features: List[str],
                                 target_column: Optional[str]) -> List[DataRecommendation]:
        """
        Generate actionable recommendations based on all analysis results
        
        Returns:
            List of DataRecommendation objects with severity and code examples
        """
        recommendations = []
        
        # Quality recommendations
        self._add_quality_recommendations(
            df, quality_metrics, numerical_features, categorical_features, recommendations
        )
        
        # Bias recommendations
        if bias_metrics:
            self._add_bias_recommendations(
                df, bias_metrics, target_column, recommendations
            )
        
        # Privacy recommendations
        if privacy_metrics:
            self._add_privacy_recommendations(
                df, privacy_metrics, recommendations
            )
        
        # Drift recommendations
        if drift_metrics:
            self._add_drift_recommendations(
                df, drift_metrics, recommendations
            )
        
        # Sort recommendations by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _add_quality_recommendations(self,
                                    df: pd.DataFrame,
                                    metrics: DataQualityMetrics,
                                    numerical_features: List[str],
                                    categorical_features: List[str],
                                    recommendations: List[DataRecommendation]):
        """Add data quality recommendations"""
        
        # Missing value recommendations
        missing_cols = {col: comp for col, comp in metrics.completeness.items() if comp < 0.98}
        for col, completeness in sorted(missing_cols.items(), key=lambda x: x[1]):
            # Calculate missing percentage
            missing_pct = (1 - completeness) * 100
            
            if missing_pct > 20:
                severity = "high"
            elif missing_pct > 5:
                severity = "medium"
            else:
                severity = "low"
                
            # Suggest different imputation methods based on column type
            if col in numerical_features:
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Missing values detected ({missing_pct:.1f}%) in numerical column '{col}'",
                    impact="May reduce model performance and lead to biased predictions",
                    recommendation="Use advanced imputation techniques for numerical data",
                    code_example=f"""
# Method 1: KNN imputation (better than mean/median for preserving relationships)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['{col}'] = imputer.fit_transform(df[['{col}']])[:,0]

# Method 2: Use iterative imputation (models each feature as a function of others)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=42)
df['{col}'] = imp.fit_transform(df[['{col}']])[:,0]
"""
                ))
            elif col in categorical_features:
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Missing values detected ({missing_pct:.1f}%) in categorical column '{col}'",
                    impact="May reduce model performance and lead to biased predictions",
                    recommendation="Use appropriate imputation techniques for categorical data",
                    code_example=f"""
# Method 1: Mode imputation (most frequent value)
mode_value = df['{col}'].mode()[0]
df['{col}'] = df['{col}'].fillna(mode_value)

# Method 2: Create a 'Missing' category
df['{col}'] = df['{col}'].fillna('Missing')

# Method 3: Use predictive model for imputation
from sklearn.ensemble import RandomForestClassifier
# First create a copy of the dataframe without the column to impute
df_temp = df.drop(columns=['{col}'])
# Split data into rows with and without missing values
known = df[df['{col}'].notna()]
unknown = df[df['{col}'].isna()]
# Train a model on known data
X_train = known.drop(columns=['{col}'])
y_train = known['{col}']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict missing values
if len(unknown) > 0:
    X_pred = unknown.drop(columns=['{col}'])
    predictions = model.predict(X_pred)
    df.loc[df['{col}'].isna(), '{col}'] = predictions
"""
                ))
                
        # Outlier recommendations
        for col, outlier_indices in metrics.outlier_scores.items():
            outlier_pct = len(outlier_indices) / len(df) * 100
            
            if outlier_pct > 5:
                severity = "high"
            elif outlier_pct > 1:
                severity = "medium"
            else:
                severity = "low"
                
            recommendations.append(DataRecommendation(
                issue_type="quality",
                severity=severity,
                feature=col,
                description=f"Outliers detected ({outlier_pct:.1f}%) in column '{col}'",
                impact="May skew model training, affect scaling, and reduce robustness",
                recommendation="Consider robust scaling, capping, or removing outliers",
                code_example=f"""
# Method 1: Cap outliers (winsorization)
from scipy import stats
df['{col}_winsorized'] = stats.mstats.winsorize(df['{col}'], limits=[0.05, 0.05])

# Method 2: Apply robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['{col}_scaled'] = scaler.fit_transform(df[['{col}']])

# Method 3: Log transformation to reduce impact of outliers
import numpy as np
if (df['{col}'] > 0).all():  # Only if all values are positive
    df['{col}_log'] = np.log1p(df['{col}'])
"""
            ))
            
        # Distribution skewness recommendations
        for col, stats_dict in metrics.distribution_metrics.items():
            if 'skewness' in stats_dict and abs(stats_dict['skewness']) > 1.0:
                skewness = stats_dict['skewness']
                
                if abs(skewness) > 3:
                    severity = "high"
                elif abs(skewness) > 2:
                    severity = "medium"
                else:
                    severity = "low"
                    
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Highly skewed distribution (skewness={skewness:.2f}) in column '{col}'",
                    impact="May violate model assumptions and reduce predictive performance",
                    recommendation="Apply appropriate transformation to normalize distribution",
                    code_example=f"""
import numpy as np
from scipy import stats

# Method 1: Box-Cox transformation (data must be positive)
if (df['{col}'] > 0).all():
    df['{col}_boxcox'], lambda_value = stats.boxcox(df['{col}'])
    print(f"Box-Cox lambda: {{lambda_value}}")

# Method 2: Yeo-Johnson transformation (works with negative values too)
df['{col}_yeojohnson'], lambda_value = stats.yeojohnson(df['{col}'])
print(f"Yeo-Johnson lambda: {{lambda_value}}")

# Method 3: Simple transformations
if skewness > 0:  # Right-skewed
    # Log transformation for positive right-skewed data
    if (df['{col}'] > 0).all():
        df['{col}_log'] = np.log1p(df['{col}'])
    # Square root for right-skewed data
    if (df['{col}'] >= 0).all():
        df['{col}_sqrt'] = np.sqrt(df['{col}'])
else:  # Left-skewed
    # Square or cube for left-skewed data
    df['{col}_squared'] = df['{col}'] ** 2
"""
                ))
                
        # High cardinality categorical features
        for col in categorical_features:
            if col in metrics.distribution_metrics and 'unique_count' in metrics.distribution_metrics[col]:
                unique_count = metrics.distribution_metrics[col]['unique_count']
                
                if unique_count > 100:
                    severity = "high"
                elif unique_count > 50:
                    severity = "medium"
                elif unique_count > 20:
                    severity = "low"
                else:
                    continue  # Not a concern
                    
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"High cardinality detected in categorical feature '{col}' ({unique_count} unique values)",
                    impact="May cause overfitting, curse of dimensionality when one-hot encoded, and reduce model performance",
                    recommendation="Use dimensionality reduction techniques for categorical features",
                    code_example=f"""
# Method 1: Group rare categories
value_counts = df['{col}'].value_counts()
# Keep top 10 categories, group the rest as 'Other'
top_categories = value_counts.nlargest(10).index
df['{col}_grouped'] = df['{col}'].apply(lambda x: x if x in top_categories else 'Other')

# Method 2: Target encoding (replace categories with target mean)
# Caution: Requires cross-validation to prevent data leakage
if 'target_column' in df:  # Replace with your actual target column
    global_mean = df['target_column'].mean()
    encoding_map = df.groupby('{col}')['target_column'].mean().to_dict()
    df['{col}_encoded'] = df['{col}'].map(encoding_map).fillna(global_mean)

# Method 3: Use hash encoding 
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=20, input_type='string')
hashed_features = hasher.transform(df['{col}'].astype(str))
hashed_df = pd.DataFrame(hashed_features.toarray())
hashed_df.columns = [f'{col}_hash_{{i}}' for i in range(20)]
df = pd.concat([df, hashed_df], axis=1)
"""
                ))
                
        # Correlation recommendations
        if metrics.correlation_matrix is not None:
            # Find highly correlated features
            corr_matrix = metrics.correlation_matrix
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.9:
                        high_corr_pairs.append((col1, col2, abs(corr)))
            
            if high_corr_pairs:
                cols1 = [pair[0] for pair in high_corr_pairs[:3]]
                cols2 = [pair[1] for pair in high_corr_pairs[:3]]
                corrs = [f"{pair[2]:.2f}" for pair in high_corr_pairs[:3]]
                
                col_pairs_str = ", ".join([f"'{c1}'/'{c2}' ({corr})" for c1, c2, corr in zip(cols1, cols2, corrs)])
                
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity="medium",
                    feature=None,
                    description=f"High multicollinearity detected between features: {col_pairs_str}",
                    impact="May cause instability in model coefficients, reduce interpretability, and make models sensitive to small changes in input",
                    recommendation="Apply dimensionality"
        def _create_executive_summary(self, quality_score: Dict[str, float], recommendations: List[Dict]) -> str:
        """Create an executive summary based on findings."""
        overall_score = quality_score["overall"]
        
        critical_issues = sum(1 for r in recommendations if r["severity"] == "Critical")
        high_issues = sum(1 for r in recommendations if r["severity"] == "High")
        
        summary_parts = []
        
        # Overall assessment
        if overall_score >= 90:
            summary_parts.append(f"This dataset demonstrates excellent quality with an overall score of {overall_score:.1f}%.")
        elif overall_score >= 75:
            summary_parts.append(f"This dataset demonstrates good quality with an overall score of {overall_score:.1f}%.")
        elif overall_score >= 60:
            summary_parts.append(f"This dataset demonstrates moderate quality with an overall score of {overall_score:.1f}%.")
        else:
            summary_parts.append(f"This dataset demonstrates poor quality with an overall score of {overall_score:.1f}%.")
        
        # Add details about specific scores
        lowest_score = min((score, name) for name, score in quality_score.items() if name != "overall")
        highest_score = max((score, name) for name, score in quality_score.items() if name != "overall")
        
        summary_parts.append(f"The strongest aspect is {highest_score[1]} ({highest_score[0]:.1f}%), while {lowest_score[1]} ({lowest_score[0]:.1f}%) requires the most attention.")
        
        # Add summary of key issues
        if critical_issues > 0 or high_issues > 0:
            issue_text = f"The analysis detected {critical_issues} critical and {high_issues} high severity issues"
            if self.target_column:
                issue_text += " that may affect model performance."
            else:
                issue_text += " that may affect data quality."
            summary_parts.append(issue_text)
        
        # Add data integrity summary
        if "missing_patterns" in self.report["data_integrity"]:
            missing_cols = sum(1 for v in self.report["data_integrity"]["missing_patterns"].get("missing_percentages", {}).values() if v > 0)
            if missing_cols > 0:
                summary_parts.append(f"Missing data detected in {missing_cols} columns which may require imputation.")
        
        # Add duplicate/corruption summary
        duplicates = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicates > 0:
            summary_parts.append(f"Found {duplicates} duplicate rows ({(duplicates/len(self.df)*100):.1f}% of dataset).")
        
        corrupted = len(self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {}))
        if corrupted > 0:
            summary_parts.append(f"Detected potentially corrupted values in {corrupted} columns.")
        
        # Add outlier summary
        outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
        if outliers:
            outlier_cols = len(outliers)
            if outlier_cols > 0:
                summary_parts.append(f"Found statistical outliers in {outlier_cols} numerical columns.")
        
        # Add class balance summary if target exists
        if self.target_column and "class_balance" in self.report:
            if self.report["class_balance"].get("is_imbalanced", False):
                ratio = self.report["class_balance"].get("imbalance_ratio", 0)
                summary_parts.append(f"Class imbalance detected with an imbalance ratio of {ratio:.1f}x between largest and smallest classes.")
        
        # Add leakage warning if detected
        if self.target_column and "leakage_detection" in self.report:
            leakage = self.report["leakage_detection"].get("high_correlation_features", [])
            if leakage:
                summary_parts.append(f"WARNING: Potential data leakage detected in {len(leakage)} features with suspicious correlations to the target.")
        
        # Add storage efficiency info
        if "storage_efficiency" in self.report:
            savings = self.report["storage_efficiency"].get("savings_percentage", 0)
            if savings > 10:
                summary_parts.append(f"Storage optimization could reduce memory usage by {savings:.1f}%.")
        
        # Add recommendation summary
        if recommendations:
            top_recommendations = sorted(recommendations, key=lambda x: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(x["severity"], 4))[:3]
            if top_recommendations:
                summary_parts.append("\nTop recommendations:")
                for i, rec in enumerate(top_recommendations, 1):
                    summary_parts.append(f"{i}. [{rec['severity']}] {rec['recommendation']}")
        
        return "\n\n".join(summary_parts)

    def export_html_report(self, filename: str = "data_quality_report.html") -> str:
        """
        Generate a comprehensive HTML report with visualizations.
        
        Args:
            filename: Name of HTML file to save
        
        Returns:
            Path to saved HTML file
        """
        # Get comprehensive report
        report_data = self.generate_comprehensive_report(include_figures=True)
        
        # Generate HTML content
        html_content = self._generate_html_report(report_data)
        
        # Save to file
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filepath
        
    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report content from report data"""
        
        # Convert matplotlib figures to base64 for embedding in HTML
        import base64
        from io import BytesIO
        
        embedded_figures = {}
        for name, fig in report_data.get("visualizations", {}).items():
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            embedded_figures[name] = f"data:image/png;base64,{img_data}"
            plt.close(fig)  # Close to prevent memory leaks
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Quality Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }
                h1, h2, h3, h4 {
                    color: #2c3e50;
                }
                .header {
                    background-color: #34495e;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                .card {
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    grid-gap: 15px;
                    margin: 20px 0;
                }
                .metric-box {
                    background-color: #f1f5f9;
                    padding: 15px;
                    border-radius: 6px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }
                .metric-label {
                    font-size: 14px;
                    color: #7f8c8d;
                }
                .figure-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .figure-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 6px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .recommendation {
                    padding: 10px;
                    margin: 10px 0;
                    border-left: 4px solid #3498db;
                    background-color: #f1f5f9;
                }
                .critical {
                    border-left-color: #e74c3c;
                }
                .high {
                    border-left-color: #e67e22;
                }
                .medium {
                    border-left-color: #f1c40f;
                }
                .low {
                    border-left-color: #2ecc71;
                }
                pre {
                    background-color: #f8f8f8;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                    font-size: 14px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .summary {
                    font-size: 16px;
                    line-height: 1.8;
                    white-space: pre-line;
                }
                .tabs {
                    display: flex;
                    margin: 20px 0 0;
                    padding: 0;
                    list-style: none;
                }
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #f1f5f9;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 5px;
                }
                .tab.active {
                    background-color: white;
                    border-bottom: 2px solid #3498db;
                    font-weight: bold;
                }
                .tab-content {
                    display: none;
                    background-color: white;
                    padding: 20px;
                    border-bottom-left-radius: 6px;
                    border-bottom-right-radius: 6px;
                    border-top-right-radius: 6px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .tab-content.active {
                    display: block;
                }
                .footer {
                    text-align: center;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Assessment Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="card">
                <h2>Executive Summary</h2>
                <div class="summary">
                    {executive_summary}
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{overall_score}%</div>
                        <div class="metric-label">Overall Quality</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{completeness_score}%</div>
                        <div class="metric-label">Completeness</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{consistency_score}%</div>
                        <div class="metric-label">Consistency</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{rows_count}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{columns_count}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'data-quality')">Data Quality</div>
                <div class="tab" onclick="openTab(event, 'visualizations')">Visualizations</div>
                <div class="tab" onclick="openTab(event, 'recommendations')">Recommendations</div>
                <div class="tab" onclick="openTab(event, 'detailed-stats')">Detailed Statistics</div>
            </div>
            
            <div id="data-quality" class="tab-content active">
                <h3>Data Integrity Issues</h3>
                <table>
                    <tr>
                        <th>Issue Type</th>
                        <th>Impact</th>
                        <th>Details</th>
                    </tr>
                    {data_quality_rows}
                </table>
                
                {class_balance_section}
                
                {storage_efficiency_section}
            </div>
            
            <div id="visualizations" class="tab-content">
                <h3>Data Quality Visualizations</h3>
                {visualizations}
            </div>
            
            <div id="recommendations" class="tab-content">
                <h3>Recommendations</h3>
                {recommendations}
            </div>
            
            <div id="detailed-stats" class="tab-content">
                <h3>Detailed Column Statistics</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing</th>
                        <th>Unique Values</th>
                        <th>Statistics</th>
                    </tr>
                    {column_stats_rows}
                </table>
            </div>
            
            <div class="footer">
                Generated by DataGuardianEnhancer | Neural-Scope | {timestamp}
            </div>
            
            <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;
                    
                    // Hide all tab content
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }
                    
                    // Remove active class from all tabs
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }
                    
                    // Show the selected tab and add an active class
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }
            </script>
        </body>
        </html>
        """
        
        # Prepare data for HTML template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        quality_scores = report_data["quality_score"]
        
        # Create data quality issues table rows
        data_quality_issues = []
        
        # Missing values
        missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
        if missing_patterns:
            miss_cols = sum(1 for v in missing_patterns.get("missing_percentages", {}).values() if v > 0)
            if miss_cols > 0:
                data_quality_issues.append({
                    "type": "Missing Values",
                    "impact": "Medium" if miss_cols > self.df.shape[1] * 0.2 else "Low",
                    "details": f"Found missing values in {miss_cols} columns"
                })
        
        # Duplicates
        duplicates = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicates > 0:
            data_quality_issues.append({
                "type": "Duplicate Rows",
                "impact": "High" if duplicates > len(self.df) * 0.05 else "Medium",
                "details": f"Found {duplicates} duplicate rows ({(duplicates/len(self.df)*100):.1f}% of dataset)"
            })
        
        # Outliers
        outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
        if outliers:
            outlier_cols = len(outliers)
            outlier_details = ", ".join([f"{col}: {info['percentage']:.1f}%" 
                                       for col, info in list(outliers.items())[:3]])
            if outlier_cols > 3:
                outlier_details += f" and {outlier_cols-3} more columns"
                
            data_quality_issues.append({
                "type": "Statistical Outliers",
                "impact": "High" if outlier_cols > 3 else "Medium",
                "details": f"Detected outliers in {outlier_cols} columns. {outlier_details}"
            })
        
        # Corrupted values
        corrupted = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
        if corrupted:
            data_quality_issues.append({
                "type": "Potentially Corrupted Values",
                "impact": "High",
                "details": f"Detected potentially corrupted values in {len(corrupted)} columns"
            })
        
        # Build data quality rows HTML
        data_quality_rows = ""
        for issue in data_quality_issues:
            data_quality_rows += f"""
            <tr>
                <td>{issue['type']}</td>
                <td>{issue['impact']}</td>
                <td>{issue['details']}</td>
            </tr>
            """
        
        # Class balance section
        class_balance_section = ""
        if self.target_column and "class_balance" in self.report:
            class_info = self.report["class_balance"]
            is_imbalanced = class_info.get("is_imbalanced", False)
            
            class_balance_section = """
            <h3>Class Balance Analysis</h3>
            <div class="card">
            """
            
            if is_imbalanced:
                ratio = class_info.get("imbalance_ratio", 0)
                class_balance_section += f"""
                <p>⚠️ <strong>Class imbalance detected</strong> with an imbalance ratio of {ratio:.1f}x between largest and smallest classes.</p>
                """
                
                if "class_distribution" in class_info:
                    class_balance_section += """
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    """
                    
                    for cls, data in class_info["class_distribution"].items():
                        class_balance_section += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{data['count']}</td>
                            <td>{data['percentage']:.1f}%</td>
                        </tr>
                        """
                    
                    class_balance_section += """
                    </table>
                    """
            else:
                class_balance_section += """
                <p>✓ Classes are well balanced in the target variable.</p>
                """
            
            class_balance_section += """
            </div>
            """
        
        # Storage efficiency section
        storage_efficiency_section = ""
        if "storage_efficiency" in self.report:
            storage_info = self.report["storage_efficiency"]
            savings = storage_info.get("savings_percentage", 0)
            
            storage_efficiency_section = """
            <h3>Storage Efficiency Analysis</h3>
            <div class="card">
            """
            
            if savings > 0:
                storage_efficiency_section += f"""
                <p>💾 Storage optimization could <strong>reduce memory usage by {savings:.1f}%</strong>.</p>
                """
                
                if "format_recommendations" in storage_info:
                    formats = storage_info["format_recommendations"]
                    if "recommendation" in formats:
                        storage_efficiency_section += f"""
                        <p><strong>Recommendation:</strong> {formats['recommendation']}</p>
                        """
                    
                    if len(formats) > 2:  # More than just recommendation and code_example
                        storage_efficiency_section += """
                        <table>
                            <tr>
                                <th>Format</th>
                                <th>Size (MB)</th>
                                <th>Relative Load Time</th>
                            </tr>
                        """
                        
                        for fmt, data in formats.items():
                            if fmt not in ["recommendation", "code_example"]:
                                storage_efficiency_section += f"""
                                <tr>
                                    <td>{fmt}</td>
                                    <td>{data.get('size_mb', 0):.2f}</td>
                                    <td>{data.get('relative_load_time', 1.0):.2f}x</td>
                                </tr>
                                """
                        
                        storage_efficiency_section += """
                        </table>
                        """
            else:
                storage_efficiency_section += """
                <p>✓ Storage is already efficient.</p>
                """
            
            storage_efficiency_section += """
            </div>
            """
        
        # Visualization section
        visualizations = ""
        for name, img_data in embedded_figures.items():
            title = " ".join(name.split("_")).title()
            visualizations += f"""
            <div class="figure-container">
                <h4>{title}</h4>
                <img src="{img_data}" alt="{title}">
            </div>
            """
        
        # Recommendations section
        recommendations_html = ""
        for rec in report_data.get("recommendations", []):
            severity = rec["severity"].lower()
            feature = f"<strong>{rec['feature']}</strong>: " if rec.get('feature') else ""
            
            recommendations_html += f"""
            <div class="recommendation {severity}">
                <h4>{rec['severity']} Issue: {feature}{rec['description']}</h4>
                <p><strong>Impact:</strong> {rec['impact']}</p>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <details>
                    <summary>Code Example</summary>
                    <pre><code>{rec['code_example']}</code></pre>
                </details>
            </div>
            """
        
        # Column statistics
        column_stats_rows = ""
        for col in self.df.columns:
            col_type = str(self.df[col].dtype)
            missing = self.df[col].isna().sum()
            missing_pct = missing / len(self.df) * 100
            unique_values = self.df[col].nunique()
            
            # Generate statistics based on column type
            if col in self.numeric_cols:
                stats = self.df[col].describe().to_dict()
                stats_text = f"""
                Min: {stats.get('min', 'N/A'):.2f}<br>
                Max: {stats.get('max', 'N/A'):.2f}<br>
                Mean: {stats.get('mean', 'N/A'):.2f}<br>
                Std Dev: {stats.get('std', 'N/A'):.2f}
                """
            else:
                top_values = self.df[col].value_counts().head(3).to_dict()
                stats_text = "Top values:<br>" + "<br>".join([f"{k}: {v}" for k, v in top_values.items()])
            
            column_stats_rows += f"""
            <tr>
                <td>{col}</td>
                <td>{col_type}</td>
                <td>{missing} ({missing_pct:.1f}%)</td>
                <td>{unique_values}</td>
                <td>{stats_text}</td>
            </tr>
            """
        
        # Fill in the template
        html_content = html_template.format(
            timestamp=timestamp,
            executive_summary=report_data.get("executive_summary", "No summary available."),
            overall_score=round(quality_scores.get("overall", 0)),
            completeness_score=round(quality_scores.get("completeness", 0)),
            consistency_score=round(quality_scores.get("consistency", 0)),
            rows_count=len(self.df),
            columns_count=len(self.df.columns),
            data_quality_rows=data_quality_rows,
            class_balance_section=class_balance_section,
            storage_efficiency_section=storage_efficiency_section,
            visualizations=visualizations,
            recommendations=recommendations_html,
            column_stats_rows=column_stats_rows
        )
        
        return html_content
    
    def save_report_to_json(self, filename: str = "data_quality_report.json") -> str:
        """
        Save the comprehensive report to a JSON file.
        
        Args:
            filename: Output JSON file name
            
        Returns:
            Path to saved JSON file
        """
        # Get report data (without figures which can't be serialized)
        report_data = self.generate_comprehensive_report(include_figures=False)
        
        # Convert any numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Convert report to serializable format
        serializable_report = convert_to_serializable(report_data)
        
        # Save to file
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2)
            
        return filepath
    
    def get_automated_fixes_pipeline(self) -> Callable:
        """
        Create a callable pipeline that applies recommended automatic fixes.
        
        Returns:
            Function that can be called on a dataframe to apply all recommended fixes
        """
        # Get recommended fixes
        recommended_fixes = self._recommend_automatic_fixes()["recommended_fixes"]
        
        # Create pipeline function
        def pipeline(df: pd.DataFrame) -> pd.DataFrame:
            """Apply all recommended fixes to the dataframe"""
            result_df, fixes_applied = self.apply_fixes(recommended_fixes)
            
            # Print summary of applied fixes
            print(f"Applied {len(fixes_applied)} automatic fixes:")
            for fix_type, details in fixes_applied.items():
                if isinstance(details, dict):
                    print(f"- {fix_type}: Applied to {len(details)} columns")
                else:
                    print(f"- {fix_type}: {details}")
                    
            return result_df
        
        return pipeline
                    