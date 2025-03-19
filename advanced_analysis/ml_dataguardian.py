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
                                # Train a simple model
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                
                # Get feature importance using SHAP
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Analyze feature importance for protected attributes
                for attr in protected_attributes:
                    if attr in df.columns:
                        # Find columns derived from this protected attribute
                        related_cols = [col for col in X.columns if col.startswith(f"{attr}_")]
                        
                        if related_cols:
                            # Calculate importance of protected attribute
                            importance = np.abs(shap_values).mean(axis=0)
                            attr_importance = sum(importance[X.columns.get_loc(col)] for col in related_cols)
                            metrics.feature_importance_bias[attr] = float(attr_importance)
            except Exception as e:
                logger.warning(f"Error calculating SHAP feature importance: {e}")
        
        # Calculate overall bias score
        bias_scores = []
        
        # Statistical parity contribution
        if metrics.statistical_parity:
            bias_scores.append(min(1.0, sum(metrics.statistical_parity.values()) / len(metrics.statistical_parity)))
            
        # Disparate impact contribution (normalized to 0-1)
        if metrics.disparate_impact:
            di_score = 0.0
            for di in metrics.disparate_impact.values():
                # Both very low and very high DI indicate bias
                di_norm = abs(di - 1.0)
                di_score += min(1.0, di_norm)
            bias_scores.append(di_score / len(metrics.disparate_impact))
            
        # Intersectional bias contribution
        if metrics.intersectional_bias_scores:
            bias_scores.append(min(1.0, sum(abs(v) for v in metrics.intersectional_bias_scores.values()) / 
                                len(metrics.intersectional_bias_scores)))
                                
        # Feature importance bias contribution
        if metrics.feature_importance_bias:
            # Normalize feature importance scores
            total_importance = sum(metrics.feature_importance_bias.values())
            if total_importance > 0:
                normalized_scores = {k: v/total_importance for k, v in metrics.feature_importance_bias.items()}
                max_score = max(normalized_scores.values())
                bias_scores.append(max_score)
        
        # Calculate final bias score
        metrics.overall_bias_score = float(np.mean(bias_scores)) if bias_scores else 0.0
        
        logger.info(f"Bias assessment completed. Overall bias score: {metrics.overall_bias_score:.4f}")
        return metrics
    
    def _should_detect_pii(self) -> bool:
        """Determine if automatic PII detection should be performed"""
        # Enable PII detection for datasets with over 1000 rows or 10+ columns
        return True
    
    def _assess_privacy(self, 
                        df: pd.DataFrame, 
                        sensitive_attributes: Optional[List[str]] = None) -> PrivacyMetrics:
        """
        Assess privacy risks and detect PII in the dataset
        
        Args:
            df: DataFrame to analyze
            sensitive_attributes: Optional list of columns known to contain sensitive information
            
        Returns:
            PrivacyMetrics with privacy risk assessment
        """
        metrics = PrivacyMetrics()
        logger.info("Starting privacy assessment")
        
        # Use provided sensitive attributes or detect automatically
        sensitive_attrs = sensitive_attributes or []
        
        # Auto-detect PII in text columns
        for col in df.columns:
            if col in sensitive_attrs:
                metrics.pii_detected[col] = ["user_specified"]
                continue
                
            if df[col].dtype == 'object':
                # Sample values to check for PII
                sample = df[col].dropna().astype(str).sample(min(1000, len(df))).tolist()
                
                detected_pii_types = []
                
                # Check each PII pattern
                for pii_type, pattern in self.pii_patterns.items():
                    # Count matches in the sample
                    matches = sum(1 for val in sample if re.search(pattern, val))
                    match_ratio = matches / len(sample) if sample else 0
                    
                    # If significant matches found, mark as PII
                    if match_ratio > 0.1:  # At least 10% of values match the pattern
                        detected_pii_types.append(pii_type)
                
                if detected_pii_types:
                    metrics.pii_detected[col] = detected_pii_types
        
        # Calculate k-anonymity
        if len(df) > 0:
            # Use quasi-identifiers (categorical columns that aren't sensitive)
            quasi_identifiers = [col for col in df.columns 
                              if col in df.select_dtypes(include=['object', 'category']).columns
                              and col not in metrics.pii_detected]
            
            if quasi_identifiers:
                # Group by all quasi-identifiers and count occurrences
                try:
                    grouped = df.groupby(quasi_identifiers).size()
                    k_anonymity = grouped.min() if not grouped.empty else len(df)
                    metrics.k_anonymity = int(k_anonymity)
                except Exception as e:
                    logger.warning(f"Error calculating k-anonymity: {e}")
                    metrics.k_anonymity = 0
        
        # Calculate uniqueness risk
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio > 0.9:  # High cardinality columns pose uniqueness risk
                metrics.uniqueness_risk[col] = float(unique_ratio)
        
        # Calculate l-diversity for sensitive attributes
        for sensitive_col in metrics.pii_detected.keys():
            if sensitive_col in df.columns:
                # Use quasi-identifiers to create groups
                quasi_identifiers = [col for col in df.columns 
                                  if col in df.select_dtypes(include=['object', 'category']).columns
                                  and col != sensitive_col
                                  and col not in metrics.pii_detected]
                
                if quasi_identifiers:
                    try:
                        # Find minimum distinct values of sensitive attribute within any group
                        l_diversity_values = []
                        for _, group in df.groupby(quasi_identifiers):
                            if len(group) > 0:
                                l_diversity_values.append(group[sensitive_col].nunique())
                        
                        if l_diversity_values:
                            metrics.l_diversity[sensitive_col] = int(min(l_diversity_values))
                    except Exception as e:
                        logger.warning(f"Error calculating l-diversity for {sensitive_col}: {e}")
        
        # Calculate overall privacy risk score (0-1, higher means higher risk)
        risk_factors = []
        
        # Risk from detected PII
        if metrics.pii_detected:
            risk_factors.append(min(1.0, len(metrics.pii_detected) / len(df.columns)))
        
        # Risk from low k-anonymity (inversely proportional)
        if metrics.k_anonymity > 0:
            k_risk = min(1.0, 10 / metrics.k_anonymity)  # Higher risk for k < 10
            risk_factors.append(k_risk)
            
        # Risk from high uniqueness
        if metrics.uniqueness_risk:
            risk_factors.append(min(1.0, sum(metrics.uniqueness_risk.values()) / len(metrics.uniqueness_risk)))
            
        # Risk from low l-diversity
        if metrics.l_diversity:
            l_risk = min(1.0, sum(3 / max(ld, 1) for ld in metrics.l_diversity.values()) / len(metrics.l_diversity))
            risk_factors.append(l_risk)
        
        # Calculate overall risk score
        metrics.overall_privacy_risk = float(np.mean(risk_factors)) if risk_factors else 0.0
        
        logger.info(f"Privacy assessment completed. Overall risk score: {metrics.overall_privacy_risk:.4f}")
        return metrics
    
    def _assess_drift(self, 
                      df: pd.DataFrame, 
                      reference_df: pd.DataFrame) -> DataDriftMetrics:
        """
        Assess data drift between current dataset and reference dataset
        
        Args:
            df: Current DataFrame to analyze
            reference_df: Reference DataFrame (e.g., training data)
            
        Returns:
            DataDriftMetrics with drift assessment
        """
        metrics = DataDriftMetrics()
        logger.info("Starting drift assessment")
        
        # Ensure DataFrames have the same columns
        common_columns = set(df.columns).intersection(set(reference_df.columns))
        if not common_columns:
            logger.warning("No common columns between current and reference datasets")
            return metrics
            
        numerical_columns = [col for col in common_columns 
                          if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(reference_df[col])]
        
        categorical_columns = [col for col in common_columns 
                            if not pd.api.types.is_numeric_dtype(df[col]) or not pd.api.types.is_numeric_dtype(reference_df[col])]
        
        # Check distribution drift for numerical features
        for col in numerical_columns:
            try:
                # Skip columns with too many missing values
                if df[col].isna().mean() > 0.3 or reference_df[col].isna().mean() > 0.3:
                    continue
                
                # Kolmogorov-Smirnov test for distribution comparison
                ks_statistic, p_value = stats.ks_2samp(
                    df[col].dropna().values, 
                    reference_df[col].dropna().values
                )
                
                # Earth Mover's Distance (Wasserstein)
                try:
                    emd = stats.wasserstein_distance(
                        df[col].dropna().values, 
                        reference_df[col].dropna().values
                    )
                except Exception:
                    emd = np.nan
                
                # Population Stability Index (PSI)
                psi = self._calculate_psi(reference_df[col], df[col])
                
                metrics.distribution_shifts[col] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'earth_movers_distance': float(emd) if not np.isnan(emd) else None,
                    'psi': float(psi),
                    'significant_drift': p_value < 0.05 and ks_statistic > 0.1
                }
                
                # Store feature drift score (normalized KS statistic)
                metrics.feature_drift[col] = float(ks_statistic)
                
            except Exception as e:
                logger.warning(f"Error calculating drift for {col}: {e}")
        
        # Check distribution drift for categorical features
        for col in categorical_columns:
            try:
                # Calculate Chi-square test for categorical distributions
                curr_counts = df[col].value_counts(normalize=True, dropna=True).to_dict()
                ref_counts = reference_df[col].value_counts(normalize=True, dropna=True).to_dict()
                
                # Create consistent category set
                all_categories = set(curr_counts.keys()).union(set(ref_counts.keys()))
                curr_dist = np.array([curr_counts.get(cat, 0) for cat in all_categories])
                ref_dist = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                
                # Ensure non-zero values for chi-square (add small epsilon)
                curr_dist = np.maximum(curr_dist, 1e-10)
                ref_dist = np.maximum(ref_dist, 1e-10)
                
                # Normalize distributions
                curr_dist = curr_dist / np.sum(curr_dist)
                ref_dist = ref_dist / np.sum(ref_dist)
                
                # Jensen-Shannon divergence as drift measure
                js_divergence = stats.entropy(curr_dist, (curr_dist + ref_dist) / 2) / 2 + \
                               stats.entropy(ref_dist, (curr_dist + ref_dist) / 2) / 2
                
                metrics.distribution_shifts[col] = {
                    'js_divergence': float(js_divergence),
                    'significant_drift': js_divergence > 0.1,
                    'category_changes': len(set(curr_counts.keys()).symmetric_difference(set(ref_counts.keys())))
                }
                
                # Store feature drift score (normalized JS divergence)
                metrics.feature_drift[col] = float(min(1.0, js_divergence))
                
            except Exception as e:
                logger.warning(f"Error calculating categorical drift for {col}: {e}")
        
        # Calculate overall drift score
        if metrics.feature_drift:
            metrics.overall_drift_score = float(np.mean(list(metrics.feature_drift.values())))
        
        logger.info(f"Drift assessment completed. Overall drift score: {metrics.overall_drift_score:.4f}")
        return metrics
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Handle non-numeric data
        if not pd.api.types.is_numeric_dtype(expected) or not pd.api.types.is_numeric_dtype(actual):
            return 0.0
            
        # Remove missing values
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        # Create equal-width bins based on expected distribution
        try:
            breaks = np.linspace(expected.min(), expected.max(), bins + 1)
            
            # Calculate bin frequencies
            expected_counts, _ = np.histogram(expected, bins=breaks)
            actual_counts, _ = np.histogram(actual, bins=breaks)
            
            # Convert to percentages and avoid division by zero
            expected_percents = expected_counts / expected.shape[0]
            actual_percents = actual_counts / actual.shape[0]
            
            # Replace zeros with small epsilon
            expected_percents = np.maximum(expected_percents, 1e-6)
            actual_percents = np.maximum(actual_percents, 1e-6)
            
            # Calculate PSI
            psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
            psi = np.sum(psi_values)
            
            return psi
        except Exception:
            return 0.0
    
    def _generate_recommendations(self,
                                 df: pd.DataFrame,
                                 quality_metrics: DataQualityMetrics,
                                 bias_metrics: Optional[BiasMetrics] = None,
                                 privacy_metrics: Optional[PrivacyMetrics] = None,
                                 drift_metrics: Optional[DataDriftMetrics] = None,
                                 numerical_features: List[str] = None,
                                 categorical_features: List[str] = None,
                                 target_column: Optional[str] = None) -> List[DataRecommendation]:
        """
        Generate comprehensive, actionable recommendations based on all findings
        
        Args:
            df: DataFrame that was analyzed
            quality_metrics: Data quality metrics
            bias_metrics: Optional bias metrics
            privacy_metrics: Optional privacy metrics
            drift_metrics: Optional drift metrics
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target_column: Name of target/label column
            
        Returns:
            List of DataRecommendation objects with actionable insights
        """
        recommendations = []
        
        # Quality recommendations - missing values
        for col, completeness in quality_metrics.completeness.items():
            if completeness < 0.9:  # More than 10% missing
                severity = "high" if completeness < 0.7 else "medium"
                
                # Different recommendations based on column type
                if col in numerical_features:
                    recommendations.append(DataRecommendation(
                        issue_type="quality",
                        severity=severity,
                        feature=col,
                        description=f"High missing value rate ({(1-completeness)*100:.1f}%) in numerical column '{col}'",
                        impact="Reduces model reliability and may introduce bias if not randomly distributed",
                        recommendation="Impute missing values with mean/median or use advanced imputation methods",
                        code_example=f"""
# Option 1: Simple imputation with median (robust to outliers)
df['{col}'] = df['{col}'].fillna(df['{col}'].median())

# Option 2: KNN imputation (preserves relationships)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['{col}'] = imputer.fit_transform(df[['{col}']])[:, 0]
"""
                    ))
                elif col in categorical_features:
                    recommendations.append(DataRecommendation(
                        issue_type="quality",
                        severity=severity,
                        feature=col,
                        description=f"High missing value rate ({(1-completeness)*100:.1f}%) in categorical column '{col}'",
                        impact="Reduces model reliability and may introduce bias",
                        recommendation="Impute with mode or create a 'Missing' category",
                        code_example=f"""
# Option 1: Create explicit 'Missing' category
df['{col}_with_missing'] = df['{col}'].fillna('Missing')

# Option 2: Mode imputation
df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])
"""
                    ))
        
        # Quality recommendations - outliers
        for col, outliers in quality_metrics.outlier_scores.items():
            outlier_pct = len(outliers) / len(df) * 100
            if outlier_pct > 1.0:  # More than 1% outliers
                severity = "medium" if outlier_pct > 5.0 else "low"
                
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity=severity,
                    feature=col,
                    description=f"Contains {outlier_pct:.1f}% outliers in column '{col}'",
                    impact="May skew statistics and reduce model performance",
                    recommendation="Consider capping/flooring, transformation, or removal based on domain knowledge",
                    code_example=f"""
# Option 1: Cap outliers at percentiles
Q1 = df['{col}'].quantile(0.25)
Q3 = df['{col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['{col}_capped'] = df['{col}'].clip(lower_bound, upper_bound)

# Option 2: Log transformation (reduces impact of outliers)
import numpy as np
df['{col}_log'] = np.log1p(df['{col}'] - df['{col}'].min() + 1 if df['{col}'].min() <= 0 else df['{col}'])
"""
                ))
        
        # Quality recommendations - skewed distributions
        for col, stats_dict in quality_metrics.distribution_metrics.items():
            if 'skewness' in stats_dict and abs(stats_dict['skewness']) > 2.0:
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity="medium",
                    feature=col,
                    description=f"Highly skewed distribution (skewness={stats_dict['skewness']:.2f}) in column '{col}'",
                    impact="May violate model assumptions and reduce performance",
                    recommendation="Apply appropriate transformation to normalize distribution",
                    code_example=f"""
import numpy as np
from scipy import stats

# Option 1: Box-Cox transformation (data must be positive)
if (df['{col}'] > 0).all():
    df['{col}_boxcox'], lambda_value = stats.boxcox(df['{col}'])
    print(f"Optimal lambda: {{lambda_value}}")

# Option 2: Yeo-Johnson transformation (works with negative values)
df['{col}_yeojohnson'], lambda_value = stats.yeojohnson(df['{col}'])
print(f"Optimal lambda: {{lambda_value}}")

# Option 3: Log transformation
if (df['{col}'] > 0).all():
    df['{col}_log'] = np.log(df['{col}'])
else:
    df['{col}_log'] = np.log(df['{col}'] - df['{col}'].min() + 1)
"""
                ))
        
        # Correlation recommendations
        if quality_metrics.correlation_matrix is not None:
            # Find highly correlated features
            corr_matrix = quality_metrics.correlation_matrix.abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j])
                             for i, j in zip(*np.where(upper_tri > 0.9))]
            
            if high_corr_pairs:
                # Create recommendation for highly correlated features
                corr_pairs_str = "\n".join([f"- {col1} & {col2}: {corr:.2f}" for col1, col2, corr in high_corr_pairs[:5]])
                if len(high_corr_pairs) > 5:
                    corr_pairs_str += f"\n- And {len(high_corr_pairs) - 5} more pairs..."
                
                recommendations.append(DataRecommendation(
                    issue_type="quality",
                    severity="medium",
                    feature=None,
                    description=f"Found {len(high_corr_pairs)} pairs of highly correlated features (>0.9)",
                    impact="Multicollinearity can destabilize model coefficients and reduce interpretability",
                    recommendation="Consider dimensionality reduction or removing redundant features",
                    code_example=f"""
# Option 1: Remove one feature from each highly correlated pair
# Correlations:
{corr_pairs_str}

# Option 2: Apply PCA to reduce dimensionality
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

numerical_features = {numerical_features}
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_features])
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_result = pca.fit_transform(scaled_data)

# Create dataframe with PCA components
pca_df = pd.DataFrame(
    data=pca_result, 
    columns=[f'PC{{i+1}}' for i in range(pca_result.shape[1])]
)
print(f"Reduced from {{len(numerical_features)}} to {{pca_result.shape[1]}} features")
"""
                ))
        
        # Bias recommendations
        if bias_metrics and bias_metrics.overall_bias_score > 0.1:
            # Find most biased attribute
            if bias_metrics.statistical_parity:
                most_biased_attr = max(bias_metrics.statistical_parity.items(), key=lambda x: x[1])
                attr_name, bias_score = most_biased_attr
                
                if bias_score > 0.1:  # Significant bias
                    severity = "high" if bias_score > 0.2 else "medium"
                    
                    recommendations.append(DataRecommendation(
                        issue_type="bias",
                        severity=severity,
                        feature=attr_name,
                        description=f"Significant statistical parity difference ({bias_score:.2f}) for attribute '{attr_name}'",
                        impact="Model predictions may discriminate against protected groups",
                        recommendation="Apply bias mitigation techniques during preprocessing or model training",
                        code_example=f"""
# Option 1: Reweighing (pre-processing technique)
from fairlearn.preprocessing import CorrelationRemover

# Remove correlation with protected attribute
remover = CorrelationRemover(sensitive_feature_names=['{attr_name}'])
X_filtered = remover.fit_transform(df.drop(columns=['{target_column}' if target_column else '']))

# Option 2: Adversarial debiasing (in-processing, using TensorFlow)
# See: https://github.com/fairlearn/fairlearn/

# Option 3: Post-processing calibration
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['{target_column}', '{attr_name}']),
    df['{target_column}'],
    test_size=0.2, random_state=42
)

# Train your model (example with RandomForest)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)

# Apply threshold optimization for fairness
threshold_optimizer = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    prefit=True
)
threshold_optimizer.fit(
    X_train, y_train, 
    sensitive_features=df.loc[X_train.index, '{attr_name}']
)
"""
                    ))
                    
            # Intersectional bias recommendations
            if bias_metrics.intersectional_bias_scores:
                most_biased_intersection = max(bias_metrics.intersectional_bias_scores.items(), key=lambda x: abs(x[1]))
                intersection_name, bias_score = most_biased_intersection
                
                if abs(bias_score) > 0.15:  # Significant intersectional bias
                    attr1, attr2 = intersection_name.split('_')
                    recommendations.append(DataRecommendation(
                        issue_type="bias",
                        severity="high",
                        feature=intersection_name,
                        description=f"Significant intersectional bias ({bias_score:.2f}) between '{attr1}' and '{attr2}'",
                        impact="Model may discriminate against specific intersectional groups",
                        recommendation="Analyze intersectional fairness metrics and apply targeted mitigation",
                        code_example=f"""
# Create intersectional groups for analysis
df['intersectional_group'] = df['{attr1}'].astype(str) + '_' + df['{attr2}'].astype(str)

# Analyze model performance across intersectional groups
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt

# After training your model and getting predictions:
# predictions = model.predict(X_test)

# Analyze fairness across intersectional groups
dpd = demographic_parity_difference(
    y_true=y_test, 
    y_pred=predictions,
    sensitive_features=df.loc[X_test.index, 'intersectional_group']
)
print(f"Demographic Parity Difference: {{dpd:.4f}}")

# Visualize predictions by intersectional group
group_performance = {{'group': [], 'positive_rate': []}}
for group, group_df in df.loc[X_test.index].groupby('intersectional_group'):
    group_indices = group_df.index
    group_performance['group'].append(group)
    group_performance['positive_rate'].append(
        predictions[group_indices].mean()
    )

performance_df = pd.DataFrame(group_performance)
performance_df.sort_values('positive_rate').plot(kind='bar', x='group', y='positive_rate')
plt.title('Prediction Rate by Intersectional Group')
plt.ylabel('Positive Prediction Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
                    ))
        
        # Privacy recommendations
        if privacy_metrics and privacy_metrics.overall_privacy_risk > 0.3:
            # PII recommendations
            if privacy_metrics.pii_detected:
                pii_columns = list(privacy_metrics.pii_detected.keys())
                pii_types = {col: types for col, types in privacy_metrics.pii_detected.items()}
                
                # Format PII types for display
                pii_info = "\n".join([f"- {col}: {', '.join(types)}" for col, types in pii_types.items()])
                
                recommendations.append(DataRecommendation(
                    issue_type="privacy",
                    severity="critical",
                    feature=None,
                    description=f"Detected PII in {len(pii_columns)} columns: {', '.join(pii_columns[:3])}{'...' if len(pii_columns) > 3 else ''}",
                    impact="Privacy risk, potential regulatory compliance issues (GDPR, CCPA, etc.)",
                    recommendation="Apply anonymization techniques before using for ML",
                    code_example=f"""
# PII detected in:
{pii_info}

# Option 1: Hashing sensitive values
import hashlib

for col in {pii_columns}:
    if col in df.columns:
        # Create a hashed version, keeping nulls as nulls
        df[f'{{col}}_hashed'] = df[col].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None
        )
        # Drop original column
        df = df.drop(columns=[col])

# Option 2: Tokenization (better than hashing for ML)
from sklearn.preprocessing import OrdinalEncoder

for col in {pii_columns}:
    if col in df.columns and df[col].dtype == 'object':
                # Create tokenized version
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[f'{col}_tokenized'] = encoder.fit_transform(df[[col]])
        # Drop original column
        df = df.drop(columns=[col])

# Option 3: Differential privacy (using diffprivlib)
# pip install diffprivlib
try:
    import diffprivlib as dp
    
    # Example for numeric columns
    # For each PII column that is numeric, apply differential privacy
    numeric_pii = [col for col in {pii_columns} if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_pii:
        # Apply Laplace noise to protect privacy
        dp_mean = dp.mean(df[col].dropna(), epsilon=1.0)
        dp_std = dp.std(df[col].dropna(), epsilon=1.0)
        # Create a perturbed version of the data
        laplace_mech = dp.mechanisms.Laplace(epsilon=1.0, sensitivity=(df[col].max() - df[col].min()))
        df[f'{col}_private'] = df[col].apply(lambda x: laplace_mech.randomise(x) if pd.notna(x) else None)
        # Drop original
        df = df.drop(columns=[col])
except ImportError:
    pass  # diffprivlib not available
"""
                ))
                
                # K-anonymity recommendations if k is low
                if privacy_metrics.k_anonymity < 5 and privacy_metrics.k_anonymity > 0:
                    recommendations.append(DataRecommendation(
                        issue_type="privacy",
                        severity="high",
                        feature=None,
                        description=f"Low k-anonymity (k={privacy_metrics.k_anonymity})",
                        impact="Individuals might be re-identifiable from quasi-identifiers",
                        recommendation="Apply generalization, suppression, or perturbation to increase anonymity",
                        code_example="""
# Increase k-anonymity through generalization
# Example: Binning age values to age groups

# Option 1: Binning numerical columns
def bin_column(df, column, bins, labels=None):
    """Create binned version of a numerical column"""
    if labels is None:
        labels = [f'Group {i+1}' for i in range(len(bins)-1)]
    df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

# Example: Age binning
if 'age' in df.columns:
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['0-18', '19-35', '36-50', '51-65', '65+']
    df = bin_column(df, 'age', bins, labels)
    df = df.drop(columns=['age'])  # Remove original column

# Option 2: Generalize categorical columns
def generalize_zipcode(df, column, digits_to_keep=3):
    """Generalize ZIP codes by keeping only first N digits"""
    df[f'{column}_generalized'] = df[column].astype(str).str[:digits_to_keep] + 'XX'
    return df

# Example: ZIP code generalization
if 'zipcode' in df.columns:
    df = generalize_zipcode(df, 'zipcode', 3)
    df = df.drop(columns=['zipcode'])  # Remove original
"""
                    ))
        
        # Drift recommendations
        if drift_metrics and drift_metrics.overall_drift_score > 0.1:
            # Find features with significant drift
            significant_drift = sorted(
                [(col, score) for col, score in drift_metrics.feature_drift.items() if score > 0.1],
                key=lambda x: x[1],
                reverse=True
            )
            
            if significant_drift:
                # Format drift info for display
                drift_info = "\n".join([f"- {col}: {score:.3f}" for col, score in significant_drift[:5]])
                if len(significant_drift) > 5:
                    drift_info += f"\n- And {len(significant_drift) - 5} more features..."
                
                recommendations.append(DataRecommendation(
                    issue_type="drift",
                    severity="high" if drift_metrics.overall_drift_score > 0.3 else "medium",
                    feature=None,
                    description=f"Significant data drift detected (score: {drift_metrics.overall_drift_score:.2f})",
                    impact="Model may perform poorly on new data due to distribution shifts",
                    recommendation="Consider model retraining, concept drift adaptation, or domain adaptation techniques",
                    code_example=f"""
# Significant drift detected in:
{drift_info}

# Option 1: Visualize distribution differences
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions for top drifted features
top_drift_features = {[col for col, _ in significant_drift[:3]]}

for col in top_drift_features:
    if col in df.columns:
        plt.figure(figsize=(10, 4))
        
        # Plot current vs reference distributions
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric features
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), color='blue', label='Current', alpha=0.6)
            sns.histplot(reference_df[col].dropna(), color='red', label='Reference', alpha=0.6)
            plt.title(f'Distribution Comparison: {col}')
            plt.legend()
            
            # QQ-plot
            plt.subplot(1, 2, 2)
            from scipy import stats
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title("Q-Q Plot")
        else:
            # For categorical features
            curr_counts = df[col].value_counts(normalize=True).sort_index()
            ref_counts = reference_df[col].value_counts(normalize=True).sort_index()
            
            # Combine and reindex
            combined = pd.DataFrame({'Current': curr_counts, 'Reference': ref_counts})
            combined = combined.fillna(0)
            
            combined.plot(kind='bar', figsize=(12, 6))
            plt.title(f'Category Distribution: {col}')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Option 2: Implement automatic drift detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Monitor drift in production
def detect_drift(reference_data, new_data, columns_to_check, contamination=0.05):
    '''
    Detect drift using Isolation Forest
    Returns: Drift score (0-1), drifted samples indices
    '''
    # Prepare data
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference_data[columns_to_check])
    
    # Train isolation forest on reference data
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(reference_scaled)
    
    # Check if new data is anomalous compared to reference
    new_scaled = scaler.transform(new_data[columns_to_check])
    predictions = iso_forest.predict(new_scaled)
    
    # Get outlier indices (-1 represents outliers)
    drifted_indices = np.where(predictions == -1)[0]
    drift_score = len(drifted_indices) / len(new_data)
    
    return drift_score, drifted_indices

# Option 3: Implement corrective measures for drift
from sklearn.preprocessing import StandardScaler

def adapt_to_drift(reference_df, current_df, features_with_drift):
    '''Apply distribution matching to adjust for drift'''
    adapted_df = current_df.copy()
    
    for feature in features_with_drift:
        if pd.api.types.is_numeric_dtype(reference_df[feature]):
            # For numeric features: quantile-based matching
            ref_quantiles = np.quantile(reference_df[feature].dropna(), np.linspace(0, 1, 100))
            curr_quantiles = np.quantile(current_df[feature].dropna(), np.linspace(0, 1, 100))
            
            # Create correction function
            from scipy.interpolate import interp1d
            correction_fn = interp1d(curr_quantiles, ref_quantiles, 
                                     bounds_error=False, 
                                     fill_value=(ref_quantiles[0], ref_quantiles[-1]))
            
            # Apply correction
            adapted_df[feature] = adapted_df[feature].apply(
                lambda x: correction_fn(x) if pd.notna(x) else x
            )
    
    return adapted_df
"""
                ))
        
        # Memory usage recommendations
        try:
            # Check memory usage - import memory_profiler here to catch import error gracefully
            try:
                import memory_profiler
                MEMORY_PROFILER_AVAILABLE = True
            except ImportError:
                MEMORY_PROFILER_AVAILABLE = False
                
            if MEMORY_PROFILER_AVAILABLE and df.memory_usage(deep=True).sum() / (1024*1024) > 500:  # If dataset is > 500 MB
                high_mem_cols = []
                for col in df.columns:
                    col_size_mb = df[col].memory_usage(deep=True) / (1024*1024)
                    if col_size_mb > 50:  # If column takes > 50 MB
                        high_mem_cols.append((col, col_size_mb))
                
                if high_mem_cols:
                    high_mem_info = "\n".join([f"- {col}: {size:.2f} MB" for col, size in sorted(high_mem_cols, key=lambda x: x[1], reverse=True)])
                    
                    recommendations.append(DataRecommendation(
                        issue_type="quality",
                        severity="medium",
                        feature=None,
                        description="High memory usage detected",
                        impact="May cause performance issues or out-of-memory errors during processing",
                        recommendation="Optimize data types to reduce memory consumption",
                        code_example=f"""
# High memory usage columns:
{high_mem_info}

# Option 1: Optimize numeric column types
def optimize_numeric_types(df):
    '''Optimize memory by downcasting numeric columns to appropriate types'''
    df_optimized = df.copy()
    
    for col in df.select_dtypes(include=['int']).columns:
        # Find min and max values
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Convert to smallest possible integer type
        if col_min >= 0:  # Unsigned
            if col_max < 256:
                df_optimized[col] = df[col].astype(np.uint8)
            elif col_max < 65536:
                df_optimized[col] = df[col].astype(np.uint16)
            elif col_max < 4294967296:
                df_optimized[col] = df[col].astype(np.uint32)
            else:
                df_optimized[col] = df[col].astype(np.uint64)
        else:  # Signed
            if col_min > -128 and col_max < 128:
                df_optimized[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32768:
                df_optimized[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483648:
                df_optimized[col] = df[col].astype(np.int32)
            else:
                df_optimized[col] = df[col].astype(np.int64)
                
    # For float columns
    for col in df.select_dtypes(include=['float']).columns:
        df_optimized[col] = df[col].astype(np.float32)  # Downsample to 32-bit float
        
    return df_optimized

# Apply optimization
df_optimized = optimize_numeric_types(df)

# Compare memory usage
original_mem = df.memory_usage(deep=True).sum() / (1024*1024)
optimized_mem = df_optimized.memory_usage(deep=True).sum() / (1024*1024)
print(f"Original memory usage: {original_mem:.2f} MB")
print(f"Optimized memory usage: {optimized_mem:.2f} MB")
print(f"Memory savings: {original_mem - optimized_mem:.2f} MB ({100*(original_mem - optimized_mem)/original_mem:.1f}%)")

# Option 2: Category encoding for high-cardinality string columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < len(df) * 0.5:  # If cardinality is less than 50% of rows
        df[col] = df[col].astype('category')
"""
                    ))
        except Exception as e:
            logger.warning(f"Error generating memory usage recommendations: {e}")
        
        return recommendations


class DataGuardianDashboard:
    """Interactive dashboard for exploring DataGuardian analysis results"""
    
    def __init__(self, report: DataGuardianReport):
        """
        Initialize dashboard with a DataGuardian report
        
        Args:
            report: DataGuardianReport to visualize
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly and Dash are required for the dashboard")
            
        from dash import Dash, html, dcc, callback, Output, Input
        
        self.report = report
        self.app = Dash(__name__)
        
        # Set up dashboard layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Configure the dashboard layout"""
        from dash import html, dcc
        
        self.app.layout = html.Div([
            html.H1(f"DataGuardian Report: {self.report.dataset_name}"),
            html.P(f"Generated on: {self.report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"),
            
            dcc.Tabs([
                # Summary Tab
                dcc.Tab(label="Summary", children=[
                    html.Div([
                        html.H3("Dataset Overview"),
                        html.Div([
                            html.Div([
                                html.H4("Rows"),
                                html.P(f"{self.report.dataset_stats['num_rows']:,}")
                            ], className="stat-card"),
                            html.Div([
                                html.H4("Columns"),
                                html.P(f"{self.report.dataset_stats['num_columns']:,}")
                            ], className="stat-card"),
                            html.Div([
                                html.H4("Memory"),
                                html.P(f"{self.report.dataset_stats['memory_usage']:.2f} MB")
                            ], className="stat-card"),
                        ], className="stat-container"),
                        
                        html.Div([
                            html.H3("Quality Scores"),
                            dcc.Graph(id="quality-gauge")
                        ]),
                        
                        html.Div([
                            html.H3("Recommendations Summary"),
                            html.Div(id="recommendation-summary")
                        ])
                    ])
                ]),
                
                # Data Quality Tab
                dcc.Tab(label="Data Quality", children=[
                    html.Div([
                        html.H3("Data Quality Analysis"),
                        html.Div([
                            html.Div([
                                html.H4("Missing Values"),
                                dcc.Graph(id="missing-values-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("Outlier Analysis"),
                                dcc.Graph(id="outlier-chart")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                        
                        html.Div([
                            html.H4("Distribution Analysis"),
                            dcc.Dropdown(
                                id="distribution-feature-dropdown",
                                options=[{'label': col, 'value': col} for col in self.report.quality_metrics.distribution_metrics.keys()],
                                value=next(iter(self.report.quality_metrics.distribution_metrics.keys()), None),
                                style={'width': '50%', 'margin-bottom': '10px'}
                            ),
                            dcc.Graph(id="distribution-chart")
                        ]),
                        
                        html.Div([
                            html.H4("Correlation Matrix"),
                            dcc.Graph(id="correlation-matrix")
                        ])
                    ])
                ]),
                
                # Bias Analysis Tab
                dcc.Tab(label="Bias Analysis", children=[
                    html.Div([
                        html.H3("Bias & Fairness Analysis"),
                        html.Div([
                            html.Div([
                                html.H4("Statistical Parity Difference"),
                                dcc.Graph(id="bias-bar-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("Intersectional Bias"),
                                dcc.Graph(id="intersectional-bias-chart")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                        
                        html.Div([
                            html.H4("Feature Importance Bias"),
                            dcc.Graph(id="feature-importance-bias-chart")
                        ])
                    ])
                ]) if self.report.bias_metrics else None,
                
                # Privacy Tab
                dcc.Tab(label="Privacy Analysis", children=[
                    html.Div([
                        html.H3("Privacy Risk Assessment"),
                        html.Div([
                            html.Div([
                                html.H4("PII Detection"),
                                dcc.Graph(id="pii-detection-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("K-Anonymity Analysis"),
                                html.P(f"K-Anonymity: {self.report.privacy_metrics.k_anonymity}"),
                                html.P("K-anonymity represents the minimum number of records that share the same combination of quasi-identifiers. Higher values mean better anonymity.")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                    ])
                ]) if self.report.privacy_metrics else None,
                
                # Drift Analysis Tab
                dcc.Tab(label="Drift Analysis", children=[
                    html.Div([
                        html.H3("Distribution Drift Analysis"),
                        html.P(f"Overall Drift Score: {self.report.drift_metrics.overall_drift_score:.4f}"),
                        html.Div([
                            html.H4("Feature Drift"),
                            dcc.Graph(id="feature-drift-chart")
                        ])
                    ])
                ]) if self.report.drift_metrics else None,
                
                # Recommendations Tab
                dcc.Tab(label="Recommendations", children=[
                    html.Div([
                        html.H3("All Recommendations"),
                        html.Div([
                            html.Div([
                                html.H4("Filter by Type"),
                                dcc.Checklist(
                                    id="recommendation-type-filter",
                                    options=[
                                        {'label': 'Quality', 'value': 'quality'},
                                        {'label': 'Bias', 'value': 'bias'},
                                        {'label': 'Privacy', 'value': 'privacy'},
                                        {'label': 'Drift', 'value': 'drift'}
                                    ],
                                    value=['quality', 'bias', 'privacy', 'drift'],
                                    inline=True
                                )
                            ]),
                            html.Div([
                                html.H4("Filter by Severity"),
                                dcc.Checklist(
                                    id="recommendation-severity-filter",
                                    options=[
                                        {'label': 'Critical', 'value': 'critical'},
                                        {'label': 'High', 'value': 'high'},
                                        {'label': 'Medium', 'value': 'medium'},
                                        {'label': 'Low', 'value': 'low'}
                                    ],
                                    value=['critical', 'high', 'medium', 'low'],
                                    inline=True
                                )
                            ])
                        ]),
                        html.Div(id="filtered-recommendations")
                    ])
                ])
            ], id="dashboard-tabs")
        ])
    
    def _setup_callbacks(self):
        """Set up interactive callbacks for the dashboard"""
        from dash import callback, Output, Input
        import plotly.express as px
        import plotly.graph_objects as go
        from dash import html
        
        # Quality Gauge
        @self.app.callback(
            Output("quality-gauge", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_quality_gauge(_):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=self.report.quality_metrics.quality_score,
                title={'text': "Overall Quality Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.5], 'color': "red"},
                           {'range': [0.5, 0.7], 'color': "orange"},
                           {'range': [0.7, 0.9], 'color': "yellow"},
                           {'range': [0.9, 1], 'color': "green"}
                       ],
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': self.report.quality_metrics.quality_score
                       }}
            ))
            return fig
            
        # Missing Values Chart
        @self.app.callback(
            Output("missing-values-chart", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_missing_values_chart(_):
            missing_data = [(col, 1 - completeness) for col, completeness in 
                           self.report.quality_metrics.completeness.items() if completeness < 1]
            missing_data.sort(key=lambda x: x[1], reverse=True)
            
            cols = [x[0] for x in missing_data]
            values = [x[1] * 100 for x in missing_data]
            
            if not cols:  # No missing values
                return go.Figure().add_annotation(
                    text="No missing values detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            fig = go.Figure(go.Bar(
                x=cols, y=values,
                marker_color='indianred'
            ))
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Column",
                yaxis_title="Missing Percentage (%)",
                yaxis=dict(range=[0, 100])
            )
            return fig
            
        # Outlier Chart
        @self.app.callback(
            Output("outlier-chart", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_outlier_chart(_):
            outlier_data = [(col, len(indices) / self.report.dataset_stats['num_rows'] * 100) 
                          for col, indices in self.report.quality_metrics.outlier_scores.items()]
            outlier_data.sort(key=lambda x: x[1], reverse=True)
            
            cols = [x[0] for x in outlier_data]
            values = [x[1] for x in outlier_data]
            
            if not cols:  # No outliers
                return go.Figure().add_annotation(
                    text="No significant outliers detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            fig = go.Figure(go.Bar(
                x=cols, y=values,
                marker_color='orange'
            ))
            fig.update_layout(
                title="Outlier Percentage by Column",
                xaxis_title="Column",
                yaxis_title="Outlier Percentage (%)"
            )
            return fig
            
        # Distribution Chart
        @self.app.callback(
            Output("distribution-chart", "figure"),
            [Input("distribution-feature-dropdown", "value")]
        )
        def update_distribution_chart(feature):
            if not feature or feature not in self.report.quality_metrics.distribution_metrics:
                return go.Figure().add_annotation(
                    text="Select a feature to view distribution statistics",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            stats_dict = self.report.quality_metrics.distribution_metrics[feature]
            
            if 'skewness' in stats_dict:  # Numeric feature
                # Create bar chart for numeric stats
                labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis']
                values = [stats_dict['mean'], stats_dict['median'], stats_dict['std'], 
                         stats_dict['min'], stats_dict['max'], stats_dict['skewness'],
                         stats_dict.get('kurtosis', 0)]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels[:5],  # Only use the first 5 stats (exclude skew/kurt)
                    y=values[:5],
                    marker_color='cornflowerblue',
                    name="Basic Stats"
                ))
                
                # Add skewness and kurtosis on secondary y-axis
                fig.add_trace(go.Bar(
                    x=labels[5:],  # Only skew/kurt
                    y=values[5:],
                    marker_color='darkblue',
                    name="Distribution Shape"
                ))
                
                fig.update_layout(
                    title=f"Distribution Statistics for {feature}",
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                )
            else:  # Categorical feature
                # For categorical features, show entropy and top values
                labels = ['Entropy', 'Top Value %', 'Unique Count']
                values = [stats_dict['entropy'], stats_dict['top_value_pct'] * 100, stats_dict['unique_count']]
                
                fig = go.Figure(go.Bar(
                    x=labels,
                    y=values,
                    marker_color=['purple', 'pink', 'rebeccapurple']
                ))
                
                fig.update_layout(
                    title=f"Distribution Statistics for {feature} (Categorical)",
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                    annotations=[
                        dict(
                            x="Top Value %", 
                            y=values[1],
                            text=f"Top: {stats_dict['top_value']}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                    ]
                )
                
            return fig
            
        # Correlation Matrix
        @self.app.callback(
            Output("correlation-matrix", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_correlation_matrix(_):
            if self.report.quality_metrics.correlation_matrix is None:
                return go.Figure().add_annotation(
                    text="No correlation matrix available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            corr = self.report.quality_metrics.correlation_matrix
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size":10}
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=max(500, len(corr.columns) * 25),
                width=max(600, len(corr.columns) * 25)
            )
            return fig
        
        # Recommendation Summary
        @self.app.callback(
            Output("recommendation-summary", "children"),
            Input("dashboard-tabs", "value")
        )
        def update_recommendation_summary(_):
            # Count recommendations by type and severity
            by_type = {}
            by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            for rec in self.report.recommendations:
                by_type[rec.issue_type] = by_type.get(rec.issue_type, 0) + 1
                by_severity[rec.severity] = by_severity.get(rec.severity, 0) + 1
            
            # Create visual summary
            return html.Div([
                html.Div([
                    html.H4("By Issue Type"),
                    html.Div([
                        html.Div([
                            html.Div(f"{count}", className="count"),
                            html.Div(issue_type.title(), className="label")
                        ], className="count-box")
                        for issue_type, count in by_type.items()
                    ], className="count-container")
                ]),
                html.Div([
                    html.H4("By Severity"),
                    html.Div([
                        html.Div([
                            html.Div(f"{by_severity['critical']}", className="count critical"),
                            html.Div("Critical", className="label")
                        ], className="count-box"),
                        html.Div([
                            html.Div(f"{by_severity['high']}", className="count high"),
                            html.Div("High", className="label")
                        ], className="count-box"),
                        html.Div([
                            html.Div(f"{by_severity['medium']}", className# Create tokenized version
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[f'{col}_tokenized'] = encoder.fit_transform(df[[col]])
        # Drop original column
        df = df.drop(columns=[col])

# Option 3: Differential privacy (using diffprivlib)
# pip install diffprivlib
try:
    import diffprivlib as dp
    
    # Example for numeric columns
    # For each PII column that is numeric, apply differential privacy
    numeric_pii = [col for col in {pii_columns} if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_pii:
        # Apply Laplace noise to protect privacy
        dp_mean = dp.mean(df[col].dropna(), epsilon=1.0)
        dp_std = dp.std(df[col].dropna(), epsilon=1.0)
        # Create a perturbed version of the data
        laplace_mech = dp.mechanisms.Laplace(epsilon=1.0, sensitivity=(df[col].max() - df[col].min()))
        df[f'{col}_private'] = df[col].apply(lambda x: laplace_mech.randomise(x) if pd.notna(x) else None)
        # Drop original
        df = df.drop(columns=[col])
except ImportError:
    pass  # diffprivlib not available
"""
                ))
                
                # K-anonymity recommendations if k is low
                if privacy_metrics.k_anonymity < 5 and privacy_metrics.k_anonymity > 0:
                    recommendations.append(DataRecommendation(
                        issue_type="privacy",
                        severity="high",
                        feature=None,
                        description=f"Low k-anonymity (k={privacy_metrics.k_anonymity})",
                        impact="Individuals might be re-identifiable from quasi-identifiers",
                        recommendation="Apply generalization, suppression, or perturbation to increase anonymity",
                        code_example="""
# Increase k-anonymity through generalization
# Example: Binning age values to age groups

# Option 1: Binning numerical columns
def bin_column(df, column, bins, labels=None):
    """Create binned version of a numerical column"""
    if labels is None:
        labels = [f'Group {i+1}' for i in range(len(bins)-1)]
    df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

# Example: Age binning
if 'age' in df.columns:
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['0-18', '19-35', '36-50', '51-65', '65+']
    df = bin_column(df, 'age', bins, labels)
    df = df.drop(columns=['age'])  # Remove original column

# Option 2: Generalize categorical columns
def generalize_zipcode(df, column, digits_to_keep=3):
    """Generalize ZIP codes by keeping only first N digits"""
    df[f'{column}_generalized'] = df[column].astype(str).str[:digits_to_keep] + 'XX'
    return df

# Example: ZIP code generalization
if 'zipcode' in df.columns:
    df = generalize_zipcode(df, 'zipcode', 3)
    df = df.drop(columns=['zipcode'])  # Remove original
"""
                    ))
        
        # Drift recommendations
        if drift_metrics and drift_metrics.overall_drift_score > 0.1:
            # Find features with significant drift
            significant_drift = sorted(
                [(col, score) for col, score in drift_metrics.feature_drift.items() if score > 0.1],
                key=lambda x: x[1],
                reverse=True
            )
            
            if significant_drift:
                # Format drift info for display
                drift_info = "\n".join([f"- {col}: {score:.3f}" for col, score in significant_drift[:5]])
                if len(significant_drift) > 5:
                    drift_info += f"\n- And {len(significant_drift) - 5} more features..."
                
                recommendations.append(DataRecommendation(
                    issue_type="drift",
                    severity="high" if drift_metrics.overall_drift_score > 0.3 else "medium",
                    feature=None,
                    description=f"Significant data drift detected (score: {drift_metrics.overall_drift_score:.2f})",
                    impact="Model may perform poorly on new data due to distribution shifts",
                    recommendation="Consider model retraining, concept drift adaptation, or domain adaptation techniques",
                    code_example=f"""
# Significant drift detected in:
{drift_info}

# Option 1: Visualize distribution differences
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions for top drifted features
top_drift_features = {[col for col, _ in significant_drift[:3]]}

for col in top_drift_features:
    if col in df.columns:
        plt.figure(figsize=(10, 4))
        
        # Plot current vs reference distributions
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric features
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), color='blue', label='Current', alpha=0.6)
            sns.histplot(reference_df[col].dropna(), color='red', label='Reference', alpha=0.6)
            plt.title(f'Distribution Comparison: {col}')
            plt.legend()
            
            # QQ-plot
            plt.subplot(1, 2, 2)
            from scipy import stats
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title("Q-Q Plot")
        else:
            # For categorical features
            curr_counts = df[col].value_counts(normalize=True).sort_index()
            ref_counts = reference_df[col].value_counts(normalize=True).sort_index()
            
            # Combine and reindex
            combined = pd.DataFrame({'Current': curr_counts, 'Reference': ref_counts})
            combined = combined.fillna(0)
            
            combined.plot(kind='bar', figsize=(12, 6))
            plt.title(f'Category Distribution: {col}')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Option 2: Implement automatic drift detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Monitor drift in production
def detect_drift(reference_data, new_data, columns_to_check, contamination=0.05):
    '''
    Detect drift using Isolation Forest
    Returns: Drift score (0-1), drifted samples indices
    '''
    # Prepare data
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference_data[columns_to_check])
    
    # Train isolation forest on reference data
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(reference_scaled)
    
    # Check if new data is anomalous compared to reference
    new_scaled = scaler.transform(new_data[columns_to_check])
    predictions = iso_forest.predict(new_scaled)
    
    # Get outlier indices (-1 represents outliers)
    drifted_indices = np.where(predictions == -1)[0]
    drift_score = len(drifted_indices) / len(new_data)
    
    return drift_score, drifted_indices

# Option 3: Implement corrective measures for drift
from sklearn.preprocessing import StandardScaler

def adapt_to_drift(reference_df, current_df, features_with_drift):
    '''Apply distribution matching to adjust for drift'''
    adapted_df = current_df.copy()
    
    for feature in features_with_drift:
        if pd.api.types.is_numeric_dtype(reference_df[feature]):
            # For numeric features: quantile-based matching
            ref_quantiles = np.quantile(reference_df[feature].dropna(), np.linspace(0, 1, 100))
            curr_quantiles = np.quantile(current_df[feature].dropna(), np.linspace(0, 1, 100))
            
            # Create correction function
            from scipy.interpolate import interp1d
            correction_fn = interp1d(curr_quantiles, ref_quantiles, 
                                     bounds_error=False, 
                                     fill_value=(ref_quantiles[0], ref_quantiles[-1]))
            
            # Apply correction
            adapted_df[feature] = adapted_df[feature].apply(
                lambda x: correction_fn(x) if pd.notna(x) else x
            )
    
    return adapted_df
"""
                ))
        
        # Memory usage recommendations
        try:
            # Check memory usage - import memory_profiler here to catch import error gracefully
            try:
                import memory_profiler
                MEMORY_PROFILER_AVAILABLE = True
            except ImportError:
                MEMORY_PROFILER_AVAILABLE = False
                
            if MEMORY_PROFILER_AVAILABLE and df.memory_usage(deep=True).sum() / (1024*1024) > 500:  # If dataset is > 500 MB
                high_mem_cols = []
                for col in df.columns:
                    col_size_mb = df[col].memory_usage(deep=True) / (1024*1024)
                    if col_size_mb > 50:  # If column takes > 50 MB
                        high_mem_cols.append((col, col_size_mb))
                
                if high_mem_cols:
                    high_mem_info = "\n".join([f"- {col}: {size:.2f} MB" for col, size in sorted(high_mem_cols, key=lambda x: x[1], reverse=True)])
                    
                    recommendations.append(DataRecommendation(
                        issue_type="quality",
                        severity="medium",
                        feature=None,
                        description="High memory usage detected",
                        impact="May cause performance issues or out-of-memory errors during processing",
                        recommendation="Optimize data types to reduce memory consumption",
                        code_example=f"""
# High memory usage columns:
{high_mem_info}

# Option 1: Optimize numeric column types
def optimize_numeric_types(df):
    '''Optimize memory by downcasting numeric columns to appropriate types'''
    df_optimized = df.copy()
    
    for col in df.select_dtypes(include=['int']).columns:
        # Find min and max values
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Convert to smallest possible integer type
        if col_min >= 0:  # Unsigned
            if col_max < 256:
                df_optimized[col] = df[col].astype(np.uint8)
            elif col_max < 65536:
                df_optimized[col] = df[col].astype(np.uint16)
            elif col_max < 4294967296:
                df_optimized[col] = df[col].astype(np.uint32)
            else:
                df_optimized[col] = df[col].astype(np.uint64)
        else:  # Signed
            if col_min > -128 and col_max < 128:
                df_optimized[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32768:
                df_optimized[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483648:
                df_optimized[col] = df[col].astype(np.int32)
            else:
                df_optimized[col] = df[col].astype(np.int64)
                
    # For float columns
    for col in df.select_dtypes(include=['float']).columns:
        df_optimized[col] = df[col].astype(np.float32)  # Downsample to 32-bit float
        
    return df_optimized

# Apply optimization
df_optimized = optimize_numeric_types(df)

# Compare memory usage
original_mem = df.memory_usage(deep=True).sum() / (1024*1024)
optimized_mem = df_optimized.memory_usage(deep=True).sum() / (1024*1024)
print(f"Original memory usage: {original_mem:.2f} MB")
print(f"Optimized memory usage: {optimized_mem:.2f} MB")
print(f"Memory savings: {original_mem - optimized_mem:.2f} MB ({100*(original_mem - optimized_mem)/original_mem:.1f}%)")

# Option 2: Category encoding for high-cardinality string columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < len(df) * 0.5:  # If cardinality is less than 50% of rows
        df[col] = df[col].astype('category')
"""
                    ))
        except Exception as e:
            logger.warning(f"Error generating memory usage recommendations: {e}")
        
        return recommendations


class DataGuardianDashboard:
    """Interactive dashboard for exploring DataGuardian analysis results"""
    
    def __init__(self, report: DataGuardianReport):
        """
        Initialize dashboard with a DataGuardian report
        
        Args:
            report: DataGuardianReport to visualize
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly and Dash are required for the dashboard")
            
        from dash import Dash, html, dcc, callback, Output, Input
        
        self.report = report
        self.app = Dash(__name__)
        
        # Set up dashboard layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Configure the dashboard layout"""
        from dash import html, dcc
        
        self.app.layout = html.Div([
            html.H1(f"DataGuardian Report: {self.report.dataset_name}"),
            html.P(f"Generated on: {self.report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"),
            
            dcc.Tabs([
                # Summary Tab
                dcc.Tab(label="Summary", children=[
                    html.Div([
                        html.H3("Dataset Overview"),
                        html.Div([
                            html.Div([
                                html.H4("Rows"),
                                html.P(f"{self.report.dataset_stats['num_rows']:,}")
                            ], className="stat-card"),
                            html.Div([
                                html.H4("Columns"),
                                html.P(f"{self.report.dataset_stats['num_columns']:,}")
                            ], className="stat-card"),
                            html.Div([
                                html.H4("Memory"),
                                html.P(f"{self.report.dataset_stats['memory_usage']:.2f} MB")
                            ], className="stat-card"),
                        ], className="stat-container"),
                        
                        html.Div([
                            html.H3("Quality Scores"),
                            dcc.Graph(id="quality-gauge")
                        ]),
                        
                        html.Div([
                            html.H3("Recommendations Summary"),
                            html.Div(id="recommendation-summary")
                        ])
                    ])
                ]),
                
                # Data Quality Tab
                dcc.Tab(label="Data Quality", children=[
                    html.Div([
                        html.H3("Data Quality Analysis"),
                        html.Div([
                            html.Div([
                                html.H4("Missing Values"),
                                dcc.Graph(id="missing-values-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("Outlier Analysis"),
                                dcc.Graph(id="outlier-chart")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                        
                        html.Div([
                            html.H4("Distribution Analysis"),
                            dcc.Dropdown(
                                id="distribution-feature-dropdown",
                                options=[{'label': col, 'value': col} for col in self.report.quality_metrics.distribution_metrics.keys()],
                                value=next(iter(self.report.quality_metrics.distribution_metrics.keys()), None),
                                style={'width': '50%', 'margin-bottom': '10px'}
                            ),
                            dcc.Graph(id="distribution-chart")
                        ]),
                        
                        html.Div([
                            html.H4("Correlation Matrix"),
                            dcc.Graph(id="correlation-matrix")
                        ])
                    ])
                ]),
                
                # Bias Analysis Tab
                dcc.Tab(label="Bias Analysis", children=[
                    html.Div([
                        html.H3("Bias & Fairness Analysis"),
                        html.Div([
                            html.Div([
                                html.H4("Statistical Parity Difference"),
                                dcc.Graph(id="bias-bar-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("Intersectional Bias"),
                                dcc.Graph(id="intersectional-bias-chart")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                        
                        html.Div([
                            html.H4("Feature Importance Bias"),
                            dcc.Graph(id="feature-importance-bias-chart")
                        ])
                    ])
                ]) if self.report.bias_metrics else None,
                
                # Privacy Tab
                dcc.Tab(label="Privacy Analysis", children=[
                    html.Div([
                        html.H3("Privacy Risk Assessment"),
                        html.Div([
                            html.Div([
                                html.H4("PII Detection"),
                                dcc.Graph(id="pii-detection-chart")
                            ], style={'width': '48%'}),
                            html.Div([
                                html.H4("K-Anonymity Analysis"),
                                html.P(f"K-Anonymity: {self.report.privacy_metrics.k_anonymity}"),
                                html.P("K-anonymity represents the minimum number of records that share the same combination of quasi-identifiers. Higher values mean better anonymity.")
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                    ])
                ]) if self.report.privacy_metrics else None,
                
                # Drift Analysis Tab
                dcc.Tab(label="Drift Analysis", children=[
                    html.Div([
                        html.H3("Distribution Drift Analysis"),
                        html.P(f"Overall Drift Score: {self.report.drift_metrics.overall_drift_score:.4f}"),
                        html.Div([
                            html.H4("Feature Drift"),
                            dcc.Graph(id="feature-drift-chart")
                        ])
                    ])
                ]) if self.report.drift_metrics else None,
                
                # Recommendations Tab
                dcc.Tab(label="Recommendations", children=[
                    html.Div([
                        html.H3("All Recommendations"),
                        html.Div([
                            html.Div([
                                html.H4("Filter by Type"),
                                dcc.Checklist(
                                    id="recommendation-type-filter",
                                    options=[
                                        {'label': 'Quality', 'value': 'quality'},
                                        {'label': 'Bias', 'value': 'bias'},
                                        {'label': 'Privacy', 'value': 'privacy'},
                                        {'label': 'Drift', 'value': 'drift'}
                                    ],
                                    value=['quality', 'bias', 'privacy', 'drift'],
                                    inline=True
                                )
                            ]),
                            html.Div([
                                html.H4("Filter by Severity"),
                                dcc.Checklist(
                                    id="recommendation-severity-filter",
                                    options=[
                                        {'label': 'Critical', 'value': 'critical'},
                                        {'label': 'High', 'value': 'high'},
                                        {'label': 'Medium', 'value': 'medium'},
                                        {'label': 'Low', 'value': 'low'}
                                    ],
                                    value=['critical', 'high', 'medium', 'low'],
                                    inline=True
                                )
                            ])
                        ]),
                        html.Div(id="filtered-recommendations")
                    ])
                ])
            ], id="dashboard-tabs")
        ])
    
    def _setup_callbacks(self):
        """Set up interactive callbacks for the dashboard"""
        from dash import callback, Output, Input
        import plotly.express as px
        import plotly.graph_objects as go
        from dash import html
        
        # Quality Gauge
        @self.app.callback(
            Output("quality-gauge", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_quality_gauge(_):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=self.report.quality_metrics.quality_score,
                title={'text': "Overall Quality Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.5], 'color': "red"},
                           {'range': [0.5, 0.7], 'color': "orange"},
                           {'range': [0.7, 0.9], 'color': "yellow"},
                           {'range': [0.9, 1], 'color': "green"}
                       ],
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': self.report.quality_metrics.quality_score
                       }}
            ))
            return fig
            
        # Missing Values Chart
        @self.app.callback(
            Output("missing-values-chart", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_missing_values_chart(_):
            missing_data = [(col, 1 - completeness) for col, completeness in 
                           self.report.quality_metrics.completeness.items() if completeness < 1]
            missing_data.sort(key=lambda x: x[1], reverse=True)
            
            cols = [x[0] for x in missing_data]
            values = [x[1] * 100 for x in missing_data]
            
            if not cols:  # No missing values
                return go.Figure().add_annotation(
                    text="No missing values detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            fig = go.Figure(go.Bar(
                x=cols, y=values,
                marker_color='indianred'
            ))
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Column",
                yaxis_title="Missing Percentage (%)",
                yaxis=dict(range=[0, 100])
            )
            return fig
            
        # Outlier Chart
        @self.app.callback(
            Output("outlier-chart", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_outlier_chart(_):
            outlier_data = [(col, len(indices) / self.report.dataset_stats['num_rows'] * 100) 
                          for col, indices in self.report.quality_metrics.outlier_scores.items()]
            outlier_data.sort(key=lambda x: x[1], reverse=True)
            
            cols = [x[0] for x in outlier_data]
            values = [x[1] for x in outlier_data]
            
            if not cols:  # No outliers
                return go.Figure().add_annotation(
                    text="No significant outliers detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            fig = go.Figure(go.Bar(
                x=cols, y=values,
                marker_color='orange'
            ))
            fig.update_layout(
                title="Outlier Percentage by Column",
                xaxis_title="Column",
                yaxis_title="Outlier Percentage (%)"
            )
            return fig
            
        # Distribution Chart
        @self.app.callback(
            Output("distribution-chart", "figure"),
            [Input("distribution-feature-dropdown", "value")]
        )
        def update_distribution_chart(feature):
            if not feature or feature not in self.report.quality_metrics.distribution_metrics:
                return go.Figure().add_annotation(
                    text="Select a feature to view distribution statistics",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            stats_dict = self.report.quality_metrics.distribution_metrics[feature]
            
            if 'skewness' in stats_dict:  # Numeric feature
                # Create bar chart for numeric stats
                labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis']
                values = [stats_dict['mean'], stats_dict['median'], stats_dict['std'], 
                         stats_dict['min'], stats_dict['max'], stats_dict['skewness'],
                         stats_dict.get('kurtosis', 0)]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels[:5],  # Only use the first 5 stats (exclude skew/kurt)
                    y=values[:5],
                    marker_color='cornflowerblue',
                    name="Basic Stats"
                ))
                
                # Add skewness and kurtosis on secondary y-axis
                fig.add_trace(go.Bar(
                    x=labels[5:],  # Only skew/kurt
                    y=values[5:],
                    marker_color='darkblue',
                    name="Distribution Shape"
                ))
                
                fig.update_layout(
                    title=f"Distribution Statistics for {feature}",
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                )
            else:  # Categorical feature
                # For categorical features, show entropy and top values
                labels = ['Entropy', 'Top Value %', 'Unique Count']
                values = [stats_dict['entropy'], stats_dict['top_value_pct'] * 100, stats_dict['unique_count']]
                
                fig = go.Figure(go.Bar(
                    x=labels,
                    y=values,
                    marker_color=['purple', 'pink', 'rebeccapurple']
                ))
                
                fig.update_layout(
                    title=f"Distribution Statistics for {feature} (Categorical)",
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                    annotations=[
                        dict(
                            x="Top Value %", 
                            y=values[1],
                            text=f"Top: {stats_dict['top_value']}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                    ]
                )
                
            return fig
            
        # Correlation Matrix
        @self.app.callback(
            Output("correlation-matrix", "figure"),
            Input("dashboard-tabs", "value")
        )
        def update_correlation_matrix(_):
            if self.report.quality_metrics.correlation_matrix is None:
                return go.Figure().add_annotation(
                    text="No correlation matrix available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            corr = self.report.quality_metrics.correlation_matrix
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size":10}
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=max(500, len(corr.columns) * 25),
                width=max(600, len(corr.columns) * 25)
            )
            return fig
        
        # Recommendation Summary
        @self.app.callback(
            Output("recommendation-summary", "children"),
            Input("dashboard-tabs", "value")
        )
        def update_recommendation_summary(_):
            # Count recommendations by type and severity
            by_type = {}
            by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            for rec in self.report.recommendations:
                by_type[rec.issue_type] = by_type.get(rec.issue_type, 0) + 1
                by_severity[rec.severity] = by_severity.get(rec.severity, 0) + 1
            
            # Create visual summary
            return html.Div([
                html.Div([
                    html.H4("By Issue Type"),
                    html.Div([
                        html.Div([
                            html.Div(f"{count}", className="count"),
                            html.Div(issue_type.title(), className="label")
                        ], className="count-box")
                        for issue_type, count in by_type.items()
                    ], className="count-container")
                ]),
                html.Div([
                    html.H4("By Severity"),
                    html.Div([
                        html.Div([
                            html.Div(f"{by_severity['critical']}", className="count critical"),
                            html.Div("Critical", className="label")
                        ], className="count-box"),
                        html.Div([
                            html.Div(f"{by_severity['high']}", className="count high"),
                            html.Div("High", className="label")
                        ], className="count-box"),
                        html.Div([
                            html.Div(f"{by_severity['medium']}", className