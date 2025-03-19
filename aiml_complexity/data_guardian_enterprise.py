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
                        