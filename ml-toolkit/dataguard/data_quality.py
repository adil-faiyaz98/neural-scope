import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import re
from scipy import stats
from sklearn.ensemble import IsolationForest

class DataQualityChecker:
    """Comprehensive data quality checker with privacy, bias, and integrity checks."""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                 protected_attributes: Optional[List[str]] = None):
        """
        Initialize the data quality checker.
        
        Args:
            df: Input DataFrame to analyze
            target_column: Target variable column name for ML tasks
            protected_attributes: List of protected/sensitive attributes for bias/privacy analysis
        """
        self.df = df
        self.target_column = target_column
        self.protected_attributes = protected_attributes or []
        
        # Identify column types
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Define PII patterns
        self.pii_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[0-9]{10,15}$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'credit_card': r'^\d{4}-?\d{4}-?\d{4}-?\d{4}$',
            'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        }
        
        self.report = {}
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run all data quality checks and return comprehensive results."""
        
        self.report = {
            "dataset_info": self._get_dataset_info(),
            "data_integrity": self._check_data_integrity(),
            "quality_score": self._calculate_quality_score(),
            "recommendations": self._generate_recommendations()
        }
        
        # Run specialized analyses if applicable
        if self.target_column:
            self.report["leakage_detection"] = self._detect_potential_leakage()
            self.report["class_balance"] = self._analyze_class_balance()
        
        if self.protected_attributes:
            self.report["bias_assessment"] = self._assess_bias()
            self.report["privacy_risk"] = self._assess_privacy_risk()
        
        # Additional analyses
        self.report["storage_efficiency"] = self._analyze_storage_efficiency()
        
        return self.report
    
    def _get_dataset_info(self) -> Dict:
        """Extract basic dataset information."""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "size_bytes": self.df.memory_usage(deep=True).sum(),
            "size_mb": self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            "column_types": {
                "numeric": len(self.numeric_cols),
                "categorical": len(self.categorical_cols),
                "datetime": len(self.datetime_cols),
                "other": len(self.df.columns) - len(self.numeric_cols) - len(self.categorical_cols) - len(self.datetime_cols)
            },
            "column_list": {
                "numeric": self.numeric_cols,
                "categorical": self.categorical_cols,
                "datetime": self.datetime_cols
            }
        }
    
    def _check_data_integrity(self) -> Dict:
        """Check for data integrity issues like missing values, duplicates, outliers."""
        integrity = {}
        
        # Analyze missing values
        missing_percentages = self.df.isna().mean().to_dict()
        missing_columns = {col: pct for col, pct in missing_percentages.items() if pct > 0}
        
        # Analyze missing patterns
        integrity["missing_patterns"] = {
            "total_missing_values": self.df.isna().sum().sum(),
            "missing_percentages": missing_columns,
            "columns_with_missing": len(missing_columns),
            "missing_pattern_matrix": self._create_missing_pattern_matrix()
        }
        
        # Check for duplicates and data corruption
        integrity["duplicates_and_corruption"] = {
            "exact_duplicates": self.df.duplicated().sum(),
            "potential_duplicates": self._detect_potential_duplicates(),
            "corrupted_values": self._detect_corrupted_values()
        }
        
        # Detect outliers in numerical columns
        integrity["outliers"] = {
            "statistical_outliers": self._detect_statistical_outliers(),
            "isolation_forest_outliers": self._detect_isolation_forest_outliers()
        }
        
        # Check for inconsistencies in categorical data
        integrity["inconsistencies"] = self._detect_inconsistencies()
        
        return integrity
    
    def _create_missing_pattern_matrix(self) -> Dict:
        """Create a matrix of missing value patterns."""
        if len(self.df) > 10000:
            # Sample for large datasets
            sample_df = self.df.sample(n=10000, random_state=42)
        else:
            sample_df = self.df
            
        # Create missing pattern matrix
        pattern_df = sample_df.isna().astype(int)
        patterns = {}
        
        for i, pattern in enumerate(pattern_df.value_counts().head(10).index):
            patterns[f"pattern_{i}"] = {
                "columns": [col for col, missing in zip(pattern_df.columns, pattern) if missing == 1],
                "count": int(pattern_df.value_counts().iloc[i]),
                "percentage": float(pattern_df.value_counts().iloc[i] / len(pattern_df) * 100)
            }
            
        return patterns
    
    def _detect_potential_duplicates(self) -> Dict:
        """Detect potential duplicates that might not be exact matches."""
        potential_dups = {}
        
        if len(self.categorical_cols) > 0:
            # Check for duplicates in key categorical columns
            subset = [col for col in self.categorical_cols if self.df[col].nunique() / len(self.df) < 0.5][:5]
            if subset:
                potential_dups["categorical_duplicates"] = int(self.df.duplicated(subset=subset).sum())
        
        return potential_dups
    
    def _detect_corrupted_values(self) -> Dict:
        """Detect potentially corrupted values in the dataset."""
        corrupted = {}
        
        # Check numeric columns for implausible values
        for col in self.numeric_cols:
            # Check for values far outside the expected range
            q1 = self.df[col].quantile(0.01)
            q99 = self.df[col].quantile(0.99)
            iqr = q99 - q1
            lower_bound = q1 - (iqr * 10)
            upper_bound = q99 + (iqr * 10)
            
            extreme_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if extreme_count > 0:
                corrupted[col] = {
                    "extreme_values": int(extreme_count),
                    "percentage": float(extreme_count / len(self.df) * 100)
                }
        
        # Check categorical columns for likely errors (very rare values)
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            rare_values = value_counts[value_counts == 1].count()
            
            if rare_values > 0 and rare_values / len(self.df) < 0.001:
                corrupted[col] = {
                    "rare_singleton_values": int(rare_values),
                    "percentage": float(rare_values / len(self.df) * 100)
                }
        
        return corrupted
    
    def _detect_statistical_outliers(self) -> Dict:
        """Detect outliers using statistical methods (Z-score, IQR)."""
        outliers = {}
        
        for col in self.numeric_cols:
            if self.df[col].isna().sum() / len(self.df) > 0.2:
                continue  # Skip columns with many missing values
                
            # Use IQR method to detect outliers
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": float(outlier_count / len(self.df) * 100),
                    "method": "IQR",
                    "bounds": {
                        "lower": float(lower_bound),
                        "upper": float(upper_bound)
                    }
                }
        
        return outliers
    
    def _detect_isolation_forest_outliers(self) -> Dict:
        """Detect outliers using Isolation Forest algorithm."""
        if len(self.numeric_cols) < 2 or len(self.df) < 100:
            return {}
            
        try:
            # Prepare numerical data, handling missing values
            numeric_df = self.df[self.numeric_cols].copy()
            for col in numeric_df.columns:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
            
            # Train Isolation Forest model
            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(numeric_df)
            
            # Count outliers (-1 is outlier)
            outlier_count = (predictions == -1).sum()
            
            return {
                "total_outliers": int(outlier_count),
                "percentage": float(outlier_count / len(self.df) * 100),
                "method": "Isolation Forest"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_inconsistencies(self) -> Dict:
        """Detect inconsistencies in categorical data."""
        inconsistencies = {}
        
        for col in self.categorical_cols:
            if self.df[col].nunique() < 2:
                continue
                
            # Check for inconsistent capitalization and spacing
            value_counts = self.df[col].astype(str).str.lower().value_counts()
            cleaned_value_counts = self.df[col].astype(str).str.lower().str.strip().value_counts()
            
            if len(value_counts) != len(cleaned_value_counts):
                inconsistencies[col] = {
                    "original_unique_values": int(self.df[col].nunique()),
                    "cleaned_unique_values": int(cleaned_value_counts.shape[0]),
                    "inconsistency_type": "capitalization_or_spacing"
                }
                
            # Check for potential typos in categorical values
            if self.df[col].nunique() > 3 and self.df[col].nunique() < 20:
                # Look for values that are very similar based on string similarity
                values = self.df[col].astype(str).unique()
                similar_groups = []
                
                # Simple check for very similar values (case insensitive)
                groups = {}
                for value in values:
                    v_lower = value.lower().strip()
                    if len(v_lower) >= 4:  # Only check non-trivial strings
                        found = False
                        for group_key in list(groups.keys()):
                            if v_lower[0:3] == group_key[0:3] and (
                                v_lower in group_key or group_key in v_lower or 
                                sum(1 for a, b in zip(v_lower, group_key) if a != b) <= 2
                            ):
                                groups[group_key].append(value)
                                found = True
                                break
                        
                        if not found:
                            groups[v_lower] = [value]
                
                # Filter only groups with potential inconsistencies
                similar_groups = [v for v in groups.values() if len(v) > 1]
                if similar_groups:
                    inconsistencies[col] = {
                        "inconsistency_type": "potential_typos",
                        "similar_groups": similar_groups
                    }
        
        return inconsistencies
    
    def _calculate_quality_score(self) -> Dict[str, float]:
        """Calculate a comprehensive data quality score."""
        scores = {}
        
        # Calculate completeness score (based on missing values)
        completeness = 100 - (self.df.isna().mean().mean() * 100)
        scores["completeness"] = completeness
        
        # Calculate uniqueness score (penalize duplicate rows)
        duplicate_pct = self.df.duplicated().mean() * 100
        scores["uniqueness"] = 100 - duplicate_pct
        
        # Calculate consistency score (based on detected inconsistencies)
        inconsistency_count = len(self._detect_inconsistencies())
        consistency_score = 100 - min(100, inconsistency_count / len(self.df.columns) * 100)
        scores["consistency"] = consistency_score
        
        # Calculate validity score (based on detected corrupted values)
        corrupted = self._detect_corrupted_values()
        invalid_count = sum(info.get("extreme_values", 0) for col, info in corrupted.items())
        validity_score = 100 - min(100, invalid_count / len(self.df) * 100)
        scores["validity"] = validity_score
        
        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.4,
            "uniqueness": 0.2,
            "consistency": 0.2,
            "validity": 0.2
        }
        
        overall_score = sum(scores[key] * weight for key, weight in weights.items())
        scores["overall"] = overall_score
        
        return scores
    
    def _detect_potential_leakage(self) -> Dict:
        """Detect potential data leakage issues."""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
            
        leakage_info = {}
        
        # Check for columns highly correlated with target
        if self.target_column in self.numeric_cols:
            corrs = {}
            for col in self.numeric_cols:
                if col != self.target_column:
                    corr = self.df[[col, self.target_column]].corr().iloc[0, 1]
                    if not pd.isna(corr) and abs(corr) > 0.9:
                        corrs[col] = float(corr)
            
            if corrs:
                leakage_info["high_correlation_features"] = corrs
        
        # Check if target can be predicted perfectly from other columns
        # This is a simplistic check - in practice would use ML model
        if self.target_column in self.categorical_cols:
            # Check if any categorical column perfectly predicts the target
            for col in self.categorical_cols:
                if col != self.target_column:
                    # Group by the column and check if target is always the same in each group
                    grouped = self.df.groupby(col)[self.target_column].nunique()
                    if (grouped == 1).all():
                        if "perfect_predictors" not in leakage_info:
                            leakage_info["perfect_predictors"] = []
                        leakage_info["perfect_predictors"].append(col)
        
        # Check for duplicate rows with different target values
        if len(self.df) > 1:
            features = [col for col in self.df.columns if col != self.target_column]
            dup_mask = self.df.duplicated(subset=features, keep=False)
            if dup_mask.sum() > 0:
                dups = self.df[dup_mask]
                inconsistent_targets = dups.groupby(features)[self.target_column].nunique() > 1
                if inconsistent_targets.any():
                    leakage_info["inconsistent_target_values"] = {
                        "count": int(inconsistent_targets.sum()),
                        "percentage": float(inconsistent_targets.sum() / len(self.df) * 100)
                    }
        
        return leakage_info
    
    def _analyze_class_balance(self) -> Dict:
        """Analyze class balance for the target variable."""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
            
        balance_info = {}
        
        # Only applicable for classification tasks
        if self.target_column in self.categorical_cols:
            # Calculate class distribution
            class_counts = self.df[self.target_column].value_counts()
            class_distribution = {}
            
            for