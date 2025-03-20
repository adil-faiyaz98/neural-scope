import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from imblearn.over_sampling import SMOTE
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class DataGuardianEnhancer:
    """
    Enhanced data quality analyzer that extends DataGuardian with advanced 
    data integrity, imbalance, diversity, and leakage detection capabilities.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize with dataframe and optional target column.
        
        Args:
            df: Input dataframe to analyze
            target_column: Target/label column name if available
        """
        self.df = df
        self.target_column = target_column
        self.numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        self.report = {
            "data_integrity": {},
            "class_balance": {},
            "feature_analysis": {},
            "leakage_detection": {},
            "storage_efficiency": {},
            "recommendations": []
        }
        
    def analyze_missing_patterns(self) -> Dict:
        """
        Advanced missing data pattern analysis to detect MAR, MCAR, or MNAR patterns.
        """
        results = {
            "missing_counts": {},
            "missing_percentages": {},
            "pattern_analysis": {},
            "correlation_matrix": None,
            "likely_pattern": "Unknown"
        }
        
        # Calculate missing counts and percentages
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = 100 * missing_count / len(self.df)
            results["missing_counts"][col] = missing_count
            results["missing_percentages"][col] = round(missing_pct, 2)
        
        # Create missing value indicators for correlation analysis
        missing_indicators = pd.DataFrame()
        for col in self.df.columns:
            if self.df[col].isna().any():
                missing_indicators[f"{col}_missing"] = self.df[col].isna().astype(int)
        
        # If we have missing indicators, analyze their correlations
        if not missing_indicators.empty:
            corr_matrix = missing_indicators.corr()
            results["correlation_matrix"] = corr_matrix
            
            # Check for strong correlations between missing indicators
            high_correlations = False
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        high_correlations = True
                        col1 = corr_matrix.columns[i].replace("_missing", "")
                        col2 = corr_matrix.columns[j].replace("_missing", "")
                        results["pattern_analysis"][f"{col1},{col2}"] = corr_matrix.iloc[i, j]
            
            # Now check if missingness correlates with observed values in other columns
            mar_evidence = {}
            for missing_col in missing_indicators.columns:
                orig_col = missing_col.replace("_missing", "")
                for data_col in self.numeric_cols:
                    if data_col != orig_col:
                        # Calculate correlation between missingness and other column values
                        valid_idx = ~self.df[data_col].isna()
                        if valid_idx.sum() > 10:  # Need enough data points
                            corr = np.corrcoef(
                                missing_indicators.loc[valid_idx, missing_col], 
                                self.df.loc[valid_idx, data_col]
                            )[0, 1]
                            if abs(corr) > 0.3:  # Threshold for significance
                                mar_evidence[f"{orig_col},{data_col}"] = round(corr, 3)
            
            # Determine likely missing pattern
            if high_correlations or mar_evidence:
                results["likely_pattern"] = "MAR (Missing At Random)"
                results["pattern_analysis"]["MAR_evidence"] = mar_evidence
            else:
                # Check if missing data percentage is very low
                if all(pct < 5 for pct in results["missing_percentages"].values()):
                    results["likely_pattern"] = "MCAR (Missing Completely At Random)"
                else:
                    results["likely_pattern"] = "Possibly MNAR (Missing Not At Random)"
        
        # Generate imputation recommendations
        results["imputation_recommendations"] = self._recommend_imputation(results)
        
        self.report["data_integrity"]["missing_patterns"] = results
        return results
    
    def _recommend_imputation(self, missing_results: Dict) -> Dict:
        """Generate imputation recommendations based on missing patterns."""
        recommendations = {}
        
        for col, pct in missing_results["missing_percentages"].items():
            if pct == 0:
                continue
                
            if pct > 80:
                recommendations[col] = "Consider dropping this column due to excessive missing values"
            elif pct > 40:
                recommendations[col] = "High missingness - consider using a missing indicator plus imputation"
            else:
                if col in self.numeric_cols:
                    # Check distribution before recommending mean/median
                    try:
                        skew = self.df[col].dropna().skew()
                        if abs(skew) > 1:
                            recommendations[col] = "Use median imputation due to skewed distribution"
                        else:
                            recommendations[col] = "Use mean imputation or KNN imputation"
                    except:
                        recommendations[col] = "Use median imputation"
                else:
                    recommendations[col] = "Use mode imputation or consider adding a 'Missing' category"
        
        # If MAR pattern detected, suggest more advanced methods
        if missing_results["likely_pattern"] == "MAR (Missing At Random)":
            recommendations["overall"] = "Consider using MICE or MissForest for imputation since MAR pattern detected"
        elif missing_results["likely_pattern"] == "MCAR (Missing Completely At Random)":
            recommendations["overall"] = "Simple imputation methods like mean/median/mode should work well"
        else:
            recommendations["overall"] = "Exercise caution with imputation as MNAR pattern may introduce bias"
            
        return recommendations
    
    def analyze_class_imbalance(self) -> Dict:
        """
        Analyze class imbalance and generate rebalancing strategies.
        """
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "No target column specified for imbalance analysis"}
            
        results = {
            "class_distribution": {},
            "imbalance_ratio": 0,
            "is_imbalanced": False,
            "minority_classes": [],
            "majority_classes": [],
            "rebalancing_recommendations": {}
        }
        
        # Calculate class distribution
        target = self.df[self.target_column]
        class_counts = target.value_counts()
        total_samples = len(target)
        
        for cls, count in class_counts.items():
            results["class_distribution"][str(cls)] = {
                "count": int(count),
                "percentage": round(100 * count / total_samples, 2)
            }
        
        # Calculate imbalance ratio (majority / minority)
        if len(class_counts) > 1:
            max_count = class_counts.max()
            min_count = class_counts.min()
            results["imbalance_ratio"] = round(max_count / min_count, 2)
            
            # Determine if dataset is imbalanced
            if results["imbalance_ratio"] > 3:
                results["is_imbalanced"] = True
                
                # Identify minority and majority classes
                minority_threshold = total_samples / (len(class_counts) * 3)
                majority_threshold = total_samples / len(class_counts) * 1.5
                
                for cls, count in class_counts.items():
                    if count < minority_threshold:
                        results["minority_classes"].append(str(cls))
                    elif count > majority_threshold:
                        results["majority_classes"].append(str(cls))
                
                # Generate rebalancing recommendations
                if len(results["minority_classes"]) > 0:
                    # Check if dataset is large enough for undersampling
                    if max_count > 1000:
                        results["rebalancing_recommendations"]["undersampling"] = {
                            "technique": "Random Undersampling",
                            "rationale": "Dataset has sufficient majority samples to undersample",
                            "implementation": "from imblearn.under_sampling import RandomUnderSampler"
                        }
                    
                    # Always suggest SMOTE for minority classes
                    results["rebalancing_recommendations"]["oversampling"] = {
                        "technique": "SMOTE",
                        "rationale": "Synthesize new minority samples to balance classes",
                        "implementation": "from imblearn.over_sampling import SMOTE"
                    }
                    
                    # Suggest class weights for certain algorithms
                    class_weights = {}
                    for cls in class_counts.index:
                        class_weights[str(cls)] = max_count / class_counts[cls]
                    
                    results["rebalancing_recommendations"]["class_weights"] = {
                        "technique": "Class weighting",
                        "rationale": "Adjust algorithm to penalize mistakes on minority classes more heavily",
                        "weights": class_weights,
                        "implementation": "model.fit(X, y, class_weight='balanced')"
                    }
                    
                    # Check if extreme imbalance for two-phase approach
                    if results["imbalance_ratio"] > 50:
                        results["rebalancing_recommendations"]["two_phase"] = {
                            "technique": "Two-phase learning",
                            "rationale": "Extreme imbalance detected. First train on balanced subset, then fine-tune.",
                            "implementation": "# 1. Train initial model on balanced data\n# 2. Fine-tune on full dataset with appropriate class weights"
                        }
        
        self.report["class_balance"] = results
        return results
    
    def detect_data_leakage(self) -> Dict:
        """
        Detect potential data leakage issues and target leakage.
        """
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "No target column specified for leakage detection"}
        
        results = {
            "high_correlation_features": [],
            "unique_identifiers": [],
            "temporal_leakage_risk": "Unknown",
            "recommendations": []
        }
        
        # 1. Detect features with suspiciously high correlation with target
        if self.target_column in self.numeric_cols:
            # For numeric target, use correlation
            for col in self.numeric_cols:
                if col != self.target_column:
                    corr = self.df[[col, self.target_column]].corr().iloc[0, 1]
                    if abs(corr) > 0.95:
                        results["high_correlation_features"].append({
                            "feature": col,
                            "correlation": round(abs(corr), 3),
                            "risk": "Very High"
                        })
                    elif abs(corr) > 0.9:
                        results["high_correlation_features"].append({
                            "feature": col,
                            "correlation": round(abs(corr), 3),
                            "risk": "High"
                        })
        else:
            # For categorical target, use mutual information
            target_encoded = LabelEncoder().fit_transform(self.df[self.target_column])
            for col in self.df.columns:
                if col != self.target_column:
                    try:
                        if col in self.numeric_cols:
                            # Bin numeric values for MI calculation
                            binned = pd.qcut(self.df[col], q=10, duplicates='drop', labels=False)
                            mi = mutual_info_score(binned.fillna(0), target_encoded)
                        else:
                            # For categorical columns
                            col_encoded = LabelEncoder().fit_transform(self.df[col].fillna('missing'))
                            mi = mutual_info_score(col_encoded, target_encoded)
                            
                        # Normalize MI by entropy of target
                        target_entropy = stats.entropy(np.bincount(target_encoded) / len(target_encoded))
                        if target_entropy > 0:
                            normalized_mi = mi / target_entropy
                            
                            if normalized_mi > 0.9:
                                results["high_correlation_features"].append({
                                    "feature": col,
                                    "mutual_information": round(normalized_mi, 3),
                                    "risk": "Very High"
                                })
                            elif normalized_mi > 0.7:
                                results["high_correlation_features"].append({
                                    "feature": col,
                                    "mutual_information": round(normalized_mi, 3),
                                    "risk": "High"
                                })
                    except:
                        pass  # Skip if MI calculation fails
        
        # 2. Detect unique identifiers that shouldn't be used as features
        for col in self.df.columns:
            if col != self.target_column:
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio > 0.9 and len(self.df) > 100:
                    results["unique_identifiers"].append({
                        "feature": col,
                        "unique_ratio": round(unique_ratio, 3),
                        "recommendation": "Consider removing this near-unique identifier to prevent overfitting"
                    })
        
        # 3. Check for datetime columns that might cause temporal leakage
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
                    
        if date_cols:
            results["temporal_columns"] = date_cols
            results["temporal_leakage_risk"] = "Potential"
            results["recommendations"].append(
                "Ensure proper time-based train-test splits to prevent future data leakage"
            )
                
        # Generate recommendations based on findings
        if results["high_correlation_features"]:
            results["recommendations"].append(
                "Investigate suspiciously high-correlation features as they may indicate target leakage"
            )
            for feat in results["high_correlation_features"]:
                if feat["risk"] == "Very High":
                    results["recommendations"].append(
                        f"Consider removing {feat['feature']} as it appears to leak target information"
                    )
        
        self.report["leakage_detection"] = results
        return results
    
    def analyze_storage_efficiency(self) -> Dict:
        """
        Analyze dataset storage efficiency and recommend optimizations.
        """
        results = {
            "current_memory_usage": {},
            "optimized_memory_usage": {},
            "total_current_mb": 0,
            "total_optimized_mb": 0,
            "savings_percentage": 0,
            "format_recommendations": {},
            "loading_benchmark": {}
        }
        
        # Calculate current memory usage
        current_memory = {}
        total_current = 0
        
        for col in self.df.columns:
            col_size = self.df[col].memory_usage(deep=True) / (1024 * 1024)  # MB
            current_memory[col] = round(col_size, 3)
            total_current += col_size
            
        results["current_memory_usage"] = current_memory
        results["total_current_mb"] = round(total_current, 3)
        
        # Calculate optimized memory usage with appropriate dtypes
        df_optimized = self.df.copy()
        optimized_memory = {}
        total_optimized = 0
        
        for col in df_optimized.columns:
            # Integer optimization
            if col in self.numeric_cols:
                if df_optimized[col].isna().any():
                    # If column has NaNs, can't use Int64
                    pass
                else:
                    col_data = df_optimized[col]
                    col_min = col_data.min()
                    col_max = col_data.max()
                    
                    # Check if values are integers
                    if np.array_equal(col_data, col_data.astype(int)):
                        if col_min >= 0:
                            if col_max < 2**8:
                                df_optimized[col] = df_optimized[col].astype(np.uint8)
                            elif col_max < 2**16:
                                df_optimized[col] = df_optimized[col].astype(np.uint16)
                            elif col_max < 2**32:
                                df_optimized[col] = df_optimized[col].astype(np.uint32)
                        else:
                            if col_min > -2**7 and col_max < 2**7:
                                df_optimized[col] = df_optimized[col].astype(np.int8)
                            elif col_min > -2**15 and col_max < 2**15:
                                df_optimized[col] = df_optimized[col].astype(np.int16)
                            elif col_min > -2**31 and col_max < 2**31:
                                df_optimized[col] = df_optimized[col].astype(np.int32)
                    
            # Category optimization for object types
            elif col in self.categorical_cols:
                if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Only if reasonably low cardinality
                    df_optimized[col] = df_optimized[col].astype('category')
            
            # Calculate optimized size
            col_size = df_optimized[col].memory_usage(deep=True) / (1024 * 1024)  # MB
            optimized_memory[col] = round(col_size, 3)
            total_optimized += col_size
        
        results["optimized_memory_usage"] = optimized_memory
        results["total_optimized_mb"] = round(total_optimized, 3)
        
        # Calculate potential savings
        if total_current > 0:
            savings = (total_current - total_optimized) / total_current * 100
            results["savings_percentage"] = round(savings, 2)
        
        # Benchmark different storage formats
        # We'll just estimate rather than actually writing files
        csv_est_size = total_current * 0.8  # Rough estimate
        parquet_est_size = total_optimized * 0.4  # Rough compression estimate
        feather_est_size = total_optimized * 0.6
        
        # Loading time estimates (relative)
        # These are rough estimates based on typical performance ratios
        results["format_recommendations"] = {
            "current": {
                "format": "In-memory pandas",
                "size_mb": round(total_current, 2),
                "relative_load_time": 1.0
            },
            "optimized_pandas": {
                "format": "Optimized pandas dtypes",
                "size_mb": round(total_optimized, 2),
                "relative_load_time": 0.9,
                "savings_pct": round(savings, 2)
            },
            "csv": {
                "format": "CSV",
                "size_mb": round(csv_est_size, 2),
                "relative_load_time": 3.0,
                "notes": "Slow loading, no compression, not recommended for large datasets"
            },
            "parquet": {
                "format": "Parquet",
                "size_mb": round(parquet_est_size, 2),
                "relative_load_time": 0.5,
                "notes": "Fast loading, good compression, columnar format ideal for analytics"
            },
            "feather": {
                "format": "Feather",
                "size_mb": round(feather_est_size, 2),
                "relative_load_time": 0.3,
                "notes": "Very fast loading, moderate compression, good for temporary storage"
            }
        }
        
        # Generate recommendations based on dataset size
        if total_current > 1000:  # If dataset is over 1GB
            results["format_recommendations"]["recommendation"] = "Use Parquet format with optimized dtypes for best balance of speed and size"
            results["format_recommendations"]["code_example"] = "df.to_parquet('dataset.parquet', compression='snappy')"
        elif total_current > 100:  # If dataset is over 100MB
            results["format_recommendations"]["recommendation"] = "Use Feather format for fastest loading or Parquet for better compression"
            results["format_recommendations"]["code_example"] = "df.to_feather('dataset.feather')"
        else:
            results["format_recommendations"]["recommendation"] = "Current format is adequate, but consider optimizing dtypes"
            results["format_recommendations"]["code_example"] = "# Optimize numeric columns\nfor col in df.select_dtypes('int64').columns:\n    df[col] = pd.to_numeric(df[col], downcast='integer')"
        
        self.report["storage_efficiency"] = results
        return results
    
    def analyze_feature_relevance(self) -> Dict:
        """
        Analyze feature relevance and identify potentially useless or redundant features.
        """
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "No target column specified for feature relevance analysis"}
            
        results = {
            "constant_features": [],
            "quasi_constant_features": [],
            "duplicate_features": [],
            "correlated_feature_groups": [],
            "low_importance_features": [],
            "recommendations": []
        }
        
        # Check for constant or near-constant features
        for col in self.df.columns:
            if col == self.target_column:
                continue
                
            unique_vals = self.df[col].nunique(dropna=False)
            if unique_vals == 1:
                results["constant_features"].append(col)
            elif unique_vals == 2:
                # Check if one value dominates (>95%)
                value_counts = self.df[col].value_counts(normalize=True, dropna=False)
                if value_counts.iloc[0] > 0.95:
                    results["quasi_constant_features"].append({
                        "feature": col,
                        "dominant_value": value_counts.index[0],
                        "frequency": round(value_counts.iloc[0] * 100, 2)
                    })
        
        # Check for duplicate features
        if len(self.df.columns) > 1:
            # Create a dictionary of duplicated columns
            duplicate_cols = {}
            
            # For numeric columns, check for duplicates
            numeric_df = self.df[self.numeric_cols].copy()
            numeric_df = numeric_df.fillna(numeric_df.mean())
            
            # Use the column correlation to find duplicates
            corr = numeric_df.corr().abs()
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            
            # Find column pairs with correlation > 0.97 (nearly duplicates)
            high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j]) 
                               for i, j in zip(*np.where(upper_tri > 0.97))]
            
            # Group correlated features
            if high_corr_pairs:
                # Build groups of correlated features
                corr_groups = []
                added_cols = set()
                
                for col1, col2, corr_val in high_corr_pairs:
                    # Skip if already added to a group
                    if col1 in added_cols and col2 in added_cols:
                        continue
                    
                    # Check if either column is in an existing group
                    found = False
                    for group in corr_groups:
                        if col1 in group['features'] or col2 in group['features']:
                            if col1 not in group['features']:
                                group['features'].append(col1)
                            if col2 not in group['features']:
                                group['features'].append(col2)
                            group['correlation_matrix'].append((col1, col2, round(corr_val, 3)))
                            found = True
                            break
                    
                    # If not found in any group, create a new group
                    if not found:
                        corr_groups.append({
                            'features': [col1, col2],
                            'correlation_matrix': [(col1, col2, round(corr_val, 3))]
                        })
                    
                    added_cols.add(col1)
                    added_cols.add(col2)
                
                results["correlated_feature_groups"] = corr_groups
        
        # Make recommendations
        if results["constant_features"]:
            results["recommendations"].append(
                f"Remove constant features: {', '.join(results['constant_features'])}"
            )
            
        if results["quasi_constant_features"]:
            quasi_features = [f["feature"] for f in results["quasi_constant_features"]]
            results["recommendations"].append(
                f"Consider removing quasi-constant features: {', '.join(quasi_features)}"
            )
            
        if results["correlated_feature_groups"]:
            for group in results["correlated_feature_groups"]:
                features = group["features"]
                results["recommendations"].append(
                    f"Highly correlated features detected: {', '.join(features)}. Consider keeping only one."
                )
        
        self.report["feature_analysis"] = results
        return results
    
    def analyze_stratification_fairness(self, sensitive_features: List[str]) -> Dict:
        """
        Analyze dataset representativeness across sensitive attributes.
        
        Args:
            sensitive_features: List of column names representing sensitive attributes
        """
        results = {
            "representation_metrics": {},
            "label_distribution": {},
            "bias_metrics": {},
            "recommendations": []
        }
        
        # Validate sensitive features
        valid_sensitive = [f for f in sensitive_features if f in self.df.columns]
        if not valid_sensitive:
            return {"error": "No valid sensitive features found in dataset"}
            
        # Check representation of each sensitive feature
        for feature in valid_sensitive:
            value_counts = self.df[feature].value_counts(normalize=True, dropna=False)
            results["representation_metrics"][feature] = {
                str(val): round(pct * 100, 2) for val, pct in value_counts.items()
            }
            
            # Check if representation is highly skewed
            max_pct = value_counts.max() * 100
            min_pct = value_counts.min() * 100
            
            if max_pct > 80 and len(value_counts) > 1:
                results["recommendations"].append(
                    f"Feature '{feature}' is highly skewed with {max_pct:.1f}% from a single group. "
                    f"Consider obtaining more diverse data for underrepresented groups."
                )
            
                # If target column exists, check label distribution per group
                if self.target_column and self.target_column in self.df.columns:
                    label_dist = {}
                    bias_scores = {}
                    overall_dist = self.df[self.target_column].value_counts(normalize=True)
                    
                    for group_val in self.df[feature].unique():
                        if pd.notna(group_val):
                            group_df = self.df[self.df[feature] == group_val]
                            group_dist = group_df[self.target_column].value_counts(normalize=True)
                            
                            # Store distribution
                            label_dist[str(group_val)] = {
                                str(label): round(pct * 100, 2) for label, pct in group_dist.items()
                            }
                            
                            # Calculate statistical parity difference
                            if self.target_column in self.categorical_cols:
                                # For classification: calculate disparate impact
                                for label in overall_dist.index:
                                    if label in group_dist:
                                        ratio = group_dist[label] / overall_dist[label]
                                        if "disparate_impact" not in bias_scores:
                                            bias_scores["disparate_impact"] = {}
                                        if str(label) not in bias_scores["disparate_impact"]:
                                            bias_scores["disparate_impact"][str(label)] = {}
                                        bias_scores["disparate_impact"][str(label)][str(group_val)] = round(ratio, 3)
                                        
                                        # Flag significant disparities
                                        if ratio < 0.8 or ratio > 1.25:  # Standard thresholds for disparate impact
                                            results["recommendations"].append(
                                                f"Potential bias detected: Group '{group_val}' in feature '{feature}' "
                                                f"has {round(abs(1-ratio)*100, 1)}% {'under' if ratio < 1 else 'over'}-representation "
                                                f"for label '{label}'"
                                            )
                            else:
                                # For regression: calculate mean difference
                                if group_df[self.target_column].count() > 10:  # Ensure enough data points
                                    overall_mean = self.df[self.target_column].mean()
                                    group_mean = group_df[self.target_column].mean()
                                    diff_pct = (group_mean - overall_mean) / overall_mean * 100
                                    
                                    if "mean_difference" not in bias_scores:
                                        bias_scores["mean_difference"] = {}
                                    bias_scores["mean_difference"][str(group_val)] = round(diff_pct, 2)
                                    
                                    # Flag significant differences
                                    if abs(diff_pct) > 20:  # Arbitrary threshold, adjust as needed
                                        results["recommendations"].append(
                                            f"Potential bias detected: Group '{group_val}' in feature '{feature}' "
                                            f"has {abs(round(diff_pct, 1))}% {'lower' if diff_pct < 0 else 'higher'} "
                                            f"average target value compared to overall population"
                                        )
                    
                    results["label_distribution"][feature] = label_dist
                    if bias_scores:
                        results["bias_metrics"][feature] = bias_scores
        
        # Recommendations for fairness
        if results["recommendations"]:
            # Add general recommendation for bias mitigation
            results["recommendations"].append(
                "Consider using fairness-aware algorithms or preprocessing methods like reweighting or "
                "fair representation learning to address potential biases in the dataset."
            )
            
            # Add specific mitigation code example
            results["bias_mitigation_code"] = """
# Example using AIF360 for bias mitigation
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

# Convert your dataframe to AIF360 format
sensitive_features = ['gender', 'race']  # Replace with your sensitive attributes
df_aif = BinaryLabelDataset(
    df=df, 
    label_names=[target_column],
    protected_attribute_names=sensitive_features
)

# Apply Disparate Impact Remover
di_remover = DisparateImpactRemover(repair_level=0.8)
df_transformed = di_remover.fit_transform(df_aif)

# Convert back to pandas DataFrame
df_fair = df_transformed.convert_to_dataframe()[0]
"""
        
        self.report["fairness_analysis"] = results
        return results
                
    def detect_duplicates_and_corruption(self) -> Dict:
        """
        Perform advanced duplicate detection and data corruption analysis.
        """
        results = {
            "exact_duplicates": 0,
            "near_duplicates": {},
            "corrupted_values": {},
            "recommendations": []
        }
        
        # Check for exact duplicates
        exact_duplicates = self.df.duplicated().sum()
        results["exact_duplicates"] = int(exact_duplicates)
        
        if exact_duplicates > 0:
            dup_pct = exact_duplicates / len(self.df) * 100
            results["recommendations"].append(
                f"Found {exact_duplicates} exact duplicate rows ({dup_pct:.2f}% of data). "
                f"Consider removing duplicates with df.drop_duplicates()."
            )
            
            # Show a sample of duplicated rows
            dup_sample = self.df[self.df.duplicated(keep=False)].head(5).to_dict('records')
            results["duplicate_examples"] = dup_sample
            
        # Check for near-duplicates (fuzzy matching on string columns)
        # For practical purposes, we'll limit this to shorter text columns
        # and sample rows to avoid excessive computation
        
        text_cols = [col for col in self.categorical_cols 
                    if self.df[col].dropna().astype(str).str.len().mean() < 50  # Avoid very long text
                    and self.df[col].nunique() > 5  # Avoid pure categorical
                    and self.df[col].nunique() / len(self.df) < 0.9]  # Avoid unique identifiers
                    
        if text_cols and len(self.df) < 50000:  # Only for reasonably sized datasets
            try:
                # Sample rows to check for near-duplicates
                sample_size = min(5000, len(self.df))
                sample_df = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
                
                # For simplicity, we'll just check a few columns for similar values
                from fuzzywuzzy import fuzz
                
                near_dups = {}
                for col in text_cols[:3]:  # Limit to first 3 text columns
                    # Get string values
                    str_values = sample_df[col].dropna().astype(str).tolist()
                    
                    if len(str_values) > 100:
                        str_values = random.sample(str_values, 100)  # Further limit for performance
                    
                    # Find similar strings
                    similar_pairs = []
                    for i in range(len(str_values)):
                        for j in range(i+1, len(str_values)):
                            similarity = fuzz.ratio(str_values[i], str_values[j])
                            if similarity > 80 and similarity < 100:  # High similarity but not identical
                                similar_pairs.append({
                                    'value1': str_values[i], 
                                    'value2': str_values[j], 
                                    'similarity': similarity
                                })
                                
                    # Keep top 5 most similar pairs
                    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                    similar_pairs = similar_pairs[:5]
                    
                    if similar_pairs:
                        near_dups[col] = similar_pairs
                
                if near_dups:
                    results["near_duplicates"] = near_dups
                    results["recommendations"].append(
                        "Found near-duplicate values in text columns. Consider text normalization "
                        "or fuzzy matching to standardize similar values."
                    )
            except ImportError:
                # If fuzzywuzzy is not installed
                results["near_duplicates"]["error"] = "Could not perform fuzzy matching. Install fuzzywuzzy for this feature."
                
        # Check for potentially corrupted values
        corrupt_values = {}
        
        # Check numeric columns for implausible values
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Get statistics
            q1, q3 = col_data.quantile([0.01, 0.99])  # Use 1% and 99% to be more conservative
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 3)
            upper_bound = q3 + (iqr * 3)
            
            # Find extreme outliers (potentially corrupted)
            extreme_low = col_data[col_data < lower_bound]
            extreme_high = col_data[col_data > upper_bound]
            
            if not extreme_low.empty or not extreme_high.empty:
                corrupt_values[col] = {
                    "extreme_low_count": len(extreme_low),
                    "extreme_high_count": len(extreme_high),
                    "normal_range": f"{lower_bound:.2f} - {upper_bound:.2f}",
                    "min_value": col_data.min(),
                    "max_value": col_data.max()
                }
                
                # Only flag as corruption if truly extreme
                if len(extreme_low) > 0 and extreme_low.min() < lower_bound * 10:
                    results["recommendations"].append(
                        f"Column '{col}' has {len(extreme_low)} potentially corrupted low values "
                        f"(min: {extreme_low.min():.2f}, expected > {lower_bound:.2f})"
                    )
                
                if len(extreme_high) > 0 and extreme_high.max() > upper_bound * 10:
                    results["recommendations"].append(
                        f"Column '{col}' has {len(extreme_high)} potentially corrupted high values "
                        f"(max: {extreme_high.max():.2f}, expected < {upper_bound:.2f})"
                    )
        
        # Check categorical columns for suspicious patterns (e.g., malformed emails)
        for col in self.categorical_cols:
            suspicious_count = 0
            sample_suspicious = []
            
            # Check if column appears to contain emails
            if 'email' in col.lower():
                values = self.df[col].dropna().astype(str)
                # Simple regex for valid email format
                valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = values[~values.str.match(valid_email_pattern)]
                
                if not invalid_emails.empty:
                    suspicious_count = len(invalid_emails)
                    sample_suspicious = invalid_emails.head(3).tolist()
                    
                    results["recommendations"].append(
                        f"Column '{col}' has {suspicious_count} ({suspicious_count/len(values)*100:.1f}%) "
                        f"potentially malformed email addresses. Consider validation and cleaning."
                    )
            
            # Check if column appears to contain dates but with invalid formats
            elif 'date' in col.lower() or 'time' in col.lower():
                values = self.df[col].dropna().astype(str)
                try:
                    pd.to_datetime(values, errors='raise')
                except:
                    # If conversion fails, try to identify which values are problematic
                    valid_mask = pd.Series(True, index=values.index)
                    for i, val in values.iteritems():
                        try:
                            pd.to_datetime(val)
                        except:
                            valid_mask.loc[i] = False
                    
                    invalid_dates = values[~valid_mask]
                    if not invalid_dates.empty:
                        suspicious_count = len(invalid_dates)
                        sample_suspicious = invalid_dates.head(3).tolist()
                        
                        results["recommendations"].append(
                            f"Column '{col}' has {suspicious_count} ({suspicious_count/len(values)*100:.1f}%) "
                            f"potentially invalid date/time values. Consider standardizing date formats."
                        )
            
            if suspicious_count > 0:
                corrupt_values[col] = {
                    "suspicious_count": suspicious_count,
                    "suspicious_examples": sample_suspicious,
                    "recommendation": "Check for data entry errors or format inconsistencies"
                }
        
        results["corrupted_values"] = corrupt_values
        
        self.report["data_integrity"]["duplicates_and_corruption"] = results
        return results
    
    def detect_outliers(self, contamination=0.05) -> Dict:
        """
        Detect outliers using multiple methods and recommend actions.
        
        Args:
            contamination: Expected proportion of outliers (default 0.05 = 5%)
        """
        results = {
            "statistical_outliers": {},
            "isolation_forest_outliers": {},
            "lof_outliers": {},
            "combined_outliers": {},
            "recommendations": []
        }
        
        # Only process if we have numeric features
        if not self.numeric_cols:
            results["error"] = "No numeric columns available for outlier detection"
            return results
            
        # Prepare numeric data for outlier detection
        numeric_df = self.df[self.numeric_cols].copy()
        
        # Handle missing values for outlier detection
        numeric_df_filled = numeric_df.fillna(numeric_df.mean())
        
        # Statistical outlier detection (Z-score)
        stat_outliers = {}
        for col in self.numeric_cols:
            col_data = numeric_df[col].dropna()
            if len(col_data) < 10:  # Skip columns with too few values
                continue
                
            # Use robust statistics to avoid outliers influencing the thresholds
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))  # Median Absolute Deviation
            
            if mad == 0:  # Avoid division by zero
                continue
                
            # Modified Z-score method (more robust than standard Z-score)
            modified_z = 0.6745 * (col_data - median) / mad
            outliers_idx = np.abs(modified_z) > 3.5  # Common threshold for modified Z-score
            
            outliers = col_data[outliers_idx]
            if len(outliers) > 0:
                stat_outliers[col] = {
                    "count": len(outliers),
                    "percentage": round(len(outliers) / len(col_data) * 100, 2),
                    "min_value": float(outliers.min()),
                    "max_value": float(outliers.max()),
                    "normal_range": f"{float(median - 3.5 * mad / 0.6745):.2f} - {float(median + 3.5 * mad / 0.6745):.2f}"
                }
        
        results["statistical_outliers"] = stat_outliers
        
        # Machine learning based outlier detection
        if len(numeric_df_filled) > 20:  # Only if we have enough samples
            try:
                # Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_forest.fit(numeric_df_filled)
                outlier_pred = iso_forest.predict(numeric_df_filled)
                outlier_idx = outlier_pred == -1  # -1 indicates outliers
                
                if outlier_idx.sum() > 0:
                    outlier_pct = outlier_idx.sum() / len(outlier_idx) * 100
                    results["isolation_forest_outliers"] = {
                        "count": int(outlier_idx.sum()),
                        "percentage": round(outlier_pct, 2),
                        "feature_importance": {}
                    }
                    
                    # Try to identify which features contribute most to outlier detection
                    if hasattr(iso_forest, 'feature_importances_'):
                        feature_importances = iso_forest.feature_importances_
                        for i, col in enumerate(numeric_df_filled.columns):
                            results["isolation_forest_outliers"]["feature_importance"][col] = round(feature_importances[i], 3)
                
                # Local Outlier Factor (for density-based outlier detection)
                if len(numeric_df_filled) < 100000:  # LOF can be slow on very large datasets
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                    lof_pred = lof.fit_predict(numeric_df_filled)
                    lof_idx = lof_pred == -1  # -1 indicates outliers
                    
                    if lof_idx.sum() > 0:
                        lof_pct = lof_idx.sum() / len(lof_idx) * 100
                        results["lof_outliers"] = {
                            "count": int(lof_idx.sum()),
                            "percentage": round(lof_pct, 2)
                        }
                        
                # Find rows that are outliers according to multiple methods
                if "isolation_forest_outliers" in results and "lof_outliers" in results:
                    combined_idx = outlier_idx & lof_idx
                    if combined_idx.sum() > 0:
                        results["combined_outliers"] = {
                            "count": int(combined_idx.sum()),
                            "percentage": round(combined_idx.sum() / len(combined_idx) * 100, 2)
                        }
            
            except Exception as e:
                results["ml_outlier_detection_error"] = str(e)
        
        # Generate recommendations
        if stat_outliers:
            columns_with_many_outliers = [col for col, info in stat_outliers.items() if info["percentage"] > 5]
            if columns_with_many_outliers:
                results["recommendations"].append(
                    f"Columns with high outlier percentage: {', '.join(columns_with_many_outliers)}. "
                    f"Consider scaling (e.g., log transform) or winsorizing extreme values."
                )
                
                results["code_example_winsorization"] = """
# Winsorize outliers (cap at percentiles)
from scipy.stats import mstats
for col in ['column1', 'column2']:
    df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])  # Cap at 1st and 99th percentiles

# Or using pandas:
for col in ['column1', 'column2']:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
"""
        
        if "isolation_forest_outliers" in results and results["isolation_forest_outliers"].get("percentage", 0) > 5:
            results["recommendations"].append(
                "Consider investigating or removing multivariate outliers detected by Isolation Forest, "
                "especially for sensitive models like fraud detection or anomaly detection."
            )
            
            results["code_example_outlier_removal"] = """
# Identify and remove outliers using Isolation Forest
from sklearn.ensemble import IsolationForest

# Fit the model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso_forest.fit_predict(df[numeric_columns])

# Create a clean dataset without outliers
df_clean = df[outlier_pred == 1].copy()  # Keep only inliers
print(f"Removed {(outlier_pred == -1).sum()} outliers ({(outlier_pred == -1).sum()/len(outlier_pred)*100:.1f}%)")
"""
        
        self.report["data_integrity"]["outliers"] = results
        return results
    
                # If target column exists, check label distribution per group
                if self.target_column and self.target_column in self.df.columns:
                    label_dist = {}
                    bias_scores = {}
                    overall_dist = self.df[self.target_column].value_counts(normalize=True)
                    
                    for group_val in self.df[feature].unique():
                        if pd.notna(group_val):
                            group_df = self.df[self.df[feature] == group_val]
                            group_dist = group_df[self.target_column].value_counts(normalize=True)
                            
                            # Store distribution
                            label_dist[str(group_val)] = {
                                str(label): round(pct * 100, 2) for label, pct in group_dist.items()
                            }
                            
                            # Calculate statistical parity difference
                            if self.target_column in self.categorical_cols:
                                # For classification: calculate disparate impact
                                for label in overall_dist.index:
                                    if label in group_dist:
                                        ratio = group_dist[label] / overall_dist[label]
                                        if "disparate_impact" not in bias_scores:
                                            bias_scores["disparate_impact"] = {}
                                        if str(label) not in bias_scores["disparate_impact"]:
                                            bias_scores["disparate_impact"][str(label)] = {}
                                        bias_scores["disparate_impact"][str(label)][str(group_val)] = round(ratio, 3)
                                        
                                        # Flag significant disparities
                                        if ratio < 0.8 or ratio > 1.25:  # Standard thresholds for disparate impact
                                            results["recommendations"].append(
                                                f"Potential bias detected: Group '{group_val}' in feature '{feature}' "
                                                f"has {round(abs(1-ratio)*100, 1)}% {'under' if ratio < 1 else 'over'}-representation "
                                                f"for label '{label}'"
                                            )
                            else:
                                # For regression: calculate mean difference
                                if group_df[self.target_column].count() > 10:  # Ensure enough data points
                                    overall_mean = self.df[self.target_column].mean()
                                    group_mean = group_df[self.target_column].mean()
                                    diff_pct = (group_mean - overall_mean) / overall_mean * 100
                                    
                                    if "mean_difference" not in bias_scores:
                                        bias_scores["mean_difference"] = {}
                                    bias_scores["mean_difference"][str(group_val)] = round(diff_pct, 2)
                                    
                                    # Flag significant differences
                                    if abs(diff_pct) > 20:  # Arbitrary threshold, adjust as needed
                                        results["recommendations"].append(
                                            f"Potential bias detected: Group '{group_val}' in feature '{feature}' "
                                            f"has {abs(round(diff_pct, 1))}% {'lower' if diff_pct < 0 else 'higher'} "
                                            f"average target value compared to overall population"
                                        )
                    
                    results["label_distribution"][feature] = label_dist
                    if bias_scores:
                        results["bias_metrics"][feature] = bias_scores
        
        # Recommendations for fairness
        if results["recommendations"]:
            # Add general recommendation for bias mitigation
            results["recommendations"].append(
                "Consider using fairness-aware algorithms or preprocessing methods like reweighting or "
                "fair representation learning to address potential biases in the dataset."
            )
            
            # Add specific mitigation code example
            results["bias_mitigation_code"] = """
# Example using AIF360 for bias mitigation
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

# Convert your dataframe to AIF360 format
sensitive_features = ['gender', 'race']  # Replace with your sensitive attributes
df_aif = BinaryLabelDataset(
    df=df, 
    label_names=[target_column],
    protected_attribute_names=sensitive_features
)

# Apply Disparate Impact Remover
di_remover = DisparateImpactRemover(repair_level=0.8)
df_transformed = di_remover.fit_transform(df_aif)

# Convert back to pandas DataFrame
df_fair = df_transformed.convert_to_dataframe()[0]
"""
        
        self.report["fairness_analysis"] = results
        return results
                
    def detect_duplicates_and_corruption(self) -> Dict:
        """
        """
        results = {
            "exact_duplicates": 0,
            "near_duplicates": {},
            "corrupted_values": {},
            "recommendations": []
        }
        
        # Check for exact duplicates
        exact_duplicates = self.df.duplicated().sum()
        results["exact_duplicates"] = int(exact_duplicates)
        
        if exact_duplicates > 0:
            dup_pct = exact_duplicates / len(self.df) * 100
            results["recommendations"].append(
                f"Found {exact_duplicates} exact duplicate rows ({dup_pct:.2f}% of data). "
                f"Consider removing duplicates with df.drop_duplicates()."
            )
            
            # Show a sample of duplicated rows
            dup_sample = self.df[self.df.duplicated(keep=False)].head(5).to_dict('records')
            results["duplicate_examples"] = dup_sample
            
        # Check for near-duplicates (fuzzy matching on string columns)
        # For practical purposes, we'll limit this to shorter text columns
        # and sample rows to avoid excessive computation
        
        text_cols = [col for col in self.categorical_cols 
                    if self.df[col].dropna().astype(str).str.len().mean() < 50  # Avoid very long text
                    and self.df[col].nunique() > 5  # Avoid pure categorical
                    and self.df[col].nunique() / len(self.df) < 0.9]  # Avoid unique identifiers
                    
        if text_cols and len(self.df) < 50000:  # Only for reasonably sized datasets
            try:
                # Sample rows to check for near-duplicates
                sample_size = min(5000, len(self.df))
                sample_df = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
                
                # For simplicity, we'll just check a few columns for similar values
                from fuzzywuzzy import fuzz
                
                near_dups = {}
                for col in text_cols[:3]:  # Limit to first 3 text columns
                    # Get string values
                    str_values = sample_df[col].dropna().astype(str).tolist()
                    
                    if len(str_values) > 100:
                        str_values = random.sample(str_values, 100)  # Further limit for performance
                    
                    # Find similar strings
                    similar_pairs = []
                    for i in range(len(str_values)):
                        for j in range(i+1, len(str_values)):
                            similarity = fuzz.ratio(str_values[i], str_values[j])
                            if similarity > 80 and similarity < 100:  # High similarity but not identical
                                similar_pairs.append({
                                    'value1': str_values[i], 
                                    'value2': str_values[j], 
                                    'similarity': similarity
                                })
                                
                    # Keep top 5 most similar pairs
                    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                    similar_pairs = similar_pairs[:5]
                    
                    if similar_pairs:
                        near_dups[col] = similar_pairs
                
                if near_dups:
                    results["near_duplicates"] = near_dups
                    results["recommendations"].append(
                        "Found near-duplicate values in text columns. Consider text normalization "
                        "or fuzzy matching to standardize similar values."
                    )
            except ImportError:
                # If fuzzywuzzy is not installed
                results["near_duplicates"]["error"] = "Could not perform fuzzy matching. Install fuzzywuzzy for this feature."
                
        # Check for potentially corrupted values
        corrupt_values = {}
        
        # Check numeric columns for implausible values
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Get statistics
            q1, q3 = col_data.quantile([0.01, 0.99])  # Use 1% and 99% to be more conservative
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 3)
            upper_bound = q3 + (iqr * 3)
            
            # Find extreme outliers (potentially corrupted)
            extreme_low = col_data[col_data < lower_bound]
            extreme_high = col_data[col_data > upper_bound]
            
            if not extreme_low.empty or not extreme_high.empty:
                corrupt_values[col] = {
                    "extreme_low_count": len(extreme_low),
                    "extreme_high_count": len(extreme_high),
                    "normal_range": f"{lower_bound:.2f} - {upper_bound:.2f}",
                    "min_value": col_data.min(),
                    "max_value": col_data.max()
                }
                
                # Only flag as corruption if truly extreme
                if len(extreme_low) > 0 and extreme_low.min() < lower_bound * 10:
                    results["recommendations"].append(
                        f"Column '{col}' has {len(extreme_low)} potentially corrupted low values "
                        f"(min: {extreme_low.min():.2f}, expected > {lower_bound:.2f})"
                    )
                
                if len(extreme_high) > 0 and extreme_high.max() > upper_bound * 10:
                    results["recommendations"].append(
                        f"Column '{col}' has {len(extreme_high)} potentially corrupted high values "
                        f"(max: {extreme_high.max():.2f}, expected < {upper_bound:.2f})"
                    )
        
        # Check categorical columns for suspicious patterns (e.g., malformed emails)
        for col in self.categorical_cols:
            suspicious_count = 0
            sample_suspicious = []
            
            # Check if column appears to contain emails
            if 'email' in col.lower():
                values = self.df[col].dropna().astype(str)
                # Simple regex for valid email format
                valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = values[~values.str.match(valid_email_pattern)]
                
                if not invalid_emails.empty:
                    suspicious_count = len(invalid_emails)
                    sample_suspicious = invalid_emails.head(3).tolist()
                    
                    results["recommendations"].append(
                        f"Column '{col}' has {suspicious_count} ({suspicious_count/len(values)*100:.1f}%) "
                        f"potentially malformed email addresses. Consider validation and cleaning."
                    )
            
            # Check if column appears to contain dates but with invalid formats
            elif 'date' in col.lower() or 'time' in col.lower():
                values = self.df[col].dropna().astype(str)
                try:
                    pd.to_datetime(values, errors='raise')
                except:
                    # If conversion fails, try to identify which values are problematic
                    valid_mask = pd.Series(True, index=values.index)
                    for i, val in values.iteritems():
                        try:
                            pd.to_datetime(val)
                        except:
                            valid_mask.loc[i] = False
                    
                    invalid_dates = values[~valid_mask]
                    if not invalid_dates.empty:
                        suspicious_count = len(invalid_dates)
                        sample_suspicious = invalid_dates.head(3).tolist()
                        
                        results["recommendations"].append(
                            f"Column '{col}' has {suspicious_count} ({suspicious_count/len(values)*100:.1f}%) "
                            f"potentially invalid date/time values. Consider standardizing date formats."
                        )
            
            if suspicious_count > 0:
                corrupt_values[col] = {
                    "suspicious_count": suspicious_count,
                    "suspicious_examples": sample_suspicious,
                    "recommendation": "Check for data entry errors or format inconsistencies"
                }
        
        results["corrupted_values"] = corrupt_values
        
        self.report["data_integrity"]["duplicates_and_corruption"] = results
        return results
    
    def detect_outliers(self, contamination=0.05) -> Dict:
        """
        Detect outliers using multiple methods and recommend actions.
        
        Args:
            contamination: Expected proportion of outliers (default 0.05 = 5%)
        """
        results = {
            "statistical_outliers": {},
            "isolation_forest_outliers": {},
            "lof_outliers": {},
            "combined_outliers": {},
            "recommendations": []
        }
        
        # Only process if we have numeric features
        if not self.numeric_cols:
            results["error"] = "No numeric columns available for outlier detection"
            return results
            
        # Prepare numeric data for outlier detection
        numeric_df = self.df[self.numeric_cols].copy()
        
        # Handle missing values for outlier detection
        numeric_df_filled = numeric_df.fillna(numeric_df.mean())
        
        # Statistical outlier detection (Z-score)
        stat_outliers = {}
        for col in self.numeric_cols:
            col_data = numeric_df[col].dropna()
            if len(col_data) < 10:  # Skip columns with too few values
                continue
                
            # Use robust statistics to avoid outliers influencing the thresholds
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))  # Median Absolute Deviation
            
            if mad == 0:  # Avoid division by zero
                continue
                
            # Modified Z-score method (more robust than standard Z-score)
            modified_z = 0.6745 * (col_data - median) / mad
            outliers_idx = np.abs(modified_z) > 3.5  # Common threshold for modified Z-score
            
            outliers = col_data[outliers_idx]
            if len(outliers) > 0:
                stat_outliers[col] = {
                    "count": len(outliers),
                    "percentage": round(len(outliers) / len(col_data) * 100, 2),
                    "min_value": float(outliers.min()),
                    "max_value": float(outliers.max()),
                    "normal_range": f"{float(median - 3.5 * mad / 0.6745):.2f} - {float(median + 3.5 * mad / 0.6745):.2f}"
                }
        
        results["statistical_outliers"] = stat_outliers
        
        # Machine learning based outlier detection
        if len(numeric_df_filled) > 20:  # Only if we have enough samples
            try:
                # Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_forest.fit(numeric_df_filled)
                outlier_pred = iso_forest.predict(numeric_df_filled)
                outlier_idx = outlier_pred == -1  # -1 indicates outliers
                
                if outlier_idx.sum() > 0:
                    outlier_pct = outlier_idx.sum() / len(outlier_idx) * 100
                    results["isolation_forest_outliers"] = {
                        "count": int(outlier_idx.sum()),
                        "percentage": round(outlier_pct, 2),
                        "feature_importance": {}
                    }
                    
                    # Try to identify which features contribute most to outlier detection
                    if hasattr(iso_forest, 'feature_importances_'):
                        feature_importances = iso_forest.feature_importances_
                        for i, col in enumerate(numeric_df_filled.columns):
                            results["isolation_forest_outliers"]["feature_importance"][col] = round(feature_importances[i], 3)
                
                # Local Outlier Factor (for density-based outlier detection)
                if len(numeric_df_filled) < 100000:  # LOF can be slow on very large datasets
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                    lof_pred = lof.fit_predict(numeric_df_filled)
                    lof_idx = lof_pred == -1  # -1 indicates outliers
                    
                    if lof_idx.sum() > 0:
                        lof_pct = lof_idx.sum() / len(lof_idx) * 100
                        results["lof_outliers"] = {
                            "count": int(lof_idx.sum()),
                            "percentage": round(lof_pct, 2)
                        }
                        
                # Find rows that are outliers according to multiple methods
                if "isolation_forest_outliers" in results and "lof_outliers" in results:
                    combined_idx = outlier_idx & lof_idx
                    if combined_idx.sum() > 0:
                        results["combined_outliers"] = {
                            "count": int(combined_idx.sum()),
                            "percentage": round(combined_idx.sum() / len(combined_idx) * 100, 2)
                        }
            
            except Exception as e:
                results["ml_outlier_detection_error"] = str(e)
        
        # Generate recommendations
        if stat_outliers:
            columns_with_many_outliers = [col for col, info in stat_outliers.items() if info["percentage"] > 5]
            if columns_with_many_outliers:
                results["recommendations"].append(
                    f"Columns with high outlier percentage: {', '.join(columns_with_many_outliers)}. "
                    f"Consider scaling (e.g., log transform) or winsorizing extreme values."
                )
                
        results["code_example_winsorization"] = """
# Winsorize outliers (cap at percentiles)
from scipy.stats import mstats
for col in ['column1', 'column2']:
    df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])  # Cap at 1st and 99th percentiles

# Or using pandas:
for col in ['column1', 'column2']:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
"""
        
        if "isolation_forest_outliers" in results and results["isolation_forest_outliers"].get("percentage", 0) > 5:
            results["recommendations"].append(
                "Consider investigating or removing multivariate outliers detected by Isolation Forest, "
                "especially for sensitive models like fraud detection or anomaly detection."
            )
            
            results["code_example_outlier_removal"] = """
# Identify and remove outliers using Isolation Forest
from sklearn.ensemble import IsolationForest

# Fit the model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso_forest.fit_predict(df[numeric_columns])

# Create a clean dataset without outliers
df_clean = df[outlier_pred == 1].copy()  # Keep only inliers
print(f"Removed {(outlier_pred == -1).sum()} outliers ({(outlier_pred == -1).sum()/len(outlier_pred)*100:.1f}%)")
"""
        
        self.report["data_integrity"]["outliers"] = results
        return results
    
    def analyze_label_quality(self) -> Dict:
        """
        Analyze the quality of labels/target variable and identify potential issues.
        """
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "No target column specified for label quality analysis"}
            
        results = {
            "label_statistics": {},
            "consistency_issues": {},
            "recommendations": []
        }
        
        target = self.df[self.target_column]
        
        # Calculate basic statistics
        self._add_basic_label_statistics(target, results)
        
        # Analyze based on target type
        if self.target_column in self.numeric_cols:
            self._analyze_numeric_target(target, results)
        else:
            self._analyze_categorical_target(target, results)
        
        # Check for probability-like values with errors
        if target.dtype.kind in 'fiu':  # If numeric
            self._check_probability_values(target, results)
        
        # Generate recommendations for missing values
        self._add_missing_value_recommendations(results)
        
        # Add code examples
        self._add_code_examples(results)
        
        self.report["label_quality"] = results
        return results
    
    def _add_basic_label_statistics(self, target: pd.Series, results: Dict) -> None:
        """Add basic statistics about target variable to results."""
        results["label_statistics"]["count"] = len(target)
        results["label_statistics"]["missing"] = int(target.isna().sum())
        results["label_statistics"]["missing_percentage"] = round(target.isna().sum() / len(target) * 100, 2)
    
    def _analyze_numeric_target(self, target: pd.Series, results: Dict) -> None:
        """Analyze a numeric target variable."""
        # Basic numeric statistics
        results["label_statistics"]["mean"] = float(target.mean())
        results["label_statistics"]["median"] = float(target.median())
        results["label_statistics"]["std"] = float(target.std())
        results["label_statistics"]["min"] = float(target.min())
        results["label_statistics"]["max"] = float(target.max())
        
        # Check for outliers/implausible values
        self._check_numeric_outliers(target, results)
    
    def _check_numeric_outliers(self, target: pd.Series, results: Dict) -> None:
        """Check for outliers in numeric target."""
        q1, q3 = target.quantile([0.01, 0.99])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 3)
        upper_bound = q3 + (iqr * 3)
        
        extreme_values = target[(target < lower_bound) | (target > upper_bound)].count()
        if extreme_values > 0:
            results["consistency_issues"]["extreme_values"] = {
                "count": int(extreme_values),
                "percentage": round(extreme_values / len(target) * 100, 2),
                "expected_range": f"{float(lower_bound):.2f} - {float(upper_bound):.2f}"
            }
            results["recommendations"].append(
                f"Target variable has {extreme_values} extreme values that may be errors. "
                f"Verify that values outside the range {float(lower_bound):.2f} - {float(upper_bound):.2f} are correct."
            )
    
    def _analyze_categorical_target(self, target: pd.Series, results: Dict) -> None:
        """Analyze a categorical target variable."""
        value_counts = target.value_counts()
        results["label_statistics"]["unique_values"] = int(target.nunique())
        results["label_statistics"]["distribution"] = {
            str(label): {
                "count": int(count),
                "percentage": round(count / len(target) * 100, 2)
            } for label, count in value_counts.items()
        }
        
        # Check for rare categories
        self._check_rare_categories(target, value_counts, results)
        
        # Check for potentially mislabeled data
        if len(value_counts) > 1 and value_counts.max() / value_counts.min() > 5:
            self._check_mislabeled_instances(target, value_counts, results)
    
    def _check_rare_categories(self, target: pd.Series, value_counts: pd.Series, results: Dict) -> None:
        """Check for rare categories in target variable."""
        rare_labels = value_counts[value_counts / len(target) < 0.01]
        if not rare_labels.empty:
            results["consistency_issues"]["rare_labels"] = {
                "count": int(len(rare_labels)),
                "labels": [str(label) for label in rare_labels.index],
                "total_instances": int(rare_labels.sum())
            }
            results["recommendations"].append(
                f"Found {len(rare_labels)} rare categories in target variable with total {rare_labels.sum()} instances. "
                f"Consider merging rare categories or using stratified sampling."
            )
    
    def _check_mislabeled_instances(self, target: pd.Series, value_counts: pd.Series, results: Dict) -> None:
        """Check for potentially mislabeled instances using a simple model."""
        try:
            # Only proceed if we have sufficient data and features
            if len(self.numeric_cols) < 3 or len(self.df) < 100:
                return
                
            # Identify majority and minority classes
            majority_class = value_counts.idxmax()
            minority_classes = [c for c in value_counts.index if c != majority_class]
            
            # Prepare data for modeling
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            X_numeric = X[self.numeric_cols].copy().fillna(X[self.numeric_cols].mean())
            
            # Train a simple model to identify mislabeling
            mislabeled = self._detect_mislabeled_with_model(X_numeric, y, minority_classes)
            
            if mislabeled:
                results["consistency_issues"]["potentially_mislabeled"] = mislabeled
                results["recommendations"].append(
                    "Potential mislabeled instances detected. Consider using techniques like "
                    "active learning or confident learning to identify and correct label errors."
                )
        except Exception:
            # Skip if analysis fails
            pass
    
    def _detect_mislabeled_with_model(self, X_numeric: pd.DataFrame, y: pd.Series, 
                                      minority_classes: List) -> List:
        """Use a simple model to detect potentially mislabeled instances."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        
        mislabeled = []
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        y_pred_proba = cross_val_predict(clf, X_numeric, y, cv=5, method='predict_proba')
        
        # Create DataFrame with prediction probabilities
        proba_df = pd.DataFrame(y_pred_proba, columns=[f"prob_{c}" for c in clf.classes_])
        proba_df['true_label'] = y.values
        
        # Identify potentially mislabeled instances
        for cls in minority_classes:
            cls_idx = proba_df['true_label'] == cls
            if cls_idx.sum() <= 10:  # Skip if too few samples
                continue
                
            prob_col = f"prob_{cls}"
            if prob_col not in proba_df.columns:
                continue
                
            suspicious = proba_df.loc[cls_idx, prob_col] < 0.1  # Very low probability
            suspicious_count = suspicious.sum()
            
            if suspicious_count > 0:
                mislabeled.append({
                    "class": str(cls),
                    "potentially_mislabeled_count": int(suspicious_count),
                    "percentage_of_class": round(suspicious_count / cls_idx.sum() * 100, 2)
                })
                
        return mislabeled
    
    def _check_probability_values(self, target: pd.Series, results: Dict) -> None:
        """Check if target looks like probabilities with invalid values."""
        # Is the target mostly in [0,1] range but has some values outside?
        target_max = target.max()
        target_min = target.min()
        has_high_values = (target > 1.0).sum() > 0
        has_low_values = (target < 0.0).sum() > 0
        
        if (target_max <= 1.2 and target_min >= -0.2 and (has_high_values or has_low_values)):
            invalid_probs = ((target > 1.0) | (target < 0.0)).sum()
            results["consistency_issues"]["invalid_probabilities"] = {
                "count": int(invalid_probs),
                "percentage": round(invalid_probs / len(target) * 100, 2)
            }
            results["recommendations"].append(
                f"Target appears to contain probability values but has {invalid_probs} values outside the [0,1] range. "
                f"Consider clipping values to valid probability range."
            )
    
    def _add_missing_value_recommendations(self, results: Dict) -> None:
        """Add recommendations for missing values in target."""
        missing_pct = results["label_statistics"].get("missing_percentage", 0)
        if missing_pct > 0:
            results["recommendations"].append(
                f"Target has {missing_pct}% missing values. "
                f"Consider imputing missing targets or removing these instances."
            )
    
    def _add_code_examples(self, results: Dict) -> None:
        """Add code examples for handling various label issues."""
        results["code_examples"] = {}
        
        # Missing values example
        if results["label_statistics"].get("missing_percentage", 0) > 0:
            results["code_examples"]["handle_missing_labels"] = """
# Option 1: Remove rows with missing target values
df_cleaned = df.dropna(subset=['target_column'])

# Option 2: For regression, impute with mean/median
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['target_column'] = imputer.fit_transform(df[['target_column']])[:, 0]
"""
        
        # Rare classes example
        if "rare_labels" in results.get("consistency_issues", {}):
            results["code_examples"]["handle_rare_classes"] = """
# Combine rare classes into an 'Other' category
min_count = 50  # Minimum instances per class
value_counts = df['target_column'].value_counts()
rare_classes = value_counts[value_counts < min_count].index.tolist()

# Apply the transformation
df['target_column_grouped'] = df['target_column'].apply(
    lambda x: 'Other' if x in rare_classes else x
)
"""
        
        # Mislabeled instances example
        if "potentially_mislabeled" in results.get("consistency_issues", {}):
            results["code_examples"]["detect_mislabeled"] = """
# Use Cleanlab to identify and correct potentially mislabeled instances
# pip install cleanlab
import cleanlab
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

# Get out-of-sample predictions
clf = RandomForestClassifier(n_estimators=100, random_state=42)
pred_probs = cross_val_predict(clf, X, y, cv=5, method='predict_proba')

# Find label issues
ranked_label_issues = cleanlab.filter.find_label_issues(
    labels=y,
    pred_probs=pred_probs,
    return_indices_ranked_by='self_confidence'
)

# Get the indices of the top 5% most likely mislabeled examples
num_issues_to_inspect = max(10, int(0.05 * len(y)))
issue_indices = ranked_label_issues[:num_issues_to_inspect]

# Show potential label issues
print(f"Potential label issues: {len(issue_indices)} instances may have incorrect labels")
"""
            
    def visualize_issues(self) -> Dict[str, plt.Figure]:
        """
        Generate visualizations for detected data issues.
        
        Returns:
            Dictionary of matplotlib figures visualizing various data quality issues
        """
        figures = {}
        
        # 1. Missing values visualization
        if "missing_patterns" in self.report["data_integrity"]:
            miss_data = self.report["data_integrity"]["missing_patterns"]["missing_percentages"]
            if miss_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                miss_df = pd.Series(miss_data).sort_values(ascending=False)
                
                # Only plot columns with missing values
                miss_df = miss_df[miss_df > 0]
                
                if not miss_df.empty:
                    miss_df.plot(kind='bar', ax=ax)
                    ax.set_title('Missing Values by Column (%)', fontsize=14)
                    ax.set_ylabel('Missing Percentage')
                    ax.set_xlabel('Column')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    figures["missing_values"] = fig
        
        # 2. Class imbalance visualization
        if self.target_column and "class_balance" in self.report:
            if "class_distribution" in self.report["class_balance"]:
                dist = self.report["class_balance"]["class_distribution"]
                if dist:
                    class_counts = {cls: data["count"] for cls, data in dist.items()}
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.bar(class_counts.keys(), class_counts.values())
                    plt.title('Class Distribution in Target Variable', fontsize=14)
                    plt.ylabel('Count')
                    plt.xlabel('Class')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    figures["class_imbalance"] = fig
        
        # 3. Outlier visualization
        if "outliers" in self.report["data_integrity"]:
            outlier_data = self.report["data_integrity"]["outliers"].get("statistical_outliers", {})
            
            if outlier_data and len(outlier_data) > 0:
                # Get top 5 columns with most outliers by percentage
                top_cols = sorted(
                    outlier_data.items(), 
                    key=lambda x: x[1]["percentage"], 
                    reverse=True
                )[:5]
                
                if top_cols:
                    fig, axes = plt.subplots(len(top_cols), 1, figsize=(10, 3*len(top_cols)))
                    if len(top_cols) == 1:
                        axes = [axes]
                        
                    for i, (col, data) in enumerate(top_cols):
                        # Create boxplot
                        axes[i].boxplot(self.df[col].dropna(), vert=False)
                        axes[i].set_title(f'Outliers in {col} ({data["percentage"]}%)', fontsize=12)
                        axes[i].set_xlabel('Value')
                        axes[i].set_yticks([])
                        
                        # Add normal range indicator
                        if "normal_range" in data:
                            normal_range = data["normal_range"].split(" - ")
                            if len(normal_range) == 2:
                                try:
                                    lower = float(normal_range[0])
                                    upper = float(normal_range[1])
                                    axes[i].axvspan(lower, upper, alpha=0.2, color='green')
                                except:
                                    pass
                    
                    plt.tight_layout()
                    figures["outliers"] = fig
        
        # 4. Storage efficiency comparison
        if "storage_efficiency" in self.report:
            formats = self.report["storage_efficiency"].get("format_recommendations", {})
            if formats:
                format_sizes = {}
                format_times = {}
                
                for fmt, data in formats.items():
                    if fmt != "recommendation" and fmt != "code_example":
                        format_sizes[fmt] = data.get("size_mb", 0)
                        format_times[fmt] = data.get("relative_load_time", 1.0)
                
                if format_sizes:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Size comparison
                    ax1.bar(format_sizes.keys(), format_sizes.values())
                    ax1.set_title('File Size by Format (MB)', fontsize=14)
                    ax1.set_ylabel('Size (MB)')
                    ax1.set_xlabel('Format')
                    ax1.set_xticks(range(len(format_sizes)))
                    ax1.set_xticklabels(format_sizes.keys(), rotation=45, ha='right')
                    
                    # Loading time comparison
                    ax2.bar(format_times.keys(), format_times.values())
                    ax2.set_title('Relative Loading Time by Format', fontsize=14)
                    ax2.set_ylabel('Relative Time')
                    ax2.set_xlabel('Format')
                    ax2.set_xticks(range(len(format_times)))
                    ax2.set_xticklabels(format_times.keys(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    figures["storage_efficiency"] = fig
        
        # 5. Feature correlation heatmap
        if self.numeric_cols and len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr().abs()
            
            # Plot only if we have a reasonable number of columns
            if len(corr_matrix) <= 30:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
                ax.set_title('Feature Correlation Heatmap', fontsize=14)
                plt.tight_layout()
                figures["correlation"] = fig
        
        return figures
    
    def apply_fixes(self, fixes_to_apply: List[str]) -> pd.DataFrame:
        """
        Apply selected fixes to the dataframe.
        
        Args:
            fixes_to_apply: List of fix types to apply ('missing', 'outliers', 'duplicates', 
                                                     'class_imbalance', 'storage', 'corrupt')
                            
        Returns:
            Cleaned/transformed DataFrame
        """
        df_fixed = self.df.copy()
        fixes_applied = {}
        
        # 1. Fix missing values
        if 'missing' in fixes_to_apply:
            missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
            if missing_patterns:
                fixes_applied['missing'] = {}
                
                # Get imputation recommendations
                recommendations = missing_patterns.get("imputation_recommendations", {})
                
                # Apply appropriate imputation for each column
                for col, recommendation in recommendations.items():
                    # Skip overall recommendation
                    if col == "overall":
                        continue
                        
                    if "mean" in recommendation.lower():
                        df_fixed[col] = df_fixed[col].fillna(df_fixed[col].mean())
                        fixes_applied['missing'][col] = "Mean imputation"
                    elif "median" in recommendation.lower():
                        df_fixed[col] = df_fixed[col].fillna(df_fixed[col].median())
                        fixes_applied['missing'][col] = "Median imputation"
                    elif "mode" in recommendation.lower() or "category" in recommendation.lower():
                        # For categorical, use mode
                        if col in self.categorical_cols:
                            mode_value = df_fixed[col].mode().iloc[0] if not df_fixed[col].mode().empty else "Missing"
                            df_fixed[col] = df_fixed[col].fillna(mode_value)
                            fixes_applied['missing'][col] = f"Mode imputation with value: {mode_value}"
                        else:
                            # Create missing category
                            df_fixed[col] = df_fixed[col].fillna("Missing")
                            fixes_applied['missing'][col] = "Created 'Missing' category"
                    elif "drop" in recommendation.lower():
                        # Don't actually drop columns here, just log the recommendation
                        fixes_applied['missing'][col] = "Column recommended for dropping (not applied)"
        
        # 2. Remove duplicates
        if 'duplicates' in fixes_to_apply:
            if self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0) > 0:
                original_rows = len(df_fixed)
                df_fixed = df_fixed.drop_duplicates()
                dropped_rows = original_rows - len(df_fixed)
                
                fixes_applied['duplicates'] = f"Removed {dropped_rows} duplicate rows"
        
        # 3. Handle outliers
        if 'outliers' in fixes_to_apply:
            outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
            if outliers:
                fixes_applied['outliers'] = {}
                
                for col, data in outliers.items():
                    # Only handle columns with high outlier percentage
                    if data.get("percentage", 0) > 5 and col in self.numeric_cols:
                        # Extract normal range
                        normal_range = data.get("normal_range", "")
                        if normal_range:
                            try:
                                lower, upper = map(float, normal_range.split(" - "))
                                
                                # Apply winsorization
                                original_values = df_fixed[col].copy()
                                df_fixed[col] = df_fixed[col].clip(lower=lower, upper=upper)
                                changed_values = (original_values != df_fixed[col]).sum()
                                
                                fixes_applied['outliers'][col] = f"Clipped {changed_values} outliers to range [{lower:.2f}, {upper:.2f}]"
                            except:
                                pass
        
        # 4. Fix class imbalance
        if 'class_imbalance' in fixes_to_apply and self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance.get("is_imbalanced", False):
                # We'll only apply SMOTE if we have numeric features and categorical target
                if (self.target_column not in self.numeric_cols and 
                    len(self.numeric_cols) >= 3 and 
                    self.df[self.target_column].notna().all()):
                    
                    try:
                        # Prepare data for SMOTE
                        features = self.numeric_cols.copy()
                        if self.target_column in features:
                            features.remove(self.target_column)
                            
                        X = df_fixed[features].fillna(df_fixed[features].mean())
                        y = df_fixed[self.target_column]
                        
                        # Apply SMOTE
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(X, y)
                        
                        # Create new dataframe with balanced classes
                        df_balanced = pd.DataFrame(X_resampled, columns=features)
                        df_balanced[self.target_column] = y_resampled
                        
                        # For the remaining columns, we'll add NaN values
                        for col in df_fixed.columns:
                            if col not in df_balanced.columns:
                                df_balanced[col] = np.nan
                        
                        # Count new samples
                        original_count = len(df_fixed)
                        new_count = len(df_balanced)
                        added_samples = new_count - original_count
                        
                        # Use the balanced dataset
                        df_fixed = df_balanced
                        
                        fixes_applied['class_imbalance'] = f"Applied SMOTE to balance classes. Added {added_samples} synthetic samples."
                    except Exception as e:
                        fixes_applied['class_imbalance_error'] = str(e)
        
        # 5. Fix corrupt values
        if 'corrupt' in fixes_to_apply:
            corrupt_data = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
            if corrupt_data:
                fixes_applied['corrupt'] = {}
                
                # Handle numeric corrupt values
                for col, data in corrupt_data.items():
                    if col in self.numeric_cols:
                        # Check if we have a normal range
                        if "normal_range" in data:
                            try:
                                normal_range = data["normal_range"].split(" - ")
                                lower = float(normal_range[0])
                                upper = float(normal_range[1])
                                
                                # Replace extreme values with NaN
                                mask = (df_fixed[col] < lower) | (df_fixed[col] > upper)
                                extreme_count = mask.sum()
                                
                                if extreme_count > 0:
                                    df_fixed.loc[mask, col] = np.nan
                                    fixes_applied['corrupt'][col] = f"Replaced {extreme_count} extreme values with NaN"
                            except:
                                pass
                    # Handle corrupt email format
                    elif col in self.categorical_cols and 'email' in col.lower():
                        if data.get("suspicious_count", 0) > 0:
                            # Simple regex for valid email format
                            valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                            mask = ~df_fixed[col].astype(str).str.match(valid_email_pattern)
                            invalid_count = mask.sum()
                            
                            if invalid_count > 0:
                                df_fixed.loc[mask, col] = np.nan
                                fixes_applied['corrupt'][col] = f"Replaced {invalid_count} invalid email formats with NaN"
                    
                    # Handle corrupt date format
                    elif col in self.categorical_cols and ('date' in col.lower() or 'time' in col.lower()):
                        if data.get("suspicious_count", 0) > 0:
                            try:
                                # Attempt to convert to datetime, marking invalid formats
                                original_values = df_fixed[col].copy()
                                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
                                invalid_count = df_fixed[col].isna().sum() - original_values.isna().sum()
                                
                                if invalid_count > 0:
                                    fixes_applied['corrupt'][col] = f"Converted column to datetime format. {invalid_count} invalid dates became NaN"
                            except:
                                pass
        
        # 6. Optimize storage if requested
        if 'storage' in fixes_to_apply:
            # Apply dtype optimizations
            for col in df_fixed.columns:
                if col in self.numeric_cols:
                    if df_fixed[col].isna().any():
                        # Skip optimization for columns with NaN
                        continue
                        
                    # Check if values are integers
                    if np.array_equal(df_fixed[col].dropna(), df_fixed[col].dropna().astype(int)):
                        # Determine best integer dtype
                        col_min = df_fixed[col].min()
                        col_max = df_fixed[col].max()
                        
                        if col_min >= 0:
                            if col_max < 2**8:
                                df_fixed[col] = df_fixed[col].astype(np.uint8)
                            elif col_max < 2**16:
                                df_fixed[col] = df_fixed[col].astype(np.uint16)
                            elif col_max < 2**32:
                                df_fixed[col] = df_fixed[col].astype(np.uint32)
                        else:
                            if col_min > -2**7 and col_max < 2**7:
                                df_fixed[col] = df_fixed[col].astype(np.int8)
                            elif col_min > -2**15 and col_max < 2**15:
                                df_fixed[col] = df_fixed[col].astype(np.int16)
                            elif col_min > -2**31 and col_max < 2**31:
                                df_fixed[col] = df_fixed[col].astype(np.int32)
                
                # Optimize object columns with categorical dtype
                elif col in self.categorical_cols:
                    if df_fixed[col].nunique() / len(df_fixed) < 0.5:  # Only for reasonably low cardinality
                        df_fixed[col] = df_fixed[col].astype('category')
            
            # Calculate memory savings
            original_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            optimized_mem = df_fixed.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            savings = original_mem - optimized_mem
            savings_pct = (savings / original_mem * 100) if original_mem > 0 else 0
            
            fixes_applied['storage'] = f"Optimized dtypes. Memory usage reduced from {original_mem:.2f}MB to {optimized_mem:.2f}MB ({savings_pct:.2f}% savings)"
        
        return df_fixed, fixes_applied
    
    def generate_comprehensive_report(self, include_figures=True) -> Dict:
        """
        Generate a comprehensive data quality report with all findings.
        
        Args:
            include_figures: Whether to include visualizations in the report
            
        Returns:
            Comprehensive report dictionary
        """
        # Run all analyses if they haven't been run already
        if not self.report["data_integrity"]:
            self.analyze_missing_patterns()
            self.detect_duplicates_and_corruption()
            self.detect_outliers()
            
        if not self.report["feature_analysis"] and self.target_column:
            self.analyze_feature_relevance()
            
        if not self.report["class_balance"] and self.target_column:
            self.analyze_class_imbalance()
            
        if not self.report["leakage_detection"] and self.target_column:
            self.detect_data_leakage()
            
        if not self.report["storage_efficiency"]:
            self.analyze_storage_efficiency()
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score()
        
        # Compile all recommendations into a single list
        all_recommendations = []
        for category, data in self.report.items():
            if isinstance(data, dict):
                for section, section_data in data.items():
                    if isinstance(section_data, dict) and "recommendations" in section_data:
                        for rec in section_data["recommendations"]:
                            all_recommendations.append({
                                "category": category,
                                "section": section,
                                "recommendation": rec,
                                "severity": self._determine_recommendation_severity(rec)
                            })
        
        # Sort recommendations by severity
        all_recommendations.sort(key=lambda x: x["severity"], reverse=True)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(quality_score, all_recommendations)
        
        # Include visualizations if requested
        figures = {}
        if include_figures:
            figures = self.visualize_issues()
        
        # Build final report
        comprehensive_report = {
            "dataset_info": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "quality_score": quality_score,
            "executive_summary": executive_summary,
            "recommendations": all_recommendations,
            "visualizations": figures,
            "detailed_findings": self.report,
            "recommended_fixes": self._recommend_automatic_fixes(),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return comprehensive_report
    
    def _calculate_quality_score(self) -> Dict[str, float]:
        """Calculate overall data quality scores."""
        scores = {
            "overall": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "relevance": 0.0,
            "balance": 0.0,
            "integrity": 0.0
        }
        
        # 1. Completeness score (based on missing data)
        missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
        if missing_patterns:
            missing_percentages = missing_patterns.get("missing_percentages", {})
            if missing_percentages:
                # Calculate average missing percentage across all columns
                avg_missing = sum(missing_percentages.values()) / len(missing_percentages) if missing_percentages else 0
                scores["completeness"] = max(0, 100 - avg_missing) / 100
        else:
            # If no missing data analysis, assume perfect completeness
            scores["completeness"] = 1.0
            
        # 2. Consistency score (based on outliers and corrupted values)
        outliers = self.report.get("data_integrity", {}).get("outliers", {})
        corrupt = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
        
        # Start with perfect score and deduct for issues
        consistency_score = 1.0
        
        # Deduct for statistical outliers
        stat_outliers = outliers.get("statistical_outliers", {})
        if stat_outliers:
            # Calculate average outlier percentage across columns
            outlier_pcts = [data.get("percentage", 0) for data in stat_outliers.values()]
            avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0
            consistency_score -= avg_outlier_pct / 200  # Deduct half of the average percentage
        
        # Deduct for corrupted values
        if corrupt:
            corrupt_deduction = min(0.3, len(corrupt) * 0.05)  # Up to 0.3 for corrupted values
            consistency_score -= corrupt_deduction
            
        scores["consistency"] = max(0, consistency_score)
        
        # 3. Balance score (for class imbalance if applicable)
        if self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance:
                if class_balance.get("is_imbalanced", False):
                    # Calculate score based on imbalance ratio
                    imbalance_ratio = class_balance.get("imbalance_ratio", 1.0)
                    balance_score = 1.0
                    
                    # Progressive deductions based on severity of imbalance
                    if imbalance_ratio > 100:
                        balance_score = 0.2
                    elif imbalance_ratio > 50:
                        balance_score = 0.4
                    elif imbalance_ratio > 20:
                        balance_score = 0.6
                    elif imbalance_ratio > 10:
                        balance_score = 0.7
                    elif imbalance_ratio > 5:
                        balance_score = 0.8
                    elif imbalance_ratio > 2:
                        balance_score = 0.9
                        
                    scores["balance"] = balance_score
                else:
                    scores["balance"] = 1.0
        else:
            # If no target column, not applicable
            scores["balance"] = 1.0
            
        # 4. Relevance score (feature importance if target exists)
        if self.target_column:
            feature_analysis = self.report.get("feature_analysis", {})
            relevance_score = 1.0
            
            # Deduct for constant or quasi-constant features
            if "constant_features" in feature_analysis:
                const_deduction = min(0.3, len(feature_analysis["constant_features"]) * 0.1)
                relevance_score -= const_deduction
                
            if "quasi_constant_features" in feature_analysis:
                quasi_const_deduction = min(0.2, len(feature_analysis["quasi_constant_features"]) * 0.05)
                relevance_score -= quasi_const_deduction
                
            # Deduct for correlated features (potential redundancy)
            if "correlated_feature_groups" in feature_analysis:
                corr_deduction = min(0.2, len(feature_analysis["correlated_feature_groups"]) * 0.05)
                relevance_score -= corr_deduction
                
            scores["relevance"] = max(0, relevance_score)
        else:
            # If no target column, not applicable
            scores["relevance"] = 1.0
            
        # 5. Integrity score (based on duplicates, leakage)
        integrity_score = 1.0
        
        # Deduct for duplicates
        duplicates = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicates > 0:
            duplicate_pct = duplicates / len(self.df) if len(self.df) > 0 else 0
            integrity_score -= duplicate_pct  # Directly deduct percentage of duplicates
            
        # Deduct for data leakage if target exists
        if self.target_column:
            leakage = self.report.get("leakage_detection", {})
            if leakage:
                high_corr_features = leakage.get("high_correlation_features", [])
                if high_corr_features:
                    # Calculate deduction based on number of suspicious features
                    high_risk_count = sum(1 for f in high_corr_features if f.get("risk") == "Very High")
                    med_risk_count = sum(1 for f in high_corr_features if f.get("risk") == "High")
                    
                    leakage_deduction = min(0.5, high_risk_count * 0.1 + med_risk_count * 0.05)
                    integrity_score -= leakage_deduction
                    
                # Deduct for temporal leakage risk
                if leakage.get("temporal_leakage_risk") == "Potential":
                    integrity_score -= 0.1
                    
        scores["integrity"] = max(0, integrity_score)
        
        # Calculate overall score as weighted average of all components
        weights = {
            "completeness": 0.25,
            "consistency": 0.25,
            "balance": 0.15,
            "relevance": 0.15,
            "integrity": 0.2
        }
        
        overall_score = sum(scores[k] * v for k, v in weights.items())
        scores["overall"] = overall_score
        
        # Convert all scores to percentages
        for key in scores:
            scores[key] = round(scores[key] * 100, 1)
            
        return scores
    
    def _determine_recommendation_severity(self, recommendation: str) -> str:
        """Determine severity level of a recommendation based on keywords."""
        recommendation = recommendation.lower()
        
        # Critical severity keywords
        if any(word in recommendation for word in ["leak", "corruption", "invalid", "bias", "drops", "critical"]):
            return "Critical"
            
        # High severity keywords
        if any(word in recommendation for word in ["remove", "duplicate", "outlier", "imbalance", "missing"]):
            return "High"
            
        # Medium severity keywords
        if any(word in recommendation for word in ["consider", "recommend", "investigate"]):
            return "Medium"
            
        # Default to low severity
        return "Low"
    
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
            # filepath: c:\Users\adilm\repositories\Python\neural-scope\advanced_analysis\data_guardian_enhancer.py
    def visualize_issues(self) -> Dict[str, plt.Figure]:
        """
        Generate visualizations for detected data issues.
        
        Returns:
            Dictionary of matplotlib figures visualizing various data quality issues
        """
        figures = {}
        
        # 1. Missing values visualization
        if "missing_patterns" in self.report["data_integrity"]:
            miss_data = self.report["data_integrity"]["missing_patterns"]["missing_percentages"]
            if miss_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                miss_df = pd.Series(miss_data).sort_values(ascending=False)
                
                # Only plot columns with missing values
                miss_df = miss_df[miss_df > 0]
                
                if not miss_df.empty:
                    miss_df.plot(kind='bar', ax=ax)
                    ax.set_title('Missing Values by Column (%)', fontsize=14)
                    ax.set_ylabel('Missing Percentage')
                    ax.set_xlabel('Column')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    figures["missing_values"] = fig
        
        # 2. Class imbalance visualization
        if self.target_column and "class_balance" in self.report:
            if "class_distribution" in self.report["class_balance"]:
                dist = self.report["class_balance"]["class_distribution"]
                if dist:
                    class_counts = {cls: data["count"] for cls, data in dist.items()}
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.bar(class_counts.keys(), class_counts.values())
                    plt.title('Class Distribution in Target Variable', fontsize=14)
                    plt.ylabel('Count')
                    plt.xlabel('Class')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    figures["class_imbalance"] = fig
        
        # 3. Outlier visualization
        if "outliers" in self.report["data_integrity"]:
            outlier_data = self.report["data_integrity"]["outliers"].get("statistical_outliers", {})
            
            if outlier_data and len(outlier_data) > 0:
                # Get top 5 columns with most outliers by percentage
                top_cols = sorted(
                    outlier_data.items(), 
                    key=lambda x: x[1]["percentage"], 
                    reverse=True
                )[:5]
                
                if top_cols:
                    fig, axes = plt.subplots(len(top_cols), 1, figsize=(10, 3*len(top_cols)))
                    if len(top_cols) == 1:
                        axes = [axes]
                        
                    for i, (col, data) in enumerate(top_cols):
                        # Create boxplot
                        axes[i].boxplot(self.df[col].dropna(), vert=False)
                        axes[i].set_title(f'Outliers in {col} ({data["percentage"]}%)', fontsize=12)
                        axes[i].set_xlabel('Value')
                        axes[i].set_yticks([])
                        
                        # Add normal range indicator
                        if "normal_range" in data:
                            normal_range = data["normal_range"].split(" - ")
                            if len(normal_range) == 2:
                                try:
                                    lower = float(normal_range[0])
                                    upper = float(normal_range[1])
                                    axes[i].axvspan(lower, upper, alpha=0.2, color='green')
                                except:
                                    pass
                    
                    plt.tight_layout()
                    figures["outliers"] = fig
        
        # 4. Storage efficiency comparison
        if "storage_efficiency" in self.report:
            formats = self.report["storage_efficiency"].get("format_recommendations", {})
            if formats:
                format_sizes = {}
                format_times = {}
                
                for fmt, data in formats.items():
                    if fmt != "recommendation" and fmt != "code_example":
                        format_sizes[fmt] = data.get("size_mb", 0)
                        format_times[fmt] = data.get("relative_load_time", 1.0)
                
                if format_sizes:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Size comparison
                    ax1.bar(format_sizes.keys(), format_sizes.values())
                    ax1.set_title('File Size by Format (MB)', fontsize=14)
                    ax1.set_ylabel('Size (MB)')
                    ax1.set_xlabel('Format')
                    ax1.set_xticks(range(len(format_sizes)))
                    ax1.set_xticklabels(format_sizes.keys(), rotation=45, ha='right')
                    
                    # Loading time comparison
                    ax2.bar(format_times.keys(), format_times.values())
                    ax2.set_title('Relative Loading Time by Format', fontsize=14)
                    ax2.set_ylabel('Relative Time')
                    ax2.set_xlabel('Format')
                    ax2.set_xticks(range(len(format_times)))
                    ax2.set_xticklabels(format_times.keys(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    figures["storage_efficiency"] = fig
        
        # 5. Feature correlation heatmap
        if self.numeric_cols and len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr().abs()
            
            # Plot only if we have a reasonable number of columns
            if len(corr_matrix) <= 30:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
                ax.set_title('Feature Correlation Heatmap', fontsize=14)
                plt.tight_layout()
                figures["correlation"] = fig
        
        return figures
    
    def apply_fixes(self, fixes_to_apply: List[str]) -> pd.DataFrame:
        """
        Apply selected fixes to the dataframe.
        
        Args:
            fixes_to_apply: List of fix types to apply ('missing', 'outliers', 'duplicates', 
                                                     'class_imbalance', 'storage', 'corrupt')
                            
        Returns:
            Cleaned/transformed DataFrame
        """
        df_fixed = self.df.copy()
        fixes_applied = {}
        
        # 1. Fix missing values
        if 'missing' in fixes_to_apply:
            missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
            if missing_patterns:
                fixes_applied['missing'] = {}
                
                # Get imputation recommendations
                recommendations = missing_patterns.get("imputation_recommendations", {})
                
                # Apply appropriate imputation for each column
                for col, recommendation in recommendations.items():
                    # Skip overall recommendation
                    if col == "overall":
                        continue
                        
                    if "mean" in recommendation.lower():
                        df_fixed[col] = df_fixed[col].fillna(df_fixed[col].mean())
                        fixes_applied['missing'][col] = "Mean imputation"
                    elif "median" in recommendation.lower():
                        df_fixed[col] = df_fixed[col].fillna(df_fixed[col].median())
                        fixes_applied['missing'][col] = "Median imputation"
                    elif "mode" in recommendation.lower() or "category" in recommendation.lower():
                        # For categorical, use mode
                        if col in self.categorical_cols:
                            mode_value = df_fixed[col].mode().iloc[0] if not df_fixed[col].mode().empty else "Missing"
                            df_fixed[col] = df_fixed[col].fillna(mode_value)
                            fixes_applied['missing'][col] = f"Mode imputation with value: {mode_value}"
                        else:
                            # Create missing category
                            df_fixed[col] = df_fixed[col].fillna("Missing")
                            fixes_applied['missing'][col] = "Created 'Missing' category"
                    elif "drop" in recommendation.lower():
                        # Don't actually drop columns here, just log the recommendation
                        fixes_applied['missing'][col] = "Column recommended for dropping (not applied)"
        
        # 2. Remove duplicates
        if 'duplicates' in fixes_to_apply:
            if self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0) > 0:
                original_rows = len(df_fixed)
                df_fixed = df_fixed.drop_duplicates()
                dropped_rows = original_rows - len(df_fixed)
                
                fixes_applied['duplicates'] = f"Removed {dropped_rows} duplicate rows"
        
        # 3. Handle outliers
        if 'outliers' in fixes_to_apply:
            outliers = self.report.get("data_integrity", {}).get("outliers", {}).get("statistical_outliers", {})
            if outliers:
                fixes_applied['outliers'] = {}
                
                for col, data in outliers.items():
                    # Only handle columns with high outlier percentage
                    if data.get("percentage", 0) > 5 and col in self.numeric_cols:
                        # Extract normal range
                        normal_range = data.get("normal_range", "")
                        if normal_range:
                            try:
                                lower, upper = map(float, normal_range.split(" - "))
                                
                                # Apply winsorization
                                original_values = df_fixed[col].copy()
                                df_fixed[col] = df_fixed[col].clip(lower=lower, upper=upper)
                                changed_values = (original_values != df_fixed[col]).sum()
                                
                                fixes_applied['outliers'][col] = f"Clipped {changed_values} outliers to range [{lower:.2f}, {upper:.2f}]"
                            except:
                                pass
        
        # 4. Fix class imbalance
        if 'class_imbalance' in fixes_to_apply and self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance.get("is_imbalanced", False):
                # We'll only apply SMOTE if we have numeric features and categorical target
                if (self.target_column not in self.numeric_cols and 
                    len(self.numeric_cols) >= 3 and 
                    self.df[self.target_column].notna().all()):
                    
                    try:
                        # Prepare data for SMOTE
                        features = self.numeric_cols.copy()
                        if self.target_column in features:
                            features.remove(self.target_column)
                            
                        X = df_fixed[features].fillna(df_fixed[features].mean())
                        y = df_fixed[self.target_column]
                        
                        # Apply SMOTE
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(X, y)
                        
                        # Create new dataframe with balanced classes
                        df_balanced = pd.DataFrame(X_resampled, columns=features)
                        df_balanced[self.target_column] = y_resampled
                        
                        # For the remaining columns, we'll add NaN values
                        for col in df_fixed.columns:
                            if col not in df_balanced.columns:
                                df_balanced[col] = np.nan
                        
                        # Count new samples
                        original_count = len(df_fixed)
                        new_count = len(df_balanced)
                        added_samples = new_count - original_count
                        
                        # Use the balanced dataset
                        df_fixed = df_balanced
                        
                        fixes_applied['class_imbalance'] = f"Applied SMOTE to balance classes. Added {added_samples} synthetic samples."
                    except Exception as e:
                        fixes_applied['class_imbalance_error'] = str(e)
        
        # 5. Fix corrupt values
        if 'corrupt' in fixes_to_apply:
            corrupt_data = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
            if corrupt_data:
                fixes_applied['corrupt'] = {}
                
                # Handle numeric corrupt values
                for col, data in corrupt_data.items():
                    if col in self.numeric_cols:
                        # Check if we have a normal range
                        if "normal_range" in data:
                            try:
                                normal_range = data["normal_range"].split(" - ")
                                lower = float(normal_range[0])
                                upper = float(normal_range[1])
                                
                                # Replace extreme values with NaN
                                mask = (df_fixed[col] < lower) | (df_fixed[col] > upper)
                                extreme_count = mask.sum()
                                
                                if extreme_count > 0:
                                    df_fixed.loc[mask, col] = np.nan
                                    fixes_applied['corrupt'][col] = f"Replaced {extreme_count} extreme values with NaN"
                            except:
                                pass
                    # Handle corrupt email format
                    elif col in self.categorical_cols and 'email' in col.lower():
                        if data.get("suspicious_count", 0) > 0:
                            # Simple regex for valid email format
                            valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                            mask = ~df_fixed[col].astype(str).str.match(valid_email_pattern)
                            invalid_count = mask.sum()
                            
                            if invalid_count > 0:
                                df_fixed.loc[mask, col] = np.nan
                                fixes_applied['corrupt'][col] = f"Replaced {invalid_count} invalid email formats with NaN"
                    
                    # Handle corrupt date format
                    elif col in self.categorical_cols and ('date' in col.lower() or 'time' in col.lower()):
                        if data.get("suspicious_count", 0) > 0:
                            try:
                                # Attempt to convert to datetime, marking invalid formats
                                original_values = df_fixed[col].copy()
                                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
                                invalid_count = df_fixed[col].isna().sum() - original_values.isna().sum()
                                
                                if invalid_count > 0:
                                    fixes_applied['corrupt'][col] = f"Converted column to datetime format. {invalid_count} invalid dates became NaN"
                            except:
                                pass
        
        # 6. Optimize storage if requested
        if 'storage' in fixes_to_apply:
            # Apply dtype optimizations
            for col in df_fixed.columns:
                if col in self.numeric_cols:
                    if df_fixed[col].isna().any():
                        # Skip optimization for columns with NaN
                        continue
                        
                    # Check if values are integers
                    if np.array_equal(df_fixed[col].dropna(), df_fixed[col].dropna().astype(int)):
                        # Determine best integer dtype
                        col_min = df_fixed[col].min()
                        col_max = df_fixed[col].max()
                        
                        if col_min >= 0:
                            if col_max < 2**8:
                                df_fixed[col] = df_fixed[col].astype(np.uint8)
                            elif col_max < 2**16:
                                df_fixed[col] = df_fixed[col].astype(np.uint16)
                            elif col_max < 2**32:
                                df_fixed[col] = df_fixed[col].astype(np.uint32)
                        else:
                            if col_min > -2**7 and col_max < 2**7:
                                df_fixed[col] = df_fixed[col].astype(np.int8)
                            elif col_min > -2**15 and col_max < 2**15:
                                df_fixed[col] = df_fixed[col].astype(np.int16)
                            elif col_min > -2**31 and col_max < 2**31:
                                df_fixed[col] = df_fixed[col].astype(np.int32)
                
                # Optimize object columns with categorical dtype
                elif col in self.categorical_cols:
                    if df_fixed[col].nunique() / len(df_fixed) < 0.5:  # Only for reasonably low cardinality
                        df_fixed[col] = df_fixed[col].astype('category')
            
            # Calculate memory savings
            original_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            optimized_mem = df_fixed.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            savings = original_mem - optimized_mem
            savings_pct = (savings / original_mem * 100) if original_mem > 0 else 0
            
            fixes_applied['storage'] = f"Optimized dtypes. Memory usage reduced from {original_mem:.2f}MB to {optimized_mem:.2f}MB ({savings_pct:.2f}% savings)"
        
        return df_fixed, fixes_applied
    
    def generate_comprehensive_report(self, include_figures=True) -> Dict:
        """
        Generate a comprehensive data quality report with all findings.
        
        Args:
            include_figures: Whether to include visualizations in the report
            
        Returns:
            Comprehensive report dictionary
        """
        # Run all analyses if they haven't been run already
        if not self.report["data_integrity"]:
            self.analyze_missing_patterns()
            self.detect_duplicates_and_corruption()
            self.detect_outliers()
            
        if not self.report["feature_analysis"] and self.target_column:
            self.analyze_feature_relevance()
            
        if not self.report["class_balance"] and self.target_column:
            self.analyze_class_imbalance()
            
        if not self.report["leakage_detection"] and self.target_column:
            self.detect_data_leakage()
            
        if not self.report["storage_efficiency"]:
            self.analyze_storage_efficiency()
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score()
        
        # Compile all recommendations into a single list
        all_recommendations = []
        for category, data in self.report.items():
            if isinstance(data, dict):
                for section, section_data in data.items():
                    if isinstance(section_data, dict) and "recommendations" in section_data:
                        for rec in section_data["recommendations"]:
                            all_recommendations.append({
                                "category": category,
                                "section": section,
                                "recommendation": rec,
                                "severity": self._determine_recommendation_severity(rec)
                            })
        
        # Sort recommendations by severity
        all_recommendations.sort(key=lambda x: x["severity"], reverse=True)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(quality_score, all_recommendations)
        
        # Include visualizations if requested
        figures = {}
        if include_figures:
            figures = self.visualize_issues()
        
        # Build final report
        comprehensive_report = {
            "dataset_info": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "quality_score": quality_score,
            "executive_summary": executive_summary,
            "recommendations": all_recommendations,
            "visualizations": figures,
            "detailed_findings": self.report,
            "recommended_fixes": self._recommend_automatic_fixes(),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return comprehensive_report
    
    def _calculate_quality_score(self) -> Dict[str, float]:
        """Calculate overall data quality scores."""
        scores = {
            "overall": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "relevance": 0.0,
            "balance": 0.0,
            "integrity": 0.0
        }
        
        # 1. Completeness score (based on missing data)
        missing_patterns = self.report.get("data_integrity", {}).get("missing_patterns", {})
        if missing_patterns:
            missing_percentages = missing_patterns.get("missing_percentages", {})
            if missing_percentages:
                # Calculate average missing percentage across all columns
                avg_missing = sum(missing_percentages.values()) / len(missing_percentages) if missing_percentages else 0
                scores["completeness"] = max(0, 100 - avg_missing) / 100
        else:
            # If no missing data analysis, assume perfect completeness
            scores["completeness"] = 1.0
            
        # 2. Consistency score (based on outliers and corrupted values)
        outliers = self.report.get("data_integrity", {}).get("outliers", {})
        corrupt = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("corrupted_values", {})
        
        # Start with perfect score and deduct for issues
        consistency_score = 1.0
        
        # Deduct for statistical outliers
        stat_outliers = outliers.get("statistical_outliers", {})
        if stat_outliers:
            # Calculate average outlier percentage across columns
            outlier_pcts = [data.get("percentage", 0) for data in stat_outliers.values()]
            avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0
            consistency_score -= avg_outlier_pct / 200  # Deduct half of the average percentage
        
        # Deduct for corrupted values
        if corrupt:
            corrupt_deduction = min(0.3, len(corrupt) * 0.05)  # Up to 0.3 for corrupted values
            consistency_score -= corrupt_deduction
            
        scores["consistency"] = max(0, consistency_score)
        
        # 3. Balance score (for class imbalance if applicable)
        if self.target_column:
            class_balance = self.report.get("class_balance", {})
            if class_balance:
                if class_balance.get("is_imbalanced", False):
                    # Calculate score based on imbalance ratio
                    imbalance_ratio = class_balance.get("imbalance_ratio", 1.0)
                    balance_score = 1.0
                    
                    # Progressive deductions based on severity of imbalance
                    if imbalance_ratio > 100:
                        balance_score = 0.2
                    elif imbalance_ratio > 50:
                        balance_score = 0.4
                    elif imbalance_ratio > 20:
                        balance_score = 0.6
                    elif imbalance_ratio > 10:
                        balance_score = 0.7
                    elif imbalance_ratio > 5:
                        balance_score = 0.8
                    elif imbalance_ratio > 2:
                        balance_score = 0.9
                        
                    scores["balance"] = balance_score
                else:
                    scores["balance"] = 1.0
        else:
            # If no target column, not applicable
            scores["balance"] = 1.0
            
        # 4. Relevance score (feature importance if target exists)
        if self.target_column:
            feature_analysis = self.report.get("feature_analysis", {})
            relevance_score = 1.0
            
            # Deduct for constant or quasi-constant features
            if "constant_features" in feature_analysis:
                const_deduction = min(0.3, len(feature_analysis["constant_features"]) * 0.1)
                relevance_score -= const_deduction
                
            if "quasi_constant_features" in feature_analysis:
                quasi_const_deduction = min(0.2, len(feature_analysis["quasi_constant_features"]) * 0.05)
                relevance_score -= quasi_const_deduction
                
            # Deduct for correlated features (potential redundancy)
            if "correlated_feature_groups" in feature_analysis:
                corr_deduction = min(0.2, len(feature_analysis["correlated_feature_groups"]) * 0.05)
                relevance_score -= corr_deduction
                
            scores["relevance"] = max(0, relevance_score)
        else:
            # If no target column, not applicable
            scores["relevance"] = 1.0
            
        # 5. Integrity score (based on duplicates, leakage)
        integrity_score = 1.0
        
        # Deduct for duplicates
        duplicates = self.report.get("data_integrity", {}).get("duplicates_and_corruption", {}).get("exact_duplicates", 0)
        if duplicates > 0:
            duplicate_pct = duplicates / len(self.df) if len(self.df) > 0 else 0
            integrity_score -= duplicate_pct  # Directly deduct percentage of duplicates
            
        # Deduct for data leakage if target exists
        if self.target_column:
            leakage = self.report.get("leakage_detection", {})
            if leakage:
                high_corr_features = leakage.get("high_correlation_features", [])
                if high_corr_features:
                    # Calculate deduction based on number of suspicious features
                    high_risk_count = sum(1 for f in high_corr_features if f.get("risk") == "Very High")
                    med_risk_count = sum(1 for f in high_corr_features if f.get("risk") == "High")
                    
                    leakage_deduction = min(0.5, high_risk_count * 0.1 + med_risk_count * 0.05)
                    integrity_score -= leakage_deduction
                    
                # Deduct for temporal leakage risk
                if leakage.get("temporal_leakage_risk") == "Potential":
                    integrity_score -= 0.1
                    
        scores["integrity"] = max(0, integrity_score)
        
        # Calculate overall score as weighted average of all components
        weights = {
            "completeness": 0.25,
            "consistency": 0.25,
            "balance": 0.15,
            "relevance": 0.15,
            "integrity": 0.2
        }
        
        overall_score = sum(scores[k] * v for k, v in weights.items())
        scores["overall"] = overall_score
        
        # Convert all scores to percentages
        for key in scores:
            scores[key] = round(scores[key] * 100, 1)
            
        return scores
    
    def _determine_recommendation_severity(self, recommendation: str) -> str:
        """Determine severity level of a recommendation based on keywords."""
        recommendation = recommendation.lower()
        
        # Critical severity keywords
        if any(word in recommendation for word in ["leak", "corruption", "invalid", "bias", "drops", "critical"]):
            return "Critical"
            
        # High severity keywords
        if any(word in recommendation for word in ["remove", "duplicate", "outlier", "imbalance", "missing"]):
            return "High"
            
        # Medium severity keywords
        if any(word in recommendation for word in ["consider", "recommend", "investigate"]):
            return "Medium"
            
        # Default to low severity
        return "Low"
    
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
            