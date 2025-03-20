from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    return missing_info

def check_duplicates(df):
    """Check for duplicate rows in the DataFrame."""
    duplicate_rows = df.duplicated().sum()
    return duplicate_rows

def detect_outliers(df, threshold=1.5):
    """Detect outliers using the IQR method."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
        outliers[col] = df[outlier_condition].index.tolist()
    return outliers

def normalize_data(df):
    """Normalize numerical features in the DataFrame."""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def encode_categorical_data(df):
    """Encode categorical features in the DataFrame."""
    encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def feature_selection(df, target, k=10):
    """Select the top k features based on ANOVA F-value."""
    X = df.drop(target, axis=1)
    y = df[target]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return df[selected_features]

def visualize_missing_values(missing_info):
    """Visualize missing values in the DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_info.index, y=missing_info.values, palette='viridis')
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_outliers(outliers):
    """Visualize outliers for each numerical feature."""
    for col, indices in outliers.items():
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Outliers in {col}')
        plt.show()