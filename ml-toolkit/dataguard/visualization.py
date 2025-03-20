from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_missing_values(df):
    """Generate a visual representation of missing values in the DataFrame."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

def plot_duplicates(df):
    """Generate a visual representation of duplicate entries in the DataFrame."""
    duplicate_count = df.duplicated().sum()
    plt.figure(figsize=(6, 4))
    plt.bar(['Duplicates', 'Unique'], [duplicate_count, len(df) - duplicate_count], color=['red', 'green'])
    plt.title('Duplicate Entries in Dataset')
    plt.ylabel('Count')
    plt.show()

def plot_outliers(df, column):
    """Generate a boxplot to visualize outliers in a specified column."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.xlabel(column)
    plt.show()

def plot_class_distribution(df, target_column):
    """Generate a bar plot to visualize class distribution in the target column."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.show()

def plot_feature_distribution(df, feature_columns):
    """Generate histograms for specified feature columns."""
    df[feature_columns].hist(bins=30, figsize=(15, 10), layout=(2, 3))
    plt.suptitle('Feature Distributions')
    plt.show()

def save_visualization_as_html(fig, filename):
    """Save the generated figure as an HTML file."""
    import plotly.offline as pyo
    pyo.plot(fig, filename=filename)

def save_visualization_as_pdf(fig, filename):
    """Save the generated figure as a PDF file."""
    fig.savefig(filename, format='pdf')