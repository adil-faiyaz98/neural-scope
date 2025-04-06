"""
Tests for the data quality optimizer module.
"""

import pytest
import pandas as pd
import numpy as np
from advanced_analysis.data_quality.optimizer import DatasetOptimizer, DataLoaderProfilingResult

# Create sample data for testing
@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    # Create a large DataFrame with various data types
    np.random.seed(42)
    n_rows = 1000
    df = pd.DataFrame({
        'id': range(n_rows),
        'name': [f'Person_{i}' for i in range(n_rows)],
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(60000, 15000, n_rows),
        'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Sales'], n_rows),
        'is_manager': np.random.choice([True, False], n_rows),
        'hire_date': pd.date_range(start='2010-01-01', periods=n_rows, freq='D'),
        'performance': np.random.uniform(1, 5, n_rows)
    })
    return df

def test_dataset_optimizer_initialization():
    """Test that the DatasetOptimizer can be initialized."""
    optimizer = DatasetOptimizer()
    assert optimizer is not None

def test_optimize_dtypes(sample_data):
    """Test optimization of data types."""
    optimizer = DatasetOptimizer()
    optimized_df, report = optimizer.optimize_dtypes(sample_data)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "original_memory" in report
    assert "optimized_memory" in report
    assert "memory_reduction_pct" in report
    assert "dtype_changes" in report
    
    # Check that memory usage was reduced
    assert report["optimized_memory"] < report["original_memory"]
    assert report["memory_reduction_pct"] > 0
    
    # Check that the optimized DataFrame has the same shape as the original
    assert optimized_df.shape == sample_data.shape

def test_remove_duplicates(sample_data):
    """Test removal of duplicate rows."""
    # Add some duplicate rows
    duplicated_df = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
    
    optimizer = DatasetOptimizer()
    deduped_df, report = optimizer.remove_duplicates(duplicated_df)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "original_rows" in report
    assert "removed_rows" in report
    assert "remaining_rows" in report
    
    # Check that duplicates were removed
    assert report["removed_rows"] == 10
    assert report["remaining_rows"] == report["original_rows"] - report["removed_rows"]
    assert deduped_df.shape[0] == report["remaining_rows"]

def test_handle_missing_values(sample_data):
    """Test handling of missing values."""
    # Add some missing values
    missing_df = sample_data.copy()
    missing_df.loc[10:20, 'age'] = np.nan
    missing_df.loc[30:40, 'salary'] = np.nan
    
    optimizer = DatasetOptimizer()
    
    # Test with drop strategy
    filled_df_drop, report_drop = optimizer.handle_missing_values(missing_df, strategy='drop')
    assert filled_df_drop.shape[0] < missing_df.shape[0]
    assert report_drop["rows_dropped"] > 0
    
    # Test with fill strategy
    filled_df_fill, report_fill = optimizer.handle_missing_values(missing_df, strategy='fill')
    assert filled_df_fill.shape[0] == missing_df.shape[0]
    assert report_fill["columns_filled"] == 2
    assert not filled_df_fill.isna().any().any()

def test_profile_dataloader():
    """Test profiling of a data loader."""
    # Create a simple data loader function
    def data_loader():
        for i in range(5):
            yield pd.DataFrame({
                'id': range(i*10, (i+1)*10),
                'value': np.random.rand(10)
            })
    
    optimizer = DatasetOptimizer()
    result = optimizer.profile_dataloader(data_loader)
    
    # Check that the result is a DataLoaderProfilingResult
    assert isinstance(result, DataLoaderProfilingResult)
    
    # Check that the result contains expected fields
    assert hasattr(result, "batch_count")
    assert hasattr(result, "total_rows")
    assert hasattr(result, "avg_batch_size")
    assert hasattr(result, "avg_load_time")
    assert hasattr(result, "memory_usage")
    
    # Check that the values are reasonable
    assert result.batch_count == 5
    assert result.total_rows == 50
    assert result.avg_batch_size == 10

def test_optimize_categorical_columns(sample_data):
    """Test optimization of categorical columns."""
    optimizer = DatasetOptimizer()
    optimized_df, report = optimizer.optimize_categorical_columns(sample_data, columns=['department'])
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "original_memory" in report
    assert "optimized_memory" in report
    assert "memory_reduction_pct" in report
    assert "converted_columns" in report
    
    # Check that memory usage was reduced
    assert report["optimized_memory"] < report["original_memory"]
    assert report["memory_reduction_pct"] > 0
    
    # Check that the department column was converted to categorical
    assert optimized_df['department'].dtype.name == 'category'

def test_optimize_dataset(sample_data):
    """Test comprehensive dataset optimization."""
    optimizer = DatasetOptimizer()
    optimized_df, report = optimizer.optimize_dataset(sample_data)
    
    # Check that the report contains expected fields
    assert isinstance(report, dict)
    assert "original_memory" in report
    assert "optimized_memory" in report
    assert "memory_reduction_pct" in report
    assert "steps" in report
    
    # Check that memory usage was reduced
    assert report["optimized_memory"] < report["original_memory"]
    assert report["memory_reduction_pct"] > 0
    
    # Check that the optimized DataFrame has the same shape as the original
    assert optimized_df.shape == sample_data.shape
