import pytest
from dataguard.data_quality import DataQualityChecker
from dataguard.preprocessing import DataPreprocessor
from dataguard.visualization import VisualizationTool

def test_missing_values_detection():
    data = {
        'A': [1, 2, None, 4],
        'B': [1, 2, 3, 4],
        'C': [None, None, None, None]
    }
    checker = DataQualityChecker(data)
    missing_values = checker.detect_missing_values()
    assert missing_values == {'A': 1, 'C': 4}

def test_duplicate_detection():
    data = {
        'A': [1, 2, 2, 4],
        'B': [1, 2, 2, 4]
    }
    checker = DataQualityChecker(data)
    duplicates = checker.detect_duplicates()
    assert duplicates == [(1, 2)]

def test_outlier_detection():
    data = {
        'A': [1, 2, 3, 100],
        'B': [1, 2, 3, 4]
    }
    checker = DataQualityChecker(data)
    outliers = checker.detect_outliers()
    assert outliers == {'A': [100]}

def test_data_preprocessing():
    data = {
        'A': [1, 2, 3, 4],
        'B': ['cat', 'dog', 'cat', 'dog']
    }
    preprocessor = DataPreprocessor(data)
    processed_data = preprocessor.normalize()
    assert processed_data['A'].max() <= 1.0
    assert processed_data['A'].min() >= 0.0

def test_visualization_tool():
    data = {
        'A': [1, 2, 3, 4],
        'B': [1, 2, 3, 4]
    }
    visualizer = VisualizationTool(data)
    fig = visualizer.plot_distribution('A')
    assert fig is not None  # Check if a figure is generated

def test_report_generation():
    data = {
        'A': [1, 2, None, 4],
        'B': [1, 2, 3, 4]
    }
    checker = DataQualityChecker(data)
    report = checker.generate_report()
    assert 'Missing Values' in report
    assert 'Duplicates' in report
    assert 'Outliers' in report