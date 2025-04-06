"""
Tests for the ML code inefficiency detection module.
"""

import pytest
from advanced_analysis.ml_advisor.inefficiency_detection import InefficiencyDetector

# Sample code snippets with inefficiencies
PANDAS_ITERROWS_CODE = """
import pandas as pd

def process_dataframe(df):
    results = []
    for index, row in df.iterrows():
        results.append(row['value'] * 2)
    return results
"""

NESTED_LOOPS_CODE = """
def compute_pairwise_distances(data):
    n = len(data)
    distances = []
    for i in range(n):
        for j in range(n):
            if i != j:
                distances.append(abs(data[i] - data[j]))
    return distances
"""

LIST_APPEND_CODE = """
def process_list(data):
    results = []
    for x in data:
        results.append(x * 2)
    return results
"""

PYTORCH_CODE = """
import torch

def train_epoch(model, dataloader, criterion, optimizer):
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Move back to CPU for processing
        outputs = outputs.cpu()
"""

def test_inefficiency_detector_initialization():
    """Test that the InefficiencyDetector can be initialized."""
    detector = InefficiencyDetector("")
    assert detector is not None

def test_detect_pandas_iterrows():
    """Test detection of pandas iterrows inefficiency."""
    detector = InefficiencyDetector(PANDAS_ITERROWS_CODE)
    inefficiencies = detector.detect_inefficiencies()
    
    # Check that pandas iterrows was detected
    assert len(inefficiencies) > 0
    assert any("pandas_iterrows" == ineff["name"] for ineff in inefficiencies)

def test_detect_nested_loops():
    """Test detection of nested loops inefficiency."""
    detector = InefficiencyDetector(NESTED_LOOPS_CODE)
    inefficiencies = detector.detect_inefficiencies()
    
    # Check that nested loops were detected
    assert len(inefficiencies) > 0
    assert any("nested_loops" == ineff["name"] for ineff in inefficiencies)

def test_detect_list_append():
    """Test detection of list append in loop inefficiency."""
    detector = InefficiencyDetector(LIST_APPEND_CODE)
    inefficiencies = detector.detect_inefficiencies()
    
    # Check that list append was detected
    assert len(inefficiencies) > 0
    assert any("append_in_loop" == ineff["name"] for ineff in inefficiencies)

def test_detect_cpu_gpu_transfer():
    """Test detection of CPU-GPU transfer inefficiency."""
    detector = InefficiencyDetector(PYTORCH_CODE)
    inefficiencies = detector.detect_inefficiencies()
    
    # Check that CPU-GPU transfer was detected
    assert len(inefficiencies) > 0
    assert any("cpu_gpu_transfer" == ineff["name"] for ineff in inefficiencies)

def test_get_optimization_suggestions():
    """Test getting optimization suggestions for detected inefficiencies."""
    detector = InefficiencyDetector(PANDAS_ITERROWS_CODE)
    suggestions = detector.get_optimization_suggestions()
    
    # Check that suggestions were generated
    assert len(suggestions) > 0
    
    # Check that suggestions contain expected fields
    for suggestion in suggestions:
        assert "type" in suggestion
        assert "severity" in suggestion
        assert "message" in suggestion
        assert "details" in suggestion
        assert "code_example" in suggestion

def test_analyze_code():
    """Test comprehensive code analysis."""
    # Combine multiple inefficiencies in one code snippet
    combined_code = PANDAS_ITERROWS_CODE + "\n\n" + NESTED_LOOPS_CODE
    
    detector = InefficiencyDetector(combined_code)
    results = detector.analyze_code()
    
    # Check that the results contain expected fields
    assert "detected_inefficiencies" in results
    assert "optimization_suggestions" in results
    
    # Check that both inefficiencies were detected
    inefficiency_names = [ineff["name"] for ineff in results["detected_inefficiencies"]]
    assert "pandas_iterrows" in inefficiency_names
    assert "nested_loops" in inefficiency_names
    
    # Check that suggestions were generated for both inefficiencies
    assert len(results["optimization_suggestions"]) >= 2
