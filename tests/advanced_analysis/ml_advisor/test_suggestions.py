"""
Tests for the ML advisor suggestions module.
"""

import pytest
import torch
import torch.nn as nn
from advanced_analysis.ml_advisor.suggestions import MLAdvisor, OptimizationSuggestion

# Skip tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Create a simple PyTorch model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@pytest.fixture
def sample_model():
    """Create a sample PyTorch model for testing."""
    return SimpleModel()

def test_ml_advisor_initialization():
    """Test that the MLAdvisor can be initialized."""
    advisor = MLAdvisor()
    assert advisor is not None

def test_ml_advisor_with_model(sample_model):
    """Test MLAdvisor with a PyTorch model."""
    advisor = MLAdvisor(model=sample_model, framework="pytorch")
    assert advisor.model is sample_model
    assert advisor.framework == "pytorch"

def test_analyze_pytorch_model(sample_model):
    """Test analysis of a PyTorch model."""
    advisor = MLAdvisor(model=sample_model, framework="pytorch")
    results = advisor._analyze_pytorch_model()
    
    # Check that the results contain expected fields
    assert "model_type" in results
    assert "layer_count" in results
    assert "parameter_count" in results
    assert "suggestions" in results
    
    # Check that the model type is correct
    assert results["model_type"] == "pytorch"
    
    # Check that the layer count is reasonable
    assert results["layer_count"] > 0
    
    # Check that the parameter count is correct
    expected_params = sum(p.numel() for p in sample_model.parameters())
    assert results["parameter_count"] == expected_params
    
    # Check that suggestions were generated
    assert len(results["suggestions"]) > 0
    
    # Check that suggestions are OptimizationSuggestion objects
    for suggestion in results["suggestions"]:
        assert isinstance(suggestion, OptimizationSuggestion)

def test_get_suggestions(sample_model):
    """Test getting optimization suggestions."""
    advisor = MLAdvisor(model=sample_model, framework="pytorch")
    advisor.analyze_model()
    
    # Get all suggestions
    all_suggestions = advisor.get_suggestions()
    assert len(all_suggestions) > 0
    
    # Get suggestions by category
    memory_suggestions = advisor.get_suggestions(category="memory")
    assert all(s.category == "memory" for s in memory_suggestions)
    
    # Get suggestions by minimum priority
    high_priority_suggestions = advisor.get_suggestions(min_priority=4)
    assert all(s.priority >= 4 for s in high_priority_suggestions)

def test_generate_optimization_report(sample_model):
    """Test generation of an optimization report."""
    advisor = MLAdvisor(model=sample_model, framework="pytorch")
    advisor.analyze_model()
    report = advisor.generate_optimization_report()
    
    # Check that the report contains expected fields
    assert "model_summary" in report
    assert "optimization_suggestions" in report
    assert "high_priority_suggestions" in report
    
    # Check that the model summary is correct
    assert report["model_summary"]["framework"] == "pytorch"
    assert report["model_summary"]["layer_count"] > 0
    assert report["model_summary"]["parameter_count"] > 0
    
    # Check that optimization suggestions are organized by category
    assert isinstance(report["optimization_suggestions"], dict)
    
    # Check that high priority suggestions exist
    assert isinstance(report["high_priority_suggestions"], list)

def test_optimization_suggestion_creation():
    """Test creation of OptimizationSuggestion objects."""
    suggestion = OptimizationSuggestion(
        category="memory",
        priority=5,
        description="Use checkpoint to reduce memory usage",
        estimated_impact="30-50% memory reduction",
        code_example="from torch.utils.checkpoint import checkpoint",
        applicable_frameworks=["pytorch"]
    )
    
    # Check that the suggestion has the correct attributes
    assert suggestion.category == "memory"
    assert suggestion.priority == 5
    assert "checkpoint" in suggestion.description
    assert "30-50%" in suggestion.estimated_impact
    assert "checkpoint" in suggestion.code_example
    assert "pytorch" in suggestion.applicable_frameworks
