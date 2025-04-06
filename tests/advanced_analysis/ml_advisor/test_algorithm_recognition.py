"""
Tests for the ML algorithm recognition module.
"""

import pytest
from advanced_analysis.ml_advisor.algorithm_recognition import MLAlgorithmRecognizer

# Sample code snippets for testing
LINEAR_REGRESSION_CODE = """
import numpy as np
from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
"""

LOGISTIC_REGRESSION_CODE = """
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_classifier(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
"""

DECISION_TREE_CODE = """
from sklearn.tree import DecisionTreeClassifier

def train_tree(X, y):
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    return model
"""

NEURAL_NETWORK_CODE = """
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_nn(model, X, y, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model
"""

def test_ml_algorithm_recognizer_initialization():
    """Test that the MLAlgorithmRecognizer can be initialized."""
    recognizer = MLAlgorithmRecognizer("")
    assert recognizer is not None

def test_identify_linear_regression():
    """Test identification of linear regression algorithm."""
    recognizer = MLAlgorithmRecognizer(LINEAR_REGRESSION_CODE)
    algorithms = recognizer.identify_algorithms()
    
    # Check that linear regression was identified
    assert len(algorithms) > 0
    assert any(algo["algorithm"] == "Linear Regression" for algo in algorithms)

def test_identify_logistic_regression():
    """Test identification of logistic regression algorithm."""
    recognizer = MLAlgorithmRecognizer(LOGISTIC_REGRESSION_CODE)
    algorithms = recognizer.identify_algorithms()
    
    # Check that logistic regression was identified
    assert len(algorithms) > 0
    assert any(algo["algorithm"] == "Logistic Regression" for algo in algorithms)

def test_identify_decision_tree():
    """Test identification of decision tree algorithm."""
    recognizer = MLAlgorithmRecognizer(DECISION_TREE_CODE)
    algorithms = recognizer.identify_algorithms()
    
    # Check that decision tree was identified
    assert len(algorithms) > 0
    assert any(algo["algorithm"] == "Decision Tree" for algo in algorithms)

def test_identify_neural_network():
    """Test identification of neural network algorithm."""
    recognizer = MLAlgorithmRecognizer(NEURAL_NETWORK_CODE)
    algorithms = recognizer.identify_algorithms()
    
    # Check that neural network was identified
    assert len(algorithms) > 0
    assert any(algo["algorithm"] == "Neural Network" for algo in algorithms)

def test_get_optimization_suggestions():
    """Test getting optimization suggestions for identified algorithms."""
    recognizer = MLAlgorithmRecognizer(NEURAL_NETWORK_CODE)
    suggestions = recognizer.get_optimization_suggestions()
    
    # Check that suggestions were generated
    assert len(suggestions) > 0
    
    # Check that suggestions contain algorithm and suggestion fields
    for suggestion in suggestions:
        assert "algorithm" in suggestion
        assert "suggestion" in suggestion
        assert "complexity" in suggestion

def test_analyze_code():
    """Test comprehensive code analysis."""
    # Combine multiple algorithms in one code snippet
    combined_code = LINEAR_REGRESSION_CODE + "\n\n" + DECISION_TREE_CODE
    
    recognizer = MLAlgorithmRecognizer(combined_code)
    results = recognizer.analyze_code()
    
    # Check that the results contain expected fields
    assert "identified_algorithms" in results
    assert "optimization_suggestions" in results
    
    # Check that both algorithms were identified
    algorithms = [algo["algorithm"] for algo in results["identified_algorithms"]]
    assert "Linear Regression" in algorithms
    assert "Decision Tree" in algorithms
    
    # Check that suggestions were generated for both algorithms
    assert len(results["optimization_suggestions"]) > 0
