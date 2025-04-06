"""
Common fixtures for tests.
"""

import pytest
import pandas as pd
import numpy as np

# Skip tests if PyTorch is not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    # Create a DataFrame with various data types and quality issues
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 7],  # Duplicate value in id
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],  # Missing value in name
        'age': [25, 30, 35, 40, 45, 50, 200],  # Outlier in age
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
    })
    return df

@pytest.fixture
def sample_code():
    """Create sample Python code for testing."""
    return """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result

def nested_function(n):
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(i * j)
        result.append(row)
    return result
"""

@pytest.fixture
def sample_ml_code():
    """Create sample ML code for testing."""
    return """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

def train_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

class SimpleNN(nn.Module):
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

@pytest.fixture
def sample_pytorch_model():
    """Create a sample PyTorch model for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
        
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
            
    return SimpleModel()

@pytest.fixture
def sample_pytorch_input():
    """Create sample input data for PyTorch model testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
        
    return torch.randn(32, 10)  # Batch size 32, input dimension 10
