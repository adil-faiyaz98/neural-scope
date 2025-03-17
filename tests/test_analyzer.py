# tests/test_analyzer.py
import pytest
from prod_ready_aiml_complexity.analyzer import ComplexityAnalyzer

def test_simple_analysis():
    code = """
def naive_knn(data):
    best = float('inf')
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                dist = abs(data[i] - data[j])
                if dist < best:
                    best = dist
    return best
"""
    analyzer = ComplexityAnalyzer(analyze_runtime=False)
    report = analyzer.analyze_function(eval(code))
    assert "inefficiencies" in report
    assert len(report["inefficiencies"]) >= 1
