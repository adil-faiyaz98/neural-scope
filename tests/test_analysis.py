
import os
import tempfile
import ast
import pytest
from aiml_complexity import analysis

def test_analyze_code_basic():
    # Simple code with no inefficiencies
    code = "x = 5\ny = x + 2\n"
    result = analysis.analyze_code(code)
    # No loops or iterrows, so inefficiencies should be empty
    assert isinstance(result.complexity_score, int)
    assert result.complexity_score >= 0
    assert result.inefficiencies == []

def test_detect_nested_loops_and_iterrows():
    code = """
import pandas as pd
for i in range(5):
    for j in range(5):
        x = i * j
df = pd.DataFrame({'a':[1,2,3]})
for idx, row in df.iterrows():
    print(row)
"""
    result = analysis.analyze_code(code)
    issues = result.inefficiencies
    # Should detect nested loops issue and iterrows issue
    assert any("Nested loops" in issue for issue in issues), "Nested loop not detected"
    assert any("iterrows" in issue for issue in issues), "iterrows usage not detected"
    # Complexity score and ops should reflect loops roughly (5*5=25 ops plus some overhead)
    assert result.estimated_operations >= 25
    assert result.complexity_score == result.estimated_operations

def test_analyze_file_consistency():
    # Write code to a temp file and compare analyze_file vs analyze_code
    code = "for i in range(10):\n    x = i * 2\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(code)
    try:
        result_file = analysis.analyze_file(tmp_name)
        result_code = analysis.analyze_code(code, filename=tmp_name)
        # Both results should be equivalent
        assert result_file.complexity_score == result_code.complexity_score
        assert result_file.estimated_operations == result_code.estimated_operations
        assert result_file.inefficiencies == result_code.inefficiencies
    finally:
        os.remove(tmp_name)

def test_analyze_directory(tmp_path):
    # Create multiple files to analyze
    code1 = "for i in range(100):\n    pass\n"
    code2 = "print(\"hello\")\n"
    file1 = tmp_path / "a.py"
    file2 = tmp_path / "b.py"
    file1.write_text(code1)
    file2.write_text(code2)
    results = analysis.analyze_directory(tmp_path, recursive=False, use_multiprocessing=False)
    # Should find 2 results for 2 files
    assert isinstance(results, list)
    assert len(results) == 2
    # Identify which result corresponds to code1 vs code2 by file attribute
    res1 = next((r for r in results if r.file and r.file.endswith("a.py")), None)
    res2 = next((r for r in results if r.file and r.file.endswith("b.py")), None)
    assert res1 is not None and res2 is not None
    # File1 has a loop with range 100, should flag large loop inefficiency
    issues1 = res1.inefficiencies
    assert any("Loop iterating 100" in issue for issue in issues1)
    # File2 has just a print, no inefficiency
    assert res2.inefficiencies == []
