import os
import tempfile
import json
from aiml_complexity import storage, analysis

def test_save_and_load_analysis_single():
    code = "x = 1\ny = 2\n"
    result = analysis.analyze_code(code)
    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_name = tmp_file.name
    tmp_file.close()
    try:
        storage.save_analysis(result, tmp_name)
        # Load it back
        data = storage.load_analysis(tmp_name)
        # The loaded data should be a dict matching the result's asdict
        from dataclasses import asdict
        expected = asdict(result)
        assert isinstance(data, dict)
        # Remove file path from comparison if it was None
        if expected.get("file") is None and "file" in data:
            # The saved JSON will have file: null vs asdict also None (Python None -> null in JSON, so should match anyway)
            pass
        assert data == expected
    finally:
        os.remove(tmp_name)

def test_save_and_load_analysis_list():
    code1 = "for i in range(3):\n    pass\n"
    code2 = "print('hi')\n"
    result1 = analysis.analyze_code(code1, filename="code1")
    result2 = analysis.analyze_code(code2, filename="code2")
    results = [result1, result2]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_name = tmp_file.name
    tmp_file.close()
    try:
        storage.save_analysis(results, tmp_name)
        data = storage.load_analysis(tmp_name)
        # Should be a list of dicts
        assert isinstance(data, list)
        assert len(data) == 2
        files = {item.get("file") for item in data}
        assert "code1" in files and "code2" in files
    finally:
        os.remove(tmp_name)
