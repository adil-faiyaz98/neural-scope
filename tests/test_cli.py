import os
import sys
import builtins
import pytest
from aiml_complexity import cli

def test_cli_single_file_output(tmp_path, capsys):
    # Create a sample Python file with a known inefficiency
    code = "for i in range(2):\n    for j in range(2):\n        x = i*j\n"
    file_path = tmp_path / "sample.py"
    file_path.write_text(code)
    # Run CLI on this file
    cli.main([str(file_path)])
    captured = capsys.readouterr()
    out = captured.out
    # Check that output contains expected sections
    assert "Complexity Score" in out
    assert "Estimated Operations" in out
    assert "Inefficiencies" in out
    assert "Nested loops detected" in out
    assert "Estimated cost on m5.large" in out

def test_cli_nonexistent_path(capsys):
    # Run CLI with a path that doesn't exist
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["nonexistent_file.py"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    err = captured.err + captured.out
    # Should log an error about path not found
    assert "Path not found" in err or "not found" in err

def test_cli_different_instance(tmp_path, capsys):
    # Create a sample file and use a different instance for cost
    code = "x = 0\nfor i in range(100000):\n    x += i\n"
    file_path = tmp_path / "loop.py"
    file_path.write_text(code)
    # Run CLI with t2.micro instance
    cli.main(["--instance", "t2.micro", str(file_path)])
    captured = capsys.readouterr()
    out = captured.out
    # The output should mention the t2.micro in cost estimation
    assert "Estimated cost on t2.micro" in out
