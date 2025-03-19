# algorithm_recognition.py
import ast
import re
from aiml_complexity.ml_patterns import MLPatternDatabase

class MLAlgorithmRecognizer:
    def __init__(self, code_str):
        self.code_str = code_str
        try:
            self.tree = ast.parse(self.code_str)
        except:
            self.tree = None

    def identify_algorithms(self):
        found_algos = []
        for algo_name, info in MLPatternDatabase.items():
            matched_patterns = [pat for pat in info["patterns"] if re.search(pat, self.code_str, flags=re.IGNORECASE)]
            if matched_patterns:
                found_algos.append((algo_name, matched_patterns))
        return found_algos
