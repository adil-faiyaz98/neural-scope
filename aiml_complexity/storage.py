"""
Module: storage.py

Provides functions to save analysis results to disk and load them back.
Uses JSON format for interoperability.
"""
import json
import logging
from pathlib import Path
from typing import Union, List
from dataclasses import is_dataclass, asdict

logger = logging.getLogger(__name__)

def save_analysis(result: Union[dict, list, object], filename: str) -> None:
    """
    Save the analysis result (or list of results) to a JSON file.
    Accepts dataclass objects (AnalysisResult) or dictionaries.
    Overwrites the file if it exists.
    """
    try:
        path = Path(filename)
        data_to_save = result
        # If result is a dataclass or list of dataclasses, convert to dict
        if is_dataclass(result):
            data_to_save = asdict(result)
        elif isinstance(result, list):
            new_list = []
            for item in result:
                if is_dataclass(item):
                    new_list.append(asdict(item))
                else:
                    new_list.append(item)
            data_to_save = new_list
        # Write JSON
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"Analysis result saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save analysis to {filename}: {e}")
        raise

def load_analysis(filename: str) -> Union[dict, list]:
    """
    Load analysis result(s) from a JSON file. Returns a dictionary or list of dictionaries.
    """
    try:
        path = Path(filename)
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Analysis result loaded from {filename}")
        return data
    except Exception as e:
        logger.error(f"Failed to load analysis from {filename}: {e}")
        raise
