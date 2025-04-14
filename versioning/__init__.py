"""
Neural-Scope Model Versioning Module

This module provides tools for model versioning and promotion:
- Model registry
- Version tracking
- Model promotion
- A/B testing
"""

from versioning.model_registry import ModelRegistry
from versioning.version_manager import VersionManager

__all__ = [
    'ModelRegistry',
    'VersionManager'
]
