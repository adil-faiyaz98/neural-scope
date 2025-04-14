"""
Neural-Scope Security Module

This module provides tools for security analysis of machine learning models:
- Vulnerability detection
- Adversarial robustness testing
- Privacy risk assessment
- Model backdoor detection
"""

from security.vulnerability_detector import VulnerabilityDetector
from security.adversarial_tester import AdversarialTester
from security.privacy_analyzer import PrivacyAnalyzer
from security.backdoor_detector import BackdoorDetector

__all__ = [
    'VulnerabilityDetector',
    'AdversarialTester',
    'PrivacyAnalyzer',
    'BackdoorDetector'
]
