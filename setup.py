#!/usr/bin/env python
"""
Setup script for Neural-Scope.
"""

import os
from setuptools import setup, find_packages

# Get version from version.py
version = {}
with open(os.path.join("advanced_analysis", "version.py")) as f:
    exec(f.read(), version)

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neural-scope",
    version=version["__version__"],
    description="A comprehensive tool for analyzing and optimizing machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adil Faiyaz",
    author_email="adilmd98@gmail.com",
    url="https://github.com/adil-faiyaz98/neural-scope",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.23.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.3.0",
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "jsonschema>=3.2.0"
    ],
    extras_require={
        "pytorch": [
            "torch>=1.7.0",
            "torchvision>=0.8.0"
        ],
        "tensorflow": [
            "tensorflow>=2.4.0",
            "tensorflow-hub>=0.10.0"
        ],
        "huggingface": [
            "transformers>=4.5.0"
        ],
        "mlflow": [
            "mlflow>=1.15.0"
        ],
        "onnx": [
            "onnx>=1.8.0",
            "onnxruntime>=1.7.0"
        ],
        "aws": [
            "boto3>=1.17.0",
            "sagemaker>=2.35.0"
        ],
        "security": [
            "foolbox>=3.3.0",
            "adversarial-robustness-toolbox>=1.9.0"
        ],
        "all": [
            "torch>=1.7.0",
            "torchvision>=0.8.0",
            "tensorflow>=2.4.0",
            "tensorflow-hub>=0.10.0",
            "transformers>=4.5.0",
            "mlflow>=1.15.0",
            "onnx>=1.8.0",
            "onnxruntime>=1.7.0",
            "boto3>=1.17.0",
            "sagemaker>=2.35.0",
            "foolbox>=3.3.0",
            "adversarial-robustness-toolbox>=1.9.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "neural-scope=advanced_analysis.cli:main",
            "neural-scope-analyze=advanced_analysis.cli:analyze_command",
            "neural-scope-optimize=advanced_analysis.cli:optimize_command",
            "neural-scope-security=security.cli:security_command",
            "neural-scope-robustness=security.cli:robustness_command",
            "neural-scope-mlflow=advanced_analysis.mlflow_integration:mlflow_command",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    keywords="machine learning, deep learning, model optimization, model analysis, CI/CD, MLflow, security, robustness, adversarial, model registry",
)
