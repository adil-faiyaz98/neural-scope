"""Setup script for the advanced_analysis package."""

import os
from setuptools import setup, find_packages

# Get version from version.py
version = {}
with open(os.path.join("advanced_analysis", "version.py")) as f:
    exec(f.read(), version)

# Read long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Advanced analysis tools for machine learning models and data"

setup(
    name="neural-scope",
    version=version["__version__"],
    description="Advanced analysis tools for machine learning models and data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adil Faiyaz",
    author_email="adilmd98@gmail.com",
    url="https://github.com/adil-faiyaz98/neural-scope",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "psutil>=5.8.0",
        "memory-profiler>=0.58.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.7.0"],
        "tensorflow": ["tensorflow>=2.4.0"],
        "visualization": ["plotly>=4.14.0", "dash>=1.19.0", "seaborn>=0.11.0"],
        "distributed": ["dask>=2.0.0", "ray>=1.0.0"],
        "compression": ["tensorflow-model-optimization>=0.5.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "bandit>=1.7.0",
            "safety>=1.10.0",
        ],
        "docs": [
            "sphinx>=3.4.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.11.0",
        ],
        "all": [
            "torch>=1.7.0",
            "tensorflow>=2.4.0",
            "plotly>=4.14.0",
            "dash>=1.19.0",
            "seaborn>=0.11.0",
            "dask>=2.0.0",
            "ray>=1.0.0",
            "tensorflow-model-optimization>=0.5.0",
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
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "neural-scope=advanced_analysis.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
