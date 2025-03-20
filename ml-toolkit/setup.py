from setuptools import setup, find_packages

setup(
    name="ml-toolkit",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "shap>=0.40.0",
        "optuna>=2.10.0",
        "py-spy>=0.3.0",
        "memory-profiler>=0.60.0",
        "pyinstrument>=4.0.0"
    ],
    entry_points="""
        [console_scripts]
        dataguard=ml_toolkit.dataguard.cli:cli
        optimizer=ml_toolkit.optimizer.cli:cli
        model-trainer=ml_toolkit.trainer.cli:cli
    """,
    python_requires=">=3.8",
    author="Neural Scope Team",
    author_email="team@neural-scope.ai",
    description="Advanced ML toolkit for data quality, model optimization and training",
    keywords="machine learning, data quality, optimization",
    url="https://github.com/yourusername/neural-scope",
)