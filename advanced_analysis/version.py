"""
Version tracking for the advanced_analysis package.

This module provides version information and tracking for the entire package
and its components.
"""

# Package version
__version__ = "0.1.0"
__version_info__ = tuple(map(int, __version__.split(".")))

# Component versions
COMPONENT_VERSIONS = {
    "core": "0.1.0",
    "algorithm_complexity": "0.1.0",
    "data_quality": "0.1.0",
    "ml_advisor": "0.1.0",
    "performance": "0.1.0",
    "utils": "0.1.0",
    "visualization": "0.1.0"
}

# Dependency requirements
DEPENDENCIES = {
    "required": [
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0"
    ],
    "optional": {
        "pytorch": "torch>=1.7.0",
        "tensorflow": "tensorflow>=2.3.0",
        "visualization": [
            "plotly>=4.0.0",
            "seaborn>=0.11.0"
        ],
        "distributed": [
            "dask>=2.0.0",
            "ray>=1.0.0"
        ],
        "compression": [
            "tensorflow-model-optimization>=0.5.0"
        ]
    }
}

def get_version():
    """
    Get the current version of the package.
    
    Returns:
        str: Version string
    """
    return __version__

def get_component_version(component):
    """
    Get the version of a specific component.
    
    Args:
        component (str): Component name
        
    Returns:
        str: Component version
    """
    return COMPONENT_VERSIONS.get(component, "unknown")

def get_version_info():
    """
    Get detailed version information for the package and all components.
    
    Returns:
        dict: Version information
    """
    return {
        "package": __version__,
        "components": COMPONENT_VERSIONS,
        "dependencies": DEPENDENCIES
    }

def check_compatibility(component1, component2):
    """
    Check if two components are compatible with each other.
    
    Args:
        component1 (str): First component name
        component2 (str): Second component name
        
    Returns:
        bool: True if compatible, False otherwise
    """
    # Get component versions
    version1 = get_component_version(component1)
    version2 = get_component_version(component2)
    
    # Parse versions
    v1_info = tuple(map(int, version1.split(".")))
    v2_info = tuple(map(int, version2.split(".")))
    
    # Check compatibility (major versions must match)
    return v1_info[0] == v2_info[0]

def check_dependency_versions():
    """
    Check if installed dependencies meet version requirements.
    
    Returns:
        dict: Dependency check results
    """
    import pkg_resources
    
    results = {
        "required": {},
        "optional": {}
    }
    
    # Check required dependencies
    for dep in DEPENDENCIES["required"]:
        name, version_spec = dep.split(">=")
        try:
            installed_version = pkg_resources.get_distribution(name).version
            meets_requirement = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(version_spec)
            results["required"][name] = {
                "required": version_spec,
                "installed": installed_version,
                "meets_requirement": meets_requirement
            }
        except pkg_resources.DistributionNotFound:
            results["required"][name] = {
                "required": version_spec,
                "installed": None,
                "meets_requirement": False
            }
    
    # Check optional dependencies
    for category, deps in DEPENDENCIES["optional"].items():
        results["optional"][category] = {}
        if isinstance(deps, str):
            deps = [deps]
        for dep in deps:
            name, version_spec = dep.split(">=")
            try:
                installed_version = pkg_resources.get_distribution(name).version
                meets_requirement = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(version_spec)
                results["optional"][category][name] = {
                    "required": version_spec,
                    "installed": installed_version,
                    "meets_requirement": meets_requirement
                }
            except pkg_resources.DistributionNotFound:
                results["optional"][category][name] = {
                    "required": version_spec,
                    "installed": None,
                    "meets_requirement": False
                }
    
    return results
