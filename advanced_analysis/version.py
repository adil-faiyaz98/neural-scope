"""
Neural-Scope Version Information
"""

__version__ = "0.3.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))

def get_version():
    """Get the current version of Neural-Scope."""
    return __version__

def get_version_info():
    """Get the current version info of Neural-Scope as a tuple."""
    return __version_info__
