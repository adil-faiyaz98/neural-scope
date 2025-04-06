"""
Utility modules for the advanced_analysis framework.

This package contains utility modules for error handling, hardware detection,
and other common functionality used across the framework.
"""

from advanced_analysis.utils.error_handling import (
    # Custom exceptions
    AdvancedAnalysisError,
    DependencyError,
    FrameworkError,
    ModelError,
    DataError,
    AnalysisError,
    HardwareError,
    
    # Decorators
    handle_errors,
    require_dependency,
    with_recovery,
    log_execution_time,
    log_call,
    
    # Utilities
    check_dependency,
    configure_file_logging,
    LoggingContext,
    
    # Error recovery
    ErrorRecoveryManager,
    error_recovery_manager
)

from advanced_analysis.utils.hardware_utils import (
    detect_hardware,
    get_optimal_batch_size,
    get_optimal_num_workers,
    is_gpu_available,
    is_tpu_available,
    get_recommended_precision,
    optimize_for_hardware
)

__all__ = [
    # Error handling
    'AdvancedAnalysisError',
    'DependencyError',
    'FrameworkError',
    'ModelError',
    'DataError',
    'AnalysisError',
    'HardwareError',
    'handle_errors',
    'require_dependency',
    'with_recovery',
    'log_execution_time',
    'log_call',
    'check_dependency',
    'configure_file_logging',
    'LoggingContext',
    'ErrorRecoveryManager',
    'error_recovery_manager',
    
    # Hardware utilities
    'detect_hardware',
    'get_optimal_batch_size',
    'get_optimal_num_workers',
    'is_gpu_available',
    'is_tpu_available',
    'get_recommended_precision',
    'optimize_for_hardware'
]
