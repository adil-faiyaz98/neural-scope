"""
Error handling, logging, and version tracking utilities for the advanced_analysis framework.

This module provides centralized error handling, custom exceptions, logging
functionality, and version tracking to ensure robust operation and graceful degradation
across the framework.
"""

# Version information
__version__ = "0.1.0"
__version_info__ = tuple(map(int, __version__.split(".")))

import logging
import sys
import traceback
import functools
import inspect
import os
import json
import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar

# Configure root logger
logger = logging.getLogger("advanced_analysis")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Define custom exceptions
class AdvancedAnalysisError(Exception):
    """Base exception for all advanced_analysis errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DependencyError(AdvancedAnalysisError):
    """Exception raised when a required dependency is missing."""

    def __init__(self, dependency: str, message: Optional[str] = None):
        self.dependency = dependency
        details = {"dependency": dependency}
        message = message or f"Required dependency '{dependency}' is not installed"
        super().__init__(message, details)


class FrameworkError(AdvancedAnalysisError):
    """Exception raised when there's an issue with a specific ML framework."""

    def __init__(self, framework: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.framework = framework
        framework_details = {"framework": framework}
        if details:
            framework_details.update(details)
        super().__init__(message, framework_details)


class ModelError(AdvancedAnalysisError):
    """Exception raised when there's an issue with a model."""

    def __init__(self, model_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        model_details = {"model_type": model_type}
        if details:
            model_details.update(details)
        super().__init__(message, model_details)


class DataError(AdvancedAnalysisError):
    """Exception raised when there's an issue with data."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class AnalysisError(AdvancedAnalysisError):
    """Exception raised when analysis fails."""

    def __init__(self, analysis_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.analysis_type = analysis_type
        analysis_details = {"analysis_type": analysis_type}
        if details:
            analysis_details.update(details)
        super().__init__(message, analysis_details)


class HardwareError(AdvancedAnalysisError):
    """Exception raised when there's an issue with hardware detection or optimization."""

    def __init__(self, hardware_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.hardware_type = hardware_type
        hardware_details = {"hardware_type": hardware_type}
        if details:
            hardware_details.update(details)
        super().__init__(message, hardware_details)


# Define error handler decorator
F = TypeVar('F', bound=Callable[..., Any])

def handle_errors(
    fallback_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False,
    expected_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """
    Decorator for handling exceptions in functions.

    Args:
        fallback_return: Value to return if an exception occurs
        log_level: Logging level for exceptions
        reraise: Whether to reraise the exception after handling
        expected_exceptions: List of exception types to handle specially

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function details for better logging
                module = func.__module__
                qualname = func.__qualname__

                # Get caller information
                frame = inspect.currentframe()
                if frame:
                    caller_frame = frame.f_back
                    if caller_frame:
                        caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
                    else:
                        caller_info = "unknown"
                else:
                    caller_info = "unknown"

                # Format error message
                error_message = f"Error in {module}.{qualname}: {str(e)}"

                # Get traceback
                tb = traceback.format_exc()

                # Log the error
                log_func = getattr(logger, logging.getLevelName(log_level).lower())
                log_func(error_message)
                log_func(f"Traceback: {tb}")
                log_func(f"Called from: {caller_info}")

                # Handle expected exceptions differently if specified
                if expected_exceptions and any(isinstance(e, exc_type) for exc_type in expected_exceptions):
                    for exc_type in expected_exceptions:
                        if isinstance(e, exc_type):
                            log_func(f"Handling expected exception type: {exc_type.__name__}")
                            break

                # Reraise if specified
                if reraise:
                    raise

                # Return fallback value
                return fallback_return

        return wrapper  # type: ignore

    return decorator


# Define dependency checker
def check_dependency(dependency: str) -> bool:
    """
    Check if a dependency is installed.

    Args:
        dependency: Name of the dependency to check

    Returns:
        True if the dependency is installed, False otherwise
    """
    try:
        __import__(dependency)
        return True
    except ImportError:
        return False


def require_dependency(dependency: str, message: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to require a dependency for a function.

    Args:
        dependency: Name of the dependency to require
        message: Optional custom error message

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not check_dependency(dependency):
                error_msg = message or f"Required dependency '{dependency}' is not installed"
                logger.error(error_msg)
                raise DependencyError(dependency, error_msg)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Define error recovery utilities
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    def can_recover(self, exception: Exception) -> bool:
        """
        Check if this strategy can recover from the exception.

        Args:
            exception: The exception to check

        Returns:
            True if this strategy can recover from the exception, False otherwise
        """
        return False

    def recover(self, exception: Exception, **context: Any) -> Any:
        """
        Attempt to recover from the exception.

        Args:
            exception: The exception to recover from
            **context: Additional context for recovery

        Returns:
            Recovery result or None if recovery failed
        """
        return None


class DependencyRecoveryStrategy(ErrorRecoveryStrategy):
    """Strategy for recovering from missing dependencies."""

    def can_recover(self, exception: Exception) -> bool:
        return isinstance(exception, (DependencyError, ImportError))

    def recover(self, exception: Exception, **context: Any) -> Any:
        if isinstance(exception, DependencyError):
            dependency = exception.dependency
        elif isinstance(exception, ImportError):
            # Extract module name from ImportError
            error_msg = str(exception)
            if "No module named" in error_msg:
                dependency = error_msg.split("'")[1]
            else:
                dependency = "unknown"
        else:
            return None

        logger.warning(f"Attempting to recover from missing dependency: {dependency}")

        # Suggest installation
        logger.info(f"You can install {dependency} with: pip install {dependency}")

        # Return fallback if provided
        if "fallback" in context:
            logger.info(f"Using fallback for {dependency}")
            return context["fallback"]

        return None


class FrameworkRecoveryStrategy(ErrorRecoveryStrategy):
    """Strategy for recovering from framework-specific errors."""

    def can_recover(self, exception: Exception) -> bool:
        return isinstance(exception, FrameworkError)

    def recover(self, exception: Exception, **context: Any) -> Any:
        if not isinstance(exception, FrameworkError):
            return None

        framework = exception.framework
        logger.warning(f"Attempting to recover from {framework} error: {exception.message}")

        # Try alternative framework if provided
        if "alternative_framework" in context and "alternative_function" in context:
            alt_framework = context["alternative_framework"]
            alt_function = context["alternative_function"]
            logger.info(f"Trying alternative framework: {alt_framework}")

            try:
                return alt_function()
            except Exception as e:
                logger.error(f"Alternative framework {alt_framework} also failed: {str(e)}")

        return None


class ModelRecoveryStrategy(ErrorRecoveryStrategy):
    """Strategy for recovering from model-specific errors."""

    def can_recover(self, exception: Exception) -> bool:
        return isinstance(exception, ModelError)

    def recover(self, exception: Exception, **context: Any) -> Any:
        if not isinstance(exception, ModelError):
            return None

        model_type = exception.model_type
        logger.warning(f"Attempting to recover from model error for {model_type}: {exception.message}")

        # Try simplified model if provided
        if "simplified_model" in context:
            logger.info(f"Trying simplified model for {model_type}")
            return context["simplified_model"]

        return None


class DataRecoveryStrategy(ErrorRecoveryStrategy):
    """Strategy for recovering from data-specific errors."""

    def can_recover(self, exception: Exception) -> bool:
        return isinstance(exception, DataError)

    def recover(self, exception: Exception, **context: Any) -> Any:
        if not isinstance(exception, DataError):
            return None

        logger.warning(f"Attempting to recover from data error: {exception.message}")

        # Try with sample data if provided
        if "sample_data" in context:
            logger.info("Using sample data as fallback")
            return context["sample_data"]

        return None


class HardwareRecoveryStrategy(ErrorRecoveryStrategy):
    """Strategy for recovering from hardware-specific errors."""

    def can_recover(self, exception: Exception) -> bool:
        return isinstance(exception, HardwareError)

    def recover(self, exception: Exception, **context: Any) -> Any:
        if not isinstance(exception, HardwareError):
            return None

        hardware_type = exception.hardware_type
        logger.warning(f"Attempting to recover from hardware error for {hardware_type}: {exception.message}")

        # Try with CPU fallback if GPU error
        if hardware_type.lower() == "gpu" and "cpu_fallback" in context:
            logger.info("Falling back to CPU execution")
            return context["cpu_fallback"]

        return None


class ErrorRecoveryManager:
    """Manager for error recovery strategies."""

    def __init__(self):
        self.strategies: List[ErrorRecoveryStrategy] = [
            DependencyRecoveryStrategy(),
            FrameworkRecoveryStrategy(),
            ModelRecoveryStrategy(),
            DataRecoveryStrategy(),
            HardwareRecoveryStrategy()
        ]

    def add_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """
        Add a recovery strategy.

        Args:
            strategy: The strategy to add
        """
        self.strategies.append(strategy)

    def attempt_recovery(self, exception: Exception, **context: Any) -> Optional[Any]:
        """
        Attempt to recover from an exception using available strategies.

        Args:
            exception: The exception to recover from
            **context: Additional context for recovery

        Returns:
            Recovery result or None if recovery failed
        """
        for strategy in self.strategies:
            if strategy.can_recover(exception):
                logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                result = strategy.recover(exception, **context)
                if result is not None:
                    logger.info(f"Recovery successful with {strategy.__class__.__name__}")
                    return result

        logger.warning("All recovery strategies failed")
        return None


# Create global error recovery manager
error_recovery_manager = ErrorRecoveryManager()


def with_recovery(func: F) -> F:
    """
    Decorator to add error recovery to a function.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with error recovery
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__module__}.{func.__qualname__}: {str(e)}")
            logger.info("Attempting recovery...")

            # Extract context from args and kwargs for recovery
            context = {}

            # Add function arguments to context
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            for param_name, param_value in bound_args.arguments.items():
                context[param_name] = param_value

            # Attempt recovery
            result = error_recovery_manager.attempt_recovery(e, **context)
            if result is not None:
                return result

            # If recovery failed, reraise the exception
            raise

    return wrapper  # type: ignore


# Define logging utilities
class LoggingContext:
    """Context manager for temporarily changing logging level."""

    def __init__(self, logger_name: Optional[str] = None, level: Optional[int] = None):
        self.logger = logging.getLogger(logger_name) if logger_name else logger
        self.level = level
        self.old_level = self.logger.level

    def __enter__(self) -> 'LoggingContext':
        if self.level is not None:
            self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.logger.setLevel(self.old_level)


def configure_file_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Configure logging to a file.

    Args:
        log_file: Path to the log file
        level: Logging level
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")


def log_execution_time(func: F) -> F:
    """
    Decorator to log function execution time.

    Args:
        func: The function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"{func.__module__}.{func.__qualname__} executed in {execution_time:.4f} seconds")
        return result

    return wrapper  # type: ignore


def log_call(log_args: bool = True, log_result: bool = False) -> Callable[[F], F]:
    """
    Decorator to log function calls.

    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function result

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function call
            if log_args:
                # Format args and kwargs for logging
                args_str = ", ".join(repr(arg) for arg in args)
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                params_str = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.info(f"Calling {func_name}({params_str})")
            else:
                logger.info(f"Calling {func_name}")

            # Call function
            result = func(*args, **kwargs)

            # Log result
            if log_result:
                logger.info(f"{func_name} returned: {repr(result)}")
            else:
                logger.info(f"{func_name} completed")

            return result

        return wrapper  # type: ignore

    return decorator
