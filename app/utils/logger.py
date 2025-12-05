"""
Logging configuration module for Golden Recipe API.

This module provides a lightweight logging enhancement that works alongside
the dnd_seccommons library's logging configuration:

- Colored console output in development (ENV=dev)
- Deduplication of log handlers to prevent duplicate log lines
- Removal of SQLAlchemy's self-added handlers (which cause duplication)
- Performance timing decorators for development/QA environments

The module does NOT replace the dnd_seccommons logging configuration,
it only adds color formatting in development and prevents handler duplication.
"""

import logging
import logging.config
import sys
import os
import json
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime, timezone


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for log levels and logger names in development environments."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    GREY = "\033[90m"  # Grey for logger names
    RESET = "\033[0m"

    def formatTime(self, record, datefmt=None):
        """Override to include milliseconds in the timestamp."""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        # Add milliseconds
        s = f"{s},{int(record.msecs):03d}"
        return s

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colored log level and logger name."""
        # Save original values
        original_levelname = record.levelname
        original_name = record.name

        # Add color to levelname
        log_color = self.COLORS.get(record.levelname, "")
        if log_color:
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"

        # Add grey color to logger name
        record.name = f"{self.GREY}{record.name}{self.RESET}"

        # Format the message (levelname and name are now colored)
        formatted = super().format(record)

        # Restore original values for next handler
        record.levelname = original_levelname
        record.name = original_name

        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production environments."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add correlation_id if present (prepared for observability integration)
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def _get_log_level() -> str:
    """
    Get log level from environment variable.

    Returns:
        str: Logging level name (defaults to INFO if not set or invalid)
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()


def _get_environment() -> str:
    """
    Get current environment from ENV variable.

    Returns:
        str: Environment name (dev, qa, prod, etc.)
    """
    return os.getenv("ENV", "prod").lower()


def configure_sqlalchemy_loggers() -> None:
    """
    Prevent SQLAlchemy from adding duplicate handlers.

    SQLAlchemy dynamically adds StreamHandlers when creating database connections,
    which causes duplicate log lines. We prevent this by monkey-patching addHandler
    on SQLAlchemy loggers to be a no-op. Logs still work via propagation to root.
    """
    def _no_op_add_handler(self, handler):
        """No-op override to prevent SQLAlchemy from adding handlers."""
        pass

    for sa_logger_name in [
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.engine.Engine",
        "sqlalchemy.pool",
    ]:
        sa_logger = logging.getLogger(sa_logger_name)
        sa_logger.handlers = []  # Clear any existing handlers
        sa_logger.propagate = True  # Ensure logs propagate to root logger
        # Override addHandler method to prevent future handler additions
        sa_logger.addHandler = _no_op_add_handler.__get__(sa_logger, logging.Logger)


def setup_logging() -> None:
    """
    Apply colored formatting to console handlers in development environment.

    This function enhances the logging configuration from dnd_seccommons by:
    1. Adding colored output to existing handlers (dev only)
    2. Removing duplicate StreamHandlers from root logger
    3. Removing handlers from SQLAlchemy loggers to prevent duplication

    Background on SQLAlchemy handler duplication:
    - SQLAlchemy dynamically adds its own StreamHandler when creating connections
    - These handlers output the same logs that propagate to root logger
    - This causes duplicate log lines (one from SQLAlchemy, one from root)
    - We prevent this by removing SQLAlchemy handlers while keeping propagate=True

    Note: This function is called:
    - Once at application startup (after dnd_seccommons initializes)
    - By middleware after each request (to catch dynamically-added handlers)
    """
    env = _get_environment()

    # Only apply colored formatter in dev environment
    if env != "dev":
        return

    # Use the same format as dnd_seccommons but with ANSI color codes
    colored_formatter = ColoredFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Deduplicate and colorize root logger handlers
    root_logger = logging.getLogger()

    # Separate StreamHandlers from FileHandlers
    stream_handlers = []
    other_handlers = []

    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            stream_handlers.append(handler)
        else:
            other_handlers.append(handler)

    # Clear all handlers
    root_logger.handlers.clear()

    # Keep only the FIRST StreamHandler and apply colored formatter
    if stream_handlers:
        stream_handlers[0].setFormatter(colored_formatter)
        root_logger.addHandler(stream_handlers[0])

    # Re-add non-stream handlers (e.g., FileHandlers) without modification
    for handler in other_handlers:
        root_logger.addHandler(handler)

    # Remove handlers from SQLAlchemy loggers to prevent duplication
    # SQLAlchemy loggers will still work via propagation to root logger
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("sqlalchemy"):
            logger = logging.getLogger(logger_name)
            if isinstance(logger, logging.Logger) and logger.handlers:
                logger.handlers.clear()
                # propagate=True ensures logs still reach root logger with colors


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name.

    This is the recommended way to create loggers throughout the application.
    It ensures consistent configuration and prevents log duplication.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)


def log_execution_time(func: Optional[Callable] = None, *, level: str = "DEBUG") -> Callable:
    """
    Decorator to log function execution time.

    Only active in dev and qa environments for performance monitoring.
    In production, this decorator does nothing to avoid overhead.

    Args:
        func: Function to decorate
        level: Log level for timing message (default: DEBUG)

    Returns:
        Callable: Decorated function

    Example:
        >>> @log_execution_time
        ... def fetch_data():
        ...     # expensive operation
        ...     pass

        >>> @log_execution_time(level="INFO")
        ... def critical_operation():
        ...     pass
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            env = _get_environment()

            # Only measure in dev/qa environments
            if env not in ("dev", "qa"):
                return f(*args, **kwargs)

            logger = get_logger(f.__module__)
            start_time = time.perf_counter()

            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                log_method = getattr(logger, level.lower(), logger.debug)
                log_method(
                    f"Function '{f.__name__}' executed in {elapsed:.4f}s"
                )

        return wrapper

    # Allow usage with or without parentheses
    if func is None:
        return decorator
    return decorator(func)
