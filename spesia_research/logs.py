"""
Centralized logging configuration for the package.

Features:
- Timestamped logs
- Log level
- Module name and line number
- Single configuration point
- Safe against double-handler issues
"""

from __future__ import annotations
import logging
import sys
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    stream: Optional[object] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    Configure global logging for the package.

    This should be called **once**, ideally at application startup.

    Parameters:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        stream (object, optional): Output stream (defaults to sys.stdout)
        log_format (str): Logging format string
        date_format (str): Timestamp format

    Returns:
        None
    """
    stream = stream or sys.stdout

    root_logger = logging.getLogger()

    # Prevent duplicated handlers in notebooks / reloads
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with module awareness.

    Usually called as:
        logger = get_logger(__name__)

    Args:
        name (str): name of the logger. Defaults to module name.

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)
