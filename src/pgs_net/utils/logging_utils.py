# src/pgs_net/utils/logging_utils.py
"""Logging setup utilities."""

import logging
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Keep track of configured loggers to avoid adding multiple handlers
_configured_loggers = set()


def get_logger(
    name: str,
    level: Optional[int] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    force_reload: bool = False,
) -> logging.Logger:
    """
    Gets and configures a logger instance with console and optional file handlers.

    Args:
        name (str): Name of the logger (e.g., __name__).
        level (Optional[int]): Logging level override (e.g., logging.INFO, logging.DEBUG).
                               Defaults to DEFAULT_LOG_LEVEL if not set and logger is new.
        log_format (str): Logging format string.
        log_file (Optional[str]): Path to an optional log file.
        force_reload (bool): If True, removes existing handlers and reconfigures.

    Returns:
        logging.Logger: Configured logger instance.

    """
    logger = logging.getLogger(name)
    effective_level = level if level is not None else DEFAULT_LOG_LEVEL

    # Avoid reconfiguring if already done and not forced
    if name in _configured_loggers and not force_reload:
        # Still allow updating the level if a new level is specified
        if level is not None:
            logger.setLevel(effective_level)
            # Update handler levels too if needed
            for handler in logger.handlers:
                handler.setLevel(effective_level)
        return logger

    # Remove existing handlers if force_reload or reconfiguring for the first time
    if force_reload or name not in _configured_loggers:
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass  # Ignore errors during close
            logger.removeHandler(handler)

    # Configure logger if it has no handlers or forced reload
    if not logger.handlers or force_reload:
        logger.setLevel(effective_level)
        formatter = logging.Formatter(log_format)

        # Console Handler (always add if no handlers present)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(effective_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional File Handler
        if log_file:
            try:
                fh = logging.FileHandler(log_file, mode="a")  # Append mode
                fh.setLevel(effective_level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                logger.error(f"Failed to create file handler for {log_file}: {e}")

        logger.propagate = False  # Prevent duplicate messages in parent loggers

    _configured_loggers.add(name)
    if force_reload:
        logger.info(f"Logger '{name}' reconfigured.")
    # else: logger.debug(f"Logger '{name}' configured.") # Can be verbose
    return logger


def set_global_log_level(level: int):
    """Sets the logging level for all known loggers managed by get_logger."""
    global DEFAULT_LOG_LEVEL
    DEFAULT_LOG_LEVEL = level
    logging.getLogger().setLevel(level)  # Set root logger level
    for name in _configured_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    logger.info(f"Global log level set to {logging.getLevelName(level)}")
