# src/pgs_net/utils/logging_utils.py
"""Logging setup utilities."""

import logging
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Keep track of configured loggers to avoid adding multiple handlers
_configured_loggers = set()


def get_logger(
    name: str, level: int = DEFAULT_LOG_LEVEL, format: str = DEFAULT_LOG_FORMAT, force_reload: bool = False
) -> logging.Logger:
    """
    Gets and configures a logger instance.

    Args:
        name (str): Name of the logger (e.g., __name__).
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        format (str): Logging format string.
        force_reload (bool): If True, removes existing handlers and reconfigures.

    Returns:
        logging.Logger: Configured logger instance.

    """
    logger = logging.getLogger(name)

    if name in _configured_loggers and not force_reload:
        # Logger already configured, just ensure level is set
        logger.setLevel(level)
        return logger

    # Remove existing handlers if force_reload or reconfiguring
    if force_reload:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()  # Close file handlers etc.

    # Configure logger if it has no handlers or forced reload
    if not logger.handlers or force_reload:
        logger.setLevel(level)
        formatter = logging.Formatter(format)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optionally add File Handler
        # fh = logging.FileHandler(f"{name}.log")
        # fh.setLevel(level)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

        logger.propagate = False  # Prevent propagation to root logger if handlers are added

    _configured_loggers.add(name)
    # logger.debug(f"Logger '{name}' configured with level {logging.getLevelName(level)}.")
    return logger


# Example basic setup for the root logger if needed, but per-module is often better
# logging.basicConfig(level=DEFAULT_LOG_LEVEL, format=DEFAULT_LOG_FORMAT)
