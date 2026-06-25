"""Structured logging configuration for the project."""

import logging
import os
import sys
from datetime import datetime

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers: dict = {}


def get_logger(name: str = "esl_completion", level: str = None) -> logging.Logger:
    """
    Get or create a configured logger.

    Args:
        name: Logger name (usually module name).
        level: Log level override. Defaults to env var LOG_LEVEL or INFO.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    log_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger


def setup_file_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Add file handler to the root project logger."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"esl_completion_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    root_logger = get_logger()
    root_logger.addHandler(file_handler)
