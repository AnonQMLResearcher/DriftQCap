"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional



def configure_logging(level: int = logging.INFO, logger_name: Optional[str] = None) -> logging.Logger:
    """Configure and return a project logger."""
    name = logger_name or "driftqcap"
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
