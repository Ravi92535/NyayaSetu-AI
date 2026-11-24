import logging
import sys
from typing import Optional


def setup_logger(name: str = "rag_project", level: Optional[str] = None) -> logging.Logger:
    """
    Sets up a structured, human-readable console logger.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Global application logger
logger = setup_logger()

__all__ = ["setup_logger", "logger"]

