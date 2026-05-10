"""
Structured logging configuration for ReHub.
Uses Python's stdlib logging with a JSON-friendly format in production.
"""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Quieten noisy libraries
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
