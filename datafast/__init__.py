"""Datafast - A Python package for synthetic text dataset generation"""

import importlib.metadata
from datafast.logger_config import configure_logger

try:
    __version__ = importlib.metadata.version("datafast")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass

def get_version():
    """Return the current version of the datafast package."""
    return __version__

__all__ = ["configure_logger", "get_version"]
