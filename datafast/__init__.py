"""Datafast - A Python package for synthetic text dataset generation"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("datafast")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass

def get_version():
    """Return the current version of the datafast package."""
    return __version__
