"""
AI TOOLBOX - Models Module
==========================

Model management, downloading, and library functionality.
"""

from .types import (
    ModelFormat,
    ModelSource,
    ModelCategory,
    ModelEntry,
    ModelSearchResult,
    ModelDetails,
    LoRAInfo,
    OllamaModelInfo,
    ModelTreeNode,
)

from .library import ModelLibrary

from .downloader import ModelDownloader

__all__ = [
    # Types
    'ModelFormat',
    'ModelSource',
    'ModelCategory',
    'ModelEntry',
    'ModelSearchResult',
    'ModelDetails',
    'LoRAInfo',
    'OllamaModelInfo',
    'ModelTreeNode',
    # Classes
    'ModelLibrary',
    'ModelDownloader',
]
