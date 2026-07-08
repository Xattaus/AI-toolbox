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
    ExtendedModelInfo,
    HFSearchResult,
    GGUFFileInfo,
)

from .library import ModelLibrary

from .downloader import ModelDownloader

from .hf_search import (
    HFSearchEngine,
    SearchFilters,
    SearchResult,
    ModelCardInfo,
    ModelCardAnalyzer,
    GGUFVariant,
)

from .hf_filters import (
    TASK_CATEGORIES,
    LIBRARIES,
    APPS,
    APP_CHOICES,
    LICENSES,
    SEARCH_PRESETS,
    SORT_OPTIONS,
    QUANTIZATION_QUALITY,
    BYTES_PER_PARAM,
    RECOMMENDED_QUANTIZATIONS,
    get_quality_stars,
    parse_model_size_from_name,
    detect_quantization_from_filename,
    get_app_compatibility,
)

__all__ = [
    # Types
    "ModelFormat",
    "ModelSource",
    "ModelCategory",
    "ModelEntry",
    "ModelSearchResult",
    "ModelDetails",
    "LoRAInfo",
    "OllamaModelInfo",
    "ModelTreeNode",
    "ExtendedModelInfo",
    "HFSearchResult",
    "GGUFFileInfo",
    # Search engine
    "HFSearchEngine",
    "SearchFilters",
    "SearchResult",
    "ModelCardInfo",
    "ModelCardAnalyzer",
    "GGUFVariant",
    # Filter constants
    "TASK_CATEGORIES",
    "LIBRARIES",
    "APPS",
    "APP_CHOICES",
    "LICENSES",
    "SEARCH_PRESETS",
    "SORT_OPTIONS",
    "QUANTIZATION_QUALITY",
    "BYTES_PER_PARAM",
    "RECOMMENDED_QUANTIZATIONS",
    # Helper functions
    "get_quality_stars",
    "parse_model_size_from_name",
    "detect_quantization_from_filename",
    "get_app_compatibility",
    # Classes
    "ModelLibrary",
    "ModelDownloader",
]
