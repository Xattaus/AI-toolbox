"""
AI Toolbox - Merging Module
===========================

Model merging utilities for combining models using various techniques.
"""

from .merger import ModelMerger, MergeMethod, MergeConfig
from .mergekit_wrapper import MergekitWrapper, MergekitMethod, MergekitConfig
from .presets import (
    MergePreset,
    PresetCategory,
    PRESETS,
    get_preset,
    list_presets,
    get_recommended_preset,
    get_presets_by_category,
)
from .config_manager import MergeConfigManager, MergeHistoryEntry

__all__ = [
    # Legacy merger
    "ModelMerger",
    "MergeMethod",
    "MergeConfig",
    # Mergekit wrapper
    "MergekitWrapper",
    "MergekitMethod",
    "MergekitConfig",
    # Presets
    "MergePreset",
    "PresetCategory",
    "PRESETS",
    "get_preset",
    "list_presets",
    "get_recommended_preset",
    "get_presets_by_category",
    # Config manager
    "MergeConfigManager",
    "MergeHistoryEntry",
]
