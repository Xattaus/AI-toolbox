"""
AI TOOLBOX - The local LLM workshop
===================================

A collection of powerful tools for working with local AI models.

New modular structure (v2.0):
- core/       - Core utilities (paths, ui, config, exceptions)
- models/     - Model management (library, downloader)
- conversion/ - GGUF conversion and quantization
- training/   - LoRA training and dataset preparation
- inference/  - AI chat, benchmark, assistant
- merging/    - Model merging
- integrations/ - MCP server and external integrations
- cli/        - Command-line interface

Backward compatible imports are provided below.
"""

__version__ = "3.0.2"
__author__ = "AI Toolbox"

# =============================================================================
# BACKWARD COMPATIBLE IMPORTS
# =============================================================================
# These imports maintain compatibility with code that used the old flat structure

# Core - paths
from .core.paths import (
    ToolboxPaths,
    get_paths,
    get_root,
    get_models_dir,
    get_downloads_dir,
    get_gguf_dir,
    get_loras_dir,
    get_merged_dir,
    get_library_file,
    get_llama_cpp_dir,
    get_config_dir,
    get_datasets_dir,
    get_processed_dir,
    get_benchmarks_dir,
)

# Core - UI
from .core.ui import (
    console,
    print_toolbox_banner,
    print_mini_banner,
    create_menu_table,
    create_model_card,
    print_success,
    print_error,
    print_warning,
    print_info,
    format_size,
)

# Core - Config
from .core.config import (
    ToolboxConfig,
    get_config,
    load_config,
    save_config,
    update_config,
)

# Core - Exceptions
from .core.exceptions import (
    ToolboxError,
    ModelError,
    ModelNotFoundError,
    ModelDownloadError,
    ModelConversionError,
    ConversionError,
    LlamaCppNotFoundError,
    QuantizationError,
    TrainingError,
    DatasetError,
    DependencyError,
    MissingDependencyError,
)

# Models
from .models.library import ModelLibrary
from .models.downloader import ModelDownloader
from .models.types import ModelEntry, ModelSearchResult, ModelDetails

# Conversion
from .conversion.converter import GGUFConverter
from .conversion.quantization import QuantizationType, QUANTIZATION_INFO

# Training
from .training.lora import LoRATrainer, LoRAConfig, TrainingConfig, FullConfig
from .training.dataset import DatasetPrep, DatasetFormat, SplitConfig, FilterConfig

# Inference
from .inference.benchmark import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
from .inference.chat import AIChat, ai_chat_menu
from .inference.assistant import ai_assistant_menu

# Merging
from .merging.merger import ModelMerger

# CLI entry point
from .cli.app import main, AIToolbox

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core - Paths
    "ToolboxPaths",
    "get_paths",
    "get_root",
    "get_models_dir",
    "get_downloads_dir",
    "get_gguf_dir",
    "get_loras_dir",
    "get_merged_dir",
    "get_library_file",
    "get_llama_cpp_dir",
    "get_config_dir",
    "get_datasets_dir",
    "get_processed_dir",
    "get_benchmarks_dir",
    # Core - UI
    "console",
    "print_toolbox_banner",
    "print_mini_banner",
    "create_menu_table",
    "create_model_card",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "format_size",
    # Core - Config
    "ToolboxConfig",
    "get_config",
    "load_config",
    "save_config",
    "update_config",
    # Core - Exceptions
    "ToolboxError",
    "ModelError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "ModelConversionError",
    "ConversionError",
    "LlamaCppNotFoundError",
    "QuantizationError",
    "TrainingError",
    "DatasetError",
    "DependencyError",
    "MissingDependencyError",
    # Models
    "ModelLibrary",
    "ModelDownloader",
    "ModelEntry",
    "ModelSearchResult",
    "ModelDetails",
    # Conversion
    "GGUFConverter",
    "QuantizationType",
    "QUANTIZATION_INFO",
    # Training
    "LoRATrainer",
    "LoRAConfig",
    "TrainingConfig",
    "FullConfig",
    "DatasetPrep",
    "DatasetFormat",
    "SplitConfig",
    "FilterConfig",
    # Inference
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "AIChat",
    "ai_chat_menu",
    "ai_assistant_menu",
    # Merging
    "ModelMerger",
    # CLI
    "main",
    "AIToolbox",
]
