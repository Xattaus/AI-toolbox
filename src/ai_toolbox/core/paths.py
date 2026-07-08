"""
AI TOOLBOX - Path Management
============================

Centralized path management for portable operation.
All paths are relative to the toolbox installation directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional


class ToolboxPaths:
    """Manages all paths for the AI Toolbox in a portable way."""

    _instance: Optional["ToolboxPaths"] = None
    _root: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._root is None:
            self._root = self._detect_root()
            self._ensure_directories()

    def _detect_root(self) -> Path:
        """
        Detect the AI Toolbox root directory.

        Priority:
        1. AITOOLBOX_ROOT environment variable
        2. Parent of the src/ai_toolbox directory (when running from source)
        3. Current working directory if it contains expected structure
        """
        # Check environment variable first
        env_root = os.environ.get("AITOOLBOX_ROOT")
        if env_root:
            root = Path(env_root)
            if root.exists():
                return root

        # Try to find root from this file's location
        # This file is at: <root>/src/ai_toolbox/core/paths.py
        this_file = Path(__file__).resolve()
        possible_root = this_file.parent.parent.parent.parent  # Up 4 levels now

        if (possible_root / "src" / "ai_toolbox").exists():
            return possible_root

        # Try current working directory
        cwd = Path.cwd()
        if (cwd / "src" / "ai_toolbox").exists():
            return cwd

        # Fallback: käytä tiedoston sijaintikansiota (ei possible_root joka voi olla väärä)
        # Tämä on turvallisempi koska tiedämme ainakin missä koodi sijaitsee
        fallback = this_file.parent.parent.parent  # <root>/src/ai_toolbox -> <root>/src
        if fallback.name == "src":
            fallback = fallback.parent  # <root>
        return fallback

    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.models_dir,
            self.downloads_dir,
            self.safetensors_dir,  # Alias for clarity
            self.gguf_dir,
            self.loras_dir,
            self.adapters_dir,  # Final LoRA adapters
            self.runs_dir,  # Training runs with checkpoints
            self.merged_dir,
            self.abliterated_dir,  # Abliterated models
            self.ollama_dir,  # Ollama modelfiles
            self.llama_cpp_dir,
            self.config_dir,
            self.datasets_dir,
            self.processed_dir,
            self.benchmarks_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """AI Toolbox root directory."""
        root = self._root
        if root is None:
            root = self._detect_root()
            self._root = root
        return root

    @property
    def models_dir(self) -> Path:
        """Main models directory."""
        return self.root / "models"

    @property
    def downloads_dir(self) -> Path:
        """Directory for HuggingFace downloads (legacy alias for safetensors_dir)."""
        return self.root / "models" / "safetensors"

    @property
    def safetensors_dir(self) -> Path:
        """Directory for SafeTensors/HuggingFace models."""
        return self.root / "models" / "safetensors"

    @property
    def gguf_dir(self) -> Path:
        """Directory for GGUF models."""
        return self.root / "models" / "gguf"

    @property
    def loras_dir(self) -> Path:
        """Directory for LoRA adapters (root)."""
        return self.root / "models" / "lora"

    @property
    def adapters_dir(self) -> Path:
        """Directory for final trained LoRA adapters."""
        return self.root / "models" / "lora" / "adapters"

    @property
    def runs_dir(self) -> Path:
        """Directory for LoRA training runs with checkpoints."""
        return self.root / "models" / "lora" / "runs"

    @property
    def merged_dir(self) -> Path:
        """Directory for merged models."""
        return self.root / "models" / "merged"

    @property
    def abliterated_dir(self) -> Path:
        """Directory for abliterated models (refusal removed)."""
        return self.root / "models" / "abliterated"

    @property
    def ollama_dir(self) -> Path:
        """Directory for Ollama modelfiles."""
        return self.root / "models" / "ollama"

    @property
    def library_file(self) -> Path:
        """Path to the library index file."""
        return self.root / "models" / "library.json"

    @property
    def llama_cpp_dir(self) -> Path:
        """Directory for llama.cpp installation."""
        return self.root / "tools" / "llama.cpp"

    @property
    def config_dir(self) -> Path:
        """Directory for configuration files."""
        return self.root / "config"

    @property
    def datasets_dir(self) -> Path:
        """Directory for datasets."""
        return self.root / "datasets"

    @property
    def processed_dir(self) -> Path:
        """Directory for processed datasets."""
        return self.root / "datasets" / "processed"

    @property
    def benchmarks_dir(self) -> Path:
        """Directory for benchmark results."""
        return self.root / "benchmarks"

    @property
    def config_file(self) -> Path:
        """Main configuration file."""
        return self.root / "config" / "settings.json"

    def get_model_path(self, model_name: str, model_type: str = "gguf") -> Path:
        """
        Get the path for a model file.

        Args:
            model_name: Name of the model
            model_type: Type of model ('gguf', 'safetensors', 'adapter', 'merged', 'ollama')

        Returns:
            Path to the model
        """
        type_dirs = {
            "gguf": self.gguf_dir,
            "safetensors": self.safetensors_dir,
            "download": self.safetensors_dir,  # Legacy alias
            "adapter": self.adapters_dir,
            "lora": self.adapters_dir,  # Alias
            "run": self.runs_dir,
            "merged": self.merged_dir,
            "abliterated": self.abliterated_dir,
            "ollama": self.ollama_dir,
        }
        base_dir = type_dirs.get(model_type, self.models_dir)
        return base_dir / model_name

    def __str__(self) -> str:
        return f"ToolboxPaths(root={self._root})"

    def __repr__(self) -> str:
        return self.__str__()


# Global singleton instance
_paths: Optional[ToolboxPaths] = None


def get_paths() -> ToolboxPaths:
    """Get the global ToolboxPaths instance."""
    global _paths
    if _paths is None:
        _paths = ToolboxPaths()
    return _paths


def reset_paths():
    """Reset the global paths instance (for testing)."""
    global _paths
    _paths = None
    ToolboxPaths._instance = None
    ToolboxPaths._root = None


# Convenience functions
def get_root() -> Path:
    """Get the toolbox root directory."""
    return get_paths().root


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_paths().models_dir


def get_downloads_dir() -> Path:
    """Get the downloads directory."""
    return get_paths().downloads_dir


def get_gguf_dir() -> Path:
    """Get the GGUF models directory."""
    return get_paths().gguf_dir


def get_loras_dir() -> Path:
    """Get the LoRA adapters directory."""
    return get_paths().loras_dir


def get_merged_dir() -> Path:
    """Get the merged models directory."""
    return get_paths().merged_dir


def get_safetensors_dir() -> Path:
    """Get the SafeTensors models directory."""
    return get_paths().safetensors_dir


def get_adapters_dir() -> Path:
    """Get the LoRA adapters directory."""
    return get_paths().adapters_dir


def get_runs_dir() -> Path:
    """Get the LoRA training runs directory."""
    return get_paths().runs_dir


def get_ollama_dir() -> Path:
    """Get the Ollama modelfiles directory."""
    return get_paths().ollama_dir


def get_abliterated_dir() -> Path:
    """Get the abliterated models directory."""
    return get_paths().abliterated_dir


def get_library_file() -> Path:
    """Get the library index file path."""
    return get_paths().library_file


def get_llama_cpp_dir() -> Path:
    """Get the llama.cpp directory."""
    return get_paths().llama_cpp_dir


def get_config_dir() -> Path:
    """Get the config directory."""
    return get_paths().config_dir


def get_datasets_dir() -> Path:
    """Get the datasets directory."""
    return get_paths().datasets_dir


def get_processed_dir() -> Path:
    """Get the processed datasets directory."""
    return get_paths().processed_dir


def get_benchmarks_dir() -> Path:
    """Get the benchmarks directory."""
    return get_paths().benchmarks_dir
