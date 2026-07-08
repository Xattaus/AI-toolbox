"""
AI TOOLBOX - Configuration Management
=====================================

Centralized configuration handling for the toolbox.
"""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

from .paths import get_paths


@dataclass
class ToolboxConfig:
    """Main configuration for AI Toolbox."""

    # General settings
    language: str = "en"
    theme: str = "default"

    # Model defaults
    default_quantization: str = "Q4_K_M"
    auto_add_to_library: bool = True

    # Download settings
    hf_token: Optional[str] = None
    download_threads: int = 4

    # Chat settings
    default_context_size: int = 4096
    default_gpu_layers: int = -1  # -1 = auto

    # Training settings
    default_batch_size: int = 4
    default_learning_rate: float = 2e-4

    # UI settings
    show_tips: bool = True
    confirm_destructive: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolboxConfig":
        """Create config from dictionary."""
        # Only use keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


# Global config instance
_config: Optional[ToolboxConfig] = None


def get_config() -> ToolboxConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config() -> ToolboxConfig:
    """Load configuration from file."""
    config_file = get_paths().config_file

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ToolboxConfig.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass

    return ToolboxConfig()


def save_config(config: Optional[ToolboxConfig] = None):
    """Save configuration to file."""
    if config is None:
        config = get_config()

    config_file = get_paths().config_file
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Atomaarinen tallennus: temp-tiedosto + rename
    temp_file = config_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        temp_file.replace(config_file)
        # The config may hold an HF token; restrict to owner-only on POSIX.
        # (chmod is a no-op for permission bits on Windows, and harmless.)
        try:
            config_file.chmod(0o600)
        except OSError:
            pass
    except (OSError, IOError, PermissionError) as e:
        # Siivoa temp-tiedosto ja ilmoita virheestä
        if temp_file.exists():
            temp_file.unlink()
        from rich.console import Console

        Console().print(f"[red]Virhe tallennettaessa asetuksia: {e}[/red]")


def update_config(**kwargs) -> ToolboxConfig:
    """Update configuration with new values."""
    config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    save_config(config)
    return config


def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = ToolboxConfig()
    save_config(_config)
    return _config
