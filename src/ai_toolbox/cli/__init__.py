"""
AI TOOLBOX - CLI Commands
=========================

Command-line interface modules for AI Toolbox.
"""

from .app import AIToolbox, main

# New unified command modules
from .model_hub_cmd import ModelHubCommands
from .gguf_tools_cmd import GGUFToolsCommands
from .training_center_cmd import TrainingCenterCommands

# Kept command modules (used internally)
from .training_cmd import TrainingCommands
from .dataset_cmd import DatasetCommands
from .benchmark_cmd import BenchmarkCommands
from .merger_cmd import MergerCommands
from .abliteration_cmd import AbliterationCommands
from .settings_cmd import SettingsCommands
from .ollama_cmd import run_ollama_wizard

# CLI helpers
from .helpers import (
    MENU_STYLE,
    console,
    create_menu_choices,
    run_menu,
    confirm_action,
    prompt_text,
    prompt_path,
    prompt_select,
    prompt_checkbox,
    prompt_number,
    press_any_key,
    run_wizard_step,
    WizardContext,
    create_choice,
    create_separator,
)

# Selection helpers
from .selection import (
    select_model,
    select_model_for_merge,
    select_model_for_training,
    select_gguf_model,
    select_dataset,
    select_dataset_for_training,
    select_lora_adapter,
    select_from_list,
    multi_select_models,
)

__all__ = [
    # Main application
    'AIToolbox',
    'main',
    # New unified command classes
    'ModelHubCommands',
    'GGUFToolsCommands',
    'TrainingCenterCommands',
    # Kept command classes (used internally)
    'TrainingCommands',
    'DatasetCommands',
    'BenchmarkCommands',
    'MergerCommands',
    'AbliterationCommands',
    'SettingsCommands',
    'run_ollama_wizard',
    # Style and console
    "MENU_STYLE",
    "console",
    # Menu helpers
    "create_menu_choices",
    "run_menu",
    "confirm_action",
    "prompt_text",
    "prompt_path",
    "prompt_select",
    "prompt_checkbox",
    "prompt_number",
    "press_any_key",
    # Wizard helpers
    "run_wizard_step",
    "WizardContext",
    # Choice builders
    "create_choice",
    "create_separator",
    # Selection helpers
    "select_model",
    "select_model_for_merge",
    "select_model_for_training",
    "select_gguf_model",
    "select_dataset",
    "select_dataset_for_training",
    "select_lora_adapter",
    "select_from_list",
    "multi_select_models",
]
