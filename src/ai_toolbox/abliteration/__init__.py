"""
AI TOOLBOX - Abliteration Module
================================

Remove refusal behavior from language models.

WARNING: This tool removes safety features from models.
Use responsibly for research and testing purposes only.
"""

from .abliterator import Abliterator, AbliterationConfig, AbliterationResult
from .prompts import get_prompts, get_test_prompts, load_prompts_from_file
from .hooks import (
    register_activation_hooks,
    get_model_layer_count,
    get_recommended_layers,
    ActivationCache,
)
from .testing import (
    AbliterationTester,
    AbliterationTestResult,
    AbliterationTestReport,
    CategoryStats,
    ComparisonReport,
)
from .hardware import (
    HardwareProfile,
    MemoryEstimate,
    RecommendedSettings,
    PreflightResult,
    detect_hardware,
    estimate_cost,
    recommend_config,
    check_preflight,
    recommend_pagefile_gb,
    build_set_pagefile_command,
    apply_pagefile_setting,
)

__all__ = [
    # Core abliteration
    'Abliterator',
    'AbliterationConfig',
    'AbliterationResult',
    # Prompts
    'get_prompts',
    'get_test_prompts',
    'load_prompts_from_file',
    # Hooks
    'register_activation_hooks',
    'get_model_layer_count',
    'get_recommended_layers',
    'ActivationCache',
    # Testing
    'AbliterationTester',
    'AbliterationTestResult',
    'AbliterationTestReport',
    'CategoryStats',
    'ComparisonReport',
    # Hardware / pre-flight
    'HardwareProfile',
    'MemoryEstimate',
    'RecommendedSettings',
    'PreflightResult',
    'detect_hardware',
    'estimate_cost',
    'recommend_config',
    'check_preflight',
    'recommend_pagefile_gb',
    'build_set_pagefile_command',
    'apply_pagefile_setting',
]
