"""
AI TOOLBOX - Conversion Module
==============================

GGUF conversion and quantization functionality.
"""

from .quantization import (
    QuantizationType,
    QuantizationInfo,
    QUANTIZATION_INFO,
    get_quantization_info,
    list_quantization_types,
    recommend_quantization,
)

from .llama_cpp import LlamaCppManager

from .converter import GGUFConverter

__all__ = [
    # Quantization
    'QuantizationType',
    'QuantizationInfo',
    'QUANTIZATION_INFO',
    'get_quantization_info',
    'list_quantization_types',
    'recommend_quantization',
    # llama.cpp
    'LlamaCppManager',
    # Converter
    'GGUFConverter',
]
