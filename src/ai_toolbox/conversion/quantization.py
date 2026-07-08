"""
AI TOOLBOX - Quantization Types
===============================

Quantization type definitions and information for GGUF conversion.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class QuantizationType(Enum):
    """Available quantization types for GGUF conversion."""

    # Full precision
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"

    # Integer quantization
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q3_K_L = "q3_k_l"
    Q3_K_M = "q3_k_m"
    Q3_K_S = "q3_k_s"
    Q2_K = "q2_k"

    # IQ quantization (importance-weighted)
    IQ4_NL = "iq4_nl"
    IQ4_XS = "iq4_xs"
    IQ3_S = "iq3_s"
    IQ3_M = "iq3_m"
    IQ3_XS = "iq3_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_S = "iq2_s"
    IQ2_M = "iq2_m"
    IQ2_XS = "iq2_xs"
    IQ2_XXS = "iq2_xxs"
    IQ1_S = "iq1_s"
    IQ1_M = "iq1_m"


@dataclass
class QuantizationInfo:
    """Information about a quantization type."""

    type: QuantizationType
    bits_per_weight: float
    description: str
    quality: str  # "highest", "very high", "high", "medium-high", "medium", "medium-low", "low", "very low", "minimal"


# Quantization type information dictionary
QUANTIZATION_INFO: Dict[QuantizationType, QuantizationInfo] = {
    QuantizationType.F32: QuantizationInfo(
        QuantizationType.F32, 32.0, "32-bit floating point (no quantization)", "highest"
    ),
    QuantizationType.F16: QuantizationInfo(
        QuantizationType.F16, 16.0, "16-bit floating point", "highest"
    ),
    QuantizationType.BF16: QuantizationInfo(
        QuantizationType.BF16, 16.0, "Brain floating point 16-bit", "highest"
    ),
    QuantizationType.Q8_0: QuantizationInfo(
        QuantizationType.Q8_0, 8.0, "8-bit quantization", "very high"
    ),
    QuantizationType.Q6_K: QuantizationInfo(QuantizationType.Q6_K, 6.5, "6-bit K-quant", "high"),
    QuantizationType.Q5_K_M: QuantizationInfo(
        QuantizationType.Q5_K_M, 5.5, "5-bit K-quant medium", "high"
    ),
    QuantizationType.Q5_K_S: QuantizationInfo(
        QuantizationType.Q5_K_S, 5.5, "5-bit K-quant small", "high"
    ),
    QuantizationType.Q5_0: QuantizationInfo(
        QuantizationType.Q5_0, 5.0, "5-bit quantization", "high"
    ),
    QuantizationType.Q5_1: QuantizationInfo(
        QuantizationType.Q5_1, 5.5, "5-bit quantization with extra precision", "high"
    ),
    QuantizationType.Q4_K_M: QuantizationInfo(
        QuantizationType.Q4_K_M, 4.5, "4-bit K-quant medium (recommended)", "medium-high"
    ),
    QuantizationType.Q4_K_S: QuantizationInfo(
        QuantizationType.Q4_K_S, 4.5, "4-bit K-quant small", "medium"
    ),
    QuantizationType.Q4_0: QuantizationInfo(
        QuantizationType.Q4_0, 4.0, "4-bit quantization", "medium"
    ),
    QuantizationType.Q4_1: QuantizationInfo(
        QuantizationType.Q4_1, 4.5, "4-bit quantization with extra precision", "medium"
    ),
    QuantizationType.Q3_K_L: QuantizationInfo(
        QuantizationType.Q3_K_L, 3.5, "3-bit K-quant large", "medium-low"
    ),
    QuantizationType.Q3_K_M: QuantizationInfo(
        QuantizationType.Q3_K_M, 3.5, "3-bit K-quant medium", "low"
    ),
    QuantizationType.Q3_K_S: QuantizationInfo(
        QuantizationType.Q3_K_S, 3.5, "3-bit K-quant small", "low"
    ),
    QuantizationType.Q2_K: QuantizationInfo(
        QuantizationType.Q2_K, 2.5, "2-bit K-quant", "very low"
    ),
    QuantizationType.IQ4_NL: QuantizationInfo(
        QuantizationType.IQ4_NL, 4.25, "4-bit importance-weighted non-linear", "medium-high"
    ),
    QuantizationType.IQ4_XS: QuantizationInfo(
        QuantizationType.IQ4_XS, 4.0, "4-bit importance-weighted extra small", "medium"
    ),
    QuantizationType.IQ3_S: QuantizationInfo(
        QuantizationType.IQ3_S, 3.5, "3-bit importance-weighted small", "medium-low"
    ),
    QuantizationType.IQ3_M: QuantizationInfo(
        QuantizationType.IQ3_M, 3.4, "3-bit importance-weighted medium", "medium-low"
    ),
    QuantizationType.IQ3_XS: QuantizationInfo(
        QuantizationType.IQ3_XS, 3.3, "3-bit importance-weighted extra small", "low"
    ),
    QuantizationType.IQ3_XXS: QuantizationInfo(
        QuantizationType.IQ3_XXS, 3.0, "3-bit importance-weighted extra extra small", "low"
    ),
    QuantizationType.IQ2_S: QuantizationInfo(
        QuantizationType.IQ2_S, 2.5, "2-bit importance-weighted small", "very low"
    ),
    QuantizationType.IQ2_M: QuantizationInfo(
        QuantizationType.IQ2_M, 2.7, "2-bit importance-weighted medium", "very low"
    ),
    QuantizationType.IQ2_XS: QuantizationInfo(
        QuantizationType.IQ2_XS, 2.3, "2-bit importance-weighted extra small", "very low"
    ),
    QuantizationType.IQ2_XXS: QuantizationInfo(
        QuantizationType.IQ2_XXS, 2.0, "2-bit importance-weighted extra extra small", "very low"
    ),
    QuantizationType.IQ1_S: QuantizationInfo(
        QuantizationType.IQ1_S, 1.5, "1-bit importance-weighted small", "minimal"
    ),
    QuantizationType.IQ1_M: QuantizationInfo(
        QuantizationType.IQ1_M, 1.75, "1-bit importance-weighted medium", "minimal"
    ),
}


def get_quantization_info(quant_type: str) -> QuantizationInfo:
    """Get quantization info by type string."""
    try:
        qt = QuantizationType(quant_type.lower())
        return QUANTIZATION_INFO[qt]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown quantization type: {quant_type}")


def list_quantization_types() -> list:
    """Get list of all available quantization types."""
    result = []
    for quant_type, info in QUANTIZATION_INFO.items():
        result.append(
            {
                "type": quant_type.value,
                "bits_per_weight": info.bits_per_weight,
                "quality": info.quality,
                "description": info.description,
            }
        )
    return result


def recommend_quantization(model_params_billions: float, available_ram_gb: float) -> list:
    """
    Recommend quantization types based on model size and available RAM.

    Args:
        model_params_billions: Model parameters in billions
        available_ram_gb: Available system RAM in GB

    Returns:
        List of recommended quantization types
    """
    recommendations = []

    for quant_type, info in QUANTIZATION_INFO.items():
        estimated_size_gb = (model_params_billions * 1e9 * info.bits_per_weight / 8) / (1024**3)
        required_ram = estimated_size_gb + 2  # Add overhead

        if required_ram <= available_ram_gb * 0.8:  # Leave 20% headroom
            recommendations.append(
                {
                    "type": quant_type.value,
                    "bits_per_weight": info.bits_per_weight,
                    "quality": info.quality,
                    "estimated_size_gb": round(estimated_size_gb, 2),
                    "required_ram_gb": round(required_ram, 2),
                    "fits_in_ram": True,
                }
            )

    # Sort by quality (bits per weight, higher is better)
    recommendations.sort(key=lambda x: x["bits_per_weight"], reverse=True)

    return recommendations[:5]
