"""Tests for conversion.quantization: info lookup, listing, recommendations."""

import pytest

from ai_toolbox.conversion.quantization import (
    QuantizationType,
    QuantizationInfo,
    get_quantization_info,
    list_quantization_types,
    recommend_quantization,
)


def test_get_quantization_info_known():
    info = get_quantization_info("q4_k_m")
    assert isinstance(info, QuantizationInfo)
    assert info.type == QuantizationType.Q4_K_M
    assert 0 < info.bits_per_weight <= 32
    assert isinstance(info.quality, str) and info.quality


def test_get_quantization_info_is_case_insensitive():
    assert get_quantization_info("Q8_0").type == QuantizationType.Q8_0


def test_get_quantization_info_unknown_raises():
    with pytest.raises(ValueError):
        get_quantization_info("not_a_quant")


def test_list_quantization_types_shape():
    items = list_quantization_types()
    assert len(items) >= 10
    for item in items:
        assert set(item) >= {"type", "bits_per_weight", "quality", "description"}


def test_bits_ordering_f16_heavier_than_q4():
    assert get_quantization_info("f16").bits_per_weight > get_quantization_info("q4_k_m").bits_per_weight


def test_recommend_quantization_returns_list():
    recs = recommend_quantization(7.0, 32.0)
    assert isinstance(recs, list)
    assert all(r["fits_in_ram"] for r in recs)


def test_recommend_more_ram_allows_more_options():
    tiny = recommend_quantization(70.0, 6.0)
    huge = recommend_quantization(70.0, 200.0)
    assert len(huge) >= len(tiny)


def test_quantization_type_value_mapping():
    assert QuantizationType("q4_k_m") is QuantizationType.Q4_K_M
    assert QuantizationType.Q8_0.value == "q8_0"
