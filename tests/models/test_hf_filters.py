"""Tests for models.hf_filters: pure name/quantization/memory helpers."""

from ai_toolbox.models.hf_filters import (
    get_quality_stars,
    estimate_model_memory,
    parse_model_size_from_name,
    detect_quantization_from_filename,
    get_app_compatibility,
)


def test_quality_stars_length_and_charset():
    stars = get_quality_stars("Q4_K_M")
    assert len(stars) == 5
    assert set(stars) <= {"*", "."}


def test_quality_stars_unknown_defaults_to_three():
    assert get_quality_stars("NOPE") == "***.."


def test_estimate_memory_scales_with_precision():
    f16 = estimate_model_memory(7_000_000_000, "F16")
    q4 = estimate_model_memory(7_000_000_000, "Q4_K_M")
    assert f16 > q4 > 0


def test_estimate_memory_zero_params():
    assert estimate_model_memory(0, "Q4_K_M") == 0


def test_parse_model_size():
    assert parse_model_size_from_name("Llama-2-7B") == "7B"
    assert parse_model_size_from_name("TinyLlama-1.1B-chat") == "1.1B"
    assert parse_model_size_from_name("some-random-model") is None


def test_detect_quantization_from_filename():
    assert detect_quantization_from_filename("mymodel-Q4_K_M.gguf") == "Q4_K_M"
    assert detect_quantization_from_filename("mymodel-f16.gguf") == "F16"
    assert detect_quantization_from_filename("plainfile.txt") is None


def test_app_compatibility_returns_list():
    assert isinstance(get_app_compatibility([], has_gguf=True), list)
    assert isinstance(get_app_compatibility(["transformers"], has_gguf=False), list)
