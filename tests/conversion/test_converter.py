"""Tests for conversion.converter: GGUF validation and size estimation."""

import json

from ai_toolbox.conversion.converter import GGUFConverter


def test_is_valid_gguf_accepts_magic_bytes(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 16)
    assert GGUFConverter._is_valid_gguf(f) is True


def test_is_valid_gguf_rejects_wrong_magic(tmp_path):
    f = tmp_path / "bad.gguf"
    f.write_bytes(b"XXXX" + b"\x00" * 16)
    assert GGUFConverter._is_valid_gguf(f) is False


def test_is_valid_gguf_rejects_missing_file(tmp_path):
    assert GGUFConverter._is_valid_gguf(tmp_path / "nope.gguf") is False


def test_estimate_model_size_missing_config(tmp_path):
    result = GGUFConverter().estimate_model_size(tmp_path)
    assert result["estimated_size_gb"] == 0
    assert "error" in result


def test_estimate_model_size_computes_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "intermediate_size": 11008,
            }
        ),
        encoding="utf-8",
    )

    result = GGUFConverter().estimate_model_size(tmp_path, quantization="q4_k_m")
    assert result["estimated_size_gb"] > 0
    assert result["original_size_gb"] > result["estimated_size_gb"]  # quant shrinks it
    assert result["total_params_billions"] > 0
    assert result["compression_ratio"] > 1


def test_estimate_size_smaller_quant_is_smaller(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "vocab_size": 32000,
                "intermediate_size": 5504,
            }
        ),
        encoding="utf-8",
    )
    conv = GGUFConverter()
    q8 = conv.estimate_model_size(tmp_path, "q8_0")["estimated_size_gb"]
    q4 = conv.estimate_model_size(tmp_path, "q4_k_m")["estimated_size_gb"]
    assert q4 < q8


def test_converter_list_quantization_types():
    assert len(GGUFConverter.list_quantization_types()) >= 10
