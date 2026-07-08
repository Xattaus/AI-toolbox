"""Tests for merging.mergekit_wrapper: config reading and architecture checks."""

import json

from ai_toolbox.merging.mergekit_wrapper import MergekitWrapper


def _info(**over):
    base = dict(
        name="M",
        hidden_size=4096,
        num_layers=32,
        vocab_size=32000,
        max_position_embeddings=8192,
        rope_scaling_type=None,
    )
    base.update(over)
    return base


def _check(infos):
    return MergekitWrapper().check_architecture_compatibility(infos)


def test_read_model_config_missing(tmp_path):
    result = MergekitWrapper().read_model_config(tmp_path)
    assert result["config_found"] is False


def test_read_model_config_parses(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
            }
        ),
        encoding="utf-8",
    )
    result = MergekitWrapper().read_model_config(tmp_path)
    assert result["config_found"] is True
    assert result["architecture"] == "LlamaForCausalLM"
    assert result["hidden_size"] == 4096
    assert result["num_layers"] == 32


def test_compat_requires_two_models():
    r = _check([_info()])
    assert r["compatible_slerp"] is False
    assert r["issues"]


def test_compat_identical_models():
    r = _check([_info(), _info()])
    assert r["identical"] is True
    assert r["compatible_slerp"] is True
    assert r["compatible_dare"] is True
    assert not r["issues"]


def test_compat_different_hidden_size_is_incompatible():
    r = _check([_info(hidden_size=4096), _info(hidden_size=2048)])
    assert r["compatible_slerp"] is False
    assert r["compatible_dare"] is False
    assert r["issues"]


def test_compat_different_layers_is_incompatible():
    r = _check([_info(num_layers=32), _info(num_layers=28)])
    assert r["compatible_slerp"] is False
    assert r["compatible_dare"] is False


def test_compat_large_vocab_diff_breaks_dare_only():
    r = _check([_info(vocab_size=32000), _info(vocab_size=32200)])
    assert r["compatible_slerp"] is True  # SLERP tolerates vocab differences
    assert r["compatible_dare"] is False
    assert r["warnings"]


def test_compat_rope_mismatch_breaks_dare_only():
    r = _check([_info(rope_scaling_type=None), _info(rope_scaling_type="linear")])
    assert r["compatible_slerp"] is True
    assert r["compatible_dare"] is False
