"""Tests for merging.config_manager: YAML config save/load/list/delete."""

import pytest

from ai_toolbox.merging.config_manager import MergeConfigManager


def _mgr(tmp_path):
    return MergeConfigManager(config_dir=str(tmp_path / "merge_configs"))


def _cfg():
    return {
        "merge_method": "slerp",
        "models": [{"model": "a"}, {"model": "b"}],
        "parameters": {"t": 0.5},
        "dtype": "bfloat16",
    }


def test_save_and_load_roundtrip(tmp_path):
    mgr = _mgr(tmp_path)
    path = mgr.save_config(_cfg(), "my-merge")
    assert path.exists()
    loaded = mgr.load_config("my-merge")
    assert loaded["merge_method"] == "slerp"
    assert loaded["parameters"]["t"] == 0.5


def test_load_unknown_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _mgr(tmp_path).load_config("nope")


def test_name_with_special_chars_is_sanitized(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.save_config(_cfg(), "My Merge! v2")
    # loading with the same human name resolves through the same sanitizer
    loaded = mgr.load_config("My Merge! v2")
    assert loaded["merge_method"] == "slerp"


def test_delete_config(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.save_config(_cfg(), "temp")
    assert mgr.delete_config("temp") is True
    assert mgr.delete_config("temp") is False
    with pytest.raises(FileNotFoundError):
        mgr.load_config("temp")


def test_list_configs_reflects_saved(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.save_config(_cfg(), "alpha")
    mgr.save_config(_cfg(), "beta")
    listed = mgr.list_configs()
    assert isinstance(listed, list)
    assert len(listed) >= 2
