"""Tests for core.config: defaults, (de)serialization, persistence, permissions."""

import stat
import sys

import pytest

from ai_toolbox.core.config import (
    ToolboxConfig,
    load_config,
    save_config,
    update_config,
)
from ai_toolbox.core.paths import get_paths


def test_defaults():
    c = ToolboxConfig()
    assert c.language == "en"
    assert c.default_quantization == "Q4_K_M"
    assert c.hf_token is None
    assert c.download_threads == 4
    assert c.confirm_destructive is True


def test_to_dict_from_dict_roundtrip():
    c = ToolboxConfig(language="fi", hf_token="tok", download_threads=8)
    restored = ToolboxConfig.from_dict(c.to_dict())
    assert restored == c


def test_from_dict_ignores_unknown_keys():
    c = ToolboxConfig.from_dict({"language": "fi", "bogus_key": 123})
    assert c.language == "fi"
    assert not hasattr(c, "bogus_key")


def test_from_dict_partial_uses_defaults():
    c = ToolboxConfig.from_dict({"language": "fi"})
    assert c.language == "fi"
    assert c.default_quantization == "Q4_K_M"  # untouched default


def test_load_config_returns_defaults_when_missing():
    assert not get_paths().config_file.exists()
    assert load_config() == ToolboxConfig()


def test_save_and_load_roundtrip():
    cfg = ToolboxConfig(language="fi", hf_token="secret", download_threads=8)
    save_config(cfg)
    assert get_paths().config_file.exists()
    assert load_config() == cfg  # load_config always reads fresh from disk


def test_load_config_tolerates_corrupt_file():
    get_paths().config_file.write_text("{ not valid json", encoding="utf-8")
    assert load_config() == ToolboxConfig()


def test_update_config_persists_only_known_fields():
    update_config(language="fi", not_a_field="x")
    reloaded = load_config()
    assert reloaded.language == "fi"
    assert not hasattr(reloaded, "not_a_field")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits only")
def test_saved_config_is_owner_only_readable():
    # Regression: HF token must not land in a world-readable file.
    save_config(ToolboxConfig(hf_token="secret"))
    mode = stat.S_IMODE(get_paths().config_file.stat().st_mode)
    assert mode == 0o600
