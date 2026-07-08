"""Tests for merging.presets: lookup, listing, recommendation, config export."""

from ai_toolbox.merging.presets import (
    MergePreset,
    PresetCategory,
    get_preset,
    list_presets,
    get_recommended_preset,
    get_presets_by_category,
)


def test_list_presets_non_empty():
    presets = list_presets()
    assert len(presets) >= 3
    assert all(isinstance(p, MergePreset) for p in presets)


def test_get_preset_by_known_key():
    # PRESETS is keyed by slug (e.g. "slerp_balanced"), not by display name.
    preset = get_preset("slerp_balanced")
    assert isinstance(preset, MergePreset)
    assert preset in list_presets()


def test_get_preset_unknown_is_none():
    assert get_preset("__does_not_exist__") is None


def test_list_presets_filtered_by_category():
    by_cat = get_presets_by_category()
    assert by_cat  # at least one category present
    for category, presets in by_cat.items():
        assert isinstance(category, PresetCategory)
        assert all(p.category == category for p in presets)
        assert list_presets(category=category) == presets


def test_recommended_preset_defaults_for_two_models():
    rec = get_recommended_preset(["some-model-a", "some-model-b"])
    assert rec is not None
    assert isinstance(rec, MergePreset)


def test_preset_to_config_dict_includes_models():
    preset = list_presets()[0]
    config = preset.to_config_dict(["model-1", "model-2"])
    assert isinstance(config, dict) and config
    flat = repr(config)
    assert "model-1" in flat and "model-2" in flat
