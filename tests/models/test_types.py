"""Tests for models.types: format/category enums and ModelEntry defaults."""

from ai_toolbox.models.types import ModelFormat, ModelCategory, ModelEntry


def test_format_from_extension():
    assert ModelFormat.from_extension("safetensors") == ModelFormat.SAFETENSORS
    assert ModelFormat.from_extension("gguf") == ModelFormat.GGUF
    assert ModelFormat.from_extension("bin") == ModelFormat.PYTORCH
    assert ModelFormat.from_extension("pt") == ModelFormat.PYTORCH
    assert ModelFormat.from_extension("xyz") == ModelFormat.UNKNOWN


def test_format_from_extension_ignores_dot_and_case():
    assert ModelFormat.from_extension(".GGUF") == ModelFormat.GGUF


def test_category_from_source():
    assert ModelCategory.from_source("huggingface") == ModelCategory.BASE
    assert ModelCategory.from_source("merged") == ModelCategory.MERGED
    assert ModelCategory.from_source("trained") == ModelCategory.ADAPTER
    assert ModelCategory.from_source("ollama") == ModelCategory.OLLAMA
    assert ModelCategory.from_source("abliterated") == ModelCategory.BASE


def test_category_from_source_unknown_defaults_to_base():
    assert ModelCategory.from_source("something-else") == ModelCategory.BASE


def test_model_entry_defaults():
    e = ModelEntry(
        id="x",
        name="X",
        source="local",
        source_id=None,
        path="/p",
        format="gguf",
        size_bytes=10,
        quantization=None,
        added_date="2026-01-01",
        tags=[],
        metadata={},
    )
    assert e.category == "base"
    assert e.parent_id is None
    assert e.children_ids == []
    assert e.training_info is None
