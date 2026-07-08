"""Tests for models.library: naming helpers, sorting, and library CRUD/persistence."""

import pytest

from ai_toolbox.models.library import (
    ModelLibrary,
    extract_model_identity,
    generate_merge_name,
    get_sort_key,
)
from ai_toolbox.models.types import ModelEntry


def _gguf(tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    return f


def _entry(**kw):
    base = dict(
        id="i",
        name="N",
        source="local",
        source_id=None,
        path="/p",
        format="gguf",
        size_bytes=0,
        quantization=None,
        added_date="2026-01-01",
        tags=[],
        metadata={},
    )
    base.update(kw)
    return ModelEntry(**base)


# --- pure naming / sorting helpers ---


def test_extract_identity_strips_path():
    out = extract_model_identity("meta-llama/Some-Model")
    assert "/" not in out and "\\" not in out and out


def test_extract_identity_documented_example():
    assert extract_model_identity("LumiOpen_Llama-Poro-2-8B-SFT-abliterated") == "Poro"


def test_generate_merge_name_two_models():
    assert generate_merge_name("slerp", ["Poro", "DarkIdol"]) == "Slerp_Poro-DarkIdol"


def test_generate_merge_name_dedupes_identical():
    name = generate_merge_name("ties", ["Poro", "Poro"])
    assert name == "Ties_Poro"
    assert "-" not in name.split("_", 1)[1]


def test_generate_merge_name_unknown_method_prefix():
    assert generate_merge_name("weirdmethod", ["Poro", "DarkIdol"]).startswith("Weirdm")


def test_generate_merge_name_truncates_many_models():
    name = generate_merge_name("slerp", ["A1", "B2", "C3", "D4", "E5", "F6"])
    assert "-+" in name  # >4 models collapsed into "-+N"


def test_get_sort_key_size_and_name():
    big = _entry(name="Zeta", size_bytes=100)
    small = _entry(name="alpha", size_bytes=50)
    assert get_sort_key(big, "size") > get_sort_key(small, "size")
    assert get_sort_key(small, "name") < get_sort_key(big, "name")


def test_get_sort_key_quant_orders_by_quality():
    assert get_sort_key(_entry(quantization="Q8_0"), "quant") > get_sort_key(
        _entry(quantization="Q4_K_M"), "quant"
    )


# --- ModelLibrary CRUD ---


def test_add_detects_format_and_quant(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    entry = lib.add_model(str(_gguf(tmp_path, "mymodel-q4_k_m.gguf")))
    assert entry.format == "gguf"
    assert entry.quantization == "Q4_K_M"
    assert lib.get_model(entry.id) is entry


def test_add_missing_path_raises(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    with pytest.raises(FileNotFoundError):
        lib.add_model(str(tmp_path / "ghost.gguf"))


def test_add_same_path_twice_is_deduped(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    f = _gguf(tmp_path, "dup.gguf")
    e1 = lib.add_model(str(f))
    e2 = lib.add_model(str(f))
    assert e1.id == e2.id
    assert len(lib.list_models()) == 1


def test_remove_model(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    e = lib.add_model(str(_gguf(tmp_path, "m.gguf")))
    assert lib.remove_model(e.id) is True
    assert lib.get_model(e.id) is None
    assert lib.remove_model("nonexistent") is False


def test_update_and_tags(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    e = lib.add_model(str(_gguf(tmp_path, "m.gguf")), name="M")
    assert lib.add_tag(e.id, "fav") is True
    assert "fav" in lib.get_model(e.id).tags
    assert lib.remove_tag(e.id, "fav") is True
    assert "fav" not in lib.get_model(e.id).tags
    assert lib.update_model(e.id, name="Renamed") is True
    assert lib.get_model(e.id).name == "Renamed"


def test_search_models(tmp_path):
    lib = ModelLibrary(library_path=str(tmp_path / "lib"), auto_scan=False)
    lib.add_model(str(_gguf(tmp_path, "llama.gguf")), name="Llama-Base")
    lib.add_model(str(_gguf(tmp_path, "mistral.gguf")), name="Mistral-Base")
    results = lib.search_models("llama")
    assert [m.name for m in results] == ["Llama-Base"]


# --- persistence / crash recovery ---


def test_index_persists_across_instances(tmp_path):
    libdir = str(tmp_path / "lib")
    f = _gguf(tmp_path, "persist-q8_0.gguf")
    added = ModelLibrary(library_path=libdir, auto_scan=False).add_model(str(f), name="Persisted")

    reloaded = ModelLibrary(library_path=libdir, auto_scan=False).get_model(added.id)
    assert reloaded is not None
    assert reloaded.name == "Persisted"
    assert reloaded.quantization == "Q8_0"


def test_corrupt_index_recovers_from_backup(tmp_path):
    libdir = tmp_path / "lib"
    lib = ModelLibrary(library_path=str(libdir), auto_scan=False)
    lib.add_model(str(_gguf(tmp_path, "one.gguf")), name="One")  # save 1 (no .bak yet)
    lib.add_model(str(_gguf(tmp_path, "two.gguf")), name="Two")  # save 2 -> .bak holds [One]

    backup = libdir / "library_index.json.bak"
    assert backup.exists()
    (libdir / "library_index.json").write_text("{ corrupt json", encoding="utf-8")

    recovered = ModelLibrary(library_path=str(libdir), auto_scan=False)
    assert {m.name for m in recovered.list_models()} == {"One"}
