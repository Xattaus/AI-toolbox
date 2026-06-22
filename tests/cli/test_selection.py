from pathlib import Path
from types import SimpleNamespace

from ai_toolbox.cli.selection import build_conversion_choices


def _model(name, path, size, source="local"):
    # Mimics models.types.ModelEntry (name/path/size_bytes/source)
    return SimpleNamespace(name=name, path=path, size_bytes=size, source=source)


def test_library_models_first_with_library_source():
    items = build_conversion_choices([_model("Qwen2.5", "/m/qwen", 100)], [])
    assert len(items) == 1
    assert items[0]["source"] == "library"
    assert items[0]["path"] == Path("/m/qwen")
    assert items[0]["name"] == "Qwen2.5"
    assert items[0]["size_bytes"] == 100


def test_downloads_appended_after_library_with_download_source():
    convertible = [_model("LibA", "/m/a", 1)]
    downloaded = [{"model_id": "org/RepoB", "path": "/d/b", "size": 2}]
    items = build_conversion_choices(convertible, downloaded)
    assert [i["source"] for i in items] == ["library", "download"]
    assert items[1]["name"] == "RepoB"          # org/ prefix stripped
    assert items[1]["path"] == Path("/d/b")
    assert items[1]["size_bytes"] == 2


def test_empty_inputs_give_empty_list():
    assert build_conversion_choices([], []) == []


def test_download_without_slash_keeps_full_id():
    items = build_conversion_choices([], [{"model_id": "localmodel", "path": "/d/x", "size": 5}])
    assert items[0]["name"] == "localmodel"
    assert items[0]["source"] == "download"
