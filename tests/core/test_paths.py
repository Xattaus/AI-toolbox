"""Tests for core.paths: singleton behaviour, root detection, model paths."""

from pathlib import Path

from ai_toolbox.core.paths import get_paths, reset_paths


def test_get_paths_is_singleton():
    assert get_paths() is get_paths()


def test_reset_paths_rebuilds_instance():
    first = get_paths()
    reset_paths()
    assert get_paths() is not first


def test_root_follows_env(isolated_toolbox_root):
    assert get_paths().root == Path(isolated_toolbox_root)


def test_directories_are_created_under_root(isolated_toolbox_root):
    p = get_paths()
    for d in (p.models_dir, p.gguf_dir, p.safetensors_dir, p.merged_dir,
              p.abliterated_dir, p.config_dir, p.datasets_dir):
        assert d.exists() and d.is_dir()
        assert Path(isolated_toolbox_root) in d.parents or d == Path(isolated_toolbox_root)


def test_model_type_directories_differ():
    p = get_paths()
    assert p.gguf_dir != p.safetensors_dir
    assert p.merged_dir != p.abliterated_dir


def test_get_model_path_maps_known_types():
    p = get_paths()
    assert p.get_model_path("m.gguf", "gguf") == p.gguf_dir / "m.gguf"
    assert p.get_model_path("m", "safetensors") == p.safetensors_dir / "m"
    assert p.get_model_path("m", "merged") == p.merged_dir / "m"
    assert p.get_model_path("m", "abliterated") == p.abliterated_dir / "m"


def test_get_model_path_aliases():
    p = get_paths()
    # 'download' is a legacy alias for safetensors; 'lora' for adapters
    assert p.get_model_path("m", "download") == p.safetensors_dir / "m"
    assert p.get_model_path("m", "lora") == p.adapters_dir / "m"


def test_get_model_path_unknown_type_falls_back_to_models_dir():
    p = get_paths()
    assert p.get_model_path("m", "nonsense") == p.models_dir / "m"


def test_config_and_library_files_under_root(isolated_toolbox_root):
    p = get_paths()
    assert Path(isolated_toolbox_root) in p.config_file.parents
    assert Path(isolated_toolbox_root) in p.library_file.parents
