"""Tests for conversion.llama_cpp: Zip Slip guard, version pin, tool lookup."""

import zipfile

import pytest

from ai_toolbox.conversion.llama_cpp import LlamaCppManager, LLAMA_CPP_TAG


def _make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_version_is_pinned():
    # Regression: reproducible builds require a concrete tag, not a range.
    assert isinstance(LLAMA_CPP_TAG, str)
    assert LLAMA_CPP_TAG.startswith("b") and LLAMA_CPP_TAG[1:].isdigit()


def test_safe_extractall_extracts_normal_members(tmp_path):
    zip_path = _make_zip(tmp_path / "ok.zip", {"a.txt": "1", "sub/b.txt": "2"})
    dest = tmp_path / "out"
    with zipfile.ZipFile(zip_path) as z:
        LlamaCppManager._safe_extractall(z, dest)
    assert (dest / "a.txt").read_text() == "1"
    assert (dest / "sub" / "b.txt").read_text() == "2"


def test_safe_extractall_blocks_parent_traversal(tmp_path):
    zip_path = _make_zip(tmp_path / "evil.zip", {"ok.txt": "1", "../escape.txt": "pwned"})
    dest = tmp_path / "out"
    with zipfile.ZipFile(zip_path) as z:
        with pytest.raises(RuntimeError, match="Zip Slip"):
            LlamaCppManager._safe_extractall(z, dest)
    # nothing should have escaped the destination
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extractall_blocks_absolute_path(tmp_path):
    # Zip members with a drive/leading-slash path resolve outside dest.
    zip_path = tmp_path / "abs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../../../../etc/passwd", "x")
    dest = tmp_path / "out"
    with zipfile.ZipFile(zip_path) as z:
        with pytest.raises(RuntimeError, match="Zip Slip"):
            LlamaCppManager._safe_extractall(z, dest)


def test_check_status_when_not_installed(tmp_path):
    mgr = LlamaCppManager(str(tmp_path / "missing"))
    status = mgr.check_status()
    assert status["installed"] is False
    assert status["convert_script"] is None


def test_find_quantize_binary_absent_returns_none(tmp_path):
    mgr = LlamaCppManager(str(tmp_path / "empty"))
    assert mgr.find_quantize_binary() is None


def test_find_quantize_binary_locates_binary(tmp_path):
    root = tmp_path / "llama"
    (root / "build" / "bin").mkdir(parents=True)
    binary = root / "build" / "bin" / "llama-quantize"
    binary.write_text("#!/bin/sh\n")
    mgr = LlamaCppManager(str(root))
    found = mgr.find_quantize_binary()
    assert found is not None and found.name.startswith("llama-quantize")
