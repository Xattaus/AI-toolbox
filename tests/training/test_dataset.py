"""Tests for training.dataset: automatic format detection."""

import json

import pytest

from ai_toolbox.training.dataset import DatasetPrep, DatasetFormat


@pytest.fixture
def prep():
    return DatasetPrep()


def _jsonl(tmp_path, obj, name="d.jsonl"):
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return f


def test_missing_file_is_unknown(prep, tmp_path):
    assert prep.detect_format(tmp_path / "nope.jsonl") == DatasetFormat.UNKNOWN


def test_detect_csv(prep, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    assert prep.detect_format(f) == DatasetFormat.CSV


def test_detect_txt(prep, tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("hello", encoding="utf-8")
    assert prep.detect_format(f) == DatasetFormat.TEXT


def test_detect_chat(prep, tmp_path):
    f = _jsonl(tmp_path, {"messages": [{"role": "user", "content": "hi"}]})
    assert prep.detect_format(f) == DatasetFormat.CHAT


def test_detect_sharegpt(prep, tmp_path):
    f = _jsonl(tmp_path, {"conversations": [{"from": "human", "value": "hi"}]})
    assert prep.detect_format(f) == DatasetFormat.SHAREGPT


def test_detect_alpaca(prep, tmp_path):
    f = _jsonl(tmp_path, {"instruction": "do", "output": "done"})
    assert prep.detect_format(f) == DatasetFormat.ALPACA


def test_detect_completion(prep, tmp_path):
    f = _jsonl(tmp_path, {"prompt": "p", "completion": "c"})
    assert prep.detect_format(f) == DatasetFormat.COMPLETION


def test_detect_json_array(prep, tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps([{"foo": "bar"}]), encoding="utf-8")
    assert prep.detect_format(f) == DatasetFormat.JSON_ARRAY


def test_detect_plain_jsonl(prep, tmp_path):
    f = _jsonl(tmp_path, {"foo": "bar"})
    assert prep.detect_format(f) == DatasetFormat.JSONL


def test_detect_unknown_extension(prep, tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"\x00\x01")
    assert prep.detect_format(f) == DatasetFormat.UNKNOWN
