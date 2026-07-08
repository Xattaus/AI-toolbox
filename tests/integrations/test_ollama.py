"""Smoke tests for integrations.ollama: Modelfile generation and name validation.

These exercise the pure logic (no running Ollama required); OllamaManager's
constructor only probes for the `ollama` binary and degrades gracefully.
"""

import pytest

from ai_toolbox.integrations.ollama import OllamaManager, ModelfileConfig


@pytest.fixture
def mgr():
    return OllamaManager()


def test_modelfile_from_uses_forward_slashes(mgr):
    mf = mgr.generate_modelfile(ModelfileConfig(gguf_path=r"C:\models\m.gguf"))
    assert "FROM C:/models/m.gguf" in mf
    assert "\\" not in mf.splitlines()[0]


def test_modelfile_includes_default_parameters(mgr):
    mf = mgr.generate_modelfile(ModelfileConfig(gguf_path="m.gguf"))
    assert "PARAMETER temperature 0.7" in mf
    assert "PARAMETER num_ctx 4096" in mf
    assert "PARAMETER top_k 40" in mf


def test_modelfile_system_prompt_block(mgr):
    mf = mgr.generate_modelfile(ModelfileConfig(gguf_path="m.gguf", system_prompt="Be helpful."))
    assert 'SYSTEM """Be helpful."""' in mf


def test_modelfile_custom_template_block(mgr):
    mf = mgr.generate_modelfile(ModelfileConfig(gguf_path="m.gguf", chat_template="{{ .Prompt }}"))
    assert 'TEMPLATE """{{ .Prompt }}"""' in mf


def test_modelfile_stop_tokens(mgr):
    mf = mgr.generate_modelfile(
        ModelfileConfig(gguf_path="m.gguf", stop_tokens=["<|end|>", "</s>"])
    )
    assert 'PARAMETER stop "<|end|>"' in mf
    assert 'PARAMETER stop "</s>"' in mf


def test_modelfile_template_name_lookup(mgr):
    prompts = mgr.get_system_prompts()
    assert prompts  # there is at least one built-in template
    key = next(iter(prompts))
    mf = mgr.generate_modelfile(ModelfileConfig(gguf_path="m.gguf", template_name=key))
    assert "SYSTEM " in mf


def test_validate_model_name_rejects_empty(mgr):
    ok, _ = mgr.validate_model_name("")
    assert ok is False


def test_validate_model_name_rejects_too_long(mgr):
    ok, _ = mgr.validate_model_name("a" * 65)
    assert ok is False


def test_validate_model_name_rejects_bad_chars(mgr):
    ok, _ = mgr.validate_model_name("bad name!")
    assert ok is False


def test_validate_model_name_accepts_valid(mgr):
    ok, _ = mgr.validate_model_name("aitb-smoketest_model1")
    assert ok is True
