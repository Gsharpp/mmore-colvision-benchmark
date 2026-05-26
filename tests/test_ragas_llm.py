"""Tests for the RAGAS judge factory. langchain_openai and ragas are imported
lazily inside the factory; we patch the import sites to avoid pulling them
into the test runtime."""

from __future__ import annotations

import sys
import types

import pytest

from benchmark_colvision.clients.ragas_llm import (
    JudgeConfig,
    judge_from_env,
    make_ragas_judge,
)


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install minimal fake langchain_openai and ragas modules in sys.modules."""
    captured: dict = {}

    class FakeChat:
        def __init__(self, **kwargs) -> None:
            captured["chat_kwargs"] = kwargs

    class FakeWrapper:
        def __init__(self, chat) -> None:
            captured["wrapped_chat"] = chat

    fake_langchain_openai = types.ModuleType("langchain_openai")
    fake_langchain_openai.ChatOpenAI = FakeChat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)

    fake_ragas = types.ModuleType("ragas")
    fake_ragas_llms = types.ModuleType("ragas.llms")
    fake_ragas_base = types.ModuleType("ragas.llms.base")
    fake_ragas_base.LangchainLLMWrapper = FakeWrapper  # type: ignore[attr-defined]
    fake_ragas_llms.base = fake_ragas_base  # type: ignore[attr-defined]
    fake_ragas.llms = fake_ragas_llms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.llms", fake_ragas_llms)
    monkeypatch.setitem(sys.modules, "ragas.llms.base", fake_ragas_base)

    return captured


def test_make_ragas_judge_passes_expected_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_modules(monkeypatch)
    cfg = JudgeConfig(
        endpoint="http://node-7:8000/",  # trailing slash to verify it gets stripped
        model="epfl-llm/meditron-70b",
        temperature=0.0,
        max_tokens=256,
        timeout_s=60.0,
    )
    wrapper = make_ragas_judge(cfg)
    assert wrapper is not None
    kwargs = captured["chat_kwargs"]
    assert kwargs["model"] == "epfl-llm/meditron-70b"
    assert kwargs["base_url"] == "http://node-7:8000/v1"
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout"] == 60.0


def test_judge_from_env_reads_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_modules(monkeypatch)
    monkeypatch.setenv("VLLM_ENDPOINT", "http://node-9:8000")
    out = judge_from_env(model="meditron-70b")
    assert out is not None


def test_judge_from_env_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_modules(monkeypatch)
    monkeypatch.delenv("VLLM_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_ENDPOINT"):
        judge_from_env(model="meditron-70b")
