"""Unit tests for the vLLM client. Real network calls are replaced by an
httpx MockTransport that hands the client a deterministic response."""

from __future__ import annotations

import json

import httpx
import pytest

from benchmark_colvision.clients.vllm_client import VLLMClient, VLLMSamplingDefaults


def _client_with(handler, *, max_retries: int = 3, backoff_base_s: float = 0.0) -> VLLMClient:
    transport = httpx.MockTransport(handler)
    return VLLMClient(
        endpoint="http://vllm.local:8000",
        model="epfl-llm/meditron-70b",
        client=httpx.Client(transport=transport),
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
    )


def test_complete_happy_path() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"choices": [{"text": "  hello back."}]})

    client = _client_with(handler)
    out = client.complete("ping", temperature=0.0, max_tokens=32, stop=["\n\n"])
    assert out == "  hello back."
    assert seen["url"] == "http://vllm.local:8000/v1/completions"
    body = seen["json"]
    assert body["model"] == "epfl-llm/meditron-70b"
    assert body["prompt"] == "ping"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 32
    assert body["stop"] == ["\n\n"]


def test_complete_uses_sampling_defaults() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"text": "ok"}]})

    transport = httpx.MockTransport(handler)
    client = VLLMClient(
        endpoint="http://vllm.local:8000",
        model="m",
        defaults=VLLMSamplingDefaults(temperature=0.3, max_tokens=64, top_p=0.9),
        client=httpx.Client(transport=transport),
    )
    client.complete("x")
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 64
    assert captured["top_p"] == 0.9


def test_complete_retries_on_5xx() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"text": "finally"}]})

    client = _client_with(handler, max_retries=4)
    assert client.complete("x") == "finally"
    assert calls["n"] == 3


def test_complete_retries_on_request_error() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"choices": [{"text": "ok"}]})

    client = _client_with(handler, max_retries=3)
    assert client.complete("x") == "ok"
    assert calls["n"] == 2


def test_complete_gives_up_after_max_retries() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    client = _client_with(handler, max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("x")


def test_complete_4xx_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = _client_with(handler, max_retries=3)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("x")
    assert calls["n"] == 1


def test_complete_empty_choices_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _client_with(handler)
    with pytest.raises(RuntimeError, match="no choices"):
        client.complete("x")


def test_endpoint_trailing_slash_is_stripped() -> None:
    captured = {"url": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"text": ""}]})

    transport = httpx.MockTransport(handler)
    client = VLLMClient(
        endpoint="http://vllm.local:8000/",
        model="m",
        client=httpx.Client(transport=transport),
    )
    client.complete("x")
    assert captured["url"] == "http://vllm.local:8000/v1/completions"
