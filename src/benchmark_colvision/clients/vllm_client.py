"""HTTP client for a vLLM server exposing the OpenAI-compatible API.

Satisfies the `LLMClient` protocol expected by
`benchmark_colvision.queries.inverse_query_gen.LLMClient` (a single `complete`
method).

Two transports, selected by `chat`:
  - chat=False → `/v1/completions` (raw prompt). Works with any model regardless
    of chat-template availability; right for base LLMs.
  - chat=True  → `/v1/chat/completions` (prompt wrapped as a single user
    message). Applies the model's chat template, which is what instruction-tuned
    models (e.g. Qwen2.5-Instruct) need to actually follow the JSON instruction.

`guided_json` is forwarded as a top-level field on both endpoints (a vLLM
extension to the OpenAI schema) so guided decoding works either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class VLLMSamplingDefaults:
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0


class VLLMClient:
    """Minimal vLLM (OpenAI-compatible) completion client with retries."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        defaults: VLLMSamplingDefaults | None = None,
        chat: bool = False,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.chat = chat
        self.defaults = defaults or VLLMSamplingDefaults()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VLLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        guided_json: dict | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": self.defaults.temperature if temperature is None else temperature,
            "max_tokens": self.defaults.max_tokens if max_tokens is None else max_tokens,
            "top_p": self.defaults.top_p if top_p is None else top_p,
        }
        if self.chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
            url = f"{self.endpoint}/v1/chat/completions"
        else:
            payload["prompt"] = prompt
            url = f"{self.endpoint}/v1/completions"
        if stop:
            payload["stop"] = stop
        if guided_json is not None:
            payload["guided_json"] = guided_json
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=payload)
            except httpx.RequestError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            if 500 <= resp.status_code < 600:  # noqa: PLR2004 — HTTP 5xx range
                last_exc = httpx.HTTPStatusError(
                    f"vLLM returned {resp.status_code}", request=resp.request, response=resp
                )
                self._sleep_backoff(attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"vLLM response had no choices: {data!r}")
            if self.chat:
                text = (choices[0].get("message") or {}).get("content", "")
            else:
                text = choices[0].get("text", "")
            if not isinstance(text, str):
                raise RuntimeError(f"vLLM choice text is not a string: {choices[0]!r}")
            return text
        assert last_exc is not None
        raise last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt + 1 < self.max_retries:
            time.sleep(self.backoff_base_s * (2**attempt))
