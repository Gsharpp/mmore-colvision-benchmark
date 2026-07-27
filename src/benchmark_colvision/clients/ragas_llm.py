"""Build the LangChain-compatible LLM that RAGAS uses as its judge.

vLLM exposes an OpenAI-compatible API, so we point `ChatOpenAI` at the vLLM
endpoint with a placeholder API key. RAGAS expects a `LangchainLLMWrapper`
around that chat model. Both `langchain_openai` and `ragas` are imported
lazily so that environments without a judge installed can still import this
module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    endpoint: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 120.0
    api_key: str = "EMPTY"  # vLLM ignores it but ChatOpenAI requires a non-empty value


def make_ragas_judge(cfg: JudgeConfig):
    """Return a `LangchainLLMWrapper(ChatOpenAI(...))` wired to a vLLM endpoint.

    Raises ImportError with a clear message if `langchain_openai` or `ragas`
    is not installed in the environment.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - tested via monkeypatch
        raise ImportError(
            "langchain_openai is required for the RAGAS judge bridge — "
            "install it on RCP with `uv pip install langchain-openai`"
        ) from e
    try:
        from ragas.llms.base import LangchainLLMWrapper  # type: ignore[import-not-found]  # noqa: PLC0415, I001
    except ImportError as e:  # pragma: no cover - tested via monkeypatch
        raise ImportError(
            "ragas is required for the judge bridge — "
            "ensure `ragas>=0.2` is installed (it is in pyproject)"
        ) from e

    chat = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=f"{cfg.endpoint.rstrip('/')}/v1",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout_s,
    )
    return LangchainLLMWrapper(chat)


def judge_from_env(model: str, endpoint_env: str = "VLLM_ENDPOINT") -> object:
    """Convenience: read the vLLM endpoint from an env var, return the judge."""
    endpoint = os.environ.get(endpoint_env)
    if not endpoint:
        raise RuntimeError(
            f"Environment variable {endpoint_env} is not set — "
            "start a vLLM server first (see scripts/rcp/submit.sh) "
            "and export the printed endpoint URL."
        )
    return make_ragas_judge(JudgeConfig(endpoint=endpoint, model=model))
