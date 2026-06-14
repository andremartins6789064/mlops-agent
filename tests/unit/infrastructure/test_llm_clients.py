from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.infrastructure.llm import base as base_module
from src.infrastructure.llm.base import BaseOpenAICompatibleClient
from src.infrastructure.llm.ollama_client import OllamaLLMClient
from src.infrastructure.llm.openai_client import OpenAILLMClient
from src.shared.exceptions import LLMClientError


@dataclass(slots=True)
class _CapturedRequest:
    model: str
    messages: list[dict[str, str]]


class _FakeCompletions:
    def __init__(
        self, *, content: str | None = "ok", should_raise: bool = False
    ) -> None:
        self.content = content
        self.should_raise = should_raise
        self.captured: _CapturedRequest | None = None

    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
        if self.should_raise:
            raise RuntimeError("provider error")
        self.captured = _CapturedRequest(model=model, messages=messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FakeOpenAI:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        completions: _FakeCompletions,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=completions)


def test_base_client_generate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_completions = _FakeCompletions(content=" hello ")

    def _factory(*, base_url: str, api_key: str, timeout: float) -> _FakeOpenAI:
        return _FakeOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            completions=fake_completions,
        )

    monkeypatch.setattr(base_module, "OpenAI", _factory)
    client = BaseOpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="smollm2:1.7b",
    )

    response = client.generate(prompt="Say hi", system_prompt="Use english.")

    assert response == "hello"
    captured = fake_completions.captured
    assert captured is not None
    assert captured.model == "smollm2:1.7b"
    assert captured.messages[0]["role"] == "system"
    assert captured.messages[1]["role"] == "user"


def test_base_client_generate_raises_for_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_completions = _FakeCompletions(content=None)

    def _factory(*, base_url: str, api_key: str, timeout: float) -> _FakeOpenAI:
        return _FakeOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            completions=fake_completions,
        )

    monkeypatch.setattr(base_module, "OpenAI", _factory)
    client = BaseOpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="smollm2:1.7b",
    )

    with pytest.raises(LLMClientError, match="empty"):
        client.generate(prompt="Say hi")


def test_base_client_generate_raises_for_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_completions = _FakeCompletions(should_raise=True)

    def _factory(*, base_url: str, api_key: str, timeout: float) -> _FakeOpenAI:
        return _FakeOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            completions=fake_completions,
        )

    monkeypatch.setattr(base_module, "OpenAI", _factory)
    client = BaseOpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="smollm2:1.7b",
    )

    with pytest.raises(LLMClientError, match="request failed"):
        client.generate(prompt="Say hi")


def test_ollama_client_uses_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_base_init(
        self: BaseOpenAICompatibleClient,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        captured.update(
            {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "timeout_seconds": timeout_seconds,
            }
        )

    monkeypatch.setattr(
        "src.infrastructure.llm.base.BaseOpenAICompatibleClient.__init__",
        _fake_base_init,
    )
    monkeypatch.setattr(
        "src.infrastructure.llm.ollama_client.settings",
        SimpleNamespace(
            llm_base_url="http://localhost:11434/v1",
            llm_api_key="ollama",
            llm_model="smollm2:1.7b",
        ),
    )

    OllamaLLMClient()

    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "ollama"
    assert captured["model"] == "smollm2:1.7b"
    assert captured["timeout_seconds"] == 60.0


def test_openai_client_uses_expected_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_base_init(
        self: BaseOpenAICompatibleClient,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        captured.update(
            {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "timeout_seconds": timeout_seconds,
            }
        )

    monkeypatch.setattr(
        "src.infrastructure.llm.base.BaseOpenAICompatibleClient.__init__",
        _fake_base_init,
    )
    monkeypatch.setattr(
        "src.infrastructure.llm.openai_client.settings",
        SimpleNamespace(
            llm_api_key="openai-key",
            llm_model="gpt-4o-mini",
        ),
    )

    OpenAILLMClient()

    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["api_key"] == "openai-key"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["timeout_seconds"] == 60.0
