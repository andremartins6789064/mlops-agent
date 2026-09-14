from __future__ import annotations

import os

import pytest

from src.shared.config import Settings, resolve_provider


def test_settings_default_values() -> None:
    settings = Settings()

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key == "ollama"
    assert settings.llm_model == "smollm2:1.7b"


def test_settings_reads_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = os.getcwd()
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    settings = Settings()

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "gpt-4o-mini"


def test_resolve_provider_uses_named_secret_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    base_url, api_key = resolve_provider("groq")

    assert base_url == "https://api.groq.com/openai/v1"
    assert api_key == "test-groq-key"


def test_resolve_provider_supports_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    base_url, api_key = resolve_provider("openrouter")

    assert base_url == "https://openrouter.ai/api/v1"
    assert api_key == "test-openrouter-key"


def test_resolve_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_provider("unknown")
