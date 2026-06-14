from __future__ import annotations

import os

import pytest

from src.shared.config import Settings


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
