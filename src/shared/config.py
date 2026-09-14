"""Application configuration."""

import os
from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Public configuration for one OpenAI-compatible provider."""

    base_url: str
    api_key_env: str
    reviewer_context_budget_tokens: int
    reviewer_inter_call_delay_seconds: float


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "smollm2:1.7b"
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


PROVIDERS = {
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="LLM_API_KEY",
        reviewer_context_budget_tokens=4_000,
        reviewer_inter_call_delay_seconds=0.0,
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        reviewer_context_budget_tokens=1_500,
        reviewer_inter_call_delay_seconds=2.0,
    ),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        reviewer_context_budget_tokens=1_500,
        reviewer_inter_call_delay_seconds=1.0,
    ),
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        reviewer_context_budget_tokens=1_500,
        reviewer_inter_call_delay_seconds=1.0,
    ),
}

_SETTINGS_API_KEY_ATTR = {
    "ollama": "llm_api_key",
    "groq": "groq_api_key",
    "openrouter": "openrouter_api_key",
    "gemini": "gemini_api_key",
}


settings = Settings()


def resolve_provider(provider: str) -> tuple[str, str]:
    """Resolve a provider name into endpoint and API key."""
    config = PROVIDERS.get(provider)
    if config is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available providers: {', '.join(sorted(PROVIDERS))}"
        )

    base_url = config.base_url
    if provider == "ollama":
        base_url = os.getenv("LLM_BASE_URL") or settings.llm_base_url or base_url

    settings_key = getattr(settings, _SETTINGS_API_KEY_ATTR[provider], None)
    api_key = os.getenv(config.api_key_env) or settings_key
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}' (set {config.api_key_env})"
        )
    return base_url, api_key


def reviewer_limits(provider: str | None) -> tuple[int, float]:
    """Return the context budget and spacing configured for a provider."""
    if provider is None:
        return 4_000, 0.0
    config = PROVIDERS.get(provider)
    if config is None:
        raise ValueError(f"Unknown LLM provider '{provider}'")
    return (
        config.reviewer_context_budget_tokens,
        config.reviewer_inter_call_delay_seconds,
    )
