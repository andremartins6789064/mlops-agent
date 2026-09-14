"""Application configuration."""

import os
from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Public configuration for one OpenAI-compatible provider."""

    base_url: str
    api_key_env: str


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "smollm2:1.7b"
    groq_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


PROVIDERS = {
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="LLM_API_KEY",
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
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
    api_key = settings.llm_api_key if provider == "ollama" else None
    if provider == "ollama":
        base_url = settings.llm_base_url or base_url
    else:
        api_key = settings.groq_api_key

    base_url = os.getenv("LLM_BASE_URL", base_url) if provider == "ollama" else base_url
    api_key = os.getenv(config.api_key_env) or api_key
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}' (set {config.api_key_env})"
        )
    return base_url, api_key
