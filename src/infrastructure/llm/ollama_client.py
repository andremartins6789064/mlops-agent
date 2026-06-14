from __future__ import annotations

from src.infrastructure.llm.base import BaseOpenAICompatibleClient
from src.shared.config import settings


class OllamaLLMClient(BaseOpenAICompatibleClient):
    """LLM client configured for an Ollama OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            model=model or settings.llm_model,
            timeout_seconds=timeout_seconds,
        )
