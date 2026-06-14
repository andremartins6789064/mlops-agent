from __future__ import annotations

from src.infrastructure.llm.base import BaseOpenAICompatibleClient
from src.shared.config import settings


class OpenAILLMClient(BaseOpenAICompatibleClient):
    """LLM client configured for OpenAI API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=api_key or settings.llm_api_key,
            model=model or settings.llm_model,
            timeout_seconds=timeout_seconds,
        )
