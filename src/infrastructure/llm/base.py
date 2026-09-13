from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.domain.interfaces import ILLMClient
from src.shared.exceptions import LLMClientError


class BaseOpenAICompatibleClient(ILLMClient):
    """OpenAI-compatible chat completions client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate text from user prompt and optional system instruction."""
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except Exception as exc:  # pragma: no cover - network/provider dependent
            msg = "LLM request failed."
            raise LLMClientError(msg) from exc

        raw_content = response.choices[0].message.content if response.choices else None
        if raw_content is None:
            msg = "LLM response is empty."
            raise LLMClientError(msg)

        if not isinstance(raw_content, str):
            msg = "LLM response content is not text."
            raise LLMClientError(msg)

        return raw_content.strip()
