from __future__ import annotations

import time

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
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
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

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                )
                break
            except Exception as exc:  # pragma: no cover - network/provider dependent
                if _is_rate_limit_error(exc) and attempt < self._max_retries:
                    time.sleep(self._retry_backoff_seconds * (2**attempt))
                    continue
                detail = _safe_error_detail(exc, api_key=self._api_key)
                msg = f"LLM request failed: {detail}"
                raise LLMClientError(msg) from exc

        raw_content = response.choices[0].message.content if response.choices else None
        if raw_content is None:
            msg = "LLM response is empty."
            raise LLMClientError(msg)

        if not isinstance(raw_content, str):
            msg = "LLM response content is not text."
            raise LLMClientError(msg)

        return raw_content.strip()


def _is_rate_limit_error(error: Exception) -> bool:
    """Return whether an exception represents an HTTP 429 response."""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 429


def _safe_error_detail(error: Exception, *, api_key: str) -> str:
    """Format provider errors without exposing credentials."""
    detail = str(error).strip() or type(error).__name__
    return detail.replace(api_key, "[REDACTED]") if api_key else detail
