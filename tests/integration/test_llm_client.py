from __future__ import annotations

import socket
from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from src.infrastructure.llm.ollama_client import OllamaLLMClient

F = TypeVar("F", bound=Callable[..., object])
integration_test = cast(Callable[[F], F], pytest.mark.integration)


def _is_ollama_server_available(host: str = "localhost", port: int = 11434) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@integration_test
def test_ollama_client_generates_text() -> None:
    if not _is_ollama_server_available():
        pytest.skip("Ollama server is not available on localhost:11434.")

    client = OllamaLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="smollm2:1.7b",
        timeout_seconds=120.0,
    )

    response = client.generate(
        prompt="Answer with one word: online or offline?",
        system_prompt=(
            "You are a concise assistant. Answer in English and return only one word."
        ),
    )

    assert isinstance(response, str)
    assert response.strip() != ""
