from __future__ import annotations

import socket
from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.infrastructure.llm.ollama_client import OllamaLLMClient
from src.infrastructure.parsers.notebook_parser import NotebookParser
from src.shared.config import settings


def _is_ollama_server_available(host: str = "localhost", port: int = 11434) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


F = TypeVar("F", bound=Callable[..., object])
integration_test = cast(Callable[[F], F], pytest.mark.integration)


@integration_test
def test_agents_generate_valid_outputs_with_ollama() -> None:
    if not _is_ollama_server_available():
        pytest.skip("Ollama server is not available on localhost:11434.")

    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    llm_client = OllamaLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=120.0,
    )

    analyzer = NotebookAnalyzerAgent(llm_client=llm_client)
    analysis = analyzer.analyze(notebook)

    assert set(analysis["cells_by_pipeline"].keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert any(
        isinstance(i, int) for v in analysis["cells_by_pipeline"].values() for i in v
    )
    assert isinstance(analysis["libraries"], list)

    architecture_agent = ArchitectureAgent(llm_client=llm_client)
    architecture_plan = architecture_agent.plan(analysis)

    assert set(architecture_plan["modules"].keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert isinstance(architecture_plan["modules"]["training"]["functions"], list)
