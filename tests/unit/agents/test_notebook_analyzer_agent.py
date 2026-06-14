from __future__ import annotations

import json

from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.domain.interfaces import ILLMClient
from src.infrastructure.parsers.notebook_parser import NotebookParser


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self._response


def test_analyzer_uses_llm_json_when_valid() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    llm_payload = {
        "cells_by_pipeline": {
            "feature_engineering": [1, 2],
            "training": [3],
            "inference": [],
            "evaluation": [],
        },
        "libraries": ["pandas", "sklearn"],
        "shared_variables": ["X", "y"],
        "summary": "Valid LLM response.",
    }
    analyzer = NotebookAnalyzerAgent(llm_client=_StubLLMClient(json.dumps(llm_payload)))

    result = analyzer.analyze(notebook)

    assert result["summary"] == "Valid LLM response."
    assert notebook.cells[3].pipeline_type.value == "training"


def test_analyzer_falls_back_to_heuristics_when_llm_json_is_invalid() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    analyzer = NotebookAnalyzerAgent(llm_client=_StubLLMClient("not-json"))

    result = analyzer.analyze(notebook)

    assert "deterministic rules" in result["summary"]
    assert 3 in result["cells_by_pipeline"]["training"]


def test_analyzer_falls_back_when_llm_schema_is_invalid() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    invalid_payload = {"cells_by_pipeline": {"feature_engineering": [1]}}
    analyzer = NotebookAnalyzerAgent(
        llm_client=_StubLLMClient(json.dumps(invalid_payload))
    )

    result = analyzer.analyze(notebook)

    assert result["cells_by_pipeline"]["feature_engineering"]
    assert notebook.cells[0].pipeline_type.value == "unknown"
