from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

import src.application.convert_notebook as convert_notebook_module
from src.agents.orchestrator import OrchestrationResult
from src.application.convert_notebook import (
    ConversionRequest,
    _build_llm_client,
    convert_notebook,
)
from src.domain.entities import CellType, Notebook, NotebookCell


def test_build_llm_client_returns_none_when_disabled() -> None:
    request = ConversionRequest(notebook_path="sample.ipynb", use_llm=False)
    assert _build_llm_client(request) is None


def test_build_llm_client_returns_client_when_enabled() -> None:
    request = ConversionRequest(
        notebook_path="sample.ipynb",
        use_llm=True,
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="token",
        llm_model="smollm2:1.7b",
    )
    client = _build_llm_client(request)
    assert client is not None


def test_build_llm_client_passes_configured_timeout(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        convert_notebook_module,
        "BaseOpenAICompatibleClient",
        _StubClient,
    )

    client = _build_llm_client(
        ConversionRequest(
            notebook_path="sample.ipynb",
            use_llm=True,
            llm_timeout_seconds=123.0,
        )
    )

    assert client is not None
    assert captured["timeout_seconds"] == 123.0


def test_convert_notebook_runs_orchestrator(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    expected_result = OrchestrationResult(
        notebook=notebook,
        notebook_analysis={},
        architecture_plan={},
    )

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured["has_exporter"] = str("exporter" in kwargs)

        def run(
            self, notebook_path: str, *, output_dir: str | None = None
        ) -> OrchestrationResult:
            captured["notebook_path"] = notebook_path
            captured["output_dir"] = output_dir or ""
            return expected_result

    monkeypatch.setattr(convert_notebook_module, "Orchestrator", _StubOrchestrator)

    result = convert_notebook(
        ConversionRequest(
            notebook_path="tests/fixtures/simple_regression.ipynb",
            output_dir="output/test-ui",
            use_llm=False,
        )
    )

    assert captured["has_exporter"] == "True"
    assert captured["notebook_path"].endswith("simple_regression.ipynb")
    assert captured["output_dir"] == "output/test-ui"
    assert result is expected_result


def test_convert_notebook_passes_feedback_to_agents(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    expected_result = OrchestrationResult(
        notebook=notebook,
        notebook_analysis={},
        architecture_plan={},
    )

    class _StubArchitectureAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["architecture_feedback"] = kwargs.get("user_feedback")

    class _StubCodeGeneratorAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["stage_feedback"] = kwargs.get("stage_feedback")

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured["architecture_agent"] = kwargs.get("architecture_agent")
            captured["code_generator"] = kwargs.get("code_generator")

        def run(
            self, notebook_path: str, *, output_dir: str | None = None
        ) -> OrchestrationResult:
            return expected_result

    monkeypatch.setattr(
        convert_notebook_module, "ArchitectureAgent", _StubArchitectureAgent
    )
    monkeypatch.setattr(
        convert_notebook_module, "CodeGeneratorAgent", _StubCodeGeneratorAgent
    )
    monkeypatch.setattr(convert_notebook_module, "Orchestrator", _StubOrchestrator)

    convert_notebook(
        ConversionRequest(
            notebook_path="tests/fixtures/simple_regression.ipynb",
            output_dir="output/test-ui",
            architecture_feedback="Prefer explicit contracts",
            stage_feedback={"training": "Add model validation"},
        )
    )
    assert captured["architecture_feedback"] == "Prefer explicit contracts"
    assert captured["stage_feedback"] == {"training": "Add model validation"}
