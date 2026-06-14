from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from src.agents.orchestrator import OrchestrationResult
from src.domain.entities import CellType, Notebook, NotebookCell
from src.domain.value_objects import QualityMetrics
from src.ui.workflow import (
    create_temp_notebook_path,
    get_stage_code,
    get_stage_tests,
    result_ready,
    run_conversion,
    summarize_review_status,
)


def _build_result() -> OrchestrationResult:
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    return OrchestrationResult(
        notebook=notebook,
        notebook_analysis={},
        architecture_plan={},
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={
            "training": "def test_train_model() -> None:\n    assert True\n"
        },
    )


def test_result_ready_depends_on_session_key() -> None:
    assert not result_ready({})
    assert result_ready({"mlops_result": "non-empty"})


def test_get_stage_code_and_tests_return_expected_values() -> None:
    result = _build_result()
    assert get_stage_code(result, stage_name="training") is not None
    assert get_stage_tests(result, stage_name="training") is not None
    assert get_stage_code(result, stage_name="inference") is None


def test_summarize_review_status_handles_missing_metrics() -> None:
    assert summarize_review_status(None) == "Métricas de revisão ainda não disponíveis."


def test_summarize_review_status_formats_values() -> None:
    status = summarize_review_status(
        QualityMetrics(
            lint_errors=0, type_errors=1, test_coverage=81.5, review_iterations=1
        )
    )
    assert "cobertura=81.5%" in status
    assert "tipos=1" in status


def test_create_temp_notebook_path_preserves_filename() -> None:
    notebook_path = create_temp_notebook_path(notebook_name="analysis.ipynb")
    assert notebook_path.name == "analysis.ipynb"


def test_run_conversion_forwards_feedback(
    monkeypatch: MonkeyPatch, tmp_path: object
) -> None:
    captured: dict[str, object] = {}
    result = _build_result()

    def _stub_convert_notebook(request: object) -> OrchestrationResult:
        captured["request"] = request
        return result

    monkeypatch.setattr("src.ui.workflow.convert_notebook", _stub_convert_notebook)

    from src.ui.session import UIConfig

    output_dir = str(tmp_path)
    conversion_result = run_conversion(
        notebook_bytes=b'{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}',
        notebook_name="demo.ipynb",
        config=UIConfig(output_dir=output_dir),
        architecture_feedback="Use cleaner modules",
        stage_feedback={"training": "Add train/valid split"},
    )
    request = captured["request"]
    assert conversion_result is result
    assert getattr(request, "architecture_feedback") == "Use cleaner modules"
    assert getattr(request, "stage_feedback") == {"training": "Add train/valid split"}
