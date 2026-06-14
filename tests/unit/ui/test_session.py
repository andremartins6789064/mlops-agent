from __future__ import annotations

from src.agents.orchestrator import OrchestrationResult
from src.domain.entities import CellType, Notebook, NotebookCell
from src.ui.session import default_config, get_result


def test_default_config_has_expected_defaults() -> None:
    config = default_config()
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_model == "smollm2:1.7b"
    assert config.output_dir == "output/ui_runs"


def test_get_result_returns_orchestration_result_only() -> None:
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    result = OrchestrationResult(
        notebook=notebook,
        notebook_analysis={},
        architecture_plan={},
    )
    assert get_result({"mlops_result": result}) is result
    assert get_result({"mlops_result": "invalid"}) is None
