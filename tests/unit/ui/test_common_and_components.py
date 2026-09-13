from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.orchestrator import OrchestrationResult
from src.application.validate_output import ValidationResult
from src.domain.entities import CellType, Notebook, NotebookCell
from src.domain.value_objects import QualityMetrics
from src.shared.provenance import StageProvenance
from src.ui.components import (
    cell_stage_table,
    code_viewer,
    metrics_card,
    provenance_card,
    validation_details,
)
from src.ui.pages import common
from src.ui.session import (
    CONFIG_KEY,
    ERROR_KEY,
    NOTEBOOK_BYTES_KEY,
    NOTEBOOK_NAME_KEY,
    RESULT_KEY,
    UIConfig,
)


@dataclass
class _FakeSpinner:
    def __enter__(self) -> _FakeSpinner:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.messages: list[str] = []

    def subheader(self, message: str) -> None:
        self.messages.append(message)

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)

    def code(self, source_code: str, language: str = "python") -> None:
        self.messages.append(f"code:{language}:{len(source_code)}")

    def title(self, message: str) -> None:
        self.messages.append(message)

    def json(self, payload: dict[str, Any]) -> None:
        self.messages.append(str(payload))

    def dataframe(self, data: object, **kwargs: object) -> None:
        self.messages.append(f"dataframe:{data}")

    def expander(self, label: str) -> _FakeSpinner:
        self.messages.append(label)
        return _FakeSpinner()

    def write(self, message: object) -> None:
        self.messages.append(str(message))

    def text_area(self, label: str, **kwargs: object) -> str:
        self.messages.append(label)
        return ""

    def caption(self, message: str) -> None:
        self.messages.append(message)

    def button(self, label: str, **kwargs: object) -> bool:
        self.messages.append(label)
        return False

    def columns(self, ncols: int) -> list[_FakeStreamlit]:
        return [self for _ in range(ncols)]

    def metric(self, label: str, value: str) -> None:
        self.messages.append(f"{label}:{value}")

    def download_button(self, **kwargs: object) -> None:
        self.messages.append(str(kwargs.get("label", "download")))

    def spinner(self, message: str) -> _FakeSpinner:
        self.messages.append(message)
        return _FakeSpinner()

    def success(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)


def _build_result(tmp_path: Any) -> OrchestrationResult:
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    zip_path = tmp_path / "demo.zip"
    zip_path.write_bytes(b"zip")
    return OrchestrationResult(
        notebook=notebook,
        notebook_analysis={"libraries": ["pandas"]},
        architecture_plan={"modules": {}},
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_file_paths={"training": "output/training.py"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
        generated_test_file_paths={"training": "output/tests/test_training.py"},
        stage_provenance={
            "training": StageProvenance(
                origin="llm",
                model="test-model",
                duration_seconds=1.25,
                parse_method="raw",
            )
        },
        quality_metrics=QualityMetrics(test_coverage=88.0),
        exported_zip_path=str(zip_path),
    )


def test_components_render_with_fake_streamlit(monkeypatch: Any) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(code_viewer, "st", fake_st)
    monkeypatch.setattr(metrics_card, "st", fake_st)
    monkeypatch.setattr(cell_stage_table, "st", fake_st)
    monkeypatch.setattr(validation_details, "st", fake_st)

    code_viewer.render_code_viewer(title="Code", source_code=None)
    code_viewer.render_code_viewer(
        title="Code", source_code="def f() -> None:\n    return\n"
    )
    metrics_card.render_metrics_card(None)
    metrics_card.render_metrics_card(QualityMetrics(test_coverage=90.0))
    cell_stage_table.render_cell_stage_table(
        notebook=Notebook(
            path="demo.ipynb",
            cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="x = 1")],
            metadata={},
        ),
        analysis={"cells_by_pipeline": {"training": [0]}},
    )
    validation_details.render_validation_details(
        validation=ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=88.0,
            lint_output="ruff passed",
            type_output="mypy passed",
            test_output="1 passed",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        ),
        validated_output_dir="output/generated",
        generated_file_paths={"training": "output/generated/src/training.py"},
    )

    assert any("Ainda não há conteúdo" in message for message in fake_st.messages)
    assert any("Cobertura" in message for message in fake_st.messages)
    assert any("dataframe:" in message for message in fake_st.messages)
    assert any("Validação real do artefato" in message for message in fake_st.messages)
    assert any("Saída do pytest" in message for message in fake_st.messages)


def test_build_cell_stage_rows_preserves_order_and_shortens_code() -> None:
    notebook = Notebook(
        path="demo.ipynb",
        cells=[
            NotebookCell(index=2, cell_type=CellType.CODE, source="x = 1\nprint(x)"),
            NotebookCell(index=5, cell_type=CellType.CODE, source=""),
        ],
        metadata={},
    )

    rows = cell_stage_table.build_cell_stage_rows(
        notebook=notebook,
        analysis={"cells_by_pipeline": {"training": [2]}},
    )

    assert rows == [
        {"Índice": 2, "Código": "x = 1 print(x)", "Estágio": "training"},
        {"Índice": 5, "Código": "(célula vazia)", "Estágio": "não atribuído"},
    ]


def test_common_helpers_update_session_and_render(
    monkeypatch: Any, tmp_path: Any
) -> None:
    fake_st = _FakeStreamlit()
    result = _build_result(tmp_path)
    fake_st.session_state[NOTEBOOK_NAME_KEY] = "simple_regression.ipynb"
    fake_st.session_state[CONFIG_KEY] = UIConfig()
    fake_st.session_state[NOTEBOOK_BYTES_KEY] = b"{}"
    monkeypatch.setattr(common, "st", fake_st)
    monkeypatch.setattr(
        common,
        "render_stage_provenance",
        lambda **kwargs: fake_st.messages.append(f"provenance:{kwargs['stage_name']}"),
    )
    monkeypatch.setattr(common, "run_conversion", lambda **kwargs: result)

    common.reset_result_state()
    common.set_result_state(
        result=result,
        error="",
        notebook_name="simple_regression.ipynb",
        notebook_bytes=b"{}",
    )
    assert RESULT_KEY in fake_st.session_state
    assert ERROR_KEY in fake_st.session_state

    loaded = common.read_result()
    assert loaded is result

    common.render_analysis_blocks(result)
    common.render_stage_page(stage_name="training", title="Training")
    common.render_download_section(result)
    common.regenerate_from_feedback()

    assert any("Saída do Notebook Analyzer" in message for message in fake_st.messages)
    assert any("Arquivos de módulos gerados" in message for message in fake_st.messages)
    assert any("Baixar projeto gerado" in message for message in fake_st.messages)
    assert any("provenance:training" in message for message in fake_st.messages)


def test_provenance_card_distinguishes_llm_and_template(
    monkeypatch: Any,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(provenance_card, "st", fake_st)

    provenance_card.render_stage_provenance(
        stage_name="training",
        provenance=StageProvenance(
            origin="llm",
            model="test-model",
            duration_seconds=1.5,
            parse_method="fenced",
        ),
    )
    provenance_card.render_stage_provenance(
        stage_name="evaluation",
        provenance=StageProvenance(
            origin="template",
            fallback_reason="invalid response",
        ),
    )

    assert any("código gerado pela LLM" in message for message in fake_st.messages)
    assert any("código gerado por template" in message for message in fake_st.messages)
    assert any(
        "Motivo do fallback: invalid response" in message
        for message in fake_st.messages
    )


def test_common_missing_result_path(monkeypatch: Any) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(common, "st", fake_st)
    common.render_missing_result_message()
    common.render_stage_page(stage_name="training", title="Training")
    assert any("Execute a conversão" in message for message in fake_st.messages)


def test_regenerate_from_feedback_handles_missing_configuration(
    monkeypatch: Any,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(common, "st", fake_st)
    common.regenerate_from_feedback()
    assert any("Configuração não encontrada" in message for message in fake_st.messages)
