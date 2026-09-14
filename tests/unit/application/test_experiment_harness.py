from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import nbformat
import pytest

from src.agents.orchestrator import OrchestrationResult
from src.application.convert_notebook import ConversionRequest
from src.application.experiment_harness import (
    DEFAULT_EXPERIMENT_NOTEBOOKS,
    InvalidExperimentNotebookError,
    resolve_generated_pipeline_dir,
    run_experiment_matrix,
    validate_experiment_notebooks,
)
from src.domain.entities import Notebook
from src.domain.value_objects import QualityMetrics
from src.shared.provenance import StageProvenance

# mypy: disable-error-code=no-untyped-call


def _write_notebook(path: Path, source: str) -> str:
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, path)
    return str(path)


def _metric_notebook(tmp_path: Path, name: str = "junior.ipynb") -> str:
    return _write_notebook(tmp_path / name, "print('final_mse=1.0')")


def _stub_converter(
    requests: list[ConversionRequest],
) -> Any:
    def converter(request: ConversionRequest) -> OrchestrationResult:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        requests.append(request)
        return _result()

    return converter


def _result(*, exported_zip_path: str | None = None) -> OrchestrationResult:
    return OrchestrationResult(
        notebook=Notebook(path="demo.ipynb", cells=[], metadata={}),
        notebook_analysis={},
        architecture_plan={},
        generated_modules={"training": "def train() -> int:\n    return 1\n"},
        stage_provenance={
            "training": StageProvenance(origin="llm", model="test-model")
        },
        quality_metrics=QualityMetrics(
            lint_errors=0,
            type_errors=1,
            test_coverage=82.5,
            review_iterations=1,
        ),
        exported_zip_path=exported_zip_path,
    )


def test_harness_writes_one_row_per_matrix_combination(tmp_path: Any) -> None:
    requests: list[ConversionRequest] = []
    notebook = _metric_notebook(tmp_path)

    output_csv = tmp_path / "results.csv"
    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["model-a", "model-b"],
        repetitions=2,
        output_csv=str(output_csv),
        output_root=str(tmp_path / "runs"),
        run_mutation=False,
        converter=_stub_converter(requests),
    )

    assert len(rows) == 4
    assert len(requests) == 4
    with output_csv.open(newline="", encoding="utf-8") as file_obj:
        saved_rows = list(csv.DictReader(file_obj))
    assert len(saved_rows) == 4
    assert {row["model"] for row in saved_rows} == {"model-a", "model-b"}
    assert all(row["equivalence_status"] == "nao_executavel" for row in saved_rows)
    assert all(row["review_status"] == "desativada" for row in saved_rows)
    assert all(row["coverage"] == "82.50" for row in saved_rows)
    assert all(row["type_errors"] == "1" for row in saved_rows)


def test_harness_records_failed_combination_without_aborting(tmp_path: Any) -> None:
    calls = 0
    notebook = _metric_notebook(tmp_path, "demo.ipynb")

    def converter(request: ConversionRequest) -> OrchestrationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model unavailable")
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        return _result()

    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["model-a", "model-b"],
        repetitions=1,
        output_csv=str(tmp_path / "results.csv"),
        output_root=str(tmp_path / "runs"),
        run_mutation=False,
        converter=converter,
    )

    assert len(rows) == 2
    assert rows[0]["error"] == "model unavailable"
    assert rows[1]["error"] == ""


def test_harness_records_provider_from_model_spec(tmp_path: Any) -> None:
    requests: list[ConversionRequest] = []
    notebook = _metric_notebook(tmp_path)

    output_csv = tmp_path / "results.csv"
    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["groq=openai/gpt-oss-120b"],
        repetitions=1,
        output_csv=str(output_csv),
        output_root=str(tmp_path / "runs"),
        run_mutation=False,
        converter=_stub_converter(requests),
    )

    assert rows[0]["model"] == "openai/gpt-oss-120b"
    assert rows[0]["provider"] == "groq"
    assert requests[0].llm_provider == "groq"


def test_harness_records_timeout_and_keeps_incremental_csv(
    tmp_path: Any,
) -> None:
    notebook = _metric_notebook(tmp_path, "demo.ipynb")

    def converter(request: ConversionRequest) -> OrchestrationResult:
        time.sleep(2)
        return _result()

    output_csv = tmp_path / "results.csv"
    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["slow-model"],
        repetitions=1,
        output_csv=str(output_csv),
        output_root=str(tmp_path / "runs"),
        max_run_seconds=1,
        run_mutation=False,
        converter=converter,
    )

    assert rows[0]["error"] == "timeout after 1s"
    with output_csv.open(newline="", encoding="utf-8") as file_obj:
        saved_rows = list(csv.DictReader(file_obj))
    assert saved_rows[0]["error"] == "timeout after 1s"


def test_resolve_generated_pipeline_dir_uses_exported_project(
    tmp_path: Any,
) -> None:
    output_dir = tmp_path / "run"
    zip_path = output_dir / "junior.zip"

    assert (
        resolve_generated_pipeline_dir(output_dir, exported_zip_path=str(zip_path))
        == output_dir / "junior"
    )
    assert (
        resolve_generated_pipeline_dir(output_dir, exported_zip_path=None) == output_dir
    )


def test_harness_equivalence_runs_entrypoint_in_exported_tree(
    tmp_path: Any,
) -> None:
    notebook = _metric_notebook(tmp_path, "junior.ipynb")

    def converter(request: ConversionRequest) -> OrchestrationResult:
        output = Path(request.output_dir)
        review_src = output / "src"
        review_src.mkdir(parents=True, exist_ok=True)
        (review_src / "training.py").write_text("def train() -> None:\n    pass\n")
        project_src = output / "junior" / "src"
        project_src.mkdir(parents=True, exist_ok=True)
        (project_src / "main.py").write_text(
            "print('final_mse=1.0')\n", encoding="utf-8"
        )
        zip_path = output / "junior.zip"
        zip_path.write_bytes(b"unused")
        return _result(exported_zip_path=str(zip_path))

    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["model-a"],
        repetitions=1,
        output_csv=str(tmp_path / "results.csv"),
        output_root=str(tmp_path / "runs"),
        run_mutation=False,
        converter=converter,
    )

    assert rows[0]["equivalence_status"] == "equivalente"
    assert rows[0]["equivalence_error"] == ""
    assert "No such file or directory" not in rows[0]["error"]


def test_harness_rejects_fixture_notebook_before_conversion(tmp_path: Any) -> None:
    requests: list[ConversionRequest] = []
    output_csv = tmp_path / "results.csv"

    def converter(request: ConversionRequest) -> OrchestrationResult:
        requests.append(request)
        return _result()

    with pytest.raises(InvalidExperimentNotebookError, match="tests/fixtures"):
        run_experiment_matrix(
            notebooks=["tests/fixtures/simple_regression.ipynb"],
            models=["model-a"],
            repetitions=1,
            output_csv=str(output_csv),
            output_root=str(tmp_path / "runs"),
            run_mutation=False,
            converter=converter,
        )

    assert requests == []
    assert not output_csv.exists()


def test_harness_rejects_notebook_without_metric_before_llm(tmp_path: Any) -> None:
    requests: list[ConversionRequest] = []
    notebook = _write_notebook(tmp_path / "score_only.ipynb", "print('score=1.0')")
    output_csv = tmp_path / "results.csv"

    def converter(request: ConversionRequest) -> OrchestrationResult:
        requests.append(request)
        return _result()

    with pytest.raises(InvalidExperimentNotebookError, match="final_mse"):
        run_experiment_matrix(
            notebooks=[notebook],
            models=["model-a"],
            repetitions=1,
            output_csv=str(output_csv),
            output_root=str(tmp_path / "runs"),
            run_mutation=False,
            converter=converter,
        )

    assert requests == []
    assert not output_csv.exists()


def test_harness_rejects_notebook_that_does_not_print_metric(
    tmp_path: Any,
) -> None:
    requests: list[ConversionRequest] = []
    notebook = _write_notebook(tmp_path / "silent.ipynb", "final_mse = 1.0")

    def converter(request: ConversionRequest) -> OrchestrationResult:
        requests.append(request)
        return _result()

    with pytest.raises(InvalidExperimentNotebookError, match="executed"):
        run_experiment_matrix(
            notebooks=[notebook],
            models=["model-a"],
            repetitions=1,
            output_csv=str(tmp_path / "results.csv"),
            output_root=str(tmp_path / "runs"),
            run_mutation=False,
            converter=converter,
        )

    assert requests == []


def test_harness_rejects_missing_notebook(tmp_path: Any) -> None:
    output_csv = tmp_path / "results.csv"
    with pytest.raises(InvalidExperimentNotebookError, match="does not exist"):
        run_experiment_matrix(
            notebooks=[str(tmp_path / "missing.ipynb")],
            models=["model-a"],
            repetitions=1,
            output_csv=str(output_csv),
            output_root=str(tmp_path / "runs"),
            run_mutation=False,
            converter=_stub_converter([]),
        )
    assert not output_csv.exists()


def test_harness_precheck_skips_duplicate_notebook_paths(tmp_path: Any) -> None:
    notebook = _metric_notebook(tmp_path)
    validate_experiment_notebooks([notebook, notebook])


def test_etapa_9_0_notebooks_pass_precheck() -> None:
    validate_experiment_notebooks(DEFAULT_EXPERIMENT_NOTEBOOKS)


def test_harness_records_nonconforming_contract_without_crashing(
    tmp_path: Any,
) -> None:
    notebook = _metric_notebook(tmp_path)

    def converter(request: ConversionRequest) -> OrchestrationResult:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        result = _result()
        result.generated_modules = {
            "feature_engineering": (
                "def load_data(file_path: str) -> object:\n    return file_path\n"
            ),
            "training": (
                "def train_model(x: object, y: object) -> object:\n    return x\n"
            ),
            "inference": (
                "def predict(model: object, x: object) -> object:\n    return x\n"
            ),
            "evaluation": (
                "def evaluate_model(a: object, b: object, c: object) -> float:\n"
                "    return 0.0\n"
            ),
        }
        return result

    rows = run_experiment_matrix(
        notebooks=[notebook],
        models=["model-a"],
        repetitions=1,
        output_csv=str(tmp_path / "results.csv"),
        output_root=str(tmp_path / "runs"),
        run_mutation=False,
        converter=converter,
    )

    assert rows[0]["contract_status"] == "nao_conforme"
    assert "feature_engineering.load_data" in rows[0]["contract_error"]
    assert "evaluation.evaluate_model" in rows[0]["contract_error"]
    assert rows[0]["error"] == ""
