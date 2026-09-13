from __future__ import annotations

import csv
import time
from typing import Any

from src.agents.orchestrator import OrchestrationResult
from src.application.convert_notebook import ConversionRequest
from src.application.experiment_harness import run_experiment_matrix
from src.domain.entities import Notebook
from src.domain.value_objects import QualityMetrics
from src.shared.provenance import StageProvenance


def _result() -> OrchestrationResult:
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
    )


def test_harness_writes_one_row_per_matrix_combination(tmp_path: Any) -> None:
    requests: list[ConversionRequest] = []

    def converter(request: ConversionRequest) -> OrchestrationResult:
        requests.append(request)
        return _result()

    output_csv = tmp_path / "results.csv"
    rows = run_experiment_matrix(
        notebooks=["junior.ipynb"],
        models=["model-a", "model-b"],
        repetitions=2,
        output_csv=str(output_csv),
        output_root=str(tmp_path / "runs"),
        converter=converter,
    )

    assert len(rows) == 4
    assert len(requests) == 4
    with output_csv.open(newline="", encoding="utf-8") as file_obj:
        saved_rows = list(csv.DictReader(file_obj))
    assert len(saved_rows) == 4
    assert {row["model"] for row in saved_rows} == {"model-a", "model-b"}
    assert all(row["equivalence_status"] == "nao_executavel" for row in saved_rows)
    assert all(row["coverage"] == "82.50" for row in saved_rows)
    assert all(row["type_errors"] == "1" for row in saved_rows)


def test_harness_records_failed_combination_without_aborting(tmp_path: Any) -> None:
    calls = 0

    def converter(request: ConversionRequest) -> OrchestrationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model unavailable")
        return _result()

    rows = run_experiment_matrix(
        notebooks=["demo.ipynb"],
        models=["model-a", "model-b"],
        repetitions=1,
        output_csv=str(tmp_path / "results.csv"),
        output_root=str(tmp_path / "runs"),
        converter=converter,
    )

    assert len(rows) == 2
    assert rows[0]["error"] == "model unavailable"
    assert rows[1]["error"] == ""


def test_harness_records_timeout_and_keeps_incremental_csv(
    tmp_path: Any,
) -> None:
    def converter(request: ConversionRequest) -> OrchestrationResult:
        time.sleep(2)
        return _result()

    output_csv = tmp_path / "results.csv"
    rows = run_experiment_matrix(
        notebooks=["demo.ipynb"],
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
