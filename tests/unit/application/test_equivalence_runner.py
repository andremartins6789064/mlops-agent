from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nbformat

from src.application.equivalence_runner import (
    EquivalenceStatus,
    extract_metric,
    notebook_source_declares_metric,
    read_notebook_metric,
    run_equivalence,
)

# mypy: disable-error-code=no-untyped-call


def _write_notebook(path: Path, source: str) -> None:
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, path)


def _write_pipeline(path: Path, metric: str) -> None:
    path.write_text(f"print('final_mse={metric}')\n", encoding="utf-8")


def test_equivalence_runner_classifies_equivalent_pipeline(tmp_path: Any) -> None:
    notebook_path = tmp_path / "source.ipynb"
    pipeline_path = tmp_path / "pipeline.py"
    _write_notebook(notebook_path, "print('final_mse=1.0')")
    _write_pipeline(pipeline_path, "1.03")

    result = run_equivalence(
        notebook_path=str(notebook_path),
        pipeline_command=[sys.executable, str(pipeline_path)],
        pipeline_dir=str(tmp_path),
        tolerance=0.05,
    )

    assert result.status == EquivalenceStatus.EQUIVALENT
    assert result.original_metric == 1.0
    assert result.generated_metric == 1.03


def test_equivalence_runner_classifies_divergent_pipeline(tmp_path: Any) -> None:
    notebook_path = tmp_path / "source.ipynb"
    pipeline_path = tmp_path / "pipeline.py"
    _write_notebook(notebook_path, "print('final_mse=1.0')")
    _write_pipeline(pipeline_path, "1.2")

    result = run_equivalence(
        notebook_path=str(notebook_path),
        pipeline_command=[sys.executable, str(pipeline_path)],
        pipeline_dir=str(tmp_path),
        tolerance=0.05,
    )

    assert result.status == EquivalenceStatus.DIVERGENT


def test_equivalence_runner_classifies_broken_pipeline(tmp_path: Any) -> None:
    notebook_path = tmp_path / "source.ipynb"
    _write_notebook(notebook_path, "print('final_mse=1.0')")

    result = run_equivalence(
        notebook_path=str(notebook_path),
        pipeline_command=[sys.executable, str(tmp_path / "missing.py")],
        pipeline_dir=str(tmp_path),
    )

    assert result.status == EquivalenceStatus.NOT_EXECUTABLE
    assert result.error is not None


def test_extract_metric_uses_last_value_and_supports_scientific_notation() -> None:
    output = "final_mse=1.0\nintermediate=2\nfinal_mse=1.2e-3\n"

    assert extract_metric(output, metric_name="final_mse") == 0.0012


def test_notebook_source_declares_metric_detects_print_and_assignment(
    tmp_path: Any,
) -> None:
    declared = tmp_path / "declared.ipynb"
    missing = tmp_path / "missing.ipynb"
    _write_notebook(declared, "print(f'final_mse={value:.12f}')")
    _write_notebook(missing, "print('score=1.0')")

    assert notebook_source_declares_metric(str(declared), metric_name="final_mse")
    assert not notebook_source_declares_metric(str(missing), metric_name="final_mse")


def test_read_notebook_metric_returns_printed_value(tmp_path: Any) -> None:
    notebook_path = tmp_path / "source.ipynb"
    _write_notebook(notebook_path, "print('final_mse=0.011202345146')")

    metric, error = read_notebook_metric(str(notebook_path), metric_name="final_mse")

    assert error is None
    assert metric == 0.011202345146
