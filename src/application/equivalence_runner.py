"""Run and compare notebook and generated-pipeline metrics."""

from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum

import nbformat
from nbclient import NotebookClient

# mypy: disable-error-code=no-untyped-call


class EquivalenceStatus(StrEnum):
    """Possible functional-equivalence verdicts."""

    EQUIVALENT = "equivalente"
    DIVERGENT = "divergente"
    NOT_EXECUTABLE = "nao_executavel"


@dataclass(slots=True)
class EquivalenceResult:
    """Outcome and evidence from an equivalence run."""

    status: EquivalenceStatus
    metric_name: str
    original_metric: float | None
    generated_metric: float | None
    tolerance: float
    original_output: str = ""
    generated_output: str = ""
    error: str | None = None
    original_duration_seconds: float = 0.0
    generated_duration_seconds: float = 0.0


def run_equivalence(
    *,
    notebook_path: str,
    pipeline_command: list[str],
    pipeline_dir: str = ".",
    metric_name: str = "final_mse",
    tolerance: float = 0.05,
    timeout_seconds: int = 120,
) -> EquivalenceResult:
    """Execute both artifacts and compare a printed metric."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    original_output, original_error, original_duration = _run_notebook(
        notebook_path=notebook_path,
        metric_name=metric_name,
        timeout_seconds=timeout_seconds,
    )
    original_metric = extract_metric(original_output, metric_name=metric_name)
    if original_error is not None or original_metric is None:
        return EquivalenceResult(
            status=EquivalenceStatus.NOT_EXECUTABLE,
            metric_name=metric_name,
            original_metric=original_metric,
            generated_metric=None,
            tolerance=tolerance,
            original_output=original_output,
            error=original_error or f"Metric '{metric_name}' was not printed.",
            original_duration_seconds=original_duration,
        )

    generated_output, generated_error, generated_duration = _run_pipeline(
        command=pipeline_command,
        pipeline_dir=pipeline_dir,
        timeout_seconds=timeout_seconds,
    )
    generated_metric = extract_metric(generated_output, metric_name=metric_name)
    if generated_error is not None or generated_metric is None:
        return EquivalenceResult(
            status=EquivalenceStatus.NOT_EXECUTABLE,
            metric_name=metric_name,
            original_metric=original_metric,
            generated_metric=generated_metric,
            tolerance=tolerance,
            original_output=original_output,
            generated_output=generated_output,
            error=generated_error or f"Metric '{metric_name}' was not printed.",
            original_duration_seconds=original_duration,
            generated_duration_seconds=generated_duration,
        )

    status = (
        EquivalenceStatus.EQUIVALENT
        if math.isclose(
            original_metric,
            generated_metric,
            abs_tol=tolerance,
            rel_tol=0.0,
        )
        else EquivalenceStatus.DIVERGENT
    )
    return EquivalenceResult(
        status=status,
        metric_name=metric_name,
        original_metric=original_metric,
        generated_metric=generated_metric,
        tolerance=tolerance,
        original_output=original_output,
        generated_output=generated_output,
        original_duration_seconds=original_duration,
        generated_duration_seconds=generated_duration,
    )


def extract_metric(output: str, *, metric_name: str) -> float | None:
    """Extract the last printed numeric value for a named metric."""
    pattern = re.compile(
        rf"^\s*{re.escape(metric_name)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
        flags=re.MULTILINE,
    )
    matches = pattern.findall(output)
    return float(matches[-1]) if matches else None


def _run_notebook(
    *, notebook_path: str, metric_name: str, timeout_seconds: int
) -> tuple[str, str | None, float]:
    started = time.perf_counter()
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        NotebookClient(
            notebook,
            timeout=timeout_seconds,
            kernel_name="python3",
            allow_errors=False,
        ).execute()
        output = _notebook_output(notebook)
        return output, None, time.perf_counter() - started
    except Exception as exc:
        return "", f"Notebook execution failed: {exc}", time.perf_counter() - started


def _run_pipeline(
    *,
    command: list[str],
    pipeline_dir: str,
    timeout_seconds: int,
) -> tuple[str, str | None, float]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=pipeline_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            "",
            f"Pipeline execution timed out after {timeout_seconds}s.",
            time.perf_counter() - started,
        )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        return (
            output,
            f"Pipeline exited with status {completed.returncode}.",
            time.perf_counter() - started,
        )
    return output, None, time.perf_counter() - started


def _notebook_output(notebook: nbformat.NotebookNode) -> str:
    outputs: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                outputs.append(str(output.get("text", "")))
            elif output.output_type == "execute_result":
                data = output.get("data", {})
                outputs.append(str(data.get("text/plain", "")))
    return "\n".join(outputs)
