"""Headless notebook/model experiment matrix harness."""

from __future__ import annotations

import ast
import csv
import shlex
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from src.agents.orchestrator import OrchestrationResult
from src.application.convert_notebook import ConversionRequest, convert_notebook
from src.application.equivalence_runner import (
    notebook_source_declares_metric,
    read_notebook_metric,
    run_equivalence,
)
from src.application.mutation_checker import run_mutation_check
from src.shared.progress import ProgressEvent

Converter = Callable[[ConversionRequest], OrchestrationResult]

DEFAULT_EXPERIMENT_NOTEBOOKS = (
    "notebooks/junior_regression.ipynb",
    "notebooks/senior_regression.ipynb",
)
PRIMARY_METRIC_NAME = "final_mse"


class ExperimentTimeout(BaseException):
    """Signal that one matrix combination exceeded its time budget."""


class InvalidExperimentNotebookError(ValueError):
    """A matrix notebook cannot produce the primary experimental metric."""


CSV_FIELDS = (
    "notebook",
    "model",
    "provider",
    "repetition",
    "duration_seconds",
    "equivalence_status",
    "original_metric",
    "generated_metric",
    "equivalence_error",
    "stage_origins",
    "fallback_stages",
    "review_iterations",
    "review_status",
    "review_error",
    "coverage",
    "lint_errors",
    "type_errors",
    "mutation_status",
    "mutation_score",
    "mutation_survivors",
    "empty_functions",
    "error",
)


def run_experiment_matrix(
    *,
    notebooks: Sequence[str],
    models: Sequence[str],
    repetitions: int,
    output_csv: str,
    output_root: str = "output/experiments",
    pipeline_command: str | None = None,
    tolerance: float = 0.05,
    timeout_seconds: int = 120,
    max_run_seconds: int = 180,
    llm_timeout_seconds: float = 300.0,
    llm_max_retries: int = 3,
    llm_retry_backoff_seconds: float = 5.0,
    enable_review: bool = False,
    review_max_llm_calls: int = 1,
    review_max_seconds: float = 120.0,
    run_mutation: bool = True,
    converter: Converter = convert_notebook,
) -> list[dict[str, str]]:
    """Run every notebook/model combination and write one CSV row per run."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if max_run_seconds <= 0:
        raise ValueError("max_run_seconds must be positive")

    validate_experiment_notebooks(
        notebooks,
        metric_name=PRIMARY_METRIC_NAME,
        timeout_seconds=timeout_seconds,
    )

    rows: list[dict[str, str]] = []
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for notebook in notebooks:
            for model_spec in models:
                provider, model = _parse_model_spec(model_spec)
                for repetition in range(1, repetitions + 1):
                    row = _run_one_with_timeout(
                        notebook=notebook,
                        model=model,
                        provider=provider,
                        repetition=repetition,
                        output_root=root,
                        pipeline_command=pipeline_command,
                        tolerance=tolerance,
                        timeout_seconds=timeout_seconds,
                        max_run_seconds=max_run_seconds,
                        llm_timeout_seconds=llm_timeout_seconds,
                        llm_max_retries=llm_max_retries,
                        llm_retry_backoff_seconds=llm_retry_backoff_seconds,
                        enable_review=enable_review,
                        review_max_llm_calls=review_max_llm_calls,
                        review_max_seconds=review_max_seconds,
                        run_mutation=run_mutation,
                        converter=converter,
                    )
                    rows.append(row)
                    writer.writerow(row)
                    file_obj.flush()
                    print(
                        f"[matrix] {notebook} | {model} | run={repetition} "
                        f"| status={row['error'] or row['equivalence_status']}",
                        flush=True,
                    )
    return rows


def validate_experiment_notebooks(
    notebooks: Sequence[str],
    *,
    metric_name: str = PRIMARY_METRIC_NAME,
    timeout_seconds: int = 120,
) -> None:
    """Reject parser fixtures and notebooks that cannot emit the primary metric."""
    seen: set[Path] = set()
    for notebook in notebooks:
        path = Path(notebook)
        if _is_parser_fixture(path):
            raise InvalidExperimentNotebookError(
                f"Notebook '{notebook}' is under tests/fixtures/ and cannot be "
                "used as experiment-matrix input. Use "
                "notebooks/junior_regression.ipynb and "
                "notebooks/senior_regression.ipynb. Parser fixtures remain valid "
                "for unit tests, PASSO 0, and T-8."
            )
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _require_printed_metric(
            path,
            metric_name=metric_name,
            timeout_seconds=timeout_seconds,
        )


def _is_parser_fixture(path: Path) -> bool:
    parts = path.expanduser().resolve(strict=False).parts
    for index, part in enumerate(parts[:-1]):
        if part == "tests" and parts[index + 1] == "fixtures":
            return True
    return False


def _require_printed_metric(
    path: Path,
    *,
    metric_name: str,
    timeout_seconds: int,
) -> None:
    if not path.is_file():
        raise InvalidExperimentNotebookError(f"Notebook '{path}' does not exist.")
    notebook_path = str(path)
    if not notebook_source_declares_metric(notebook_path, metric_name=metric_name):
        raise InvalidExperimentNotebookError(
            f"Notebook '{notebook_path}' does not print '{metric_name}'. "
            "Refuse this input before calling the LLM. Experiment notebooks "
            "must print the primary metric used by functional equivalence."
        )
    metric, error = read_notebook_metric(
        notebook_path,
        metric_name=metric_name,
        timeout_seconds=timeout_seconds,
    )
    if error is not None or metric is None:
        raise InvalidExperimentNotebookError(
            f"Notebook '{notebook_path}' did not produce '{metric_name}' when "
            f"executed: {error or 'metric missing from output'}"
        )


def _run_one_with_timeout(
    *,
    notebook: str,
    model: str,
    provider: str | None,
    repetition: int,
    output_root: Path,
    pipeline_command: str | None,
    tolerance: float,
    timeout_seconds: int,
    max_run_seconds: int,
    llm_timeout_seconds: float,
    llm_max_retries: int,
    llm_retry_backoff_seconds: float,
    enable_review: bool,
    review_max_llm_calls: int,
    review_max_seconds: float,
    run_mutation: bool,
    converter: Converter,
) -> dict[str, str]:
    row = _empty_row(
        notebook=notebook,
        model=model,
        provider=provider,
        repetition=repetition,
    )

    def _raise_timeout(signum: int, frame: object) -> None:
        raise ExperimentTimeout

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, max_run_seconds)
    try:
        return _run_one(
            notebook=notebook,
            model=model,
            provider=provider,
            repetition=repetition,
            output_root=output_root,
            pipeline_command=pipeline_command,
            tolerance=tolerance,
            timeout_seconds=timeout_seconds,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff_seconds=llm_retry_backoff_seconds,
            enable_review=enable_review,
            review_max_llm_calls=review_max_llm_calls,
            review_max_seconds=review_max_seconds,
            run_mutation=run_mutation,
            converter=converter,
        )
    except ExperimentTimeout:
        row["error"] = f"timeout after {max_run_seconds}s"
        row["duration_seconds"] = str(max_run_seconds)
        return row
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_one(
    *,
    notebook: str,
    model: str,
    provider: str | None,
    repetition: int,
    output_root: Path,
    pipeline_command: str | None,
    tolerance: float,
    timeout_seconds: int,
    llm_timeout_seconds: float,
    llm_max_retries: int,
    llm_retry_backoff_seconds: float,
    enable_review: bool,
    review_max_llm_calls: int,
    review_max_seconds: float,
    run_mutation: bool,
    converter: Converter,
) -> dict[str, str]:
    started = time.perf_counter()
    run_name = f"{provider or 'default'}-{_safe_name(model)}"
    output_dir = output_root / (f"{Path(notebook).stem}-{run_name}-run{repetition}")
    row = _empty_row(
        notebook=notebook,
        model=model,
        provider=provider,
        repetition=repetition,
    )
    try:

        def report_progress(event: ProgressEvent) -> None:
            print(
                f"[{model}] {event.completed}/{event.total} {event.message}",
                flush=True,
            )

        result = converter(
            ConversionRequest(
                notebook_path=notebook,
                output_dir=str(output_dir),
                use_llm=True,
                llm_model=model,
                llm_provider=provider,
                enable_review=enable_review,
                review_max_llm_calls=review_max_llm_calls,
                review_max_seconds=review_max_seconds,
                llm_timeout_seconds=llm_timeout_seconds,
                llm_max_retries=llm_max_retries,
                llm_retry_backoff_seconds=llm_retry_backoff_seconds,
                progress_callback=report_progress,
            )
        )
        _add_result_metrics(row, result)
        _add_equivalence(
            row,
            notebook=notebook,
            output_dir=output_dir,
            pipeline_command=pipeline_command,
            tolerance=tolerance,
            timeout_seconds=timeout_seconds,
        )
        if run_mutation:
            mutation = run_mutation_check(
                str(output_dir),
                timeout_seconds=timeout_seconds,
            )
            row["mutation_status"] = mutation.status
            if mutation.baseline_passed:
                row["mutation_score"] = f"{mutation.mutation_score:.4f}"
                row["mutation_survivors"] = str(mutation.survived_mutations)
            else:
                row["error"] = mutation.baseline_error or "mutation baseline failed"
        row["empty_functions"] = str(_count_empty_functions(result.generated_modules))
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
    row["duration_seconds"] = f"{time.perf_counter() - started:.3f}"
    return row


def _empty_row(
    *, notebook: str, model: str, provider: str | None, repetition: int
) -> dict[str, str]:
    return {field: "" for field in CSV_FIELDS} | {
        "notebook": notebook,
        "model": model,
        "provider": provider or "",
        "repetition": str(repetition),
    }


def _parse_model_spec(model_spec: str) -> tuple[str | None, str]:
    """Parse ``provider=model`` while preserving provider-specific model IDs."""
    if "=" not in model_spec:
        return None, model_spec
    provider, model = model_spec.split("=", 1)
    if not provider or not model:
        raise ValueError(
            f"Invalid model specification '{model_spec}'; expected provider=model"
        )
    return provider, model


def _add_result_metrics(row: dict[str, str], result: OrchestrationResult) -> None:
    if not result.review_enabled:
        row["review_status"] = "desativada"
    elif result.review_incomplete or result.quality_metrics is None:
        row["review_status"] = "inconclusiva"
    else:
        row["review_status"] = "concluida"
    row["review_error"] = result.review_error or ""
    provenance = result.stage_provenance or {}
    row["stage_origins"] = ";".join(
        f"{stage}:{item.origin}" for stage, item in sorted(provenance.items())
    )
    row["fallback_stages"] = ";".join(
        stage for stage, item in sorted(provenance.items()) if item.origin == "template"
    )
    if result.quality_metrics is not None:
        metrics = result.quality_metrics
        row["review_iterations"] = str(metrics.review_iterations)
        row["coverage"] = f"{metrics.test_coverage:.2f}"
        row["lint_errors"] = str(metrics.lint_errors)
        row["type_errors"] = str(metrics.type_errors)


def _add_equivalence(
    row: dict[str, str],
    *,
    notebook: str,
    output_dir: Path,
    pipeline_command: str | None,
    tolerance: float,
    timeout_seconds: int,
) -> None:
    command = (
        pipeline_command.format(
            output_dir=str(output_dir),
            notebook=notebook,
        )
        if pipeline_command is not None
        else f"{sys.executable} src/main.py"
    )
    result = run_equivalence(
        notebook_path=notebook,
        pipeline_command=shlex.split(command),
        pipeline_dir=str(output_dir),
        tolerance=tolerance,
        timeout_seconds=timeout_seconds,
    )
    row["equivalence_status"] = result.status.value
    row["original_metric"] = _format_optional(result.original_metric)
    row["generated_metric"] = _format_optional(result.generated_metric)
    row["equivalence_error"] = result.error or ""


def _count_empty_functions(modules: dict[str, str] | None) -> int:
    if modules is None:
        return 0
    empty = 0
    for source in modules.values():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_empty(
                node
            ):
                empty += 1
    return empty


def _is_empty(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    meaningful: list[ast.stmt] = []
    for statement in node.body:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            if isinstance(statement.value.value, str):
                continue
        meaningful.append(statement)
    return len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
