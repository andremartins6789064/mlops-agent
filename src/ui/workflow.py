from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents.orchestrator import OrchestrationResult
from src.application.convert_notebook import ConversionRequest, convert_notebook
from src.domain.value_objects import QualityMetrics
from src.shared.progress import ProgressCallback
from src.ui.session import UIConfig

STAGE_NAMES = ("feature_engineering", "training", "inference", "evaluation")


def run_conversion(
    *,
    notebook_bytes: bytes,
    notebook_name: str,
    config: UIConfig,
    architecture_feedback: str | None = None,
    stage_feedback: dict[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OrchestrationResult:
    """Persist uploaded notebook and run conversion pipeline."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_root = (
        Path(config.output_dir) / f"{timestamp}-{notebook_name.replace(' ', '_')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    notebook_path = run_root / notebook_name
    notebook_path.write_bytes(notebook_bytes)

    request = ConversionRequest(
        notebook_path=str(notebook_path),
        output_dir=str(run_root / "generated"),
        use_llm=config.use_llm,
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_model=config.llm_model,
        enable_review=config.enable_review,
        architecture_feedback=architecture_feedback,
        stage_feedback=stage_feedback,
        progress_callback=progress_callback,
    )
    return convert_notebook(request)


def get_stage_code(
    result: OrchestrationResult | None,
    *,
    stage_name: str,
) -> str | None:
    """Return generated module code for the selected stage."""
    if result is None or result.generated_modules is None:
        return None
    stage_code = result.generated_modules.get(stage_name)
    if isinstance(stage_code, str):
        return stage_code
    return None


def get_stage_tests(
    result: OrchestrationResult | None,
    *,
    stage_name: str,
) -> str | None:
    """Return generated tests for the selected stage."""
    if result is None or result.generated_tests is None:
        return None
    stage_tests = result.generated_tests.get(stage_name)
    if isinstance(stage_tests, str):
        return stage_tests
    return None


def result_ready(state: dict[str, Any]) -> bool:
    """Indicate if conversion output is available in session state."""
    return bool(state.get("mlops_result"))


def summarize_review_status(
    metrics: QualityMetrics | None,
    *,
    review_enabled: bool = True,
    review_incomplete: bool = False,
) -> str:
    """Build a one-line review status string for the UI."""
    if not review_enabled:
        return "Reviewer desativado; métricas de revisão não foram coletadas."
    if review_incomplete:
        return "Revisão inconclusiva; verifique os erros registrados."
    if metrics is None:
        return "Métricas de revisão ainda não disponíveis."
    status = "aprovada" if metrics.meets_minimum_coverage() else "precisa de ajustes"
    return (
        f"Revisão {status}: lint={metrics.lint_errors}, "
        f"tipos={metrics.type_errors}, cobertura={metrics.test_coverage:.1f}%."
    )


def create_temp_notebook_path(*, notebook_name: str) -> Path:
    """Provide a temporary notebook path for external callers/tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="mlops-agent-ui-"))
    return temp_dir / notebook_name
