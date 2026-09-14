from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.test_generator import PipelineTestGeneratorAgent
from src.application.validate_output import ValidationResult
from src.domain.entities import Notebook, Pipeline, PipelineStage, PipelineType
from src.domain.interfaces import IExporter, INotebookParser
from src.domain.pipeline_contract import check_pipeline_contract, format_contract_issues
from src.domain.value_objects import QualityMetrics
from src.shared.logger import StructuredExecutionLogger
from src.shared.progress import ProgressCallback, ProgressEvent
from src.shared.provenance import StageProvenance
from src.shared.python_source import extract_imported_libraries


@dataclass(slots=True)
class OrchestrationResult:
    """Container for orchestrated pipeline conversion outputs."""

    notebook: Notebook
    notebook_analysis: dict[str, Any]
    architecture_plan: dict[str, Any]
    generated_modules: dict[str, str] | None = None
    generated_file_paths: dict[str, str] | None = None
    generated_tests: dict[str, str] | None = None
    generated_test_file_paths: dict[str, str] | None = None
    stage_provenance: dict[str, StageProvenance] | None = None
    validation_result: ValidationResult | None = None
    validated_output_dir: str | None = None
    quality_metrics: QualityMetrics | None = None
    review_enabled: bool = False
    review_incomplete: bool = False
    review_error: str | None = None
    exported_zip_path: str | None = None
    execution_log_path: str | None = None


class Orchestrator:
    """Coordinate notebook analyzer and architecture agent."""

    def __init__(
        self,
        *,
        notebook_parser: INotebookParser,
        notebook_analyzer: NotebookAnalyzerAgent,
        architecture_agent: ArchitectureAgent,
        code_generator: CodeGeneratorAgent | None = None,
        test_generator: PipelineTestGeneratorAgent | None = None,
        reviewer: ReviewerAgent | None = None,
        exporter: IExporter | None = None,
    ) -> None:
        self._notebook_parser = notebook_parser
        self._notebook_analyzer = notebook_analyzer
        self._architecture_agent = architecture_agent
        self._code_generator = code_generator
        self._test_generator = test_generator
        self._reviewer = reviewer
        self._exporter = exporter

    def run(
        self,
        notebook_path: str,
        *,
        output_dir: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        """Run analysis, architecture planning, and optional code generation."""
        self._notify(
            progress_callback,
            phase="start",
            message="Iniciando conversão.",
            completed=0,
        )
        notebook = self._notebook_parser.parse(notebook_path)
        self._notify(
            progress_callback,
            phase="parse",
            message="Notebook carregado.",
            completed=1,
        )
        notebook_analysis = self._notebook_analyzer.analyze(notebook)
        self._notify(
            progress_callback,
            phase="analysis",
            message="Análise do notebook concluída.",
            completed=2,
        )
        architecture_plan = self._architecture_agent.plan(notebook_analysis)
        self._notify(
            progress_callback,
            phase="architecture",
            message="Plano de arquitetura concluído.",
            completed=3,
        )

        run_output_dir = output_dir
        if run_output_dir is None and (
            self._code_generator is not None
            or self._test_generator is not None
            or self._reviewer is not None
        ):
            run_output_dir = "output"
        execution_logger = (
            StructuredExecutionLogger(Path(run_output_dir) / "execution.log")
            if run_output_dir is not None
            else None
        )
        if execution_logger is not None:
            execution_logger.log(
                "execution_started",
                notebook_path=notebook_path,
                review_enabled=self._reviewer is not None,
            )

        generated_modules: dict[str, str] | None = None
        generated_file_paths: dict[str, str] | None = None
        generated_tests: dict[str, str] | None = None
        generated_test_file_paths: dict[str, str] | None = None
        quality_metrics: QualityMetrics | None = None
        review_incomplete = False
        review_error: str | None = None
        validation_result: ValidationResult | None = None
        exported_zip_path: str | None = None
        stage_provenance: dict[str, StageProvenance] = {}

        if self._code_generator is not None:
            generated_modules = self._code_generator.generate_modules(
                notebook=notebook,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
                progress_callback=progress_callback,
            )
            if run_output_dir is not None:
                generated_file_paths = self._code_generator.write_modules(
                    generated_modules=generated_modules,
                    output_dir=run_output_dir,
                )
            stage_provenance = self._code_generator.stage_provenance
            self._annotate_contract(generated_modules, stage_provenance)
            if execution_logger is not None:
                for stage_name, provenance in stage_provenance.items():
                    execution_logger.log(
                        "stage_generated",
                        stage=stage_name,
                        **provenance.to_dict(),
                    )
                issues = check_pipeline_contract(generated_modules)
                execution_logger.log(
                    "contract_checked",
                    contract_ok=not issues,
                    contract_error=format_contract_issues(issues),
                )

        if self._test_generator is not None and generated_modules is not None:
            generated_tests = self._test_generator.generate_tests(
                generated_modules=generated_modules
            )
            if run_output_dir is not None:
                tests_dir = str(Path(run_output_dir) / "tests")
                generated_test_file_paths = self._test_generator.write_tests(
                    generated_tests=generated_tests,
                    output_dir=tests_dir,
                )
            self._notify(
                progress_callback,
                phase="tests",
                message="Testes gerados.",
                completed=8,
            )

        if (
            self._reviewer is not None
            and run_output_dir is not None
            and generated_modules is not None
            and generated_tests is not None
        ):
            review_result = self._reviewer.review(
                project_dir=run_output_dir,
                generated_modules=generated_modules,
                generated_tests=generated_tests,
            )
            quality_metrics = review_result.quality_metrics
            review_incomplete = review_result.review_incomplete
            review_error = review_result.review_error
            validation_result = review_result.validation_result
            self._notify(
                progress_callback,
                phase="review",
                message="Revisão e validação concluídas.",
                completed=9,
            )
            if execution_logger is not None:
                execution_logger.log(
                    "review_completed",
                    iterations=review_result.iterations,
                    has_errors=review_result.validation_result.has_errors,
                    test_coverage=review_result.quality_metrics.test_coverage,
                    review_incomplete=review_result.review_incomplete,
                    review_error=review_result.review_error,
                )

        if (
            self._exporter is not None
            and run_output_dir is not None
            and generated_modules is not None
            and generated_tests is not None
        ):
            pipeline = self._build_pipeline(
                notebook=notebook,
                generated_modules=generated_modules,
                generated_tests=generated_tests,
                architecture_plan=architecture_plan,
                quality_metrics=quality_metrics,
            )
            exported_zip_path = self._exporter.export(
                pipeline,
                run_output_dir,
                libraries=self._collect_project_libraries(
                    notebook_analysis=notebook_analysis,
                    generated_modules=generated_modules,
                ),
            )
            self._notify(
                progress_callback,
                phase="export",
                message="Exportação do projeto concluída.",
                completed=10,
            )

        if execution_logger is not None:
            execution_logger.log(
                "execution_completed",
                stages=list(stage_provenance),
                exported_zip_path=exported_zip_path,
            )

        return OrchestrationResult(
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
            generated_modules=generated_modules,
            generated_file_paths=generated_file_paths,
            generated_tests=generated_tests,
            generated_test_file_paths=generated_test_file_paths,
            stage_provenance=stage_provenance,
            validation_result=validation_result,
            validated_output_dir=(
                run_output_dir if validation_result is not None else None
            ),
            quality_metrics=quality_metrics,
            review_enabled=self._reviewer is not None,
            review_incomplete=review_incomplete,
            review_error=review_error,
            exported_zip_path=exported_zip_path,
            execution_log_path=(
                str(execution_logger.log_path) if execution_logger is not None else None
            ),
        )

    @staticmethod
    def _annotate_contract(
        generated_modules: dict[str, str],
        provenance: dict[str, StageProvenance],
    ) -> None:
        """Record signature mismatches on provenance without failing the run."""
        issues = check_pipeline_contract(generated_modules)
        messages: dict[str, list[str]] = {}
        for issue in issues:
            messages.setdefault(issue.module, []).append(
                f"{issue.function}: {issue.message}"
            )
        for module_name, details in messages.items():
            item = provenance.get(module_name)
            if item is None:
                continue
            item.contract_ok = False
            item.contract_error = "; ".join(details)

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        *,
        phase: str,
        message: str,
        completed: int,
    ) -> None:
        if callback is not None:
            callback(
                ProgressEvent(
                    phase=phase,
                    message=message,
                    completed=completed,
                    total=10,
                )
            )

    def _build_pipeline(
        self,
        *,
        notebook: Notebook,
        generated_modules: dict[str, str],
        generated_tests: dict[str, str],
        architecture_plan: dict[str, Any],
        quality_metrics: QualityMetrics | None,
    ) -> Pipeline:
        stage_types = (
            PipelineType.FEATURE_ENGINEERING,
            PipelineType.TRAINING,
            PipelineType.INFERENCE,
            PipelineType.EVALUATION,
        )
        stages: list[PipelineStage] = []
        for stage_type in stage_types:
            stage_key = stage_type.value
            stage_cells = [
                cell for cell in notebook.cells if cell.pipeline_type == stage_type
            ]
            stages.append(
                PipelineStage(
                    pipeline_type=stage_type,
                    cells=stage_cells,
                    generated_code=generated_modules.get(stage_key),
                    generated_tests=generated_tests.get(stage_key),
                )
            )
        return Pipeline(
            notebook=notebook,
            stages=stages,
            architecture_plan=architecture_plan,
            quality_metrics=quality_metrics,
        )

    def _collect_project_libraries(
        self,
        *,
        notebook_analysis: dict[str, Any],
        generated_modules: dict[str, str],
    ) -> list[str]:
        """Combine analyzed libraries with imports in generated source."""
        libraries = {
            library
            for library in notebook_analysis.get("libraries", [])
            if isinstance(library, str)
        }
        for source in generated_modules.values():
            libraries.update(extract_imported_libraries(source))
        return sorted(libraries)
