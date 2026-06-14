from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.test_generator import PipelineTestGeneratorAgent
from src.domain.entities import Notebook
from src.domain.interfaces import INotebookParser
from src.domain.value_objects import QualityMetrics


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
    quality_metrics: QualityMetrics | None = None


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
    ) -> None:
        self._notebook_parser = notebook_parser
        self._notebook_analyzer = notebook_analyzer
        self._architecture_agent = architecture_agent
        self._code_generator = code_generator
        self._test_generator = test_generator
        self._reviewer = reviewer

    def run(
        self,
        notebook_path: str,
        *,
        output_dir: str | None = None,
    ) -> OrchestrationResult:
        """Run analysis, architecture planning, and optional code generation."""
        notebook = self._notebook_parser.parse(notebook_path)
        notebook_analysis = self._notebook_analyzer.analyze(notebook)
        architecture_plan = self._architecture_agent.plan(notebook_analysis)

        run_output_dir = output_dir
        if run_output_dir is None and (
            self._code_generator is not None
            or self._test_generator is not None
            or self._reviewer is not None
        ):
            run_output_dir = "output"

        generated_modules: dict[str, str] | None = None
        generated_file_paths: dict[str, str] | None = None
        generated_tests: dict[str, str] | None = None
        generated_test_file_paths: dict[str, str] | None = None
        quality_metrics: QualityMetrics | None = None

        if self._code_generator is not None:
            generated_modules = self._code_generator.generate_modules(
                notebook=notebook,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
            )
            if run_output_dir is not None:
                generated_file_paths = self._code_generator.write_modules(
                    generated_modules=generated_modules,
                    output_dir=run_output_dir,
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

        return OrchestrationResult(
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
            generated_modules=generated_modules,
            generated_file_paths=generated_file_paths,
            generated_tests=generated_tests,
            generated_test_file_paths=generated_test_file_paths,
            quality_metrics=quality_metrics,
        )
