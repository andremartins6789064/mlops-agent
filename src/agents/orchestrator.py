from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.domain.entities import Notebook
from src.domain.interfaces import INotebookParser


@dataclass(slots=True)
class OrchestrationResult:
    """Container for stage 4 orchestration outputs."""

    notebook: Notebook
    notebook_analysis: dict[str, Any]
    architecture_plan: dict[str, Any]
    generated_modules: dict[str, str] | None = None
    generated_file_paths: dict[str, str] | None = None


class Orchestrator:
    """Coordinate notebook analyzer and architecture agent."""

    def __init__(
        self,
        *,
        notebook_parser: INotebookParser,
        notebook_analyzer: NotebookAnalyzerAgent,
        architecture_agent: ArchitectureAgent,
        code_generator: CodeGeneratorAgent | None = None,
    ) -> None:
        self._notebook_parser = notebook_parser
        self._notebook_analyzer = notebook_analyzer
        self._architecture_agent = architecture_agent
        self._code_generator = code_generator

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

        generated_modules: dict[str, str] | None = None
        generated_file_paths: dict[str, str] | None = None
        if self._code_generator is not None:
            generated_modules = self._code_generator.generate_modules(
                notebook=notebook,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
            )
            if output_dir is not None:
                generated_file_paths = self._code_generator.write_modules(
                    generated_modules=generated_modules,
                    output_dir=output_dir,
                )

        return OrchestrationResult(
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
            generated_modules=generated_modules,
            generated_file_paths=generated_file_paths,
        )
