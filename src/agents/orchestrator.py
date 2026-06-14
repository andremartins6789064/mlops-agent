from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.domain.entities import Notebook
from src.domain.interfaces import INotebookParser


@dataclass(slots=True)
class OrchestrationResult:
    """Container for stage 4 orchestration outputs."""

    notebook: Notebook
    notebook_analysis: dict[str, Any]
    architecture_plan: dict[str, Any]


class Orchestrator:
    """Coordinate notebook analyzer and architecture agent."""

    def __init__(
        self,
        *,
        notebook_parser: INotebookParser,
        notebook_analyzer: NotebookAnalyzerAgent,
        architecture_agent: ArchitectureAgent,
    ) -> None:
        self._notebook_parser = notebook_parser
        self._notebook_analyzer = notebook_analyzer
        self._architecture_agent = architecture_agent

    def run(self, notebook_path: str) -> OrchestrationResult:
        """Run analysis and architecture planning for a notebook path."""
        notebook = self._notebook_parser.parse(notebook_path)
        notebook_analysis = self._notebook_analyzer.analyze(notebook)
        architecture_plan = self._architecture_agent.plan(notebook_analysis)
        return OrchestrationResult(
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
        )
