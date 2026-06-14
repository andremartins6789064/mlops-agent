from __future__ import annotations

from dataclasses import dataclass

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.agents.orchestrator import OrchestrationResult, Orchestrator
from src.agents.reviewer import ReviewerAgent
from src.agents.test_generator import PipelineTestGeneratorAgent
from src.domain.interfaces import ILLMClient
from src.infrastructure.exporters.zip_exporter import ZipExporter
from src.infrastructure.llm.base import BaseOpenAICompatibleClient
from src.infrastructure.parsers.notebook_parser import NotebookParser
from src.shared.config import settings


@dataclass(slots=True)
class ConversionRequest:
    """Input configuration for notebook conversion workflow."""

    notebook_path: str
    output_dir: str = "output"
    use_llm: bool = False
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    architecture_feedback: str | None = None
    stage_feedback: dict[str, str] | None = None


def convert_notebook(request: ConversionRequest) -> OrchestrationResult:
    """Execute full notebook-to-project conversion pipeline."""
    llm_client = _build_llm_client(request)
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(llm_client=llm_client),
        architecture_agent=ArchitectureAgent(
            llm_client=llm_client,
            user_feedback=request.architecture_feedback,
        ),
        code_generator=CodeGeneratorAgent(
            llm_client=llm_client,
            stage_feedback=request.stage_feedback or {},
        ),
        test_generator=PipelineTestGeneratorAgent(llm_client=llm_client),
        reviewer=ReviewerAgent(llm_client=llm_client),
        exporter=ZipExporter(),
    )
    return orchestrator.run(request.notebook_path, output_dir=request.output_dir)


def _build_llm_client(request: ConversionRequest) -> ILLMClient | None:
    if not request.use_llm:
        return None
    return BaseOpenAICompatibleClient(
        base_url=request.llm_base_url or settings.llm_base_url,
        api_key=request.llm_api_key or settings.llm_api_key,
        model=request.llm_model or settings.llm_model,
    )
