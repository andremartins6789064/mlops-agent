from __future__ import annotations

from pathlib import Path

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.agents.orchestrator import Orchestrator
from src.domain.entities import PipelineType
from src.infrastructure.parsers.notebook_parser import NotebookParser


def test_orchestrator_returns_valid_analysis_and_plan() -> None:
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
    )

    result = orchestrator.run("tests/fixtures/simple_regression.ipynb")

    cells_by_pipeline = result.notebook_analysis["cells_by_pipeline"]
    assert set(cells_by_pipeline.keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert all(
        isinstance(index, int)
        for indexes in cells_by_pipeline.values()
        for index in indexes
    )

    architecture_modules = result.architecture_plan["modules"]
    assert set(architecture_modules.keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert architecture_modules["training"]["functions"]


def test_orchestrator_sets_pipeline_labels_on_cells() -> None:
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
    )

    result = orchestrator.run("tests/fixtures/simple_regression.ipynb")
    labels = {cell.index: cell.pipeline_type for cell in result.notebook.cells}

    assert labels[1] == PipelineType.FEATURE_ENGINEERING
    assert labels[2] == PipelineType.FEATURE_ENGINEERING
    assert labels[3] == PipelineType.TRAINING


def test_orchestrator_generates_stage_modules(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
        code_generator=CodeGeneratorAgent(),
    )

    result = orchestrator.run(
        "tests/fixtures/simple_regression.ipynb",
        output_dir=str(tmp_path),
    )

    assert result.generated_modules is not None
    assert result.generated_file_paths is not None
    assert set(result.generated_modules.keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    for stage_file in result.generated_file_paths.values():
        assert Path(stage_file).exists()
