from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.notebook_analyzer import NotebookAnalyzerAgent
from src.agents.orchestrator import Orchestrator
from src.agents.reviewer import ReviewerAgent
from src.agents.test_generator import PipelineTestGeneratorAgent
from src.application.validate_output import ValidationResult
from src.domain.entities import PipelineType
from src.infrastructure.exporters import ZipExporter
from src.infrastructure.parsers.notebook_parser import NotebookParser
from src.shared.progress import ProgressEvent


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
        assert Path(stage_file).parent.name == "src"
        assert Path(stage_file).exists()


def test_orchestrator_generates_tests_and_quality_metrics(tmp_path: Path) -> None:
    def _validator(project_dir: str) -> ValidationResult:
        return ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=85.0,
            lint_output="",
            type_output="",
            test_output="",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        )

    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
        code_generator=CodeGeneratorAgent(),
        test_generator=PipelineTestGeneratorAgent(),
        reviewer=ReviewerAgent(validator=_validator),
    )

    result = orchestrator.run(
        "tests/fixtures/simple_regression.ipynb",
        output_dir=str(tmp_path),
    )

    assert result.generated_tests is not None
    assert result.generated_test_file_paths is not None
    assert len(result.generated_test_file_paths) == 4
    assert result.quality_metrics is not None
    assert result.quality_metrics.test_coverage == 85.0
    assert result.stage_provenance is not None
    assert set(result.stage_provenance) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert all(
        provenance.origin == "template"
        for provenance in result.stage_provenance.values()
    )
    assert result.execution_log_path is not None
    log_records = [
        json.loads(line)
        for line in Path(result.execution_log_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(record["event"] == "stage_generated" for record in log_records) == 4
    assert any(record["event"] == "review_completed" for record in log_records)


def test_orchestrator_emits_progress_events_in_order(tmp_path: Path) -> None:
    def _validator(project_dir: str) -> ValidationResult:
        return ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=85.0,
            lint_output="",
            type_output="",
            test_output="",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        )

    events: list[ProgressEvent] = []
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
        code_generator=CodeGeneratorAgent(),
        test_generator=PipelineTestGeneratorAgent(),
        reviewer=ReviewerAgent(validator=_validator),
    )

    orchestrator.run(
        "tests/fixtures/simple_regression.ipynb",
        output_dir=str(tmp_path),
        progress_callback=events.append,
    )

    assert [event.phase for event in events] == [
        "start",
        "parse",
        "analysis",
        "architecture",
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
        "tests",
        "review",
    ]
    assert [event.completed for event in events] == list(range(10))
    assert all(event.total == 10 for event in events)


def test_orchestrator_reviewer_real_validation_meets_threshold(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
        code_generator=CodeGeneratorAgent(),
        test_generator=PipelineTestGeneratorAgent(),
        reviewer=ReviewerAgent(),
    )

    result = orchestrator.run(
        "tests/fixtures/simple_regression.ipynb",
        output_dir=str(tmp_path),
    )

    assert result.quality_metrics is not None
    assert result.quality_metrics.lint_errors == 0
    assert result.quality_metrics.type_errors == 0
    assert result.quality_metrics.test_coverage >= 80.0


def test_orchestrator_exports_zip_archive(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        notebook_parser=NotebookParser(),
        notebook_analyzer=NotebookAnalyzerAgent(),
        architecture_agent=ArchitectureAgent(),
        code_generator=CodeGeneratorAgent(),
        test_generator=PipelineTestGeneratorAgent(),
        reviewer=ReviewerAgent(),
        exporter=ZipExporter(),
    )

    result = orchestrator.run(
        "tests/fixtures/simple_regression.ipynb",
        output_dir=str(tmp_path),
    )

    assert result.exported_zip_path is not None
    zip_path = Path(result.exported_zip_path)
    assert zip_path.exists()
    with ZipFile(zip_path, "r") as zip_file:
        names = set(zip_file.namelist())
    assert "src/feature_engineering.py" in names
    assert "tests/test_feature_engineering.py" in names
    assert "requirements.txt" in names
