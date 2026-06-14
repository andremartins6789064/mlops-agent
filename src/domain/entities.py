from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain.value_objects import QualityMetrics


class CellType(StrEnum):
    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


class PipelineType(StrEnum):
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class NotebookCell:
    index: int
    cell_type: CellType
    source: str
    outputs: list[str] = field(default_factory=list)
    pipeline_type: PipelineType = PipelineType.UNKNOWN


@dataclass(slots=True)
class Notebook:
    path: str
    cells: list[NotebookCell]
    metadata: dict[str, Any]


@dataclass(slots=True)
class PipelineStage:
    pipeline_type: PipelineType
    cells: list[NotebookCell]
    generated_code: str | None = None
    generated_tests: str | None = None


@dataclass(slots=True)
class Pipeline:
    notebook: Notebook
    stages: list[PipelineStage]
    architecture_plan: dict[str, Any] | None = None
    quality_metrics: QualityMetrics | None = None
