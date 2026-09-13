from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from src.domain.entities import Notebook, NotebookCell, PipelineType
from src.domain.interfaces import ILLMClient
from src.shared.llm_parsing import parse_json_object


class NotebookAnalyzerAgent:
    """Analyze notebook cells and classify them by pipeline stage."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/notebook_analyzer.txt",
    ) -> None:
        self._llm_client = llm_client
        self._prompt_path = prompt_path

    def analyze(self, notebook: Notebook) -> dict[str, Any]:
        """Return notebook analysis in a JSON-serializable structure."""
        if self._llm_client is not None:
            llm_result = self._analyze_with_llm(notebook)
            if llm_result is not None:
                self._apply_pipeline_labels(notebook, llm_result["cells_by_pipeline"])
                return llm_result

        heuristic_result = self._analyze_with_heuristics(notebook)
        self._apply_pipeline_labels(notebook, heuristic_result["cells_by_pipeline"])
        return heuristic_result

    def _analyze_with_llm(self, notebook: Notebook) -> dict[str, Any] | None:
        if self._llm_client is None:
            return None
        prompt = self._build_prompt(notebook)
        raw_response = self._llm_client.generate(prompt=prompt)
        parsed = parse_json_object(raw_response).value
        if parsed is None:
            return None
        if not self._is_valid_analysis(parsed):
            return None
        return cast(dict[str, Any], parsed)

    def _analyze_with_heuristics(self, notebook: Notebook) -> dict[str, Any]:
        grouped: dict[PipelineType, list[int]] = {
            PipelineType.FEATURE_ENGINEERING: [],
            PipelineType.TRAINING: [],
            PipelineType.INFERENCE: [],
            PipelineType.EVALUATION: [],
        }
        detected_libraries: set[str] = set()
        variable_mentions: dict[str, int] = {}

        for cell in notebook.cells:
            detected_libraries.update(self._extract_libraries(cell.source))
            stage = self._infer_pipeline_type(cell)
            if stage in grouped:
                grouped[stage].append(cell.index)
            for variable_name in self._extract_variable_mentions(cell.source):
                variable_mentions[variable_name] = (
                    variable_mentions.get(variable_name, 0) + 1
                )

        shared_variables = sorted(
            name for name, count in variable_mentions.items() if count >= 2
        )
        summary = (
            "Notebook analyzed with deterministic rules "
            "for pipeline stage classification."
        )
        return {
            "cells_by_pipeline": {
                "feature_engineering": grouped[PipelineType.FEATURE_ENGINEERING],
                "training": grouped[PipelineType.TRAINING],
                "inference": grouped[PipelineType.INFERENCE],
                "evaluation": grouped[PipelineType.EVALUATION],
            },
            "libraries": sorted(detected_libraries),
            "shared_variables": shared_variables,
            "summary": summary,
        }

    def _infer_pipeline_type(self, cell: NotebookCell) -> PipelineType:
        if cell.cell_type.value != "code":
            return PipelineType.UNKNOWN

        source = cell.source.lower()
        scores: dict[PipelineType, int] = {
            PipelineType.FEATURE_ENGINEERING: 0,
            PipelineType.TRAINING: 0,
            PipelineType.INFERENCE: 0,
            PipelineType.EVALUATION: 0,
        }

        feature_keywords = (
            "read_csv",
            "dataframe",
            "dropna",
            "drop_duplicates",
            "fillna",
            "standardscaler",
            "minmaxscaler",
            "onehot",
            "labelencoder",
            "train_test_split",
            "feature",
            "preprocess",
        )
        training_keywords = (
            "fit(",
            "train",
            "classifier(",
            "regression(",
            "xgboost",
            "lightgbm",
            "randomforest",
            "model =",
        )
        inference_keywords = (
            "predict(",
            "inference",
            "infer",
            "load_model",
        )
        evaluation_keywords = (
            "score",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "rmse",
            "mse",
            "r2",
            "confusion_matrix",
            "classification_report",
        )

        for keyword in feature_keywords:
            if keyword in source:
                scores[PipelineType.FEATURE_ENGINEERING] += 1
        for keyword in training_keywords:
            if keyword in source:
                scores[PipelineType.TRAINING] += 1
        for keyword in inference_keywords:
            if keyword in source:
                scores[PipelineType.INFERENCE] += 1
        for keyword in evaluation_keywords:
            if keyword in source:
                scores[PipelineType.EVALUATION] += 1

        if self._is_imports_only(source):
            scores[PipelineType.FEATURE_ENGINEERING] += 1

        best_stage = max(scores, key=lambda stage: scores[stage])
        if scores[best_stage] == 0:
            return PipelineType.UNKNOWN
        return best_stage

    def _extract_libraries(self, source: str) -> set[str]:
        libraries: set[str] = set()
        import_pattern = re.compile(
            r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))",
            re.MULTILINE,
        )
        for match in import_pattern.findall(source):
            module_name = match[0] or match[1]
            top_level_name = module_name.split(".")[0]
            if top_level_name:
                libraries.add(top_level_name)
        return libraries

    def _extract_variable_mentions(self, source: str) -> list[str]:
        return re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", source)

    def _is_imports_only(self, source: str) -> bool:
        stripped_lines = [line.strip() for line in source.splitlines() if line.strip()]
        if not stripped_lines:
            return False
        return all(
            line.startswith("import ") or line.startswith("from ")
            for line in stripped_lines
        )

    def _build_prompt(self, notebook: Notebook) -> str:
        prompt_template = Path(self._prompt_path).read_text(encoding="utf-8")
        notebook_payload = {
            "path": notebook.path,
            "cells": [
                {
                    "index": cell.index,
                    "cell_type": cell.cell_type.value,
                    "source": cell.source,
                }
                for cell in notebook.cells
            ],
        }
        return (
            f"{prompt_template}\n\nNotebook JSON:\n"
            f"{json.dumps(notebook_payload, ensure_ascii=True)}"
        )

    def _is_valid_analysis(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        cells_by_pipeline = payload.get("cells_by_pipeline")
        if not isinstance(cells_by_pipeline, dict):
            return False
        required_keys = {"feature_engineering", "training", "inference", "evaluation"}
        if set(cells_by_pipeline.keys()) != required_keys:
            return False
        for indexes in cells_by_pipeline.values():
            if not isinstance(indexes, list) or not all(
                isinstance(i, int) for i in indexes
            ):
                return False
        return True

    def _apply_pipeline_labels(
        self, notebook: Notebook, cells_by_pipeline: dict[str, list[int]]
    ) -> None:
        pipeline_map = {
            "feature_engineering": PipelineType.FEATURE_ENGINEERING,
            "training": PipelineType.TRAINING,
            "inference": PipelineType.INFERENCE,
            "evaluation": PipelineType.EVALUATION,
        }
        labels_by_cell: dict[int, PipelineType] = {}
        for pipeline_key, indexes in cells_by_pipeline.items():
            pipeline_type = pipeline_map.get(pipeline_key)
            if pipeline_type is None:
                continue
            for index in indexes:
                labels_by_cell[index] = pipeline_type
        for cell in notebook.cells:
            cell.pipeline_type = labels_by_cell.get(cell.index, PipelineType.UNKNOWN)
