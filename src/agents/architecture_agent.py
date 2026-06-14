from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from src.domain.interfaces import ILLMClient


class ArchitectureAgent:
    """Build architecture plan from notebook analysis."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/architecture_agent.txt",
        target_structure_path: str = "configs/target_structure.yaml",
    ) -> None:
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._target_structure_path = target_structure_path

    def plan(self, notebook_analysis: dict[str, Any]) -> dict[str, Any]:
        """Return architecture JSON with module contracts."""
        if self._llm_client is not None:
            llm_result = self._plan_with_llm(notebook_analysis)
            if llm_result is not None:
                return llm_result
        return self._plan_with_templates(notebook_analysis)

    def _plan_with_llm(
        self, notebook_analysis: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._llm_client is None:
            return None
        prompt = self._build_prompt(notebook_analysis)
        raw_response = self._llm_client.generate(prompt=prompt)
        parsed = self._parse_json_response(raw_response)
        if parsed is None:
            return None
        if not self._is_valid_plan(parsed):
            return None
        return cast(dict[str, Any], parsed)

    def _plan_with_templates(self, notebook_analysis: dict[str, Any]) -> dict[str, Any]:
        libraries = notebook_analysis.get("libraries", [])
        base_functions = {
            "feature_engineering": [
                "load_data",
                "clean_data",
                "prepare_features",
                "split_data",
            ],
            "training": ["train_model", "save_model"],
            "inference": ["load_model", "predict"],
            "evaluation": ["evaluate_model", "build_metrics_report"],
        }
        if (
            "sklearn" in libraries
            and "scale_features" not in base_functions["feature_engineering"]
        ):
            base_functions["feature_engineering"].append("scale_features")

        return {
            "modules": {
                "feature_engineering": {
                    "functions": base_functions["feature_engineering"],
                    "inputs": "raw dataset path or dataframe",
                    "outputs": "processed features and train/test splits",
                },
                "training": {
                    "functions": base_functions["training"],
                    "inputs": "training features and labels",
                    "outputs": "trained model artifact",
                },
                "inference": {
                    "functions": base_functions["inference"],
                    "inputs": "trained model artifact and new samples",
                    "outputs": "predictions array",
                },
                "evaluation": {
                    "functions": base_functions["evaluation"],
                    "inputs": "true values and predictions",
                    "outputs": "evaluation metrics dictionary",
                },
            }
        }

    def _build_prompt(self, notebook_analysis: dict[str, Any]) -> str:
        prompt_template = Path(self._prompt_path).read_text(encoding="utf-8")
        target_structure = Path(self._target_structure_path).read_text(encoding="utf-8")
        notebook_analysis_json = json.dumps(notebook_analysis, ensure_ascii=True)
        return (
            f"{prompt_template}\n\nTarget structure YAML:\n{target_structure}\n\n"
            f"Notebook analysis JSON:\n{notebook_analysis_json}"
        )

    def _is_valid_plan(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        modules = payload.get("modules")
        if not isinstance(modules, dict):
            return False
        required = {"feature_engineering", "training", "inference", "evaluation"}
        if set(modules.keys()) != required:
            return False
        for module_data in modules.values():
            if not isinstance(module_data, dict):
                return False
            functions = module_data.get("functions")
            if not isinstance(functions, list) or not all(
                isinstance(function_name, str) for function_name in functions
            ):
                return False
        return True

    def _parse_json_response(self, raw_response: str) -> Any | None:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL
        )
        if fenced_match is not None:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                return None

        object_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if object_match is not None:
            try:
                return json.loads(object_match.group(0))
            except json.JSONDecodeError:
                return None
        return None
