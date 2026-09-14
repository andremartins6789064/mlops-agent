from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.domain.entities import Notebook
from src.domain.interfaces import ILLMClient
from src.domain.pipeline_contract import prompt_signatures
from src.shared.config import settings
from src.shared.llm_parsing import parse_json_object, parse_python_block
from src.shared.progress import ProgressCallback, ProgressEvent
from src.shared.provenance import StageProvenance
from src.shared.python_source import python_syntax_error


class CodeGeneratorAgent:
    """Generate Python modules for each pipeline stage."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/code_generator.txt",
        stage_feedback: dict[str, str] | None = None,
        context_budget_chars: int = 12_000,
    ) -> None:
        if context_budget_chars <= 0:
            raise ValueError("context_budget_chars must be greater than zero")
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._context_budget_chars = context_budget_chars
        self._stage_provenance: dict[str, StageProvenance] = {}
        self._last_context_truncated = False
        self._stage_feedback = {
            key: value.strip()
            for key, value in (stage_feedback or {}).items()
            if value.strip()
        }

    def generate_modules(
        self,
        *,
        notebook: Notebook,
        notebook_analysis: dict[str, Any],
        architecture_plan: dict[str, Any],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, str]:
        """Return generated python source code for all pipeline modules."""
        modules = ["feature_engineering", "training", "inference", "evaluation"]
        generated: dict[str, str] = {}
        self._stage_provenance = {}
        for module_name in modules:
            started_at = time.perf_counter()
            self._last_context_truncated = False
            llm_code, parse_method, fallback_reason = self._generate_module_with_llm(
                module_name=module_name,
                notebook=notebook,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
            )
            if llm_code is not None:
                generated[module_name] = llm_code
                origin = "llm"
            else:
                generated[module_name] = self._generate_module_with_templates(
                    module_name=module_name,
                    notebook_analysis=notebook_analysis,
                    architecture_plan=architecture_plan,
                )
                origin = "template"
            self._stage_provenance[module_name] = StageProvenance(
                origin=origin,
                fallback_reason=fallback_reason,
                model=self._model_name(),
                duration_seconds=time.perf_counter() - started_at,
                context_truncated=self._last_context_truncated,
                parse_method=parse_method,
            )
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase=module_name,
                        message=f"Estágio concluído: {module_name}.",
                        completed=4 + modules.index(module_name),
                        total=10,
                    )
                )
        return generated

    @property
    def stage_provenance(self) -> dict[str, StageProvenance]:
        """Return provenance for the most recent module generation."""
        return dict(self._stage_provenance)

    def write_modules(
        self,
        *,
        generated_modules: dict[str, str],
        output_dir: str = "output",
    ) -> dict[str, str]:
        """Persist generated modules to disk and return file path map."""
        output_path = Path(output_dir) / "src"
        output_path.mkdir(parents=True, exist_ok=True)
        file_map: dict[str, str] = {}
        for module_name, source_code in generated_modules.items():
            file_path = output_path / f"{module_name}.py"
            file_path.write_text(source_code, encoding="utf-8")
            file_map[module_name] = str(file_path)
        return file_map

    def _generate_module_with_llm(
        self,
        *,
        module_name: str,
        notebook: Notebook,
        notebook_analysis: dict[str, Any],
        architecture_plan: dict[str, Any],
    ) -> tuple[str | None, str | None, str | None]:
        if self._llm_client is None:
            return None, None, "LLM client is not configured"
        prompt = self._build_prompt(
            module_name=module_name,
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
        )
        raw_response = self._llm_client.generate(prompt=prompt)
        python_result = parse_python_block(raw_response)
        if python_result.method == "fenced" and isinstance(python_result.value, str):
            return self._accept_python_response(
                python_result.value, python_result.method
            )
        json_result = parse_json_object(raw_response)
        parsed_payload = json_result.value
        if isinstance(parsed_payload, dict):
            module_code = parsed_payload.get("module_code")
            if isinstance(module_code, str) and "def " in module_code:
                return self._accept_python_response(
                    module_code.strip(), json_result.method
                )
        python_code = python_result.value
        if isinstance(python_code, str):
            return self._accept_python_response(python_code, python_result.method)
        return (
            None,
            None,
            f"Unable to parse LLM response (JSON: {json_result.method}; "
            f"Python: {python_result.method})",
        )

    def _accept_python_response(
        self, source: str, parse_method: str
    ) -> tuple[str | None, str | None, str | None]:
        """Accept only syntactically valid Python returned by the LLM."""
        syntax_error = python_syntax_error(source)
        if syntax_error is not None:
            return None, None, f"Generated Python is invalid: {syntax_error}"
        return source, parse_method, None

    def _generate_module_with_templates(
        self,
        *,
        module_name: str,
        notebook_analysis: dict[str, Any],
        architecture_plan: dict[str, Any],
    ) -> str:
        libraries = notebook_analysis.get("libraries", [])
        modules = architecture_plan.get("modules", {})
        stage_plan = modules.get(module_name, {})
        functions = stage_plan.get("functions", [])
        if not isinstance(functions, list):
            functions = []

        imports = self._build_imports(module_name=module_name, libraries=libraries)
        function_blocks = [
            self._build_function_stub(
                module_name=module_name, function_name=str(function_name)
            )
            for function_name in functions
        ]
        if not function_blocks:
            function_blocks = [
                self._build_function_stub(
                    module_name=module_name, function_name=f"run_{module_name}"
                )
            ]

        header = f'"""{module_name}.py\nGenerated automatically by mlops-agent.\n"""\n'
        feedback_note = self._stage_feedback.get(module_name)
        if feedback_note:
            header = (
                f'"""{module_name}.py\nGenerated automatically by mlops-agent.\n'
                f"User feedback: {feedback_note}\n"
                '"""\n'
            )
        return f"{header}\n{imports}\n\n" + "\n\n".join(function_blocks) + "\n"

    def _build_prompt(
        self,
        *,
        module_name: str,
        notebook: Notebook,
        notebook_analysis: dict[str, Any],
        architecture_plan: dict[str, Any],
    ) -> str:
        prompt_template = Path(self._prompt_path).read_text(encoding="utf-8")
        prompt_template = prompt_template.replace(
            "__REQUIRED_SIGNATURES__",
            prompt_signatures(module_name=module_name),
        )
        stage_cells = self._build_stage_cell_context(
            module_name=module_name,
            notebook=notebook,
            notebook_analysis=notebook_analysis,
        )
        self._last_context_truncated = bool(stage_cells["truncated"])
        payload = {
            "module_name": module_name,
            "notebook_path": notebook.path,
            "stage_cells": stage_cells,
            "shared_variables": notebook_analysis.get("shared_variables", []),
            "module_plan": architecture_plan.get("modules", {}).get(module_name, {}),
            "stage_feedback": self._stage_feedback.get(module_name),
            "required_signatures": prompt_signatures(module_name=module_name),
        }
        return f"{prompt_template}\n\nGeneration context JSON:\n{json.dumps(payload)}"

    def _model_name(self) -> str | None:
        """Return the configured model name when available."""
        if self._llm_client is None:
            return None
        model = getattr(self._llm_client, "model", None)
        if isinstance(model, str):
            return model
        private_model = getattr(self._llm_client, "_model", None)
        if isinstance(private_model, str):
            return private_model
        return settings.llm_model

    def _build_stage_cell_context(
        self,
        *,
        module_name: str,
        notebook: Notebook,
        notebook_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the selected stage's source with an explicit character budget."""
        cells_by_pipeline = notebook_analysis.get("cells_by_pipeline", {})
        indexes = (
            cells_by_pipeline.get(module_name, [])
            if isinstance(cells_by_pipeline, dict)
            else []
        )
        valid_indexes = (
            {index for index in indexes if isinstance(index, int)}
            if isinstance(indexes, list)
            else set()
        )
        selected_cells = [
            cell for cell in notebook.cells if cell.index in valid_indexes
        ]

        remaining = self._context_budget_chars
        serialized_cells: list[dict[str, Any]] = []
        truncated = False
        original_chars = sum(len(cell.source) for cell in selected_cells)
        included_chars = 0
        for cell in selected_cells:
            source = cell.source
            included_source = source[:remaining]
            included_chars += len(included_source)
            cell_was_truncated = len(included_source) < len(source)
            serialized_cells.append(
                {
                    "index": cell.index,
                    "cell_type": cell.cell_type.value,
                    "source": included_source,
                    "truncated": cell_was_truncated,
                }
            )
            remaining -= len(included_source)
            truncated = truncated or cell_was_truncated
            if remaining == 0:
                break

        return {
            "module_name": module_name,
            "cells": serialized_cells,
            "context_budget_chars": self._context_budget_chars,
            "original_source_chars": original_chars,
            "included_source_chars": included_chars,
            "truncated": truncated,
        }

    def _build_imports(self, *, module_name: str, libraries: Any) -> str:
        import_lines = ["from __future__ import annotations", "from typing import Any"]
        if module_name in {"training", "inference"}:
            import_lines.extend(["import pickle", "from pathlib import Path"])
        return "\n".join(import_lines)

    def _build_function_stub(self, *, module_name: str, function_name: str) -> str:
        if function_name == "load_data":
            return (
                "def load_data(path: str | None = None) -> tuple[Any, Any]:\n"
                '    """Return (features, labels) without requiring a file."""\n'
                "    if path is None:\n"
                "        return [], []\n"
                "    return [path], []\n"
            )
        if function_name == "split_data":
            return (
                "def split_data(\n"
                "    features: Any, labels: Any\n"
                ") -> tuple[Any, Any, Any, Any]:\n"
                '    """Split dataset into train/test placeholders."""\n'
                "    return features, features, labels, labels\n"
            )
        if function_name == "clean_data":
            return (
                "def clean_data(features: Any, labels: Any) -> tuple[Any, Any]:\n"
                '    """Passthrough cleaner placeholder."""\n'
                "    return features, labels\n"
            )
        if function_name == "prepare_features":
            return (
                "def prepare_features(features: Any, labels: Any) -> tuple[Any, Any]:\n"
                '    """Passthrough feature placeholder."""\n'
                "    return features, labels\n"
            )
        if function_name == "train_model":
            return (
                "def train_model(features: Any, labels: Any) -> Any:\n"
                '    """Train model placeholder using provided data."""\n'
                '    model = {"features": features, "labels": labels}\n'
                "    return model\n"
            )
        if function_name == "save_model":
            return (
                "def save_model(model: Any, model_path: str) -> str:\n"
                '    """Persist model artifact with pickle."""\n'
                "    path = Path(model_path)\n"
                '    with path.open("wb") as file_obj:\n'
                "        pickle.dump(model, file_obj)\n"
                "    return str(path)\n"
            )
        if function_name == "load_model":
            return (
                "def load_model(model_path: str) -> Any:\n"
                '    """Load pickled model artifact."""\n'
                "    path = Path(model_path)\n"
                '    with path.open("rb") as file_obj:\n'
                "        return pickle.load(file_obj)\n"
            )
        if function_name == "predict":
            return (
                "def predict(model: Any, features: Any) -> Any:\n"
                '    """Run prediction using model interface when available."""\n'
                '    if hasattr(model, "predict"):\n'
                "        return model.predict(features)\n"
                "    return features\n"
            )
        if function_name in {"evaluate_model", "build_metrics_report"}:
            return (
                f"def {function_name}(test_labels: Any, predictions: Any) -> "
                "dict[str, float]:\n"
                '    """Return a metrics mapping that includes final_mse."""\n'
                '    return {"final_mse": 0.0}\n'
            )
        return (
            f"def {function_name}(*args: Any, **kwargs: Any) -> Any:\n"
            f'    """{module_name} stage function '
            'generated from architecture plan."""\n'
            "    return None\n"
        )
