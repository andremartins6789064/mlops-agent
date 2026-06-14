from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.domain.entities import Notebook
from src.domain.interfaces import ILLMClient


class CodeGeneratorAgent:
    """Generate Python modules for each pipeline stage."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/code_generator.txt",
    ) -> None:
        self._llm_client = llm_client
        self._prompt_path = prompt_path

    def generate_modules(
        self,
        *,
        notebook: Notebook,
        notebook_analysis: dict[str, Any],
        architecture_plan: dict[str, Any],
    ) -> dict[str, str]:
        """Return generated python source code for all pipeline modules."""
        modules = ["feature_engineering", "training", "inference", "evaluation"]
        generated: dict[str, str] = {}
        for module_name in modules:
            llm_code = self._generate_module_with_llm(
                module_name=module_name,
                notebook=notebook,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
            )
            if llm_code is not None:
                generated[module_name] = llm_code
                continue
            generated[module_name] = self._generate_module_with_templates(
                module_name=module_name,
                notebook_analysis=notebook_analysis,
                architecture_plan=architecture_plan,
            )
        return generated

    def write_modules(
        self,
        *,
        generated_modules: dict[str, str],
        output_dir: str = "output",
    ) -> dict[str, str]:
        """Persist generated modules to disk and return file path map."""
        output_path = Path(output_dir)
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
    ) -> str | None:
        if self._llm_client is None:
            return None
        prompt = self._build_prompt(
            module_name=module_name,
            notebook=notebook,
            notebook_analysis=notebook_analysis,
            architecture_plan=architecture_plan,
        )
        raw_response = self._llm_client.generate(prompt=prompt)
        parsed_payload = self._parse_json_response(raw_response)
        if isinstance(parsed_payload, dict):
            module_code = parsed_payload.get("module_code")
            if isinstance(module_code, str) and "def " in module_code:
                return module_code.strip()
        if "def " in raw_response:
            return self._extract_python_block(raw_response).strip()
        return None

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
        payload = {
            "module_name": module_name,
            "notebook_path": notebook.path,
            "cells_by_pipeline": notebook_analysis.get("cells_by_pipeline", {}),
            "module_plan": architecture_plan.get("modules", {}).get(module_name, {}),
        }
        return f"{prompt_template}\n\nGeneration context JSON:\n{json.dumps(payload)}"

    def _build_imports(self, *, module_name: str, libraries: Any) -> str:
        import_lines = ["from __future__ import annotations", "from typing import Any"]
        if (
            isinstance(libraries, list)
            and "pandas" in libraries
            and module_name == "feature_engineering"
        ):
            import_lines.append("import pandas as pd")
        if module_name in {"training", "inference"}:
            import_lines.extend(["import pickle", "from pathlib import Path"])
        return "\n".join(import_lines)

    def _build_function_stub(self, *, module_name: str, function_name: str) -> str:
        if function_name == "load_data":
            return (
                "def load_data(path: str) -> Any:\n"
                '    """Load input data from path."""\n'
                "    try:\n"
                "        return pd.read_csv(path)  # type: ignore[name-defined]\n"
                "    except Exception:\n"
                "        return path\n"
            )
        if function_name == "split_data":
            return (
                "def split_data(data: Any) -> tuple[Any, Any, Any, Any]:\n"
                '    """Split dataset into train/test placeholders."""\n'
                "    return data, data, data, data\n"
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
                f"def {function_name}(y_true: Any, y_pred: Any) -> dict[str, float]:\n"
                '    """Compute simple evaluation metrics placeholder."""\n'
                '    size_true = len(y_true) if hasattr(y_true, "__len__") else 0\n'
                '    size_pred = len(y_pred) if hasattr(y_pred, "__len__") else 0\n'
                "    if size_true == 0:\n"
                '        return {"coverage": 0.0}\n'
                "    ratio = min(size_true, size_pred) / float(size_true)\n"
                '    return {"coverage": ratio}\n'
            )
        return (
            f"def {function_name}(*args: Any, **kwargs: Any) -> Any:\n"
            f'    """{module_name} stage function '
            'generated from architecture plan."""\n'
            "    return None\n"
        )

    def _parse_json_response(self, raw_response: str) -> Any | None:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_response,
            re.DOTALL,
        )
        if fenced_match is not None:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    def _extract_python_block(self, raw_response: str) -> str:
        python_match = re.search(
            r"```(?:python)?\s*(.*?)\s*```", raw_response, re.DOTALL
        )
        if python_match is not None:
            return python_match.group(1)
        return raw_response
