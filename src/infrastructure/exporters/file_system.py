from __future__ import annotations

from pathlib import Path

_PACKAGE_NAME_ALIASES = {
    "sklearn": "scikit-learn",
}


class FileSystemOutputWriter:
    """Write generated project artifacts to disk."""

    def write_stage_modules(
        self, *, project_root: str, generated_modules: dict[str, str]
    ) -> dict[str, str]:
        src_root = Path(project_root) / "src"
        src_root.mkdir(parents=True, exist_ok=True)
        file_map: dict[str, str] = {}
        for stage_name, source_code in generated_modules.items():
            file_path = src_root / f"{stage_name}.py"
            file_path.write_text(source_code, encoding="utf-8")
            file_map[stage_name] = str(file_path)
        return file_map

    def write_stage_tests(
        self, *, project_root: str, generated_tests: dict[str, str]
    ) -> dict[str, str]:
        tests_root = Path(project_root) / "tests"
        tests_root.mkdir(parents=True, exist_ok=True)
        file_map: dict[str, str] = {}
        for stage_name, source_code in generated_tests.items():
            file_path = tests_root / f"test_{stage_name}.py"
            file_path.write_text(source_code, encoding="utf-8")
            file_map[stage_name] = str(file_path)
        return file_map

    def write_requirements(
        self, *, project_root: str, libraries: list[str] | None = None
    ) -> str:
        requirements_path = Path(project_root) / "requirements.txt"
        normalized = sorted(
            {_PACKAGE_NAME_ALIASES.get(library, library) for library in libraries or []}
        )
        requirements_path.write_text(
            "\n".join(normalized) + ("\n" if normalized else "")
        )
        return str(requirements_path)

    def write_readme(
        self,
        *,
        project_root: str,
        project_name: str,
        libraries: list[str] | None = None,
    ) -> str:
        readme_path = Path(project_root) / "README.md"
        libs = ", ".join(sorted(set(libraries or []))) or "not detected"
        content = (
            f"# {project_name}\n\n"
            "Generated automatically by mlops-agent.\n\n"
            "## Project layout\n\n"
            "- `src/`: pipeline modules\n"
            "- `tests/`: generated tests\n"
            "- `python src/main.py`: execute the complete pipeline\n"
            "- `requirements.txt`: dependencies\n\n"
            f"Detected libraries: {libs}\n"
        )
        readme_path.write_text(content, encoding="utf-8")
        return str(readme_path)

    def write_entrypoint(self, *, project_root: str) -> str:
        """Write the standalone entrypoint for the generated pipeline."""
        entrypoint_path = Path(project_root) / "src" / "main.py"
        entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
        entrypoint_path.write_text(
            '''"""Executable entrypoint for the generated ML pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import evaluation
import feature_engineering
import inference
import training


def _call_with_fallback(function: Any, *args: Any) -> Any:
    """Call a generated function, retrying without optional context."""
    try:
        return function(*args)
    except TypeError:
        if len(args) > 1:
            return function(args[0])
        raise


def run_pipeline() -> float:
    """Run all generated stages and return the final MSE metric."""
    loaded = _call_with_fallback(feature_engineering.load_data, "")
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("load_data must return features and labels")
    features, labels = loaded

    clean_data = getattr(feature_engineering, "clean_data", None)
    if callable(clean_data):
        features, labels = _call_with_fallback(clean_data, features, labels)
    prepare_features = getattr(feature_engineering, "prepare_features", None)
    if callable(prepare_features):
        features = prepare_features(features)

    split_data = getattr(feature_engineering, "split_data", None)
    if not callable(split_data):
        raise RuntimeError("feature_engineering.split_data is required")
    split = _call_with_fallback(split_data, features, labels)
    if not isinstance(split, tuple) or len(split) != 4:
        raise RuntimeError("split_data must return four values")
    train_features, test_features, train_labels, test_labels = split

    model = training.train_model(train_features, train_labels)
    predictions = inference.predict(model, test_features)
    metrics = evaluation.evaluate_model(test_labels, predictions)
    if not isinstance(metrics, Mapping):
        raise RuntimeError("evaluate_model must return a metrics mapping")

    value = metrics.get("final_mse", metrics.get("mse"))
    if value is None:
        raise RuntimeError("evaluation metrics must include final_mse or mse")
    return float(value)


if __name__ == "__main__":
    print(f"final_mse={run_pipeline():.12f}")
''',
            encoding="utf-8",
        )
        return str(entrypoint_path)

    def write_pyproject(self, *, project_root: str, project_name: str) -> str:
        pyproject_path = Path(project_root) / "pyproject.toml"
        content = (
            "[project]\n"
            f'name = "{project_name}"\n'
            'version = "0.1.0"\n'
            'description = "Generated by mlops-agent"\n'
            'requires-python = ">=3.11"\n\n'
            "[tool.pytest.ini_options]\n"
            'testpaths = ["tests"]\n'
        )
        pyproject_path.write_text(content, encoding="utf-8")
        return str(pyproject_path)
