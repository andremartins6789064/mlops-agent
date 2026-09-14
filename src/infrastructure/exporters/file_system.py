from __future__ import annotations

from pathlib import Path

from src.domain.pipeline_contract import entrypoint_source
from src.shared.generated_project import render_pyproject, render_requirements


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
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        requirements_path.write_text(
            render_requirements(libraries or []), encoding="utf-8"
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
        entrypoint_path.write_text(entrypoint_source(), encoding="utf-8")
        return str(entrypoint_path)

    def write_pyproject(
        self,
        *,
        project_root: str,
        project_name: str,
        libraries: list[str] | None = None,
    ) -> str:
        pyproject_path = Path(project_root) / "pyproject.toml"
        pyproject_path.parent.mkdir(parents=True, exist_ok=True)
        pyproject_path.write_text(
            render_pyproject(project_name=project_name, libraries=libraries or []),
            encoding="utf-8",
        )
        return str(pyproject_path)
