from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.domain.entities import Pipeline, PipelineType
from src.domain.interfaces import IExporter
from src.infrastructure.exporters.file_system import FileSystemOutputWriter


class ZipExporter(IExporter):
    """Export generated pipeline project as a zip archive."""

    def __init__(self, *, writer: FileSystemOutputWriter | None = None) -> None:
        self._writer = writer or FileSystemOutputWriter()

    def export(self, pipeline: Pipeline, output_dir: str, **kwargs: object) -> str:
        project_name = str(
            kwargs.get("project_name")
            or Path(pipeline.notebook.path).stem.replace(" ", "_")
            or "generated_project"
        )
        libraries = kwargs.get("libraries")
        libs_list = list(libraries) if isinstance(libraries, list) else []

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        project_root = base_dir / project_name
        project_root.mkdir(parents=True, exist_ok=True)

        generated_modules = self._collect_stage_codes(pipeline=pipeline)
        generated_tests = self._collect_stage_tests(pipeline=pipeline)

        self._writer.write_stage_modules(
            project_root=str(project_root),
            generated_modules=generated_modules,
        )
        self._writer.write_stage_tests(
            project_root=str(project_root),
            generated_tests=generated_tests,
        )
        self._writer.write_requirements(
            project_root=str(project_root), libraries=libs_list
        )
        self._writer.write_readme(
            project_root=str(project_root),
            project_name=project_name,
            libraries=libs_list,
        )
        self._writer.write_pyproject(
            project_root=str(project_root), project_name=project_name
        )

        zip_path = base_dir / f"{project_name}.zip"
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
            for file_path in sorted(project_root.rglob("*")):
                if file_path.is_file():
                    zip_file.write(
                        file_path, arcname=str(file_path.relative_to(project_root))
                    )
        return str(zip_path)

    def _collect_stage_codes(self, *, pipeline: Pipeline) -> dict[str, str]:
        modules: dict[str, str] = {}
        for stage in pipeline.stages:
            if stage.generated_code is None:
                continue
            if stage.pipeline_type == PipelineType.UNKNOWN:
                continue
            modules[stage.pipeline_type.value] = stage.generated_code
        return modules

    def _collect_stage_tests(self, *, pipeline: Pipeline) -> dict[str, str]:
        tests: dict[str, str] = {}
        for stage in pipeline.stages:
            if stage.generated_tests is None:
                continue
            if stage.pipeline_type == PipelineType.UNKNOWN:
                continue
            tests[stage.pipeline_type.value] = stage.generated_tests
        return tests
