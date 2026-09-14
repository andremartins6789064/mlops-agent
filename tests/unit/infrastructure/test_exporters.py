from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import nbformat

from src.application.equivalence_runner import EquivalenceStatus, run_equivalence
from src.domain.entities import (
    CellType,
    Notebook,
    NotebookCell,
    Pipeline,
    PipelineStage,
    PipelineType,
)
from src.infrastructure.exporters.file_system import FileSystemOutputWriter
from src.infrastructure.exporters.zip_exporter import ZipExporter

# mypy: disable-error-code=no-untyped-call


def _build_pipeline() -> Pipeline:
    notebook = Notebook(
        path="tests/fixtures/simple_regression.ipynb",
        cells=[NotebookCell(index=0, cell_type=CellType.CODE, source="print('ok')")],
        metadata={},
    )
    stages = [
        PipelineStage(
            pipeline_type=PipelineType.FEATURE_ENGINEERING,
            cells=[],
            generated_code="def load_data(path: str) -> str:\n    return path\n",
            generated_tests="def test_feature() -> None:\n    assert True\n",
        ),
        PipelineStage(
            pipeline_type=PipelineType.TRAINING,
            cells=[],
            generated_code=(
                "def train_model(x: object, y: object) -> dict[str, bool]:\n"
                "    return {'ok': True}\n"
            ),
            generated_tests="def test_training() -> None:\n    assert True\n",
        ),
    ]
    return Pipeline(notebook=notebook, stages=stages)


def test_file_system_writer_creates_expected_artifacts(tmp_path: Path) -> None:
    writer = FileSystemOutputWriter()
    root = tmp_path / "project"
    writer.write_stage_modules(
        project_root=str(root),
        generated_modules={"inference": "def predict() -> int:\n    return 1\n"},
    )
    writer.write_stage_tests(
        project_root=str(root),
        generated_tests={"inference": "def test_predict() -> None:\n    assert True\n"},
    )
    writer.write_requirements(
        project_root=str(root), libraries=["numpy", "pandas", "numpy", "sklearn"]
    )
    writer.write_readme(
        project_root=str(root), project_name="demo", libraries=["numpy"]
    )
    writer.write_pyproject(project_root=str(root), project_name="demo")

    assert (root / "src" / "inference.py").exists()
    assert (root / "tests" / "test_inference.py").exists()
    assert (root / "requirements.txt").read_text(
        encoding="utf-8"
    ) == "numpy\npandas\nscikit-learn\n"
    assert (root / "README.md").exists()
    assert (root / "pyproject.toml").exists()


def test_file_system_writer_creates_executable_entrypoint(tmp_path: Path) -> None:
    writer = FileSystemOutputWriter()
    root = tmp_path / "project"
    writer.write_stage_modules(
        project_root=str(root),
        generated_modules={
            "feature_engineering": (
                "def load_data(path: str) -> tuple[list[int], list[int]]:\n"
                "    return [1, 2], [2, 4]\n\n"
                "def split_data(X: list[int], y: list[int]) -> "
                "tuple[list[int], list[int], list[int], list[int]]:\n"
                "    return X, X, y, y\n"
            ),
            "training": (
                "def train_model(X: list[int], y: list[int]) -> dict[str, int]:\n"
                "    return {'coefficient': 2}\n"
            ),
            "inference": (
                "def predict(model: dict[str, int], X: list[int]) -> list[int]:\n"
                "    return [model['coefficient'] * value for value in X]\n"
            ),
            "evaluation": (
                "def evaluate_model(y_true: list[int], y_pred: list[int]) -> "
                "dict[str, float]:\n"
                "    return {'final_mse': 0.0}\n"
            ),
        },
    )
    entrypoint = writer.write_entrypoint(project_root=str(root))

    result = subprocess.run(
        [sys.executable, entrypoint],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "final_mse=0.000000000000" in result.stdout

    notebook_path = root / "source.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("print('final_mse=0.0')")]
        ),
        notebook_path,
    )
    equivalence = run_equivalence(
        notebook_path=str(notebook_path),
        pipeline_command=[sys.executable, "src/main.py"],
        pipeline_dir=str(root),
    )
    assert equivalence.status == EquivalenceStatus.EQUIVALENT


def test_zip_exporter_creates_valid_archive_with_expected_layout(
    tmp_path: Path,
) -> None:
    exporter = ZipExporter()
    pipeline = _build_pipeline()

    archive_path = exporter.export(
        pipeline,
        str(tmp_path),
        project_name="generated_demo",
        libraries=["pandas", "sklearn"],
    )

    archive = Path(archive_path)
    assert archive.exists()
    with ZipFile(archive, "r") as zip_file:
        names = set(zip_file.namelist())

    assert "src/feature_engineering.py" in names
    assert "src/training.py" in names
    assert "src/main.py" in names
    assert "tests/test_feature_engineering.py" in names
    assert "tests/test_training.py" in names
    assert "requirements.txt" in names
    assert "pyproject.toml" in names
    assert "README.md" in names
