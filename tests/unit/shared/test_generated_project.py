from __future__ import annotations

import tomllib
from pathlib import Path

from src.shared.generated_project import (
    PACKAGE_NAME_ALIASES,
    detect_project_libraries,
    normalize_package_names,
    parse_project_name,
    render_pyproject,
    render_requirements,
    write_project_metadata,
)


def _declared_packages(root: Path) -> tuple[list[str], list[str]]:
    requirements = [
        line
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parsed = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = parsed["project"]["dependencies"]
    assert isinstance(dependencies, list)
    return requirements, list(dependencies)


def test_normalize_package_names_maps_sklearn_and_deduplicates() -> None:
    assert PACKAGE_NAME_ALIASES["sklearn"] == "scikit-learn"
    assert normalize_package_names(
        ["numpy", "sklearn", "pandas", "numpy", "sklearn"]
    ) == ["numpy", "pandas", "scikit-learn"]


def test_normalize_package_names_skips_generated_modules() -> None:
    assert normalize_package_names(["training", "numpy", "main"]) == ["numpy"]


def test_render_functions_declare_the_same_packages() -> None:
    libraries = ["sklearn", "numpy", "pandas"]
    requirements = render_requirements(libraries).splitlines()
    parsed = tomllib.loads(render_pyproject(project_name="demo", libraries=libraries))

    assert requirements == ["numpy", "pandas", "scikit-learn"]
    assert parsed["project"]["dependencies"] == requirements


def test_write_project_metadata_keeps_requirements_and_pyproject_aligned(
    tmp_path: Path,
) -> None:
    write_project_metadata(
        tmp_path,
        project_name="simple_regression",
        libraries=["sklearn", "pandas"],
    )
    requirements, pyproject_deps = _declared_packages(tmp_path)

    assert requirements == ["pandas", "scikit-learn"]
    assert pyproject_deps == requirements
    assert (
        parse_project_name((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
        == "simple_regression"
    )


def test_detect_project_libraries_reads_sklearn_from_generated_source(
    tmp_path: Path,
) -> None:
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "training.py").write_text(
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression\n\n"
        "def train_model() -> LinearRegression:\n"
        "    return LinearRegression()\n",
        encoding="utf-8",
    )

    assert detect_project_libraries(tmp_path) == ["numpy", "scikit-learn"]


def test_detect_project_libraries_reads_existing_packaging_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        render_pyproject(project_name="demo", libraries=["sklearn"]),
        encoding="utf-8",
    )

    assert detect_project_libraries(tmp_path) == ["pandas", "scikit-learn"]


def test_parse_project_name_ignores_malformed_lines() -> None:
    assert parse_project_name("name\nname = \n") is None
    assert parse_project_name('name = "demo"\n') == "demo"


def test_libraries_from_invalid_pyproject_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("not = [toml", encoding="utf-8")
    assert detect_project_libraries(tmp_path) == []

    (tmp_path / "pyproject.toml").write_text("project = []\n", encoding="utf-8")
    assert detect_project_libraries(tmp_path) == []

    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = { invalid = true }\n",
        encoding="utf-8",
    )
    assert detect_project_libraries(tmp_path) == []


def test_pyproject_template_lives_in_a_single_module() -> None:
    exporter = Path("src/infrastructure/exporters/file_system.py").read_text(
        encoding="utf-8"
    )
    validator = Path("src/application/validate_output.py").read_text(encoding="utf-8")

    assert "[project]" not in exporter
    assert "[project]" not in validator
    assert "render_pyproject" in exporter
    assert "write_project_metadata" in validator
