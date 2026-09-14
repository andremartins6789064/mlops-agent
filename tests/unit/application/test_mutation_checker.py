from __future__ import annotations

from pathlib import Path
from typing import Any

from src.application.mutation_checker import run_mutation_check


def _write_project(root: Path, test_source: str) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "calculator.py").write_text(
        "def add(first: int, second: int) -> int:\n    return first + second\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_calculator.py").write_text(
        test_source,
        encoding="utf-8",
    )


def test_mutation_checker_detects_meaningful_test_suite(tmp_path: Any) -> None:
    _write_project(
        tmp_path,
        "from src.calculator import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
    )

    result = run_mutation_check(str(tmp_path))

    assert result.total_mutations == 1
    assert result.killed_mutations == 1
    assert result.survived_mutations == 0
    assert result.mutation_score == 1.0
    assert result.passed


def test_mutation_checker_rejects_placeholder_suite(tmp_path: Any) -> None:
    _write_project(
        tmp_path,
        "from src import calculator\n\n"
        "def test_module_exists() -> None:\n"
        "    assert hasattr(calculator, 'add')\n",
    )

    result = run_mutation_check(str(tmp_path))

    assert result.total_mutations == 1
    assert result.killed_mutations == 0
    assert result.survived_mutations == 1
    assert result.mutation_score == 0.0
    assert not result.passed


def test_mutation_checker_rejects_invalid_baseline_suite(tmp_path: Any) -> None:
    _write_project(
        tmp_path,
        "import module_that_does_not_exist\n\n"
        "def test_never_collects() -> None:\n"
        "    assert True\n",
    )

    result = run_mutation_check(str(tmp_path))

    assert result.status == "suite_invalida"
    assert not result.baseline_passed
    assert result.total_mutations == 0
    assert result.mutation_score == 0.0
    assert not result.passed
    assert result.baseline_error is not None


def test_mutation_checker_handles_project_without_functions(tmp_path: Any) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "constants.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_constants.py").write_text(
        "def test_value() -> None:\n    assert True\n",
        encoding="utf-8",
    )

    result = run_mutation_check(str(tmp_path))

    assert result.total_mutations == 0
    assert not result.passed
