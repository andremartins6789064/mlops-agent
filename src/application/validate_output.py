from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ValidationResult:
    """Result of quality validation commands over generated output."""

    lint_errors: int
    type_errors: int
    test_coverage: float
    lint_output: str
    type_output: str
    test_output: str
    lint_exit_code: int
    type_exit_code: int
    test_exit_code: int
    minimum_coverage: float = 80.0
    timed_out: bool = False

    @property
    def has_errors(self) -> bool:
        return (
            self.lint_exit_code != 0
            or self.type_exit_code != 0
            or self.test_exit_code != 0
            or self.lint_errors > 0
            or self.type_errors > 0
            or self.test_coverage < self.minimum_coverage
            or self.timed_out
        )


def validate_output(
    project_dir: str,
    *,
    timeout_seconds: int = 60,
    minimum_coverage: float = 80.0,
) -> ValidationResult:
    """Run lint, type-check, and tests for generated project files."""
    root = Path(project_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_pyproject(root)
    requirements_args = (
        ["--with-requirements", "requirements.txt"]
        if (root / "requirements.txt").exists()
        else []
    )

    lint = _run_command(
        [
            "uv",
            "run",
            "--no-project",
            *requirements_args,
            "--with",
            "ruff",
            "ruff",
            "check",
            ".",
        ],
        cwd=root,
        timeout_seconds=timeout_seconds,
    )
    type_check = _run_command(
        [
            "uv",
            "run",
            "--no-project",
            *requirements_args,
            "--with",
            "mypy",
            "mypy",
            ".",
            "--ignore-missing-imports",
        ],
        cwd=root,
        timeout_seconds=timeout_seconds,
    )
    tests = _run_command(
        [
            "uv",
            "run",
            "--no-project",
            *requirements_args,
            "--with",
            "pytest",
            "--with",
            "pytest-cov",
            "pytest",
            "--cov=src",
            "--cov-report=term-missing",
        ],
        cwd=root,
        timeout_seconds=timeout_seconds,
    )

    lint_errors = len(re.findall(r"^[A-Z]\d{3}\s", lint.output, flags=re.MULTILINE))
    type_errors = len(
        re.findall(r"^\S+:\d+:\s+error:", type_check.output, flags=re.MULTILINE)
    )
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", tests.output)
    coverage = float(coverage_match.group(1)) if coverage_match is not None else 0.0

    return ValidationResult(
        lint_errors=lint_errors,
        type_errors=type_errors,
        test_coverage=coverage,
        lint_output=lint.output,
        type_output=type_check.output,
        test_output=tests.output,
        lint_exit_code=lint.exit_code,
        type_exit_code=type_check.exit_code,
        test_exit_code=tests.exit_code,
        minimum_coverage=minimum_coverage,
        timed_out=lint.timed_out or type_check.timed_out or tests.timed_out,
    )


@dataclass(slots=True)
class _CommandResult:
    output: str
    exit_code: int
    timed_out: bool = False


def _ensure_pyproject(root: Path) -> None:
    """Create minimal project metadata before running validation."""
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        return
    pyproject_path.write_text(
        "[project]\n"
        'name = "generated-project"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n\n'
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n',
        encoding="utf-8",
    )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> _CommandResult:
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        output = f"{stdout}\n{stderr}\nCommand timed out after {timeout_seconds}s."
        return _CommandResult(output=output.strip(), exit_code=124, timed_out=True)
    merged_output = f"{completed.stdout}\n{completed.stderr}".strip()
    return _CommandResult(output=merged_output, exit_code=completed.returncode)
