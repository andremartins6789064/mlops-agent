from __future__ import annotations

import importlib
import subprocess
import tomllib
from types import SimpleNamespace
from typing import Any

validate_module = importlib.import_module("src.application.validate_output")


def test_validate_output_parses_tool_outputs(monkeypatch: Any, tmp_path: Any) -> None:
    outputs = [
        SimpleNamespace(stdout="E501 example\n", stderr="", returncode=1),
        SimpleNamespace(
            stdout="file.py:1: error: Incompatible types\n", stderr="", returncode=1
        ),
        SimpleNamespace(stdout="TOTAL 10 1 90%\n", stderr="", returncode=0),
    ]
    commands: list[list[str]] = []

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        commands.append(args[0])
        return outputs.pop(0)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    result = validate_module.validate_output(str(tmp_path))

    assert result.lint_errors == 1
    assert result.type_errors == 1
    assert result.test_coverage == 90.0
    assert result.has_errors
    assert (tmp_path / "pyproject.toml").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert all("--no-project" in command for command in commands)
    assert all("--with-requirements" in command for command in commands)


def test_validate_output_reports_timeout(monkeypatch: Any, tmp_path: Any) -> None:
    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    result = validate_module.validate_output(str(tmp_path), timeout_seconds=1)

    assert result.timed_out is True
    assert result.has_errors
    assert "timed out" in result.lint_output
    assert result.lint_exit_code == 124


def test_validate_output_installs_detected_sklearn_in_isolated_env(
    monkeypatch: Any, tmp_path: Any
) -> None:
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "training.py").write_text(
        "from sklearn.linear_model import LinearRegression\n\n"
        "def train_model() -> object:\n"
        "    return LinearRegression()\n",
        encoding="utf-8",
    )
    outputs = [
        SimpleNamespace(stdout="", stderr="", returncode=0),
        SimpleNamespace(stdout="", stderr="", returncode=0),
        SimpleNamespace(stdout="TOTAL 10 2 80%\n", stderr="", returncode=0),
    ]
    commands: list[list[str]] = []

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        commands.append(args[0])
        return outputs.pop(0)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    result = validate_module.validate_output(str(tmp_path))

    requirements = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    pyproject = tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert requirements.splitlines() == ["scikit-learn"]
    assert pyproject["project"]["dependencies"] == ["scikit-learn"]
    assert result.test_coverage == 80.0
    assert all("--no-project" in command for command in commands)
    assert all(
        command[command.index("--with-requirements") + 1] == "requirements.txt"
        for command in commands
    )


def test_validate_output_rewrites_legacy_pyproject_without_dependencies(
    monkeypatch: Any, tmp_path: Any
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "legacy-artifact"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "training.py").write_text(
        "from sklearn.linear_model import LinearRegression\n",
        encoding="utf-8",
    )
    outputs = [
        SimpleNamespace(stdout="", stderr="", returncode=0),
        SimpleNamespace(stdout="", stderr="", returncode=0),
        SimpleNamespace(stdout="TOTAL 4 0 100%\n", stderr="", returncode=0),
    ]

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        return outputs.pop(0)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    validate_module.validate_output(str(tmp_path))

    pyproject = tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert pyproject["project"]["name"] == "legacy-artifact"
    assert pyproject["project"]["dependencies"] == ["scikit-learn"]
    assert requirements.splitlines() == ["scikit-learn"]
