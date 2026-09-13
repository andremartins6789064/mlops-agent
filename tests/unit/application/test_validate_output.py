from __future__ import annotations

import importlib
import subprocess
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
    assert all("--no-project" in command for command in commands)


def test_validate_output_reports_timeout(monkeypatch: Any, tmp_path: Any) -> None:
    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    result = validate_module.validate_output(str(tmp_path), timeout_seconds=1)

    assert result.timed_out is True
    assert result.has_errors
    assert "timed out" in result.lint_output
    assert result.lint_exit_code == 124
