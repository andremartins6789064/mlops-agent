from __future__ import annotations

import importlib
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

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        return outputs.pop(0)

    monkeypatch.setattr(validate_module.subprocess, "run", _fake_run)

    result = validate_module.validate_output(str(tmp_path))

    assert result.lint_errors == 1
    assert result.type_errors == 1
    assert result.test_coverage == 90.0
    assert result.has_errors
