"""Utilities for validating and inspecting generated Python source."""

from __future__ import annotations

import ast
import sys

_STDLIB_MODULES = set(sys.stdlib_module_names) | {"__future__"}


def python_syntax_error(source: str) -> str | None:
    """Return a syntax error message, or ``None`` when source is valid."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        return f"{exc.msg} ({location})"
    return None


def extract_imported_libraries(source: str) -> set[str]:
    """Return top-level library names imported by valid Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    libraries: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            libraries.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            libraries.add(node.module.split(".")[0])
    return libraries - _STDLIB_MODULES
