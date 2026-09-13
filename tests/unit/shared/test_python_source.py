from __future__ import annotations

from src.shared.python_source import (
    extract_imported_libraries,
    python_syntax_error,
)


def test_python_syntax_error_detects_invalid_generated_code() -> None:
    error = python_syntax_error("def train() -> None:\n    return X = value\n")

    assert error is not None
    assert "invalid syntax" in error


def test_python_syntax_error_accepts_valid_code() -> None:
    assert python_syntax_error("def train() -> None:\n    return None\n") is None


def test_extract_imported_libraries_reads_import_statements() -> None:
    source = (
        "import pandas as pd\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from pathlib import Path\n"
    )

    assert extract_imported_libraries(source) == {"pandas", "sklearn"}


def test_extract_imported_libraries_ignores_invalid_source() -> None:
    assert extract_imported_libraries("return X = value") == set()
