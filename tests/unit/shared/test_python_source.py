from __future__ import annotations

from src.shared.python_source import (
    extract_imported_libraries,
    generated_test_matches_module,
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


def test_generated_test_matches_module_rejects_unknown_import() -> None:
    test_source = "from eva import eval_model\n\ndef test_eval() -> None:\n    pass\n"
    module_source = (
        "def evaluate_model(values: list[int]) -> int:\n    return len(values)\n"
    )

    assert not generated_test_matches_module(test_source, module_source, "evaluation")


def test_generated_test_matches_module_rejects_wrong_argument_count() -> None:
    test_source = (
        "import training as stage_module\n\n"
        "def test_train() -> None:\n"
        "    stage_module.train_model([1])\n"
    )
    module_source = (
        "def train_model(features: list[int], labels: list[int]) -> object:\n"
        "    return features\n"
    )

    assert not generated_test_matches_module(test_source, module_source, "training")
