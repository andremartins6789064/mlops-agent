from __future__ import annotations

from src.shared.python_source import (
    GENERATED_TEST_SRC_BOOTSTRAP,
    extract_imported_libraries,
    generated_test_bootstraps_src,
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


def test_extract_imported_libraries_ignores_stdlib() -> None:
    source = (
        "import random\nimport os.path\nfrom math import sqrt\nimport numpy as np\n"
    )

    assert extract_imported_libraries(source) == {"numpy"}


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


def _training_test_with_src_bootstrap() -> str:
    return (
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"{GENERATED_TEST_SRC_BOOTSTRAP}\n"
        "import training as stage_module\n\n"
        "def test_train() -> None:\n"
        "    stage_module.train_model([1], [2])\n"
    )


def test_generated_test_bootstraps_src_detects_template_insert() -> None:
    source = (
        "import sys\n"
        "from pathlib import Path\n"
        f"{GENERATED_TEST_SRC_BOOTSTRAP}\n"
        "import training as stage_module\n"
    )
    assert generated_test_bootstraps_src(source)


def test_generated_test_bootstraps_src_rejects_missing_insert() -> None:
    source = "from training import train_model\n\ndef test_train() -> None:\n    pass\n"
    assert not generated_test_bootstraps_src(source)


def test_generated_test_matches_module_rejects_missing_src_bootstrap() -> None:
    test_source = (
        "import training as stage_module\n\n"
        "def test_train() -> None:\n"
        "    stage_module.train_model([1], [2])\n"
    )
    module_source = (
        "def train_model(features: list[int], labels: list[int]) -> object:\n"
        "    return features\n"
    )
    assert not generated_test_matches_module(test_source, module_source, "training")


def test_generated_test_matches_module_accepts_src_bootstrap() -> None:
    module_source = (
        "def train_model(features: list[int], labels: list[int]) -> object:\n"
        "    return features\n"
    )
    assert generated_test_matches_module(
        _training_test_with_src_bootstrap(), module_source, "training"
    )
