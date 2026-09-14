from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.domain.pipeline_contract import (
    PRIMARY_METRIC_NAME,
    check_pipeline_contract,
    entrypoint_source,
    format_contract_issues,
    prompt_signatures,
)
from src.infrastructure.exporters.file_system import FileSystemOutputWriter

# Signatures taken from the groq_20b / T-29 conversion: correct notebook logic,
# wrong public contract (DataFrame-in, mapping-out vs. the agreed call shape).
_GROQ_LIKE_FEATURE_ENGINEERING = """
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df, df.iloc[:, 0]
"""

_CONFORMING_MODULES = {
    "feature_engineering": (
        "def load_data() -> tuple[list[int], list[int]]:\n"
        "    return [1, 2], [2, 4]\n\n"
        "def split_data(features: list[int], labels: list[int]) -> "
        "tuple[list[int], list[int], list[int], list[int]]:\n"
        "    return features, features, labels, labels\n"
    ),
    "training": (
        "def train_model(train_features: list[int], train_labels: list[int]) "
        "-> dict[str, int]:\n"
        "    return {'coefficient': 2}\n"
    ),
    "inference": (
        "def predict(model: dict[str, int], test_features: list[int]) -> list[int]:\n"
        "    return [model['coefficient'] * value for value in test_features]\n"
    ),
    "evaluation": (
        "def evaluate_model(test_labels: list[int], predictions: list[int]) "
        "-> dict[str, float]:\n"
        "    return {'final_mse': 0.0}\n"
    ),
}


def test_prompt_and_entrypoint_share_metric_and_signatures() -> None:
    source = entrypoint_source()
    prompt = prompt_signatures()

    assert f'METRIC_NAME = "{PRIMARY_METRIC_NAME}"' in source
    assert "_call_with_fallback" not in source
    assert "load_data()" in source
    assert "load_data()" in prompt
    assert "evaluate_model(test_labels, predictions)" in prompt
    assert "must not require a path" in prompt


def test_groq_like_dataframe_decomposition_is_reported_not_raised() -> None:
    modules = {
        "feature_engineering": _GROQ_LIKE_FEATURE_ENGINEERING,
        "training": _CONFORMING_MODULES["training"],
        "inference": _CONFORMING_MODULES["inference"],
        "evaluation": (
            "def evaluate_model(test_x: list[float], test_y: list[float], "
            "intercept: float, coefficient: float) -> float:\n"
            "    return 0.0\n"
        ),
    }

    issues = check_pipeline_contract(modules)

    assert issues
    serialized = format_contract_issues(issues)
    assert "feature_engineering.load_data" in serialized
    assert "evaluation.evaluate_model" in serialized
    assert "feature_engineering.split_data" in serialized


def test_conforming_modules_have_no_issues() -> None:
    assert check_pipeline_contract(_CONFORMING_MODULES) == []


def test_entrypoint_from_contract_runs_conforming_modules(tmp_path: Path) -> None:
    writer = FileSystemOutputWriter()
    root = tmp_path / "project"
    writer.write_stage_modules(
        project_root=str(root), generated_modules=_CONFORMING_MODULES
    )
    writer.write_entrypoint(project_root=str(root))

    completed = subprocess.run(
        [sys.executable, "src/main.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert f"{PRIMARY_METRIC_NAME}=0.000000000000" in completed.stdout
