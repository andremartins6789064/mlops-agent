import sys
from pathlib import Path

# Bootstrap to allow imports from src/ during test collection
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from typing import Sequence, Dict

import evaluation


def test_evaluate_model_correct_mse() -> None:
    """evaluate_model should compute the correct mean squared error."""
    test_labels: Sequence[float] = [1.0, 2.0, 3.0]
    predictions: Sequence[float] = [1.0, 2.0, 4.0]
    result = evaluation.evaluate_model(test_labels, predictions)
    expected_mse = (0.0 + 0.0 + 1.0) / 3  # 0^2 + 0^2 + 1^2
    assert isinstance(result, dict)
    assert "final_mse" in result
    assert isinstance(result["final_mse"], float)
    assert result["final_mse"] == expected_mse


def test_evaluate_model_raises_value_error_on_length_mismatch() -> None:
    """evaluate_model should raise ValueError when input lengths differ."""
    test_labels: Sequence[float] = [1.0, 2.0]
    predictions: Sequence[float] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="test_labels and predictions must have the same length"):
        evaluation.evaluate_model(test_labels, predictions)


def test_evaluate_model_works_with_tuples() -> None:
    """evaluate_model should accept any Sequence, including tuples."""
    test_labels: Sequence[float] = (10.0, 20.0)
    predictions: Sequence[float] = (12.0, 18.0)
    result = evaluation.evaluate_model(test_labels, predictions)
    expected_mse = ((10.0 - 12.0) ** 2 + (20.0 - 18.0) ** 2) / 2
    assert result["final_mse"] == expected_mse


def test_build_metrics_report_returns_same_as_evaluate_model() -> None:
    """build_metrics_report should delegate to evaluate_model and return the same result."""
    test_labels: Sequence[float] = [5.0, 5.0, 5.0]
    predictions: Sequence[float] = [6.0, 4.0, 5.5]
    expected = evaluation.evaluate_model(test_labels, predictions)
    report = evaluation.build_metrics_report(test_labels, predictions)
    assert isinstance(report, dict)
    assert report == expected
    assert "final_mse" in report


def test_build_metrics_report_type_hints() -> None:
    """Ensure that build_metrics_report returns a Dict[str, float] as per its signature."""
    test_labels: Sequence[float] = [0.0, 1.0]
    predictions: Sequence[float] = [0.0, 1.0]
    report = evaluation.build_metrics_report(test_labels, predictions)
    assert isinstance(report, dict)
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in report.items())