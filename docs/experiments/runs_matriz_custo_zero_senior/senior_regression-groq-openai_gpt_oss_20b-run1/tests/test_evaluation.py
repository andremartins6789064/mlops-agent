# tests/test_evaluation.py
import sys
from pathlib import Path

# Bootstrap to add the src directory to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
from typing import MutableMapping

import pytest

from evaluation import _mean, evaluate_model, build_metrics_report


def test_mean_with_non_empty_iterable() -> None:
    """_mean should correctly compute the mean of a non‑empty iterable."""
    values = [1.0, 2.0, 3.0, 4.0]
    expected = 2.5
    assert _mean(values) == expected


def test_mean_with_empty_iterable_returns_nan() -> None:
    """_mean should return NaN when the iterable is empty."""
    empty_iter = ()
    result = _mean(empty_iter)
    assert math.isnan(result)


def test_mean_with_generator() -> None:
    """_mean should work with any iterable, e.g. a generator."""
    generator = (x for x in [10, 20, 30])
    assert _mean(generator) == 20.0


def test_evaluate_model_returns_mse_for_matching_inputs() -> None:
    """evaluate_model should return a mapping containing the correct MSE."""
    labels = [1.0, 2.0, 3.0]
    predictions = [1.0, 2.0, 3.0]
    result = evaluate_model(labels, predictions)

    assert isinstance(result, MutableMapping)
    assert "final_mse" in result
    assert result["final_mse"] == 0.0


def test_evaluate_model_raises_value_error_on_mismatch() -> None:
    """evaluate_model should raise ValueError when input lengths differ."""
    labels = [1.0, 2.0]
    predictions = [1.0]
    with pytest.raises(ValueError, match="Number of true labels and predictions must be equal"):
        evaluate_model(labels, predictions)


def test_build_metrics_report_computes_all_metrics() -> None:
    """build_metrics_report should compute final_mse, mae, and r2 correctly."""
    labels = [1.0, 2.0, 3.0]
    predictions = [1.0, 2.0, 3.0]
    report = build_metrics_report(labels, predictions)

    assert isinstance(report, MutableMapping)
    assert report["final_mse"] == 0.0
    assert report["mae"] == 0.0
    assert report["r2"] == 1.0


def test_build_metrics_report_r2_nan_when_variance_zero() -> None:
    """If the true labels have zero variance, r2 should be NaN."""
    labels = [2.0, 2.0, 2.0]
    predictions = [1.0, 2.0, 3.0]
    report = build_metrics_report(labels, predictions)

    assert math.isnan(report["r2"])


def test_build_metrics_report_with_nonzero_errors() -> None:
    """Test the metric calculations with non‑trivial predictions."""
    labels = [1.0, 2.0, 3.0]
    predictions = [4.0, 5.0, 6.0]

    report = build_metrics_report(labels, predictions)

    # MSE: ((1-4)^2 + (2-5)^2 + (3-6)^2) / 3 = 9.0
    assert report["final_mse"] == 9.0
    # MAE: ((3 + 3 + 3) / 3) = 3.0
    assert report["mae"] == 3.0
    # R^2: 1 - residual_variance / total_variance
    # total_variance = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2
    # residual_variance = 27
    # r2 = 1 - 27/2 = -12.5
    assert math.isclose(report["r2"], -12.5, rel_tol=1e-9)


def test_evaluate_and_report_consistency() -> None:
    """The MSE from evaluate_model should match the MSE in the detailed report."""
    labels = [0.0, 1.0, 2.0, 3.0]
    predictions = [0.5, 1.5, 2.5, 2.5]
    eval_result = evaluate_model(labels, predictions)
    report = build_metrics_report(labels, predictions)

    assert math.isclose(eval_result["final_mse"], report["final_mse"], rel_tol=1e-12)