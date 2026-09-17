# src/evaluation.py

"""
Utility functions for evaluating regression predictions.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, MutableMapping


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of an iterable of numbers."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else math.nan


def evaluate_model(test_labels: Iterable[float], predictions: Iterable[float]) -> MutableMapping[str, float]:
    """
    Compute evaluation metrics for regression predictions.

    Parameters
    ----------
    test_labels : Iterable[float]
        Ground‑truth target values.
    predictions : Iterable[float]
        Predicted target values.

    Returns
    -------
    MutableMapping[str, float]
        A mapping that always contains a numeric ``final_mse`` key.
        Additional metrics may be added in :func:`build_metrics_report`.
    """
    labels = list(test_labels)
    preds = list(predictions)

    if len(labels) != len(preds):
        raise ValueError("Number of true labels and predictions must be equal.")

    squared_errors = [(l - p) ** 2 for l, p in zip(labels, preds)]
    mse = _mean(squared_errors)

    return {"final_mse": mse}


def build_metrics_report(
    test_labels: Iterable[float], predictions: Iterable[float]
) -> Mapping[str, float]:
    """
    Build a detailed metrics report for regression predictions.

    Parameters
    ----------
    test_labels : Iterable[float]
        Ground‑truth target values.
    predictions : Iterable[float]
        Predicted target values.

    Returns
    -------
    Mapping[str, float]
        Dictionary containing `final_mse`, `mae`, and `r2` metrics.
    """
    labels = list(test_labels)
    preds = list(predictions)

    if len(labels) != len(preds):
        raise ValueError("Number of true labels and predictions must be equal.")

    # Mean Squared Error
    squared_errors = [(l - p) ** 2 for l, p in zip(labels, preds)]
    mse = _mean(squared_errors)

    # Mean Absolute Error
    absolute_errors = [abs(l - p) for l, p in zip(labels, preds)]
    mae = _mean(absolute_errors)

    # R² Score
    mean_label = _mean(labels)
    total_variance = sum((l - mean_label) ** 2 for l in labels)
    residual_variance = sum((l - p) ** 2 for l, p in zip(labels, preds))
    r2 = 1 - (residual_variance / total_variance) if total_variance != 0 else float("nan")

    return {"final_mse": mse, "mae": mae, "r2": r2}