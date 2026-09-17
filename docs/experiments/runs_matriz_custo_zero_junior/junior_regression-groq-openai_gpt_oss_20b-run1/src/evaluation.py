# evaluation.py
"""Utility module for evaluating regression models.

This module implements a small API for calculating regression metrics
and for building a comprehensive metrics report. It is designed to be
invoked from ``src/main.py`` and to return a mapping that contains a
numeric ``final_mse`` value, as required by the evaluation pipeline.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Dict


def evaluate_model(
    test_labels: Sequence[float],
    predictions: Sequence[float],
) -> Mapping[str, float]:
    """
    Compute the mean squared error (MSE) between the true labels and
    model predictions.

    Parameters
    ----------
    test_labels : Sequence[float]
        The true target values for the test set.
    predictions : Sequence[float]
        The predictions produced by a model for the test set.

    Returns
    -------
    Mapping[str, float]
        A dictionary containing the key ``"final_mse"`` mapped to the
        calculated MSE value.

    Raises
    ------
    ValueError
        If the two input sequences are of different lengths.

    Notes
    -----
    The calculation follows the definition of MSE:

        MSE = (1 / N) * Σ (y_true - y_pred)²

    where N is the number of test samples.
    """
    if len(test_labels) != len(predictions):
        raise ValueError("test_labels and predictions must have the same length")

    squared_errors = (
        (actual - predicted) ** 2
        for actual, predicted in zip(test_labels, predictions)
    )
    mse = sum(squared_errors) / len(test_labels)

    return {"final_mse": mse}


def build_metrics_report(
    test_labels: Sequence[float],
    predictions: Sequence[float],
) -> Dict[str, float]:
    """
    Build a metrics report dictionary that can be extended with additional
    metrics in the future.

    Parameters
    ----------
    test_labels : Sequence[float]
        The true target values for the test set.
    predictions : Sequence[float]
        The predictions produced by a model for the test set.

    Returns
    -------
    Dict[str, float]
        A dictionary containing at least the ``"final_mse"`` metric.
        Additional metrics can be added to this dictionary in the
        future if required.

    Notes
    -----
    The function currently delegates to :func:`evaluate_model` for the
    MSE calculation. This design allows the report builder to remain
    lightweight and easily extensible.
    """
    return evaluate_model(test_labels, predictions)