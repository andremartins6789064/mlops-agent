# src/inference.py
"""
Inference module for trained regression models.

This module provides a small API to load a previously trained model and
to generate predictions on new data samples. The implementation
mirrors the logic used in the original notebook while keeping the
interface minimal and type‑safe.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def load_model(path: str) -> Any:
    """
    Load a trained model artifact from disk.

    Parameters
    ----------
    path
        Path to the model file. Supports both pickle and joblib formats.
        The file extension is used to determine the appropriate loader:
        ``.joblib`` or ``.pkl``/``.pickle`` are accepted.

    Returns
    -------
    Any
        The deserialized model object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = p.suffix.lower()
    if ext in {".joblib", ".pkl", ".pickle"}:
        # Prefer joblib for larger objects, fallback to pickle if needed.
        try:
            return joblib.load(p)
        except Exception:
            with p.open("rb") as f:
                return pickle.load(f)
    raise ValueError(f"Unsupported model file extension: {ext}")


def predict(model: Any, test_features: np.ndarray) -> np.ndarray:
    """
    Generate predictions using a trained regression model.

    Parameters
    ----------
    model
        The model object returned by :func:`load_model`. The model
        must implement a ``predict`` method compatible with scikit‑learn.
    test_features
        2‑D array of shape (n_samples, n_features) containing the
        input data for which predictions are requested.

    Returns
    -------
    np.ndarray
        1‑D array of predictions with shape (n_samples,).

    Notes
    -----
    The function forwards the call to ``model.predict`` and returns the
    result as a NumPy array. It performs minimal validation on the input
    array shape to catch common misuse.

    Raises
    ------
    AttributeError
        If ``model`` does not have a ``predict`` method.
    ValueError
        If ``test_features`` is not a 2‑D array.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("The provided model does not have a 'predict' method.")

    test_features = np.asarray(test_features)
    if test_features.ndim != 2:
        raise ValueError(f"Expected 2-D array for test_features, got {test_features.ndim}‑D.")

    predictions = model.predict(test_features)
    return np.asarray(predictions)