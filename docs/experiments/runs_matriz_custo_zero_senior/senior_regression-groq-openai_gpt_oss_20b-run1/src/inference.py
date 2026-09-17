#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference module for the senior regression pipeline.

This module provides lightweight utilities for loading a trained model
and generating predictions on new data.  The implementation follows
the logic from the original notebook and is compatible with
type‑checking tools such as mypy (strict mode) and linter
ruff.  No external dependencies beyond the standard library,
NumPy, pandas, and scikit‑learn / joblib are required.

The public API consists of:

* ``load_model`` – Load a serialized model artifact.
* ``predict``    – Produce predictions for new feature data.

The module also defines a set of shared constants that can be reused
by other stages of the pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np
import pandas as pd

__all__ = ["load_model", "predict"]

# Shared variables – paths and configuration used across stages.
# These can be overridden by the caller if necessary.
DATA_PATH: str | None = None
MODEL_PATH: str | None = "model.pkl"
DEFAULT_SEED: int | None = 42
DEFAULT_SPLIT: float | None = 0.8
FEATURE_SCALE: bool | None = True


def load_model(path: str | Path | None = None) -> Any:
    """
    Load a trained model artifact.

    The function attempts to deserialize a model using joblib first,
    falling back to the Python pickle module if necessary.  The
    ``path`` argument may be omitted; in that case the global
    ``MODEL_PATH`` is used.  An exception is raised if a model file
    cannot be found.

    Parameters
    ----------
    path : str | Path | None
        Path to the serialized model file.  If ``None``, the global
        ``MODEL_PATH`` is used.

    Returns
    -------
    Any
        The deserialized model object, which must expose a ``predict``
        method compatible with scikit‑learn estimators.

    Raises
    ------
    FileNotFoundError
        If the model file cannot be located.
    """
    model_path = Path(path) if path is not None else Path(MODEL_PATH or "")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Prefer joblib for efficiency; fall back to pickle.
    try:
        from joblib import load  # type: ignore[import-not-found]

        return load(model_path)
    except Exception:
        with model_path.open("rb") as f:
            return pickle.load(f)


def predict(
    model: Any,
    test_features: Union[np.ndarray, pd.DataFrame, Iterable[Iterable[float]]],
) -> np.ndarray:
    """
    Generate predictions from a trained model.

    The input ``test_features`` can be a NumPy array, a pandas
    DataFrame, or any iterable of iterables.  It is converted to
    a NumPy array before calling the model's ``predict`` method.

    Parameters
    ----------
    model : Any
        The trained model object with a ``predict`` method.
    test_features : np.ndarray | pd.DataFrame | Iterable[Iterable[float]]
        Feature matrix for which predictions are requested.

    Returns
    -------
    np.ndarray
        Predicted target values.
    """
    if isinstance(test_features, pd.DataFrame):
        features = test_features.values
    else:
        features = np.asarray(test_features, dtype=float)

    predictions = model.predict(features)
    return np.asarray(predictions, dtype=float)