from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Sequence

# Shared constants used by the pipeline
DATA_PATH: str = "/data/hardcoded/junior_regression.csv"
MODEL_PATH: str = "/models/hardcoded/senior_model.bin"
DEFAULT_SEED: int = 2026
DEFAULT_SPLIT: int = 21          # Percent of data reserved for testing
FEATURE_SCALE: float = 3.0

__all__ = [
    "load_data",
    "split_data",
    "clean_data",
    "prepare_features",
]


def load_data(path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw dataset from CSV.

    Parameters
    ----------
    path : str | None, default=None
        Path to the CSV file. If ``None`` the module-level ``DATA_PATH`` is used.

    Returns
    -------
    features : pd.DataFrame
        DataFrame containing feature columns.
    labels : pd.Series
        Series containing the target column (assumed to be named ``"y"``).
    """
    csv_path = Path(path) if path is not None else Path(DATA_PATH)
    df = pd.read_csv(csv_path)

    if "y" not in df.columns:
        raise ValueError("Expected target column named 'y' in the CSV file.")

    labels = df.pop("y")
    return df, labels


def clean_data(features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows with missing values from the feature set and the label vector.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    labels : pd.Series
        Target Series.

    Returns
    -------
    cleaned_features : pd.DataFrame
        Features with missing rows dropped.
    cleaned_labels : pd.Series
        Labels with corresponding rows dropped.
    """
    df = features.copy()
    df["__label__"] = labels
    cleaned = df.dropna()
    cleaned_features = cleaned.drop(columns=["__label__"])
    cleaned_labels = cleaned["__label__"]
    return cleaned_features, cleaned_labels


def prepare_features(features: pd.DataFrame, labels: pd.Series | None = None) -> pd.DataFrame:
    """
    Apply a simple feature scaling transformation to numeric columns.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    labels : pd.Series | None, default=None
        Optional target Series (ignored for scaling).

    Returns
    -------
    scaled_features : pd.DataFrame
        Features scaled by ``FEATURE_SCALE``. Non‑numeric columns are left unchanged.
    """
    numeric_cols = features.select_dtypes(include=np.number).columns
    scaled = features.copy()
    scaled[numeric_cols] = scaled[numeric_cols] * FEATURE_SCALE
    return scaled


def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
) -> Sequence[pd.DataFrame | pd.Series]:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    labels : pd.Series
        Target Series.

    Returns
    -------
    train_features : pd.DataFrame
        Training features.
    test_features : pd.DataFrame
        Testing features.
    train_labels : pd.Series
        Training labels.
    test_labels : pd.Series
        Testing labels.
    """
    test_size = DEFAULT_SPLIT / 100.0
    train_f, test_f, train_l, test_l = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=DEFAULT_SEED,
        shuffle=True,
    )
    return train_f, test_f, train_l, test_l