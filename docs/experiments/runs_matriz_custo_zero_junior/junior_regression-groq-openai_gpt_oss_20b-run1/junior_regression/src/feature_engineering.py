# feature_engineering.py
"""Feature engineering for the junior regression example.

The module reproduces the data generation and preprocessing logic from
the original notebook.  It exposes a minimal public API used by the
training pipeline.

The functions are fully typed and documented for clarity.  All random
choices are seeded to make the behaviour deterministic.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

# --- constants used by the original notebook ---------------------------------
DATA_PATH = "/data/hardcoded/junior_regression.csv"  # kept for reference
MODEL_PATH = "/models/hardcoded/junior_model.bin"     # kept for reference
LEARNING_RATE = 0.01
EPOCHS = 1_000
SEED = 2026

# Initialise random seed
random.seed(SEED)


def load_data() -> Tuple[List[float], List[float]]:
    """
    Generate the raw dataset.

    Returns
    -------
    features
        List of feature values (``raw_x``).
    labels
        Corresponding list of labels (``raw_y``).
    """
    # Generate raw feature values
    raw_x: List[float] = [index / 10.0 for index in range(30)]

    # Generate corresponding labels with linear relationship + noise
    raw_y: List[float] = [
        2.75 * value + 1.25 + random.uniform(-0.15, 0.15)
        for value in raw_x
    ]

    return raw_x, raw_y


def split_data(
    features: List[float],
    labels: List[float],
    split_at: int = 21,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    features
        The list of raw feature values.
    labels
        The list of raw label values.
    split_at
        The index at which to split the data into training (before)
        and testing (after).  Defaults to 21 to match the notebook.

    Returns
    -------
    train_features, test_features, train_labels, test_labels
        The four lists resulting from the split.
    """
    train_features = features[:split_at]
    test_features = features[split_at:]
    train_labels = labels[:split_at]
    test_labels = labels[split_at:]

    return train_features, test_features, train_labels, test_labels


def clean_data(
    features: List[float], labels: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Optional cleaning step.  The original notebook does not modify the
    data, so this function simply returns its inputs unchanged.

    Parameters
    ----------
    features
        Raw feature list.
    labels
        Raw label list.

    Returns
    -------
    Tuple[List[float], List[float]]
        Unmodified features and labels.
    """
    return features, labels


def prepare_features(
    features: List[float]
) -> List[float]:
    """
    Apply the preprocessing transformation used in the original notebook:
    divide each feature by 3.0.

    Parameters
    ----------
    features
        Raw feature list.

    Returns
    -------
    List[float]
        Scaled feature list.
    """
    return [value / 3.0 for value in features]