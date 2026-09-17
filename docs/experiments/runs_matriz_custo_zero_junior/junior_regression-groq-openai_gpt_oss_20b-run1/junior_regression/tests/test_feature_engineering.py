import sys
from pathlib import Path
import random
import math

# Add src to sys.path so the module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

# Import the module under test
from feature_engineering import (
    load_data,
    split_data,
    clean_data,
    prepare_features,
)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _generate_expected_features() -> list[float]:
    """Return the deterministic feature list produced by load_data."""
    return [i / 10.0 for i in range(30)]


def _expected_label_value(x: float) -> float:
    """Return the noise‑free label value for a given feature."""
    return 2.75 * x + 1.25


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_load_data_output_shape_and_type() -> None:
    """
    Ensure load_data returns two lists of length 30.
    """
    features, labels = load_data()
    assert isinstance(features, list) and isinstance(labels, list)
    assert len(features) == 30
    assert len(labels) == 30


def test_load_data_feature_values() -> None:
    """
    Verify that the generated feature values match the deterministic sequence.
    """
    features, _ = load_data()
    expected = _generate_expected_features()
    assert features == expected


def test_load_data_label_noise_bounds() -> None:
    """
    Each label must lie within +/- 0.15 of the expected linear value.
    """
    features, labels = load_data()
    for x, y in zip(features, labels):
        expected = _expected_label_value(x)
        diff = y - expected
        # Noise is uniform in (-0.15, 0.15), so diff should lie in this interval
        assert -0.15 <= diff <= 0.15
        # Also check the absolute difference is bounded
        assert abs(diff) <= 0.15


def test_split_data_default_split() -> None:
    """
    Splitting with default split_at (21) should produce the expected train/test split.
    """
    features, labels = load_data()
    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = split_data(features, labels)

    # Train part
    assert train_features == features[:21]
    assert train_labels == labels[:21]

    # Test part
    assert test_features == features[21:]
    assert test_labels == labels[21:]


def test_split_data_custom_split() -> None:
    """
    Splitting with a custom split index should behave correctly.
    """
    features, labels = load_data()
    split_at = 10
    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = split_data(features, labels, split_at=split_at)

    assert train_features == features[:split_at]
    assert test_features == features[split_at:]
    assert train_labels == labels[:split_at]
    assert test_labels == labels[split_at:]


def test_clean_data_returns_same_objects() -> None:
    """
    clean_data should return the same list objects it receives.
    """
    features, labels = load_data()
    cleaned_features, cleaned_labels = clean_data(features, labels)
    # The function should return the original objects unchanged
    assert cleaned_features is features
    assert cleaned_labels is labels


def test_prepare_features_scaling() -> None:
    """
    prepare_features should divide each feature by 3.0.
    """
    features, _ = load_data()
    scaled = prepare_features(features)
    # Validate that each element is divided by 3.0
    for original, scaled_val in zip(features, scaled):
        assert math.isclose(scaled_val, original / 3.0, rel_tol=1e-9)