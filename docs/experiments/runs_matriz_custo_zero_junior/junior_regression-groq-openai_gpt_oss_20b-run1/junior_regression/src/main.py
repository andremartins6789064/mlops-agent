"""Executable entrypoint for the generated ML pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import evaluation
import feature_engineering
import inference
import training

METRIC_NAME = "final_mse"


def run_pipeline() -> float:
    """Run all generated stages and return the primary metric."""
    loaded = feature_engineering.load_data()
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("load_data must return (features, labels)")
    features, labels = loaded

    clean_data = getattr(feature_engineering, "clean_data", None)
    if callable(clean_data):
        cleaned = clean_data(features, labels)
        if isinstance(cleaned, tuple) and len(cleaned) == 2:
            features, labels = cleaned

    prepare_features = getattr(feature_engineering, "prepare_features", None)
    if callable(prepare_features):
        prepared = prepare_features(features, labels)
        if isinstance(prepared, tuple) and len(prepared) == 2:
            features, labels = prepared
        elif prepared is not None:
            features = prepared

    split = feature_engineering.split_data(features, labels)
    if (
        isinstance(split, (str, bytes))
        or not isinstance(split, Sequence)
        or len(split) != 4
    ):
        raise RuntimeError("split_data must return four values")
    train_features, test_features, train_labels, test_labels = split

    model = training.train_model(train_features, train_labels)
    predictions = inference.predict(model, test_features)
    metrics = evaluation.evaluate_model(test_labels, predictions)
    if not isinstance(metrics, Mapping):
        raise RuntimeError("evaluate_model must return a metrics mapping")

    value = metrics.get(METRIC_NAME, metrics.get("mse"))
    if value is None:
        raise RuntimeError(f"evaluation metrics must include {METRIC_NAME} or mse")
    return float(value)


if __name__ == "__main__":
    print(f"{METRIC_NAME}={run_pipeline():.12f}")
