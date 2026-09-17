# src/training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

__all__ = ["train_model", "save_model"]


@dataclass
class LinearRegressionModel:
    """Simple linear regression model trained on one‑variable data.

    Attributes
    ----------
    coefficient : float
        Slope of the fitted line.
    intercept : float
        Intercept of the fitted line.
    mse : float
        Mean‑squared error on the training data.
    """

    coefficient: float
    intercept: float
    mse: float

    def predict(self, features: Sequence[float]) -> List[float]:
        """Predict target values for a sequence of feature values.

        Parameters
        ----------
        features : Sequence[float]
            Feature values to predict.

        Returns
        -------
        List[float]
            Predicted target values.
        """
        return [self.intercept + self.coefficient * x for x in features]


def train_model(train_features: Sequence[float], train_labels: Sequence[float]) -> LinearRegressionModel:
    """
    Train a one‑variable linear regression model using the closed‑form solution.

    The implementation follows the notebook logic:
    1. Compute means of the features and labels.
    2. Center the data.
    3. Compute the coefficient (slope) as the covariance divided by the variance.
    4. Compute the intercept.
    5. Compute predictions and the training mean‑squared error.

    Parameters
    ----------
    train_features : Sequence[float]
        The input feature values (e.g., x values).
    train_labels : Sequence[float]
        The corresponding target values (e.g., y values).

    Returns
    -------
    LinearRegressionModel
        The trained model, including coefficient, intercept, and training MSE.
    """
    # Ensure inputs are sequences of the same length
    if len(train_features) != len(train_labels):
        raise ValueError("train_features and train_labels must have the same length")

    # Calculate means
    mean_x = sum(train_features) / len(train_features)
    mean_y = sum(train_labels) / len(train_labels)

    # Center data
    centered_x = [x - mean_x for x in train_features]
    centered_y = [y - mean_y for y in train_labels]

    # Compute coefficient and intercept
    numerator = sum(a * b for a, b in zip(centered_x, centered_y))
    denominator = sum(v * v for v in centered_x)
    if denominator == 0:
        raise ZeroDivisionError("Variance of features is zero; cannot fit linear model.")
    coefficient = numerator / denominator
    intercept = mean_y - coefficient * mean_x

    # Compute predictions and MSE
    train_predictions = [intercept + coefficient * x for x in train_features]
    mse = sum((actual - predicted) ** 2 for actual, predicted in zip(train_labels, train_predictions)) / len(train_labels)

    print(f"fit coefficient={coefficient:.6f}, intercept={intercept:.6f}")

    return LinearRegressionModel(coefficient=coefficient, intercept=intercept, mse=mse)


def save_model(model: LinearRegressionModel, path: Path) -> None:
    """
    Persist the model coefficients to a file.

    The model is saved in a simple text format: two lines
    containing the coefficient and intercept respectively.

    Parameters
    ----------
    model : LinearRegressionModel
        The model to persist.
    path : Path
        Destination file path. The parent directory is created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{model.coefficient}\n{model.intercept}\n")