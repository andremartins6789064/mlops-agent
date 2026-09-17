import random
from typing import *
from dataclasses import *

@dataclass(frozen=True)
class PreprocessedDataset:
    train_features: List[float]
    train_targets: List[float]
    test_features: List[float]
    test_targets: List[float]

def make_dataset(seed: int, sample_count: int = 30) -> PreprocessedDataset:
    """Generate the deterministic dataset used by both experiment notebooks."""
    generator = random.Random(seed)
    features = [index / 10.0 for index in range(sample_count)]
    targets = [
        2.75 * value + 1.25 + generator.uniform(-0.15, 0.15) for value in features
    ]
    train_features, train_targets, test_features, test_targets = split_dataset(
        features, targets, DEFAULT_SPLIT
    )
    train_features = scale_features(train_features, FEATURE_SCALE)
    test_features = scale_features(test_features, FEATURE_SCALE)
    return PreprocessedDataset(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

def split_dataset(
    features: Sequence[float],
    targets: Sequence[float],
    split_at: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Split features and targets into deterministic train and test partitions."""
    return (
        list(features[:split_at]),
        list(targets[:split_at]),
        list(features[split_at:]),
        list(targets[split_at:]),
    )

def scale_features(features: Sequence[float], scale: float) -> List[float]:
    """Apply the single preprocessing transform used by train and test."""
    return [value / scale for value in features]

def fit_linear_regression(
    features: Sequence[float], targets: Sequence[float]
) -> Tuple[float, float]:
    """Fit a one-feature least-squares linear regression."""
    mean_x = sum(features) / len(features)
    mean_y = sum(targets) / len(targets)
    centered_x = [value - mean_x for value in features]
    centered_y = [value - mean_y for value in targets]
    coefficient = sum(a * b for a, b in zip(centered_x, centered_y)) / sum(
        value * value for value in centered_x
    )
    intercept = mean_y - coefficient * mean_x
    return intercept, coefficient

def predict(
    features: Sequence[float], intercept: float, coefficient: float
) -> List[float]:
    """Generate predictions from a fitted linear regression."""
    return [intercept + coefficient * value for value in features]

def mean_squared_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Compute the regression mean squared error."""
    return sum((actual_value - predicted_value) ** 2 for actual_value, predicted_value in zip(actual, predicted)) / len(actual)

def run_experiment(seed: int = DEFAULT_SEED) -> float:
    """Train and evaluate the deterministic senior pipeline."""
    dataset = make_dataset(seed)
    intercept, coefficient = fit_linear_regression(dataset.train_features, dataset.train_targets)
    predictions = predict(dataset.test_features, intercept, coefficient)
    return mean_squared_error(dataset.test_targets, predictions)

def load_model(model_artifact_path: str | None = None) -> Tuple[float, float]:
    """Load the pre-trained model weights."""
    # Dummy implementation for demonstration
    return fit_linear_regression(range(10), range(10))  # Replace with actual loading logic

def predict(model: Tuple[float, float], new_samples_X: List[float]) -> List[float]:
    """Generate predictions from a fitted linear regression model."""
    intercept, coefficient = model
    return [intercept + coefficient * value for value in new_samples_X]