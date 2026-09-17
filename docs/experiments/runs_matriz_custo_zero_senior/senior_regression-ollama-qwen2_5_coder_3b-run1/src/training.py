import random


# Shared variables and context plan constants
DEFAULT_SEED = 42
DATA_PATH = "path/to/data"
MODEL_PATH = "model/path"


def make_dataset(seed: int = DEFAULT_SEED, sample_count: int = 30) -> tuple[list[float], list[float]]:
    """Generate a deterministic dataset using given seed and sample count."""
    features = [index / 10.0 for index in range(sample_count)]
    targets = [
        2.75 * value + 1.25 + random.uniform(-0.15, 0.15) for value in features
    ]
    return features, targets


def split_dataset(
    features: Sequence[float], targets: Sequence[float], split_at: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Split the dataset into training and test partitions."""
    return (
        list(features[:split_at]),
        list(targets[:split_at]),
        list(features[split_at:]),
        list(targets[split_at:]),
    )


def scale_features(features: Sequence[float], scale: float) -> list[float]:
    """
    Scale features using the specified scale factor.
    
    Parameters:
    - features (Sequence[float]): The input features to be scaled.
    - scale (float): The scaling factor for each feature.
    
    Returns:
    - list[float]: The scaled features.
    """
    return [value / scale for value in features]


def fit_linear_regression(
    features: Sequence[float], targets: Sequence[float]
) -> tuple[float, float]:
    """Fit a linear regression model for a single feature."""
    mean_x = sum(features) / len(features)
    mean_y = sum(targets) / len(targets)
    centered_x = [value - mean_x for value in features]
    centered_y = [value - mean_y for value in targets]
    
    coefficient = sum(a * b for a, b in zip(centered_x, centered_y)) / sum(
        value * value for value in centered_x
    )
    intercept = mean_y - coefficient * mean_x
    
    return intercept, coefficient


def predict(features: Sequence[float], intercept: float, coefficient: float) -> list[float]:
    """
    Predict target values using the fitted linear regression model.
    
    Parameters:
    - features (Sequence[float]): The input features for prediction.
    - intercept (float): The y-intercept of the linear model.
    - coefficient (float): The slope of the linear model.
    
    Returns:
    - list[float]: The predicted target values.
    """
    return [intercept + coefficient * value for value in features]


def mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Calculate the mean squared error between actual and predicted values.
    
    Parameters:
    - actual (Sequence[float]): The actual target values.
    - predicted (Sequence[float]): The predicted target values.
    
    Returns:
    - float: The mean squared error value.
    """
    return sum((actual_value - predicted_value) ** 2 for actual_value, predicted_value in zip(actual, predicted)) / len(actual)


def save_model(model: object, path: str | None = MODEL_PATH) -> None:
    """Save the trained model to the specified path."""
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError("Model path cannot be None for saving.")


def train_model(train_features: Sequence[float], train_labels: Sequence[float]) -> object:
    """
    Train a linear regression model using the provided training data.
    
    Parameters:
    - train_features (Sequence[float]): The input features for training.
    - train_labels (Sequence[float]): The target labels for training.
    
    Returns:
    - object: The trained machine learning model.
    """
    intercept, coefficient = fit_linear_regression(train_features, train_labels)
    return {"intercept": intercept, "coefficient": coefficient}


def run_experiment(seed: int = DEFAULT_SEED) -> float:
    """Train and evaluate the deterministic senior pipeline."""
    features, targets = make_dataset(seed)
    train_features, train_targets, test_features, test_targets = split_dataset(
        features, targets, DEFAULT_SPLIT
    )
    train_features = scale_features(train_features, FEATURE_SCALE)
    test_features = scale_features(test_features, FEATURE_SCALE)
    
    intercept, coefficient = fit_linear_regression(train_features, train_targets)
    predictions = predict(test_features, intercept, coefficient)
    mse = mean_squared_error(test_targets, predictions)
    
    # Save the trained model
    save_model((intercept, coefficient), MODEL_PATH)
    
    return mse


# Ensure the trained model artifact is returned properly when used as input/output
def train_model_artifact(train_data: dict[str: Sequence[float]], path: str | None = MODEL_PATH) -> tuple[dict[str: float], str]:
    """Get or save the trained linear regression model artifact."""
    # This is a placeholder for future implementation logic that might involve reading
    # previously saved models, training them again if necessary, etc.
    
    intercept, coefficient = train_model(train_data["train_X"], train_data["train_y"])
    
    # Construct and return the train model artifact containing the coefficients and path
    train_model_artifact = {"intercept": intercept, "coefficient": coefficient}
    
    save_model((intercept, coefficient), MODEL_PATH)
    
    return train_model_artifact, MODEL_PATH