from typing import List, Sequence

def load_data(path: str | None = None) -> Tuple[List[float], List[float]]:
    """Load and preprocess the dataset."""
    if path is not None:
        # Load data from file if a path is provided
        with open(path, 'r') as file:
            train_data = [list(map(float, line.strip().split(','))) for line in file]
        return [(item[0] / 3.0, item[1]) for row in train_data], [row[1] for _, row in enumerate(train_data)]
    else:
        # In-memory dataset if no path is provided
        train_raw = [
            (9.4, 25),
            (1.3, -21),
            (3.6, -50),
            (8.9, -13),
            (0.0, 37)
        ]
        return [(row[0] / 3.0, row[1]) for row in train_raw], [row[1] for _, row in enumerate(train_raw)]

def apply_transformation(features: Sequence[float]) -> List[float]:
    """Apply the transformation to the features."""
    train_x = features
    x2 = [value * value for value in train_x]
    tmp = sum(x2)
    print(f"train features={len(train_x)}, train scale={tmp:.3f}")
    return train_x

def train_model(train_x: List[float], train_labels: List[float]) -> Any:
    """Fit a statistical model to the transformed data."""
    # Fit a real one-variable linear regression in global scope.
    mean_x = sum(train_x) / len(train_x)
    mean_y = sum(train_labels) / len(train_labels)
    centered_x = [value - mean_x for value in train_x]
    centered_y = [value - mean_y for value in train_labels]
    coefficient = sum(a * b for a, b in zip(centered_x, centered_y)) / sum(value * value for value in centered_x)
    intercept = mean_y - coefficient * mean_x
    train_predictions = [intercept + coefficient * value for value in train_x]
    train_mse = sum((actual - predicted) ** 2 for actual, predicted in zip(train_labels, train_predictions)) / len(train_labels)
    print(f"fit coefficient={coefficient:.6f}, intercept={intercept:.6f}")
    
    return {"trained_model": "model_artifact"}, train_mse

def save_model(model_artifact: Any, file_path: str) -> None:
    """Save the trained model."""
    # Implement model saving logic here
    pass