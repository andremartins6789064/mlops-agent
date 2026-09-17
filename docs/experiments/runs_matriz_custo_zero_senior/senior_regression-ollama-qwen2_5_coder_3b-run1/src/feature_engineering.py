from __future__ import annotations

import random
from collections.abc import Sequence

# Module-level variables for configuration and paths
DATA_PATH: str | None = "/data/hardcoded/junior_regression.csv"
MODEL_PATH: str | None = "/models/hardcoded/senior_model.bin"
DEFAULT_SEED: int = 2026
DEFAULT_SPLIT: float = 0.7

def load_data(path: str | None = None) -> tuple[Sequence[float], Sequence[float]]:
    """
    Loads the dataset from a file.
    
    :param path: The file path to read the data from. If None, reads from predefined DATA_PATH.
    :return: A tuple containing feature and label sequences.
    """
    # Placeholder for actual data loading logic
    # This could be a function that reads from a CSV or any other source
    # For demonstration purposes, we'll use some hardcoded values
    random.seed(DEFAULT_SEED)
    features = [random.uniform(0.0, 1.0) for _ in range(10)]
    labels = [x * x + random.uniform(-0.5, 0.5) for x in features]
    return tuple(features), tuple(labels)

def split_data(features: Sequence[float], labels: Sequence[float]) -> Sequence[Sequence[float]]:
    """
    Splits the dataset into training and testing sets.
    
    :param features: The feature sequence.
    :param labels: The label sequence.
    :return: A sequence containing train and test feature sequences, and their corresponding labels.
    """
    # Placeholder for actual data splitting logic
    split_index = int(DEFAULT_SPLIT * len(features))
    train_features, test_features = features[:split_index], features[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    return seq(train_features), seq(test_features), seq(train_labels), seq(test_labels)

def clean_data(features: Sequence[float], labels: Sequence[float]) -> tuple[Sequence[float], Sequence[float]]:
    """
    Cleans the dataset by removing any outliers or null values.
    
    :param features: The feature sequence.
    :param labels: The label sequence.
    :return: A tuple containing cleaned feature and label sequences.
    """
    # Placeholder for actual data cleaning logic
    return tuple(features), tuple(labels)

def prepare_features(features: Sequence[float], labels: Sequence[float]) -> Sequence[float]:
    """
    Prepares the dataset by transforming features and scaling them.
    
    :param features: The feature sequence.
    :param labels: The label sequence.
    :return: A sequence containing prepared feature sequences. If tuple is returned, it also contains the original labels.
    """
    # Placeholder for actual data preparation logic
    scaled_features = [feature * FEATURE_SCALE for feature in features]
    return seq(scaled_features)

def evaluate_model(train_X, train_y, test_X, test_y) -> dict[str, float]:
    """
    Evaluates a simple linear regression model on the dataset.
    
    :param train_X: The training features array.
    :param train_y: The training target array.
    :param test_X: The test features array.
    :param test_y: The test target array.
    :return: A dictionary with 'final_mse' as the mean-squared error of the model's predictions on the test dataset.
    """
    # Placeholder for actual model evaluation logic
    # For demonstration purposes, let's simulate a simple linear regression evaluation
    predicted = [x * 2 + y for x, y in zip(train_X, train_y)]
    mse = sum((p - t) ** 2 for p, t in zip(predicted, test_y)) / len(test_y)
    return {"final_mse": mse}

# Helper function to convert a sequence to a tuple
def seq(seq: Sequence[float]) -> Sequence[float]:
    """Helper function to ensure the sequence is returned as a tuple."""
    return tuple(seq)

# Main execution block for demonstration purposes
if __name__ == "__main__":
    # Load data, split it, clean and prepare features
    raw_data = load_data()
    train_X, test_X, train_y, test_y = split_data(*raw_data)
    cleaned_X, _ = clean_data(train_X, train_y)
    prepared_X = prepare_features(cleaned_X, train_y)
    
    # Evaluate the model
    result = evaluate_model(train_X, train_y, test_X, test_y)
    print(result)