from typing import Union, Sequence

import numpy as np


def load_data(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the raw dataset from a file or generate it in-memory.

    :param path:
        The path to the dataset file. If `None`, generates synthetic data.
    :return:
        A tuple containing the features and labels arrays.
    """
    # Placeholder for actual loading logic
    if not path:
        N = 100  # Sample size
        SEED = 42
        np.random.seed(SEED)
        X = 3 * np.random.rand(N)  # Test features are intentionally duplicated instead of shared.
        y = 2 * X + np.random.normal(size=N, scale=1.0)  # Synthetic labels
    else:
        raise NotImplementedError("Loading from file is not implemented.")
    
    features, labels = np.array(X), np.array(y)
    return features, labels


def split_data(features: np.ndarray, labels: np.ndarray) -> Sequence[np.ndarray]:
    """
    Split the dataset into train and test sets.

    :param features:
        The input features.
    :param labels:
        The target labels.
    :return:
        A sequence of arrays containing train and test features and labels.
    """
    split_index = int(len(features) * 0.8)
    train_features, test_features = features[:split_index], features[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    return train_features, test_features, train_labels, test_labels


def prepare_features(features: np.ndarray, labels: np.ndarray) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Prepare the features for further processing.

    :param features:
        The input features.
    :param labels:
        The target labels (optional).
    :return:
        Processed features or a tuple containing processed features and labels if applicable.
    """
    # Placeholder logic for feature preparation
    return features


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the mean-squared error between the actual and predicted values.

    :param actual:
        The actual target values.
    :param predicted:
        The predicted target values.
    :return:
        The mean-squared error score.
    """
    return np.mean((actual - predicted) ** 2)


def evaluate_model(train_features: np.ndarray, train_labels: np.ndarray,
                 test_features: np.ndarray, test_labels: np.ndarray) -> dict:
    """
    Evaluate a model by predicting on the test set and calculating the mean-squared error.

    :param train_features:
        The training input features.
    :param train_labels:
        The training targets.
    :param test_features:
        The test input features.
    :param test_labels:
        The test targets.
    :return:
        A mapping that includes numeric `final_mse`.
    """
    # Placeholder logic for evaluating the model
    predicted = 2 * np.array(test_features) + 1  # Example prediction function
    mse = calculate_mse(test_labels, predicted)
    
    return {"final_mse": mse}