import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

def test_load_data():
    data, labels = load_data()
    expected_data = [(3.8, 25), (0.4333333333333333, -21), (1.2, -50), (2.9666666666666667, -13), (0.0, 37)]
    expected_labels = [-21, -50, -13, -4.0, 37]
    assert data == expected_data
    assert labels == expected_labels

def test_load_data_file():
    path = 'path_to_dataset.csv'
    with open(path, 'w') as file:
        file.write('9.4,-21\n')
        file.write('1.3,-50\n')

    data, labels = load_data(path)
    assert data == [(3.8, 25), (0.4333333333333333, -21), (1.2, -50), (2.9666666666666667, -13), (0.0, 37)]
    assert labels == [-21, -50, -13, -4.0, 37]

def test_apply_transformation():
    features = [9.4, 1.3, 3.6, 8.9, 0.0]
    transformed_features = apply_transformation(features)
    expected_transformed_features = [3.1333333333333335, 0.4333333333333333, 1.2, 2.9666666666666667, 0.0]
    assert all(abs(a - b) < 1e-10 for a, b in zip(transformed_features, expected_transformed_features))

def test_train_model():
    train_x = [9.4 / 3.0, 1.3 / 3.0, 3.6 / 3.0, 8.9 / 3.0, 0.0]
    train_labels = [-21, -50, -13, -13, 37]
    trained_model_artifact, mse = train_model(train_x, train_labels)
    expected_mse = sum((actual - expected) ** 2 for actual, expected in zip(expected_train_labels, [4.399982615] * len(train_labels))) / len(train_labels)
    assert trained_model_artifact['trained_model'] == 'model_artifact'
    assert abs(mse - expected_mse) < 1e-10

def test_save_model():
    model_artifact = "train"
    file_path = "path_to_model.pickle"
    save_model(model_artifact, file_path)