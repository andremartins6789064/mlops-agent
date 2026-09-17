# tests/test_training.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pickle
import pytest

from training import train_model, save_model


def test_train_model_returns_dict_with_features_and_labels():
    """train_model should return a dict containing the original features and labels."""
    features = [1, 2, 3]
    labels = [0, 1, 0]
    model = train_model(features, labels)

    assert isinstance(model, dict), "Returned model is not a dict."
    assert set(model.keys()) == {"features", "labels"}, "Keys are not 'features' and 'labels'."
    assert model["features"] is features, "Features not stored by reference."
    assert model["labels"] is labels, "Labels not stored by reference."


def test_train_model_with_none_and_empty_inputs():
    """train_model should handle None and empty structures."""
    model_none = train_model(None, None)
    assert isinstance(model_none, dict)
    assert model_none["features"] is None
    assert model_none["labels"] is None

    empty_features: list[int] = []
    empty_labels: list[int] = []
    model_empty = train_model(empty_features, empty_labels)
    assert model_empty["features"] is empty_features
    assert model_empty["labels"] is empty_labels


def test_save_model_persists_and_returns_path(tmp_path: Path):
    """save_model should write a pickle file and return its path as a string."""
    model = {"foo": "bar"}
    model_path = tmp_path / "my_model.pkl"

    returned_path = save_model(model, str(model_path))
    assert returned_path == str(model_path), "Returned path does not match the input path."
    assert model_path.is_file(), "Model file was not created."

    with model_path.open("rb") as fp:
        loaded = pickle.load(fp)
    assert loaded == model, "Loaded model does not match the original."


def test_save_model_returns_correct_string_path(tmp_path: Path):
    """save_model should return a string representation of the path."""
    model = {"a": 1}
    path_obj = tmp_path / "output.pkl"
    path_str = str(path_obj)

    result = save_model(model, path_str)
    assert isinstance(result, str), "Result is not a string."
    assert result == path_str, "Returned string path differs from the input."


def test_save_model_raises_for_non_picklable_objects(tmp_path: Path):
    """save_model should raise a PicklingError when the model is not pickleable."""
    # lambda functions are not pickleable
    non_picklable = {"fn": lambda x: x}
    model_path = tmp_path / "bad.pkl"

    with pytest.raises(pickle.PicklingError):
        save_model(non_picklable, str(model_path))