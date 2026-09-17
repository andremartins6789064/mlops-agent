import sys
from pathlib import Path

# Ensure the `src` package is importable during test collection
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from training import LinearRegressionModel, train_model, save_model


def test_train_model_simple_linear_relationship() -> None:
    """Training on a perfect linear relationship should return slope 2.0 and intercept 0.0."""
    features: list[float] = [1, 2, 3, 4]
    labels: list[float] = [2, 4, 6, 8]
    model = train_model(features, labels)
    assert isinstance(model, LinearRegressionModel)
    assert model.coefficient == pytest.approx(2.0, rel=1e-6)
    assert model.intercept == pytest.approx(0.0, rel=1e-6)
    # Mean squared error should be zero for perfect fit
    assert model.mse == pytest.approx(0.0, abs=1e-12)


def test_train_model_zero_variance_raises() -> None:
    """Features with zero variance should raise ZeroDivisionError."""
    features: list[float] = [3.0, 3.0, 3.0]
    labels: list[float] = [1.0, 2.0, 3.0]
    with pytest.raises(ZeroDivisionError, match="Variance of features is zero"):
        train_model(features, labels)


def test_train_model_mismatched_length_raises() -> None:
    """Mismatched feature and label lengths should raise ValueError."""
    features: list[float] = [1.0, 2.0]
    labels: list[float] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="must have the same length"):
        train_model(features, labels)


def test_predict_method() -> None:
    """The model's predict method should apply the linear equation."""
    features: list[float] = [1, 2, 3, 4]
    labels: list[float] = [2, 4, 6, 8]
    model = train_model(features, labels)
    test_features: list[float] = [5.0, 10.0]
    predictions = model.predict(test_features)
    assert predictions == pytest.approx([10.0, 20.0], rel=1e-6)


def test_save_model_writes_correctly(tmp_path: Path) -> None:
    """`save_model` writes the coefficient and intercept on separate lines."""
    features: list[float] = [0, 1]
    labels: list[float] = [0, 2]
    model = train_model(features, labels)
    file_path: Path = tmp_path / "model.txt"
    save_model(model, file_path)
    # The file should exist and contain two lines
    assert file_path.is_file()
    content: str = file_path.read_text(encoding="utf-8")
    lines: list[str] = content.strip().splitlines()
    assert len(lines) == 2
    coeff_from_file: float = float(lines[0])
    intercept_from_file: float = float(lines[1])
    assert coeff_from_file == pytest.approx(model.coefficient, rel=1e-6)
    assert intercept_from_file == pytest.approx(model.intercept, rel=1e-6)


def test_save_model_creates_parent_directory(tmp_path: Path) -> None:
    """`save_model` creates parent directories if they do not exist."""
    features: list[float] = [1]
    labels: list[float] = [2]
    model = train_model(features, labels)
    nested_dir: Path = tmp_path / "nested" / "dir"
    file_path: Path = nested_dir / "model.txt"
    # Parent directories should not exist initially
    assert not nested_dir.exists()
    save_model(model, file_path)
    # After saving, parent directories and the file should exist
    assert nested_dir.is_dir()
    assert file_path.is_file()