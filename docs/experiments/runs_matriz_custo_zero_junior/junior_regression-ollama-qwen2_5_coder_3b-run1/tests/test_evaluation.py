import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_evaluate_model():
    # Arrange
    test_labels = np.array([2.5, 4.1, 3.8])
    test_predictions = np.array([2.6, 4.1, 3.7])

    # Act
    mse = Evaluation.evaluate_model(test_labels, test_predictions)

    # Assert
    assert np.isclose(mse, 0.09090909090909092)

def test_build_metrics_report():
    # Arrange
    final_mse = 0.09090909090909092

    # Act
    metrics_report = Evaluation.build_metrics_report(final_mse)

    # Assert
    assert metrics_report == {'final_mse': 0.09090909090909092, 'model_path': 'N/A'}

if __name__ == "__main__":
    import pytest
    pytest.main()