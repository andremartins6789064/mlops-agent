from typing import List, Tuple, Dict
import numpy as np

class Evaluation:
    @staticmethod
    def evaluate_model(test_labels: np.ndarray, test_predictions: np.ndarray) -> float:
        """
        Evaluate the model by calculating the mean-squared error between actual and predicted values.

        Parameters:
        - test_labels: np.array of actual target values.
        - test_predictions: np.array of predicted target values.

        Returns:
        - mse: Mean-squared error for the predictions.
        """
        mse = np.mean((test_labels - test_predictions) ** 2)
        return mse

    @staticmethod
    def build_metrics_report(final_mse: float, model_path: str | None = None) -> Dict[str, int | float | str]:
        """
        Create a metrics report combining final MSE and optional model path.

        Parameters:
        - final_mse: Mean-squared error.
        - model_path: Optional path to the trained model (string).

        Returns:
        - metrics_report: A dictionary containing final MSE and optionally model path.
                            Keys are 'final_mse' and, if provided, 'model_path'.
        """
        return {'final_mse': final_mse, 'model_path': model_path or 'N/A'}

# Example usage
def main():
    # In this example, load_data would be called before the actual evaluation
    test_predictions = [intercept + coefficient * value for value in test_x]
    final_mse = Evaluation.evaluate_model(test_y, test_predictions)
    metrics_report = Evaluation.build_metrics_report(final_mse)

    print(f"Final MSE: {final_mse:.12f}")
    # Assuming you have a logging framework like logging or an output console to handle the metrics report

if __name__ == "__main__":
    main()